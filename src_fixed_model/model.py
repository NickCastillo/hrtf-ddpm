"""
model.py — Diffusion model (DDPM) and U-Net architecture for HRTF personalisation.

Architecture matches arxiv 2501.02871 (Albarracin et al., ICASSP 2025) as closely
as the HUTUBS dataset allows. Deviations from the paper are documented explicitly.

U-Net architecture (Section III-C):
  - sequence_channels = (4, 8, 16, 32, 64): five encoder/decoder scales.
  - Each Block: (i) Conv1d+ReLU+BN, (ii) conditioning injections,
                (iii) second Conv1d with NO activation, (iv) stride-2/transposed conv.
  - SelfAttention (4 heads) after each ENCODER block.
  - Binaural output: audio_channels=2 (left + right ear jointly).

Anthropometric features (Section III-A):
  The paper references CIPIC's N=27 features (17 head + 10 pinna). However,
  HUTUBS provides a related but different set: 13 head/torso features (CIPIC
  x1-x9, x12, x14, x16, x17 — missing x10, x11, x13, x15) and 12 pinna
  features per ear (d1-d10 + theta1 + theta2, two extra vs CIPIC's d1-d8).
  We use all 37 available HUTUBS features (13 head + 24 ear = 12L + 12R).
  This is documented as a methodological note: the paper's N=27 claim is not
  exactly reproducible from HUTUBS.

Changes vs. src_fixed_model/model.py:
  [ARCH] sequence_channels: (64,128,256,512,1024) → (4,8,16,32,64).
         This was the cause of the ~26s/epoch slowdown (100× more parameters).
  [ARCH] Block.conv2 (second conv, no activation) is now called in forward().
         Previously defined but dead code — never executed.
  [ARCH] Self-attention moved from decoder to encoder path (paper spec).
  [ARCH] time_embedding_dims: 256 → 128 (restores the faster old model's value;
         paper does not specify this hyperparameter).
  [BUG]  backward(): t.item()==0 fix retained from previous session.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

class DiffusionModel:
    """
    Linear-schedule DDPM forward/backward process.

    Parameters
    ----------
    start_schedule : float   β₀ (paper: 1e-4)
    end_schedule   : float   β_T (paper: 0.02)
    timesteps      : int     T   (paper: 600)
    """

    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps=600):
        self.start_schedule = start_schedule
        self.end_schedule   = end_schedule
        self.timesteps      = timesteps

        self.betas          = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas         = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_0, t, device='cpu'):
        """
        DDPM forward (noising): q(x_t | x_0) = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε
        Returns (x_t, ε).
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod_t = self.get_index_from_list(
            self.alphas_cumprod.sqrt(), t, x_0.shape
        )
        sqrt_one_minus_alpha_cumprod_t = self.get_index_from_list(
            torch.sqrt(1.0 - self.alphas_cumprod), t, x_0.shape
        )
        x_t = (sqrt_alpha_cumprod_t.to(device) * x_0.to(device)
               + sqrt_one_minus_alpha_cumprod_t.to(device) * noise.to(device))
        return x_t, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        """
        One DDPM denoising step: p(x_{t-1} | x_t).

        Parameters
        ----------
        x     : Tensor (B, C, L)   noisy signal at step t.
        t     : Tensor (B,)        current timestep indices.
        model : nn.Module          noise-prediction U-Net.
        **kwargs                   forwarded to model (labels, head_embedding, etc.)
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1.0 - self.alphas_cumprod), t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(
            torch.sqrt(1.0 / self.alphas), t, x.shape
        )

        predicted_noise = model(x, t, **kwargs)

        mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # BUG FIX (retained): original code had 'if t == 0' where t is a Tensor.
        # A non-empty Tensor is always truthy, so the else branch (adding noise)
        # was never executed, making the sampler fully deterministic.
        # Fixed to t.item() == 0.
        if torch.all(t == 0):
            return mean

        else:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(betas_t) * noise

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        """Gather schedule values at timestep t and broadcast over (B, C, L)."""
        batch_size = t.shape[0]
        out = values.gather(-1, t.long().cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal embedding for diffusion timestep t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Self-attention block
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension of a 1-D feature map.

    Paper (Section III-C): "self-attention layers with 4 attention heads are
    integrated after each downsampling block."

    Input / output shape: (B, C, L).
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.ln  = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ff  = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # (B, C, L) → (B, L, C) for MHA, then back.
        x   = x.permute(0, 2, 1)
        x_n = self.ln(x)
        attn, _ = self.mha(x_n, x_n, x_n)
        x = x + attn
        x = x + self.ff(x)
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# U-Net block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    One encoder or decoder block of the U-Net (paper Section III-C).

    Processing order:
      (i)   Conv1d(k=3) + ReLU + BN
      (ii)  Add conditioning signals to feature maps:
              • sinusoidal time embedding  (via linear projection)
              • head anthropometrics       (13 features → channels_out)
              • ear anthropometrics        (24 features → channels_out)
              • DOA label                  (nn.Embedding lookup)
      (iii) Second Conv1d(k=3) with NO activation  [paper spec]
      (iv)  Stride-2 Conv1d (encoder) or ConvTranspose1d (decoder)

    Anthropometric dims (HUTUBS CSV, all available features):
      HEAD_DIM = 13  (CIPIC x1-x9, x12, x14, x16, x17)
      EARS_DIM = 24  (12 left pinna + 12 right pinna: d1-d10, theta1, theta2 each)

    Parameters
    ----------
    channels_in, channels_out : int
    time_embedding_dims       : int
    labels      : int or False   Number of DOA classes (440 for HUTUBS).
    head_embedding : bool
    ears_embedding : bool
    downsample  : bool   True = encoder block; False = decoder block.
    """

    # All 37 available HUTUBS anthropometric features.
    # See dataset.py and AntrhopometricMeasures.csv for the exact column mapping.
    # Note: paper cites CIPIC N=27 (17 head + 10 pinna) but HUTUBS provides a
    # different set; we use everything available rather than dropping valid data.
    HEAD_DIM = 13   # cols 1-13: x1-x9, x12, x14, x16, x17
    EARS_DIM = 24   # cols 14-37: L_d1..L_d10, L_theta1, L_theta2,
                    #             R_d1..R_d10, R_theta1, R_theta2

    def __init__(
        self,
        channels_in,
        channels_out,
        time_embedding_dims,
        labels,
        head_embedding,
        ears_embedding,
        kernel_size=3,
        downsample=True,
    ):
        super().__init__()

        self.time_embedding  = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels          = labels
        self.head_embedding  = head_embedding
        self.ears_embedding  = ears_embedding
        self.downsample      = downsample

        padding = kernel_size // 2

        if labels:
            self.label_emb = nn.Embedding(labels, channels_out)

        if downsample:
            self.conv1 = nn.Conv1d(channels_in,     channels_out, kernel_size, padding=padding)
            self.final = nn.Conv1d(channels_out,    channels_out, 4, 2, 1)
        else:
            # Skip connection doubles the input channel count in the decoder.
            self.conv1 = nn.Conv1d(2 * channels_in, channels_out, kernel_size, padding=padding)
            self.final = nn.ConvTranspose1d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1   = nn.BatchNorm1d(channels_out)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu     = nn.ReLU()

        if head_embedding:
            self.head_mlp = nn.Linear(self.HEAD_DIM, channels_out)

        if ears_embedding:
            self.ears_mlp = nn.Linear(self.EARS_DIM, channels_out)

        # Second conv with no activation (paper Section III-C).
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size, padding=padding)

    def forward(self, x, t, **kwargs):
        # (i) First conv + BN + ReLU.
        o = self.bnorm1(self.relu(self.conv1(x)))

        # (ii) Conditioning injections — each projected to (B, channels_out)
        #      then broadcast over the temporal dimension as (B, channels_out, 1).
        o = o + self.relu(self.time_mlp(self.time_embedding(t))).unsqueeze(2)

        if self.head_embedding:
            head_meas = kwargs.get('head_embedding')
            if head_meas is not None:
                o = o + self.head_mlp(head_meas.float()).unsqueeze(2)

        if self.ears_embedding:
            ear_meas = kwargs.get('ears_embedding')
            if ear_meas is not None:
                o = o + self.ears_mlp(ear_meas.float()).unsqueeze(2)

        if self.labels:
            label = kwargs.get('labels')
            if label is not None:
                o = o + self.label_emb(label).squeeze(1).unsqueeze(2)

        # (iii) Second conv, no activation.
        o = self.conv2(o)

        # (iv) Downsample / upsample.
        return self.final(o)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    1-D U-Net noise predictor for HRTF personalisation.

    Input : (B, 2, L)   noisy binaural HRIR at diffusion step t.
    Output: (B, 2, L)   predicted noise ε_θ(x_t, t, r, a).

    Encoder: 4 downsampling blocks (sequence_channels pairs),
             each followed by SelfAttention (4 heads).
    Decoder: 4 upsampling blocks with skip connections from the encoder.
    Stem   : Conv1d(audio_channels → sequence_channels[0]).
    Head   : Conv1d(sequence_channels[0] → audio_channels).

    Parameters
    ----------
    audio_channels      : int     2 = binaural (paper default).
    time_embedding_dims : int     Sinusoidal embedding dimension (paper unspecified).
    labels              : int or False   DOA classes (440 for full HUTUBS sphere).
    head_embedding      : bool    Inject head/torso anthropometrics.
    ears_embedding      : bool    Inject pinna anthropometrics.
    sequence_channels   : tuple   Encoder channel widths. Paper: (4, 8, 16, 32, 64).
    """

    def __init__(
        self,
        audio_channels=2,
        time_embedding_dims=128,
        labels=False,
        head_embedding=False,
        ears_embedding=False,
        sequence_channels=(4, 8, 16, 32, 64),
    ):
        super().__init__()

        # Stem.
        self.conv1 = nn.Conv1d(audio_channels, sequence_channels[0], 3, padding=1)

        # Encoder: one Block per adjacent channel pair.
        self.downsampling = nn.ModuleList([
            Block(ch_in, ch_out, time_embedding_dims,
                  labels, head_embedding, ears_embedding, downsample=True)
            for ch_in, ch_out in zip(sequence_channels, sequence_channels[1:])
        ])

        # Self-attention after each encoder block (paper: "after each downsampling block").
        self.encoder_attentions = nn.ModuleList([
            SelfAttention(ch_out, num_heads=4)
            for ch_out in sequence_channels[1:]
        ])

        # Decoder: mirrors encoder.
        sc_rev = sequence_channels[::-1]
        self.upsampling = nn.ModuleList([
            Block(ch_in, ch_out, time_embedding_dims,
                  labels, head_embedding, ears_embedding, downsample=False)
            for ch_in, ch_out in zip(sc_rev, sc_rev[1:])
        ])

        # Head.
        self.conv2 = nn.Conv1d(sequence_channels[0], audio_channels, 1)

    def forward(self, x, t, **kwargs):
        residuals = []

        o = self.conv1(x)

        # Encoder: downsample → attention → store skip.
        for ds, attn in zip(self.downsampling, self.encoder_attentions):
            o = ds(o, t, **kwargs)
            o = attn(o)
            residuals.append(o)

        # Decoder: pad if needed → cat skip → upsample.
        for us, res in zip(self.upsampling, reversed(residuals)):
            if o.shape[2] != res.shape[2]:
                o = F.pad(o, (0, res.shape[2] - o.shape[2]))
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)
