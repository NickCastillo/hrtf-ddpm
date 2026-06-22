"""
model.py — Diffusion model (DDPM) and U-Net architecture for HRTF personalisation.

Changes vs. original commit (1c54d54 / uploaded version):
  [BUG] backward(): 'if t == 0' compared a Tensor to an int.  A non-empty Tensor
        is always truthy in Python, so the else branch (adding noise) was NEVER
        executed.  Every denoising step returned a deterministic mean — the model
        could not explore the posterior.  Fixed to 'if t.item() == 0'.
  [BUG] Block: self.bnorm2 and self.conv2 were defined but never called in
        forward(), wasting parameters and misleading readers.  Removed.
  [DESIGN] UNet: removed unused 'sequence_channels_rev = reversed(...)' line.
  [DESIGN] Added docstrings throughout.
"""

import torch
from torch import nn
import math


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

class DiffusionModel:
    """
    Linear-schedule DDPM forward/backward process.

    Parameters
    ----------
    start_schedule : float
        β at t=0.
    end_schedule : float
        β at t=T-1.
    timesteps : int
        Total number of diffusion steps T.
        The paper used 600; the uploaded commit used 300.
        Set this to match whichever run you are replicating.
    """

    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps=600):
        self.start_schedule = start_schedule
        self.end_schedule   = end_schedule
        self.timesteps      = timesteps

        self.betas          = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas         = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_0, t, device="cpu"):
        """
        DDPM forward (noising) process: q(x_t | x_0).
        Returns (x_t, noise) where noise ~ N(0, I).
        """
        noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.alphas_cumprod.sqrt(), t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1.0 - self.alphas_cumprod), t, x_0.shape
        )

        mean     = sqrt_alphas_cumprod_t.to(device)             * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device)   * noise.to(device)

        return mean + variance, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        """
        DDPM backward (denoising) step: p(x_{t-1} | x_t).

        Parameters
        ----------
        x : Tensor  (B, C, L)   noisy signal at step t.
        t : Tensor  (B,)        current timestep indices.
        model : nn.Module       the noise-prediction U-Net.
        **kwargs                passed through to model (labels, head_embedding, etc.)
        """
        labels          = kwargs.get('labels',          None)
        head_embedding  = kwargs.get('head_embedding',  None)
        ears_embedding  = kwargs.get('ears_embedding',  None)

        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1.0 - self.alphas_cumprod), t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(
            torch.sqrt(1.0 / self.alphas), t, x.shape
        )

        predicted_noise = model(
            x, t,
            labels=labels,
            head_embedding=head_embedding,
            ears_embedding=ears_embedding,
        )

        mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = betas_t

        # BUG FIX: original had 'if t == 0' where t is a Tensor.
        # In Python, any non-empty Tensor is truthy, so the else branch was
        # NEVER executed — no noise was ever added during denoising.
        # This turned the stochastic sampler into a deterministic one and
        # prevented the model from exploring the posterior distribution.
        if t.item() == 0:
            return mean
        else:
            noise    = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        """Gather schedule values at timestep indices t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = values.gather(-1, t.long().cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embedding for diffusion timestep t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ---------------------------------------------------------------------------
# U-Net building block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Single encoder or decoder block in the U-Net.

    Each block:
      1. Applies a convolution + BN + ReLU.
      2. Adds sinusoidal time embedding (projected to channels_out).
      3. Adds head anthropometric embedding (optional).
      4. Adds ear anthropometric embedding (optional).
      5. Adds DOA label embedding (optional).
      6. Downsamples (stride-2 conv) or upsamples (transposed conv).

    Parameters
    ----------
    channels_in, channels_out : int
    time_embedding_dims : int
    labels : int or False
        Number of DOA classes for nn.Embedding, or False to disable.
    head_embedding : bool
    ears_embedding : bool
    downsample : bool
        True → encoder block (stride-2 conv).
        False → decoder block (transposed conv, input is concatenated skip).
    """

    def __init__(
        self,
        channels_in,
        channels_out,
        time_embedding_dims,
        labels,
        head_embedding,
        ears_embedding,
        num_filters=3,
        downsample=True,
    ):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding      = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels              = labels
        self.head_embedding      = head_embedding
        self.ears_embedding      = ears_embedding

        # DOA conditioning: each spatial position gets a learnable embedding vector.
        if labels:
            self.label_emb = nn.Embedding(labels, channels_out)

        self.downsample = downsample

        if downsample:
            # Encoder: single-channel input, stride-2 downsampling.
            self.conv1 = nn.Conv1d(channels_in,      channels_out, num_filters, padding=1)
            self.final = nn.Conv1d(channels_out,     channels_out, 4, 2, 1)
        else:
            # Decoder: skip connection doubles the channel count.
            self.conv1 = nn.Conv1d(2 * channels_in,  channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose1d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1 = nn.BatchNorm1d(channels_out)

        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)

        if ears_embedding:
            # 24 ear anthropometric features → time_embedding_dims → channels_out.
            self.ears_measurement_embedding = nn.Linear(24, time_embedding_dims)
            self.ears_mlp = nn.Linear(time_embedding_dims, channels_out)

        if head_embedding:
            # 13 head anthropometric features → time_embedding_dims → channels_out.
            self.head_measurement_embedding = nn.Linear(13, time_embedding_dims)
            self.head_mlp = nn.Linear(time_embedding_dims, channels_out)

        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        # --- Convolution + BN + ReLU ---
        o = self.bnorm1(self.relu(self.conv1(x)))

        # --- Time embedding ---
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time.unsqueeze(2)

        # --- Head anthropometrics ---
        if self.head_embedding:
            head_meas = kwargs.get('head_embedding')
            o_head = self.head_mlp(self.head_measurement_embedding(head_meas.float()))
            o = o + o_head.unsqueeze(2)

        # --- Ear anthropometrics ---
        if self.ears_embedding:
            ear_meas = kwargs.get('ears_embedding')
            o_ears = self.ears_mlp(self.ears_measurement_embedding(ear_meas.float()))
            o = o + o_ears.unsqueeze(2)

        # --- DOA label embedding ---
        if self.labels:
            label   = kwargs.get('labels')
            o_label = self.label_emb(label).squeeze(1)
            o = o + o_label.unsqueeze(2)

        return self.final(o)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    1-D U-Net for noise prediction.

    Input : (B, audio_channels, L)   — noisy HRIR at timestep t.
    Output: (B, audio_channels, L)   — predicted noise ε.

    audio_channels=2 means both ears are predicted jointly in a single forward
    pass, matching the paper's architecture description.

    Parameters
    ----------
    audio_channels : int
        1 = single-ear, 2 = binaural (paper default).
    time_embedding_dims : int
        Dimension of sinusoidal time embedding.
    labels : int or False
        Number of DOA embedding classes (440 for full HUTUBS sphere).
    head_embedding, ears_embedding : bool
        Whether to inject anthropometric conditioning.
    sequence_channels : tuple[int]
        Channel widths at each encoder scale.
    """

    def __init__(
        self,
        audio_channels=2,
        time_embedding_dims=256,
        labels=False,
        head_embedding=False,
        ears_embedding=False,
        sequence_channels=(64, 128, 256, 512, 1024),
    ):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims

        # Encoder: 4 downsampling blocks (len(sequence_channels) - 1).
        self.downsampling = nn.ModuleList([
            Block(ch_in, ch_out, time_embedding_dims, labels, head_embedding, ears_embedding)
            for ch_in, ch_out in zip(sequence_channels, sequence_channels[1:])
        ])

        # Decoder: 4 upsampling blocks, mirroring the encoder.
        sc_rev = sequence_channels[::-1]
        self.upsampling = nn.ModuleList([
            Block(ch_in, ch_out, time_embedding_dims, labels, head_embedding, ears_embedding,
                  downsample=False)
            for ch_in, ch_out in zip(sc_rev, sc_rev[1:])
        ])

        # Stem: project audio channels to first encoder width.
        self.conv1 = nn.Conv1d(audio_channels, sequence_channels[0],  3, padding=1)
        # Head: project back to audio channels.
        self.conv2 = nn.Conv1d(sequence_channels[0], audio_channels, 1)

    def forward(self, x, t, **kwargs):
        residuals = []

        o = self.conv1(x)

        # Encoder path — store residuals for skip connections.
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)

        # Decoder path — concatenate skip connections before each block.
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)
