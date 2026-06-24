import torch
from torch import nn
import math
import numpy as np


class DiffusionModel:
    """
    DDPM with 600 timesteps as per the paper (Sec. IV-A).
    Training cost is independent of timesteps — only inference is affected.
    """
    def __init__(self, start_schedule=1e-4, end_schedule=0.02, timesteps=600):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps

        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def forward(self, x_0, t, device="cpu"):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape
        )
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return mean + variance, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        labels = kwargs.get('labels', None)
        head_embedding = kwargs.get('head_embedding', None)
        ears_embedding = kwargs.get('ears_embedding', None)

        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            torch.sqrt(1. - self.alphas_cumprod), t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)

        denoise_model = model(x, t, labels=labels, head_embedding=head_embedding, ears_embedding=ears_embedding)
        mean = sqrt_recip_alphas_t * (x - betas_t * denoise_model / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        out = values.gather(-1, t.long().cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SelfAttention1d(nn.Module):
    """
    Lightweight self-attention for 1D feature maps.
    Applied only at the two deepest encoder levels and the bottleneck
    to keep compute manageable (paper: 4 attention heads).
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)          # (B, L, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)          # (B, C, L)
        return x + h                     # residual


class Block(nn.Module):
    """
    U-Net block matching the paper architecture:
      conv -> GroupNorm -> ReLU -> conv (no activation)  [two convolutions per block]
      + time embedding, label embedding, head/ear measurement embeddings
      + final downsample (stride-2 conv) or upsample (transposed conv)

    GroupNorm replaces BatchNorm for stability at small spatial sizes.
    """
    def __init__(self, channels_in, channels_out, time_embedding_dims,
                 labels, head_embedding, ears_embedding,
                 num_filters=3, downsample=True, use_attention=False):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        self.head_embedding = head_embedding
        self.ears_embedding = ears_embedding
        self.downsample = downsample

        # ── Conditioning projections ──────────────────────────────────────────
        if labels:
            self.label_emb = nn.Embedding(labels, channels_out)

        if head_embedding:
            self.head_measurement_embedding = nn.Linear(13, time_embedding_dims)
            self.head_mlp = nn.Linear(time_embedding_dims, channels_out)

        if ears_embedding:
            self.ears_measurement_embedding = nn.Linear(24, time_embedding_dims)
            self.ears_mlp = nn.Linear(time_embedding_dims, channels_out)

        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)

        # ── Two convolutional layers per block (paper Sec. III-C) ─────────────
        if downsample:
            self.conv1 = nn.Conv1d(channels_in, channels_out, num_filters, padding=1)
        else:
            self.conv1 = nn.Conv1d(2 * channels_in, channels_out, num_filters, padding=1)

        # Second conv with no activation (as stated in the paper)
        self.conv2 = nn.Conv1d(channels_out, channels_out, 3, padding=1)

        # GroupNorm — stable at small spatial lengths unlike BatchNorm
        num_groups = min(8, channels_out)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels_out)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels_out)

        # Downsample / upsample
        if downsample:
            self.final = nn.Conv1d(channels_out, channels_out, 4, 2, 1)
        else:
            self.final = nn.ConvTranspose1d(channels_out, channels_out, 4, 2, 1)

        self.relu = nn.ReLU()

        # Optional self-attention (applied at deep levels only)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention1d(channels_out, num_heads=4)

    def forward(self, x, t, **kwargs):
        # First conv: conv -> GroupNorm -> ReLU
        o = self.relu(self.gnorm1(self.conv1(x)))

        # Time embedding
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time.unsqueeze(2)

        # Head anthropometric conditioning
        if self.head_embedding:
            head_meas = kwargs.get('head_embedding')
            o_head = self.head_mlp(self.head_measurement_embedding(head_meas.float()))
            o = o + o_head.unsqueeze(2)

        # Ear anthropometric conditioning
        if self.ears_embedding:
            ear_meas = kwargs.get('ears_embedding')
            o_ears = self.ears_mlp(self.ears_measurement_embedding(ear_meas.float()))
            o = o + o_ears.unsqueeze(2)

        # DOA label conditioning
        if self.labels:
            label = kwargs.get('labels')
            o_label = self.label_emb(label).squeeze(1)
            o = o + o_label.unsqueeze(2)

        # Second conv: conv -> GroupNorm (no activation, per paper)
        o = self.gnorm2(self.conv2(o))

        # Self-attention (bottleneck + deepest two encoder levels)
        if self.use_attention:
            o = self.attn(o)

        return self.final(o)


class UNet(nn.Module):
    """
    U-Net with:
      - Reduced channel sequence (32,64,128,256,512) for speed
      - Self-attention at the 3 deepest levels (last 2 encoder + bottleneck connection)
      - GroupNorm throughout
      - Dual conv per block matching the paper
    """
    def __init__(self, audio_channels=2, time_embedding_dims=256,
                 labels=False, head_embedding=False, ears_embedding=False,
                 sequence_channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims

        n_levels = len(sequence_channels) - 1  # 4 transitions

        # Attention at the deepest 2 downsampling and 2 upsampling blocks
        def use_attn_down(i):
            return i >= (n_levels - 2)   # last 2 encoder blocks

        def use_attn_up(i):
            return i < 2                  # first 2 decoder blocks (mirror)

        self.downsampling = nn.ModuleList([
            Block(
                channels_in=sequence_channels[i],
                channels_out=sequence_channels[i + 1],
                time_embedding_dims=time_embedding_dims,
                labels=labels,
                head_embedding=head_embedding,
                ears_embedding=ears_embedding,
                downsample=True,
                use_attention=use_attn_down(i),
            )
            for i in range(n_levels)
        ])

        seq_rev = sequence_channels[::-1]
        self.upsampling = nn.ModuleList([
            Block(
                channels_in=seq_rev[i],
                channels_out=seq_rev[i + 1],
                time_embedding_dims=time_embedding_dims,
                labels=labels,
                head_embedding=head_embedding,
                ears_embedding=ears_embedding,
                downsample=False,
                use_attention=use_attn_up(i),
            )
            for i in range(n_levels)
        ])

        self.conv1 = nn.Conv1d(audio_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv1d(sequence_channels[0], audio_channels, 1)

    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)
        return self.conv2(o)
