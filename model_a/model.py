import torch
from torch import nn
import math


class DiffusionModel:
    """DDPM — 600 timesteps, linear schedule β: 1e-4→0.02."""
    def __init__(self, start_schedule=1e-4, end_schedule=0.02, timesteps=600):
        self.start_schedule = start_schedule
        self.end_schedule   = end_schedule
        self.timesteps      = timesteps
        self.betas          = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas         = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_0, t, device='cpu'):
        noise      = torch.randn_like(x_0)
        sqrt_acp   = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_1macp = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
        return (sqrt_acp.to(device) * x_0.to(device) +
                sqrt_1macp.to(device) * noise.to(device)), noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        betas_t      = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_1macp_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        pred = model(x, t, **kwargs)
        mean = sqrt_recip_t * (x - betas_t * pred / sqrt_1macp_t)
        if t[0] == 0:
            return mean
        return mean + torch.sqrt(betas_t) * torch.randn_like(x)

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        out = values.gather(-1, t.long().cpu())
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(t.device)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SelfAttention1d(nn.Module):
    """Self-attention for 1D feature maps — 4 heads, GroupNorm pre-norm, residual."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)


class Block(nn.Module):
    """
    U-Net block — previous model (Model A):
      conv → GroupNorm → ReLU → conv → GroupNorm (no activation)
      + time / label / head / ear conditioning injections
      + stride-2 downsample or transposed upsample
      + optional self-attention
    """
    def __init__(self, channels_in, channels_out, time_embedding_dims,
                 labels, head_embedding, ears_embedding,
                 downsample=True, use_attention=False):
        super().__init__()

        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels         = labels
        self.head_embedding = head_embedding
        self.ears_embedding = ears_embedding
        self.downsample     = downsample

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

        in_ch = channels_in if downsample else 2 * channels_in
        self.conv1 = nn.Conv1d(in_ch,        channels_out, 3, padding=1)
        # Second conv with no activation (as stated in the paper)

        self.conv2 = nn.Conv1d(channels_out, channels_out, 3, padding=1)
         # GroupNorm — stable at small spatial lengths unlike BatchNorm

        ng = min(8, channels_out)
        self.gn1 = nn.GroupNorm(ng, channels_out)
        self.gn2 = nn.GroupNorm(ng, channels_out)

        self.relu = nn.ReLU()

        if downsample:
            self.resample = nn.Conv1d(channels_out, channels_out, 4, stride=2, padding=1)
        else:
            self.resample = nn.ConvTranspose1d(channels_out, channels_out, 4, stride=2, padding=1)
        # Optional self-attention (applied at deep levels only)
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention1d(channels_out, num_heads=4)

    def forward(self, x, t, **kwargs):
        # First conv: conv -> GroupNorm -> ReLU

        o = self.relu(self.gn1(self.conv1(x)))
        # Time embedding
        o = o + self.relu(self.time_mlp(self.time_embedding(t))).unsqueeze(2)

        # Head anthropometric conditioning
        if self.head_embedding:
            head_meas = kwargs.get('head_embedding')
            o = o + self.head_mlp(self.head_measurement_embedding(head_meas.float())).unsqueeze(2)

        # Ear anthropometric conditioning

        if self.ears_embedding:
            ear_meas = kwargs.get('ears_embedding')
            o = o + self.ears_mlp(self.ears_measurement_embedding(ear_meas.float())).unsqueeze(2)

        # DOA label conditioning

        if self.labels:
            label = kwargs.get('labels')
            o = o + self.label_emb(label).squeeze(1).unsqueeze(2)

        # Second conv: conv -> GroupNorm (no activation, per paper)
        o = self.gn2(self.conv2(o))
        # Self-attention (bottleneck + deepest two encoder levels)
        if self.use_attention:
            o = self.attn(o)

        return self.resample(o)


class UNet(nn.Module):
    """
    Previous model (Model A):
      - 4 encoder / 4 decoder blocks
      - Channel sequence: (32, 64, 128, 256, 512)
      - Self-attention at encoder blocks 3 & 4 only (deepest two)
      - GroupNorm throughout, dual conv per block
      - All HUTUBS anthropometric features: 13 head + 24 ear
    """
    def __init__(self, audio_channels=2, time_embedding_dims=256,
                 labels=False, head_embedding=False, ears_embedding=False,
                 sequence_channels=(32, 64, 128, 256, 512)):
        super().__init__()

        n_levels = len(sequence_channels) - 1   # 4

        common = dict(
            time_embedding_dims=time_embedding_dims,
            labels=labels,
            head_embedding=head_embedding,
            ears_embedding=ears_embedding,
        )

        # Attention at the deepest 2 encoder blocks only
        self.encoder = nn.ModuleList([
            Block(sequence_channels[i], sequence_channels[i + 1],
                  downsample=True,
                  use_attention=(i >= n_levels - 2),
                  **common)
            for i in range(n_levels)
        ])

        seq_rev = sequence_channels[::-1]
        self.decoder = nn.ModuleList([
            Block(seq_rev[i], seq_rev[i + 1],
                  downsample=False,
                  use_attention=(i < 2),   # mirror: first 2 decoder blocks
                  **common)
            for i in range(n_levels)
        ])

        self.stem = nn.Conv1d(audio_channels, sequence_channels[0], 3, padding=1)
        self.out  = nn.Conv1d(sequence_channels[0], audio_channels, 1)

    def forward(self, x, t, **kwargs):
        o = self.stem(x)
        skips = []
        for enc in self.encoder:
            o = enc(o, t, **kwargs)
            skips.append(o)
        for dec, skip in zip(self.decoder, reversed(skips)):
            o = dec(torch.cat([o, skip], dim=1), t, **kwargs)
        return self.out(o)