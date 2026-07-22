import torch
from torch import nn
import math
import torchvision


class DiffusionModel:
    """
    DDPM — 600 timesteps, linear schedule β: 1e-4→0.02 (paper Sec. IV-A).
    Training cost is independent of T; only inference is affected.
    """
    def __init__(self, start_schedule=1e-4, end_schedule=0.02, timesteps=600):
        self.start_schedule = start_schedule
        self.end_schedule   = end_schedule
        self.timesteps      = timesteps

        self.betas           = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas          = 1 - self.betas
        self.alphas_cumprod  = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_0, t, device='cpu'):
        noise = torch.randn_like(x_0)
        sqrt_acp   = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_1macp = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
        return sqrt_acp.to(device) * x_0.to(device) + sqrt_1macp.to(device) * noise.to(device), noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        betas_t        = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_1macp_t   = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_t   = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)

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
    """Sinusoidal timestep embedding (standard DDPM formulation)."""
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
    """
    Self-attention for 1D feature maps.
    Paper: 4 attention heads after every downsampling block (Sec. III-C).
    Pre-norm + residual — not stated in paper but standard practice;
    kept as an improvement since paper does not specify otherwise.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        # GroupNorm pre-norm — improvement over no norm (paper unspecified)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)      # (B, L, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)           # residual


class ImageEncoder(nn.Module):
    """
    Encodes a subject's L/R ear-crop images into one compact feature vector,
    computed ONCE per sample -- not once per block -- then fed into every
    block exactly like the ear-measurement vector (see Block.image_fc).

    Uses a frozen, ImageNet-pretrained MobileNetV2 as a feature extractor
    rather than a CNN trained from scratch.

    L and R images go through the SAME shared backbone separately (it
    expects standard 3-channel input, matching its pretrained weights --
    stacking L+R into 6 channels the way the from-scratch version did
    would destroy the meaning of the pretrained first-layer weights) and
    their pooled features are concatenated before the projection head.
    Expects images as (B, 2, 3, H, W): dim 1 is [L, R].
    """
    def __init__(self, out_dim):
        super().__init__()
        backbone = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features   # conv stack only, no classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.last_channel    # 1280 for mobilenet_v2

        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        self.fc = nn.Linear(feat_dim * 2, out_dim)   # *2: L and R concatenated

    def train(self, mode=True):
        # Keep the frozen backbone's BatchNorm layers in eval mode (fixed
        # running stats) even when the surrounding UNet is set to .train() --
        # otherwise those stats would drift to this tiny, ear-photo-only
        # dataset instead of staying at their ImageNet-trained values.
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, images):
        left, right = images[:, 0], images[:, 1]
        with torch.no_grad():   # frozen -- no need to build a backward graph
            feat_l = self.pool(self.features(left)).flatten(1)
            feat_r = self.pool(self.features(right)).flatten(1)
        return self.fc(torch.cat([feat_l, feat_r], dim=1))


class Block(nn.Module):
    """
    U-Net block per paper Sec. III-C:
      (i)  Conv1d k=3 pad=1 → BN → ReLU
      (ii) Conv1d k=3 pad=1 → BN  (no activation)
      + conditioning injections (time, DOA label, head, ear, image)
      + downsample: Conv1d k=4 stride=2  /  upsample: ConvTranspose1d k=4 stride=2
      + self-attention after every downsampling block (paper)

    Normalisation: paper states BatchNorm. We use GroupNorm (improvement)
    because BatchNorm degrades at small spatial lengths in deep encoder
    levels (L=16 at the bottleneck for the default 4-block encoder — see
    UNet docstring for the full level-by-level breakdown), and paper does
    not justify the choice.

    Conditioning fusion: paper Sec. III-C concatenates the conditioning
    embeddings with the feature map (channel-wise) rather than adding them.
    Each conditioning signal (time, DOA label, head, ear) is projected to
    channels_out, broadcast across the sequence length, concatenated with
    the post-conv1 feature map along the channel dimension, and fused back
    down to channels_out via a 1x1 conv. This replaces the previous
    addition-based injection (discrepancy #8 vs. the paper).
    """
    def __init__(self, channels_in, channels_out, time_embedding_dims,
                 labels, head_dim, ear_dim, image_dim,
                 downsample=True, use_attention=False):
        super().__init__()

        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.downsample     = downsample

        # ── Conditioning projections (FC layers concatenated to feature maps, Sec.III-C) ─
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)

        if labels:
            self.label_emb = nn.Embedding(labels, channels_out)
        self.labels = labels

        if head_dim:
            self.head_fc = nn.Sequential(
                nn.Linear(head_dim, time_embedding_dims),
                nn.ReLU(),
                nn.Linear(time_embedding_dims, channels_out),
            )
        self.head_dim = head_dim

        if ear_dim:
            self.ear_fc = nn.Sequential(
                nn.Linear(ear_dim, time_embedding_dims),
                nn.ReLU(),
                nn.Linear(time_embedding_dims, channels_out),
            )
        self.ear_dim = ear_dim

        if image_dim:
            self.image_fc = nn.Sequential(
                nn.Linear(image_dim, time_embedding_dims),
                nn.ReLU(),
                nn.Linear(time_embedding_dims, channels_out),
            )
        self.image_dim = image_dim

        # ── Convolutional layers (paper Sec. III-C, both k=3 pad=1) ───────────
        in_ch = channels_in if downsample else 2 * channels_in
        self.conv1 = nn.Conv1d(in_ch,          channels_out, 3, padding=1)
        self.conv2 = nn.Conv1d(channels_out,   channels_out, 3, padding=1)

        # ── Conditioning fusion (concatenation, not addition) ─────────────────
        # Tensors concatenated along the channel dim: the feature map itself,
        # plus time (always active), plus label/head/ear/image if active.
        n_concat_tensors = (2 + int(bool(labels)) + int(bool(head_dim))
                             + int(bool(ear_dim)) + int(bool(image_dim)))
        self.cond_fuse = nn.Conv1d(channels_out * n_concat_tensors, channels_out, kernel_size=1)

        # GroupNorm (improvement over paper's BatchNorm — see docstring)
        ng = min(8, channels_out)
        self.gn1 = nn.GroupNorm(ng, channels_out)
        self.gn2 = nn.GroupNorm(ng, channels_out)

        self.relu = nn.ReLU()

        # ── Stride conv / transposed conv (paper: k=4 stride=2) ───────────────
        if downsample:
            self.resample = nn.Conv1d(channels_out, channels_out, 4, stride=2, padding=1)
        else:
            self.resample = nn.ConvTranspose1d(channels_out, channels_out, 4, stride=2, padding=1)

        # ── Self-attention (paper: after every downsampling block, 4 heads) ────
        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention1d(channels_out, num_heads=4)

    def forward(self, x, t, **kwargs):
        # (i) conv → GroupNorm → ReLU
        o = self.relu(self.gn1(self.conv1(x)))
        L = o.shape[-1]

        # Project every active conditioning signal to channels_out and
        # broadcast across the sequence length, then concatenate with o
        # along the channel dimension (paper: concatenation, not addition).
        cond_feats = [o]

        time_feat = self.relu(self.time_mlp(self.time_embedding(t)))       # (B, channels_out)
        cond_feats.append(time_feat.unsqueeze(2).expand(-1, -1, L))

        if self.head_dim:
            head_feat = self.head_fc(kwargs['head_embedding'].float())      # (B, channels_out)
            cond_feats.append(head_feat.unsqueeze(2).expand(-1, -1, L))

        if self.ear_dim:
            ear_feat = self.ear_fc(kwargs['ears_embedding'].float())        # (B, channels_out)
            cond_feats.append(ear_feat.unsqueeze(2).expand(-1, -1, L))

        if self.image_dim:
            img_feat = self.image_fc(kwargs['image_embedding'].float())     # (B, channels_out)
            cond_feats.append(img_feat.unsqueeze(2).expand(-1, -1, L))

        if self.labels:
            label_feat = self.label_emb(kwargs['labels']).view(o.shape[0], -1)  # (B, channels_out)
            cond_feats.append(label_feat.unsqueeze(2).expand(-1, -1, L))

        o = torch.cat(cond_feats, dim=1)     # (B, channels_out * n_cond, L)
        o = self.cond_fuse(o)                # fuse back down to channels_out

        # (ii) conv → GroupNorm  (no activation — paper)
        o = self.gn2(self.conv2(o))

        # Self-attention (downsampling blocks only, per paper)
        if self.use_attention:
            o = self.attn(o)

        return self.resample(o)


class UNet(nn.Module):
    """
    U-Net matching paper Sec. III-C as closely as possible:
      - 4 encoder blocks (see note below on "5 encoder blocks" in the
        paper vs. this code), channel sizes (4, 8, 16, 32, 64) × base_channels
      - 4 decoder blocks mirroring the encoder
      - Self-attention with 4 heads, by default on the two deepest
        encoder blocks + bottleneck only (see attn_full_encoder below)
      - Skip connections via concatenation
      - Conditioning injected at every block via FC layers, fused by
        channel-wise concatenation (paper), not addition

    base_channels: scalar multiplier. Paper states (4,8,16,32,64); since those
    are very narrow for a 2-channel 256-sample signal we expose this as a
    parameter. Default=1 matches the paper literally. Set to e.g. 8 to get
    (32,64,128,256,512) if more capacity is needed.

    attn_full_encoder: ablation switch (default False). When False
    (default), attention is restricted to encoder blocks 2-3 (indices
    >= 2 of 4) — the two deepest/most-compressed blocks — per the
    "~4-6x speedup, negligible quality loss" rationale in the encoder
    construction below. Setting attn_full_encoder=True restores attention 
    on *all* 4 encoder blocks (matching the paper's "after every downsampling 
    block"), for a controlled comparison of final LSD between the two placements.

    keep_5fold is preserved — only architecture changes.

    Conditioning: head_dim, ear_dim, and image_dim are each independently
    switchable (0 = branch fully disabled, no weights created, no kwarg
    read) — this is what lets the same Block/UNet code serve all four
    SONICOM ablation conditions plus the HUTUBS baseline:
        A unconditioned:    ear_dim=0,  image_dim=0
        B anthro-only:      ear_dim=24, image_dim=0   (= HUTUBS baseline)
        C image-only:       ear_dim=0,  image_dim=N
        D anthro + image:   ear_dim=24, image_dim=N
    head_dim stays 0 in all of the above (neither dataset has head/torso
    data — see dataset.py) but is kept as a parameter for possible future use.
    image_dim is the width of the feature vector ImageEncoder produces, not
    a pixel count — the encoder itself lives on the UNet, not each Block,
    and runs once per sample (see forward()).
    """
    CHANNEL_MULTS = (4, 8, 16, 32, 64)   # paper Sec. III-C

    def __init__(self, audio_channels=2, time_embedding_dims=256,
                 labels=440, head_dim=0, ear_dim=24, image_dim=0,
                 base_channels=8, attn_full_encoder=False):
        super().__init__()

        seq = [m * base_channels for m in self.CHANNEL_MULTS]  # 5 levels
        n   = len(seq)   # 5 channel levels -> 4 encoder/decoder blocks (see docstring)

        # Stem: map audio channels → first channel size
        self.stem   = nn.Conv1d(audio_channels, seq[0], 3, padding=1)
        self.out    = nn.Conv1d(seq[0], audio_channels, 1)

        # Image encoder runs once per sample (not once per block, unlike
        # the ear/head FC layers which are cheap enough to duplicate per
        # block) -- its output feeds into every block via image_embedding,
        # the same way ears_embedding does. image_dim=0 disables it.
        self.image_encoder = ImageEncoder(out_dim=image_dim) if image_dim else None

        common = dict(
            time_embedding_dims=time_embedding_dims,
            labels=labels,
            head_dim=head_dim,
            ear_dim=ear_dim,
            image_dim=image_dim,
        )

        # Encoder: by default, attention only on the two deepest blocks
        # (indices 2,3 of 4 — see docstring for exact spatial lengths).
        # Paper specifies attention after every block; attn_full_encoder
        # restores that behaviour for the ablation comparing final LSD
        # between "attention everywhere" and this default placement.
        self.attn_full_encoder = attn_full_encoder
        self.encoder = nn.ModuleList([
            Block(
                seq[i], seq[i+1], downsample=True,
                use_attention=(True if attn_full_encoder else (i >= 2)),
                **common,
            )
            for i in range(n - 1)
        ])

        # Bottleneck: same-channel block with attention, no skip concat,
        # no spatial change. Use a dedicated small module to avoid the
        # 2*C_in skip-concat assumption in Block's decoder path.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(seq[-1], seq[-1], 3, padding=1),
            nn.GroupNorm(min(8, seq[-1]), seq[-1]),
            nn.ReLU(),
            nn.Conv1d(seq[-1], seq[-1], 3, padding=1),
            nn.GroupNorm(min(8, seq[-1]), seq[-1]),
        )
        self.bottleneck_attn = SelfAttention1d(seq[-1], num_heads=4)

        # Decoder: mirrors encoder; skip concat doubles input channels
        seq_rev = seq[::-1]
        self.decoder = nn.ModuleList([
            Block(seq_rev[i], seq_rev[i+1], downsample=False,
                  use_attention=False, **common)   # paper only specifies attn on encoder
            for i in range(n - 1)
        ])

    def forward(self, x, t, **kwargs):
        # Encode images once (if this run uses them) and hand the resulting
        # feature vector to every block through the same kwargs dict used
        # for labels/head/ear -- Blocks never touch raw pixels themselves.
        if self.image_encoder is not None:
            kwargs['image_embedding'] = self.image_encoder(kwargs['images'])

        o = self.stem(x)
        skips = []
        for enc in self.encoder:
            o = enc(o, t, **kwargs)
            skips.append(o)

        o = self.bottleneck_attn(self.bottleneck(o))

        for dec, skip in zip(self.decoder, reversed(skips)):
            o = dec(torch.cat([o, skip], dim=1), t, **kwargs)

        return self.out(o)