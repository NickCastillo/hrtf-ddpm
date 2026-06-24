import matplotlib.pyplot as plt
import numpy as np
import torch

# ── LSD constants matching paper Eq. 9 ───────────────────────────────────────
SR       = 44100
HRIR_LEN = 256
K_BANDS  = 44      # paper: K=44 frequency bins
F_MAX    = 15000   # paper: 0–15 kHz


def plot_noise_distribution(noise, predicted_noise, epoch, plot_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(noise.cpu().numpy()[0, 0], label='GT L',   linewidth=0.5, marker='o', markersize=1)
    axes[0].plot(noise.cpu().numpy()[0, 1], label='GT R',   linewidth=0.5, marker='o', markersize=1)
    axes[0].set_title('GT Noise'); axes[0].grid(); axes[0].legend()

    axes[1].plot(predicted_noise.cpu().numpy()[0, 0], label='Pred L', linewidth=0.5, marker='o', markersize=1)
    axes[1].plot(predicted_noise.cpu().numpy()[0, 1], label='Pred R', linewidth=0.5, marker='o', markersize=1)
    axes[1].set_title('Predicted Noise'); axes[1].grid(); axes[1].legend()

    axes[2].hist(noise.cpu().numpy().flatten(),           density=True, alpha=0.8, label='GT')
    axes[2].hist(predicted_noise.cpu().numpy().flatten(), density=True, alpha=0.8, label='Pred')
    axes[2].set_title('Distribution'); axes[2].legend()

    fig.suptitle(f'Noise distribution — epoch {epoch}')
    if plot_path:
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def nmse(hrir_test, hrir_gen):
    """NMSE averaged over L and R channels."""
    out = 0.0
    for ch in range(2):
        num = torch.mean((hrir_test[ch] - hrir_gen[ch]) ** 2)
        den = torch.mean(hrir_test[ch] ** 2) + 1e-12
        out += num / den
    return out / 2


def lsd(hrir_test, hrir_gen, points, sr=SR):
    """
    Log-Spectral Distortion — paper Eq. 9.

    LSD(H, H_hat) = sqrt( 1/(L*K) * sum_l sum_k (20 log10 |H(r_l,k)| / |H_hat(r_l,k)|)^2 )

    Matches paper exactly:
      - K=44 frequency bands evenly spaced between 0 and 15 kHz
      - Averaged over both L (ch 0) and R (ch 1) ears
      - hrir_test / hrir_gen: lists of tensors or arrays, each shape (2, HRIR_LEN)
    """
    freqs     = np.fft.rfftfreq(HRIR_LEN, d=1.0 / sr)          # 129 bins
    band_mask = freqs <= F_MAX
    band_idx  = np.where(band_mask)[0]
    # Evenly subsample to exactly K_BANDS within [0, F_MAX]
    sel = band_idx[np.linspace(0, len(band_idx) - 1, K_BANDS, dtype=int)]

    # Stack into arrays: (n_points, 2, HRIR_LEN)
    def to_np(x):
        return x.numpy() if isinstance(x, torch.Tensor) else np.array(x)

    gt  = np.stack([to_np(h) for h in hrir_test],  axis=0).astype(np.float32)
    gen = np.stack([to_np(h) for h in hrir_gen],   axis=0).astype(np.float32)

    eps   = 1e-12
    total = 0.0
    for ch in range(2):
        H_gt  = np.abs(np.fft.rfft(gt[:,  ch, :]))[:, sel]   # (n_points, K)
        H_gen = np.abs(np.fft.rfft(gen[:, ch, :]))[:, sel]
        log_r = 20 * np.log10((H_gt + eps) / (H_gen + eps))
        total += np.sum(log_r ** 2)

    return float(np.sqrt(total / (2 * points * K_BANDS)))
