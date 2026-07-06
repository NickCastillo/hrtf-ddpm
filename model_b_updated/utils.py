import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
SR       = 44100
HRIR_LEN = 256
K_BANDS  = 44      # paper Eq. 9: K=44 frequency bins
F_MAX    = 15000   # paper: 0–15 kHz


# ── EMA (Exponential Moving Average of model weights) ─────────────────────────
class EMA:
    """
    Exponential moving average of a model's parameters.

    Diffusion models are known to benefit from evaluating/sampling with an
    EMA of the training weights rather than the raw (noisier) weights at
    any single step — the EMA copy tends to generalize better and produces
    smoother samples. Only floating-point parameters are tracked; GroupNorm
    (used throughout this U-Net instead of BatchNorm) has no running-stat
    buffers, so there is nothing else to average.

    Usage:
        ema = EMA(model, decay=0.999)
        ...
        optimizer.step()
        ema.update(model)              # call once per optimizer step
        ...
        ema.apply_shadow(model)        # swap in EMA weights (e.g. for val/inference)
        ... run validation / sampling ...
        ema.restore(model)             # swap raw training weights back
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


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


def _get_freq_bins(sr=SR):
    """Return the K=44 frequency bin indices used for LSD (shared by all callers)."""
    freqs     = np.fft.rfftfreq(HRIR_LEN, d=1.0 / sr)
    band_mask = freqs <= F_MAX
    band_idx  = np.where(band_mask)[0]
    return band_idx[np.linspace(0, len(band_idx) - 1, K_BANDS, dtype=int)]


def _to_np(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _stack(hrirs):
    """Stack list of (2, L) tensors/arrays → (N, 2, L) float32 array."""
    return np.stack([_to_np(h) for h in hrirs], axis=0).astype(np.float32)


def combined_loss(noise, predicted_noise, freq_weight=0.3, sr=SR, k_bands=K_BANDS, f_max=F_MAX):
    """
    Training loss combining the paper's time-domain term with a
    frequency-magnitude term.

    - Time term: L1 on the raw (noise, predicted_noise) signals — the
      paper's stated loss (Sec. III-C / discrepancy #5).
    - Frequency term: L1 on FFT magnitude, restricted to the *same* K=44
      frequency bands (0–15 kHz) used by the LSD evaluation metric
      (`lsd()` below, paper Eq. 9). This nudges training to directly
      reduce error in the bands the reported metric actually measures,
      rather than only minimizing uniform time-domain error.

    final_loss = (1 - freq_weight) * L1_time + freq_weight * L1_freq_mag

    noise, predicted_noise: (B, 2, L) tensors — ground-truth / predicted
    diffusion noise (epsilon), not the HRIR itself. FFT is computed in
    float32 regardless of the ambient autocast dtype for numerical safety.
    """
    l1_time = F.l1_loss(noise, predicted_noise)

    band_idx = torch.as_tensor(_get_freq_bins(sr=sr), device=noise.device, dtype=torch.long)

    noise_fft = torch.fft.rfft(noise.float(), dim=-1)
    pred_fft  = torch.fft.rfft(predicted_noise.float(), dim=-1)
    mag_noise = torch.abs(noise_fft).index_select(-1, band_idx)
    mag_pred  = torch.abs(pred_fft).index_select(-1, band_idx)
    l1_freq   = F.l1_loss(mag_noise, mag_pred)

    return (1 - freq_weight) * l1_time + freq_weight * l1_freq


def lsd(hrir_test, hrir_gen, points, sr=SR):
    """
    Log-Spectral Distortion — paper Eq. 9.
    K=44 bands 0–15 kHz.
    Returns dict with keys 'L', 'R', 'avg' (all floats, dB).
    'avg' matches the paper exactly (L and R averaged).
    """
    sel = _get_freq_bins(sr)
    gt  = _stack(hrir_test)
    gen = _stack(hrir_gen)
    eps = 1e-12
    lsd_per_ch = []
    for ch in range(2):
        H_gt  = np.abs(np.fft.rfft(gt[:,  ch, :]))[:, sel]
        H_gen = np.abs(np.fft.rfft(gen[:, ch, :]))[:, sel]
        sq    = np.sum((20 * np.log10((H_gt + eps) / (H_gen + eps))) ** 2)
        lsd_per_ch.append(float(np.sqrt(sq / (points * K_BANDS))))
    return {
        'L':   lsd_per_ch[0],
        'R':   lsd_per_ch[1],
        'avg': float(np.sqrt((lsd_per_ch[0]**2 + lsd_per_ch[1]**2) / 2)),
    }


# ── ITD ───────────────────────────────────────────────────────────────────────

def compute_itd(hrir, sr=SR, window_n=4, prominence=0.05e7):
    """
    Estimate ITD for a single HRIR via energy-based onset detection,
    matching the author's implementation exactly.

    Method (per author's notebook):
      1. Square the signal per channel → instantaneous energy
      2. Convolve with a Hann window of length N → local energy function
      3. Differentiate and half-wave rectify → energy novelty function
      4. Find first peak above prominence threshold → onset sample
      5. ITD = (onset_L - onset_R) / sr * 1e6  [µs]

    If no peak is found in a channel, falls back to argmax of the
    novelty function to avoid NaN propagation.

    Returns ITD in microseconds. Positive = L leads R, Negative = R leads L.
    """
    h   = _to_np(hrir).astype(np.float64)   # (2, L)
    w   = scipy.signal.windows.hann(window_n)
    onsets = []

    for ch in range(2):
        sq      = h[ch] ** 2
        energy  = np.convolve(sq, w ** 2, mode='same')
        diff    = np.diff(energy)
        diff    = np.concatenate((diff, np.array([0.0])))
        novelty = np.where(diff > 0, diff, 0.0)   # half-wave rectify

        peaks, _ = scipy.signal.find_peaks(novelty, prominence=prominence)
        if len(peaks) > 0:
            onset = peaks[0]
        else:
            # Fallback: first maximum of novelty
            onset = int(np.argmax(novelty))
        onsets.append(onset)

    itd_samples = onsets[0] - onsets[1]             # L onset − R onset
    return float(itd_samples / sr * 1e6)             # → µs


def itd_error(hrir_test_list, hrir_gen_list, sr=SR):
    """
    Mean absolute ITD error over all positions (µs).
    hrir_test_list / hrir_gen_list: lists of (2, L) tensors or arrays.
    """
    errors = []
    for gt, gen in zip(hrir_test_list, hrir_gen_list):
        itd_gt  = compute_itd(gt,  sr)
        itd_gen = compute_itd(gen, sr)
        errors.append(abs(itd_gt - itd_gen))
    return float(np.mean(errors))


# ── PBC ───────────────────────────────────────────────────────────────────────

def _erb_filters(sr=SR, n_filters=40, f_low=50.0):
    """
    Generate centre frequencies for an ERB (Equivalent Rectangular Bandwidth)
    auditory filterbank spanning f_low to sr/2.

    ERB scale per Moore & Glasberg (1983):
        ERB(f) = 24.7 * (4.37*f/1000 + 1)

    Returns array of centre frequencies in Hz.
    """
    f_high  = sr / 2.0
    erb_low  = 24.7 * (4.37 * f_low  / 1000 + 1)
    erb_high = 24.7 * (4.37 * f_high / 1000 + 1)
    erbs     = np.linspace(erb_low, erb_high, n_filters)
    # Invert ERB → Hz
    cfs = (erbs / 24.7 - 1) / 4.37 * 1000
    return cfs


def _gammatone_response(freqs, cf, bw_factor=1.019):
    """
    Approximate gammatone filter magnitude response at given FFT frequencies.
    Uses the simplified Glasberg & Moore ERB bandwidth:
        ERB(cf) = 24.7 * (4.37*cf/1000 + 1)
    bw_factor = 1.019 is the standard correction for a 4th-order gammatone.
    """
    erb = 24.7 * (4.37 * cf / 1000 + 1)
    bw  = bw_factor * erb
    # 4th-order gammatone magnitude approximated as squared Lorentzian
    response = 1.0 / (1.0 + ((freqs - cf) / (bw / 2)) ** 2) ** 4
    return response / (response.max() + 1e-12)


def pbc(hrir_test_list, hrir_gen_list, sr=SR, n_filters=40):
    """
    Perceptual Blur Criterion (PBC) — binaural auditory model metric.

    Computes the mean spectral distortion between GT and generated HRTFs
    weighted by an ERB gammatone filterbank, giving perceptually-weighted
    frequency emphasis (more weight to speech-important mid frequencies).

    Formula per position p, channel c, filter k:
        D(p,c,k) = 20*log10( sum_f |H_gt(f)| * g_k(f) )
                          - 20*log10( sum_f |H_gen(f)| * g_k(f) )
        PBC = sqrt( mean_{p,c,k} D(p,c,k)^2 )   [dB]

    This is the standard formulation used in HRTF personalisation
    literature (Enzner 2008, Grigoriev et al.). Lower = more perceptually
    similar. Paper reports PBC alongside LSD and ITD (Sec. IV-B).

    hrir_test_list / hrir_gen_list: lists of (2, L) tensors or arrays.
    Returns scalar float (dB).
    """
    gt  = _stack(hrir_test_list)    # (N, 2, L)
    gen = _stack(hrir_gen_list)

    N = gt.shape[0]
    freqs = np.fft.rfftfreq(HRIR_LEN, d=1.0 / sr)   # (129,)
    cfs   = _erb_filters(sr=sr, n_filters=n_filters)  # (n_filters,)

    # Pre-compute filter responses: (n_filters, n_freqs)
    filters = np.stack([_gammatone_response(freqs, cf) for cf in cfs], axis=0)

    eps    = 1e-12
    total  = 0.0
    count  = 0

    for ch in range(2):
        H_gt  = np.abs(np.fft.rfft(gt[:,  ch, :]))   # (N, 129)
        H_gen = np.abs(np.fft.rfft(gen[:, ch, :]))   # (N, 129)

        for k in range(n_filters):
            g = filters[k]                             # (129,)
            # Weighted energy per position
            E_gt  = H_gt  @ g + eps                   # (N,)
            E_gen = H_gen @ g + eps                   # (N,)
            D = 20 * np.log10(E_gt / E_gen)           # (N,)
            total += np.sum(D ** 2)
            count += N

    return float(np.sqrt(total / count))