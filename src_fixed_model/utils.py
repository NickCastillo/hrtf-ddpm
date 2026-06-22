"""
utils.py — LSD computation, training diagnostics, and plotting helpers.

Two LSD variants are provided:

  lsd_paper(hrir_test_list, hrir_gen_list)
      Replicates the paper's formula (arxiv 2501.02871, Eq. 9 / Section IV-B):
        - K=44 frequency bands, linearly spaced between 0 and 15 kHz
        - One-sided FFT magnitude (np.abs of rfft)
        - Formula: sqrt( 1/(L*K) * sum_l sum_k [20 log10 |H_l_k / Ĥ_l_k|]^2 )
          where L = number of DOA locations, K = 44 bands
        - Binaural: applied independently to each ear, then averaged
      This is the metric to use when comparing against the paper's 5.1 dB.

      Note: the previous session's lsd_paper() delegated to spatialaudiometrics
      which uses 20–20000 Hz (~116 bins) — that does NOT match the paper.
      This implementation directly codes Eq. 9 with the paper's stated K and range.

  lsd_corrected(hrir_test_list, hrir_gen_list, channel=None)
      A technically clean variant for internal model comparison:
        - One-sided FFT bins 1..87 (excludes DC, capped at 15 kHz)
        - Covers ~172 Hz to ~14 987 Hz at 44100 Hz / 256 samples
        - Binaural: mean of left and right ear LSD
      Same frequency scope as lsd_paper() but using 87 bins instead of 44,
      giving finer frequency resolution. Use as your internal baseline metric.

Changes vs. previous session's utils.py:
  [FIX] lsd_paper() now implements the paper's formula directly (K=44, 0–15 kHz)
        instead of delegating to spatialaudiometrics (which used 20–20 kHz,
        ~116 bins — not what the paper describes).
  [FIX] spatialaudiometrics dependency removed entirely.
  The lsd_corrected() and plot/nmse helpers are unchanged.
"""

import matplotlib
matplotlib.use('Agg')   # headless-safe; no GUI pop-up in Colab / SSH
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
FS      = 44100   # HUTUBS sample rate
N_FFT   = 256     # HRIR length (samples)

# Paper's 44 frequency bands: linearly spaced from 0 to 15 000 Hz.
# We find the nearest FFT bin for each target frequency.
# rfftfreq(256) gives bins 0, 172.3, 344.5, ..., 22 050 Hz (129 bins).
_FREQS      = np.fft.rfftfreq(N_FFT, 1.0 / FS)           # shape (129,)
_TARGET_HZ  = np.linspace(0, 15000, 44)                  # 44 points, 0–15 kHz
PAPER_BINS  = np.array(
    [np.argmin(np.abs(_FREQS - f)) for f in _TARGET_HZ], dtype=int
)   # 44 unique bin indices into the one-sided FFT


# ---------------------------------------------------------------------------
# Noise-prediction diagnostic plot
# ---------------------------------------------------------------------------

def plot_noise_distribution(noise, predicted_noise, epoch,
                            plot_path=None, show=False):
    """
    Plot ground-truth vs. predicted noise waveforms and histograms.

    Parameters
    ----------
    noise, predicted_noise : Tensor  (B, 2, L)
    epoch : int
    plot_path : str or None   — save figure here if given.
    show : bool               — call plt.show() (disable in headless environments).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(noise.cpu().numpy()[0, 0], label='GT L',
                 linewidth=0.5, marker='o', markersize=1)
    axes[0].plot(noise.cpu().numpy()[0, 1], label='GT R',
                 linewidth=0.5, marker='o', markersize=1)
    axes[0].set_title('GT Noise'); axes[0].grid(); axes[0].legend()

    axes[1].plot(predicted_noise.cpu().numpy()[0, 0], label='Pred L',
                 linewidth=0.5, marker='o', markersize=1)
    axes[1].plot(predicted_noise.cpu().numpy()[0, 1], label='Pred R',
                 linewidth=0.5, marker='o', markersize=1)
    axes[1].set_title('Predicted Noise'); axes[1].grid(); axes[1].legend()

    axes[2].hist(noise.cpu().numpy().flatten(),
                 density=True, alpha=0.8, label='GT')
    axes[2].hist(predicted_noise.cpu().numpy().flatten(),
                 density=True, alpha=0.8, label='Pred')
    axes[2].set_title('Noise Distribution'); axes[2].legend()

    fig.suptitle(f'Noise distribution — epoch {epoch}')
    fig.tight_layout()

    # Save BEFORE show() — show() clears the figure in headless environments.
    if plot_path:
        plt.savefig(plot_path, dpi=100)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# NMSE (quick sanity-check metric)
# ---------------------------------------------------------------------------

def nmse(hrir_test, hrir_gen):
    """
    Normalised MSE averaged over left and right ears.
    hrir_test, hrir_gen : Tensor (2, L) — single subject, single DOA.
    """
    sq_l = torch.mean((hrir_test[0] - hrir_gen[0]) ** 2)
    sq_r = torch.mean((hrir_test[1] - hrir_gen[1]) ** 2)
    pw_l = torch.mean(hrir_test[0] ** 2)
    pw_r = torch.mean(hrir_test[1] ** 2)
    return ((sq_l / (pw_l + 1e-8)) + (sq_r / (pw_r + 1e-8))) / 2.0


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Version 1 — Paper-replicating LSD
# ---------------------------------------------------------------------------

def lsd_paper(hrir_test_list, hrir_gen_list):
    """
    LSD matching the paper's formula exactly (arxiv 2501.02871, Section IV-B).

    The paper states:
      "we evaluated K=44 frequency bands between 0 and 15 kHz"
      LSD = sqrt( 1/(L*K) * sum_l sum_k [20 log10 |H(r_l,k) / Ĥ(r_l,k)|]^2 )

    Implementation:
      - One-sided FFT magnitude via np.fft.rfft (includes DC at bin 0).
      - 44 bands selected as nearest FFT bins to linspace(0, 15000, 44).
      - Applied to each ear independently; result is the mean of L and R.
      - Outer average over DOAs and bands is taken jointly (one sqrt, not per-DOA).

    Parameters
    ----------
    hrir_test_list : list of array-like, each shape (2, L)
        Ground-truth HRIRs for one subject, one entry per DOA.
    hrir_gen_list  : list of array-like, each shape (2, L)
        Generated HRIRs in the same DOA order.

    Returns
    -------
    lsd_mean  : float  — mean binaural LSD in dB (0–15 kHz, K=44 bands).
    lsd_per_ear : tuple(float, float) — (LSD_left, LSD_right).
    """
    n_locs = len(hrir_test_list)
    if n_locs == 0:
        return float('nan'), (float('nan'), float('nan'))

    K = len(PAPER_BINS)   # 44
    eps = 1e-8

    lsd_ears = []
    for ch in (0, 1):  # left ear, then right ear
        sq_sum = 0.0
        for p in range(n_locs):
            h_test = _to_numpy(hrir_test_list[p])[ch]   # (L,)
            h_gen  = _to_numpy(hrir_gen_list[p])[ch]

            # One-sided FFT magnitude, select the 44 paper bands.
            mag_test = np.abs(np.fft.rfft(h_test))[PAPER_BINS]
            mag_gen  = np.abs(np.fft.rfft(h_gen))[PAPER_BINS]

            log_ratio = 20.0 * np.log10(
                (mag_test + eps) / (mag_gen + eps)
            )
            sq_sum += np.sum(log_ratio ** 2)

        # Paper formula: sqrt( 1/(L*K) * total_squared_sum )
        lsd_ears.append(float(np.sqrt(sq_sum / (n_locs * K))))

    lsd_mean = float(np.mean(lsd_ears))
    return lsd_mean, (lsd_ears[0], lsd_ears[1])


# ---------------------------------------------------------------------------
# Version 2 — Corrected LSD (same frequency scope as paper, finer resolution)
# ---------------------------------------------------------------------------

# Positive-frequency bins (no DC) up to 15 kHz, matching the paper's upper
# frequency limit. At 44100 Hz / 256 samples: bins 1..87, covering
# ~172 Hz to ~14 987 Hz (87 bins vs the paper's 44 selected bands).
_CORRECTED_BINS = np.array(
    [i for i, f in enumerate(_FREQS) if 0 < f <= 15000], dtype=int
)   # shape (87,)


def lsd_corrected(hrir_test_list, hrir_gen_list, channel=None):
    """
    Corrected LSD for internal model comparison.

    Fixes vs. the original uploaded lsd():
      Bug 1: used all 256 FFT bins (DC + negative freqs) — denominator 2.2×
             too large, underestimating per-bin distortion.
             Fix: positive-frequency bins only, no DC, capped at 15 kHz
             (bins 1..87, ~172 Hz to ~14 987 Hz).
      Bug 2: only evaluated left ear (channel[0]).
             Fix: evaluate both ears and return their mean by default.

    Frequency scope matches lsd_paper() (0-15 kHz) but uses 87 bins instead
    of 44, giving finer resolution. The only remaining difference between the
    two variants is the number of frequency samples.

    Parameters
    ----------
    hrir_test_list, hrir_gen_list : list of array-like, each shape (2, L).
    channel : int or None
        None (default) -> binaural mean.  0 -> left only.  1 -> right only.

    Returns
    -------
    float  LSD in dB.
    """
    n_locs = len(hrir_test_list)
    if n_locs == 0:
        return float('nan')

    K = len(_CORRECTED_BINS)   # 87

    channels = [0, 1] if channel is None else [channel]
    eps = 1e-8
    lsd_per_ch = []

    for ch in channels:
        sq_sum = 0.0
        for p in range(n_locs):
            h_test = _to_numpy(hrir_test_list[p])[ch]
            h_gen  = _to_numpy(hrir_gen_list[p])[ch]

            mag_test = np.abs(np.fft.rfft(h_test))[_CORRECTED_BINS]
            mag_gen  = np.abs(np.fft.rfft(h_gen))[_CORRECTED_BINS]

            log_ratio = 20.0 * np.log10((mag_gen + eps) / (mag_test + eps))
            sq_sum += np.sum(log_ratio ** 2)

        lsd_per_ch.append(float(np.sqrt(sq_sum / (n_locs * K))))

    return float(np.mean(lsd_per_ch))
