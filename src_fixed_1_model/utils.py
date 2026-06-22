"""
utils.py
Fixes applied vs original:
  - hrir2hrtf: 'plot_path' parameter unused → replaced with explicit plot_dir.
  - energy_loss renamed to energy_loss_fn to avoid name collision in main.py.
  - plt.show() removed (blocks Colab); save before close.
  - matplotlib Agg backend set for server/Colab safety.

Binaural fixes:
  - hrir2hrtf now processes BOTH ears independently.
    Input tensors are (N, 2, L); left=channel 0, right=channel 1.
  - Returns separate dB arrays for left and right:
        hrtf_pred_db_l, hrtf_test_db_l,
        hrtf_pred_db_r, hrtf_test_db_r,
        positions
  - Plots saved for both ears with _l / _r suffixes.
  - error_freq_binaural computes mean LSD across both ears.
  - plot_noise_distribution already plotted both channels; labels clarified.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_noise_distribution(noise, predicted_noise, epoch, plot_path=None):
    """
    noise / predicted_noise: (B, 1, L)  — single-ear tensors passed from main.py
    Both channels of the original binaural tensor are plotted by main.py calling
    this function twice (once per ear), or by passing the full (B, 2, L) tensor
    in which case channels 0 and 1 are shown side-by-side.
    """
    n = noise.cpu().numpy()
    p = predicted_noise.cpu().numpy()
    n_ch = n.shape[1]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    for ch, label in zip(range(n_ch), ['L', 'R']):
        axes[0].plot(n[0, ch], label=f'GT {label}',   linewidth=0.5, marker='o', markersize=1)
        axes[1].plot(p[0, ch], label=f'Pred {label}', linewidth=0.5, marker='o', markersize=1)

    axes[0].set_title('Ground Truth Noise'); axes[0].grid(); axes[0].legend()
    axes[1].set_title('Predicted Noise');    axes[1].grid(); axes[1].legend()
    axes[2].hist(n.flatten(), density=True, alpha=0.8, label='GT Noise')
    axes[2].hist(p.flatten(), density=True, alpha=0.8, label='Pred Noise')
    axes[2].legend()

    fig.suptitle(f'Noise distribution – epoch {epoch}')
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
    plt.close()


def hrir2hrtf(hrir_test, hrir_pred, subject_id, plot_dir=None):
    """
    Convert binaural HRIR tensors to HRTF magnitude (dB) at critical bands.

    Parameters
    ----------
    hrir_test, hrir_pred : torch.Tensor  shape (N_points, 2, L)
                           channel 0 = left ear, channel 1 = right ear
                           (padding already trimmed before calling)
    subject_id           : int
    plot_dir             : str | None

    Returns
    -------
    hrtf_pred_db_l : np.ndarray  (N_points, n_bands)  left  predicted
    hrtf_test_db_l : np.ndarray  (N_points, n_bands)  left  ground truth
    hrtf_pred_db_r : np.ndarray  (N_points, n_bands)  right predicted
    hrtf_test_db_r : np.ndarray  (N_points, n_bands)  right ground truth
    positions      : np.ndarray  random point indices used for plots
    """
    def _to_np(t, ch):
        arr = t[:, ch, :].numpy() if isinstance(t, torch.Tensor) else t[:, ch, :]
        return arr

    def _db_bands(hrir_np, K, nearest_indices):
        hrtf   = np.fft.fft(hrir_np)
        db     = 20 * np.log10(np.abs(hrtf) + 1e-12)
        return hrtf, db, db[:, nearest_indices]

    critical_bands = np.array([
        200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
        2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
    ])

    # Use left ear for freq axis (both ears have same L)
    L = hrir_test.shape[2]
    K = np.fft.fftfreq(L, 1 / 44100)
    nearest_indices = np.array([np.abs(K - freq).argmin() for freq in critical_bands])

    test_l_np = _to_np(hrir_test, 0)
    pred_l_np = _to_np(hrir_pred, 0)
    test_r_np = _to_np(hrir_test, 1)
    pred_r_np = _to_np(hrir_pred, 1)

    hrtf_test_l, db_test_l, hrtf_test_db_l = _db_bands(test_l_np, K, nearest_indices)
    hrtf_pred_l, db_pred_l, hrtf_pred_db_l = _db_bands(pred_l_np, K, nearest_indices)
    hrtf_test_r, db_test_r, hrtf_test_db_r = _db_bands(test_r_np, K, nearest_indices)
    hrtf_pred_r, db_pred_r, hrtf_pred_db_r = _db_bands(pred_r_np, K, nearest_indices)

    n_plots   = min(10, hrir_test.shape[0])
    positions = np.random.choice(hrir_test.shape[0], size=n_plots, replace=False).astype(int)

    if plot_dir is not None:
        subj_dir = os.path.join(plot_dir, f'sub_{subject_id}')
        os.makedirs(subj_dir, exist_ok=True)

        for pos in positions:
            for ear, (db_t, db_p, hrir_t, hrir_p, tag) in enumerate([
                (db_test_l, db_pred_l, test_l_np, pred_l_np, 'l'),
                (db_test_r, db_pred_r, test_r_np, pred_r_np, 'r'),
            ]):
                side = 'Left' if tag == 'l' else 'Right'

                # HRTF magnitude
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(K[:L // 2], db_t[pos, :L // 2], label='Test')
                ax.plot(K[:L // 2], db_p[pos, :L // 2], label='Predicted')
                ax.set_title(f'{side} HRTF – source position {pos}')
                ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude (dB)')
                ax.set_xlim(20, 20000); ax.set_xscale('log')
                ax.grid(True); ax.legend()
                plt.savefig(os.path.join(subj_dir, f'hrtf_{tag}_pos{pos}.jpg'))
                plt.close()

                # HRIR time domain
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(hrir_p[pos], label='Predicted')
                ax.plot(hrir_t[pos], label='Sample', linestyle='dashed')
                ax.grid(True); ax.legend()
                ax.set_title(f'{side} HRIR – position {pos}')
                ax.set_xlabel('Sample Index'); ax.set_ylabel('Amplitude')
                plt.savefig(os.path.join(subj_dir, f'hrir_{tag}_pos{pos}.jpg'))
                plt.close()

    return hrtf_pred_db_l, hrtf_test_db_l, hrtf_pred_db_r, hrtf_test_db_r, positions


def error_freq(hrir_pred, hrir_test):
    """Squared log-spectral difference (single ear)."""
    return (hrir_pred.float() - hrir_test.float()) ** 2


def error_freq_binaural(pred_l, test_l, pred_r, test_r):
    """
    Mean LSD across both ears.
    Inputs are np.ndarray (N_points, n_bands).
    Returns a scalar tensor.
    """
    err_l = error_freq(torch.from_numpy(pred_l), torch.from_numpy(test_l))
    err_r = error_freq(torch.from_numpy(pred_r), torch.from_numpy(test_r))
    return torch.sqrt(torch.mean((err_l + err_r) / 2.0))


def energy_loss_fn(predicted, target):
    """Renamed from energy_loss to avoid shadowing the local tensor in main.py."""
    return torch.sum((torch.sum(predicted ** 2) - torch.sum(target ** 2)) ** 2)
