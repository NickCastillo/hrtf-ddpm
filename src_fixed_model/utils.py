"""
utils.py
Fixes applied vs original:
  - hrir2hrtf: 'plot_path' parameter was declared but never used; the function
    built its own directory string as '' (empty) and tried to save into it.
    Now accepts an explicit plot_dir and uses it correctly; also returns a
    3-tuple consistent with how main.py calls it (hrtf_pred, hrtf_test, sample).
  - energy_loss renamed to energy_loss_fn to avoid collision with the local
    tensor variable of the same name used in main.py.
  - plot_noise_distribution: plt.show() removed (blocks execution in Colab/
    non-interactive sessions); saving now happens before plt.close().
"""

import matplotlib
matplotlib.use('Agg')          # non-interactive backend; safe for Colab & servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import scipy.io


def plot_noise_distribution(noise, predicted_noise, epoch, plot_path=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    axes[0].plot(noise.cpu().numpy()[0, 0], label='GT Noise L',   linewidth=0.5, marker='o', markersize=1)
    axes[0].plot(noise.cpu().numpy()[0, 1], label='GT Noise R',   linewidth=0.5, marker='o', markersize=1)
    axes[0].grid(); axes[0].legend()

    axes[1].plot(predicted_noise.cpu().numpy()[0, 0], label='Pred Noise L', linewidth=0.5, marker='o', markersize=1)
    axes[1].plot(predicted_noise.cpu().numpy()[0, 1], label='Pred Noise R', linewidth=0.5, marker='o', markersize=1)
    axes[1].grid(); axes[1].legend()

    axes[2].hist(noise.cpu().numpy().flatten(),           density=True, alpha=0.8, label='Ground Truth Noise')
    axes[2].hist(predicted_noise.cpu().numpy().flatten(), density=True, alpha=0.8, label='Predicted Noise')
    axes[2].legend()

    fig.suptitle(f'Noise distribution epoch: {epoch}')

    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
    plt.close()


def plot_hrir(hrir_test, hrir_pred, position, id, plot_path):
    plt.figure(figsize=(14, 5))
    plt.plot(hrir_test[0], label='Sample',    linestyle='dashed')
    plt.plot(hrir_pred[0], label='Predicted')
    plt.grid(True); plt.legend()
    plt.title(f'Left HRIR position: {position}')
    plt.xlabel('Sample Index'); plt.ylabel('Amplitude')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def hrir2hrtf(hrir_test, hrir_pred, subject_id, plot_dir=None):
    """
    Convert HRIR tensors to HRTF magnitude in dB at critical bands.

    Parameters
    ----------
    hrir_test, hrir_pred : torch.Tensor  shape (N_points, 1, L)  (already trimmed of padding)
    subject_id           : int   used only for plot filenames
    plot_dir             : str | None  if given, plots are saved here

    Returns
    -------
    hrtf_pred_db : np.ndarray  (N_points, n_bands)
    hrtf_test_db : np.ndarray  (N_points, n_bands)
    sample_idx   : np.ndarray  random positions used for plots
    """
    # squeeze channel dim → (N, L)
    hrir_test_np = hrir_test[:, 0, :].numpy() if isinstance(hrir_test, torch.Tensor) else hrir_test[:, 0, :]
    hrir_pred_np = hrir_pred[:, 0, :].numpy() if isinstance(hrir_pred, torch.Tensor) else hrir_pred[:, 0, :]

    hrtf_test = np.fft.fft(hrir_test_np)
    hrtf_pred = np.fft.fft(hrir_pred_np)

    hrtf_test_db = 20 * np.log10(np.abs(hrtf_test) + 1e-12)
    hrtf_pred_db = 20 * np.log10(np.abs(hrtf_pred) + 1e-12)

    critical_bands = np.array([
        200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
        2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
    ])

    L = hrtf_test.shape[1]
    K = np.fft.fftfreq(L, 1 / 44100)
    nearest_indices = np.array([np.abs(K - freq).argmin() for freq in critical_bands])

    hrtf_pred_db_bands = hrtf_pred_db[:, nearest_indices]
    hrtf_test_db_bands = hrtf_test_db[:, nearest_indices]

    n_plots = min(10, len(hrir_test_np))
    positions = np.random.choice(len(hrir_test_np), size=n_plots, replace=False).astype(int)

    if plot_dir is not None:
        subj_dir = os.path.join(plot_dir, f'sub_{subject_id}')
        os.makedirs(subj_dir, exist_ok=True)

        for position in positions:
            # HRTF magnitude plot
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(K[:L // 2], hrtf_test_db[position, :L // 2], label='Test')
            ax.plot(K[:L // 2], hrtf_pred_db[position, :L // 2], label='Predicted')
            ax.set_title(f'Left HRTF source position: {position}')
            ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude (dB)')
            ax.set_xlim(20, 20000); ax.set_xscale('log')
            ax.grid(True); ax.legend()
            plt.savefig(os.path.join(subj_dir, f'hrtf_l_pos_{position}.jpg'))
            plt.close()

            # HRIR time-domain plot
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(hrir_pred_np[position], label='Predicted')
            ax.plot(hrir_test_np[position], label='Sample', linestyle='dashed')
            ax.grid(True); ax.legend()
            ax.set_title(f'Left HRIR position: {position}')
            ax.set_xlabel('Sample Index'); ax.set_ylabel('Amplitude')
            plt.savefig(os.path.join(subj_dir, f'hrir_l_pos_{position}.jpg'))
            plt.close()

    return hrtf_pred_db_bands, hrtf_test_db_bands, positions


def error_freq(hrir_pred, hrir_test):
    """Squared log-spectral difference per point and band."""
    ratio = (hrir_pred.float() - hrir_test.float()) ** 2
    return ratio


# Renamed from energy_loss to energy_loss_fn to avoid shadowing the local
# tensor variable of the same name used during training in main.py.
def energy_loss_fn(predicted, target):
    return torch.sum((torch.sum(predicted ** 2) - torch.sum(target ** 2)) ** 2)
