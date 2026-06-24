import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_noise_distribution(noise, predicted_noise, epoch, plot_path=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    axes[0].plot(noise.cpu().numpy()[0, 0], label='GT Noise L', linewidth=0.5, marker='o', markersize=1)
    axes[0].plot(noise.cpu().numpy()[0, 1], label='GT Noise R', linewidth=0.5, marker='o', markersize=1)
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(predicted_noise.cpu().numpy()[0, 0], label='Pred Noise L', linewidth=0.5, marker='o', markersize=1)
    axes[1].plot(predicted_noise.cpu().numpy()[0, 1], label='Pred Noise R', linewidth=0.5, marker='o', markersize=1)
    axes[1].grid()
    axes[1].legend()

    axes[2].hist(noise.cpu().numpy().flatten(), density=True, alpha=0.8, label='Ground Truth Noise')
    axes[2].hist(predicted_noise.cpu().numpy().flatten(), density=True, alpha=0.8, label='Predicted Noise')
    axes[2].legend()

    fig.suptitle(f'Noise distribution — epoch {epoch}')

    if plot_path:
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()


def nmse(hrir_test, hrir_gen):
    """Normalised Mean Squared Error, averaged across L and R channels."""
    sq_error_left = torch.mean((hrir_test[0] - hrir_gen[0]) ** 2)
    sq_error_right = torch.mean((hrir_test[1] - hrir_gen[1]) ** 2)

    power_left = torch.mean(hrir_test[0] ** 2)
    power_right = torch.mean(hrir_test[1] ** 2)

    nmse_left = sq_error_left / (power_left + 1e-12)
    nmse_right = sq_error_right / (power_right + 1e-12)

    return (nmse_left + nmse_right) / 2


def lsd(hrir_test, hrir_gen, points, sr, plot=False):
    """
    Log-Spectral Distortion between lists of test and generated HRIRs.
    Uses left channel only (index 0) per the paper's formulation.
    """
    hrtf_gen_list = []
    hrtf_test_list = []

    for point in range(points):
        hrtf_gen_list.append(np.fft.fft(hrir_gen[point][0].numpy()
                                         if isinstance(hrir_gen[point][0], torch.Tensor)
                                         else hrir_gen[point][0]))
        hrtf_test_list.append(np.fft.fft(hrir_test[point][0].numpy()
                                           if isinstance(hrir_test[point][0], torch.Tensor)
                                           else hrir_test[point][0]))

    H_gen = np.array(hrtf_gen_list)
    H_test = np.array(hrtf_test_list)
    K = H_gen.shape[1]

    # Avoid log(0)
    eps = 1e-12
    log_ratio = 20 * np.log10((np.abs(H_gen) + eps) / (np.abs(H_test) + eps))
    squared_diffs = np.sum(log_ratio ** 2) / (points * K)
    LSD = np.sqrt(squared_diffs)

    return LSD
