"""
Configurable loss function for the L1 / L2 / combined / log_combined loss
ablation.

Kept as its own file, separate from main_model/utils.py, so the production
training script (main_model/main.py) is never touched by this experiment.
loss_exp/main.py imports compute_loss() from here instead of calling
utils.combined_loss() directly.

For loss_type='combined', this calls main_model/utils.py's own
combined_loss() rather than re-implementing the frequency-band math --
that guarantees byte-for-byte identical behaviour to the production loss
(including the norm='ortho' FFT fix), with zero risk of the two
implementations silently drifting apart over time.

loss_type='log_combined' is the new addition (see chat): same structure
as 'combined' -- (1 - freq_weight) * L1_time + freq_weight * L1_freq --
but the frequency term is computed on LOG-magnitude instead of linear
magnitude. Rationale: LSD/PBC (the actual evaluation metrics) measure
error as a log-ratio (20*log10(H_gt/H_gen)), which weights proportional/
relative spectral differences roughly equally whether they occur at a
loud peak or a quiet notch. The production 'combined' loss's frequency
term is linear-magnitude L1, dominated by the loudest parts of the
spectrum, and can under-fit quiet spectral notches (e.g. pinna notches)
that barely move a linear L1 loss but contribute heavily to LSD/PBC in dB
terms. log_combined aligns the training objective with that log-domain
evaluation criterion instead. See _log_magnitude_combined_loss()'s
docstring below for the one caveat worth knowing before you look at its
TensorBoard curve.
"""
import os
import sys

import torch
import torch.nn.functional as F

# ── Make main_model/ importable ──────────────────────────────────────────────
# loss_exp/ and main_model/ are assumed to be sibling folders:
#   hrtf-ddpm/
#     main_model/   (dataset.py, model.py, utils.py, main.py -- untouched)
#     loss_exp/     (this file + main.py)
_MAIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'main_model')
if _MAIN_MODEL_DIR not in sys.path:
    sys.path.insert(0, _MAIN_MODEL_DIR)

from utils import combined_loss as _production_combined_loss  # noqa: E402
from utils import _get_freq_bins  # noqa: E402  -- same K=44, 0-15kHz bins LSD/combined_loss use

LOSS_TYPES = ('l1', 'l2', 'combined', 'log_combined')


def _log_magnitude_combined_loss(noise, predicted_noise, freq_weight=0.3, eps=1e-8,
                                  return_components=False):
    """
    Same structure as main_model/utils.combined_loss (time-domain L1 term
    at weight (1 - freq_weight), frequency term at weight freq_weight, the
    same K=44 band-limited FFT bins as LSD via utils._get_freq_bins(), FFT
    computed with norm='ortho' for the same scale-matching reason
    documented in combined_loss's own docstring) -- but the frequency term
    is L1 on LOG-magnitude instead of linear magnitude:

        L1_freq_log = L1( log(|FFT(noise)| + eps), log(|FFT(pred_noise)| + eps) )

    CAVEAT worth knowing: this operates on the diffusion NOISE (epsilon),
    same as the production 'combined' loss -- not on a reconstructed HRIR
    estimate. Pure Gaussian noise's FFT magnitude has no consistent
    "spectral shape" the way clean audio does, and can land very close to
    zero in any given bin/sample purely by chance; taking log() there
    amplifies small absolute differences into large loss values and can
    make training noisier than the linear-magnitude version. The eps
    floor keeps this finite, but if this loss's curve looks unusually
    noisy/unstable on TensorBoard compared to 'combined', that's the
    likely reason. A more faithful (but more involved) version would
    reconstruct an x_0 estimate from (noise, predicted_noise, t, the
    diffusion schedule) and compute the log-magnitude term on that instead
    of on raw noise -- flagged here rather than built, since the direct
    swap was what was asked for first; ask if you want that version too
    once this one's been screened.
    """
    l1_time = F.l1_loss(noise, predicted_noise)

    band_idx = torch.as_tensor(_get_freq_bins(), device=noise.device, dtype=torch.long)

    noise_fft = torch.fft.rfft(noise.float(), dim=-1, norm='ortho')
    pred_fft  = torch.fft.rfft(predicted_noise.float(), dim=-1, norm='ortho')
    mag_noise = torch.abs(noise_fft).index_select(-1, band_idx)
    mag_pred  = torch.abs(pred_fft).index_select(-1, band_idx)

    log_mag_noise = torch.log(mag_noise + eps)
    log_mag_pred  = torch.log(mag_pred + eps)
    l1_freq_log   = F.l1_loss(log_mag_noise, log_mag_pred)

    loss = (1 - freq_weight) * l1_time + freq_weight * l1_freq_log
    if return_components:
        return loss, l1_time.detach(), l1_freq_log.detach()
    return loss


def compute_loss(noise, predicted_noise, loss_type, freq_weight=0.3, return_components=False):
    """
    noise, predicted_noise: (B, 2, L) ground-truth / predicted diffusion
    noise (epsilon) -- same convention as main_model/utils.combined_loss.

    loss_type:
      'l1'           -- plain time-domain L1 (mean absolute error).
      'l2'           -- plain time-domain L2 (mean squared error).
      'combined'     -- main_model's production loss: (1 - freq_weight) *
                        L1_time + freq_weight * L1_freq_mag (FFT
                        norm='ortho', same K=44 frequency bands as the LSD
                        metric, linear magnitude).
      'log_combined' -- same as 'combined' but the frequency term is
                        L1 on LOG-magnitude instead of linear magnitude
                        (see _log_magnitude_combined_loss's docstring).

    freq_weight is only used when loss_type is 'combined' or
    'log_combined' (ignored, but accepted, for 'l1'/'l2' so callers don't
    need to special-case the function signature per loss type).

    Returns loss, or (loss, component_a, component_b) if return_components:
      'l1'/'l2'      -> component_a = loss itself, component_b = 0.0
                        (there's only one term -- nothing else to log)
      'combined'     -> component_a = l1_time, component_b = l1_freq
      'log_combined' -> component_a = l1_time, component_b = l1_freq_log
                        (both match the two TensorBoard scalars main.py
                        already tracks for the production combined loss)
    """
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
        if return_components:
            return loss, loss.detach(), torch.tensor(0.0)
        return loss

    if loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
        if return_components:
            return loss, loss.detach(), torch.tensor(0.0)
        return loss

    if loss_type == 'combined':
        return _production_combined_loss(
            noise, predicted_noise, freq_weight=freq_weight,
            return_components=return_components,
        )

    if loss_type == 'log_combined':
        return _log_magnitude_combined_loss(
            noise, predicted_noise, freq_weight=freq_weight,
            return_components=return_components,
        )

    raise ValueError(f"Unknown loss_type '{loss_type}' -- expected one of {LOSS_TYPES}")
