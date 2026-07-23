"""
Configurable loss function for the L1 / L2 / combined loss ablation.

Kept as its own file, separate from main_model/utils.py, so the production
training script (main_model/main.py) is never touched by this experiment.
loss_exp/main.py imports compute_loss() from here instead of calling
utils.combined_loss() directly.

For loss_type='combined', this calls main_model/utils.py's own
combined_loss() rather than re-implementing the frequency-band math --
that guarantees byte-for-byte identical behaviour to the production loss
(including the norm='ortho' FFT fix), with zero risk of the two
implementations silently drifting apart over time.
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

LOSS_TYPES = ('l1', 'l2', 'combined')


def compute_loss(noise, predicted_noise, loss_type, freq_weight=0.3, return_components=False):
    """
    noise, predicted_noise: (B, 2, L) ground-truth / predicted diffusion
    noise (epsilon) -- same convention as main_model/utils.combined_loss.

    loss_type:
      'l1'       -- plain time-domain L1 (mean absolute error).
      'l2'       -- plain time-domain L2 (mean squared error).
      'combined' -- main_model's production loss: (1 - freq_weight) *
                    L1_time + freq_weight * L1_freq_mag (FFT norm='ortho',
                    same K=44 frequency bands as the LSD metric).

    freq_weight is only used when loss_type == 'combined' (ignored, but
    accepted, for 'l1'/'l2' so callers don't need to special-case the
    function signature per loss type).

    Returns loss, or (loss, component_a, component_b) if return_components:
      'l1'/'l2'  -> component_a = loss itself, component_b = 0.0
                    (there's only one term -- nothing else to log)
      'combined' -> component_a = l1_time, component_b = l1_freq
                    (matches the two TensorBoard scalars main.py already
                    tracks for the production combined loss)
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

    raise ValueError(f"Unknown loss_type '{loss_type}' -- expected one of {LOSS_TYPES}")
