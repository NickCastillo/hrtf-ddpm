# HRTF DDPM — Model B (HUTUBS Baseline Reimplementation)

Reimplementation of the conditional DDPM from arXiv:2501.02871 for HRTF
personalization, evaluated on HUTUBS (Phase 1 baseline validation).

## Architecture (brief)

A DDPM that predicts the added noise on 2-channel (L/R) HRIR waveforms
(256 samples), conditioned on: diffusion timestep, measurement
point / direction-of-arrival, and anthropometric head + ear features.

- **Backbone:** 1D U-Net — 5 encoder blocks → bottleneck → 5 decoder
  blocks, skip connections via concatenation.
- **Each block:** `Conv1d(k=3) → norm → ReLU → Conv1d(k=3) → norm`,
  with conditioning fused in before the second conv, then
  downsample (`Conv1d k=4 s=2`) or upsample (`ConvTranspose1d k=4 s=2`).
- **Self-attention:** 4-head attention in the deeper encoder blocks.
- **Conditioning:** timestep, measurement-point label, head measurements,
  and ear measurements are each projected and fused into every block.

## What matches the paper

| Aspect              | Paper                                  | This code                              |
| ------------------- | -------------------------------------- | -------------------------------------- |
| Timesteps           | 600                                    | 600                                    |
| Beta schedule       | linear, 1e-4 → 0.02                    | same                                   |
| U-Net depth         | 5 encoder + 5 decoder blocks           | same                                   |
| Channel-mult ratios | (4, 8, 16, 32, 64)                     | same ratios (see below for scale)      |
| Conv kernels        | k=3 pad=1 (blocks), k=4 s=2 (resample) | same                                   |
| Self-attention      | 4 heads                                | same                                   |
| Skip connections    | concatenation                          | same                                   |
| Conditioning fusion | concatenation                          | **now matches** (was addition — fixed) |
| LSD metric          | Eq. 9, K=44 bands, 0–15 kHz            | same                                   |

## What differs from the paper (and why)

| Aspect                   | Paper                                                                   | This code                                           | Reason                                                                                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Normalization            | BatchNorm                                                               | GroupNorm                                           | BatchNorm degrades at small spatial lengths (e.g. L=8 at the bottleneck)                                                                                                                               |
| Channel width            | literal (4,8,16,32,64)                                                  | ×`base_channels` (default 8 → 32,64,128,256,512)    | Literal paper sizes (~40K params) are non-functional; width is now a tunable parameter                                                                                                                 |
| Self-attention placement | after every downsampling block                                          | only on the two deepest blocks                      | ~4–6× speedup, negligible quality loss at shallow (uncompressed) resolutions                                                                                                                           |
| Loss                     | L2 (implied)                                                            | Combined L1 (time) + L1 (freq. magnitude)           | Discrepancy vs. paper's released code (L1); freq. term added as an experiment aligned with the LSD metric                                                                                                |
| LR schedule              | simple decay (paper text)                                               | StepLR + linear warmup                              | Monotonic decay pairs better with early-stopping checkpointing than cosine schedules                                                                                                                   |
| Evaluation protocol      | random 80/20 **sample-level** split (leaks test subjects into training) | **subject-level** k-fold CV, no leakage             | This is possibly the primary source of the ~7.6 dB (ours) vs. 5.1 dB (paper) LSD gap. k-fold was chosen to speed up training, the source code also includes a leaky protocol which has now been fixed. |
| Anthropometric features  | count/selection unclear                                                 | head_dim=13, ear_dim=24 (sigmoid-normalized, Eq. 4) | Pending clarification from paper's authors. At the moment all HUTUBS features are used.                                                                                                                |

## EMA + combined loss (new)

**EMA:** `--ema_decay` (default `0.999`), `--use_ema` (default `true`)

An exponential moving average of the U-Net weights is maintained
alongside normal training. Each optimizer step updates the EMA copy;
validation/early-stopping/checkpointing swap the EMA weights in for the
duration of the validation pass (then restore the raw training weights
before the next epoch), so `best_val_loss` reflects the EMA model's
performance. Checkpoints store both `model_state_dict` (raw) and
`ema_state_dict`. Inference loads `ema_state_dict` by default
(`--use_ema true`); set it to `false` to use raw weights instead, and
older checkpoints without an EMA copy fall back to raw weights with a
warning.

**Combined loss:** `--loss_freq_weight` (default `0.3`)

The training loss now combines the paper's time-domain L1 term with an
L1 term on FFT magnitude, restricted to the same K=44 frequency bands
(0–15 kHz) used by the LSD evaluation metric:

```
loss = (1 - loss_freq_weight) * L1(noise, pred) + loss_freq_weight * L1(|FFT(noise)|, |FFT(pred)|)
```

Both terms operate on the predicted diffusion noise (epsilon), not the
HRIR itself. `loss_freq_weight=0` recovers the original plain L1 loss.
Implemented as `combined_loss()` in `utils.py`, reusing the same
frequency-bin selection (`_get_freq_bins`) as `lsd()` so the loss is
aligned with what's actually being measured at evaluation time.

**Bugfix (post-first-run):** the FFT is computed with `norm='ortho'`.
An unnormalized FFT's magnitude scales with `sqrt(signal_length)`
relative to the time-domain amplitude (~16x for length-256 unit-variance
noise), so without this the frequency term dominated the sum almost
regardless of `loss_freq_weight`, silently turning a 70/30 blend into
something closer to 15/85 and degrading results. `Loss/train_l1_time`
and `Loss/train_l1_freq` are now logged separately to TensorBoard so
this kind of scale mismatch is visible going forward.

## Other notes

- **Splits caching:** `checkpoint_dir/splits.json` + `splits_meta.json`.
  The metadata file records the `k_folds` setting used to build the
  cached splits and forces a regeneration if it changes, so stale index
  caches can't silently be reused.
- **Precision:** FP16 autocast by default; falls back to FP32 for
  `base_channels=16` (large model) to avoid overflow.
- **Metrics:** LSD (L/R + avg, Eq. 9), ITD error (energy-onset
  detection), PBC (ERB gammatone filterbank), NMSE — saved per-subject
  and per-fold as `.mat`, plus Excel summaries.
- Crash-safe training: atomic `progress.json` writes, full
  optimizer/scheduler state in checkpoints.

## Files

| File         | Contents                                                                 |
| ------------ | ------------------------------------------------------------------------ |
| `model.py`   | `DiffusionModel` (noise schedule), `UNet` architecture                   |
| `dataset.py` | `HUTUBSDataset` — loading, normalization, k-fold splitting             |
| `main.py`    | CLI entrypoint — training & inference per fold                           |
| `utils.py`   | Metrics (LSD, ITD, PBC, NMSE) and plotting                               |
