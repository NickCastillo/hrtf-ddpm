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
| Loss                     | L2 (implied)                                                            | L1                                                  | Discrepancy vs. paper's released code; kept as current default, still open                                                                                                                             |
| LR schedule              | simple decay (paper text)                                               | StepLR + linear warmup                              | Monotonic decay pairs better with early-stopping checkpointing than cosine schedules                                                                                                                   |
| Evaluation protocol      | random 80/20 **sample-level** split (leaks test subjects into training) | **subject-level** k-fold CV, no leakage             | This is possibly the primary source of the ~7.6 dB (ours) vs. 5.1 dB (paper) LSD gap. k-fold was chosen to speed up training, the source code also includes a leaky protocol which has now been fixed. |
| Anthropometric features  | count/selection unclear                                                 | head_dim=13, ear_dim=24 (sigmoid-normalized, Eq. 4) | Pending clarification from paper's authors. At the moment all HUTUBS features are used.                                                                                                                |

## Data augmentation — L/R mirroring (new)

**Flag:** `--data_augmentation true/false` (default `false`)

Exploits approximate left/right symmetry of the head to synthesize an
extra sample from every real one:

1. Swap the L/R HRIR channels.
2. Re-label the measurement point as its **mirrored** source position
   (azimuth → `(360 − azimuth) % 360`; elevation/radius unchanged).
3. Swap the L/R halves of the ear measurements (head measurements are
   left unchanged — assumed non-lateralized).

This roughly doubles the number of training samples per subject.

**Important:** mirrored samples are only ever placed in the **training**
split — `get_kfold_splits` guarantees validation and test always contain
real, physically measured samples only.

**Assumptions to verify before trusting results:**

- SOFA `SourcePosition` columns are `(azimuth, elevation, radius)` under
  the mirroring convention above (a warning is printed if the grid
  doesn't check out as symmetric).
- The anthropometric CSV's ear-measurement columns are laid out as
  `[left-ear features][right-ear features]` in two equal halves.

## Other notes

- **Splits caching:** `checkpoint_dir/splits.json` + `splits_meta.json`.
  The metadata file records the `data_augmentation` / `k_folds` setting
  used to build the cached splits and forces a regeneration if either
  changes, so stale index caches can't silently be reused.
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
| `dataset.py` | `HUTUBSDataset` — loading, normalization, k-fold splitting, augmentation |
| `main.py`    | CLI entrypoint — training & inference per fold                           |
| `utils.py`   | Metrics (LSD, ITD, PBC, NMSE) and plotting                               |
