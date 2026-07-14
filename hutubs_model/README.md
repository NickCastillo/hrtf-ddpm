# HRTF DDPM — HUTUBS_model (HUTUBS Baseline Reimplementation)

Reimplementation of the conditional DDPM from arXiv:2501.02871 for HRTF
personalization, evaluated on HUTUBS (Phase 0/1 baseline validation).
Referred to as **HUTUBS_model** to distinguish it from the four SONICOM
ablation conditions (unconditioned / anthro-only / image-only /
anthro+image) it is later compared against.

## Architecture (brief)

A DDPM that predicts the added noise on 2-channel (L/R) HRIR waveforms
(256 samples), conditioned on: diffusion timestep, measurement
point / direction-of-arrival, and anthropometric **ear** features.

- **Backbone:** 1D U-Net — 4 encoder blocks → bottleneck → 4 decoder
  blocks, skip connections via concatenation. (5 channel levels —
  (4,8,16,32,64)×base_channels — give 4 transitions/blocks, not 5; see
  `model.py`'s `UNet` docstring for the full level-by-level spatial
  lengths and why an earlier version of this doc miscounted it as 5.)
- **Each block:** `Conv1d(k=3) → norm → ReLU → Conv1d(k=3) → norm`,
  with conditioning fused in before the second conv, then
  downsample (`Conv1d k=4 s=2`) or upsample (`ConvTranspose1d k=4 s=2`).
- **Self-attention:** 4-head attention, by default on the two deepest
  encoder blocks (L=64 and L=32) plus the bottleneck (L=16). See
  "Self-attention placement ablation" below for the `--full_attention`
  switch that puts attention on all 4 encoder blocks instead.
- **Conditioning:** timestep, measurement-point label, and ear
  measurements are each projected and fused into every block. Head/torso
  measurements are not used (`head_dim=0`) — see "Feature mismatch fix"
  below.

## What matches the paper

| Aspect              | Paper                                  | This code                              |
| ------------------- | -------------------------------------- | -------------------------------------- |
| Timesteps           | 600                                    | 600                                    |
| Beta schedule       | linear, 1e-4 → 0.02                    | same                                   |
| Channel-mult ratios | (4, 8, 16, 32, 64)                     | same ratios (see below for scale)      |
| Conv kernels        | k=3 pad=1 (blocks), k=4 s=2 (resample) | same                                   |
| Self-attention      | 4 heads                                | same                                   |
| Skip connections    | concatenation                          | same                                   |
| Conditioning fusion | concatenation                          | **now matches** (was addition — fixed) |
| LSD metric          | Eq. 9, K=44 bands, 0–15 kHz            | same                                   |

## What differs from the paper (and why)

| Aspect                   | Paper                                                                   | This code                                                                      | Reason                                                                                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| U-Net depth              | "5 encoder blocks" (paper text)                                         | 4 encoder blocks (4 transitions across 5 channel levels)                       | See "Blocks vs. channel levels" below — a channel-mult tuple with 5 entries produces 4 transitions, not 5; this is a naming/fencepost mismatch, not a missing layer.                                   |
| Normalization            | BatchNorm                                                               | GroupNorm                                                                      | BatchNorm degrades at small spatial lengths (bottleneck is L=16, not L=8 — see below)                                                                                                                  |
| Channel width            | literal (4,8,16,32,64)                                                  | ×`base_channels` (default 8 → 32,64,128,256,512)                               | Literal paper sizes (~40K params) are non-functional; width is now a tunable parameter                                                                                                                 |
| Self-attention placement | after every downsampling block                                          | by default, only on the two deepest blocks (`--full_attention` restores all 4) | ~4–6× speedup, assumed negligible quality loss at shallow (uncompressed) resolutions — **not yet empirically ablated on HUTUBS_model**; see "Self-attention placement ablation" below.                 |
| Loss                     | L2 (implied)                                                            | Combined L1 (time) + L1 (freq. magnitude)                                      | Discrepancy vs. paper's released code (L1); freq. term added as an experiment aligned with the LSD metric                                                                                              |
| LR schedule              | simple decay (paper text)                                               | StepLR + linear warmup                                                         | Monotonic decay pairs better with early-stopping checkpointing than cosine schedules                                                                                                                   |
| Evaluation protocol      | random 80/20 **sample-level** split (leaks test subjects into training) | **subject-level** k-fold CV, no leakage                                        | This is possibly the primary source of the ~7.6 dB (ours) vs. 5.1 dB (paper) LSD gap. k-fold was chosen to speed up training, the source code also includes a leaky protocol which has now been fixed. |
| Anthropometric features  | count/selection unclear                                                 | ear_dim=24 only, head_dim=0 (sigmoid-normalized, Eq. 4)                        | Fixed in Phase 0 (see below): head/torso columns are never loaded, so conditioning matches SONICOM's ear-only anthropometric CSV (d1–d10, θ1, θ2) and later ablation conditions.                       |

## Blocks vs. channel levels

`CHANNEL_MULTS = (4, 8, 16, 32, 64)` is 5 channel _levels_. A `Block` is
a _transition_ between two consecutive levels (`seq[i] -> seq[i+1]`),
so 5 levels produce 4 transitions/blocks (`range(n - 1)` over `n=5` in
`model.py`), the same fencepost relationship as "5 posts, 4 fence
panels." Earlier docs/comments in this codebase said "5 encoder
blocks," which doesn't match what the code does.

With `audio_channels=2`, `HRIR_LEN=256`, and each block halving the
sequence length on downsample (`k=4, s=2, p=1`), the actual
level-by-level spatial lengths are:

| Stage                               | Input L | Output L | Attention (default) |
| ----------------------------------- | ------- | -------- | ------------------- |
| stem (channel proj, no resample)    | —       | 256      | —                   |
| encoder block 0 (seq0→seq1)         | 256     | 128      | off                 |
| encoder block 1 (seq1→seq2)         | 128     | 64       | off                 |
| encoder block 2 (seq2→seq3)         | 64      | 32       | **on** (at L=64)    |
| encoder block 3 (seq3→seq4)         | 32      | 16       | **on** (at L=32)    |
| bottleneck (seq4→seq4, no resample) | 16      | 16       | **on** (at L=16)    |

Attention is applied at a block's _input_ resolution, before that
block's own downsample (see `Block.forward`) — so "attention on the two
deepest blocks" means L=64 and L=32, and the bottleneck (always
attended) runs at L=16. An earlier version of this docstring claimed
the bottleneck was at L=8; that was wrong given the 4-block encoder
above wasn't accounted for correctly the first time. This also means
attention's O(L²·C) cost is already fairly small everywhere it's
applied (4096, 1024, and 256 position pairs respectively) — the
deepest blocks' conv layers, where channel count grows to 256→512,
dominate compute more than attention does at this point.

## Self-attention placement ablation

`--full_attention true` sets `attn_full_encoder=True` on `UNet`,
enabling attention on all 4 encoder blocks (matching the paper's "after
every downsampling block") instead of just the two deepest by default.
This exists to test, empirically, whether the "~4-6x speedup, negligible
quality loss" rationale for the default placement actually holds on
HUTUBS_model — that claim was a design assumption carried over into
this codebase, not something previously measured here. Use
`--model_name HUTUBS_model_full_attn` (or similar) alongside it so the
run's checkpoints/results/TensorBoard logs and exported per-subject
metrics land in a separate, clearly-tagged location instead of
overwriting the baseline run — `metrics_per_subject.xlsx` and the
per-fold `.mat` summaries both carry a `model` column and a
`full_attention` flag for exactly this comparison.

## Feature mismatch fix (Phase 0)

HUTUBS_model previously loaded and conditioned on all anthropometric
columns, including 13 head/torso measurements alongside the 24 ear
measurements. SONICOM's anthropometric CSV only provides the 12
ear-specific parameters (d1–d10, θ1, θ2) — there are no head/torso
columns — so a model conditioned on head/torso features can't be
directly compared against, or reused as a pretrained checkpoint for,
the SONICOM ablation conditions.

Fix: `dataset.py` no longer reads the head/torso CSV columns at all
(only ear columns are loaded, normalized, and exposed), and `model.py` /
`main.py` default to `head_dim=0`, which fully disables the head
conditioning branch. Subjects 18, 79, and 92 continue to be excluded
(incomplete/NaN anthropometric rows) — `dataset.py`'s
`EXCLUDED_SUBJECTS` set and the pre-normalization row-drop were already
correct and are unchanged by this fix.

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
  and per-fold as `.mat`, plus Excel summaries. Every per-subject export
  (`.mat` and `metrics_per_subject.xlsx`) carries an explicit
  `subject_id` and a `model` tag (from `--model_name`, default
  `"HUTUBS_model"`) rather than relying on iteration order, so results
  join cleanly on `subject_id` against other runs/conditions for a
  paired Wilcoxon signed-rank test on per-subject LSD (or ITD/PBC/NMSE).
- **Output namespacing:** `--checkpoint_dir` / `--results_dir` /
  `--runs_dir` default to `./checkpoints|results|runs/<model_name>`, so
  different runs (baseline, attention ablation, future SONICOM
  conditions) don't overwrite each other's checkpoints or results as
  long as each is given a distinct `--model_name`.
- Crash-safe training: atomic `progress.json` writes, full
  optimizer/scheduler state in checkpoints.

## Files

| File         | Contents                                                   |
| ------------ | ---------------------------------------------------------- |
| `model.py`   | `DiffusionModel` (noise schedule), `UNet` architecture     |
| `dataset.py` | `HUTUBSDataset` — loading, normalization, k-fold splitting |
| `main.py`    | CLI entrypoint — training & inference per fold             |
| `utils.py`   | Metrics (LSD, ITD, PBC, NMSE) and plotting                 |
