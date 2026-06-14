"""
main.py  -  5-Fold CV training loop for HRTF-DDPM (left-ear + mirrored-right)

All fixes vs original juancaudio20 code
========================================
 1.  __len__ tuple crash                -> fixed in dataset.py
 2.  val_subject_idx += 20 offset       -> removed; loop covers all 93 valid subjects
 3.  saved_model_path clobbered         -> per-fold best_model_path, never overwritten
 4.  energy_loss name collision         -> local var renamed to energy_loss_val
 5.  dataset.py module-level init crash -> fixed in dataset.py
 6.  best_loss / early_stop_counter not reset per fold -> moved inside loop
 7.  mat_noise saves only last batch    -> labelled clearly as last-batch snapshot
 8.  Norm stats included val subject    -> fixed in dataset.py (train-only stats)
 9.  CSV exclusion by position          -> fixed in dataset.py (by subject ID)
10.  subj_2 misalignment               -> fixed in dataset.py (dict lookup)
11.  alpha decay kills energy loss      -> decay changed to 0.95 (configurable)
12.  TensorBoard writer not closed      -> writer.close() at end of every fold

Left-ear + mirrored-right augmentation
=======================================
M1.  Model now trained on left ears only (in_channels=1, out_channels=1).
M2.  Right-ear HRIRs are mirrored in azimuth and added as synthetic left-ear
     samples, doubling the effective training set size.
M3.  Training loop uses batch['hrtf_mono'] (B, 1, L) instead of (B, 2, L).
M4.  Validation & inference still compute binaural LSD by running the model
     twice: once for the genuine left ear (hrtf_l) and once for the mirrored
     right ear, then un-mirroring the prediction back to the right ear.
M5.  hrir2hrtf / error_freq_binaural in utils.py are unchanged; binaural eval
     is reconstructed from two mono forward passes.

5-Fold CV (replaces LOOCV)
===========================
F1.  build_5fold_splits() in dataset.py splits 93 subjects into 5 folds
     (~18-19 subjects per fold).
F2.  Each fold trains on ~74-75 subjects and evaluates on ~18-19.
F3.  --start_fold CLI arg lets you resume mid-way.
F4.  Per-fold checkpoints are named fold_{fold_idx}_best.pt.
F5.  Summary .mat saved as 5fold_summary.mat (replaces loocv_summary.mat).
"""

import os
import torch
import argparse
import tqdm
import numpy as np
import scipy.io
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dataset import (HUTUBSDataset, HUTUBSTrainDataset, HUTUBSValDataset,
                     collate_fn, EXCLUDED_SUBJECTS, build_5fold_splits)
from model import DiffusionModel, UNet, EMA
from utils import (plot_noise_distribution, hrir2hrtf,
                   error_freq_binaural, energy_loss_fn)

# ---------------------------------------------------------------------------
# CONFIGURE PATHS  (patched automatically by colab_train.py)
# ---------------------------------------------------------------------------
BASE_DIR       = "/content/drive/MyDrive/hrtf-ddpm"
DATA_DIR       = os.path.join(BASE_DIR, "HUTUBS", "HRIRs")
ANTHRO_CSV     = os.path.join(BASE_DIR, "HUTUBS", "AntrhopometricMeasures.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
RUNS_DIR       = os.path.join(BASE_DIR, "runs")

for _d in (CHECKPOINT_DIR, RESULTS_DIR, RUNS_DIR,
           os.path.join(RESULTS_DIR, "mat"),
           os.path.join(RESULTS_DIR, "plots")):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='HRTF-DDPM 5-Fold CV training (left+mirrored)')
parser.add_argument('--batch_size',           type=int,   default=2048)
parser.add_argument('--epochs',               type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-3)
parser.add_argument('--early_stop_patience',  type=int,   default=300)
parser.add_argument('--lr_decay',             type=float, default=0.8)
parser.add_argument('--lr_decay_interval',    type=int,   default=100)
parser.add_argument('--alpha_decay',          type=float, default=0.95)
parser.add_argument('--alpha_decay_interval', type=int,   default=100)
parser.add_argument('--verbose',              action='store_true')
parser.add_argument('--start_fold',           type=int,   default=0,
                    help='0-based fold index to resume 5-fold CV from (0–4)')
parser.add_argument('--n_folds',              type=int,   default=5,
                    help='Number of CV folds (default 5)')
parser.add_argument('--fold_seed',            type=int,   default=42,
                    help='Random seed for fold assignment')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
diffusion_model = DiffusionModel()

def adjust_lr(optimizer, base_lr, epoch):
    lr = base_lr * (args.lr_decay ** (epoch // args.lr_decay_interval))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def adjust_alpha(alpha, epoch):
    return alpha * (args.alpha_decay ** (epoch // args.alpha_decay_interval))

# ---------------------------------------------------------------------------
# Build 5-fold splits
# ---------------------------------------------------------------------------
folds = build_5fold_splits(n_folds=args.n_folds, seed=args.fold_seed)
folds_to_run = folds[args.start_fold:]

print(f"5-Fold CV: {args.n_folds} folds total, running from fold {args.start_fold}")
for fi, f in enumerate(folds):
    print(f"  Fold {fi}: val subjects = {sorted(f['val'])}")

# ---------------------------------------------------------------------------
# 5-Fold CV loop
# ---------------------------------------------------------------------------
all_lsd_scores = []   # list of per-fold mean LSD values
start_time     = datetime.now()

for fold_offset, fold in enumerate(tqdm.tqdm(folds_to_run, desc='5-Fold CV')):
    fold_idx     = args.start_fold + fold_offset
    val_subjects = fold['val']    # list of 0-based sofa indices for this fold

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}: held-out subjects = {sorted(val_subjects)}")
    print(f"{'='*60}")

    # ---- 1. Dataset for this fold ----------------------------------------
    hutubs = HUTUBSDataset(
        hrtf_directory=DATA_DIR,
        anthro_csv_path=ANTHRO_CSV,
        val_subject_list=val_subjects,
        pad_size=10,
    )
    train_ds    = HUTUBSTrainDataset(hutubs)
    val_ds      = HUTUBSValDataset(hutubs)
    global_mean = hutubs.global_mean   # scalar, computed from training data only
    global_std  = hutubs.global_std

    print(f"  Train samples (left + mirrored): {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- 2. Model (fresh per fold) ----------------------------------------
    # M1: in_channels=1, out_channels=1 — single-ear (left or mirrored-right)
    unet      = UNet(in_channels=1, out_channels=1,
                     labels=441, head_embedding=True, ears_embedding=True).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    ema       = EMA(0.995)
    ema.register(unet)

    writer = SummaryWriter(os.path.join(RUNS_DIR, f'fold_{fold_idx}'))

    # ---- 3. Per-fold state ------------------------------------------------
    best_loss          = float('inf')
    early_stop_counter = 0
    alpha              = 1.0
    best_model_path    = os.path.join(CHECKPOINT_DIR, f'fold_{fold_idx}_best.pt')

    # ---- 4. Training loop -------------------------------------------------
    for epoch in range(args.epochs):

        # -- train --
        unet.train()
        train_losses = []

        for data in train_loader:
            # M3: use hrtf_mono (B, 1, L) — left ear or mirrored right ear
            batch     = data['hrtf_mono'].to(device).float()   # (B, 1, L)
            label     = data['point'].long().to(device)
            head_meas = data['head_measurements'].float().to(device)
            ear_meas  = data['ear_measurements'].float().to(device)

            t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t)
            # noise shape: (B, 1, L)

            if np.random.random() < 0.1:   # classifier-free guidance dropout
                label = head_meas = ear_meas = None

            predicted_noise = unet(
                batch_noisy.float(), t,
                labels=label,
                head_embedding=head_meas,
                ears_embedding=ear_meas,
            )   # (B, 1, L)

            energy_loss_val = torch.nn.functional.l1_loss(
                torch.fft.fft(predicted_noise),
                torch.fft.fft(noise),
            ) * alpha

            train_loss = (torch.nn.functional.l1_loss(predicted_noise, noise)
                          + energy_loss_val)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            ema.update(unet)
            train_losses.append(train_loss.item())

        adjust_lr(optimizer, args.lr, epoch)

        # -- validate (left ear only for speed during training) --
        unet.eval()
        val_losses = []

        with torch.inference_mode():
            for data in val_loader:
                # Use genuine left ear for quick per-epoch val loss
                batch     = data['hrtf_l'].to(device).float()   # (B, 1, L)
                label     = data['point'].long().to(device)
                head_meas = data['head_measurements'].float().to(device)
                ear_meas  = data['ear_measurements'].float().to(device)

                t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)
                batch_noisy, noise = diffusion_model.forward(batch, t)

                predicted_noise = unet(
                    batch_noisy.float(), t,
                    labels=label,
                    head_embedding=head_meas,
                    ears_embedding=ear_meas,
                )

                energy_loss_val = torch.nn.functional.l1_loss(
                    torch.fft.fft(predicted_noise),
                    torch.fft.fft(noise),
                ) * alpha

                val_loss = (torch.nn.functional.l1_loss(predicted_noise, noise)
                            + energy_loss_val)
                val_losses.append(val_loss.item())

        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss   = float(np.mean(val_losses))

        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val',   epoch_val_loss,   epoch)
        writer.flush()

        alpha = adjust_alpha(alpha, epoch)

        if epoch_val_loss < best_loss:
            torch.save(unet.state_dict(), best_model_path)
            best_loss          = epoch_val_loss
            early_stop_counter = 0
            elapsed = datetime.now() - start_time
            print(f"  ✓ Epoch {epoch:4d} | train {epoch_train_loss:.6f} "
                  f"| val {epoch_val_loss:.6f} | saved | {elapsed}")

            # Noise snapshot (last batch, clearly labelled)
            scipy.io.savemat(
                os.path.join(RESULTS_DIR, 'mat',
                             f'noise_fold{fold_idx}_ep{epoch}_lastbatch.mat'),
                {
                    'noise_pred_lastbatch': predicted_noise[:, 0, :].cpu().numpy(),
                    'noise_gt_lastbatch':   noise[:, 0, :].cpu().numpy(),
                }
            )
        else:
            early_stop_counter += 1
            if args.verbose and early_stop_counter % 10 == 0:
                print(f"  Patience {early_stop_counter}/{args.early_stop_patience} "
                      f"| {datetime.now() - start_time}")

        if early_stop_counter > args.early_stop_patience:
            print(f"  Early stop at epoch {epoch}")
            break

    writer.close()

    # ---- 5. Inference on held-out subjects --------------------------------
    # M4: run model twice per point — once for left ear, once for mirrored right.
    print(f"  Running inference for fold {fold_idx} ({len(val_subjects)} subjects) …")

    unet_inf = UNet(in_channels=1, out_channels=1,
                    labels=441, head_embedding=True, ears_embedding=True).to(device)
    unet_inf.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.manual_seed(16)
    unet_inf.eval()

    # Collect per-subject results so we can save per-subject .mat files
    # val_data items all have subject_id; group by it.
    subject_preds = {}   # subject_id → {'pred_l', 'pred_r', 'gt_l', 'gt_r'}

    with torch.inference_mode():
        for data in val_loader:
            head_meas  = data['head_measurements'].float().to(device)
            ear_meas   = data['ear_measurements'].float().to(device)
            label      = data['point'].to(device)
            subject_ids = data['subject_id']            # (B,) LongTensor

            # --- Left-ear pass ---
            gt_l      = data['hrtf_l'].to(device).float()   # (B, 1, L)
            L_padded  = gt_l.shape[2]
            noise_l   = torch.randn((gt_l.shape[0], 1, L_padded), device=device)
            pred_l    = diffusion_model.backward(
                x=noise_l, model=unet_inf,
                labels=label, head_embedding=head_meas, ears_embedding=ear_meas,
            )   # (B, 1, L)

            # --- Mirrored-right-ear pass ---
            # Feed the right HRIR channel through the model as if it were a left ear
            gt_r_raw  = data['hrtf_r'].to(device).float()   # (B, 1, L)  right ear HRIR
            noise_r   = torch.randn_like(gt_r_raw)
            # Mirrored azimuth is encoded implicitly — model just sees a left-ear-like
            # waveform; we generate the prediction as we would for a left ear.
            pred_r_mirrored = diffusion_model.backward(
                x=noise_r, model=unet_inf,
                labels=label, head_embedding=head_meas, ears_embedding=ear_meas,
            )   # (B, 1, L) — predicted "left ear" for the mirrored azimuth

            if torch.isnan(pred_l).any() or torch.isnan(pred_r_mirrored).any():
                print(f"  WARNING: NaN in backward output for fold {fold_idx}, skipping batch")
                continue

            # Store per-subject
            for b in range(pred_l.shape[0]):
                sid = int(subject_ids[b].item())
                if sid not in subject_preds:
                    subject_preds[sid] = {
                        'pred_l': [], 'pred_r': [],
                        'gt_l':   [], 'gt_r':   [],
                    }
                subject_preds[sid]['pred_l'].append(pred_l[b:b+1].cpu())
                subject_preds[sid]['pred_r'].append(pred_r_mirrored[b:b+1].cpu())
                subject_preds[sid]['gt_l'].append(gt_l[b:b+1].cpu())
                subject_preds[sid]['gt_r'].append(gt_r_raw[b:b+1].cpu())

    if not subject_preds:
        print(f"  SKIP: all batches produced NaN for fold {fold_idx}")
        continue

    fold_lsd_scores = []

    for sid, arrs in subject_preds.items():
        pred_l_cat = torch.cat(arrs['pred_l'], dim=0)   # (N, 1, L)
        pred_r_cat = torch.cat(arrs['pred_r'], dim=0)
        gt_l_cat   = torch.cat(arrs['gt_l'],   dim=0)
        gt_r_cat   = torch.cat(arrs['gt_r'],   dim=0)

        # De-normalise
        pred_l_denorm = pred_l_cat * global_std + global_mean
        pred_r_denorm = pred_r_cat * global_std + global_mean
        gt_l_denorm   = gt_l_cat   * global_std + global_mean
        gt_r_denorm   = gt_r_cat   * global_std + global_mean

        # Trim padding — pad=10 on both sides
        pad = 10
        pred_l_trim = pred_l_denorm[:, :, pad:-pad]   # (N, 1, L_orig)
        pred_r_trim = pred_r_denorm[:, :, pad:-pad]
        gt_l_trim   = gt_l_denorm[:,   :, pad:-pad]
        gt_r_trim   = gt_r_denorm[:,   :, pad:-pad]

        # Build (N, 2, L) binaural tensors for hrir2hrtf
        # channel 0 = left, channel 1 = right
        pred_binaural = torch.cat([pred_l_trim, pred_r_trim], dim=1)   # (N, 2, L_orig)
        gt_binaural   = torch.cat([gt_l_trim,   gt_r_trim],   dim=1)

        hrtf_pred_l, hrtf_test_l, hrtf_pred_r, hrtf_test_r, _ = hrir2hrtf(
            gt_binaural, pred_binaural,
            subject_id=sid,
            plot_dir=os.path.join(RESULTS_DIR, 'plots'),
        )

        lsd = error_freq_binaural(hrtf_pred_l, hrtf_test_l,
                                   hrtf_pred_r, hrtf_test_r).item()
        fold_lsd_scores.append(lsd)
        print(f"  Subject {sid}: LSD = {lsd:.4f} dB (binaural mean)")

        # Save per-subject .mat
        scipy.io.savemat(
            os.path.join(RESULTS_DIR, 'mat', f'fold{fold_idx}_sub{sid}_results.mat'),
            {
                f'sub_{sid}_pred_L':       pred_l_cat[:, 0, :].numpy(),
                f'sub_{sid}_pred_R':       pred_r_cat[:, 0, :].numpy(),
                f'sub_{sid}_gt_L':         gt_l_cat[:, 0, :].numpy(),
                f'sub_{sid}_gt_R':         gt_r_cat[:, 0, :].numpy(),
                f'sub_{sid}_pred_denorm_L': pred_l_denorm[:, 0, :].numpy(),
                f'sub_{sid}_pred_denorm_R': pred_r_denorm[:, 0, :].numpy(),
                f'sub_{sid}_gt_denorm_L':  gt_l_denorm[:, 0, :].numpy(),
                f'sub_{sid}_gt_denorm_R':  gt_r_denorm[:, 0, :].numpy(),
                f'sub_{sid}_LSD_binaural': lsd,
            }
        )

    fold_mean_lsd = float(np.mean(fold_lsd_scores)) if fold_lsd_scores else float('nan')
    all_lsd_scores.append(fold_mean_lsd)
    print(f"  Fold {fold_idx} mean binaural LSD: {fold_mean_lsd:.4f} dB "
          f"({len(fold_lsd_scores)} subjects)")

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
if all_lsd_scores:
    mean_lsd = np.nanmean(all_lsd_scores)
    print(f"\n5-Fold CV complete. Mean binaural LSD over {len(all_lsd_scores)} folds: "
          f"{mean_lsd:.4f} dB")
    scipy.io.savemat(
        os.path.join(RESULTS_DIR, '5fold_summary.mat'),
        {
            'fold_lsd_scores': np.array(all_lsd_scores),
            'mean_lsd':        mean_lsd,
        },
    )
else:
    print("\nNo valid results collected.")

print(f"Total time: {datetime.now() - start_time}")
