"""
main.py  -  Full LOOCV training loop for HRTF-DDPM (binaural)

All fixes vs original juancaudio20 code
========================================
 1.  __len__ tuple crash                -> fixed in dataset.py
 2.  val_subject_idx += 20 offset       -> removed; loop covers all 93 valid subjects
 3.  saved_model_path clobbered         -> per-subject best_model_path, never overwritten
 4.  energy_loss name collision         -> local var renamed to energy_loss_val
 5.  dataset.py module-level init crash -> fixed in dataset.py
 6.  best_loss / early_stop_counter not reset per fold -> moved inside loop
 7.  mat_noise saves only last batch    -> labelled clearly as last-batch snapshot
 8.  Norm stats included val subject    -> fixed in dataset.py (train-only stats)
 9.  CSV exclusion by position          -> fixed in dataset.py (by subject ID)
10.  subj_2 misalignment               -> fixed in dataset.py (dict lookup)
11.  alpha decay kills energy loss      -> decay changed to 0.95 (configurable)
12.  TensorBoard writer not closed      -> writer.close() at end of every fold

Binaural fixes (both ears)
===========================
B1.  Training used batch[:, 0, :] only -> both ears now fed as (B, 2, L) tensor
B2.  Validation used left ear only     -> val loop now uses full (B, 2, L) tensor
B3.  Inference generated (B, 1, 276)   -> generates (B, 2, 276) for both ears
B4.  hrir2hrtf called with 1-channel   -> now receives (N, 2, L) binaural tensor
B5.  LSD computed on left ear only     -> error_freq_binaural averages both ears
B6.  .mat results saved left-only      -> pred/gt saved with _l and _r arrays
B7.  UNet input/output channels        -> in_channels and out_channels set to 2
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
                     collate_fn, EXCLUDED_SUBJECTS)
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
parser = argparse.ArgumentParser(description='HRTF-DDPM LOOCV training (binaural)')
parser.add_argument('--batch_size',           type=int,   default=2048)
parser.add_argument('--epochs',               type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-3)
parser.add_argument('--early_stop_patience',  type=int,   default=300)
parser.add_argument('--lr_decay',             type=float, default=0.8)
parser.add_argument('--lr_decay_interval',    type=int,   default=100)
parser.add_argument('--alpha_decay',          type=float, default=0.95)
parser.add_argument('--alpha_decay_interval', type=int,   default=100)
parser.add_argument('--verbose',              action='store_true')
parser.add_argument('--start_subject',        type=int,   default=0,
                    help='0-based subject index to resume LOOCV from')
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
# Subject list
# ---------------------------------------------------------------------------
ALL_SUBJECTS   = [i for i in range(96) if i not in EXCLUDED_SUBJECTS]
VALID_SUBJECTS = [i for i in ALL_SUBJECTS if i >= args.start_subject]
print(f"Running LOOCV over {len(VALID_SUBJECTS)} subjects "
      f"(start_subject={args.start_subject})")

# ---------------------------------------------------------------------------
# LOOCV loop
# ---------------------------------------------------------------------------
all_lsd_scores = []
start_time     = datetime.now()

for val_subject_idx in tqdm.tqdm(VALID_SUBJECTS, desc='LOOCV fold'):

    print(f"\n{'='*60}")
    print(f"  Fold: held-out subject {val_subject_idx} (pp{val_subject_idx + 1})")
    print(f"{'='*60}")

    # ---- 1. Dataset for this fold ----------------------------------------
    hutubs = HUTUBSDataset(
        hrtf_directory=DATA_DIR,
        anthro_csv_path=ANTHRO_CSV,
        val_sub_idx=val_subject_idx,
        pad_size=10,
    )
    train_ds    = HUTUBSTrainDataset(hutubs)
    val_ds      = HUTUBSValDataset(hutubs)
    global_mean = hutubs.global_mean   # scalar, computed from training data only
    global_std  = hutubs.global_std

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- 2. Model (fresh per fold) ----------------------------------------
    # B7: in_channels=2, out_channels=2 so both ears flow through the UNet
    unet      = UNet(in_channels=2, out_channels=2,
                     labels=441, head_embedding=True, ears_embedding=True).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    ema       = EMA(0.995)
    ema.register(unet)

    writer = SummaryWriter(os.path.join(RUNS_DIR, f'fold_{val_subject_idx}'))

    # ---- 3. Per-fold state (FIX 6) ----------------------------------------
    best_loss          = float('inf')
    early_stop_counter = 0
    alpha              = 1.0
    best_model_path    = os.path.join(CHECKPOINT_DIR,
                                      f'subject_{val_subject_idx}_best.pt')

    # ---- 4. Training loop -------------------------------------------------
    for epoch in range(args.epochs):

        # -- train --
        unet.train()
        train_losses = []

        for data in train_loader:
            # B1: use full binaural tensor (B, 2, L) instead of left ear only
            batch     = data['hrtf'].to(device).float()   # (B, 2, L)
            label     = data['point'].long().to(device)
            head_meas = data['head_measurements'].float().to(device)
            ear_meas  = data['ear_measurements'].float().to(device)

            t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t)
            # noise shape: (B, 2, L)  — matches both-channel batch

            if np.random.random() < 0.1:   # classifier-free guidance dropout
                label = head_meas = ear_meas = None

            predicted_noise = unet(
                batch_noisy.float(), t,
                labels=label,
                head_embedding=head_meas,
                ears_embedding=ear_meas,
                dropout_prob=0.2,
            )   # (B, 2, L)

            # FIX 4: energy_loss_val avoids shadowing the energy_loss_fn import
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

        # -- validate --
        unet.eval()
        val_losses = []

        with torch.inference_mode():
            for data in val_loader:
                # B2: full binaural tensor in validation too
                batch     = data['hrtf'].to(device).float()   # (B, 2, L)
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
                    dropout_prob=0.0,
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

        # FIX 11: gentler alpha decay keeps energy term active
        alpha = adjust_alpha(alpha, epoch)

        if epoch_val_loss < best_loss:
            torch.save(unet.state_dict(), best_model_path)
            best_loss          = epoch_val_loss
            early_stop_counter = 0
            elapsed = datetime.now() - start_time
            print(f"  ✓ Epoch {epoch:4d} | train {epoch_train_loss:.6f} "
                  f"| val {epoch_val_loss:.6f} | saved | {elapsed}")

            # FIX 7: noise snapshot clearly labelled as last-batch only
            scipy.io.savemat(
                os.path.join(RESULTS_DIR, 'mat',
                             f'noise_sub{val_subject_idx}_ep{epoch}_lastbatch.mat'),
                {
                    'noise_pred_L_lastbatch': predicted_noise[:, 0, :].cpu().numpy(),
                    'noise_pred_R_lastbatch': predicted_noise[:, 1, :].cpu().numpy(),
                    'noise_gt_L_lastbatch':   noise[:, 0, :].cpu().numpy(),
                    'noise_gt_R_lastbatch':   noise[:, 1, :].cpu().numpy(),
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

    # FIX 12: close TensorBoard writer before next fold
    writer.close()

    # ---- 5. Inference on held-out subject ---------------------------------
    print(f"  Running inference for subject {val_subject_idx} …")

    # B7: load with same 2-channel architecture
    unet_inf = UNet(in_channels=2, out_channels=2,
                    labels=441, head_embedding=True, ears_embedding=True).to(device)
    unet_inf.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.manual_seed(16)
    unet_inf.eval()

    all_preds   = []
    all_gt      = []

    with torch.inference_mode():
        for data in val_loader:
            batch     = data['hrtf'].to(device).float()        # (B, 2, L)
            label     = data['point'].to(device)
            head_meas = data['head_measurements'].float().to(device)
            ear_meas  = data['ear_measurements'].float().to(device)

            L_padded = batch.shape[2]

            # B3: generate (B, 2, L) — both ears simultaneously
            audio_result = torch.randn((batch.shape[0], 2, L_padded), device=device)
            audio_result = diffusion_model.backward(
                x=audio_result, model=unet_inf,
                labels=label,
                head_embedding=head_meas,
                ears_embedding=ear_meas,
            )   # (B, 2, L)

            if torch.isnan(audio_result).any():
                print(f"  WARNING: NaN in backward output, subject {val_subject_idx}")
                continue

            all_preds.append(audio_result.cpu())
            all_gt.append(batch.cpu())

    if not all_preds:
        print(f"  SKIP: all batches produced NaN for subject {val_subject_idx}")
        continue

    pred_cat = torch.cat(all_preds, dim=0)   # (N, 2, L)
    gt_cat   = torch.cat(all_gt,    dim=0)   # (N, 2, L)

    # De-normalise
    pred_denorm = (pred_cat * global_std) + global_mean
    gt_denorm   = (gt_cat   * global_std) + global_mean

    # Trim padding
    pad = 10
    pred_trim = pred_denorm[:, :, pad:-pad]   # (N, 2, L_orig)
    gt_trim   = gt_denorm[:,   :, pad:-pad]

    # B4: pass full (N, 2, L) binaural tensors to hrir2hrtf
    hrtf_pred_l, hrtf_test_l, hrtf_pred_r, hrtf_test_r, _ = hrir2hrtf(
        gt_trim, pred_trim,
        subject_id=val_subject_idx,
        plot_dir=os.path.join(RESULTS_DIR, 'plots'),
    )

    # B5: LSD averaged across both ears
    lsd = error_freq_binaural(hrtf_pred_l, hrtf_test_l,
                               hrtf_pred_r, hrtf_test_r).item()
    all_lsd_scores.append(lsd)
    print(f"  LSD subject {val_subject_idx}: {lsd:.4f} dB (binaural mean)")

    # B6: save both ears separately in .mat
    scipy.io.savemat(
        os.path.join(RESULTS_DIR, 'mat', f'sub_{val_subject_idx}_results.mat'),
        {
            f'sub_{val_subject_idx}_pred_L':       pred_cat[:, 0, :].numpy(),
            f'sub_{val_subject_idx}_pred_R':       pred_cat[:, 1, :].numpy(),
            f'sub_{val_subject_idx}_gt_L':         gt_cat[:, 0, :].numpy(),
            f'sub_{val_subject_idx}_gt_R':         gt_cat[:, 1, :].numpy(),
            f'sub_{val_subject_idx}_pred_denorm_L': pred_denorm[:, 0, :].numpy(),
            f'sub_{val_subject_idx}_pred_denorm_R': pred_denorm[:, 1, :].numpy(),
            f'sub_{val_subject_idx}_gt_denorm_L':  gt_denorm[:, 0, :].numpy(),
            f'sub_{val_subject_idx}_gt_denorm_R':  gt_denorm[:, 1, :].numpy(),
            f'sub_{val_subject_idx}_LSD_binaural': lsd,
        }
    )

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
if all_lsd_scores:
    mean_lsd = np.mean(all_lsd_scores)
    print(f"\nLOOCV complete. Mean binaural LSD over {len(all_lsd_scores)} subjects: "
          f"{mean_lsd:.4f} dB")
    scipy.io.savemat(
        os.path.join(RESULTS_DIR, 'loocv_summary.mat'),
        {'all_lsd_scores': np.array(all_lsd_scores), 'mean_lsd': mean_lsd},
    )
else:
    print("\nNo valid results collected.")

print(f"Total time: {datetime.now() - start_time}")
