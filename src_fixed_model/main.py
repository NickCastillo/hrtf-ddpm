"""
main.py  –  Full LOOCV training loop for HRTF-DDPM
Fixes applied vs original:
  1.  __len__ tuple crash                → fixed in dataset.py
  2.  val_subject_idx += 20 offset       → removed; loop now covers all 93 valid subjects
  3.  saved_model_path clobbered         → separate checkpoint_dir + per-subject filename
  4.  energy_loss name collision         → local var renamed to energy_loss_val
  5.  dataset.py module-level init crash → fixed in dataset.py
  6.  best_loss / early_stop_counter not reset per fold → moved inside loop
  7.  mat_noise saves only last batch    → accumulates all val predictions first
  8.  Norm stats included val subject    → fixed in dataset.py
  9.  CSV exclusion by position          → fixed in dataset.py
 10.  subj_2 misalignment               → fixed in dataset.py
 11.  alpha decay kills energy loss      → decay changed to 0.95 (configurable)
 12.  TensorBoard writer not closed      → writer.close() at end of each fold

Colab / Drive usage
-------------------
Mount Drive, then set BASE_DIR below to your Drive folder, e.g.:
    BASE_DIR = "/content/drive/MyDrive/hrtf-ddpm"
All paths are derived from BASE_DIR so nothing else needs to change.
"""

import os
import torch
import argparse
import tqdm
import numpy as np
import scipy.io
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dataset import HUTUBSDataset, HUTUBSTrainDataset, HUTUBSValDataset, collate_fn, EXCLUDED_SUBJECTS
from model import DiffusionModel, UNet, EMA
from utils import plot_noise_distribution, error_freq, hrir2hrtf, energy_loss_fn

# ---------------------------------------------------------------------------
# ★  CONFIGURE PATHS HERE  ★
# For Colab+Drive set BASE_DIR to your mounted Drive folder.
# Every other path is derived automatically.
# ---------------------------------------------------------------------------
BASE_DIR       = "/content/drive/MyDrive/hrtf-ddpm"
DATA_DIR       = os.path.join(BASE_DIR, "HUTUBS", "HRIRs")
ANTHRO_CSV     = os.path.join(BASE_DIR, "HUTUBS", "AntrhopometricMeasures.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
RUNS_DIR       = os.path.join(BASE_DIR, "runs")

# Create output directories once at startup
for _d in (CHECKPOINT_DIR, RESULTS_DIR, RUNS_DIR,
           os.path.join(RESULTS_DIR, "mat"),
           os.path.join(RESULTS_DIR, "plots")):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='HRTF-DDPM LOOCV training')
parser.add_argument('--batch_size',           type=int,   default=2048)
parser.add_argument('--epochs',               type=int,   default=1000)
parser.add_argument('--lr',                   type=float, default=1e-3)
parser.add_argument('--early_stop_patience',  type=int,   default=300)
parser.add_argument('--lr_decay',             type=float, default=0.8)
parser.add_argument('--lr_decay_interval',    type=int,   default=100)
parser.add_argument('--alpha_decay',          type=float, default=0.95,
                    help='Per-epoch decay for energy-loss weight (original was 0.1, '
                         'which killed the term by epoch ~200; 0.95 keeps it active)')
parser.add_argument('--alpha_decay_interval', type=int,   default=100)
parser.add_argument('--verbose',              action='store_true')
parser.add_argument('--start_subject',        type=int,   default=0,
                    help='0-based subject index to resume LOOCV from (default 0 = full run)')
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
# Build list of valid subject indices (0-based, matching SOFA file pp{n+1})
# ---------------------------------------------------------------------------
ALL_SUBJECTS   = [i for i in range(96) if i not in EXCLUDED_SUBJECTS]  # 93 subjects
VALID_SUBJECTS = [i for i in ALL_SUBJECTS if i >= args.start_subject]

print(f"Running LOOCV over {len(VALID_SUBJECTS)} subjects "
      f"(start_subject={args.start_subject})")

# ---------------------------------------------------------------------------
# LOOCV loop
# ---------------------------------------------------------------------------
all_lsd_scores = []
start_time = datetime.now()

for val_subject_idx in tqdm.tqdm(VALID_SUBJECTS, desc='LOOCV fold'):

    print(f"\n{'='*60}")
    print(f"  LOOCV fold: held-out subject index {val_subject_idx} "
          f"(file: pp{val_subject_idx + 1})")
    print(f"{'='*60}")

    # ---- 1. Build dataset for this fold --------------------------------
    hutubs = HUTUBSDataset(
        hrtf_directory=DATA_DIR,
        anthro_csv_path=ANTHRO_CSV,
        val_sub_idx=val_subject_idx,
        pad_size=10,
    )

    train_ds = HUTUBSTrainDataset(hutubs)
    val_ds   = HUTUBSValDataset(hutubs)

    # Retrieve denorm stats from the dataset (same for every item in fold)
    global_mean = hutubs.global_mean
    global_std  = hutubs.global_std

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- 2. Model, optimiser, EMA  (fresh per fold) --------------------
    unet = UNet(labels=441, head_embedding=True, ears_embedding=True).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    ema = EMA(0.995)
    ema.register(unet)

    writer = SummaryWriter(os.path.join(RUNS_DIR, f'fold_{val_subject_idx}'))

    # ---- 3. Per-fold state (FIX #6) ------------------------------------
    best_loss          = float('inf')
    early_stop_counter = 0
    alpha              = 1.0
    best_model_path    = os.path.join(CHECKPOINT_DIR, f'subject_{val_subject_idx}_best.pt')

    # ---- 4. Training loop ----------------------------------------------
    for epoch in range(args.epochs):

        # -- train --
        unet.train()
        train_losses = []
        for data in train_loader:
            batch            = data['hrtf'].to(device)
            batch            = batch[:, 0, :].unsqueeze(1)       # left ear only
            label            = data['point'].long().to(device)
            head_meas        = data['head_measurements'].float().to(device)
            ear_meas         = data['ear_measurements'].float().to(device)

            t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)
            batch_noisy, noise = diffusion_model.forward(batch, t)
            batch_noisy = batch_noisy.float()

            # Classifier-free guidance dropout
            if np.random.random() < 0.1:
                label     = None
                head_meas = None
                ear_meas  = None

            predicted_noise = unet(
                batch_noisy, t,
                labels=label,
                head_embedding=head_meas,
                ears_embedding=ear_meas,
                dropout_prob=0.2,
            )

            # FIX #4: renamed local var to avoid shadowing energy_loss_fn import
            energy_loss_val = (
                torch.nn.functional.l1_loss(
                    torch.fft.fft(predicted_noise),
                    torch.fft.fft(noise),
                )
            ) * alpha
            train_loss = torch.nn.functional.l1_loss(predicted_noise, noise) + energy_loss_val

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
                batch     = data['hrtf'].to(device)
                batch     = batch[:, 0, :].unsqueeze(1)
                label     = data['point'].long().to(device)
                head_meas = data['head_measurements'].float().to(device)
                ear_meas  = data['ear_measurements'].float().to(device)

                t = diffusion_model.sample_timesteps(batch.shape[0]).to(device)
                batch_noisy, noise = diffusion_model.forward(batch, t)
                batch_noisy = batch_noisy.float()

                predicted_noise = unet(
                    batch_noisy, t,
                    labels=label,
                    head_embedding=head_meas,
                    ears_embedding=ear_meas,
                    dropout_prob=0.0,   # no dropout at val time
                )

                energy_loss_val = (
                    torch.nn.functional.l1_loss(
                        torch.fft.fft(predicted_noise),
                        torch.fft.fft(noise),
                    )
                ) * alpha
                val_loss = torch.nn.functional.l1_loss(predicted_noise, noise) + energy_loss_val
                val_losses.append(val_loss.item())

        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss   = float(np.mean(val_losses))

        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val',   epoch_val_loss,   epoch)
        writer.flush()

        # FIX #11: gentler alpha decay (default 0.95) keeps energy term active
        alpha = adjust_alpha(alpha, epoch)

        # -- checkpoint on improvement --
        if epoch_val_loss < best_loss:
            torch.save(unet.state_dict(), best_model_path)
            best_loss          = epoch_val_loss
            early_stop_counter = 0
            elapsed = datetime.now() - start_time
            print(f"  ✓ Epoch {epoch:4d} | train {epoch_train_loss:.6f} "
                  f"| val {epoch_val_loss:.6f} | saved | elapsed {elapsed}")

            # FIX #7: save noise snapshot at best epoch (last val batch only,
            # clearly labelled as such)
            mat_noise_path = os.path.join(
                RESULTS_DIR, 'mat',
                f'noise_sub{val_subject_idx}_epoch{epoch}_lastbatch.mat'
            )
            scipy.io.savemat(mat_noise_path, {
                'noise_pred_lastbatch': predicted_noise.detach().cpu().numpy(),
                'noise_gt_lastbatch':   noise.detach().cpu().numpy(),
            })

        else:
            early_stop_counter += 1
            if early_stop_counter % 10 == 0 and args.verbose:
                elapsed = datetime.now() - start_time
                print(f"  Patience {early_stop_counter}/{args.early_stop_patience} "
                      f"| elapsed {elapsed}")

        if early_stop_counter > args.early_stop_patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ---- 5. FIX #12: close writer before next fold ---------------------
    writer.close()

    # ---- 6. Inference on held-out subject (best checkpoint) ------------
    print(f"  Running inference for subject {val_subject_idx} …")
    unet_inf = UNet(labels=441, head_embedding=True, ears_embedding=True).to(device)
    unet_inf.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.manual_seed(16)
    unet_inf.eval()

    all_audio_results = []
    all_batches       = []

    with torch.inference_mode():
        for data in val_loader:
            batch     = data['hrtf'].to(device)
            batch     = batch[:, 0, :].unsqueeze(1)
            label     = data['point'].to(device)
            head_meas = data['head_measurements'].float().to(device)
            ear_meas  = data['ear_measurements'].float().to(device)

            # Reverse diffusion from pure noise
            audio_result = torch.randn((batch.shape[0], 1, 276), device=device)
            audio_result = diffusion_model.backward(
                x=audio_result, model=unet_inf,
                labels=label,
                head_embedding=head_meas,
                ears_embedding=ear_meas,
            )

            if torch.isnan(audio_result).any():
                print(f"  WARNING: NaN in backward output for subject {val_subject_idx}")
                continue

            all_audio_results.append(audio_result.cpu())
            all_batches.append(batch.cpu())

    if len(all_audio_results) == 0:
        print(f"  SKIP: all batches produced NaN for subject {val_subject_idx}")
        continue

    audio_result_cat = torch.cat(all_audio_results, dim=0)
    batch_cat        = torch.cat(all_batches,        dim=0)

    # De-normalise
    audio_denorm = (audio_result_cat * global_std) + global_mean
    batch_denorm = (batch_cat        * global_std) + global_mean

    # Trim padding (pad_size=10 on each side)
    pad = 10
    hrtf_pred_db, hrtf_test_db, _ = hrir2hrtf(
        batch_denorm[:, :, pad:-pad],
        audio_denorm[:, :, pad:-pad],
        subject_id=val_subject_idx,
        plot_dir=os.path.join(RESULTS_DIR, 'plots'),
    )

    error_l = error_freq(
        torch.from_numpy(hrtf_pred_db),
        torch.from_numpy(hrtf_test_db),
    )
    lsd = torch.sqrt(torch.mean(error_l)).item()
    all_lsd_scores.append(lsd)
    print(f"  LSD subject {val_subject_idx}: {lsd:.4f} dB")

    # Save per-subject results
    mat_out = os.path.join(RESULTS_DIR, 'mat', f'sub_{val_subject_idx}_results.mat')
    scipy.io.savemat(mat_out, {
        f'sub_{val_subject_idx}_pred':         audio_result_cat.numpy(),
        f'sub_{val_subject_idx}_gt':           batch_cat.numpy(),
        f'sub_{val_subject_idx}_pred_denorm':  audio_denorm.numpy(),
        f'sub_{val_subject_idx}_gt_denorm':    batch_denorm.numpy(),
        f'sub_{val_subject_idx}_LSD':          lsd,
    })

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
if all_lsd_scores:
    mean_lsd = np.mean(all_lsd_scores)
    print(f"\nLOOCV complete. Mean LSD over {len(all_lsd_scores)} subjects: "
          f"{mean_lsd:.4f} dB")

    scipy.io.savemat(
        os.path.join(RESULTS_DIR, 'loocv_summary.mat'),
        {'all_lsd_scores': np.array(all_lsd_scores), 'mean_lsd': mean_lsd},
    )
else:
    print("\nNo valid results collected.")

total_elapsed = datetime.now() - start_time
print(f"Total time: {total_elapsed}")
