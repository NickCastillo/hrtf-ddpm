"""
main.py — Training and inference for HRTF personalisation DDPM.

Training (k-fold cross-validation):
    python main.py --training --folds 5 --epochs 1000 --batch_size 64 \
                   --results_dir results/

Inference (after training all folds):
    python main.py --results_dir results/

K-fold protocol
---------------
93 valid subjects are partitioned into K folds of roughly equal size.
For each fold k:
  - Test  = subjects in fold k
  - Val   = subjects in fold (k+1) % K
  - Train = all remaining subjects (K-2 folds), with ear-mirroring augmentation

Optimizer (paper Section IV-A)
-------------------------------
Adam, lr=0.001, StepLR decay of 20% every 100 epochs.
1000 epochs, early stopping with patience=200.

LR default is 1e-3 (paper value). Previous version used 5e-5 (too low).

Anthropometric features
-----------------------
All 37 available HUTUBS features are used:
  13 head/torso (x1-x9, x12, x14, x16, x17) + 24 ear (12L + 12R pinna).
The paper references CIPIC N=27 but HUTUBS provides a different set;
see dataset.py for the full explanation.

Architecture (model.py)
------------------------
Matches arxiv 2501.02871 Section III-C:
  sequence_channels=(4,8,16,32,64), self-attention after each encoder block,
  second conv with no activation in each Block, binaural output (audio_channels=2).
"""

import os
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import tqdm
import scipy.io

from dataset import HUTUBSDataset, collate_fn, EXCLUDED_SUBJECT_IDS, N_SOFA_FILES
from model import DiffusionModel, UNet
from utils import plot_noise_distribution, lsd_paper, lsd_corrected


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='HRTF DDPM — k-fold training / inference')
parser.add_argument('--training',            action='store_true')
parser.add_argument('--folds',               type=int,   default=5,
                    help='Number of cross-validation folds (default: 5)')
parser.add_argument('--epochs',              type=int,   default=1000)
parser.add_argument('--batch_size',          type=int,   default=64)
parser.add_argument('--lr',                  type=float, default=1e-3,
                    help='Initial learning rate (paper: 0.001)')
parser.add_argument('--early_stop_patience', type=int,   default=200)
parser.add_argument('--results_dir',         type=str,   default='results')
parser.add_argument('--hrtf_directory',      type=str,
                    default='/content/drive/MyDrive/hrtf-ddpm/HUTUBS/HRIRs')
parser.add_argument('--anthro_csv_path',     type=str,
                    default='/content/drive/MyDrive/hrtf-ddpm/HUTUBS/AntrhopometricMeasures.csv')
parser.add_argument('--p_uncond',            type=float, default=0.0,
                    help='CFG dropout probability. Paper does not use CFG (default=0.0).')
parser.add_argument('--timesteps',           type=int,   default=600,
                    help='Diffusion steps (paper: 600)')
parser.add_argument('--start_fold',          type=int,   default=0,
                    help='Resume training from this fold index (default: 0). '
                         'Folds below this value are skipped entirely.')
parser.add_argument('--inference_batch_size', type=int, default=440,
                    help='Number of HRTF positions denoised in parallel during inference. '
                         'Use 440 to process one full HUTUBS subject at once; lower this if you run out of GPU memory.')
parser.add_argument('--verbose',             action='store_true')
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)

if args.verbose:
    print(f'folds={args.folds}, start_fold={args.start_fold}, timesteps={args.timesteps}, lr={args.lr}, '
          f'batch_size={args.batch_size}, inference_batch_size={args.inference_batch_size}, '
          f'early_stop={args.early_stop_patience}, p_uncond={args.p_uncond}')


# ---------------------------------------------------------------------------
# Subject list and fold assignment
# ---------------------------------------------------------------------------
all_subject_ids = sorted(
    sid for sid in range(1, N_SOFA_FILES + 1)
    if sid not in EXCLUDED_SUBJECT_IDS
)
n_subjects = len(all_subject_ids)   # 93

K = args.folds

fold_assignments = [[] for _ in range(K)]
for i, sid in enumerate(all_subject_ids):
    fold_assignments[i % K].append(sid)

if args.verbose:
    for k, subjects in enumerate(fold_assignments):
        print(f'  Fold {k}: {len(subjects)} subjects — {subjects[:3]}…')


# ---------------------------------------------------------------------------
# Diffusion schedule (shared across folds — schedule only, no weights)
# ---------------------------------------------------------------------------
diffusion_model = DiffusionModel(timesteps=args.timesteps)


# ---------------------------------------------------------------------------
# Helper: build UNet with consistent hyperparameters
# ---------------------------------------------------------------------------
def make_unet():
    """Instantiate the U-Net matching the paper architecture."""
    return UNet(
        audio_channels    = 2,                      # binaural
        time_embedding_dims = 128,                  # paper unspecified; kept small
        labels            = 440,                    # HUTUBS DOA positions
        head_embedding    = True,
        ears_embedding    = True,
        sequence_channels = (4, 8, 16, 32, 64),    # paper Section III-C
    )


def _stack_subject_chunk(items, start, end, device):
    """Create one inference batch from a subject's measurement-position items.

    The old code denoised one measurement position at a time. This helper keeps
    the same conditioning information but stacks positions into a batch so the
    600 reverse-diffusion steps run in parallel on the GPU.
    """
    chunk = items[start:end]

    hrir_gt = torch.stack([item['hrtf'] for item in chunk], dim=0)
    head = torch.stack([item['head_measurements'] for item in chunk], dim=0).float().to(device)
    ears = torch.stack([item['ear_measurements'] for item in chunk], dim=0).float().to(device)
    labels = torch.tensor(
        [item['measurement_point'] for item in chunk],
        dtype=torch.long,
        device=device,
    )

    return hrir_gt, head, ears, labels


def infer_subject_batched(items, unet, diffusion_model, norm_mean, norm_std, device, inference_batch_size):
    """Generate predictions for all measurement positions of one subject.

    This is mathematically the same reverse DDPM loop as the original inference
    code, but it processes many positions simultaneously. The output order is
    preserved, so LSD and the saved .mat files are computed in the same order as
    before.
    """
    if inference_batch_size <= 0:
        raise ValueError('--inference_batch_size must be a positive integer')

    hrir_gt_list = []
    hrir_pred_list = []

    with torch.inference_mode():
        for start in range(0, len(items), inference_batch_size):
            end = min(start + inference_batch_size, len(items))
            hrir_gt, head, ears, labels = _stack_subject_chunk(items, start, end, device)

            batch_size = end - start
            L = hrir_gt.shape[-1]
            audio_result = torch.randn((batch_size, 2, L), device=device)

            for i in reversed(range(diffusion_model.timesteps)):
                t = torch.full((batch_size,), i, dtype=torch.long, device=device)
                audio_result = diffusion_model.backward(
                    x              = audio_result,
                    t              = t,
                    model          = unet,
                    labels         = labels,
                    head_embedding = head,
                    ears_embedding = ears,
                )

            pred_denorm = (audio_result.detach().cpu() * norm_std) + norm_mean
            gt_denorm = (hrir_gt.cpu() * norm_std) + norm_mean

            hrir_pred_list.extend(pred_denorm.numpy())
            hrir_gt_list.extend(gt_denorm.numpy())

    return hrir_gt_list, hrir_pred_list


# ---------------------------------------------------------------------------
# Training — iterate over all K folds
# ---------------------------------------------------------------------------
if args.training:

    all_fold_lsd_v1 = []
    all_fold_lsd_v2 = []

    for fold_k in range(K):

        if fold_k < args.start_fold:
            print(f'Skipping fold {fold_k} (--start_fold={args.start_fold})')
            continue

        print(f"\n{'='*60}")
        print(f'FOLD {fold_k + 1} / {K}')
        print(f"{'='*60}")

        fold_dir = os.path.join(args.results_dir, f'fold_{fold_k}')
        os.makedirs(fold_dir, exist_ok=True)

        # Test: fold k.  Val: fold (k+1)%K.  Train: everything else.
        test_subject_ids  = fold_assignments[fold_k]
        val_subject_ids   = fold_assignments[(fold_k + 1) % K]
        train_subject_ids = [
            sid for i, fold in enumerate(fold_assignments)
            for sid in fold
            if i != fold_k and i != (fold_k + 1) % K
        ]

        if args.verbose:
            print(f'  Train: {len(train_subject_ids)} subjects')
            print(f'  Val:   {len(val_subject_ids)} subjects — {val_subject_ids}')
            print(f'  Test:  {len(test_subject_ids)} subjects — {test_subject_ids}')

        # Training set: augment=True enables ear-mirroring (doubles set size).
        # Val / test: augment=False — always evaluated on real unmodified HRIRs.
        train_dataset = HUTUBSDataset(
            hrtf_directory  = args.hrtf_directory,
            anthro_csv_path = args.anthro_csv_path,
            subject_ids     = train_subject_ids,
            augment         = True,
        )

        norm_kwargs = dict(
            norm_mean        = train_dataset.norm_mean,
            norm_std         = train_dataset.norm_std,
            norm_anthro_mean = train_dataset.norm_anthro_mean,
            norm_anthro_std  = train_dataset.norm_anthro_std,
        )

        val_dataset = HUTUBSDataset(
            hrtf_directory  = args.hrtf_directory,
            anthro_csv_path = args.anthro_csv_path,
            subject_ids     = val_subject_ids,
            augment         = False,
            **norm_kwargs,
        )

        test_dataset = HUTUBSDataset(
            hrtf_directory  = args.hrtf_directory,
            anthro_csv_path = args.anthro_csv_path,
            subject_ids     = test_subject_ids,
            augment         = False,
            **norm_kwargs,
        )

        if args.verbose:
            real_train = len(train_subject_ids) * 440
            print(f'  Train items: {len(train_dataset)} '
                  f'({real_train} real + {len(train_dataset) - real_train} mirrored)')
            print(f'  Val items:   {len(val_dataset)}')
            print(f'  Test items:  {len(test_dataset)}')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
            drop_last=True, collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
            drop_last=False, collate_fn=collate_fn,
        )

        # Fresh model for each fold.
        unet = make_unet().to(device)

        # Paper optimizer: Adam lr=0.001, 20% decay every 100 epochs.
        optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.9
        )

        model_path      = os.path.join(fold_dir, 'model.pt')
        noise_plot_path = os.path.join(fold_dir, 'noise_distribution.png')

        best_val_loss    = float('inf')
        early_stop_count = 0

        for epoch in tqdm.tqdm(range(args.epochs),
                                desc=f'Fold {fold_k} training', unit='epoch'):

            # ---- Training ----
            unet.train()
            train_losses = []

            for data in train_loader:
                batch             = data['hrtf'].to(device)
                label             = data['measurement_point'].to(device)
                head_measurements = data['head_measurements'].to(device)
                ears_measurements = data['ear_measurements'].to(device)

                t = torch.randint(
                    0, diffusion_model.timesteps, (batch.shape[0],),
                    dtype=torch.long, device=device
                )

                batch_noisy, noise = diffusion_model.forward(batch, t, device)
                batch_noisy = batch_noisy.float()

                # CFG dropout: disabled by default — paper does not use CFG.
                if args.p_uncond > 0.0 and random.random() < args.p_uncond:
                    label             = None
                    head_measurements = None
                    ears_measurements = None

                predicted_noise = unet(
                    batch_noisy, t,
                    labels         = label,
                    head_embedding = head_measurements,
                    ears_embedding = ears_measurements,
                )

                loss = torch.nn.functional.l1_loss(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # ---- Validation (noise-prediction loss, not full denoising) ----
            unet.eval()
            val_losses = []

            with torch.no_grad():
                for data in val_loader:
                    batch             = data['hrtf'].to(device)
                    label             = data['measurement_point'].to(device)
                    head_measurements = data['head_measurements'].to(device)
                    ears_measurements = data['ear_measurements'].to(device)

                    t = torch.randint(
                        0, diffusion_model.timesteps, (batch.shape[0],),
                        dtype=torch.long, device=device
                    )

                    batch_noisy, noise = diffusion_model.forward(batch, t, device)
                    batch_noisy = batch_noisy.float()

                    predicted_noise = unet(
                        batch_noisy, t,
                        labels         = label,
                        head_embedding = head_measurements,
                        ears_embedding = ears_measurements,
                    )
                    val_losses.append(
                        torch.nn.functional.l1_loss(noise, predicted_noise).item()
                    )

            mean_train = np.mean(train_losses)
            mean_val   = np.mean(val_losses)

            if args.verbose or epoch % 50 == 0:
                print(f'  Epoch {epoch:4d} | Train {mean_train:.6f} | '
                      f'Val {mean_val:.6f} | '
                      f'Patience {early_stop_count}/{args.early_stop_patience} | '
                      f'LR {scheduler.get_last_lr()[0]:.2e}')

            if mean_val < best_val_loss:
                best_val_loss    = mean_val
                early_stop_count = 0
                torch.save(unet.state_dict(), model_path)
                if args.verbose:
                    print(f'  ✓ Saved (val {best_val_loss:.6f})')
                with torch.no_grad():
                    plot_noise_distribution(
                        noise.detach().cpu(),
                        predicted_noise.detach().cpu(),
                        epoch,
                        plot_path=noise_plot_path,
                    )
            else:
                early_stop_count += 1
                if early_stop_count > args.early_stop_patience:
                    print(f'  Early stopping at epoch {epoch}.')
                    break

            scheduler.step()

        print(f'Fold {fold_k} training complete. Best val loss: {best_val_loss:.6f}')

        # ---- Inference on this fold's test set ----
        print(f'Running inference on fold {fold_k} test subjects…')
        unet.load_state_dict(torch.load(model_path, map_location=device))
        unet.eval()

        norm_mean = train_dataset.norm_mean
        norm_std  = train_dataset.norm_std

        subject_items = defaultdict(list)
        for item in test_dataset.items:
            subject_items[item['subject_id']].append(item)

        fold_lsd_v1 = []
        fold_lsd_v2 = []

        for subject_id, items in subject_items.items():
            hrir_gt_list, hrir_pred_list = infer_subject_batched(
                items                = items,
                unet                 = unet,
                diffusion_model      = diffusion_model,
                norm_mean            = norm_mean,
                norm_std             = norm_std,
                device               = device,
                inference_batch_size = args.inference_batch_size,
            )

            lsd_v1, (lsd_v1_l, lsd_v1_r) = lsd_paper(hrir_gt_list, hrir_pred_list)
            lsd_v2 = lsd_corrected(hrir_gt_list, hrir_pred_list)

            fold_lsd_v1.append(lsd_v1)
            fold_lsd_v2.append(lsd_v2)

            print(f'  Subject {subject_id:3d} | '
                  f'LSD v1: {lsd_v1:.3f} dB [L={lsd_v1_l:.3f} R={lsd_v1_r:.3f}] | '
                  f'LSD v2: {lsd_v2:.3f} dB')

            scipy.io.savemat(
                os.path.join(fold_dir, f'sub_{subject_id}_results.mat'),
                {
                    f'sub_{subject_id}_pred':     np.stack(hrir_pred_list),
                    f'sub_{subject_id}_gt':       np.stack(hrir_gt_list),
                    f'sub_{subject_id}_lsd_v1':   lsd_v1,
                    f'sub_{subject_id}_lsd_v1_l': lsd_v1_l,
                    f'sub_{subject_id}_lsd_v1_r': lsd_v1_r,
                    f'sub_{subject_id}_lsd_v2':   lsd_v2,
                }
            )

        fold_mean_v1 = float(np.mean(fold_lsd_v1))
        fold_mean_v2 = float(np.mean(fold_lsd_v2))
        all_fold_lsd_v1.append(fold_mean_v1)
        all_fold_lsd_v2.append(fold_mean_v2)

        print(f'Fold {fold_k} mean LSD — v1: {fold_mean_v1:.3f} dB | '
              f'v2: {fold_mean_v2:.3f} dB')

        scipy.io.savemat(
            os.path.join(fold_dir, 'fold_summary.mat'),
            {
                'test_subject_ids':   np.array(test_subject_ids),
                'mean_lsd_v1':        fold_mean_v1,
                'mean_lsd_v2':        fold_mean_v2,
                'per_subject_lsd_v1': np.array(fold_lsd_v1),
                'per_subject_lsd_v2': np.array(fold_lsd_v2),
            }
        )

    # ---- Cross-fold summary ----
    print(f"\n{'='*60}")
    print(f'{K}-FOLD CROSS-VALIDATION SUMMARY')
    print(f"{'='*60}")
    print(f'LSD v1 (paper, K=44, 0-15 kHz): '
          f'{np.mean(all_fold_lsd_v1):.3f} ± {np.std(all_fold_lsd_v1):.3f} dB')
    print(f'LSD v2 (corrected, 87 bins):     '
          f'{np.mean(all_fold_lsd_v2):.3f} ± {np.std(all_fold_lsd_v2):.3f} dB')
    print(f'Per-fold v1: {[f"{x:.3f}" for x in all_fold_lsd_v1]}')
    print(f'Per-fold v2: {[f"{x:.3f}" for x in all_fold_lsd_v2]}')

    scipy.io.savemat(
        os.path.join(args.results_dir, 'cv_summary.mat'),
        {
            'n_folds':          K,
            'mean_lsd_v1':      float(np.mean(all_fold_lsd_v1)),
            'std_lsd_v1':       float(np.std(all_fold_lsd_v1)),
            'mean_lsd_v2':      float(np.mean(all_fold_lsd_v2)),
            'std_lsd_v2':       float(np.std(all_fold_lsd_v2)),
            'per_fold_lsd_v1':  np.array(all_fold_lsd_v1),
            'per_fold_lsd_v2':  np.array(all_fold_lsd_v2),
        }
    )


# ---------------------------------------------------------------------------
# Inference-only mode
# ---------------------------------------------------------------------------
else:
    print('Inference mode: loading saved fold models from results_dir…')

    all_fold_lsd_v1 = []
    all_fold_lsd_v2 = []

    for fold_k in range(K):
        fold_dir   = os.path.join(args.results_dir, f'fold_{fold_k}')
        model_path = os.path.join(fold_dir, 'model.pt')

        if not os.path.exists(model_path):
            print(f'  Fold {fold_k}: no model at {model_path}, skipping.')
            continue

        test_subject_ids  = fold_assignments[fold_k]
        train_subject_ids = [
            sid for i, fold in enumerate(fold_assignments)
            for sid in fold
            if i != fold_k and i != (fold_k + 1) % K
        ]

        # Load training-set normalisation stats.
        train_dataset = HUTUBSDataset(
            hrtf_directory  = args.hrtf_directory,
            anthro_csv_path = args.anthro_csv_path,
            subject_ids     = train_subject_ids,
            augment         = False,
        )
        norm_kwargs = dict(
            norm_mean        = train_dataset.norm_mean,
            norm_std         = train_dataset.norm_std,
            norm_anthro_mean = train_dataset.norm_anthro_mean,
            norm_anthro_std  = train_dataset.norm_anthro_std,
        )
        test_dataset = HUTUBSDataset(
            hrtf_directory  = args.hrtf_directory,
            anthro_csv_path = args.anthro_csv_path,
            subject_ids     = test_subject_ids,
            augment         = False,
            **norm_kwargs,
        )

        unet = make_unet()
        unet.load_state_dict(torch.load(model_path, map_location=device))
        unet.eval().to(device)

        norm_mean = train_dataset.norm_mean
        norm_std  = train_dataset.norm_std

        subject_items = defaultdict(list)
        for item in test_dataset.items:
            subject_items[item['subject_id']].append(item)

        fold_lsd_v1 = []
        fold_lsd_v2 = []

        for subject_id, items in tqdm.tqdm(subject_items.items(),
                                            desc=f'Fold {fold_k}', unit='subject'):
            hrir_gt_list, hrir_pred_list = infer_subject_batched(
                items                = items,
                unet                 = unet,
                diffusion_model      = diffusion_model,
                norm_mean            = norm_mean,
                norm_std             = norm_std,
                device               = device,
                inference_batch_size = args.inference_batch_size,
            )

            lsd_v1, (lsd_v1_l, lsd_v1_r) = lsd_paper(hrir_gt_list, hrir_pred_list)
            lsd_v2 = lsd_corrected(hrir_gt_list, hrir_pred_list)
            fold_lsd_v1.append(lsd_v1)
            fold_lsd_v2.append(lsd_v2)

            print(f'  Fold {fold_k} Sub {subject_id:3d} | '
                  f'v1: {lsd_v1:.3f} dB [L={lsd_v1_l:.3f} R={lsd_v1_r:.3f}] | '
                  f'v2: {lsd_v2:.3f} dB')

            scipy.io.savemat(
                os.path.join(fold_dir, f'sub_{subject_id}_results.mat'),
                {
                    f'sub_{subject_id}_pred':     np.stack(hrir_pred_list),
                    f'sub_{subject_id}_gt':       np.stack(hrir_gt_list),
                    f'sub_{subject_id}_lsd_v1':   lsd_v1,
                    f'sub_{subject_id}_lsd_v1_l': lsd_v1_l,
                    f'sub_{subject_id}_lsd_v1_r': lsd_v1_r,
                    f'sub_{subject_id}_lsd_v2':   lsd_v2,
                }
            )

        fold_mean_v1 = float(np.mean(fold_lsd_v1))
        fold_mean_v2 = float(np.mean(fold_lsd_v2))
        all_fold_lsd_v1.append(fold_mean_v1)
        all_fold_lsd_v2.append(fold_mean_v2)
        print(f'Fold {fold_k} mean — v1: {fold_mean_v1:.3f} | v2: {fold_mean_v2:.3f}')

    print(f"\n{'='*60}")
    print(f'{K}-FOLD SUMMARY')
    print(f'LSD v1: {np.mean(all_fold_lsd_v1):.3f} ± {np.std(all_fold_lsd_v1):.3f} dB')
    print(f'LSD v2: {np.mean(all_fold_lsd_v2):.3f} ± {np.std(all_fold_lsd_v2):.3f} dB')
