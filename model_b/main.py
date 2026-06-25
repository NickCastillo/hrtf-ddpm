import torch
import argparse
import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for remote servers
import matplotlib.pyplot as plt
import torchaudio
import os
import json
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import HUTUBSDataset, collate_fn
from model import DiffusionModel, UNet
from utils import plot_noise_distribution, nmse, lsd

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TF32 on Ampere GPUs — free ~5-10% speedup, no effect on model behaviour
torch.set_float32_matmul_precision('high')

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='HRTF DDPM — per-fold training & inference')
parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True,
                    help='"train" or "infer"')
parser.add_argument('--fold', type=int, default=None,
                    help='Which fold to run (1-based). Omit to run all folds.')
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1000)
# ── LR & Scheduler ────────────────────────────────────────────────────────────
# StepLR (paper: lr=1e-3, 20% decay every 100 epochs) is preferred over
# CosineAnnealing because:
#   1. StepLR is monotonically decreasing — early stopping always saves the
#      checkpoint at the model's most fine-grained convergence point.
#      CosineAnnealing oscillates; with early stopping the saved checkpoint
#      can land at any arbitrary point in the cosine cycle.
#   2. Cosine restarts help escape local minima over long budgets; with
#      ~1000 epochs and early stopping at ~400-600, they add noise not value.
# A 10-epoch linear warmup is prepended to stabilise attention layers at init.
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_warmup_epochs', type=int, default=10)
parser.add_argument('--lr_step', type=int, default=100)
parser.add_argument('--lr_gamma', type=float, default=0.8)
parser.add_argument('--early_stop_patience', type=int, default=200)
parser.add_argument('--early_stop_min_epoch', type=int, default=100)
# ── Paths ─────────────────────────────────────────────────────────────────────
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                    help='Where to save model weights and splits.json')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='Where to save .mat files, plots, and .wav files')
parser.add_argument('--runs_dir', type=str, default='./runs',
                    help='Where to save TensorBoard event files')
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--hrtf_directory', type=str,
                    default='/nas/home/jalbarracin/datasets/HUTUBS/HRIRs')
parser.add_argument('--anthro_csv_path', type=str,
                    default='/nas/home/jalbarracin/datasets/HUTUBS/AntrhopometricMeasures.csv')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# ── Directory layout ──────────────────────────────────────────────────────────
#
#   checkpoint_dir/          ← splits.json + unet_fold1.pt ...
#   runs_dir/                ← TensorBoard event files per fold
#   results_dir/
#     mat/fold_N/            ← per-subject .mat files
#     plots/fold_N/          ← noise distribution + HRIR overlay PNGs
#     fold_N/sub_M/          ← generated .wav files
#
CKPT_DIR  = args.checkpoint_dir
RES_DIR   = args.results_dir
RUNS_DIR  = args.runs_dir
MAT_DIR   = os.path.join(RES_DIR, 'mat')
PLOTS_DIR = os.path.join(RES_DIR, 'plots')

for d in [RUNS_DIR, CKPT_DIR, RES_DIR, MAT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Pre-create per-fold run dirs so TensorBoard sees them immediately
for fi in range(5):   # max k_folds; harmless extras are ignored
    os.makedirs(os.path.join(RUNS_DIR, f'fold_{fi + 1}'), exist_ok=True)

# splits.json lives in checkpoints/ so it travels with the model weights
SPLITS_PATH = os.path.join(CKPT_DIR, 'splits.json')

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
hutubs_dataset = HUTUBSDataset(
    hrtf_directory=args.hrtf_directory,
    anthro_csv_path=args.anthro_csv_path,
)
print(f"Dataset size: {len(hutubs_dataset)} samples")

if os.path.exists(SPLITS_PATH):
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    print(f"Loaded existing splits from {SPLITS_PATH}")
else:
    splits = hutubs_dataset.get_kfold_splits(k=args.k_folds)
    splits_serialisable = [
        {k: [int(x) for x in v] for k, v in s.items()}
        for s in splits
    ]
    with open(SPLITS_PATH, 'w') as f:
        json.dump(splits_serialisable, f, indent=2)
    print(f"Saved splits to {SPLITS_PATH}")

# Resolve folds to run
if args.fold is not None:
    assert 1 <= args.fold <= len(splits), \
        f"--fold must be 1–{len(splits)}, got {args.fold}"
    fold_indices = [args.fold - 1]
else:
    fold_indices = list(range(len(splits)))

# ── Diffusion model ───────────────────────────────────────────────────────────
diffusion_model = DiffusionModel()   # 600 timesteps
NUM_CLASSES = 440


def build_unet():
    """
    Paper architecture: channel mults (4,8,16,32,64) x base_channels, 5 encoder blocks,
    self-attention after every encoder block (4 heads).
    base_channels=8 gives (32,64,128,256,512). Set to 1 for literal paper sizes.
    head_dim=13, ear_dim=24: all HUTUBS anthropometric features kept.
    """
    unet = UNet(
        audio_channels=2,
        labels=NUM_CLASSES,
        head_dim=13,
        ear_dim=24,
        base_channels=8,
    ).to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  UNet parameters: {n_params:,}")
    # torch.compile fuses kernels — ~20-30% speedup. First epoch is slower
    # (compilation overhead) but all subsequent epochs benefit.
    # Paper does not specify; neutral change.
    if hasattr(torch, 'compile'):
        unet = torch.compile(unet)
    return unet


def ckpt_path(fold_idx):
    return os.path.join(CKPT_DIR, f'unet_fold{fold_idx + 1}.pt')


def build_subject_point_index(dataset):
    idx_map = {}
    for i, item in enumerate(dataset.normalized_dataset):
        idx_map[(item['subject_id'], item['measurement_point'])] = i
    return idx_map


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def train_fold(fold_idx, split):
    fold_tag = f'fold_{fold_idx + 1}'
    print(f"\n{'='*60}")
    print(f"  TRAINING  FOLD {fold_idx + 1}/{args.k_folds}  |  "
          f"test subjects: {split['test_subjects']}")
    print(f"{'='*60}")

    # ── TensorBoard writer for this fold ─────────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, fold_tag))

    # Log hyperparameters once per fold so they appear in the HParams tab
    writer.add_hparams(
        hparam_dict={
            'lr': args.lr,
            'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_step': args.lr_step,
            'lr_gamma': args.lr_gamma,
            'batch_size': args.BATCH_SIZE,
            'early_stop_patience': args.early_stop_patience,
            'fold': fold_idx + 1,
        },
        metric_dict={'best_val_loss': float('inf')},   # updated at end
    )

    train_loader = DataLoader(
        Subset(hutubs_dataset, split['train']),
        batch_size=args.BATCH_SIZE, shuffle=True,
        num_workers=2, drop_last=True, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        Subset(hutubs_dataset, split['val']),
        batch_size=args.BATCH_SIZE, shuffle=False,
        num_workers=2, drop_last=False, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    unet = build_unet()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.lr_warmup_epochs,
    )
    step_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, step_scheduler],
        milestones=[args.lr_warmup_epochs],
    )

    # GradScaler for mixed precision — prevents FP16 underflow during backward
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_val_loss    = float('inf')
    early_stop_count = 0
    model_path       = ckpt_path(fold_idx)
    plots_fold_dir   = os.path.join(PLOTS_DIR, fold_tag)
    os.makedirs(plots_fold_dir, exist_ok=True)

    for epoch in tqdm.tqdm(range(args.epochs), desc=fold_tag, unit='epoch'):

        # ── Train ─────────────────────────────────────────────────────────────
        unet.train()
        train_losses = []
        for data in train_loader:
            batch = data['hrtf'].to(device, non_blocking=True).float()
            label = data['measurement_point'].to(device, non_blocking=True)
            head  = data['head_measurements'].to(device, non_blocking=True)
            ears  = data['ear_measurements'].to(device, non_blocking=True)

            t = torch.randint(0, diffusion_model.timesteps,
                              (batch.shape[0],), device=device).long()
            batch_noisy, noise = diffusion_model.forward(batch, t, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                predicted_noise = unet(
                    batch_noisy.float(), t,
                    labels=label, head_embedding=head, ears_embedding=ears,
                )
                loss = torch.nn.functional.l1_loss(noise, predicted_noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        unet.eval()
        val_losses = []
        last_noise = last_pred = None
        with torch.no_grad():
            for data in val_loader:
                batch = data['hrtf'].to(device, non_blocking=True).float()
                label = data['measurement_point'].to(device, non_blocking=True)
                head  = data['head_measurements'].to(device, non_blocking=True)
                ears  = data['ear_measurements'].to(device, non_blocking=True)

                t = torch.randint(0, diffusion_model.timesteps,
                                  (batch.shape[0],), device=device).long()
                batch_noisy, noise = diffusion_model.forward(batch, t, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    pred = unet(
                        batch_noisy.float(), t,
                        labels=label, head_embedding=head, ears_embedding=ears,
                    )
                val_losses.append(
                    torch.nn.functional.l1_loss(noise, pred).item()
                )
                last_noise, last_pred = noise, pred   # keep last batch for plots

        mean_train = np.mean(train_losses)
        mean_val   = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]

        # ── TensorBoard scalars ───────────────────────────────────────────────
        writer.add_scalar('Loss/train',      mean_train, epoch)
        writer.add_scalar('Loss/val',        mean_val,   epoch)
        writer.add_scalar('LR/learning_rate', current_lr, epoch)
        writer.add_scalar('EarlyStopping/patience_counter', early_stop_count, epoch)

        # Gradient norm (useful for diagnosing training stability)
        total_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in unet.parameters() if p.grad is not None
        ) ** 0.5
        writer.add_scalar('Gradients/total_norm', total_grad_norm, epoch)

        # ── Noise-distribution plot → TensorBoard + disk (every 50 epochs) ───
        if epoch % 50 == 0 and last_noise is not None:
            plot_path = os.path.join(plots_fold_dir, f'noise_ep{epoch:04d}.png')
            plot_noise_distribution(last_noise, last_pred, epoch, plot_path=plot_path)
            # Load saved PNG and push to TensorBoard as an image
            import torchvision.transforms.functional as tvf
            from PIL import Image
            img = Image.open(plot_path)
            img_tensor = tvf.to_tensor(img)          # (C, H, W) in [0,1]
            writer.add_image(f'NoiseDist/epoch_{epoch:04d}', img_tensor, epoch)

        if args.verbose or epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Train {mean_train:.4f} | Val {mean_val:.4f} | "
                  f"LR {current_lr:.2e} | Patience {early_stop_count}/{args.early_stop_patience}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save({
                'epoch':      torch.tensor(epoch),
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':   torch.tensor(best_val_loss),
                'fold':       torch.tensor(fold_idx + 1),
            }, model_path)
            early_stop_count = 0
            if args.verbose:
                print(f"  ✓ Checkpoint saved — val {best_val_loss:.4f}")
        elif epoch >= args.early_stop_min_epoch:
            early_stop_count += 1

        writer.flush()

        if early_stop_count > args.early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Update hparams with final metric so the HParams tab shows real values
    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_step': args.lr_step, 'lr_gamma': args.lr_gamma,
            'batch_size': args.BATCH_SIZE,
            'early_stop_patience': args.early_stop_patience,
            'fold': fold_idx + 1,
        },
        metric_dict={'best_val_loss': best_val_loss},
    )
    writer.close()
    print(f"  Fold {fold_idx + 1} done — best val loss: {best_val_loss:.4f}")
    return best_val_loss


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
def infer_fold(fold_idx, split, subj_point_index):
    model_path = ckpt_path(fold_idx)
    if not os.path.exists(model_path):
        print(f"No checkpoint at {model_path} — skipping fold {fold_idx + 1}")
        return [], []

    fold_tag = f'fold_{fold_idx + 1}'
    print(f"\n{'='*60}")
    print(f"  INFERENCE  FOLD {fold_idx + 1}/{args.k_folds}  |  "
          f"test subjects: {split['test_subjects']}")
    print(f"{'='*60}")

    # ── TensorBoard writer for inference metrics ──────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, fold_tag))

    unet = build_unet()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    if hasattr(torch, 'compile'):
        unet = torch.compile(unet)

    # Pre-compute schedule tensors on GPU once
    betas_gpu              = diffusion_model.betas.to(device)
    alphas_gpu             = diffusion_model.alphas.to(device)
    sqrt_recip_alphas_gpu  = torch.sqrt(1.0 / alphas_gpu)
    sqrt_one_minus_acp_gpu = torch.sqrt(1.0 - diffusion_model.alphas_cumprod.to(device))

    # Output dirs
    wav_dir   = os.path.join(RES_DIR, fold_tag)
    mat_fold  = os.path.join(MAT_DIR,  fold_tag)
    plot_fold = os.path.join(PLOTS_DIR, fold_tag)
    for d in [wav_dir, mat_fold, plot_fold]:
        os.makedirs(d, exist_ok=True)

    # Resume support
    progress_path = os.path.join(wav_dir, 'progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            print(f"  Resuming — {len(progress['done_subjects'])} subjects already done.")
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: progress.json corrupted, starting fold fresh.")
            progress = {'done_subjects': [], 'lsd': [], 'nmse': []}
    else:
        progress = {'done_subjects': [], 'lsd': [], 'nmse': []}

    done_set  = set(progress['done_subjects'])
    fold_lsd  = progress['lsd']
    fold_nmse = progress['nmse']
    INFER_BATCH = 64

    for subject_id in tqdm.tqdm(split['test_subjects'], desc=f'{fold_tag} infer'):
        if subject_id in done_set:
            continue

        hrir_sub  = []
        hrir_tsub = []
        nmse_sub  = []
        sub_wav_dir = os.path.join(wav_dir, f'sub_{subject_id}')
        os.makedirs(sub_wav_dir, exist_ok=True)

        data_ref = hutubs_dataset[subj_point_index[(subject_id, 0)]]
        head_1 = data_ref['head_measurements'].to(device).float()
        ears_1 = data_ref['ear_measurements'].to(device).float()
        g_std  = data_ref['global_std']
        g_mean = data_ref['global_mean']

        gt_hrirs     = []
        valid_points = []
        for c in range(440):
            key = (subject_id, c)
            if key not in subj_point_index:
                continue
            gt_hrirs.append(hutubs_dataset[subj_point_index[key]]['hrtf'])
            valid_points.append(c)

        n_points = len(valid_points)
        if n_points == 0:
            continue

        # ── Batched denoising (all positions in parallel) ─────────────────────
        all_results = []
        torch.manual_seed(42)

        for start in range(0, n_points, INFER_BATCH):
            end = min(start + INFER_BATCH, n_points)
            b   = end - start
            pts = valid_points[start:end]

            x            = torch.randn(b, 2, 256, device=device)
            labels_batch = torch.tensor(pts, device=device)
            head_batch   = head_1.unsqueeze(0).expand(b, -1)
            ears_batch   = ears_1.unsqueeze(0).expand(b, -1)

            with torch.no_grad():
                for i in reversed(range(diffusion_model.timesteps)):
                    t_batch      = torch.full((b,), i, dtype=torch.long, device=device)
                    betas_t      = betas_gpu[i].view(1, 1, 1)
                    sqrt_recip_t = sqrt_recip_alphas_gpu[i].view(1, 1, 1)
                    sqrt_omacp_t = sqrt_one_minus_acp_gpu[i].view(1, 1, 1)

                    predicted_noise = unet(
                        x, t_batch,
                        labels=labels_batch,
                        head_embedding=head_batch,
                        ears_embedding=ears_batch,
                    )
                    mean = sqrt_recip_t * (x - betas_t * predicted_noise / sqrt_omacp_t)
                    x = mean + (torch.sqrt(betas_t) * torch.randn_like(x) if i > 0 else 0)

            all_results.append(x.cpu())

        results_tensor = torch.cat(all_results, dim=0)   # (n_points, 2, 256)

        # ── Metrics, plots, saving ────────────────────────────────────────────
        gen_hrirs_mat  = []
        gt_hrirs_mat   = []

        for j, c in enumerate(valid_points):
            audio_result = results_tensor[j]
            hrir_test    = gt_hrirs[j]

            if torch.isnan(audio_result).any():
                print(f"  NaN: subject {subject_id}, point {c} — skipping")
                continue

            err = nmse(hrir_test=hrir_test, hrir_gen=audio_result)
            nmse_sub.append(err.item())

            # Save .wav (denormalised)
            hrir_save = (audio_result * g_std) + g_mean
            torchaudio.save(
                uri=os.path.join(sub_wav_dir, f'pos_{c}.wav'),
                src=hrir_save, sample_rate=44100,
            )

            hrir_sub.append(audio_result)
            hrir_tsub.append(hrir_test)
            gen_hrirs_mat.append(audio_result.float().numpy())
            gt_hrirs_mat.append(hrir_test.float().numpy())

        # ── Per-subject .mat ──────────────────────────────────────────────────
        if gen_hrirs_mat:
            lsd_val_sub = lsd(hrir_tsub, hrir_sub, len(hrir_sub), sr=44100) if hrir_sub else float('nan')
            sio.savemat(
                os.path.join(mat_fold, f'sub_{subject_id}.mat'),
                {
                    'hrir_gen':    np.array(gen_hrirs_mat,  dtype=np.float32),  # (n_valid, 2, 256)
                    'hrir_gt':     np.array(gt_hrirs_mat,   dtype=np.float32),
                    'nmse_values': np.array(nmse_sub,       dtype=np.float64),
                    'lsd_value':   np.float64(lsd_val_sub),
                    'subject_id':  np.int32(subject_id),
                    'positions':   np.array(valid_points,   dtype=np.int32),
                }
            )

        # ── Per-subject HRIR overlay plot → disk + TensorBoard ───────────────
        if hrir_sub:
            lsd_val = lsd_val_sub
            fold_lsd.append(lsd_val)
            fold_nmse.extend(nmse_sub)

            # Sample plot: first DOA of the subject
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(hrir_tsub[0][0].numpy(),  label='GT L',  linewidth=0.8)
            axes[0].plot(hrir_sub[0][0].numpy(),   label='Gen L', linewidth=0.8, linestyle='--')
            axes[0].set_title('Left channel — DOA 0')
            axes[0].legend(); axes[0].grid()

            axes[1].plot(hrir_tsub[0][1].numpy(),  label='GT R',  linewidth=0.8)
            axes[1].plot(hrir_sub[0][1].numpy(),   label='Gen R', linewidth=0.8, linestyle='--')
            axes[1].set_title('Right channel — DOA 0')
            axes[1].legend(); axes[1].grid()

            fig.suptitle(f'Subject {subject_id} | LSD={lsd_val:.3f} dB  '
                         f'NMSE={np.mean(nmse_sub):.4f}')
            plot_file = os.path.join(plot_fold, f'sub_{subject_id}_hrir.png')
            fig.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Push to TensorBoard under Inference/
            from PIL import Image
            import torchvision.transforms.functional as tvf
            img_tensor = tvf.to_tensor(Image.open(plot_file))
            writer.add_image(f'Inference/sub_{subject_id}_hrir', img_tensor, fold_idx + 1)
            writer.add_scalar(f'Inference/LSD_sub_{subject_id}',  float(lsd_val),           fold_idx + 1)
            writer.add_scalar(f'Inference/NMSE_sub_{subject_id}', float(np.mean(nmse_sub)), fold_idx + 1)

            print(f"  Subject {subject_id}: LSD={lsd_val:.3f} dB  "
                  f"NMSE={np.mean(nmse_sub):.4f}")

        # ── Persist progress (atomic write — crash-safe) ──────────────────────
        progress['done_subjects'].append(int(subject_id))
        progress['lsd']  = [float(v) for v in fold_lsd]
        progress['nmse'] = [float(v) for v in fold_nmse]
        tmp_path = progress_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(progress, f)
        os.replace(tmp_path, progress_path)   # atomic on POSIX and Windows

    # ── Fold-level summary scalars ─────────────────────────────────────────────
    if fold_lsd:
        writer.add_scalar('Inference/mean_LSD',  np.mean(fold_lsd),  fold_idx + 1)
        writer.add_scalar('Inference/mean_NMSE', np.mean(fold_nmse), fold_idx + 1)

        # Fold-level .mat
        sio.savemat(
            os.path.join(MAT_DIR, f'fold_{fold_idx + 1}_summary.mat'),
            {
                'lsd_per_subject':  np.array(fold_lsd),
                'nmse_per_position': np.array(fold_nmse),
                'test_subjects':    np.array(split['test_subjects']),
                'mean_lsd':         float(np.mean(fold_lsd)),
                'mean_nmse':        float(np.mean(fold_nmse)),
            }
        )

    writer.flush()
    writer.close()
    print(f"  Fold {fold_idx + 1} — mean LSD={np.mean(fold_lsd):.3f} dB  "
          f"mean NMSE={np.mean(fold_nmse):.4f}")
    return fold_lsd, fold_nmse


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ═════════════════════════════════════════════════════════════════════════════
if args.mode == 'train':
    all_val_losses = []
    for fi in fold_indices:
        all_val_losses.append(train_fold(fi, splits[fi]))

    if len(all_val_losses) > 1:
        print(f"\nCross-validation summary:")
        print(f"  Per-fold val losses : {[f'{v:.4f}' for v in all_val_losses]}")
        print(f"  Mean ± std          : "
              f"{np.mean(all_val_losses):.4f} ± {np.std(all_val_losses):.4f}")

else:
    subj_point_index = build_subject_point_index(hutubs_dataset)
    all_lsd, all_nmse_vals = [], []

    for fi in fold_indices:
        fl, fn = infer_fold(fi, splits[fi], subj_point_index)
        all_lsd.extend(fl)
        all_nmse_vals.extend(fn)

    if all_lsd:
        print(f"\nOverall LSD  : {np.mean(all_lsd):.3f} ± {np.std(all_lsd):.3f} dB")
        print(f"Overall NMSE : {np.mean(all_nmse_vals):.4f} ± {np.std(all_nmse_vals):.4f}")

        # Global summary .mat
        sio.savemat(
            os.path.join(MAT_DIR, 'all_folds_summary.mat'),
            {
                'lsd_all':  np.array(all_lsd),
                'nmse_all': np.array(all_nmse_vals),
                'mean_lsd':  float(np.mean(all_lsd)),
                'mean_nmse': float(np.mean(all_nmse_vals)),
            }
        )
        # Also save Excel for convenience
        pd.DataFrame({'lsd': all_lsd}).to_excel(
            os.path.join(RES_DIR, 'lsd_values.xlsx'), index=False)
        pd.DataFrame({'nmse': all_nmse_vals}).to_excel(
            os.path.join(RES_DIR, 'nmse_values.xlsx'), index=False)