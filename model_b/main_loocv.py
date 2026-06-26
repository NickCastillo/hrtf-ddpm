"""
main_loocv.py — Leave-One-Out Cross-Validation training for Model B.

LOOCV strategy (matching the paper exactly):
  - Test set  : 1 subject (the left-out subject)
  - Val set   : VAL_SUBJECTS randomly chosen subjects, fixed across all rounds
                (held out for early stopping only — not the test subject)
  - Train set : remaining 92 - VAL_SUBJECTS subjects

This script is independent — it imports model, dataset, and utils from
the model_b source directory. Point --src_dir to wherever those files live.

Usage — run 3 specific subjects to estimate LOOCV performance:
    python main_loocv.py \
        --src_dir /content/hrtf-ddpm/model_b \
        --loocv_subjects 1 45 83 \
        --checkpoint_dir /content/drive/MyDrive/master-thesis/loocv/checkpoints \
        --results_dir    /content/drive/MyDrive/master-thesis/loocv/results \
        --runs_dir       /content/drive/MyDrive/master-thesis/loocv/runs \
        --hrtf_directory /content/hrtf-ddpm/HUTUBS/HRIRs \
        --anthro_csv_path /content/hrtf-ddpm/HUTUBS/AnthroPometricMeasures.csv \
        --mode train

    python main_loocv.py \
        --src_dir /content/hrtf-ddpm/model_b \
        --loocv_subjects 1 45 83 \
        --checkpoint_dir /content/drive/MyDrive/master-thesis/loocv/checkpoints \
        --results_dir    /content/drive/MyDrive/master-thesis/loocv/results \
        --runs_dir       /content/drive/MyDrive/master-thesis/loocv/runs \
        --hrtf_directory /content/hrtf-ddpm/HUTUBS/HRIRs \
        --anthro_csv_path /content/hrtf-ddpm/HUTUBS/AnthroPometricMeasures.csv \
        --mode infer
"""

import sys
import os
import argparse
import json
import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# ── Inject model_b source dir so we reuse its model, dataset, utils ───────────
parser = argparse.ArgumentParser(description='HRTF DDPM — LOOCV training & inference')
parser.add_argument('--src_dir', type=str,
                    default='/content/hrtf-ddpm/model_b',
                    help='Path to directory containing model.py, dataset.py, utils.py')
parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True)
parser.add_argument('--loocv_subjects', type=int, nargs='+', default=None,
                    help='Subject IDs to use as test subjects. '
                         'Omit to run full LOOCV over all 93 subjects.')
parser.add_argument('--val_subjects', type=int, default=5,
                    help='Number of subjects held out for validation / early stopping '
                         '(same fixed set across all LOOCV rounds, seed=42)')
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_warmup_epochs', type=int, default=10)
parser.add_argument('--lr_step', type=int, default=100)
parser.add_argument('--lr_gamma', type=float, default=0.8)
parser.add_argument('--early_stop_patience', type=int, default=200)
parser.add_argument('--early_stop_min_epoch', type=int, default=100)
parser.add_argument('--checkpoint_dir', type=str, default='./loocv/checkpoints')
parser.add_argument('--results_dir',    type=str, default='./loocv/results')
parser.add_argument('--runs_dir',       type=str, default='./loocv/runs')
parser.add_argument('--hrtf_directory', type=str,
                    default='/nas/home/jalbarracin/datasets/HUTUBS/HRIRs')
parser.add_argument('--anthro_csv_path', type=str,
                    default='/nas/home/jalbarracin/datasets/HUTUBS/AntrhopometricMeasures.csv')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# Insert model_b source so imports resolve correctly
sys.path.insert(0, os.path.abspath(args.src_dir))
from model   import DiffusionModel, UNet          # model_b architecture
from dataset import HUTUBSDataset, collate_fn
from utils   import plot_noise_distribution, nmse, lsd, itd_error, pbc

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')

# ── Directories ───────────────────────────────────────────────────────────────
CKPT_DIR  = args.checkpoint_dir
RES_DIR   = args.results_dir
RUNS_DIR  = args.runs_dir
MAT_DIR   = os.path.join(RES_DIR, 'mat')
PLOTS_DIR = os.path.join(RES_DIR, 'plots')

for d in [RUNS_DIR, CKPT_DIR, RES_DIR, MAT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
hutubs_dataset = HUTUBSDataset(
    hrtf_directory=args.hrtf_directory,
    anthro_csv_path=args.anthro_csv_path,
)
all_subjects = hutubs_dataset.valid_subject_indices   # list of valid 1-based IDs
print(f"Dataset size: {len(hutubs_dataset)} samples across {len(all_subjects)} subjects")

# ── LOOCV subject list ────────────────────────────────────────────────────────
if args.loocv_subjects is not None:
    # Validate requested subjects exist
    invalid = [s for s in args.loocv_subjects if s not in all_subjects]
    if invalid:
        raise ValueError(f"Subjects not in dataset: {invalid}. "
                         f"Valid range: {min(all_subjects)}–{max(all_subjects)}")
    test_subjects = args.loocv_subjects
else:
    test_subjects = list(all_subjects)   # full LOOCV

print(f"LOOCV rounds to run: {len(test_subjects)} "
      f"(subjects: {test_subjects})")

# ── Fixed validation set (same across all LOOCV rounds) ──────────────────────
# Exclude test subjects from val pool so val set is always clean.
# Seed fixed so val set is reproducible across restarts.
rng = np.random.default_rng(42)
val_pool = [s for s in all_subjects if s not in set(test_subjects)]
val_fixed = [int(x) for x in rng.choice(val_pool, size=min(args.val_subjects, len(val_pool)),
                                              replace=False)]
print(f"Fixed validation subjects (all rounds): {sorted(val_fixed)}")

# Save the val set alongside checkpoints for reproducibility
val_meta_path = os.path.join(CKPT_DIR, 'loocv_meta.json')
if not os.path.exists(val_meta_path):
    with open(val_meta_path, 'w') as f:
        json.dump({
            'val_subjects':  sorted(val_fixed),
            'test_subjects': sorted(test_subjects),
            'all_subjects':  sorted(all_subjects),
        }, f, indent=2)

# ── Subject → dataset-index lookup ───────────────────────────────────────────
subj_to_indices = {}
for i, item in enumerate(hutubs_dataset.normalized_dataset):
    subj_to_indices.setdefault(item['subject_id'], []).append(i)

subj_point_index = {
    (item['subject_id'], item['measurement_point']): i
    for i, item in enumerate(hutubs_dataset.normalized_dataset)
}

# ── Model constants ───────────────────────────────────────────────────────────
diffusion_model = DiffusionModel()   # 600 timesteps
NUM_CLASSES = 440


def build_unet():
    unet = UNet(
        audio_channels=2,
        labels=NUM_CLASSES,
        head_dim=13,
        ear_dim=24,
        base_channels=8,   # (32,64,128,256,512) — paper-like model B
    ).to(device)
    n = sum(p.numel() for p in unet.parameters())
    print(f"  UNet parameters: {n:,}")
    if hasattr(torch, 'compile'):
        unet = torch.compile(unet)
    return unet


def ckpt_path(test_subject_id):
    return os.path.join(CKPT_DIR, f'unet_loocv_sub{test_subject_id}.pt')


def get_split_indices(test_subject_id):
    """
    For a given test subject, return train/val/test dataset indices.
    Val set is always val_fixed (same across rounds).
    Train = all subjects except test and val.
    """
    val_set  = set(val_fixed)
    test_set = {test_subject_id}
    train_indices, val_indices, test_indices = [], [], []
    for sid, indices in subj_to_indices.items():
        if sid in test_set:
            test_indices.extend(indices)
        elif sid in val_set:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return train_indices, val_indices, test_indices


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def train_round(test_subject_id):
    tag = f'loocv_sub{test_subject_id}'
    print(f"\n{'='*60}")
    print(f"  LOOCV TRAIN  |  test subject: {test_subject_id}  |  "
          f"val subjects: {sorted(val_fixed)}")
    print(f"{'='*60}")

    train_indices, val_indices, _ = get_split_indices(test_subject_id)
    print(f"  Train: {len(train_indices)} samples  "
          f"Val: {len(val_indices)} samples")

    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, tag))
    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_step': args.lr_step, 'lr_gamma': args.lr_gamma,
            'batch_size': args.BATCH_SIZE,
            'early_stop_patience': args.early_stop_patience,
            'test_subject': test_subject_id,
        },
        metric_dict={'best_val_loss': float('inf')},
    )

    train_loader = DataLoader(
        Subset(hutubs_dataset, train_indices),
        batch_size=args.BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        Subset(hutubs_dataset, val_indices),
        batch_size=args.BATCH_SIZE, shuffle=False,
        num_workers=4, drop_last=False, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    unet      = build_unet()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.lr_warmup_epochs,
    )
    step_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, step_sched],
        milestones=[args.lr_warmup_epochs],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_val_loss    = float('inf')
    early_stop_count = 0
    model_path       = ckpt_path(test_subject_id)
    plots_dir        = os.path.join(PLOTS_DIR, tag)
    os.makedirs(plots_dir, exist_ok=True)

    for epoch in tqdm.tqdm(range(args.epochs), desc=tag, unit='epoch'):

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
                pred = unet(batch_noisy.float(), t,
                            labels=label, head_embedding=head, ears_embedding=ears)
                loss = torch.nn.functional.l1_loss(noise, pred)
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
                    pred = unet(batch_noisy.float(), t,
                                labels=label, head_embedding=head, ears_embedding=ears)
                val_losses.append(
                    torch.nn.functional.l1_loss(noise, pred).item()
                )
                last_noise, last_pred = noise, pred

        mean_train = np.mean(train_losses)
        mean_val   = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar('Loss/train',       mean_train, epoch)
        writer.add_scalar('Loss/val',         mean_val,   epoch)
        writer.add_scalar('LR/learning_rate', current_lr, epoch)
        writer.add_scalar('EarlyStopping/patience_counter', early_stop_count, epoch)

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in unet.parameters() if p.grad is not None
        ) ** 0.5
        writer.add_scalar('Gradients/total_norm', grad_norm, epoch)

        if epoch % 50 == 0 and last_noise is not None:
            plot_path = os.path.join(plots_dir, f'noise_ep{epoch:04d}.png')
            plot_noise_distribution(last_noise, last_pred, epoch, plot_path=plot_path)
            import torchvision.transforms.functional as tvf
            from PIL import Image
            writer.add_image(f'NoiseDist/epoch_{epoch:04d}',
                             tvf.to_tensor(Image.open(plot_path)), epoch)

        if args.verbose or epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Train {mean_train:.4f} | Val {mean_val:.4f} | "
                  f"LR {current_lr:.2e} | Patience {early_stop_count}/{args.early_stop_patience}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save({
                'epoch':              torch.tensor(epoch),
                'model_state_dict':   unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':           torch.tensor(best_val_loss),
                'test_subject':       torch.tensor(test_subject_id),
            }, model_path)
            early_stop_count = 0
            if args.verbose:
                print(f"  ✓ Saved (val {best_val_loss:.4f})")
        elif epoch >= args.early_stop_min_epoch:
            early_stop_count += 1

        writer.flush()

        if early_stop_count > args.early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_step': args.lr_step, 'lr_gamma': args.lr_gamma,
            'batch_size': args.BATCH_SIZE,
            'early_stop_patience': args.early_stop_patience,
            'test_subject': test_subject_id,
        },
        metric_dict={'best_val_loss': best_val_loss},
    )
    writer.close()
    print(f"  Subject {test_subject_id} done — best val loss: {best_val_loss:.4f}")
    return best_val_loss


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
def infer_round(test_subject_id):
    model_path = ckpt_path(test_subject_id)
    if not os.path.exists(model_path):
        print(f"  No checkpoint for subject {test_subject_id} — skipping")
        return None

    tag = f'loocv_sub{test_subject_id}'
    print(f"\n{'='*60}")
    print(f"  LOOCV INFER  |  test subject: {test_subject_id}")
    print(f"{'='*60}")

    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, tag))

    unet = build_unet()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()

    # Pre-compute schedule tensors on GPU
    betas_gpu              = diffusion_model.betas.to(device)
    alphas_gpu             = diffusion_model.alphas.to(device)
    sqrt_recip_alphas_gpu  = torch.sqrt(1.0 / alphas_gpu)
    sqrt_one_minus_acp_gpu = torch.sqrt(1.0 - diffusion_model.alphas_cumprod.to(device))

    mat_dir  = os.path.join(MAT_DIR,   tag)
    plot_dir = os.path.join(PLOTS_DIR, tag)
    prog_dir = os.path.join(RES_DIR,   tag)
    for d in [mat_dir, plot_dir, prog_dir]:
        os.makedirs(d, exist_ok=True)

    # Resume support
    progress_path  = os.path.join(prog_dir, 'progress.json')
    empty_progress = {'done': False, 'lsd_L': None, 'lsd_R': None,
                      'lsd_avg': None, 'itd': None, 'pbc': None,
                      'nmse_values': []}
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            for k, v in empty_progress.items():
                progress.setdefault(k, v)
            if progress.get('done'):
                print(f"  Subject {test_subject_id} already complete — loading from progress.")
                return {k: progress[k] for k in
                        ['lsd_L', 'lsd_R', 'lsd_avg', 'itd', 'pbc', 'nmse_values']}
        except (json.JSONDecodeError, KeyError):
            progress = empty_progress.copy()
    else:
        progress = empty_progress.copy()

    # Collect all 440 positions for this subject
    gt_hrirs     = []
    valid_points = []
    for c in range(440):
        key = (test_subject_id, c)
        if key not in subj_point_index:
            continue
        gt_hrirs.append(hutubs_dataset[subj_point_index[key]]['hrtf'])
        valid_points.append(c)

    n_points = len(valid_points)
    if n_points == 0:
        print(f"  No positions found for subject {test_subject_id}")
        return None

    data_ref = hutubs_dataset[subj_point_index[(test_subject_id, valid_points[0])]]
    head_1 = data_ref['head_measurements'].to(device).float()
    ears_1 = data_ref['ear_measurements'].to(device).float()

    # ── Batched denoising ─────────────────────────────────────────────────────
    INFER_BATCH = 64
    all_results = []
    torch.manual_seed(42)

    for start in tqdm.tqdm(range(0, n_points, INFER_BATCH),
                            desc=f'  sub{test_subject_id} denoise', leave=False):
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

                pred = unet(x, t_batch,
                            labels=labels_batch,
                            head_embedding=head_batch,
                            ears_embedding=ears_batch)
                mean = sqrt_recip_t * (x - betas_t * pred / sqrt_omacp_t)
                x = mean + (torch.sqrt(betas_t) * torch.randn_like(x) if i > 0 else 0)

        all_results.append(x.cpu())

    results_tensor = torch.cat(all_results, dim=0)   # (n_points, 2, 256)

    # ── Metrics ───────────────────────────────────────────────────────────────
    hrir_sub      = []
    hrir_tsub     = []
    nmse_vals     = []
    gen_hrirs_mat = []
    gt_hrirs_mat  = []

    for j, c in enumerate(valid_points):
        audio_result = results_tensor[j]
        hrir_test    = gt_hrirs[j]

        if torch.isnan(audio_result).any():
            print(f"  NaN at position {c} — skipping")
            continue

        nmse_vals.append(nmse(hrir_test, audio_result).item())
        hrir_sub.append(audio_result)
        hrir_tsub.append(hrir_test)
        gen_hrirs_mat.append(audio_result.float().numpy())
        gt_hrirs_mat.append(hrir_test.float().numpy())

    if not hrir_sub:
        print(f"  No valid positions for subject {test_subject_id}")
        return None

    lsd_vals    = lsd(hrir_tsub, hrir_sub, len(hrir_sub), sr=44100)
    lsd_L       = lsd_vals['L']
    lsd_R       = lsd_vals['R']
    lsd_avg     = lsd_vals['avg']
    itd_val     = itd_error(hrir_tsub, hrir_sub, sr=44100)
    pbc_val     = pbc(hrir_tsub, hrir_sub, sr=44100)

    print(f"  Subject {test_subject_id}: "
          f"LSD_L={lsd_L:.3f}  LSD_R={lsd_R:.3f}  LSD_avg={lsd_avg:.3f} dB  "
          f"ITD={itd_val:.2f} µs  PBC={pbc_val:.3f} dB  "
          f"NMSE={np.mean(nmse_vals):.4f}")

    # ── Save .mat ─────────────────────────────────────────────────────────────
    sio.savemat(
        os.path.join(mat_dir, f'sub_{test_subject_id}.mat'),
        {
            'hrir_gen':    np.array(gen_hrirs_mat, dtype=np.float32),
            'hrir_gt':     np.array(gt_hrirs_mat,  dtype=np.float32),
            'nmse_values': np.array(nmse_vals,     dtype=np.float64),
            'lsd_L':       np.float64(lsd_L),
            'lsd_R':       np.float64(lsd_R),
            'lsd_avg':     np.float64(lsd_avg),
            'itd_error':   np.float64(itd_val),
            'pbc_value':   np.float64(pbc_val),
            'subject_id':  np.int32(test_subject_id),
            'positions':   np.array(valid_points, dtype=np.int32),
        }
    )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer.add_scalar('Inference/LSD_L',   float(lsd_L),             test_subject_id)
    writer.add_scalar('Inference/LSD_R',   float(lsd_R),             test_subject_id)
    writer.add_scalar('Inference/LSD_avg', float(lsd_avg),           test_subject_id)
    writer.add_scalar('Inference/ITD',     float(itd_val),           test_subject_id)
    writer.add_scalar('Inference/PBC',     float(pbc_val),           test_subject_id)
    writer.add_scalar('Inference/NMSE',    float(np.mean(nmse_vals)),test_subject_id)
    writer.flush()
    writer.close()

    # ── Atomic progress write ─────────────────────────────────────────────────
    progress.update({
        'done':       True,
        'lsd_L':      float(lsd_L),
        'lsd_R':      float(lsd_R),
        'lsd_avg':    float(lsd_avg),
        'itd':        float(itd_val),
        'pbc':        float(pbc_val),
        'nmse_values':[float(v) for v in nmse_vals],
    })
    tmp = progress_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(progress, f)
    os.replace(tmp, progress_path)

    return {'lsd_L': lsd_L, 'lsd_R': lsd_R, 'lsd_avg': lsd_avg,
            'itd': itd_val, 'pbc': pbc_val, 'nmse_values': nmse_vals}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ═════════════════════════════════════════════════════════════════════════════
if args.mode == 'train':
    val_losses = []
    for subj in test_subjects:
        vl = train_round(subj)
        val_losses.append((subj, vl))

    print(f"\n{'='*60}")
    print(f"LOOCV Training Summary ({len(val_losses)} rounds)")
    for subj, vl in val_losses:
        print(f"  Subject {subj:3d}: best val loss = {vl:.4f}")

else:   # infer
    results = []
    for subj in test_subjects:
        r = infer_round(subj)
        if r is not None:
            results.append({'subject_id': subj, **{
                k: v for k, v in r.items() if k != 'nmse_values'
            }, 'nmse_mean': float(np.mean(r['nmse_values']))})

    if results:
        all_lsd_L   = [r['lsd_L']    for r in results]
        all_lsd_R   = [r['lsd_R']    for r in results]
        all_lsd_avg = [r['lsd_avg']  for r in results]
        all_itd     = [r['itd']      for r in results]
        all_pbc     = [r['pbc']      for r in results]
        all_nmse    = [r['nmse_mean']for r in results]

        print(f"\n{'='*60}")
        print(f"LOOCV Inference Summary ({len(results)} subjects)")
        print(f"  LSD_L  : {np.mean(all_lsd_L):.3f} ± {np.std(all_lsd_L):.3f} dB")
        print(f"  LSD_R  : {np.mean(all_lsd_R):.3f} ± {np.std(all_lsd_R):.3f} dB")
        print(f"  LSD_avg: {np.mean(all_lsd_avg):.3f} ± {np.std(all_lsd_avg):.3f} dB")
        print(f"  ITD    : {np.mean(all_itd):.2f} ± {np.std(all_itd):.2f} µs")
        print(f"  PBC    : {np.mean(all_pbc):.3f} ± {np.std(all_pbc):.3f} dB")
        print(f"  NMSE   : {np.mean(all_nmse):.4f} ± {np.std(all_nmse):.4f}")
        print(f"  Paper  : 5.1 dB LSD")
        print(f"{'='*60}")

        # Save summary
        sio.savemat(
            os.path.join(MAT_DIR, 'loocv_summary.mat'),
            {
                'subject_ids':  np.array([r['subject_id'] for r in results], dtype=np.int32),
                'lsd_L_all':    np.array(all_lsd_L,   dtype=np.float64),
                'lsd_R_all':    np.array(all_lsd_R,   dtype=np.float64),
                'lsd_avg_all':  np.array(all_lsd_avg, dtype=np.float64),
                'itd_all':      np.array(all_itd,     dtype=np.float64),
                'pbc_all':      np.array(all_pbc,     dtype=np.float64),
                'nmse_all':     np.array(all_nmse,    dtype=np.float64),
                'mean_lsd_avg': float(np.mean(all_lsd_avg)),
                'mean_itd':     float(np.mean(all_itd)),
                'mean_pbc':     float(np.mean(all_pbc)),
                'mean_nmse':    float(np.mean(all_nmse)),
            }
        )
        pd.DataFrame(results).to_excel(
            os.path.join(RES_DIR, 'loocv_metrics_per_subject.xlsx'), index=False)