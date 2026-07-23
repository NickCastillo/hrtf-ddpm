"""
Loss-function ablation driver (L1 vs L2 vs combined loss).

This file is a restructured copy of main_model/main.py: same dataset
loading, same UNet, same training/inference loops, same metrics. The two
things that differ:

  1. The loss function used in the train/val step is selected via
     --loss_type (l1 / l2 / combined / all) and computed by
     loss_exp/losses.py::compute_loss(), instead of always calling
     utils.combined_loss().
  2. --condition defaults to 'A' (unconditioned -- ear_dim=0, image_dim=0)
     instead of 'B', and -- unlike main_model/main.py -- this script does
     NOT force --dataset hutubs to condition B. The whole point of this
     script is to screen the loss function on the simplest/cheapest model
     (no subject-level conditioning) so the comparison isn't confounded by
     conditioning effects; that requires condition A to be selectable on
     HUTUBS too. Mechanically this is safe: ear_dim=0 just means the
     UNet's Block never reads the ears_embedding kwarg, whether or not the
     dataset supplies it.

--loss_type all (the default) runs every loss type back-to-back in ONE
invocation, sequentially, each under its own namespaced
checkpoint/results/runs directory (so they never clobber each other), and
prints + saves a comparison table at the end. Every loss type trains/
evaluates on the IDENTICAL fold split (dataset.get_kfold_splits is
seeded and computed once, up front, before the loss-type loop) so the
comparison is apples-to-apples.

Recommended usage for a cheap first screen (see chat discussion): run a
single, representative fold rather than the full k_folds --

    python loss_exp/main.py --mode train --dataset hutubs --fold 3
    python loss_exp/main.py --mode infer --dataset hutubs --fold 3

Both commands default to --condition A --loss_type all, so each trains/
infers L1, then L2, then combined, on fold 3 only. Pick "3" (or whatever
fold) as the fold whose baseline LSD is closest to your existing 5-fold
mean, so it's representative rather than an outlier -- see chat.
"""
import torch
import argparse
import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for remote servers
import matplotlib.pyplot as plt
import os
import sys
import json
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# ── Make main_model/ importable ──────────────────────────────────────────────
# loss_exp/ and main_model/ are assumed to be sibling folders (see module
# docstring). This must run before the dataset/model/utils imports below.
_MAIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'main_model')
if _MAIN_MODEL_DIR not in sys.path:
    sys.path.insert(0, _MAIN_MODEL_DIR)

from dataset import HUTUBSDataset, SONICOMDataset, collate_fn
from model import DiffusionModel, UNet
from utils import plot_noise_distribution, nmse, lsd, itd_error, pbc, EMA, load_matching_state_dict
from losses import compute_loss, LOSS_TYPES   # local to loss_exp/ -- no path hack needed

# Conditioning per SONICOM ablation condition (ear_dim, use_image). head_dim
# stays 0 everywhere -- neither dataset has head/torso measurements.
# IMAGE_FEAT_DIM is the width of the vector ImageEncoder produces (see
# model.py), not a flag -- there's no reason to tune it per run.
IMAGE_FEAT_DIM = 32
CONDITIONS = {
    'A': dict(ear_dim=0,  use_image=False),   # unconditioned
    'B': dict(ear_dim=24, use_image=False),   # anthro-only (= HUTUBS baseline)
    'C': dict(ear_dim=0,  use_image=True),    # image-only
    'D': dict(ear_dim=24, use_image=True),    # anthro + image
}

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TF32 on Ampere GPUs — free ~5-10% speedup, no effect on model behaviour
torch.set_float32_matmul_precision('high')

def str2bool(v):
    """Allow --flag true/false, yes/no, 1/0, t/f on the command line."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='HRTF DDPM — loss function ablation (L1 / L2 / combined)')
parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True,
                    help='"train" or "infer"')
parser.add_argument('--fold', type=int, nargs='+', default=None,
                    help='Which fold(s) to run (1-based), space-separated, '
                         'e.g. --fold 2 3 4 5. A single value (--fold 3) still '
                         'works as before. Omit to run all folds. For a cheap '
                         'first screen, pick ONE representative fold (see '
                         'module docstring / chat discussion).')
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1000)
# ── LR & Scheduler ────────────────────────────────────────────────────────────
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_warmup_epochs', type=int, default=10)
parser.add_argument('--lr_plateau_patience', type=int, default=25,
                    help='Epochs with no val-loss improvement before halving the LR.')
parser.add_argument('--lr_plateau_factor', type=float, default=0.5,
                    help='Multiplicative LR reduction factor on plateau.')
parser.add_argument('--lr_min', type=float, default=1e-6,
                    help='Floor for the LR — plateau reductions stop here.')
parser.add_argument('--early_stop_patience', type=int, default=200)
parser.add_argument('--early_stop_min_epoch', type=int, default=100)
# ── EMA ───────────────────────────────────────────────────────────────────────
parser.add_argument('--ema_decay', type=float, default=0.999)
parser.add_argument('--use_ema', type=str2bool, default=True,
                    help='true/false — whether inference loads the EMA weights.')
# ── Loss ──────────────────────────────────────────────────────────────────────
parser.add_argument('--loss_type', type=str, choices=list(LOSS_TYPES) + ['all'], default='all',
                    help='Which loss to train/evaluate with. "all" (default) runs '
                         'l1, l2, and combined back-to-back in this single '
                         'invocation, each under its own namespaced output dirs, '
                         'and prints/saves a comparison table at the end.')
parser.add_argument('--loss_freq_weight', type=float, default=0.3,
                    help='Weight on the frequency-magnitude term of the combined '
                         'loss (ignored for loss_type l1/l2): final = (1 - w) * '
                         'L1_time + w * L1_freq_mag.')
# ── Architecture ablations ───────────────────────────────────────────────────
parser.add_argument('--full_attention', type=str2bool, default=False)
# ── Dataset / conditioning ────────────────────────────────────────────────────
parser.add_argument('--dataset', type=str, choices=['hutubs', 'sonicom'], default='hutubs',
                    help='Which dataset to train/infer on.')
parser.add_argument('--condition', type=str, choices=list(CONDITIONS), default='A',
                    help='Conditioning condition: A=unconditioned (default here -- '
                         'see module docstring for why), B=anthro-only, '
                         'C=image-only, D=anthro+image. Unlike main_model/main.py, '
                         'this script does NOT force --dataset hutubs to condition B.')
parser.add_argument('--pretrained_checkpoint', type=str, default=None)
parser.add_argument('--reset_cond_fuse', type=str2bool, default=True)
# ── Model identity / paths ───────────────────────────────────────────────────
parser.add_argument('--model_name', type=str, default=None,
                    help='Base tag for this experiment. Each loss type actually '
                         'run gets its own "<model_name>_<loss_type>" namespaced '
                         'checkpoint/results/runs dir so l1/l2/combined never '
                         'clobber each other. Defaults to '
                         '"<DATASET>_lossexp_<condition>".')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='Only honoured when --loss_type is a single value (not '
                         '"all") — with multiple loss types this would make them '
                         'overwrite each other, so it is ignored and the auto '
                         'per-loss-type path is used instead.')
parser.add_argument('--results_dir', type=str, default=None,
                    help='Same caveat as --checkpoint_dir.')
parser.add_argument('--runs_dir', type=str, default=None,
                    help='Same caveat as --checkpoint_dir.')
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--hrtf_directory', type=str, default=None,
                    help='Defaults to ./HUTUBS/HRIRs or ./SONICOM/HRIRs depending on --dataset.')
parser.add_argument('--anthro_csv_path', type=str, default=None,
                    help='Defaults to ./HUTUBS/AnthropometricMeasures.csv or '
                         './SONICOM/AnthropometricMeasures.csv depending on --dataset.')
parser.add_argument('--image_dir', type=str, default=None,
                    help='Ear-crop image directory (SONICOM conditions C/D only). '
                         'Defaults to ./SONICOM/cropped.')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# NOTE: main_model/main.py forces --dataset hutubs to condition B here.
# This script deliberately does NOT do that — see module docstring.

# Fill in dataset-specific defaults for whichever path args weren't given explicitly.
DATASET_DEFAULTS = {
    'hutubs':  dict(hrtf_directory='./HUTUBS/HRIRs',
                    anthro_csv_path='./HUTUBS/AnthropometricMeasures.csv'),
    'sonicom': dict(hrtf_directory='./SONICOM/HRIRs',
                    anthro_csv_path='./SONICOM/AnthropometricMeasures.csv',
                    image_dir='./SONICOM/cropped'),
}
for key, default in DATASET_DEFAULTS[args.dataset].items():
    if getattr(args, key) is None:
        setattr(args, key, default)

BASE_MODEL_NAME = args.model_name or f'{args.dataset.upper()}_lossexp_{args.condition}'

loss_types_to_run = list(LOSS_TYPES) if args.loss_type == 'all' else [args.loss_type]
if args.loss_type == 'all' and (args.checkpoint_dir or args.results_dir or args.runs_dir):
    print("Note: --checkpoint_dir/--results_dir/--runs_dir are ignored when "
          "--loss_type all runs multiple losses in one invocation (each loss "
          "type gets its own auto-namespaced dir instead, so they don't "
          "overwrite each other).")

print(f"Base model name: {BASE_MODEL_NAME}  |  dataset={args.dataset}  |  "
      f"condition={args.condition}  |  loss_type(s)={loss_types_to_run}  |  "
      f"full_attention={args.full_attention}")

# ── Dataset (loaded ONCE — shared across every loss type in this run) ────────
print("Loading dataset...")
condition = CONDITIONS[args.condition]
if args.dataset == 'hutubs':
    dataset = HUTUBSDataset(
        hrtf_directory=args.hrtf_directory,
        anthro_csv_path=args.anthro_csv_path,
    )
else:
    dataset = SONICOMDataset(
        hrtf_directory=args.hrtf_directory,
        anthro_csv_path=args.anthro_csv_path,
        image_dir=args.image_dir if condition['use_image'] else None,
    )
print(f"Dataset size: {len(dataset)} samples  |  measurement points: {dataset.measurement_points}")

# ── Fold splits (computed ONCE, up front) ────────────────────────────────────
# get_kfold_splits is seeded (seed=42, see dataset.py) and depends only on
# the dataset's subject list, so calling it once here and reusing the same
# in-memory splits for every loss type guarantees L1/L2/combined all
# train/evaluate on IDENTICAL fold membership — required for a fair
# comparison. (Each loss type's checkpoint dir still gets its own
# splits.json written to disk below, purely so that dir stays
# self-contained/portable like a normal main_model/main.py run — not
# because it's re-derived independently.)
BASE_SPLITS = dataset.get_kfold_splits(k=args.k_folds)

if args.fold is not None:
    for f in args.fold:
        assert 1 <= f <= len(BASE_SPLITS), \
            f"--fold values must be 1–{len(BASE_SPLITS)}, got {f}"
    fold_indices = [f - 1 for f in args.fold]
else:
    fold_indices = list(range(len(BASE_SPLITS)))

# ── Diffusion model ───────────────────────────────────────────────────────────
diffusion_model = DiffusionModel()   # 600 timesteps
NUM_CLASSES = dataset.measurement_points   # 440 for HUTUBS, 793 for SONICOM

# ── Precision selection ───────────────────────────────────────────────────────
BASE_CHANNELS = 8
USE_FP16 = (BASE_CHANNELS != 16)
PRECISION_DTYPE = torch.float16 if USE_FP16 else torch.float32
print(f"Precision: {'FP16 (autocast)' if USE_FP16 else 'FP32 (large model — NaN-safe)'}")


def build_unet():
    unet = UNet(
        audio_channels=2,
        labels=NUM_CLASSES,
        head_dim=0,     # neither dataset has head/torso measurements
        ear_dim=condition['ear_dim'],
        image_dim=(IMAGE_FEAT_DIM if condition['use_image'] else 0),
        base_channels=BASE_CHANNELS,
        attn_full_encoder=args.full_attention,
    ).to(device)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  UNet parameters: {n_params:,}  |  base_channels={BASE_CHANNELS}  |  "
          f"condition={args.condition}  |  loss_type={CURRENT_LOSS_TYPE}  |  "
          f"full_attention={args.full_attention}")
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
    print(f"  TRAINING  FOLD {fold_idx + 1}/{args.k_folds}  |  loss_type={CURRENT_LOSS_TYPE}  |  "
          f"test subjects: {split['test_subjects']}")
    print(f"{'='*60}")

    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, fold_tag))

    writer.add_hparams(
        hparam_dict={
            'lr': args.lr,
            'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_plateau_patience': args.lr_plateau_patience,
            'lr_plateau_factor': args.lr_plateau_factor,
            'lr_min': args.lr_min,
            'batch_size': args.BATCH_SIZE,
            'early_stop_patience': args.early_stop_patience,
            'ema_decay': args.ema_decay,
            'loss_type': CURRENT_LOSS_TYPE,
            'loss_freq_weight': args.loss_freq_weight,
            'condition': args.condition,
            'full_attention': args.full_attention,
            'model_name': MODEL_NAME,
            'fold': fold_idx + 1,
        },
        metric_dict={'best_val_loss': float('inf')},
    )

    train_loader = DataLoader(
        Subset(dataset, split['train']),
        batch_size=args.BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        Subset(dataset, split['val']),
        batch_size=args.BATCH_SIZE, shuffle=False,
        num_workers=4, drop_last=False, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    unet = build_unet()
    if args.pretrained_checkpoint:
        load_matching_state_dict(
            unet,
            os.path.join(args.pretrained_checkpoint, f'unet_fold{fold_idx + 1}.pt'),
            reset_cond_fuse=args.reset_cond_fuse,
        )
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    ema = EMA(unet, decay=args.ema_decay)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.lr_warmup_epochs,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr=args.lr_min,
    )

    scaler = torch.cuda.amp.GradScaler(init_scale=1024, enabled=(device.type == 'cuda' and USE_FP16))

    best_val_loss    = float('inf')
    early_stop_count = 0
    model_path       = ckpt_path(fold_idx)
    plots_fold_dir   = os.path.join(PLOTS_DIR, fold_tag)
    os.makedirs(plots_fold_dir, exist_ok=True)

    for epoch in tqdm.tqdm(range(args.epochs), desc=f'{fold_tag}[{CURRENT_LOSS_TYPE}]', unit='epoch'):

        # ── Train ─────────────────────────────────────────────────────────────
        unet.train()
        train_losses = []
        train_comp_a = []
        train_comp_b = []
        for data in train_loader:
            batch = data['hrtf'].to(device, non_blocking=True).float()
            label = data['measurement_point'].to(device, non_blocking=True)
            ears  = data['ear_measurements'].to(device, non_blocking=True)
            images = data['image'].to(device, non_blocking=True) if 'image' in data else None

            t = torch.randint(0, diffusion_model.timesteps,
                              (batch.shape[0],), device=device).long()
            batch_noisy, noise = diffusion_model.forward(batch, t, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=PRECISION_DTYPE, enabled=USE_FP16):
                predicted_noise = unet(
                    batch_noisy.float(), t,
                    labels=label, ears_embedding=ears, images=images,
                )
                # Loss selection lives entirely in loss_exp/losses.py::compute_loss —
                # this is the one line that main_model/main.py's train_fold doesn't have.
                loss, comp_a, comp_b = compute_loss(
                    noise, predicted_noise, loss_type=CURRENT_LOSS_TYPE,
                    freq_weight=args.loss_freq_weight, return_components=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.5)
            did_step = not any(torch.isnan(p.grad).any()
                                for p in unet.parameters() if p.grad is not None)
            if did_step:
                scaler.step(optimizer)
            scaler.update()
            if did_step:
                ema.update(unet)
            if not torch.isnan(loss):
                train_losses.append(loss.item())
                train_comp_a.append(comp_a.item())
                train_comp_b.append(comp_b.item())

        # ── Validate (using EMA weights — swapped in for the duration) ──────────
        ema.apply_shadow(unet)
        unet.eval()
        val_losses = []
        last_noise = last_pred = None
        with torch.no_grad():
            for data in val_loader:
                batch = data['hrtf'].to(device, non_blocking=True).float()
                label = data['measurement_point'].to(device, non_blocking=True)
                ears  = data['ear_measurements'].to(device, non_blocking=True)
                images = data['image'].to(device, non_blocking=True) if 'image' in data else None

                t = torch.randint(0, diffusion_model.timesteps,
                                  (batch.shape[0],), device=device).long()
                batch_noisy, noise = diffusion_model.forward(batch, t, device)
                with torch.autocast(device_type=device.type, dtype=PRECISION_DTYPE, enabled=USE_FP16):
                    pred = unet(
                        batch_noisy.float(), t,
                        labels=label, ears_embedding=ears, images=images,
                    )
                val_losses.append(
                    compute_loss(noise, pred, loss_type=CURRENT_LOSS_TYPE,
                                 freq_weight=args.loss_freq_weight).item()
                )
                last_noise, last_pred = noise, pred
        ema.restore(unet)

        mean_train = np.mean(train_losses)
        mean_val   = np.mean(val_losses)

        if epoch < args.lr_warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(mean_val)
        current_lr = optimizer.param_groups[0]['lr']

        # ── TensorBoard scalars ───────────────────────────────────────────────
        # For loss_type='combined', comp_a/comp_b are l1_time/l1_freq (same
        # two scalars main_model/main.py logs). For 'l1'/'l2', comp_a equals
        # the loss itself and comp_b is always 0 — there's only one term to
        # log — see compute_loss()'s docstring in losses.py.
        writer.add_scalar('Loss/train',      mean_train, epoch)
        writer.add_scalar('Loss/val',        mean_val,   epoch)
        writer.add_scalar('Loss/train_component_a', np.mean(train_comp_a), epoch)
        writer.add_scalar('Loss/train_component_b', np.mean(train_comp_b), epoch)
        writer.add_scalar('LR/learning_rate', current_lr, epoch)
        writer.add_scalar('EarlyStopping/patience_counter', early_stop_count, epoch)

        total_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in unet.parameters() if p.grad is not None
        ) ** 0.5
        writer.add_scalar('Gradients/total_norm', total_grad_norm, epoch)

        if epoch % 50 == 0 and last_noise is not None:
            if not torch.isnan(last_pred).any():
                plot_path = os.path.join(plots_fold_dir, f'noise_ep{epoch:04d}.png')
                plot_noise_distribution(last_noise, last_pred, epoch, plot_path=plot_path)
            import torchvision.transforms.functional as tvf
            from PIL import Image
            img = Image.open(plot_path)
            img_tensor = tvf.to_tensor(img)
            writer.add_image(f'NoiseDist/epoch_{epoch:04d}', img_tensor, epoch)

        if args.verbose or epoch % 50 == 0:
            print(f"  [{CURRENT_LOSS_TYPE}] Epoch {epoch:4d} | Train {mean_train:.4f} | Val {mean_val:.4f} | "
                  f"LR {current_lr:.2e} | Patience {early_stop_count}/{args.early_stop_patience}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save({
                'epoch':      torch.tensor(epoch),
                'model_state_dict': unet.state_dict(),
                'ema_state_dict':   ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
                'val_loss':   torch.tensor(best_val_loss),
                'fold':       torch.tensor(fold_idx + 1),
                'loss_type':  CURRENT_LOSS_TYPE,
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

    writer.add_hparams(
        hparam_dict={
            'lr': args.lr, 'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_plateau_patience': args.lr_plateau_patience,
            'lr_plateau_factor': args.lr_plateau_factor,
            'lr_min': args.lr_min,
            'batch_size': args.BATCH_SIZE,
            'early_stop_patience': args.early_stop_patience,
            'ema_decay': args.ema_decay,
            'loss_type': CURRENT_LOSS_TYPE,
            'loss_freq_weight': args.loss_freq_weight,
            'condition': args.condition,
            'full_attention': args.full_attention,
            'model_name': MODEL_NAME,
            'fold': fold_idx + 1,
        },
        metric_dict={'best_val_loss': best_val_loss},
    )
    writer.close()
    print(f"  Fold {fold_idx + 1} [{CURRENT_LOSS_TYPE}] done — best val loss: {best_val_loss:.4f}")
    return best_val_loss


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
def infer_fold(fold_idx, split, subj_point_index):
    model_path = ckpt_path(fold_idx)
    if not os.path.exists(model_path):
        print(f"No checkpoint at {model_path} — skipping fold {fold_idx + 1}")
        return [], [], [], [], [], [], [], []

    fold_tag = f'fold_{fold_idx + 1}'
    print(f"\n{'='*60}")
    print(f"  INFERENCE  FOLD {fold_idx + 1}/{args.k_folds}  |  loss_type={CURRENT_LOSS_TYPE}  |  "
          f"test subjects: {split['test_subjects']}")
    print(f"{'='*60}")

    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, fold_tag))

    unet = build_unet()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if args.use_ema and 'ema_state_dict' in ckpt:
        full_state = unet.state_dict()
        n_ema = len(ckpt['ema_state_dict'])
        full_state.update(ckpt['ema_state_dict'])
        unet.load_state_dict(full_state)
        print(f"  Loaded EMA weights from {model_path} "
              f"({n_ema}/{len(full_state)} tensors were EMA-tracked; the remaining "
              f"{len(full_state) - n_ema} are frozen/non-trainable and kept their "
              f"freshly constructed values)")
    else:
        if args.use_ema:
            print(f"  Warning: --use_ema=True but checkpoint has no 'ema_state_dict' "
                  f"(older checkpoint?) — falling back to raw weights.")
        unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    if hasattr(torch, 'compile'):
        unet = torch.compile(unet)

    betas_gpu              = diffusion_model.betas.to(device)
    alphas_gpu             = diffusion_model.alphas.to(device)
    sqrt_recip_alphas_gpu  = torch.sqrt(1.0 / alphas_gpu)
    sqrt_one_minus_acp_gpu = torch.sqrt(1.0 - diffusion_model.alphas_cumprod.to(device))

    wav_dir   = os.path.join(RES_DIR, fold_tag)
    mat_fold  = os.path.join(MAT_DIR,  fold_tag)
    plot_fold = os.path.join(PLOTS_DIR, fold_tag)
    for d in [wav_dir, mat_fold, plot_fold]:
        os.makedirs(d, exist_ok=True)

    _empty_progress = {
        'done_subjects': [], 'subject_ids': [],
        'lsd_L': [], 'lsd_R': [], 'lsd_avg': [], 'itd': [], 'pbc': [],
        'nmse': [], 'nmse_subj': [],
    }
    progress_path = os.path.join(wav_dir, 'progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            for k, v in _empty_progress.items():
                progress.setdefault(k, v)
            print(f"  Resuming — {len(progress['done_subjects'])} subjects already done.")
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: progress.json corrupted, starting fold fresh.")
            progress = dict(_empty_progress)
    else:
        progress = dict(_empty_progress)

    done_set     = set(progress['done_subjects'])
    fold_subject_ids = progress['subject_ids']
    fold_lsd_L   = progress['lsd_L']
    fold_lsd_R   = progress['lsd_R']
    fold_lsd_avg = progress['lsd_avg']
    fold_itd     = progress['itd']
    fold_pbc     = progress['pbc']
    fold_nmse    = progress['nmse']
    fold_nmse_subj = progress['nmse_subj']
    INFER_BATCH = 64

    for subject_id in tqdm.tqdm(split['test_subjects'], desc=f'{fold_tag}[{CURRENT_LOSS_TYPE}] infer'):
        if subject_id in done_set:
            continue

        hrir_sub  = []
        hrir_tsub = []
        nmse_sub  = []

        data_ref = dataset[subj_point_index[(subject_id, 0)]]
        ears_1  = data_ref['ear_measurements'].to(device).float()
        images_1 = data_ref['image'].to(device).float() if 'image' in data_ref else None
        g_std  = data_ref['global_std']
        g_mean = data_ref['global_mean']

        gt_hrirs     = []
        valid_points = []
        for c in range(dataset.measurement_points):
            key = (subject_id, c)
            if key not in subj_point_index:
                continue
            gt_hrirs.append(dataset[subj_point_index[key]]['hrtf'])
            valid_points.append(c)

        n_points = len(valid_points)
        if n_points == 0:
            continue

        all_results = []
        torch.manual_seed(42)

        for start in range(0, n_points, INFER_BATCH):
            end = min(start + INFER_BATCH, n_points)
            b   = end - start
            pts = valid_points[start:end]

            x            = torch.randn(b, 2, 256, device=device)
            labels_batch = torch.tensor(pts, device=device)
            ears_batch   = ears_1.unsqueeze(0).expand(b, -1)

            images_batch = images_1.unsqueeze(0).expand(b, -1, -1, -1, -1) if images_1 is not None else None

            with torch.no_grad():
                for i in reversed(range(diffusion_model.timesteps)):
                    t_batch      = torch.full((b,), i, dtype=torch.long, device=device)
                    betas_t      = betas_gpu[i].view(1, 1, 1)
                    sqrt_recip_t = sqrt_recip_alphas_gpu[i].view(1, 1, 1)
                    sqrt_omacp_t = sqrt_one_minus_acp_gpu[i].view(1, 1, 1)

                    predicted_noise = unet(
                        x, t_batch,
                        labels=labels_batch,
                        ears_embedding=ears_batch,
                        images=images_batch,
                    )
                    mean = sqrt_recip_t * (x - betas_t * predicted_noise / sqrt_omacp_t)
                    x = mean + (torch.sqrt(betas_t) * torch.randn_like(x) if i > 0 else 0)

            all_results.append(x.cpu())

        results_tensor = torch.cat(all_results, dim=0)

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

            hrir_sub.append(audio_result)
            hrir_tsub.append(hrir_test)
            gen_hrirs_mat.append(audio_result.float().numpy())
            gt_hrirs_mat.append(hrir_test.float().numpy())

        if gen_hrirs_mat:
            n_valid      = len(hrir_sub)
            lsd_vals     = lsd(hrir_tsub, hrir_sub, n_valid, sr=44100)
            lsd_L_sub    = lsd_vals['L']
            lsd_R_sub    = lsd_vals['R']
            lsd_avg_sub  = lsd_vals['avg']
            itd_val_sub  = itd_error(hrir_tsub, hrir_sub, sr=44100)
            pbc_val_sub  = pbc(hrir_tsub, hrir_sub, sr=44100)
            sio.savemat(
                os.path.join(mat_fold, f'sub_{subject_id}.mat'),
                {
                    'hrir_gen':    np.array(gen_hrirs_mat,  dtype=np.float32),
                    'hrir_gt':     np.array(gt_hrirs_mat,   dtype=np.float32),
                    'nmse_values': np.array(nmse_sub,       dtype=np.float64),
                    'lsd_L':       np.float64(lsd_L_sub),
                    'lsd_R':       np.float64(lsd_R_sub),
                    'lsd_avg':     np.float64(lsd_avg_sub),
                    'itd_error':   np.float64(itd_val_sub),
                    'pbc_value':   np.float64(pbc_val_sub),
                    'subject_id':  np.int32(subject_id),
                    'positions':   np.array(valid_points,   dtype=np.int32),
                }
            )

        if hrir_sub:
            fold_subject_ids.append(int(subject_id))
            fold_lsd_L.append(lsd_L_sub)
            fold_lsd_R.append(lsd_R_sub)
            fold_lsd_avg.append(lsd_avg_sub)
            fold_itd.append(itd_val_sub)
            fold_pbc.append(pbc_val_sub)
            fold_nmse.extend(nmse_sub)
            fold_nmse_subj.append(float(np.mean(nmse_sub)))

            writer.add_scalar(f'Inference/LSD_L_sub_{subject_id}',  float(lsd_L_sub),         fold_idx + 1)
            writer.add_scalar(f'Inference/LSD_R_sub_{subject_id}',  float(lsd_R_sub),         fold_idx + 1)
            writer.add_scalar(f'Inference/LSD_avg_sub_{subject_id}',float(lsd_avg_sub),        fold_idx + 1)
            writer.add_scalar(f'Inference/ITD_sub_{subject_id}',    float(itd_val_sub),        fold_idx + 1)
            writer.add_scalar(f'Inference/PBC_sub_{subject_id}',    float(pbc_val_sub),        fold_idx + 1)
            writer.add_scalar(f'Inference/NMSE_sub_{subject_id}',   float(np.mean(nmse_sub)),  fold_idx + 1)

            print(f"  [{CURRENT_LOSS_TYPE}] Subject {subject_id}: "
                  f"LSD_L={lsd_L_sub:.3f}  LSD_R={lsd_R_sub:.3f}  LSD_avg={lsd_avg_sub:.3f} dB  "
                  f"ITD={itd_val_sub:.2f} µs  "
                  f"PBC={pbc_val_sub:.3f} dB  "
                  f"NMSE={np.mean(nmse_sub):.4f}")

        progress['done_subjects'].append(int(subject_id))
        progress['subject_ids'] = [int(v) for v in fold_subject_ids]
        progress['lsd_L']   = [float(v) for v in fold_lsd_L]
        progress['lsd_R']   = [float(v) for v in fold_lsd_R]
        progress['lsd_avg'] = [float(v) for v in fold_lsd_avg]
        progress['itd']     = [float(v) for v in fold_itd]
        progress['pbc']     = [float(v) for v in fold_pbc]
        progress['nmse']    = [float(v) for v in fold_nmse]
        progress['nmse_subj'] = [float(v) for v in fold_nmse_subj]
        tmp_path = progress_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(progress, f)
        os.replace(tmp_path, progress_path)

    if fold_lsd_avg:
        writer.add_scalar('Inference/mean_LSD_L',   float(np.mean(fold_lsd_L)),   fold_idx + 1)
        writer.add_scalar('Inference/mean_LSD_R',   float(np.mean(fold_lsd_R)),   fold_idx + 1)
        writer.add_scalar('Inference/mean_LSD_avg', float(np.mean(fold_lsd_avg)), fold_idx + 1)
        writer.add_scalar('Inference/mean_ITD',     float(np.mean(fold_itd)),     fold_idx + 1)
        writer.add_scalar('Inference/mean_PBC',     float(np.mean(fold_pbc)),     fold_idx + 1)
        writer.add_scalar('Inference/mean_NMSE',    float(np.mean(fold_nmse)),    fold_idx + 1)

        sio.savemat(
            os.path.join(MAT_DIR, f'fold_{fold_idx + 1}_summary.mat'),
            {
                'subject_id_per_subject': np.array(fold_subject_ids, dtype=np.int32),
                'lsd_L_per_subject':  np.array(fold_lsd_L,   dtype=np.float64),
                'lsd_R_per_subject':  np.array(fold_lsd_R,   dtype=np.float64),
                'lsd_avg_per_subject':np.array(fold_lsd_avg, dtype=np.float64),
                'itd_per_subject':    np.array(fold_itd,     dtype=np.float64),
                'pbc_per_subject':    np.array(fold_pbc,     dtype=np.float64),
                'nmse_per_subject':   np.array(fold_nmse_subj, dtype=np.float64),
                'nmse_per_position':  np.array(fold_nmse,    dtype=np.float64),
                'test_subjects':      np.array(split['test_subjects'], dtype=np.int32),
                'model':              MODEL_NAME,
                'loss_type':          CURRENT_LOSS_TYPE,
                'full_attention':     bool(args.full_attention),
                'mean_lsd_L':         float(np.mean(fold_lsd_L)),
                'mean_lsd_R':         float(np.mean(fold_lsd_R)),
                'mean_lsd_avg':       float(np.mean(fold_lsd_avg)),
                'mean_itd':           float(np.mean(fold_itd)),
                'mean_pbc':           float(np.mean(fold_pbc)),
                'mean_nmse':          float(np.mean(fold_nmse)),
            }
        )

    writer.flush()
    writer.close()
    print(f"  Fold {fold_idx + 1} [{CURRENT_LOSS_TYPE}] — "
          f"LSD_L={np.mean(fold_lsd_L):.3f}  "
          f"LSD_R={np.mean(fold_lsd_R):.3f}  "
          f"LSD_avg={np.mean(fold_lsd_avg):.3f} dB  "
          f"ITD={np.mean(fold_itd):.2f} µs  "
          f"PBC={np.mean(fold_pbc):.3f} dB  "
          f"NMSE={np.mean(fold_nmse):.4f}")
    return fold_subject_ids, fold_lsd_L, fold_lsd_R, fold_lsd_avg, fold_itd, fold_pbc, fold_nmse, fold_nmse_subj


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH — loops over every loss type requested (l1 / l2 / combined,
# or just one if --loss_type wasn't "all"), each under its own namespaced
# checkpoint/results/runs dir so they never overwrite each other. Comparison
# summary is printed + saved once every loss type has run.
# ═════════════════════════════════════════════════════════════════════════════
comparison = {}   # loss_type -> summary dict, used for the final comparison table

for CURRENT_LOSS_TYPE in loss_types_to_run:
    MODEL_NAME = f'{BASE_MODEL_NAME}_{CURRENT_LOSS_TYPE}'

    single_loss_run = (args.loss_type != 'all')
    CKPT_DIR  = args.checkpoint_dir if (single_loss_run and args.checkpoint_dir) else os.path.join('./checkpoints', MODEL_NAME)
    RES_DIR   = args.results_dir    if (single_loss_run and args.results_dir)    else os.path.join('./results',     MODEL_NAME)
    RUNS_DIR  = args.runs_dir       if (single_loss_run and args.runs_dir)       else os.path.join('./runs',        MODEL_NAME)
    MAT_DIR   = os.path.join(RES_DIR, 'mat')
    PLOTS_DIR = os.path.join(RES_DIR, 'plots')

    for d in [RUNS_DIR, CKPT_DIR, RES_DIR, MAT_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)
    for fi in range(5):   # max k_folds; harmless extras are ignored
        os.makedirs(os.path.join(RUNS_DIR, f'fold_{fi + 1}'), exist_ok=True)

    # splits.json is written into this loss type's own checkpoint dir so it
    # stays self-contained (matches main_model/main.py's convention) — but
    # the actual fold membership always comes from BASE_SPLITS computed once
    # above, never re-derived per loss type.
    SPLITS_PATH      = os.path.join(CKPT_DIR, 'splits.json')
    SPLITS_META_PATH = os.path.join(CKPT_DIR, 'splits_meta.json')
    if not os.path.exists(SPLITS_PATH) or not os.path.exists(SPLITS_META_PATH):
        splits_serialisable = [
            {k: [int(x) for x in v] for k, v in s.items()}
            for s in BASE_SPLITS
        ]
        with open(SPLITS_PATH, 'w') as f:
            json.dump(splits_serialisable, f, indent=2)
        with open(SPLITS_META_PATH, 'w') as f:
            json.dump({'k_folds': args.k_folds}, f)
    splits = BASE_SPLITS

    print(f"\n{'#'*70}\n#  LOSS TYPE: {CURRENT_LOSS_TYPE}  |  model_name={MODEL_NAME}\n{'#'*70}")

    if args.mode == 'train':
        fold_val_losses = []
        for fi in fold_indices:
            fold_val_losses.append(train_fold(fi, splits[fi]))

        comparison[CURRENT_LOSS_TYPE] = {
            'per_fold_val_loss': fold_val_losses,
            'mean_val_loss': float(np.mean(fold_val_losses)),
            'std_val_loss': float(np.std(fold_val_losses)),
        }
        if len(fold_val_losses) > 1:
            print(f"\n[{CURRENT_LOSS_TYPE}] Cross-validation summary:")
            print(f"  Per-fold val losses : {[f'{v:.4f}' for v in fold_val_losses]}")
            print(f"  Mean ± std          : "
                  f"{np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}")

    else:
        subj_point_index = build_subject_point_index(dataset)

        for fi in fold_indices:
            infer_fold(fi, splits[fi], subj_point_index)

        all_subject_ids, all_lsd_L, all_lsd_R, all_lsd_avg = [], [], [], []
        all_itd_vals, all_pbc_vals, all_nmse_vals, all_nmse_subj = [], [], [], []
        for fi in range(len(splits)):
            summary_path = os.path.join(MAT_DIR, f'fold_{fi + 1}_summary.mat')
            if not os.path.exists(summary_path):
                continue
            m = sio.loadmat(summary_path)
            all_subject_ids.extend(m['subject_id_per_subject'].ravel().tolist())
            all_lsd_L.extend(m['lsd_L_per_subject'].ravel().tolist())
            all_lsd_R.extend(m['lsd_R_per_subject'].ravel().tolist())
            all_lsd_avg.extend(m['lsd_avg_per_subject'].ravel().tolist())
            all_itd_vals.extend(m['itd_per_subject'].ravel().tolist())
            all_pbc_vals.extend(m['pbc_per_subject'].ravel().tolist())
            all_nmse_subj.extend(m['nmse_per_subject'].ravel().tolist())
            all_nmse_vals.extend(m['nmse_per_position'].ravel().tolist())

        if all_lsd_avg:
            print(f"\n[{CURRENT_LOSS_TYPE}] Overall LSD_L  : {np.mean(all_lsd_L):.3f} ± {np.std(all_lsd_L):.3f} dB")
            print(f"[{CURRENT_LOSS_TYPE}] Overall LSD_R  : {np.mean(all_lsd_R):.3f} ± {np.std(all_lsd_R):.3f} dB")
            print(f"[{CURRENT_LOSS_TYPE}] Overall LSD_avg: {np.mean(all_lsd_avg):.3f} ± {np.std(all_lsd_avg):.3f} dB")
            print(f"[{CURRENT_LOSS_TYPE}] Overall ITD    : {np.mean(all_itd_vals):.2f} ± {np.std(all_itd_vals):.2f} µs")
            print(f"[{CURRENT_LOSS_TYPE}] Overall PBC    : {np.mean(all_pbc_vals):.3f} ± {np.std(all_pbc_vals):.3f} dB")
            print(f"[{CURRENT_LOSS_TYPE}] Overall NMSE   : {np.mean(all_nmse_vals):.4f} ± {np.std(all_nmse_vals):.4f}")

            sio.savemat(
                os.path.join(MAT_DIR, 'all_folds_summary.mat'),
                {
                    'subject_id_all': np.array(all_subject_ids, dtype=np.int32),
                    'lsd_L_all':  np.array(all_lsd_L,    dtype=np.float64),
                    'lsd_R_all':  np.array(all_lsd_R,    dtype=np.float64),
                    'lsd_avg_all':np.array(all_lsd_avg,  dtype=np.float64),
                    'itd_all':    np.array(all_itd_vals,  dtype=np.float64),
                    'pbc_all':    np.array(all_pbc_vals,  dtype=np.float64),
                    'nmse_all':   np.array(all_nmse_vals, dtype=np.float64),
                    'model':      MODEL_NAME,
                    'loss_type':  CURRENT_LOSS_TYPE,
                    'full_attention': bool(args.full_attention),
                    'mean_lsd_L':  float(np.mean(all_lsd_L)),
                    'mean_lsd_R':  float(np.mean(all_lsd_R)),
                    'mean_lsd_avg':float(np.mean(all_lsd_avg)),
                    'mean_itd':    float(np.mean(all_itd_vals)),
                    'mean_pbc':    float(np.mean(all_pbc_vals)),
                    'mean_nmse':   float(np.mean(all_nmse_vals)),
                }
            )
            pd.DataFrame({
                'subject_id': all_subject_ids,
                'model':   MODEL_NAME,
                'loss_type': CURRENT_LOSS_TYPE,
                'lsd_L':   all_lsd_L,
                'lsd_R':   all_lsd_R,
                'lsd_avg': all_lsd_avg,
                'itd':     all_itd_vals,
                'pbc':     all_pbc_vals,
                'nmse':    all_nmse_subj,
            }).to_excel(os.path.join(RES_DIR, 'metrics_per_subject.xlsx'), index=False)

            pd.DataFrame({
                'nmse': all_nmse_vals,
            }).to_excel(os.path.join(RES_DIR, 'nmse_per_position.xlsx'), index=False)

            comparison[CURRENT_LOSS_TYPE] = {
                'lsd_avg_mean': float(np.mean(all_lsd_avg)), 'lsd_avg_std': float(np.std(all_lsd_avg)),
                'itd_mean':     float(np.mean(all_itd_vals)), 'itd_std':     float(np.std(all_itd_vals)),
                'pbc_mean':     float(np.mean(all_pbc_vals)), 'pbc_std':     float(np.std(all_pbc_vals)),
                'nmse_mean':    float(np.mean(all_nmse_vals)), 'nmse_std':   float(np.std(all_nmse_vals)),
                'n_subjects':   len(all_subject_ids),
            }


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-LOSS-TYPE COMPARISON — only printed/saved when more than one loss
# type actually ran in this invocation (i.e. --loss_type all).
# ═════════════════════════════════════════════════════════════════════════════
if len(comparison) > 1:
    print(f"\n{'='*70}\n  LOSS ABLATION COMPARISON — {BASE_MODEL_NAME}\n{'='*70}")
    rows = []
    for lt, stats in comparison.items():
        rows.append({'loss_type': lt, **stats})
        if args.mode == 'train':
            print(f"  {lt:>9s}  best_val_loss = {stats['mean_val_loss']:.4f} ± {stats['std_val_loss']:.4f}"
                  f"  (per fold: {[f'{v:.4f}' for v in stats['per_fold_val_loss']]})")
        else:
            print(f"  {lt:>9s}  LSD_avg = {stats['lsd_avg_mean']:.3f} ± {stats['lsd_avg_std']:.3f} dB  |  "
                  f"ITD = {stats['itd_mean']:.2f} ± {stats['itd_std']:.2f} µs  |  "
                  f"PBC = {stats['pbc_mean']:.3f} ± {stats['pbc_std']:.3f} dB  |  "
                  f"NMSE = {stats['nmse_mean']:.4f} ± {stats['nmse_std']:.4f}")

    comparison_dir = './results'
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_path = os.path.join(comparison_dir, f'{BASE_MODEL_NAME}_loss_comparison_{args.mode}.xlsx')
    pd.DataFrame(rows).to_excel(comparison_path, index=False)
    print(f"\nComparison table saved to {comparison_path}")
