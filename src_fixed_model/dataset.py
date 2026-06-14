"""
dataset.py  –  HUTUBSDataset
Fixes applied vs original:
  - Removed module-level dataset instantiation (crashed on import in Colab).
  - __len__ now returns a single int (was returning a tuple, breaking DataLoader).
  - __getitem__ with integer index now works correctly for DataLoader iteration.
  - Normalisation statistics computed from TRAINING subjects only (no val-subject
    data leak).
  - CSV row exclusion now done by subject-ID value, not by row position.
  - subj_2 counter replaced by a clean subject_id → row lookup dict so
    anthropometric measurements are always aligned to the right subject.
  - HUTUBSTrainDataset / HUTUBSValDataset are two lightweight wrapper classes
    that share the pre-computed stats from HUTUBSDataset.

Binaural → Left-ear-only + mirrored-right augmentation:
  - The model is now trained on LEFT ears only.
  - Right ears are horizontally mirrored in azimuth and appended to the training
    set as additional "left-ear" samples, effectively doubling the data.
  - hrtf / hrtf_l / hrtf_r keys are retained in items, but the Dataset wrappers
    expose a single 'hrtf_mono' key of shape (1, L) which contains either a
    genuine left-ear HRIR or a mirrored right-ear HRIR.
  - Mirrored items are flagged with 'mirrored': True for debugging.

5-Fold CV:
  - ALL_SUBJECTS (93 valid subjects) are split into 5 roughly equal folds by
    build_5fold_splits().
  - HUTUBSDataset now accepts a val_subject_list (list of 0-based sofa indices)
    instead of a single val_sub_idx, removing the single-subject LOOCV constraint.
  - HUTUBSTrainDataset / HUTUBSValDataset wrappers are unchanged.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from pysofaconventions import SOFAFile
import numpy as np
import torch.nn.functional as F


# Subjects whose SOFA files are known-bad; excluded from every split.
EXCLUDED_SUBJECTS = {17, 78, 91}   # 0-based indices (pp18, pp79, pp92)

ALL_SUBJECTS = [i for i in range(96) if i not in EXCLUDED_SUBJECTS]   # 93 subjects


def build_5fold_splits(subjects=None, n_folds=5, seed=42):
    """
    Return a list of n_folds dicts, each with keys 'train' and 'val'
    containing lists of 0-based sofa subject indices.

    Parameters
    ----------
    subjects : list[int] | None
        Subjects to split.  Defaults to ALL_SUBJECTS (93 valid subjects).
    n_folds  : int
        Number of folds (default 5).
    seed     : int
        Random seed for reproducible shuffling.

    Returns
    -------
    list[dict]  length == n_folds, each {'train': [...], 'val': [...]}
    """
    if subjects is None:
        subjects = ALL_SUBJECTS
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()
    fold_sizes = [len(shuffled) // n_folds + (1 if i < len(shuffled) % n_folds else 0)
                  for i in range(n_folds)]
    splits, start = [], 0
    for size in fold_sizes:
        val   = shuffled[start:start + size]
        train = shuffled[:start] + shuffled[start + size:]
        splits.append({'train': train, 'val': val})
        start += size
    return splits


def _mirror_item(item):
    """
    Convert a right-ear sample into a synthetic left-ear sample.

    Horizontal mirroring means:
      azimuth   → -azimuth  (mod 360 kept in [0, 360) for consistency)
      elevation → unchanged
      hrir      → take the right-ear channel (index 1) as the new mono signal

    All keys expected by collate_fn are included so that mirrored items can
    be batched alongside genuine items without KeyError.  The 'hrtf' and
    'hrtf_l' / 'hrtf_r' fields are re-derived from the mirrored mono signal
    for collation consistency (hrtf_mono is the canonical training signal).
    """
    az_mirrored  = (-item['azimuth']) % 360.0
    mono         = item['hrtf'][1:2, :]   # right-ear channel treated as left (1, L)
    # Build a dummy binaural tensor so collate_fn can stack 'hrtf' uniformly.
    # Both channels are set to the mirrored mono signal — it is only used for
    # the collate stack and is not consumed during training (hrtf_mono is used).
    dummy_biaural = torch.cat([mono, mono], dim=0)    # (2, L)
    return {
        'hrtf':              dummy_biaural,   # (2, L) — kept for collate_fn compat
        'hrtf_l':            mono,            # (1, L)
        'hrtf_r':            mono,            # (1, L) — same as hrtf_l for mirrored
        'hrtf_mono':         mono,            # (1, L) — the actual training signal
        'point':             item['point'],
        'azimuth':           az_mirrored,
        'elevation':         item['elevation'],
        'subject_id':        item['subject_id'],
        'head_measurements': item['head_measurements'],
        'ear_measurements':  item['ear_measurements'],
        'global_mean':       item['global_mean'],
        'global_std':        item['global_std'],
        'mirrored':          True,
    }


class HUTUBSDataset:
    """
    Loads all HUTUBS subjects, builds global normalisation from training
    subjects only, and exposes .train_data / .val_data lists of dicts.

    Training items contain both the genuine left-ear sample AND a mirrored
    right-ear sample, doubling the effective training set size.

    Each item exposes:
        'hrtf_mono' : (1, L+2*pad)  single-channel HRIR used for training
        'hrtf'      : (2, L+2*pad)  full binaural HRIR (for reference / val)
        'hrtf_l'    : (1, L+2*pad)  left  ear (for reference / val)
        'hrtf_r'    : (1, L+2*pad)  right ear (for reference / val)
        ... plus point, azimuth, elevation, subject_id,
            head_measurements, ear_measurements, global_mean, global_std,
            mirrored (bool, True only for augmented right-ear copies)

    Parameters
    ----------
    hrtf_directory   : path to folder containing pp{n}_HRIRs_measured.sofa
    anthro_csv_path  : path to AntrhopometricMeasures.csv
    val_subject_list : list[int]  0-based sofa indices to hold out (the val fold)
    pad_size         : reflect-padding samples added to each HRIR
    """

    def __init__(self, hrtf_directory, anthro_csv_path, val_subject_list, pad_size=10):
        bad = set(val_subject_list) & EXCLUDED_SUBJECTS
        if bad:
            raise ValueError(
                f"val_subject_list contains excluded subjects: {bad}"
            )
        self.hrtf_directory  = hrtf_directory
        self.anthro_csv_path = anthro_csv_path
        self.val_subject_set = set(val_subject_list)
        self.pad_size        = pad_size
        self._load()

    def _pad(self, audio):
        # audio: (2, L)  →  (2, L + 2*pad_size)
        return F.pad(audio, (self.pad_size, self.pad_size), mode='reflect')

    def _load(self):
        af_csv = pd.read_csv(self.anthro_csv_path, header=0)

        # Map 0-based sofa_idx → row index in the CSV (CSV is 1-based subject IDs)
        csv_subject_ids     = af_csv.iloc[:, 0].values
        sofa_idx_to_csv_row = {int(sid) - 1: row
                               for row, sid in enumerate(csv_subject_ids)}

        head_cols = slice(1, 14)
        ear_cols  = slice(14, 26)

        anthro_train_rows  = []
        train_hrtf_tensors = []
        train_items_raw    = []   # genuine items before augmentation
        val_items          = []

        for sofa_idx in range(96):
            if sofa_idx in EXCLUDED_SUBJECTS:
                continue

            file_path = os.path.join(
                self.hrtf_directory, f'pp{sofa_idx + 1}_HRIRs_measured.sofa'
            )
            sofa             = SOFAFile(file_path, 'r')
            source_positions = sofa.getVariableValue('SourcePosition')
            hrtf_data        = sofa.getDataIR()   # (N_points, 2, L)

            csv_row   = sofa_idx_to_csv_row[sofa_idx]
            head_meas = torch.from_numpy(
                af_csv.iloc[csv_row, head_cols].values.astype(np.float64)
            )
            ear_meas  = torch.from_numpy(
                af_csv.iloc[csv_row, ear_cols].values.astype(np.float64)
            )

            is_val = (sofa_idx in self.val_subject_set)

            for point in range(440):
                p         = source_positions[point]
                azimuth   = float(p[0])
                elevation = float(p[1])

                # Keep BOTH ears: shape (2, L)
                hrtf_point = torch.from_numpy(hrtf_data[point, :, :].data)  # (2, L)
                hrtf_point = self._pad(hrtf_point)                           # (2, L+2*pad)

                if torch.isnan(hrtf_point).any():
                    print(f"NaN at subject {sofa_idx}, point {point} – skipped.")
                    continue

                item = {
                    'hrtf':              hrtf_point,           # (2, L+2*pad)  binaural
                    'hrtf_l':            hrtf_point[0:1, :],   # (1, L+2*pad)  left
                    'hrtf_r':            hrtf_point[1:2, :],   # (1, L+2*pad)  right
                    'hrtf_mono':         hrtf_point[0:1, :],   # (1, L+2*pad)  left (default)
                    'point':             point,
                    'azimuth':           azimuth,
                    'elevation':         elevation,
                    'subject_id':        sofa_idx + 1,
                    'head_measurements': head_meas,
                    'ear_measurements':  ear_meas,
                    'mirrored':          False,
                }

                if is_val:
                    val_items.append(item)
                else:
                    train_items_raw.append(item)
                    train_hrtf_tensors.append(hrtf_point)
                    if not anthro_train_rows or anthro_train_rows[-1] != csv_row:
                        anthro_train_rows.append(csv_row)

        # Normalisation stats from TRAINING data only (both channels together)
        all_train_hrtf   = torch.stack(train_hrtf_tensors)   # (N_train, 2, L)
        self.global_mean = torch.mean(all_train_hrtf)
        self.global_std  = torch.std(all_train_hrtf)

        # Anthropometric stats from training subjects only
        anthro_train     = torch.from_numpy(
            af_csv.iloc[anthro_train_rows, 1:].values.astype(np.float64)
        )
        self.anthro_mean = torch.mean(anthro_train.float())
        self.anthro_std  = torch.std(anthro_train.float())

        # Normalise, then augment training set with mirrored right ears
        normalised_train = self._normalise(train_items_raw)
        mirrored_train   = [_mirror_item(item) for item in normalised_train]

        # Train = genuine left ears + mirrored right ears (2× size)
        self.train_data  = normalised_train + mirrored_train
        self.val_data    = self._normalise(val_items)

    def _sigmoid_norm_anthro(self, meas):
        return torch.reciprocal(
            1 + torch.exp(-(meas.float() - self.anthro_mean) / self.anthro_std)
        )

    def _normalise(self, items):
        normalised = []
        for item in items:
            hrtf_norm = (item['hrtf'].float() - self.global_mean) / self.global_std
            normalised.append({
                'hrtf':              hrtf_norm,               # (2, L)  binaural normalised
                'hrtf_l':            hrtf_norm[0:1, :],       # (1, L)  left  normalised
                'hrtf_r':            hrtf_norm[1:2, :],       # (1, L)  right normalised
                'hrtf_mono':         hrtf_norm[0:1, :],       # (1, L)  left (default mono)
                'point':             item['point'],
                'azimuth':           item['azimuth'],
                'elevation':         item['elevation'],
                'subject_id':        item['subject_id'],
                'head_measurements': self._sigmoid_norm_anthro(item['head_measurements']),
                'ear_measurements':  self._sigmoid_norm_anthro(item['ear_measurements']),
                'global_mean':       self.global_mean,
                'global_std':        self.global_std,
                'mirrored':          item.get('mirrored', False),
            })
        return normalised


# ---------------------------------------------------------------------------
# Lightweight Dataset wrappers
# ---------------------------------------------------------------------------

class HUTUBSTrainDataset(Dataset):
    def __init__(self, hutubs_dataset: HUTUBSDataset):
        self.data = hutubs_dataset.train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class HUTUBSValDataset(Dataset):
    def __init__(self, hutubs_dataset: HUTUBSDataset):
        self.data = hutubs_dataset.val_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        'hrtf':              torch.stack([item['hrtf']      for item in batch]),  # (B, 2, L)
        'hrtf_l':            torch.stack([item['hrtf_l']    for item in batch]),  # (B, 1, L)
        'hrtf_r':            torch.stack([item['hrtf_r']    for item in batch]),  # (B, 1, L)
        'hrtf_mono':         torch.stack([item['hrtf_mono'] for item in batch]),  # (B, 1, L)
        'point':             torch.tensor([item['point']     for item in batch]),
        'azimuth':           torch.tensor([item['azimuth']   for item in batch]),
        'elevation':         torch.tensor([item['elevation'] for item in batch]),
        'subject_id':        torch.LongTensor([item['subject_id'] for item in batch]),
        'head_measurements': torch.stack([item['head_measurements'] for item in batch]),
        'ear_measurements':  torch.stack([item['ear_measurements']  for item in batch]),
        'global_mean':       batch[0]['global_mean'],
        'global_std':        batch[0]['global_std'],
        'mirrored':          torch.tensor([item.get('mirrored', False) for item in batch]),
    }
