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
    that share the pre-computed stats from HUTUBSDataset; this is the cleanest
    way to feed separate DataLoaders without reloading the files.
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


class HUTUBSDataset:
    """
    Loads all HUTUBS subjects, builds a global normalisation from training
    subjects only, and exposes .train_data / .val_data lists of dicts ready
    to be wrapped by HUTUBSTrainDataset / HUTUBSValDataset.

    Parameters
    ----------
    hrtf_directory   : path to folder containing pp{n}_HRIRs_measured.sofa
    anthro_csv_path  : path to AntrhopometricMeasures.csv
    val_sub_idx      : 0-based subject index to hold out (0..95, excl. EXCLUDED)
    pad_size         : reflect-padding samples added to each HRIR
    """

    def __init__(self, hrtf_directory, anthro_csv_path, val_sub_idx, pad_size=10):
        if val_sub_idx in EXCLUDED_SUBJECTS:
            raise ValueError(
                f"val_sub_idx={val_sub_idx} is in the excluded set {EXCLUDED_SUBJECTS}."
            )
        self.hrtf_directory = hrtf_directory
        self.anthro_csv_path = anthro_csv_path
        self.val_sub_idx = val_sub_idx
        self.pad_size = pad_size
        self._load()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _pad(self, audio):
        return F.pad(audio, (self.pad_size, self.pad_size), mode='reflect')

    def _load(self):
        af_csv = pd.read_csv(self.anthro_csv_path, header=0)

        # Build a mapping: 0-based subject index → row in the CSV by subject ID.
        # The CSV first column is the subject ID (1-based in HUTUBS).
        # We index by position in the file list (0-based) to stay consistent
        # with how the SOFA files are named (pp1 … pp96 → indices 0 … 95).
        # We keep a dict {sofa_idx: csv_row_index} for the non-excluded subjects.
        csv_subject_ids = af_csv.iloc[:, 0].values          # shape (96,) or similar
        sofa_idx_to_csv_row = {}
        for csv_row, subj_id in enumerate(csv_subject_ids):
            sofa_idx = int(subj_id) - 1                     # HUTUBS is 1-based
            sofa_idx_to_csv_row[sofa_idx] = csv_row

        # Columns: head measurements 1:14, ear measurements 14:26 (original
        # code used 26: for ears in one place and 14:26 in another – we
        # consistently use 1:14 for head, 14:26 for ears).
        head_cols = slice(1, 14)
        ear_cols  = slice(14, 26)

        # Anthropometric stats will be computed over training subjects only.
        anthro_train_rows = []
        train_hrtf_tensors = []    # used to compute norm stats (train only)
        train_items = []
        val_items   = []

        for sofa_idx in range(96):                          # pp1..pp96 → 0..95
            if sofa_idx in EXCLUDED_SUBJECTS:
                continue

            file_path = os.path.join(
                self.hrtf_directory, f'pp{sofa_idx + 1}_HRIRs_measured.sofa'
            )
            sofa = SOFAFile(file_path, 'r')
            source_positions = sofa.getVariableValue('SourcePosition')
            hrtf_data = sofa.getDataIR()

            csv_row = sofa_idx_to_csv_row[sofa_idx]
            head_meas = torch.from_numpy(
                af_csv.iloc[csv_row, head_cols].values.astype(np.float64)
            )
            ear_meas  = torch.from_numpy(
                af_csv.iloc[csv_row, ear_cols].values.astype(np.float64)
            )

            is_val = (sofa_idx == self.val_sub_idx)

            for point in range(440):
                p         = source_positions[point]
                azimuth   = float(p[0])
                elevation = float(p[1])

                hrtf_point = torch.from_numpy(hrtf_data[point, :, :].data)
                hrtf_point = self._pad(hrtf_point)           # (2, L+2*pad)

                if torch.isnan(hrtf_point).any():
                    print(f"NaN detected at subject {sofa_idx}, point {point} – skipped.")
                    continue

                item = {
                    'hrtf':             hrtf_point,
                    'point':            point,
                    'azimuth':          azimuth,
                    'elevation':        elevation,
                    'subject_id':       sofa_idx + 1,        # 1-based, for reference
                    'head_measurements': head_meas,
                    'ear_measurements':  ear_meas,
                }

                if is_val:
                    val_items.append(item)
                else:
                    train_items.append(item)
                    train_hrtf_tensors.append(hrtf_point)
                    if len(anthro_train_rows) == 0 or anthro_train_rows[-1] != csv_row:
                        anthro_train_rows.append(csv_row)

        # ---- normalisation stats from TRAINING data only ----
        all_train_hrtf = torch.stack(train_hrtf_tensors)    # (N_train, 2, L)
        self.global_mean = torch.mean(all_train_hrtf)
        self.global_std  = torch.std(all_train_hrtf)

        # Anthropometric stats from training subjects only
        anthro_train = torch.from_numpy(
            af_csv.iloc[anthro_train_rows, 1:].values.astype(np.float64)
        )
        self.anthro_mean = torch.mean(anthro_train.float())
        self.anthro_std  = torch.std(anthro_train.float())

        # ---- normalise and store ----
        self.train_data = self._normalise(train_items)
        self.val_data   = self._normalise(val_items)

    def _sigmoid_norm_anthro(self, meas):
        """Sigmoid-normalise a measurement tensor using training anthro stats."""
        return torch.reciprocal(
            1 + torch.exp(-(meas.float() - self.anthro_mean) / self.anthro_std)
        )

    def _normalise(self, items):
        normalised = []
        for item in items:
            normalised.append({
                'hrtf': (item['hrtf'].float() - self.global_mean) / self.global_std,
                'point':            item['point'],
                'azimuth':          item['azimuth'],
                'elevation':        item['elevation'],
                'subject_id':       item['subject_id'],
                'head_measurements': self._sigmoid_norm_anthro(item['head_measurements']),
                'ear_measurements':  self._sigmoid_norm_anthro(item['ear_measurements']),
                'global_mean':      self.global_mean,
                'global_std':       self.global_std,
            })
        return normalised


# ---------------------------------------------------------------------------
# Lightweight Dataset wrappers so each split has its own DataLoader
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
# collate_fn  (unchanged from original, kept here for co-location)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        'hrtf':              torch.stack([item['hrtf'] for item in batch]),
        'point':             torch.tensor([item['point'] for item in batch]),
        'azimuth':           torch.tensor([item['azimuth'] for item in batch]),
        'elevation':         torch.tensor([item['elevation'] for item in batch]),
        'subject_id':        torch.LongTensor([item['subject_id'] for item in batch]),
        'head_measurements': torch.stack([item['head_measurements'] for item in batch]),
        'ear_measurements':  torch.stack([item['ear_measurements'] for item in batch]),
        'global_mean':       batch[0]['global_mean'],
        'global_std':        batch[0]['global_std'],
    }
