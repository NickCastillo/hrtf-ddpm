"""
dataset.py — HUTUBS HRIR dataset loader with optional ear-mirroring augmentation.

Anthropometric features
-----------------------
HUTUBS provides 37 anthropometric features per subject (all used):
  Cols 1-13  : head/torso (CIPIC x1-x9, x12, x14, x16, x17)
  Cols 14-25 : left pinna  (L_d1-L_d10, L_theta1, L_theta2)
  Cols 26-37 : right pinna (R_d1-R_d10, R_theta1, R_theta2)

The paper (arxiv 2501.02871) references CIPIC's N=27 features (17 head + 10
pinna per ear), but HUTUBS provides a related but non-identical set. We use
all 37 available features rather than dropping valid measurements. This is
documented as a methodological note in the thesis.

Anthropometric normalisation
----------------------------
Features are normalised with a sigmoid transform rather than z-score:
  sigmoid((x - μ) / σ)
where μ and σ are the mean and std computed across ALL anthropometric values
(head + ear features jointly) from the training set only. This maps every
feature into (0, 1) regardless of original scale, which is beneficial for
HUTUBS where head/torso features (mm) and pinna angles (degrees) differ by
an order of magnitude. The single shared μ/σ scalar (not per-feature) matches
the original author's implementation and keeps the interface simple.

Ear-mirroring augmentation
--------------------------
For each real (subject, DOA) training item, a mirrored counterpart is appended:
  - HRIR channels swapped: [L, R] → [R, L]
  - DOA label replaced with azimuth-mirrored grid point (via pre-built LUT)
  - Anthropometrics unchanged (bilateral symmetry assumption)
Controlled by the `augment` flag. Only training datasets use augment=True.

K-fold cross-validation
-----------------------
Fold assignment is handled entirely in main.py. The dataset class receives
subject_ids directly and has no knowledge of fold structure.
"""

import os

import numpy as np
import pandas as pd
import torch
from pysofaconventions import SOFAFile
from torch.utils.data import Dataset


# Subjects with known data quality issues in HUTUBS (1-indexed, matching CSV).
EXCLUDED_SUBJECT_IDS = {18, 79, 92}

# Total number of SOFA files shipped with HUTUBS.
N_SOFA_FILES = 96

# Number of spatial measurement positions per subject.
N_POINTS = 440


# ---------------------------------------------------------------------------
# Mirror look-up table
# ---------------------------------------------------------------------------

def build_mirror_lut(source_positions):
    """
    Build a look-up table mapping each DOA index to its azimuth-mirrored index.

    For each point p with (azimuth, elevation):
      az_mirror = (360 - azimuth) % 360,  elevation unchanged.
      lut[p] = index of nearest grid point to (az_mirror, elevation).

    Parameters
    ----------
    source_positions : ndarray (N_POINTS, 3)
        Columns: [azimuth_deg, elevation_deg, radius_m]

    Returns
    -------
    lut : ndarray (N_POINTS,) int
    """
    azimuths   = source_positions[:, 0]
    elevations = source_positions[:, 1]
    lut        = np.zeros(N_POINTS, dtype=int)

    for p in range(N_POINTS):
        az_mirror = (360.0 - azimuths[p]) % 360.0
        el        = elevations[p]

        az_diff = np.abs(azimuths - az_mirror)
        el_diff = np.abs(elevations - el)
        az_diff = np.minimum(az_diff, 360.0 - az_diff)   # wrap to [0, 180]

        lut[p] = int(np.argmin(az_diff + el_diff))

    return lut


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HUTUBSDataset(Dataset):
    """
    HUTUBS binaural HRIR dataset.

    Parameters
    ----------
    hrtf_directory  : str
    anthro_csv_path : str
    subject_ids     : list[int] or None
        1-indexed subject IDs to include. None = all non-excluded subjects.
    augment         : bool
        If True, append azimuth-mirrored copies of every item (training only).
    norm_mean, norm_std : torch.Tensor or None
        Global HRIR normalisation stats. None = compute from this split.
    norm_anthro_mean, norm_anthro_std : torch.Tensor or None
        Shared scalar stats for sigmoid anthropometric normalisation.
        Computed jointly across all head + ear features from the training set.
        Pass the training-set values to val/test datasets to avoid data leakage.
        None = compute from this split (training dataset only).
    """

    def __init__(
        self,
        hrtf_directory,
        anthro_csv_path,
        subject_ids=None,
        augment=False,
        norm_mean=None,
        norm_std=None,
        norm_anthro_mean=None,
        norm_anthro_std=None,
    ):
        self.hrtf_directory     = hrtf_directory
        self.anthro_csv_path    = anthro_csv_path
        self.subject_ids_filter = set(subject_ids) if subject_ids is not None else None
        self.augment            = augment

        self._load_data(
            norm_mean, norm_std,
            norm_anthro_mean, norm_anthro_std,
        )

    def _load_data(
        self,
        norm_mean, norm_std,
        norm_anthro_mean, norm_anthro_std,
    ):
        # --- 1. Anthropometric CSV ---
        # All 37 HUTUBS features used (see module docstring).
        # Col 0 = SubjectID, cols 1-13 = head, cols 14-37 = ears (12L + 12R).
        af_csv = pd.read_csv(self.anthro_csv_path, header=0)
        subject_ids_csv = af_csv.iloc[:, 0].values.astype(int)

        head_measurements_raw = torch.from_numpy(
            af_csv.iloc[:, 1:14].values.astype(np.float32)
        )   # (N_subjects, 13)

        ear_measurements_raw = torch.from_numpy(
            af_csv.iloc[:, 14:].values.astype(np.float32)
        )   # (N_subjects, 24)  — cols 14-37

        head_measurements_raw[torch.isnan(head_measurements_raw)] = 0.0
        ear_measurements_raw[torch.isnan(ear_measurements_raw)]   = 0.0

        csv_row_by_subject = {int(sid): i for i, sid in enumerate(subject_ids_csv)}

        # --- 2. Load SOFA files ---
        source_positions = None
        sofa_files       = {}

        for n in range(N_SOFA_FILES):
            subject_id = n + 1
            if subject_id in EXCLUDED_SUBJECT_IDS:
                continue
            if (self.subject_ids_filter is not None
                    and subject_id not in self.subject_ids_filter):
                continue

            file_path = os.path.join(
                self.hrtf_directory, f'pp{subject_id}_HRIRs_measured.sofa'
            )
            sofa = SOFAFile(file_path, 'r')
            sofa_files[subject_id] = sofa

            if source_positions is None:
                source_positions = sofa.getVariableValue('SourcePosition')

        if source_positions is None:
            raise RuntimeError(
                'No valid SOFA files found — check hrtf_directory and subject_ids.'
            )

        # --- 3. Mirror LUT (same grid for all subjects; computed once) ---
        self.mirror_lut = build_mirror_lut(source_positions)

        # --- 4. Raw item list ---
        hrtf_points = []

        for subject_id, sofa_file in sofa_files.items():
            hrtf_data = sofa_file.getDataIR()   # (N_POINTS, 2, L)

            for point in range(N_POINTS):
                hrtf_point = hrtf_data[point, :, :].data   # (2, L)

                if np.isnan(hrtf_point).any():
                    print(f'NaN — subject {subject_id}, point {point}')
                    continue

                hrtf_points.append({
                    'hrtf':       hrtf_point,
                    'point':      point,
                    'subject_id': subject_id,
                    'head_row':   csv_row_by_subject[subject_id],
                })

        if not hrtf_points:
            raise RuntimeError('Dataset is empty after filtering.')

        # --- 5. Normalisation stats ---
        all_hrtf = torch.from_numpy(
            np.array([item['hrtf'] for item in hrtf_points], dtype=np.float32)
        )   # (N_items, 2, L)

        if norm_mean is None:     norm_mean     = all_hrtf.mean()
        if norm_std  is None:     norm_std      = all_hrtf.std()

        # Shared scalar stats for sigmoid normalisation — computed jointly
        # across all anthropometric values (head + ear) from this split.
        # A single μ/σ keeps the transform simple and matches the author's
        # implementation. NaN-zeroed values are excluded implicitly since
        # they were already zeroed above before stacking.
        if norm_anthro_mean is None or norm_anthro_std is None:
            all_anthro = torch.cat([
                head_measurements_raw,   # (N_csv_rows, 13)
                ear_measurements_raw,    # (N_csv_rows, 24)
            ], dim=1)                    # (N_csv_rows, 37)
            if norm_anthro_mean is None:
                norm_anthro_mean = all_anthro.mean()
            if norm_anthro_std  is None:
                norm_anthro_std  = all_anthro.std().clamp(min=1e-8)

        self.norm_mean        = norm_mean
        self.norm_std         = norm_std
        self.norm_anthro_mean = norm_anthro_mean
        self.norm_anthro_std  = norm_anthro_std

        # --- 6. Build normalised item list (+ optional mirrored copies) ---
        self.items = []

        for i, item in enumerate(hrtf_points):
            hrtf_tensor = all_hrtf[i]   # (2, L)
            row         = item['head_row']

            norm_item = {
                'hrtf':              (hrtf_tensor - norm_mean) / norm_std,
                'subject_id':        item['subject_id'],
                'measurement_point': item['point'],
                'head_measurements': torch.sigmoid(
                    (head_measurements_raw[row] - norm_anthro_mean) / norm_anthro_std
                ),
                'ear_measurements':  torch.sigmoid(
                    (ear_measurements_raw[row] - norm_anthro_mean) / norm_anthro_std
                ),
                'global_mean':       norm_mean,
                'global_std':        norm_std,
                'is_mirrored':       False,
            }
            self.items.append(norm_item)

            if self.augment:
                # Azimuth-mirrored counterpart:
                #   - swap HRIR channels [L, R] → [R, L]
                #   - replace DOA label with mirror LUT entry
                #   - head anthropometrics unchanged (bilateral symmetry)
                #   - ear anthropometrics: swap L/R halves to match channel swap.
                #     Layout: [L_d1..L_d10, L_theta1, L_theta2,   (indices 0-11)
                #              R_d1..R_d10, R_theta1, R_theta2]   (indices 12-23)
                #     When HRIR channels swap, model channel 0 = right ear, so
                #     right pinna features must move to indices 0-11. Not swapping
                #     this causes right-ear HRIR to be paired with left pinna
                #     features, producing a systematic L/R LSD asymmetry.
                ear_raw      = ear_measurements_raw[row]                        # (24,)
                ear_mirrored = torch.cat([ear_raw[12:], ear_raw[:12]], dim=0)   # swap L↔R

                mirrored_item = {
                    'hrtf':              (hrtf_tensor[[1, 0], :] - norm_mean) / norm_std,
                    'subject_id':        item['subject_id'],
                    'measurement_point': int(self.mirror_lut[item['point']]),
                    'head_measurements': torch.sigmoid(
                        (head_measurements_raw[row] - norm_anthro_mean) / norm_anthro_std
                    ),
                    'ear_measurements':  torch.sigmoid(
                        (ear_mirrored - norm_anthro_mean) / norm_anthro_std
                    ),
                    'global_mean':       norm_mean,
                    'global_std':        norm_std,
                    'is_mirrored':       True,
                }
                self.items.append(mirrored_item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Stack a list of per-item dicts into batched tensors."""
    return {
        'hrtf':              torch.stack([item['hrtf']              for item in batch]),
        'measurement_point': torch.tensor([item['measurement_point'] for item in batch]),
        'subject_id':        torch.LongTensor([item['subject_id']    for item in batch]),
        'head_measurements': torch.stack([item['head_measurements'] for item in batch]),
        'ear_measurements':  torch.stack([item['ear_measurements']  for item in batch]),
        'global_mean':       batch[0]['global_mean'],
        'global_std':        batch[0]['global_std'],
    }
