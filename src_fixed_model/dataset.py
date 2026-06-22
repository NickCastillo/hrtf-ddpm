"""
dataset.py — HUTUBS HRIR dataset loader with optional ear-mirroring augmentation.

Ear-mirroring augmentation
--------------------------
For each real (subject, DOA) item in the training set, a mirrored counterpart
is appended:
  - HRIR channels are swapped: [L, R] → [R, L]
  - DOA label is replaced with the azimuth-mirrored point index:
      az_mirror = (360 - az) % 360, elevation unchanged
      The nearest grid point to (az_mirror, el) is found via a pre-built LUT.
  - Anthropometric measurements are unchanged (bilateral symmetry assumption).

This is acoustically valid: a sound at azimuth θ heard by the original subject
is equivalent to the same sound at azimuth (360-θ) heard by a laterally mirrored
version of that subject — and since we assume bilateral symmetry, that mirrored
subject has the same measurements. The augmentation doubles the effective training
set size and encourages the model to learn the L/R symmetry relationship.

Augmentation is controlled by the `augment` flag (default False). Only training
datasets should be constructed with augment=True. Val and test datasets must use
augment=False so evaluation is always on real, unmodified HRIRs.

K-fold cross-validation
-----------------------
Fold assignment is handled entirely in main.py, which passes the appropriate
subject_ids list to each HUTUBSDataset instance. The dataset class itself has
no knowledge of fold structure.

Changes vs. previous version:
  [NEW] build_mirror_lut(): builds the 440-element DOA mirror look-up table
        from the SOFA source position grid.
  [NEW] augment parameter: when True, appends mirrored items after real items.
  [NEW] is_mirrored flag stored per-item for downstream analysis.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from pysofaconventions import SOFAFile
import numpy as np


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
      az_mirror = (360 - azimuth) % 360
      Find the grid point q closest to (az_mirror, elevation).
      lut[p] = q

    Parameters
    ----------
    source_positions : ndarray (N_POINTS, 3)
        Columns: [azimuth_deg, elevation_deg, radius_m]

    Returns
    -------
    lut : ndarray (N_POINTS,) int
        lut[p] is the DOA index of the azimuth-mirrored counterpart of p.
    """
    azimuths   = source_positions[:, 0]  # degrees, 0-360
    elevations = source_positions[:, 1]  # degrees

    lut = np.zeros(N_POINTS, dtype=int)

    for p in range(N_POINTS):
        az_mirror = (360.0 - azimuths[p]) % 360.0
        el        = elevations[p]

        # Angular distance on the sphere to each candidate point.
        # We match on azimuth AND elevation so that only the azimuth is reflected.
        az_diff = np.abs(azimuths   - az_mirror)
        el_diff = np.abs(elevations - el)

        # Wrap azimuth difference to [0, 180].
        az_diff = np.minimum(az_diff, 360.0 - az_diff)

        # Combined angular distance (equal weight, both in degrees).
        dist = az_diff + el_diff
        lut[p] = int(np.argmin(dist))

    return lut


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HUTUBSDataset(Dataset):
    """
    HUTUBS binaural HRIR dataset.

    Parameters
    ----------
    hrtf_directory : str
    anthro_csv_path : str
    subject_ids : list[int] or None
        1-indexed subject IDs to include. None = all non-excluded subjects.
    augment : bool
        If True, append azimuth-mirrored copies of every item (training only).
    norm_mean, norm_std : torch.Tensor or None
        Global HRIR normalisation stats. None = compute from this split.
    norm_head_mean, norm_head_std : torch.Tensor or None
    norm_ears_mean, norm_ears_std : torch.Tensor or None
    """

    def __init__(
        self,
        hrtf_directory,
        anthro_csv_path,
        subject_ids=None,
        augment=False,
        norm_mean=None,
        norm_std=None,
        norm_head_mean=None,
        norm_head_std=None,
        norm_ears_mean=None,
        norm_ears_std=None,
    ):
        self.hrtf_directory      = hrtf_directory
        self.anthro_csv_path     = anthro_csv_path
        self.subject_ids_filter  = set(subject_ids) if subject_ids is not None else None
        self.augment             = augment

        self._load_data(
            norm_mean, norm_std,
            norm_head_mean, norm_head_std,
            norm_ears_mean, norm_ears_std,
        )

    def _load_data(
        self,
        norm_mean, norm_std,
        norm_head_mean, norm_head_std,
        norm_ears_mean, norm_ears_std,
    ):
        # --- 1. Anthropometric CSV ---
        af_csv = pd.read_csv(self.anthro_csv_path, header=0)
        subject_ids_csv       = af_csv.iloc[:, 0].values.astype(int)
        head_measurements_raw = torch.from_numpy(
            af_csv.iloc[:, 1:14].values.astype(np.float32)
        )
        ear_measurements_raw  = torch.from_numpy(
            af_csv.iloc[:, 14:].values.astype(np.float32)
        )
        head_measurements_raw[torch.isnan(head_measurements_raw)] = 0.0
        ear_measurements_raw[torch.isnan(ear_measurements_raw)]   = 0.0

        csv_row_by_subject = {int(sid): i for i, sid in enumerate(subject_ids_csv)}

        # --- 2. Load SOFA files ---
        source_positions = None
        sofa_files = {}

        for n in range(N_SOFA_FILES):
            subject_id = n + 1
            if subject_id in EXCLUDED_SUBJECT_IDS:
                continue
            if self.subject_ids_filter is not None and subject_id not in self.subject_ids_filter:
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
                "No valid SOFA files found — check hrtf_directory and subject_ids."
            )

        # --- 3. Build mirror LUT from the SOFA grid ---
        # This is the same for all subjects; computed once here.
        self.mirror_lut = build_mirror_lut(source_positions)

        # --- 4. Raw item list ---
        hrtf_points = []

        for subject_id, sofa_file in sofa_files.items():
            hrtf_data = sofa_file.getDataIR()  # (N_POINTS, 2, L)

            for point in range(N_POINTS):
                hrtf_point = hrtf_data[point, :, :].data  # (2, L)

                if np.isnan(hrtf_point).any():
                    print(f"NaN detected — subject {subject_id}, point {point}")
                    continue

                hrtf_points.append({
                    'hrtf':       hrtf_point,
                    'point':      point,
                    'subject_id': subject_id,
                    'head_row':   csv_row_by_subject[subject_id],
                })

        if len(hrtf_points) == 0:
            raise RuntimeError("Dataset is empty after filtering.")

        # --- 5. Normalisation stats ---
        all_hrtf = torch.from_numpy(
            np.array([item['hrtf'] for item in hrtf_points], dtype=np.float32)
        )  # (N_items, 2, L)

        if norm_mean is None:
            norm_mean = all_hrtf.mean()
        if norm_std is None:
            norm_std = all_hrtf.std()
        if norm_head_mean is None:
            norm_head_mean = head_measurements_raw.mean()
        if norm_head_std is None:
            norm_head_std = head_measurements_raw.std()
        if norm_ears_mean is None:
            norm_ears_mean = ear_measurements_raw.mean()
        if norm_ears_std is None:
            norm_ears_std = ear_measurements_raw.std()

        self.norm_mean      = norm_mean
        self.norm_std       = norm_std
        self.norm_head_mean = norm_head_mean
        self.norm_head_std  = norm_head_std
        self.norm_ears_mean = norm_ears_mean
        self.norm_ears_std  = norm_ears_std

        # --- 6. Build normalised item list ---
        self.items = []

        for i, item in enumerate(hrtf_points):
            hrtf_tensor = all_hrtf[i]  # (2, L)
            row         = item['head_row']

            normalised_item = {
                'hrtf':              (hrtf_tensor - norm_mean) / norm_std,
                'subject_id':        item['subject_id'],
                'measurement_point': item['point'],
                'head_measurements': (head_measurements_raw[row] - norm_head_mean) / norm_head_std,
                'ear_measurements':  (ear_measurements_raw[row]  - norm_ears_mean)  / norm_ears_std,
                'global_mean':       norm_mean,
                'global_std':        norm_std,
                'is_mirrored':       False,
            }
            self.items.append(normalised_item)

            # --- 7. Ear-mirroring augmentation (training set only) ---
            # For each real item, append an azimuth-mirrored counterpart:
            #   - Swap HRIR channels [L, R] → [R, L]
            #   - Replace DOA label with the mirror LUT entry for this point
            #   - Anthropometrics are unchanged (bilateral symmetry assumption)
            # The mirrored item is a new, independent sample from the model's
            # perspective — it has a different DOA label and swapped channels.
            if self.augment:
                mirrored_hrtf  = hrtf_tensor[[1, 0], :]  # swap channels
                mirrored_point = int(self.mirror_lut[item['point']])

                mirrored_item = {
                    'hrtf':              (mirrored_hrtf - norm_mean) / norm_std,
                    'subject_id':        item['subject_id'],
                    'measurement_point': mirrored_point,
                    'head_measurements': (head_measurements_raw[row] - norm_head_mean) / norm_head_std,
                    'ear_measurements':  (ear_measurements_raw[row]  - norm_ears_mean)  / norm_ears_std,
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
