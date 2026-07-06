import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, Subset
from pysofaconventions import *
import numpy as np


# Subjects with incomplete/corrupted data (excluded as in original)
EXCLUDED_SUBJECTS = {18, 79, 92}
TOTAL_SUBJECTS = 96
MEASUREMENT_POINTS = 440


class HUTUBSDataset(Dataset):
    """
    HUTUBS dataset loader with:
      - Sigmoid normalization for anthropometric features (Eq. 4 in paper)
        instead of z-score, bounding features to (0,1) for conditioning stability.
      - Subject-level indexing to enable proper LOOCV / k-fold splits
        (no cross-subject leakage).
    """

    def __init__(self, hrtf_directory, anthro_csv_path):
        self.hrtf_directory = hrtf_directory
        self.anthro_csv_path = anthro_csv_path
        self.load_data()

    # ── Sigmoid normalization per Eq. (4) of the paper ───────────────────────
    @staticmethod
    def sigmoid_normalize(x, mean, std):
        """Normalize then squash to (0,1) via sigmoid."""
        return 1.0 / (1.0 + torch.exp(-(x - mean) / (std + 1e-8)))

    def load_data(self):
        sofa_files = []
        valid_subject_indices = []  # 1-based subject IDs that are NOT excluded

        for n in range(1, TOTAL_SUBJECTS + 1):
            if n in EXCLUDED_SUBJECTS:
                continue
            file_path = os.path.join(self.hrtf_directory, f'pp{n}_HRIRs_measured.sofa')
            sofa = SOFAFile(file_path, 'r')
            sofa_files.append((n, sofa))
            valid_subject_indices.append(n)

        self.valid_subject_indices = valid_subject_indices  # list of valid 1-based IDs

        # Load source positions from first file (same for all subjects)
        _, first_sofa = sofa_files[0]
        sourcePositions = first_sofa.getVariableValue('SourcePosition')
        self.source_positions = sourcePositions

        # ── Collect all HRIR points ───────────────────────────────────────────
        hrtf_points = []
        for subj_list_idx, (subj_id, sofa_file) in enumerate(sofa_files):
            hrtf_data = sofa_file.getDataIR()
            for point in range(MEASUREMENT_POINTS):
                hrtf_point = hrtf_data[point, :, :].data
                if np.isnan(hrtf_point).any():
                    print(f"NaN detected at subject: {subj_id}, point: {point} — skipping")
                    continue
                hrtf_points.append({
                    'hrtf': hrtf_point,
                    'point': point,
                    'subj_list_idx': subj_list_idx,   # index into valid_subject_indices
                    'subj_id': subj_id,               # original 1-based subject ID
                })

        # ── Global HRIR statistics for normalization ──────────────────────────
        all_hrtf = np.array([item['hrtf'] for item in hrtf_points])
        self.global_mean = float(np.mean(all_hrtf))
        self.global_std = float(np.std(all_hrtf))

        # ── Anthropometric measurements ───────────────────────────────────────
        af_csv = pd.read_csv(self.anthro_csv_path, header=0)
        subject_ids_csv = af_csv.iloc[:, 0].values          # 1-based IDs
        head_meas_raw = torch.from_numpy(af_csv.iloc[:, 1:14].values.astype(np.float32))
        ear_meas_raw = torch.from_numpy(af_csv.iloc[:, 14:].values.astype(np.float32))

        # Drop excluded subjects' rows (e.g. 18/79/92 — present in the CSV
        # but fully NaN) *before* any imputation or normalization statistics
        # are computed.
        valid_mask = np.isin(subject_ids_csv, valid_subject_indices)
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            print(f"[anthro] Dropping {n_dropped} excluded-subject row(s) from anthro CSV "
                  f"before computing normalization statistics: "
                  f"{sorted(subject_ids_csv[~valid_mask].tolist())}")
        subject_ids_csv = subject_ids_csv[valid_mask]
        head_meas_raw   = head_meas_raw[torch.from_numpy(valid_mask)]
        ear_meas_raw    = ear_meas_raw[torch.from_numpy(valid_mask)]

        # Replace NaN with column mean before normalization
        for col in range(head_meas_raw.shape[1]):
            col_mean = head_meas_raw[:, col][~torch.isnan(head_meas_raw[:, col])].mean()
            head_meas_raw[:, col] = torch.where(
                torch.isnan(head_meas_raw[:, col]),
                col_mean.expand_as(head_meas_raw[:, col]),
                head_meas_raw[:, col]
            )
        for col in range(ear_meas_raw.shape[1]):
            col_mean = ear_meas_raw[:, col][~torch.isnan(ear_meas_raw[:, col])].mean()
            ear_meas_raw[:, col] = torch.where(
                torch.isnan(ear_meas_raw[:, col]),
                col_mean.expand_as(ear_meas_raw[:, col]),
                ear_meas_raw[:, col]
            )

        # Sigmoid normalization (Eq. 4)
        head_mean = head_meas_raw.mean(dim=0)
        head_std = head_meas_raw.std(dim=0)
        ear_mean = ear_meas_raw.mean(dim=0)
        ear_std = ear_meas_raw.std(dim=0)

        head_meas_norm = self.sigmoid_normalize(head_meas_raw, head_mean, head_std)
        ear_meas_norm = self.sigmoid_normalize(ear_meas_raw, ear_mean, ear_std)

        # Build a lookup: 1-based subject_id -> normalized measurements
        self.head_meas_by_id = {}
        self.ear_meas_by_id = {}
        for i, sid in enumerate(subject_ids_csv):
            self.head_meas_by_id[int(sid)] = head_meas_norm[i]
            self.ear_meas_by_id[int(sid)] = ear_meas_norm[i]

        # ── Build normalized dataset ──────────────────────────────────────────
        g_mean = self.global_mean
        g_std = self.global_std

        self.normalized_dataset = [
            {
                'hrtf': (torch.from_numpy(item['hrtf']).float() - g_mean) / g_std,
                'subject_id': item['subj_id'],
                'subj_list_idx': item['subj_list_idx'],
                'measurement_point': item['point'],
                'head_measurements': self.head_meas_by_id[item['subj_id']],
                'ear_measurements': self.ear_meas_by_id[item['subj_id']],
                'global_std': g_std,
                'global_mean': g_mean,
            }
            for item in hrtf_points
        ]

    def __len__(self):
        return len(self.normalized_dataset)

    def __getitem__(self, idx):
        return self.normalized_dataset[idx]

    # ── Subject-level k-fold splits (no leakage) ─────────────────────────────
    def get_kfold_splits(self, k=5, val_frac=0.15, seed=42):
        """
        Returns a list of k dicts, each with keys 'train', 'val', 'test'
        containing lists of dataset indices.

        Subjects are shuffled at the subject level before folding so that
        each fold's test set is a contiguous block of subjects.
        val_frac: fraction of remaining (non-test) subjects used for validation.
        """
        rng = np.random.default_rng(seed)
        subjects = np.array(self.valid_subject_indices)
        rng.shuffle(subjects)

        subject_folds = np.array_split(subjects, k)

        # Build subject_id -> list of dataset indices
        subj_to_indices = {}
        for i, item in enumerate(self.normalized_dataset):
            sid = item['subject_id']
            subj_to_indices.setdefault(sid, []).append(i)

        splits = []
        for fold_idx in range(k):
            test_subjects = set(subject_folds[fold_idx].tolist())
            remaining_subjects = [s for s in subjects if s not in test_subjects]

            n_val = max(1, int(len(remaining_subjects) * val_frac))
            val_subjects = set(remaining_subjects[:n_val])
            train_subjects = set(remaining_subjects[n_val:])

            train_indices, val_indices, test_indices = [], [], []
            for sid, indices in subj_to_indices.items():
                if sid in train_subjects:
                    train_indices.extend(indices)
                elif sid in val_subjects:
                    val_indices.extend(indices)
                elif sid in test_subjects:
                    test_indices.extend(indices)

            splits.append({
                'train': train_indices,
                'val': val_indices,
                'test': test_indices,
                'test_subjects': sorted(test_subjects),
            })
            print(
                f"Fold {fold_idx + 1}: "
                f"train={len(train_subjects)} subjects ({len(train_indices)} samples), "
                f"val={len(val_subjects)} subjects ({len(val_indices)} samples), "
                f"test={len(test_subjects)} subjects ({len(test_indices)} samples)"
            )
        return splits


def collate_fn(batch):
    return {
        'hrtf': torch.stack([item['hrtf'] for item in batch]),
        'measurement_point': torch.tensor([item['measurement_point'] for item in batch]),
        'subject_id': torch.LongTensor([item['subject_id'] for item in batch]),
        'head_measurements': torch.stack([item['head_measurements'] for item in batch]),
        'ear_measurements': torch.stack([item['ear_measurements'] for item in batch]),
    }