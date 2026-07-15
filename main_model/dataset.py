import os
import re
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pysofaconventions import *
from PIL import Image
import torchvision.transforms.functional as tvf


IMAGE_SIZE = 128   # ear crops are resized to this (square) before the CNN image encoder


class HRTFDataset(Dataset):
    """
    Shared HRTF dataset logic for HUTUBS and SONICOM. A subclass only sets
    FILE_REGEX (how to find HRIR files and parse a subject ID from their
    filename in hrtf_directory) -- everything else (anthro loading,
    normalization, k-fold splitting, optional ear-image loading) lives here
    so the two datasets can't silently drift apart.

    Subject discovery/exclusion is dynamic rather than hardcoded:
      - subjects come from whatever .sofa files actually match FILE_REGEX
        (no fixed subject-count loop, so SONICOM's non-contiguous P0XXX
        IDs work the same way as HUTUBS's contiguous ones)
      - a subject is dropped if its anthro CSV row is entirely NaN, or if
        it's missing a SOFA file or CSV row entirely (no per-dataset
        hardcoded exclusion list)
      - measurement point count is read from the first SOFA file rather
        than assumed, since HUTUBS (440 points) and SONICOM (793 points)
        use different measurement grids
    """
    FILE_REGEX = None   # set by subclass, e.g. re.compile(r'pp(\d+)_HRIRs_measured\.sofa$')

    def __init__(self, hrtf_directory, anthro_csv_path, image_dir=None):
        self.hrtf_directory = hrtf_directory
        self.anthro_csv_path = anthro_csv_path
        self.image_dir = image_dir   # set only for SONICOM conditions that use images (C/D)
        self.load_data()

    # ── Sigmoid normalization per Eq. (4) of the paper ───────────────────────
    @staticmethod
    def sigmoid_normalize(x, mean, std):
        """Normalize then squash to (0,1) via sigmoid."""
        return 1.0 / (1.0 + torch.exp(-(x - mean) / (std + 1e-8)))

    # ── Ear-image loading (SONICOM only, conditions C/D) ─────────────────────
    def _load_ear_image(self, subject_id):
        """Load+resize the L/R ear crops for one subject, stacked as a (6, H, W) tensor."""
        sides = []
        for side in ('L', 'R'):
            path = os.path.join(self.image_dir, f'P{subject_id:04d}_{side}.jpg')
            img = Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            sides.append(tvf.to_tensor(img))   # (3, H, W) in [0,1]
        return torch.cat(sides, dim=0)   # (6, H, W)

    def load_data(self):
        # ── Discover subjects from HRIR files present in hrtf_directory ─────────
        sofa_files = []   # list of (subject_id, SOFAFile)
        for path in sorted(glob.glob(os.path.join(self.hrtf_directory, '*.sofa'))):
            m = self.FILE_REGEX.match(os.path.basename(path))
            if m:
                sofa_files.append((int(m.group(1)), SOFAFile(path, 'r')))
        if not sofa_files:
            raise RuntimeError(f"No .sofa files matching {self.FILE_REGEX.pattern} in {self.hrtf_directory}")

        # ── Anthro CSV: ear columns selected by name (L_/R_ prefix), not position ─
        # so head/torso columns (present in HUTUBS, absent in SONICOM) never
        # need to be counted or sliced around -- this is what lets the same
        # loading code work for both CSVs.
        try:
            af_csv = pd.read_csv(self.anthro_csv_path, header=0, sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            print(f"[anthro] {self.anthro_csv_path} isn't valid UTF-8 -- retrying as Latin-1")
            af_csv = pd.read_csv(self.anthro_csv_path, header=0, sep=None, engine='python', encoding='latin-1')
        if 'SubjectID' not in af_csv.columns:
            raise ValueError(
                f"'{self.anthro_csv_path}' has no 'SubjectID' column (found: {list(af_csv.columns)}). "
                f"This loader expects the wide layout (SubjectID, L_d1..L_theta2, R_d1..R_theta2) -- "
                f"if this is SONICOM's original long-format export (SONICOM ID/Ear/d1(cm).. columns), "
                f"it needs reshaping first, not just re-pointing this path at it."
            )
        ear_cols = [c for c in af_csv.columns if c.startswith('L_') or c.startswith('R_')]
        subject_ids_csv = af_csv['SubjectID'].values
        ear_meas_raw = torch.from_numpy(af_csv[ear_cols].values.astype(np.float32))

        # A subject is auto-excluded if its ear-feature row is entirely NaN
        # (replaces a hardcoded per-dataset exclusion list -- HUTUBS's old
        # {18, 79, 92} list is exactly the set this detects automatically).
        row_all_nan = np.isnan(ear_meas_raw.numpy()).all(axis=1)
        if row_all_nan.any():
            print(f"[anthro] Dropping {int(row_all_nan.sum())} subject(s) with fully-NaN "
                  f"ear features: {sorted(subject_ids_csv[row_all_nan].tolist())}")
        subject_ids_csv = subject_ids_csv[~row_all_nan]
        ear_meas_raw    = ear_meas_raw[~torch.from_numpy(row_all_nan)]

        # Keep only subjects with both a SOFA file and a valid CSV row.
        sofa_ids  = {sid for sid, _ in sofa_files}
        valid_ids = sofa_ids & set(subject_ids_csv.tolist())
        no_csv  = sofa_ids - valid_ids
        no_sofa = set(subject_ids_csv.tolist()) - sofa_ids
        if no_csv:
            print(f"[subjects] HRIR present but no valid anthro row, dropped: {sorted(no_csv)}")
        if no_sofa:
            print(f"[subjects] Anthro row present but no HRIR file, dropped: {sorted(no_sofa)}")

        sofa_files = sorted((sid, s) for sid, s in sofa_files if sid in valid_ids)
        self.valid_subject_indices = [sid for sid, _ in sofa_files]

        # ── Measurement grid: read point count from the data itself ────────────
        self.measurement_points = sofa_files[0][1].getDataIR().shape[0]

        # ── Collect all HRIR points ─────────────────────────────────────────────
        hrtf_points = []
        for subj_list_idx, (subj_id, sofa_file) in enumerate(sofa_files):
            hrtf_data = sofa_file.getDataIR()
            for point in range(self.measurement_points):
                hrtf_point = hrtf_data[point, :, :].data
                if np.isnan(hrtf_point).any():
                    print(f"NaN detected at subject: {subj_id}, point: {point} — skipping")
                    continue
                hrtf_points.append({
                    'hrtf': hrtf_point,
                    'point': point,
                    'subj_list_idx': subj_list_idx,
                    'subj_id': subj_id,
                })

        # ── Global HRIR statistics for normalization ──────────────────────────
        all_hrtf = np.array([item['hrtf'] for item in hrtf_points])
        self.global_mean = float(np.mean(all_hrtf))
        self.global_std = float(np.std(all_hrtf))

        # Replace NaN with column mean before normalization
        for col in range(ear_meas_raw.shape[1]):
            col_mean = ear_meas_raw[:, col][~torch.isnan(ear_meas_raw[:, col])].mean()
            ear_meas_raw[:, col] = torch.where(
                torch.isnan(ear_meas_raw[:, col]),
                col_mean.expand_as(ear_meas_raw[:, col]),
                ear_meas_raw[:, col]
            )

        # Sigmoid normalization (Eq. 4)
        ear_mean = ear_meas_raw.mean(dim=0)
        ear_std = ear_meas_raw.std(dim=0)
        ear_meas_norm = self.sigmoid_normalize(ear_meas_raw, ear_mean, ear_std)

        # Build a lookup: subject_id -> normalized measurements
        self.ear_meas_by_id = {int(sid): ear_meas_norm[i] for i, sid in enumerate(subject_ids_csv)}

        # ── Build normalized dataset ──────────────────────────────────────────
        g_mean = self.global_mean
        g_std = self.global_std

        self.normalized_dataset = [
            {
                'hrtf': (torch.from_numpy(item['hrtf']).float() - g_mean) / g_std,
                'subject_id': item['subj_id'],
                'subj_list_idx': item['subj_list_idx'],
                'measurement_point': item['point'],
                'ear_measurements': self.ear_meas_by_id[item['subj_id']],
                'global_std': g_std,
                'global_mean': g_mean,
            }
            for item in hrtf_points
        ]

    def __len__(self):
        return len(self.normalized_dataset)

    def __getitem__(self, idx):
        item = self.normalized_dataset[idx]
        if self.image_dir is not None:
            # Images are decoded on demand rather than precomputed/cached in
            # normalized_dataset -- there can be hundreds of HRIR points per
            # subject, and re-storing the same image tensor for each one
            # would be wasteful; a 128x128 JPEG decode is cheap per sample.
            item = dict(item)
            item['image'] = self._load_ear_image(item['subject_id'])
        return item

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


class HUTUBSDataset(HRTFDataset):
    """HUTUBS: files named pp{n}_HRIRs_measured.sofa. No ear images available."""
    FILE_REGEX = re.compile(r'pp(\d+)_HRIRs_measured\.sofa$')


class SONICOMDataset(HRTFDataset):
    """
    SONICOM: files named P{n:04d}_FreeFieldComp_44kHz.sofa. Pass image_dir
    to also load each subject's L/R ear crops (P{n:04d}_L.jpg / _R.jpg) for
    the image-conditioning experiments (conditions C/D) -- leave it as None
    for conditions A/B, which don't need images at all.
    """
    FILE_REGEX = re.compile(r'P(\d+)_FreeFieldComp_44kHz\.sofa$')


def collate_fn(batch):
    out = {
        'hrtf': torch.stack([item['hrtf'] for item in batch]),
        'measurement_point': torch.tensor([item['measurement_point'] for item in batch]),
        'subject_id': torch.LongTensor([item['subject_id'] for item in batch]),
        'ear_measurements': torch.stack([item['ear_measurements'] for item in batch]),
    }
    if 'image' in batch[0]:
        out['image'] = torch.stack([item['image'] for item in batch])
    return out