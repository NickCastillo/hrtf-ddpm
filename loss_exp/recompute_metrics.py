"""
Recompute LSD / ITD / PBC from already-generated HRIRs, without rerunning
diffusion sampling or touching any checkpoint.

Why this exists: fixing a metric function (e.g. the ERB-rate bug / PBC's
15kHz cap in utils.py) never requires retraining -- combined_loss doesn't
call pbc()/_erb_filters at all, and its own frequency term was already
capped at F_MAX=15000. It also doesn't strictly require a fresh
`--mode infer` run: inference is deterministic given a checkpoint
(torch.manual_seed(42) per subject in infer_fold), so the HRIRs already
saved to results/<model_name>/mat/fold_N/sub_*.mat during a previous
`--mode infer` run are exactly what a fresh run would regenerate. This
script reloads those saved arrays and recomputes lsd()/itd_error()/pbc()
directly -- no GPU, no model, no 600-step sampling, just seconds.

It overwrites, per model_name it processes:
  results/<model_name>/mat/fold_N_summary.mat   (per-fold aggregates)
  results/<model_name>/mat/all_folds_summary.mat
  results/<model_name>/metrics_per_subject.xlsx
and, if more than one loss type is processed in one invocation, also
writes a fresh comparison table (mirroring loss_exp/main.py's own
comparison output):
  results/<base_model_name_or_recomputed>_loss_comparison_infer.xlsx

NOTE: this only helps for metric-only changes (LSD/ITD/PBC formula
fixes). If you ever change the model, training loss, or diffusion
sampling itself, there's no shortcut -- you do need a real re-train /
re-infer.

Usage -- auto-discover every loss type under one base name:
    python loss_exp/recompute_metrics.py --base_model_name SONICOM_lossexp_A

Usage -- explicit list of result dirs:
    python loss_exp/recompute_metrics.py --model_names \
        SONICOM_lossexp_A_l1 SONICOM_lossexp_A_l2 SONICOM_lossexp_A_combined

Optional: --pbc_f_max lets you override pbc()'s frequency cap for this
recompute only (e.g. --pbc_f_max 22050 to reproduce the old full-Nyquist
PBC for comparison, without touching utils.py's default).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

# ── Make main_model/ importable (same convention as losses.py / main.py) ────
_MAIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'main_model')
if _MAIN_MODEL_DIR not in sys.path:
    sys.path.insert(0, _MAIN_MODEL_DIR)

from utils import lsd, itd_error, pbc  # noqa: E402
from losses import LOSS_TYPES          # noqa: E402  -- local to loss_exp/, ('l1','l2','combined')


def process_model_dir(results_root, model_name, loss_type, pbc_f_max):
    """
    Recomputes metrics for one results/<model_name>/ dir from its saved
    per-subject .mat files. Returns an aggregate stats dict, or None if
    nothing was found to process.
    """
    res_dir = os.path.join(results_root, model_name)
    mat_dir = os.path.join(res_dir, 'mat')
    if not os.path.isdir(mat_dir):
        print(f"  [{model_name}] no mat/ dir found at {mat_dir} — skipping")
        return None

    fold_dirs = sorted(d for d in glob.glob(os.path.join(mat_dir, 'fold_*')) if os.path.isdir(d))

    all_subject_ids, all_lsd_L, all_lsd_R, all_lsd_avg = [], [], [], []
    all_itd, all_pbc, all_nmse_subj, all_nmse_pos = [], [], [], []

    for fold_dir in fold_dirs:
        fold_tag = os.path.basename(fold_dir)
        sub_files = sorted(glob.glob(os.path.join(fold_dir, 'sub_*.mat')))
        if not sub_files:
            continue

        f_subject_ids, f_lsd_L, f_lsd_R, f_lsd_avg = [], [], [], []
        f_itd, f_pbc, f_nmse_subj, f_nmse_pos = [], [], [], []

        for sf in sub_files:
            d = sio.loadmat(sf)
            hrir_gen = np.asarray(d['hrir_gen'])   # (n_valid, 2, 256)
            hrir_gt  = np.asarray(d['hrir_gt'])
            sid = int(np.asarray(d['subject_id']).ravel()[0])
            n_valid = hrir_gen.shape[0]
            if n_valid == 0:
                continue

            gt_list  = [hrir_gt[i]  for i in range(n_valid)]
            gen_list = [hrir_gen[i] for i in range(n_valid)]

            lsd_vals = lsd(gt_list, gen_list, n_valid, sr=44100)
            itd_val  = itd_error(gt_list, gen_list, sr=44100)
            pbc_val  = (pbc(gt_list, gen_list, sr=44100, f_max=pbc_f_max) if pbc_f_max is not None
                        else pbc(gt_list, gen_list, sr=44100))   # uses utils.py's own default (15000, post-fix)
            nmse_vals = np.asarray(d['nmse_values']).ravel().tolist()

            f_subject_ids.append(sid)
            f_lsd_L.append(lsd_vals['L']); f_lsd_R.append(lsd_vals['R']); f_lsd_avg.append(lsd_vals['avg'])
            f_itd.append(itd_val); f_pbc.append(pbc_val)
            f_nmse_subj.append(float(np.mean(nmse_vals)))
            f_nmse_pos.extend(nmse_vals)

        if not f_subject_ids:
            continue

        # Preserve whatever extra metadata the old summary had (test_subjects,
        # full_attention) -- this script only corrects metric VALUES, not the
        # run's identity/config.
        summary_path = os.path.join(mat_dir, f'{fold_tag}_summary.mat')
        old_meta = {}
        if os.path.exists(summary_path):
            old = sio.loadmat(summary_path)
            for key in ('test_subjects', 'full_attention'):
                if key in old:
                    old_meta[key] = old[key]

        sio.savemat(summary_path, {
            'subject_id_per_subject': np.array(f_subject_ids, dtype=np.int32),
            'lsd_L_per_subject':  np.array(f_lsd_L,   dtype=np.float64),
            'lsd_R_per_subject':  np.array(f_lsd_R,   dtype=np.float64),
            'lsd_avg_per_subject':np.array(f_lsd_avg, dtype=np.float64),
            'itd_per_subject':    np.array(f_itd,     dtype=np.float64),
            'pbc_per_subject':    np.array(f_pbc,     dtype=np.float64),
            'nmse_per_subject':   np.array(f_nmse_subj, dtype=np.float64),
            'nmse_per_position':  np.array(f_nmse_pos,  dtype=np.float64),
            'model':      model_name,
            'loss_type':  loss_type,
            'mean_lsd_L':   float(np.mean(f_lsd_L)),
            'mean_lsd_R':   float(np.mean(f_lsd_R)),
            'mean_lsd_avg': float(np.mean(f_lsd_avg)),
            'mean_itd':     float(np.mean(f_itd)),
            'mean_pbc':     float(np.mean(f_pbc)),
            'mean_nmse':    float(np.mean(f_nmse_pos)),
            **old_meta,
        })
        print(f"  [{model_name}] {fold_tag}: recomputed {len(f_subject_ids)} subjects  |  "
              f"LSD_avg={np.mean(f_lsd_avg):.3f} dB  ITD={np.mean(f_itd):.2f} µs  "
              f"PBC={np.mean(f_pbc):.3f} dB  NMSE={np.mean(f_nmse_pos):.4f}")

        all_subject_ids.extend(f_subject_ids)
        all_lsd_L.extend(f_lsd_L); all_lsd_R.extend(f_lsd_R); all_lsd_avg.extend(f_lsd_avg)
        all_itd.extend(f_itd); all_pbc.extend(f_pbc)
        all_nmse_subj.extend(f_nmse_subj); all_nmse_pos.extend(f_nmse_pos)

    if not all_lsd_avg:
        print(f"  [{model_name}] no per-subject .mat files found under {mat_dir} — skipping")
        return None

    sio.savemat(os.path.join(mat_dir, 'all_folds_summary.mat'), {
        'subject_id_all': np.array(all_subject_ids, dtype=np.int32),
        'lsd_L_all':  np.array(all_lsd_L,    dtype=np.float64),
        'lsd_R_all':  np.array(all_lsd_R,    dtype=np.float64),
        'lsd_avg_all':np.array(all_lsd_avg,  dtype=np.float64),
        'itd_all':    np.array(all_itd,      dtype=np.float64),
        'pbc_all':    np.array(all_pbc,      dtype=np.float64),
        'nmse_all':   np.array(all_nmse_pos, dtype=np.float64),
        'model':      model_name,
        'loss_type':  loss_type,
        'mean_lsd_L':  float(np.mean(all_lsd_L)),
        'mean_lsd_R':  float(np.mean(all_lsd_R)),
        'mean_lsd_avg':float(np.mean(all_lsd_avg)),
        'mean_itd':    float(np.mean(all_itd)),
        'mean_pbc':    float(np.mean(all_pbc)),
        'mean_nmse':   float(np.mean(all_nmse_pos)),
    })

    pd.DataFrame({
        'subject_id': all_subject_ids,
        'model':     model_name,
        'loss_type': loss_type,
        'lsd_L':   all_lsd_L,
        'lsd_R':   all_lsd_R,
        'lsd_avg': all_lsd_avg,
        'itd':     all_itd,
        'pbc':     all_pbc,
        'nmse':    all_nmse_subj,
    }).to_excel(os.path.join(res_dir, 'metrics_per_subject.xlsx'), index=False)

    print(f"  [{model_name}] OVERALL — LSD_avg={np.mean(all_lsd_avg):.3f} ± {np.std(all_lsd_avg):.3f} dB  |  "
          f"ITD={np.mean(all_itd):.2f} ± {np.std(all_itd):.2f} µs  |  "
          f"PBC={np.mean(all_pbc):.3f} ± {np.std(all_pbc):.3f} dB  |  "
          f"NMSE={np.mean(all_nmse_pos):.4f} ± {np.std(all_nmse_pos):.4f}")

    return {
        'lsd_avg_mean': float(np.mean(all_lsd_avg)), 'lsd_avg_std': float(np.std(all_lsd_avg)),
        'itd_mean':     float(np.mean(all_itd)),      'itd_std':     float(np.std(all_itd)),
        'pbc_mean':     float(np.mean(all_pbc)),      'pbc_std':     float(np.std(all_pbc)),
        'nmse_mean':    float(np.mean(all_nmse_pos)), 'nmse_std':    float(np.std(all_nmse_pos)),
        'n_subjects':   len(all_subject_ids),
    }


def main():
    parser = argparse.ArgumentParser(description='Recompute LSD/ITD/PBC from saved HRIRs (no re-inference).')
    parser.add_argument('--results_root', type=str, default='./results')
    parser.add_argument('--base_model_name', type=str, default=None,
                        help='Auto-discovers "<base>_l1", "<base>_l2", "<base>_combined" under '
                             '--results_root (whichever of the three actually exist).')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='Explicit list of results/ dir names to process instead of '
                             '--base_model_name. loss_type is inferred from an "_l1"/"_l2"/'
                             '"_combined" suffix if present, else the whole name is used as-is.')
    parser.add_argument('--pbc_f_max', type=float, default=None,
                        help='Override pbc()\'s frequency cap for this recompute only (utils.py\'s '
                             'own default is used if omitted -- 15000 Hz after the fix). Pass '
                             '22050 to reproduce the old full-Nyquist PBC for comparison.')
    args = parser.parse_args()

    if not args.base_model_name and not args.model_names:
        parser.error('Provide --base_model_name or --model_names.')

    if args.model_names:
        jobs = []
        for name in args.model_names:
            loss_type = next((lt for lt in LOSS_TYPES if name.endswith(f'_{lt}')), name)
            jobs.append((loss_type, name))
    else:
        jobs = []
        for lt in LOSS_TYPES:
            candidate = f'{args.base_model_name}_{lt}'
            if os.path.isdir(os.path.join(args.results_root, candidate, 'mat')):
                jobs.append((lt, candidate))
        if not jobs:
            parser.error(f"No '{args.base_model_name}_{{{','.join(LOSS_TYPES)}}}' dirs with a "
                         f"mat/ folder found under {args.results_root}.")

    print(f"Recomputing metrics for: {[name for _, name in jobs]}")
    if args.pbc_f_max is not None:
        print(f"PBC f_max override: {args.pbc_f_max} Hz")

    comparison = {}
    for loss_type, model_name in jobs:
        print(f"\n{'='*60}\n  {model_name}  (loss_type={loss_type})\n{'='*60}")
        stats = process_model_dir(args.results_root, model_name, loss_type, args.pbc_f_max)
        if stats is not None:
            comparison[loss_type] = stats

    if len(comparison) > 1:
        print(f"\n{'='*70}\n  RECOMPUTED COMPARISON\n{'='*70}")
        rows = []
        for lt, stats in comparison.items():
            rows.append({'loss_type': lt, **stats})
            print(f"  {lt:>9s}  LSD_avg = {stats['lsd_avg_mean']:.3f} ± {stats['lsd_avg_std']:.3f} dB  |  "
                  f"ITD = {stats['itd_mean']:.2f} ± {stats['itd_std']:.2f} µs  |  "
                  f"PBC = {stats['pbc_mean']:.3f} ± {stats['pbc_std']:.3f} dB  |  "
                  f"NMSE = {stats['nmse_mean']:.4f} ± {stats['nmse_std']:.4f}")

        tag = args.base_model_name or 'recomputed'
        out_path = os.path.join(args.results_root, f'{tag}_loss_comparison_infer.xlsx')
        pd.DataFrame(rows).to_excel(out_path, index=False)
        print(f"\nComparison table saved to {out_path}")


if __name__ == '__main__':
    main()
