"""
Paired Wilcoxon signed-rank comparison across loss types (l1 / l2 /
combined / log_combined, or however many of losses.LOSS_TYPES you've
actually run) -- the loss-ablation counterpart to main_model/compare_
conditions.py's paired statistical testing for the SONICOM A/B/C/D
conditioning ablation. Named compare_losses.py (not compare_conditions.py)
so it doesn't collide with or get confused for that script -- same idea,
different axis of comparison.

Why pairing by subject_id is valid here: loss_exp/main.py computes the
k-fold split ONCE (seed=42) and reuses it for every loss type, so every
loss type evaluates the IDENTICAL held-out subjects in every fold. That
means metrics_per_subject.xlsx for each loss type (after
recompute_metrics.py has merged all folds you've run into one file per
loss type) contain the same subject_ids, each subject's HRTF reconstructed
once per loss type, from the same ground truth. Each subject is its own
control, which is what makes the Wilcoxon SIGNED-RANK test (as opposed to
an unpaired test) appropriate and more powerful than comparing independent
means.

Generalized to however many loss types you've actually run (originally
hardcoded to exactly l1/l2/combined; now handles log_combined -- or any
future addition to losses.LOSS_TYPES -- without further changes here).
For every metric and every pairwise combination of the loss types found,
this runs:
  - scipy.stats.wilcoxon (two-sided by default -- doesn't bake in a
    directional hypothesis after already having seen which loss looked
    better descriptively)
  - the matched-pairs rank-biserial correlation as effect size (per your
    existing compare_conditions.py convention)
  - Holm-Bonferroni correction across the whole table, since running
    multiple pairwise comparisons x N metrics is multiple testing and an
    uncorrected p < 0.05 per cell would overstate significance

Usage -- auto-discover every loss type present under one base name (only
those with an actual metrics_per_subject.xlsx are used, so this works
whether you've run all four loss types or just two of them, e.g. 'combined'
and the new 'log_combined' for a quick one-fold screen):
    python loss_exp/compare_losses.py --base_model_name SONICOM_lossexp_A

Usage -- explicit results/ dir names, any number >= 2, in any order:
    python loss_exp/compare_losses.py --model_names \
        SONICOM_lossexp_A_combined SONICOM_lossexp_A_log_combined

Requires each dir's metrics_per_subject.xlsx to already exist (produced by
`--mode infer` and/or recompute_metrics.py).
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── Make loss_exp/ (for LOSS_TYPES) importable regardless of cwd ─────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from losses import LOSS_TYPES  # noqa: E402  -- ('l1','l2','combined','log_combined', ...)

DEFAULT_METRICS = ['lsd_avg', 'itd', 'pbc', 'nmse']


def rank_biserial_from_diffs(diffs):
    """
    Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test:
        r = (W+ - W-) / (W+ + W-)
    W+/W- are the sums of ranks of |diff| assigned to positive/negative
    signed differences (average ranks for ties in magnitude; zero
    differences dropped first, matching scipy's default zero_method='wilcox'
    so this lines up with the p-value scipy reports).

    r near 0  -> the two losses are practically indistinguishable subject-by-subject.
    |r| near 1 -> almost every subject moved in the same direction.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return np.nan, 0
    ranks = pd.Series(np.abs(diffs)).rank().values   # average ranks for ties
    W_pos = ranks[diffs > 0].sum()
    W_neg = ranks[diffs < 0].sum()
    return (W_pos - W_neg) / (W_pos + W_neg), n


def holm_bonferroni(pvalues):
    """
    Holm's step-down procedure — controls the family-wise error rate
    across the whole comparison table without being as conservative as a
    flat Bonferroni correction. Returns adjusted p-values in the ORIGINAL
    order of `pvalues`.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[order[rank]] = min(running_max, 1.0)
    return adjusted


def load_metrics(results_root, model_name):
    path = os.path.join(results_root, model_name, 'metrics_per_subject.xlsx')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `--mode infer` and/or recompute_metrics.py "
            f"for '{model_name}' first."
        )
    df = pd.read_excel(path)
    if df['subject_id'].duplicated().any():
        dupes = df.loc[df['subject_id'].duplicated(keep=False), 'subject_id'].unique()
        raise ValueError(
            f"{path} has duplicate subject_id rows ({sorted(dupes.tolist())}) — a subject "
            f"should appear at most once across all folds you've merged into this file. "
            f"Check recompute_metrics.py wasn't run twice into overlapping fold sets."
        )
    return df.set_index('subject_id')


def main():
    parser = argparse.ArgumentParser(description='Paired Wilcoxon signed-rank comparison across loss types.')
    parser.add_argument('--results_root', type=str, default='./results')
    parser.add_argument('--base_model_name', type=str, default=None,
                        help='Auto-discovers "<base>_<loss_type>" for every loss_type in '
                             'losses.LOSS_TYPES under --results_root, using whichever ones '
                             'actually have a metrics_per_subject.xlsx (so this works whether '
                             "you've run all of them or just a couple, e.g. only 'combined' "
                             "and 'log_combined' for a quick one-fold screen).")
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='Explicit results/ dir names to compare (2 or more), instead of '
                             '--base_model_name. loss_type label for each is inferred from an '
                             '"_<loss_type>" suffix matching losses.LOSS_TYPES if present, else '
                             'the directory name itself is used as the label.')
    parser.add_argument('--metrics', type=str, nargs='+', default=DEFAULT_METRICS,
                        help=f'Subject-level metric columns to test. Default: {DEFAULT_METRICS} '
                             '(all lower-is-better; add lsd_L / lsd_R if you want per-ear detail).')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--alternative', type=str, default='two-sided',
                        choices=['two-sided', 'less', 'greater'],
                        help='scipy.stats.wilcoxon alternative. Default two-sided — avoids baking '
                             'in a directional hypothesis after already having seen the descriptive numbers.')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Defaults to "<results_root>/<base_model_name or \'loss\'>_wilcoxon.xlsx".')
    args = parser.parse_args()

    # ── Resolve which (label, results_dir_name) pairs to load ────────────────
    if args.model_names:
        if len(args.model_names) < 2:
            parser.error('--model_names needs at least 2 directories to compare.')
        jobs = []
        for name in args.model_names:
            label = next((lt for lt in LOSS_TYPES if name.endswith(f'_{lt}')), name)
            jobs.append((label, name))
    elif args.base_model_name:
        jobs = []
        for lt in LOSS_TYPES:
            candidate = f'{args.base_model_name}_{lt}'
            if os.path.exists(os.path.join(args.results_root, candidate, 'metrics_per_subject.xlsx')):
                jobs.append((lt, candidate))
        if len(jobs) < 2:
            parser.error(
                f"Found {len(jobs)} loss type(s) with a metrics_per_subject.xlsx under "
                f"'{args.base_model_name}_{{{','.join(LOSS_TYPES)}}}' — need at least 2 to compare. "
                f"Run `--mode infer` (and/or recompute_metrics.py) for more loss types first."
            )
    else:
        parser.error('Provide --base_model_name or --model_names.')

    print(f"Comparing: {[label for label, _ in jobs]}")
    dfs = {label: load_metrics(args.results_root, name) for label, name in jobs}
    PAIRS = list(itertools.combinations(dfs.keys(), 2))

    rows = []
    for metric in args.metrics:
        for a, b in PAIRS:
            df_a, df_b = dfs[a], dfs[b]
            if metric not in df_a.columns or metric not in df_b.columns:
                print(f"  [{metric} {a} vs {b}] column '{metric}' not found in one of the two "
                      f"files — skipping.")
                continue

            shared_ids = df_a.index.intersection(df_b.index)
            missing_a  = df_a.index.difference(df_b.index)
            missing_b  = df_b.index.difference(df_a.index)
            if len(missing_a) or len(missing_b):
                print(f"  [{metric} {a} vs {b}] warning: {len(missing_a)} subjects only in "
                      f"{a}, {len(missing_b)} only in {b} — dropped from this pair (using "
                      f"{len(shared_ids)} shared subjects). This shouldn't happen if both "
                      f"loss types ran inference on the same folds — check for a partial run.")

            x = df_a.loc[shared_ids, metric].values.astype(float)
            y = df_b.loc[shared_ids, metric].values.astype(float)
            diffs = x - y

            if len(diffs[diffs != 0]) == 0:
                print(f"  [{metric} {a} vs {b}] all paired differences are zero — skipping.")
                continue

            stat, p = wilcoxon(x, y, alternative=args.alternative, zero_method='wilcox')
            r, n_nonzero = rank_biserial_from_diffs(diffs)

            rows.append({
                'metric':          metric,
                'group_a':         a,
                'group_b':         b,
                'n_subjects':      len(shared_ids),
                'n_nonzero_diffs': n_nonzero,
                'mean_a':          float(np.mean(x)),
                'mean_b':          float(np.mean(y)),
                'mean_diff':       float(np.mean(diffs)),      # a - b
                'median_diff':     float(np.median(diffs)),
                'wilcoxon_W':      float(stat),
                'p_value':         float(p),
                'rank_biserial_r': float(r),
                'better':          a if np.mean(diffs) < 0 else b,  # lower = better, all default metrics
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        print("No comparisons produced any rows — nothing to save.")
        return

    result_df['p_holm']           = holm_bonferroni(result_df['p_value'].values)
    result_df['significant_holm'] = result_df['p_holm'] < args.alpha

    pd.set_option('display.width', 200)
    print(f"\n{'='*100}\n  PAIRED WILCOXON SIGNED-RANK — LOSS FUNCTION COMPARISON\n{'='*100}")
    print(result_df.to_string(index=False))

    out_path = args.out_path or os.path.join(
        args.results_root, f'{args.base_model_name or "loss"}_wilcoxon.xlsx'
    )
    result_df.to_excel(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
