"""Summarize saved preflight checkpoints; never fits a model."""
import json
from pathlib import Path
from src.training.checkpoint import read_checkpoint


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'reports/experiments/b2a-preflight-v1'
    summary = read_checkpoint(folder / 'summary.json')
    by_method = {g['config']['method']: g for g in summary['groups'].values()}
    sfs = by_method['forward_sfs']
    sfs_fits = sfs['actual_fit_counts']['estimator'] - len(sfs['completed_fold_indices'])
    total_sfs = sum(sum(p - step for step in range(k)) * 25 for p in [27, 28] for k in [10,15,20,25])
    train_counts = []
    for path in (folder / 'folds').glob('*.json'):
        data = read_checkpoint(path)
        if data['task']['config']['method'] == 'select_k_best':
            train_counts.append(data['result']['train']['rows'])
    row_factor = sum(train_counts) / (max(train_counts) * len(train_counts))
    sfs_reference = total_sfs / sfs_fits * sfs['seconds']
    fast = (8 * sum(by_method[m]['seconds'] for m in ['select_k_best', 'permutation', 'lasso'])
            + (480 / by_method['rfe']['actual_fit_counts']['estimator']) * by_method['rfe']['seconds'])
    estimate = dict(status='ESTIMATE_NOT_MEASURED_MATRIX', planned_configs=50, outer_fits=250,
                    sfs_inner_fits_without_cross_k_cache=total_sfs,
                    measured_sfs_outer_fold_indices=sfs['completed_fold_indices'],
                    measured_sfs_inner_fits=sfs_fits, measured_sfs_seconds=sfs['seconds'],
                    non_sfs_non_all_reference_seconds=fast,
                    sfs_row_weighted_scenario_seconds=sfs_reference * row_factor,
                    sfs_largest_fold_equal_cost_scenario_seconds=sfs_reference,
                    subtotal_scenario_seconds=[fast + sfs_reference * row_factor, fast + sfs_reference],
                    planning_buffer_factor=1.5,
                    buffered_subtotal_scenario_seconds=[1.5 * (fast + sfs_reference * row_factor), 1.5 * (fast + sfs_reference)],
                    caveat='Not bounds or guarantees. Does not measure the complete matrix. '
                           'SFS K>10, research28, ALL fits, serialization, process startup and later Validation '
                           'are not all timed in equivalent conditions. Correlation, Lasso convergence, width and machine load '
                           'can change runtime. 1.5 buffer is a planning assumption. No cross-K cache speedup claimed.')
    report = dict(purpose='B2-A readiness only; no model selection', preflight=summary, matrix_estimate=estimate)
    (root / 'reports/experiments/b2a-readiness-report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(estimate, indent=2))
    for group in summary['groups'].values():
        print(json.dumps({k: v for k, v in group.items() if k != 'folds'}))


if __name__ == '__main__':
    main()
