"""Small Training-only SFS timing probe and analytical, NOT executed matrix costs."""
import json
import time
import warnings
from pathlib import Path
from src.features.selection import TemporalFeatureSelector
from src.training.config import ExperimentConfig
from src.training.pipeline import candidate_features
from src.training.cv import load_training


def main():
    root = Path(__file__).resolve().parents[2]
    frame, _ = load_training(root)
    X = frame.iloc[:360][candidate_features(ExperimentConfig())]
    y = frame.Appliances.iloc[:360]
    selector = TemporalFeatureSelector(method='forward_sfs', k=10, inner_splits=5, seed=42)
    tick = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        selector.fit(X, y)
    elapsed = time.perf_counter() - tick
    fitted = len(selector.audit_)
    total = sum(sum(p - step for step in range(k)) * 5 * 5
                for p in [27, 28] for k in [10, 15, 20, 25])
    shared = sum(sum(p - step for step in range(25)) * 5 * 5 for p in [27, 28])
    reference = elapsed / fitted * total
    result = dict(purpose='timing_probe_only_not_research_conclusion', status='PASS',
                  rows=360, candidates=27, k=10, inner_splits=5, actual_inner_pipeline_fits=fitted,
                  seconds=elapsed, warnings=[str(w.message) for w in caught],
                  selected_features=selector.selected_features_,
                  planned_matrix_status='NOT_RUN', planned_configs=50, planned_outer_fits=250,
                  planned_sfs_inner_pipeline_fits=total,
                  possible_sfs_shared_path_inner_pipeline_fits=shared,
                  same_per_fit_cost_reference_seconds=reference,
                  row_scaling_scenario_seconds=reference * len(frame) / len(X),
                  caveat='Both timing extrapolations are planning scenarios, not measured matrix runtime or guarantees. '
                         'Full rows, conditioning, BLAS, audit I/O and caching affect cost. '
                         'Shared-path caching is a future batching proposal, not implemented here.')
    out = root / 'reports/experiments/b1-sfs-cost-probe.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
