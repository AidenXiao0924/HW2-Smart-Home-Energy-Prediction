import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from src.training.config import ExperimentConfig
from src.training.cv import load_training, run_cv


def main():
    root = Path(__file__).resolve().parents[2]
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != root / '.venv':
        raise RuntimeError('Official project .venv Python 3.11 required')
    raw_config = (root / 'configs/b1_smoke.json').read_bytes()
    settings = json.loads(raw_config)
    frame, manifest = load_training(root)
    subset = frame.iloc[:settings['training_prefix_rows']]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root / 'reports/experiments' / ('b1-smoke-' + run_id)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_bytes(raw_config)
    common = {key: settings[key] for key in ['seed', 'outer_splits', 'inner_splits',
                                             'candidate_set', 'permutation_repeats']}
    started = time.perf_counter()
    results = []
    for entry in settings['experiments']:
        config = ExperimentConfig(**common, **entry)
        try:
            result = run_cv(subset, config, frame.index)
            result['status'] = 'PASS'
        except Exception as exc:
            result = dict(status='FAIL', experiment_id=config.identifier, error=f'{type(exc).__name__}: {exc}')
            (out / f'{config.identifier}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            raise
        (out / f'{config.identifier}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        results.append(dict(experiment_id=config.identifier, method=config.method, k=config.k,
                            seconds=result['seconds'], cv_rmse_mean=result['cv_rmse_mean'],
                            warnings=sum(len(f['warnings']) for f in result['folds'])))
        print(json.dumps(results[-1]), flush=True)
    summary = dict(status='PASS', purpose=settings['purpose'], run_id=run_id, python=sys.version,
                   packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy']},
                   train_sha256=manifest['partitions']['train']['sha256'],
                   raw_sha256=manifest['raw_sha256'], config_sha256=hashlib.sha256(raw_config).hexdigest(),
                   total_seconds=time.perf_counter() - started, results=results)
    summary['source_sha256'] = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((root / 'src').rglob('*.py'))}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(str(out), flush=True)


if __name__ == '__main__':
    main()
