"""Verify Part A on Python 3.11 and compare preserved Python 3.12 evidence."""
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

from src.data.prepare import prepare


def main():
    root = Path(__file__).resolve().parents[2]
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != root / '.venv':
        raise RuntimeError('Run with the official project Python 3.11 .venv')
    modules = ['numpy', 'pandas', 'sklearn', 'scipy', 'statsmodels.api',
               'matplotlib', 'seaborn', 'joblib', 'pytest']
    for module in modules:
        importlib.import_module(module)
    packages = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'statsmodels',
                'matplotlib', 'seaborn', 'joblib', 'pytest']
    environment = dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
                       imports_verified=modules,
                       packages={p: importlib.metadata.version(p) for p in packages})
    (root / 'reports/environment/verification-311.json').write_text(
        json.dumps(environment, indent=2), encoding='utf-8')
    prepare(root)
    reports = root / 'reports/data'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    previous = load(reports / 'baseline-312/quality_report.json')
    current = load(reports / 'quality_report.json')
    old_split = load(reports / 'baseline-312/split_manifest.json')
    new_split = load(reports / 'split_manifest.json')
    keys = ['raw_sha256', 'rows', 'columns', 'time_start', 'time_end', 'candidate_features',
            'candidate_count', 'deployable_count', 'missing', 'duplicate_rows',
            'duplicate_timestamps', 'original_time_sorted', 'physical_range_violations',
            'interval_seconds_counts', 'dtypes']
    checks = {key: previous[key] == current[key] for key in keys}
    checks['complete_split_manifest_identical'] = old_split == new_split
    for name, partition in new_split['partitions'].items():
        actual = hashlib.sha256((root / f'data/processed/{name}.csv').read_bytes()).hexdigest()
        checks[f'{name}_export_hash_matches_manifest'] = actual == partition['sha256']
    ignores = (root / '.gitignore').read_text().splitlines()
    checks['required_gitignore_rules_present'] = all(
        rule in ignores for rule in ['.venv*/', 'data/raw/', 'data/processed/'])
    evidence = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                    note='Descriptive floating summary values may differ by numerical library version; not fitted parameters.',
                    python=sys.version, raw_sha256=current['raw_sha256'])
    (root / 'reports/tests/part-a-311-consistency.json').write_text(
        json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(environment, indent=2))
    print(json.dumps(evidence, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Cross-environment consistency failed; inspect evidence')


if __name__ == '__main__':
    main()
