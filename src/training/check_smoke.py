"""Compare two real smoke runs without fitting or evaluating any model."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    summaries = [json.loads((folder / 'summary.json').read_text()) for folder in [args.first, args.second]]
    checks = {key: summaries[0][key] == summaries[1][key]
              for key in ['train_sha256', 'raw_sha256', 'config_sha256', 'packages']}
    first_ids, second_ids = [{r['experiment_id'] for r in s['results']} for s in summaries]
    checks['same_experiment_ids'] = first_ids == second_ids
    for name in sorted(first_ids & second_ids):
        x, y = [json.loads((folder / f'{name}.json').read_text()) for folder in [args.first, args.second]]
        checks[name] = (x['config'] == y['config'] and x['input_sha256'] == y['input_sha256']
                        and len(x['folds']) == len(y['folds']) == 5
                        and all(a['metrics'] == b['metrics'] and a['selected_features'] == b['selected_features']
                                for a, b in zip(x['folds'], y['folds'])))
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                  first=str(args.first), second=str(args.second),
                  scope='All fold metrics, selected features and input/config/package identity; timing excluded.')
    root = Path(__file__).resolve().parents[2]
    (root / 'reports/tests/part-b1-smoke-reproducibility.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise RuntimeError('Smoke reproducibility mismatch')


if __name__ == '__main__':
    main()
