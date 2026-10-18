"""B2-A only: finite allowlisted preflight configs, one isolated/checkpointed worker per fold."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from src.training.audit import boundary, compact_audit
from src.training.checkpoint import atomic_json, read_checkpoint, digest, run_lock
from src.training.config import ExperimentConfig
from src.training.cv import load_training
from src.training.pipeline import candidate_features, fit_pipeline
from src.training.telemetry import fit_counts, MemorySampler
from src.evaluation.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[2]


def log(folder, event):
    record = dict(at_utc=datetime.now(timezone.utc).isoformat(), **event)
    with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        # Flush each event so Ctrl+C retains completed progress records.
        import os
        os.fsync(stream.fileno())
    print(json.dumps(record), flush=True)


def run_identity(root, plan, manifest):
    sources = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root / 'src').rglob('*.py'))}
    return dict(schema_version='1.0.0', plan=plan, python=sys.version,
                packages={p: importlib.metadata.version(p) for p in
                          ['numpy', 'pandas', 'scikit-learn', 'scipy', 'threadpoolctl']},
                source_sha256=sources, train_sha256=manifest['partitions']['train']['sha256'],
                raw_sha256=manifest['raw_sha256'], training_rows=manifest['partitions']['train']['rows'])


def tasks_from_plan(plan):
    if plan['purpose'] != 'full_training_preflight_not_research_conclusion':
        raise ValueError('Only B2-A preflight purpose is enabled')
    if len(plan['experiments']) > 5 or plan['blas_threads'] != 1:
        raise ValueError('B2-A permits at most five configs and one BLAS thread')
    tasks = []
    seen = set()
    for entry in plan['experiments']:
        config = ExperimentConfig(method=entry['method'], k=entry['k'], candidate_set=entry['candidate_set'],
                                  seed=plan['seed'], outer_splits=plan['outer_splits'],
                                  inner_splits=plan['inner_splits'], permutation_repeats=plan['permutation_repeats'])
        config.validate()
        for fold in entry['folds']:
            if not isinstance(fold, int) or fold not in range(5):
                raise ValueError('Invalid outer fold')
            key = f'{config.identifier}-fold-{fold}'
            if key in seen:
                raise ValueError('Duplicate task')
            seen.add(key)
            tasks.append(dict(key=key, config=asdict(config), fold=fold))
    return tasks


def evaluate_fold(frame, config, fold, callback=None):
    X, y = frame[candidate_features(config)], frame.Appliances
    train, valid = list(TimeSeriesSplit(n_splits=5).split(X))[fold]
    started = time.perf_counter()
    with MemorySampler() as memory, fit_counts(callback) as telemetry, threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline = fit_pipeline(config, X.iloc[train], y.iloc[train])
            prediction = pipeline.predict(X.iloc[valid])
        selector = pipeline.named_steps['selector']
        result = dict(fold=fold, candidate_set=config.candidate_set, candidate_count=X.shape[1],
                      k=config.k, selected_features=selector.selected_features_,
                      deployment_eligible=config.deployment_eligible,
                      train=boundary(X.iloc[train]), outer_validation=boundary(X.iloc[valid]),
                      fit_components={name: step.fit_boundary_ for name, step in pipeline.named_steps.items()},
                      selector_audit=compact_audit(selector.audit_),
                      selector_contract=dict(method=config.method, alpha=config.selector_lasso_alpha,
                                             lasso=selector.lasso_metadata_, scores=selector.scores_,
                                             selection_path=selector.selection_path_,
                                             all_bypassed=(config.k == 'ALL')),
                      metrics=regression_metrics(y.iloc[valid], prediction,
                                                 0 if config.model == 'mean' else len(selector.selected_features_),
                                                 config.model in ['ridge', 'lasso']),
                      predictions=np.asarray(prediction, dtype=float).tolist(),
                      prediction_sha256=hashlib.sha256(np.asarray(prediction, dtype='<f8').tobytes()).hexdigest(),
                      warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])
    result.update(seconds=time.perf_counter() - started, memory=memory.report(),
                  actual_fit_counts=dict(telemetry['counts']), fit_shapes=dict(telemetry['shapes']),
                  completed_outer_folds=1, configured_outer_folds=5, configured_inner_folds=config.inner_splits,
                  executed_inner_splits=config.inner_splits if config.method in ['forward_sfs', 'permutation'] and config.k != 'ALL' else 0,
                  purpose='preflight_timing_not_research_conclusion')
    return result


def worker(folder, task_key):
    stored = read_checkpoint(folder / 'manifest.json')
    frame, manifest = load_training(ROOT)
    if digest(run_identity(ROOT, stored['identity']['plan'], manifest)) != stored['identity_hash']:
        raise ValueError('Worker environment/data/source differs from checkpoint identity')
    task = next(t for t in stored['tasks'] if t['key'] == task_key)
    config = ExperimentConfig(**task['config'])
    result = evaluate_fold(frame, config, task['fold'], lambda e: log(folder, dict(task=task_key, **e)))
    payload = dict(status='PASS', identity_hash=stored['identity_hash'], task=task, result=result)
    atomic_json(folder / 'folds' / f'{task_key}.json', payload)


def summarize(folder, stored):
    results, missing = [], []
    for task in stored['tasks']:
        path = folder / 'folds' / f"{task['key']}.json"
        if not path.exists():
            missing.append(task['key'])
            continue
        payload = read_checkpoint(path)
        if payload['identity_hash'] != stored['identity_hash'] or payload['task'] != task or payload['status'] != 'PASS':
            raise ValueError('Completed fold identity/task mismatch')
        results.append(payload)
    groups = {}
    for payload in results:
        config = ExperimentConfig(**payload['task']['config'])
        result = payload['result']
        group = groups.setdefault(config.identifier, dict(config=asdict(config), candidate_count=result['candidate_count'],
                                  completed_fold_indices=[], seconds=0., actual_fit_counts={}, folds=[], peak_sampled_rss_mib=0.))
        group['completed_fold_indices'].append(result['fold'])
        group['seconds'] += result['seconds']
        group['peak_sampled_rss_mib'] = max(group['peak_sampled_rss_mib'], result['memory']['peak_sampled_rss'])
        for name, count in result['actual_fit_counts'].items():
            group['actual_fit_counts'][name] = group['actual_fit_counts'].get(name, 0) + count
        group['folds'].append(dict(fold=result['fold'], rmse=result['metrics']['rmse'],
                                  features=result['selected_features'], memory=result['memory'],
                                  lasso=result['selector_contract']['lasso'], warnings=result['warnings']))
    summary = dict(status='PASS' if not missing else 'NOT_RUN', purpose='preflight_not_research_conclusion',
                   identity_hash=stored['identity_hash'], training_rows=stored['identity']['training_rows'],
                   completed_folds=len(results), planned_folds=len(stored['tasks']), missing_tasks=missing,
                   groups=groups, total_fold_compute_seconds=sum(g['seconds'] for g in groups.values()))
    atomic_json(folder / 'summary.json', summary)
    return summary


def run_preflight(root, plan_path, folder, max_new_folds=None):
    root, folder = Path(root), Path(folder).resolve()
    plan = json.loads(Path(plan_path).read_text())
    tasks = tasks_from_plan(plan)
    _, manifest = load_training(root)
    identity = run_identity(root, plan, manifest)
    with run_lock(folder):
        path = folder / 'manifest.json'
        if path.exists():
            stored = read_checkpoint(path)
            if stored['identity_hash'] != digest(identity) or stored['tasks'] != tasks:
                raise ValueError('Resume refused: config/source/data/environment identity changed')
        else:
            stored = dict(identity=identity, identity_hash=digest(identity), tasks=tasks)
            atomic_json(path, stored)
        log(folder, dict(event='run_start_or_resume', identity_hash=stored['identity_hash']))
        # Validate every existing checkpoint BEFORE executing any new fits.
        summary = summarize(folder, stored)
        new = 0
        try:
            for task in tasks:
                if (folder / 'folds' / f"{task['key']}.json").exists():
                    log(folder, dict(event='fold_resume_skip', task=task['key']))
                    continue
                if max_new_folds is not None and new >= max_new_folds:
                    log(folder, dict(event='controlled_pause', completed_new_folds=new))
                    break
                log(folder, dict(event='fold_start', task=task['key'], config=task['config']))
                tick = time.perf_counter()
                process = subprocess.Popen([sys.executable, '-m', 'src.training.preflight',
                                            '--worker', task['key'], '--output', str(folder)], cwd=root)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                if code:
                    raise RuntimeError(f'Worker failed with exit {code}: {task["key"]}')
                summary = summarize(folder, stored)
                log(folder, dict(event='fold_checkpointed', task=task['key'], worker_wall_seconds=time.perf_counter()-tick))
                new += 1
        except BaseException as exc:
            log(folder, dict(event='interrupted_or_error', error=repr(exc), traceback=traceback.format_exc()))
            summarize(folder, stored)
            raise
        summary = summarize(folder, stored)
        log(folder, dict(event='run_exit', completed=summary['completed_folds'], planned=summary['planned_folds']))
        return summary


def main():
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != ROOT / '.venv':
        raise RuntimeError('Official Python 3.11 .venv required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/b2a_preflight.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/experiments/b2a-preflight-v1')
    parser.add_argument('--max-new-folds', type=int)
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.worker)
    else:
        run_preflight(ROOT, args.plan, args.output, args.max_new_folds)


if __name__ == '__main__':
    main()
