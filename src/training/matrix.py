"""Formal Training-only 50-config matrix, fold checkpoints and explicit ALL aliases."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.training.checkpoint import atomic_json, read_checkpoint, digest, run_lock
from src.training.config import ExperimentConfig
from src.training.cv import load_training
from src.training.pipeline import candidate_features
from src.training.preflight import evaluate_fold, log, run_identity
from src.features.selection import METHODS
from src.evaluation.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'reports/experiments/b2b-matrix-v1'


def config_from(plan, method, k, candidate, model='mlr'):
    return ExperimentConfig(method=method, k=k, candidate_set=candidate, model=model,
                            seed=plan['seed'], outer_splits=plan['outer_splits'],
                            inner_splits=plan['inner_splits'], permutation_repeats=plan['permutation_repeats'],
                            selector_lasso_alpha=plan['selector_lasso_alpha']).validate()


def classification(config, kind):
    if kind != 'matrix': return 'BASELINE_ONLY'
    if config.candidate_set == 'research28': return 'RESEARCH_CONTROL_LIGHTS'
    if config.k == 10: return 'RESEARCH_ONLY'
    return 'DEPLOYMENT_ELIGIBLE_CONFIGURATION_NOT_SELECTED'


def build_tasks(plan):
    if (plan['purpose'] != 'formal_training_feature_selection_cv' or plan['outer_splits'] != 5
        or plan['inner_splits'] != 5 or plan['blas_threads'] != 1 or plan['rmse_std_ddof'] != 0
        or set(plan['method_order']) != set(METHODS) or len(plan['method_order']) != 5
        or plan['candidate_sets'] != ['deployable27', 'research28'] or plan['k'] != [10,15,20,25,'ALL']):
        raise ValueError('Formal matrix requires exactly the approved 50 configurations and nested 5-fold CV')
    if plan['method_order'] != ['select_k_best','rfe','lasso','permutation','forward_sfs']:
        raise ValueError('Run inexpensive selectors before Permutation and Forward SFS')
    tasks = []
    def add(config, kind, batch, source=None):
        for fold in range(5):
            tasks.append(dict(key=f'{kind}-{config.identifier}-fold-{fold}', kind=kind,
                              batch=batch, config=asdict(config), config_sha256=digest(asdict(config)),
                              experiment_id=config.identifier, fold=fold,
                              classification=classification(config, kind),
                              research_only=(config.k == 10 or config.candidate_set == 'research28'),
                              source_task=f'matrix-{source.identifier}-fold-{fold}' if source else None))
    for candidate in plan['candidate_sets']:
        add(config_from(plan,'select_k_best','ALL',candidate,'mean'), 'baseline_mean', '00-mean-'+candidate)
    for position, method in enumerate(plan['method_order']):
        for candidate in plan['candidate_sets']:
            batch = f'{position+1:02d}-{candidate}-{method}'
            for k in plan['k']:
                source = config_from(plan,'select_k_best','ALL',candidate) if k == 'ALL' and method != 'select_k_best' else None
                add(config_from(plan,method,k,candidate), 'matrix', batch, source)
    for candidate in plan['candidate_sets']:
        config = config_from(plan,'select_k_best','ALL',candidate)
        add(config, 'baseline_full_mlr', '06-full-mlr-'+candidate, config)
    assert len([t for t in tasks if t['kind'] == 'matrix']) == 250
    return tasks


def identity_for(plan, split):
    identity = run_identity(ROOT, plan, split)
    identity.update(pdd_sha256=hashlib.sha256((ROOT/'docs/pdd/Final_PDD_v1.0.md').read_bytes()).hexdigest(),
                    training_split_sha256=digest(split['partitions']['train']))
    return identity


def all_equivalent(first, second):
    a, b = deepcopy(first), deepcopy(second)
    return (a.pop('method') in METHODS and b.pop('method') in METHODS and a == b
            and a['k'] == 'ALL' and a['model'] == 'mlr')


def validate_payload(payload, task, manifest):
    if (payload['status'] != 'PASS' or payload['task'] != task
        or payload['identity_hash'] != manifest['identity_hash']
        or payload['dataset_sha256'] != manifest['identity']['raw_sha256']
        or payload['split_sha256'] != manifest['identity']['training_split_sha256']):
        raise ValueError('Fold checkpoint identity mismatch')


def reuse_all(folder, task, manifest):
    started = time.perf_counter()
    source_task = next(t for t in manifest['tasks'] if t['key'] == task['source_task'])
    source_path = folder/'folds'/f"{source_task['key']}.json"
    source = read_checkpoint(source_path)
    validate_payload(source, source_task, manifest)
    if source['execution'] != 'COMPUTED' or not all_equivalent(source_task['config'], task['config']):
        raise ValueError('ALL alias must point directly to an equivalent actually-computed MLR fold')
    result = deepcopy(source['result'])
    result.update(actual_fit_counts={}, fit_shapes={}, fit_components={}, selector_audit=dict(records=[],scopes={}),
                  seconds=0., memory=None, purpose='formal_training_cv_explicit_equivalent_reuse')
    result['selector_contract']['method'] = task['config']['method']
    payload = envelope(task, manifest, result, 'REUSED_ALL')
    payload['equivalence'] = dict(source_task=source_task['key'], source_experiment_id=source_task['experiment_id'],
                                source_checkpoint_payload_sha256=digest(source),
                                source_checkpoint_file_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                source_prediction_sha256=source['result']['prediction_sha256'],
                                source_compute_seconds=source['result']['seconds'],
                                source_actual_fit_counts=source['result']['actual_fit_counts'],
                                rule='ALL makes every selector an identity transform; same raw columns, fold, imputer, scaler, MLR and hyperparameters. '
                                     'No estimator or selector fit was executed for this alias.',
                                equivalence_tests='tests/python/test_matrix.py::test_all_methods_numerically_identical',
                                alias_materialization_seconds=time.perf_counter()-started)
    atomic_json(folder/'folds'/f"{task['key']}.json", payload)


def envelope(task, manifest, result, execution):
    return dict(status='PASS', purpose='formal_training_cv_not_test_performance', execution=execution,
                identity_hash=manifest['identity_hash'], dataset_sha256=manifest['identity']['raw_sha256'],
                split_sha256=manifest['identity']['training_split_sha256'], task=task, result=result)


def worker(folder, key):
    manifest = read_checkpoint(folder/'manifest.json')
    frame, split = load_training(ROOT)
    if digest(identity_for(manifest['identity']['plan'], split)) != manifest['identity_hash']:
        raise ValueError('Worker source/data/environment changed; refusing to fit')
    task = next(t for t in manifest['tasks'] if t['key'] == key)
    if task['source_task']: raise ValueError('Aliases must not execute a worker')
    result = evaluate_fold(frame, ExperimentConfig(**task['config']), task['fold'],
                           lambda event: log(folder,dict(task=key,**event)))
    result['purpose'] = 'formal_training_cv_not_test_performance'
    atomic_json(folder/'folds'/f'{key}.json', envelope(task,manifest,result,'COMPUTED'))


def index_entry(path, payload):
    return dict(task=payload['task']['key'], kind=payload['task']['kind'], batch=payload['task']['batch'],
                experiment_id=payload['task']['experiment_id'], fold=payload['task']['fold'],
                execution=payload['execution'], seconds=payload['result']['seconds'],
                checkpoint_payload_sha256=digest(payload),
                checkpoint_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_matrix(folder=DEFAULT_OUTPUT, max_new_folds=None):
    folder = Path(folder).resolve()
    plan = json.loads((ROOT/'configs/b2b_matrix.json').read_text())
    tasks = build_tasks(plan)
    _, split = load_training(ROOT)
    identity = identity_for(plan,split)
    with run_lock(folder):
        path = folder/'manifest.json'
        if path.exists():
            manifest = read_checkpoint(path)
            if manifest['identity_hash'] != digest(identity) or manifest['tasks'] != tasks:
                raise ValueError('Resume refused: source, configuration, data or environment changed')
        else:
            manifest = dict(identity=identity,identity_hash=digest(identity),tasks=tasks,
                            expected_matrix_configs=50,expected_matrix_folds=250,expected_baseline_folds=20)
            atomic_json(path,manifest)
        completed = {}
        # Validate ALL existing checkpoints before performing any new work.
        for task in tasks:
            saved = folder/'folds'/f"{task['key']}.json"
            if saved.exists():
                payload = read_checkpoint(saved)
                validate_payload(payload,task,manifest)
                completed[task['key']] = index_entry(saved,payload)
        def persist():
            atomic_json(folder/'completion.json',dict(status='PASS' if len(completed)==len(tasks) else 'NOT_RUN',
                        identity_hash=manifest['identity_hash'],completed_matrix_folds=sum(t['kind']=='matrix' for t in completed.values()),
                        completed_baseline_folds=sum(t['kind']!='matrix' for t in completed.values()),
                        computed=sum(t['execution']=='COMPUTED' for t in completed.values()),
                        reused=sum(t['execution']=='REUSED_ALL' for t in completed.values()),
                        remaining=len(tasks)-len(completed),checkpoints=list(completed.values())))
        persist()
        log(folder,dict(event='matrix_start_or_resume',completed=len(completed),planned=len(tasks)))
        new, previous_batch = 0, None
        try:
            for task in tasks:
                if task['key'] in completed: continue
                if max_new_folds is not None and new >= max_new_folds:
                    log(folder,dict(event='controlled_pause',completed_new=new))
                    break
                if task['batch'] != previous_batch:
                    log(folder,dict(event='batch_start',batch=task['batch'],max_k_configs=5))
                    previous_batch=task['batch']
                log(folder,dict(event='fold_start',task=task['key'],method=task['config']['method'],k=task['config']['k'],
                               candidates=task['config']['candidate_set'],kind=task['kind']))
                tick=time.perf_counter()
                if task['source_task']:
                    reuse_all(folder,task,manifest)
                else:
                    logs=folder/'logs'
                    logs.mkdir(exist_ok=True)
                    with (logs/f"{task['key']}.log").open('a',encoding='utf-8') as stream:
                        process=subprocess.Popen([sys.executable,'-m','src.training.matrix','--worker',task['key'],
                                                  '--output',str(folder)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                        try:
                            while True:
                                try:
                                    code=process.wait(timeout=30)
                                    break
                                except subprocess.TimeoutExpired:
                                    log(folder,dict(event='fold_running',task=task['key'],elapsed_seconds=time.perf_counter()-tick))
                        except BaseException:
                            process.terminate()
                            process.wait()
                            raise
                    if code: raise RuntimeError(f'Worker exit {code}; see logs/{task["key"]}.log')
                saved=folder/'folds'/f"{task['key']}.json"
                payload=read_checkpoint(saved)
                validate_payload(payload,task,manifest)
                completed[task['key']]=index_entry(saved,payload)
                persist()
                log(folder,dict(event='fold_checkpointed',task=task['key'],execution=payload['execution'],
                                wall_seconds=time.perf_counter()-tick,matrix_done=sum(t['kind']=='matrix' for t in completed.values())))
                new+=1
        except BaseException as exc:
            log(folder,dict(event='interrupted_or_error',error=repr(exc),traceback=traceback.format_exc()))
            persist()
            raise
        persist()
        log(folder,dict(event='matrix_exit',completed=len(completed),planned=len(tasks)))


def verify_matrix(folder=DEFAULT_OUTPUT):
    folder=Path(folder)
    manifest=read_checkpoint(folder/'manifest.json')
    frame,split=load_training(ROOT)
    checks=dict(identity=digest(identity_for(manifest['identity']['plan'],split))==manifest['identity_hash'])
    tasks=manifest['tasks']
    allowed=set(frame.index.strftime('%Y-%m-%d %H:%M:%S'))
    fold_positions=list(TimeSeriesSplit(n_splits=5).split(frame))
    for task in tasks:
        saved=folder/'folds'/f"{task['key']}.json"
        payload=read_checkpoint(saved)
        validate_payload(payload,task,manifest)
        result=payload['result']
        config=ExperimentConfig(**task['config'])
        train,valid=fold_positions[task['fold']]
        train_ids=frame.index[train].strftime('%Y-%m-%d %H:%M:%S').tolist()
        valid_ids=frame.index[valid].strftime('%Y-%m-%d %H:%M:%S').tolist()
        safe=(result['train']['timestamps']==train_ids and result['outer_validation']['timestamps']==valid_ids)
        safe &= set(train_ids+valid_ids)<=allowed and max(train_ids)<min(valid_ids)
        predicted=np.asarray(result['predictions'],dtype='<f8')
        safe &= hashlib.sha256(predicted.tobytes()).hexdigest()==result['prediction_sha256']
        metrics=regression_metrics(frame.Appliances.iloc[valid],predicted,
                                   0 if config.model=='mean' else len(result['selected_features']))
        safe &= metrics==result['metrics']
        count=len(candidate_features(config)) if config.k=='ALL' else config.k
        safe &= len(result['selected_features'])==count and len(set(result['selected_features']))==count
        safe &= set(result['selected_features'])<=set(candidate_features(config))
        safe &= task['research_only']==(config.k==10 or config.candidate_set=='research28')
        if payload['execution']=='COMPUTED':
            safe &= not task['source_task']
            safe &= all(b['timestamps']==train_ids for b in result['fit_components'].values())
            audit=result['selector_audit']
            for record in audit['records']:
                safe &= all(set(audit['scopes'][key]['timestamps'])<=set(train_ids) for key in record['components'].values())
                if 'train_scope' in record:
                    inner_train=audit['scopes'][record['train_scope']]['timestamps']
                    inner_valid=audit['scopes'][record['validation_scope']]['timestamps']
                    safe &= set(inner_train+inner_valid)<=set(train_ids) and max(inner_train)<min(inner_valid)
                    safe &= all(audit['scopes'][key]['timestamps']==inner_train for key in record['components'].values())
            if config.k=='ALL': expected=dict(selector=1,imputer=1,scaler=1,estimator=1)
            elif config.method=='select_k_best': expected=dict(selector=1,imputer=2,scaler=2,estimator=1)
            elif config.method=='rfe': expected=dict(selector=1,imputer=2,scaler=2,estimator=len(candidate_features(config))-count+2,
                                                   rfe_estimator=len(candidate_features(config))-count+1)
            elif config.method=='lasso': expected=dict(selector=1,imputer=2,scaler=2,estimator=2)
            else:
                n=config.inner_splits if config.method=='permutation' else sum(len(candidate_features(config))-i for i in range(count))*config.inner_splits
                expected=dict(selector=1,imputer=n+1,scaler=n+1,estimator=n+1)
            safe &= result['actual_fit_counts']==expected
        else:
            source_path=folder/'folds'/f"{task['source_task']}.json"
            source=read_checkpoint(source_path)
            safe &= all_equivalent(task['config'],source['task']['config']) and source['execution']=='COMPUTED'
            safe &= digest(source)==payload['equivalence']['source_checkpoint_payload_sha256']
            safe &= hashlib.sha256(source_path.read_bytes()).hexdigest()==payload['equivalence']['source_checkpoint_file_sha256']
            safe &= result['predictions']==source['result']['predictions'] and result['metrics']==source['result']['metrics']
            safe &= result['actual_fit_counts']=={} and result['seconds']==0 and result['memory'] is None
        checks[task['key']]=bool(safe)
    matrix_tasks=[t for t in tasks if t['kind']=='matrix']
    checks['50_configs']=len({t['experiment_id'] for t in matrix_tasks})==50
    checks['250_matrix_folds']=len(matrix_tasks)==250
    checks['20_baseline_folds']=len(tasks)-len(matrix_tasks)==20
    completion=read_checkpoint(folder/'completion.json')
    checks['completion']=completion['status']=='PASS' and completion['remaining']==0
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                checked_folds=len(tasks),matrix_configs=50,matrix_folds=250,baseline_folds=20,
                computed=completion['computed'],reused=completion['reused'],identity_hash=manifest['identity_hash'])
    atomic_json(folder/'integrity.json',result)
    if result['status']!='PASS': raise ValueError('Matrix integrity/isolation/metrics verification failed')
    return result


def main():
    if sys.version_info[:2]!=(3,11) or Path(sys.prefix).resolve()!=ROOT/'.venv':
        raise RuntimeError('Official Python 3.11 .venv required')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--worker')
    parser.add_argument('--max-new-folds',type=int)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(args.output.resolve(),args.worker)
    elif args.verify: print(json.dumps(verify_matrix(args.output),indent=2))
    else: run_matrix(args.output,args.max_new_folds)


if __name__=='__main__': main()
