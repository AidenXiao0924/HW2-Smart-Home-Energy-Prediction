"""B3: lock Training-CV shortlist, tune on Training, evaluate limited Validation, decide MLR scheme."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.features.engineering import DEPLOYABLE_FEATURES
from src.features.selection import TemporalFeatureSelector
from src.training.audit import boundary,validate_xy,processed_estimator,compact_audit
from src.training.b3_io import DatasetGate,write_once,protected_validation_load,sha,now
from src.training.checkpoint import read_checkpoint,digest,run_lock
from src.training.config import ExperimentConfig
from src.training.cv import load_training
from src.training.telemetry import fit_counts,track_fit
from src.evaluation.metrics import regression_metrics
from src.training.b3_models import FrozenColumns,model_pipeline

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'reports/experiments/b3-validation-v1'


def source_snapshot():
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}


def shortlist(table, protocol):
    valid=table[(table.candidate_set=='deployable27')&table.k.astype(str).isin(['15','20','25','ALL'])].copy()
    if len(valid)!=20: raise ValueError('Expected all 20 legal B2-B deployment configuration records')
    ranked=valid[valid.k.astype(str)!='ALL'].sort_values(
        ['cv_rmse_mean','cv_rmse_std','feature_jaccard_mean','compute_seconds','effective_k','experiment_id'],
        ascending=[True,True,False,True,True,True])
    selected=ranked.head(protocol['non_all_shortlist_count'])
    baseline=valid[(valid.method=='select_k_best')&(valid.k.astype(str)=='ALL')]
    if len(baseline)!=1 or len(selected)!=3:raise ValueError('Shortlist must contain three ranked schemes and one Full-27 MLR')
    rows=[]
    for position,row in enumerate(pd.concat([selected,baseline]).to_dict(orient='records')):
        rows.append(dict(experiment_id=row['experiment_id'],method=row['method'],k='ALL' if str(row['k'])=='ALL' else int(row['k']),
                         effective_k=int(row['effective_k']),candidate_set='deployable27',
                         cv_rmse_mean=float(row['cv_rmse_mean']),cv_rmse_std=float(row['cv_rmse_std']),
                         feature_jaccard_mean=float(row['feature_jaccard_mean']),compute_seconds=float(row['compute_seconds']),
                         rationale=f'Legal non-ALL CV rank {position+1}, lexicographic predeclared criteria' if position<3
                                   else 'Mandatory Full-Feature 27-feature MLR deployment baseline'))
    return rows


def lock_candidates(folder=FOLDER):
    folder=Path(folder)
    protocol_path=ROOT/'configs/b3_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    with DatasetGate(ROOT,'lock') as gate:
        manifest_path=ROOT/'reports/evaluation/b2b/research_manifest.json'
        research=json.loads(manifest_path.read_text())
        if not research['integrity']['status']=='PASS':raise ValueError('B2-B research integrity not PASS')
        table_path=ROOT/'reports/evaluation/b2b/feature_selection_results.csv'
        if sha(table_path)!=research['files']['reports/evaluation/b2b/feature_selection_results.csv']:
            raise ValueError('Training CV table hash mismatch')
        table=pd.read_csv(table_path,dtype={'k':str})
        candidates=shortlist(table,protocol)
        payload=dict(schema_version='1.0.0',locked_at_utc=now(),protocol=protocol,protocol_sha256=sha(protocol_path),
                     b2b_research_manifest_sha256=sha(manifest_path),b2b_cv_table_sha256=sha(table_path),
                     training_csv_sha256=json.loads((ROOT/'reports/data/split_manifest.json').read_text())['partitions']['train']['sha256'],
                     source_sha256=source_snapshot(),candidates=candidates,
                     packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib']},
                     python=sys.version,data_reads=gate.reads,
                     validation_not_opened=True,no_calibration_or_test=True)
        path=folder/'candidate_lock.json'
        if path.exists():
            previous=read_checkpoint(path)
            for key in ['protocol_sha256','b2b_cv_table_sha256','source_sha256','training_csv_sha256','candidates']:
                if previous[key]!=payload[key]:raise ValueError('Locked B3 plan changed; preserve this version and record any repair explicitly')
            return previous
        write_once(path,payload)
    print(json.dumps(dict(event='candidate_lock_created',candidates=candidates,validation_not_opened=True),indent=2),flush=True)
    return payload


def compact_components(pipeline,X):
    expected=boundary(X)
    result={}
    for name,step in pipeline.named_steps.items():
        if step.fit_boundary_!=expected:raise ValueError('Model preprocessing/estimator fit scope is not the provided Training scope')
        result[name]={k:v for k,v in step.fit_boundary_.items() if k!='timestamps'}
    return result


def select_features(X,y,candidate,protocol):
    validate_xy(X,y)
    selector=TemporalFeatureSelector(method=candidate['method'],k=candidate['k'],inner_splits=5,
               seed=protocol['seed'],lasso_alpha=protocol['selector_lasso_alpha'],permutation_repeats=protocol['permutation_repeats'],
               expected_features=tuple(DEPLOYABLE_FEATURES)).fit(X,y)
    if not 14<=len(selector.selected_features_)<=27:raise ValueError('Illegal selection size')
    return dict(features=selector.selected_features_,fit_scope=boundary(X),
                selector_audit=compact_audit(selector.audit_),scores=selector.scores_,selection_path=selector.selection_path_,
                source='Training-only; NEVER full-Training selection reused inside a tuning CV fold')


def get_or_write(path,producer):
    if path.exists():return read_checkpoint(path)
    value=producer()
    write_once(path,value)
    return value


def tune_scheme(frame,candidate,protocol,folder):
    X,y=frame[DEPLOYABLE_FEATURES],frame.Appliances
    rows=[]
    directory=Path(folder)/'training'/candidate['experiment_id']
    for fold,(train,valid) in enumerate(TimeSeriesSplit(n_splits=5).split(X)):
        selection_path=directory/f'fold-{fold}-selection.json'
        selection=get_or_write(selection_path,lambda:select_features(X.iloc[train],y.iloc[train],candidate,protocol))
        def evaluate():
            evaluations=[]
            fixed=[('mlr',1.)]+[('ridge',a) for a in protocol['ridge_alphas']]+[('lasso',a) for a in protocol['lasso_alphas']]
            for model,alpha in fixed:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    pipeline=model_pipeline(selection['features'],model,alpha,protocol).fit(X.iloc[train],y.iloc[train])
                    predicted=pipeline.predict(X.iloc[valid])
                evaluations.append(dict(model=model,alpha=alpha,fold=fold,features=selection['features'],
                    metrics=regression_metrics(y.iloc[valid],predicted,len(selection['features']),model!='mlr'),
                    fit_components=compact_components(pipeline,X.iloc[train]),
                    cv_scope=boundary(X.iloc[valid]),selection_checkpoint_sha256=sha(selection_path),
                    warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]))
            return dict(fold=fold,training_scope=boundary(X.iloc[train]),evaluations=evaluations,
                        preprocessing_policy='Selection shared only within this Training CV fold; every model/alpha refits its own imputer/scaler')
        record=get_or_write(directory/f'fold-{fold}-models.json',evaluate)
        rows+=record['evaluations']
        print(json.dumps(dict(event='training_tuning_fold',candidate=candidate['experiment_id'],fold=fold)),flush=True)
    summary=[]
    for model in ['mlr','ridge','lasso']:
        alphas=[1.] if model=='mlr' else protocol[model+'_alphas']
        for alpha in alphas:
            values=[r['metrics']['rmse'] for r in rows if r['model']==model and r['alpha']==alpha]
            if len(values)!=5:raise ValueError('Incomplete alpha CV')
            summary.append(dict(model=model,alpha=alpha,cv_rmse_mean=float(np.mean(values)),cv_rmse_std=float(np.std(values))))
    chosen={model:min([r for r in summary if r['model']==model],key=lambda r:(r['cv_rmse_mean'],r['alpha']))['alpha']
            for model in ['ridge','lasso']}
    result=dict(candidate=candidate,alpha_grid_summary=summary,chosen_alpha=chosen,
                selection_rule=protocol['alpha_selection'],validation_used=False)
    path=directory/'tuning.json'
    return get_or_write(path,lambda:result)


def train_candidates(folder=FOLDER):
    folder=Path(folder)
    lock=read_checkpoint(folder/'candidate_lock.json')
    if lock['source_sha256']!=source_snapshot():raise ValueError('Source differs from locked pre-Validation plan')
    if (folder/'training_manifest.json').exists():return read_checkpoint(folder/'training_manifest.json')
    protocol=lock['protocol']
    models=[]
    started=time.perf_counter()
    with DatasetGate(ROOT,'train') as gate,threadpool_limits(limits=1),fit_counts() as telemetry:
        frame,_=load_training(ROOT)
        if sha(ROOT/'data/processed/train.csv')!=lock['training_csv_sha256']:raise ValueError('Training hash differs from locked CV source')
        X,y=frame[DEPLOYABLE_FEATURES],frame.Appliances
        for candidate in lock['candidates']:
            tuning=tune_scheme(frame,candidate,protocol,folder)
            directory=folder/'training'/candidate['experiment_id']
            full=get_or_write(directory/'full-selection.json',lambda:select_features(X,y,candidate,protocol))
            for model in ['mlr','ridge','lasso']:
                alpha=1. if model=='mlr' else tuning['chosen_alpha'][model]
                key=candidate['experiment_id']+'-'+model
                record_path=directory/(model+'-train-fit.json')
                if record_path.exists():
                    record=read_checkpoint(record_path)
                    if sha(ROOT/record['artifact_path'])!=record['artifact_sha256']:raise ValueError('Existing Train artifact changed')
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        pipeline=model_pipeline(full['features'],model,alpha,protocol).fit(X,y)
                    transformed=pipeline[:-1].transform(X).to_numpy()
                    estimator=pipeline.named_steps['estimator'].estimator_
                    if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_):raise ValueError('Nonfinite trained coefficients')
                    artifact=ROOT/'models/offline/b3-train-v1'/(key+'.joblib')
                    artifact.parent.mkdir(parents=True,exist_ok=True)
                    if artifact.exists():raise FileExistsError('Orphan artifact preserved; record incident instead of overwriting')
                    joblib.dump(pipeline,artifact)
                    condition=float(np.linalg.cond(transformed))
                    record=dict(model_key=key,candidate=candidate,model=model,alpha=None if model=='mlr' else alpha,
                        estimator_class=type(estimator).__module__+'.'+type(estimator).__name__,features=full['features'],
                        artifact_path=str(artifact.relative_to(ROOT)).replace('\\','/'),artifact_sha256=sha(artifact),
                        fit_scope=boundary(X),fit_components=compact_components(pipeline,X),
                        selected_feature_checkpoint_sha256=sha(directory/'full-selection.json'),
                        tuning_checkpoint_sha256=sha(directory/'tuning.json'),
                        numerical_diagnostics=dict(rank=int(np.linalg.matrix_rank(transformed)),p=len(full['features']),
                                                   condition_number=condition if np.isfinite(condition) else None,
                                                   condition_status='FINITE' if np.isfinite(condition) else 'RANK_DEFICIENT',coefficients_finite=True),
                        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                        final_fit=False,fit_partition='TRAIN_ONLY')
                    write_once(record_path,record)
                models.append(record)
            print(json.dumps(dict(event='candidate_training_complete',candidate=candidate['experiment_id'],features=full['features'])),flush=True)
    result=dict(status='PASS',lock_sha256=sha(folder/'candidate_lock.json'),source_sha256=source_snapshot(),
                models=models,actual_fit_counts=dict(telemetry['counts']),data_reads=gate.reads,
                seconds=time.perf_counter()-started,validation_opened=False,final_fit=False)
    write_once(folder/'training_manifest.json',result)
    return result


def load_validation():
    split=json.loads((ROOT/'reports/data/split_manifest.json').read_text())['partitions']['validation']
    path=ROOT/'data/processed/validation.csv'
    if sha(path)!=split['sha256']:raise ValueError('Validation CSV hash mismatch')
    frame=pd.read_csv(path)
    frame['date']=pd.to_datetime(frame.date,format='%Y-%m-%d %H:%M:%S',errors='raise')
    frame=frame.set_index('date')
    if len(frame)!=split['rows'] or str(frame.index.min())!=split['start'] or str(frame.index.max())!=split['end']:
        raise ValueError('Validation boundary mismatch')
    validate_xy(frame[DEPLOYABLE_FEATURES],frame.Appliances)
    return frame,split


def evaluate_validation(folder=FOLDER,resume=False):
    folder=Path(folder)
    lock=read_checkpoint(folder/'candidate_lock.json')
    trained=read_checkpoint(folder/'training_manifest.json')
    if lock['source_sha256']!=source_snapshot():raise ValueError('Source changed since pre-Validation lock')
    if (folder/'evaluation_manifest.json').exists():return read_checkpoint(folder/'evaluation_manifest.json')
    ledger_path=folder/'validation_access.json'
    if ledger_path.exists():
        if not resume:raise RuntimeError('Partial Validation evaluation preserved; explicit --resume-validation records a resume event')
        write_once(folder/('resume-'+now().replace(':','-')+'.json'),dict(at_utc=now(),reason='Resume incomplete locked evaluation only; no new candidates or alpha search'))
    else:
        write_once(ledger_path,dict(started_at_utc=now(),lock_sha256=sha(folder/'candidate_lock.json'),
            training_manifest_sha256=sha(folder/'training_manifest.json'),
            allowed_evaluations=[m['model_key'] for m in trained['models']],planned_comparisons=len(trained['models']),
            policy='One predefined 4-scheme x 3-model comparison. No adaptation to Validation results.'))
    results=[]
    with DatasetGate(ROOT,'validation') as gate,threadpool_limits(limits=1),fit_counts() as telemetry:
        frame,split=protected_validation_load(ROOT,folder,load_validation)
        for record in trained['models']:
            path=folder/'validation'/(record['model_key']+'.json')
            if path.exists():
                result=read_checkpoint(path)
            else:
                artifact=ROOT/record['artifact_path']
                if sha(artifact)!=record['artifact_sha256']:raise ValueError('Train-only model hash differs before inference')
                pipeline=joblib.load(artifact)
                prediction=pipeline.predict(frame[DEPLOYABLE_FEATURES])
                result=dict(model_key=record['model_key'],candidate=record['candidate'],model=record['model'],alpha=record['alpha'],
                    features=record['features'],metrics=regression_metrics(frame.Appliances,prediction,len(record['features']),record['model']!='mlr'),
                    predictions=np.asarray(prediction,dtype=float).tolist(),actual=frame.Appliances.tolist(),
                    validation_scope=boundary(frame[DEPLOYABLE_FEATURES]),validation_sha256=split['sha256'],
                    lock_sha256=sha(folder/'candidate_lock.json'),train_artifact_sha256=record['artifact_sha256'],
                    prediction_sha256=hashlib.sha256(np.asarray(prediction,dtype='<f8').tobytes()).hexdigest(),
                    negative_estimate_count=int((prediction<0).sum()),evaluated_at_utc=now(),
                    metric_split='VALIDATION_NOT_TEST',fit_performed=False)
                write_once(path,result)
            results.append(result)
            print(json.dumps(dict(event='validation_evaluation_complete',model=record['model_key'],metrics=result['metrics'])),flush=True)
        if telemetry['counts']:raise ValueError('Any Fit during Validation inference is forbidden')
    manifest=dict(status='PASS',lock_sha256=sha(folder/'candidate_lock.json'),training_manifest_sha256=sha(folder/'training_manifest.json'),
                  validation_access_sha256=sha(ledger_path),evaluations=results,validation_fit_calls=dict(telemetry['counts']),
                  data_reads=gate.reads,comparisons=len(results),no_calibration_or_test=True)
    if len(results)!=12:raise ValueError('Expected only the twelve predeclared comparisons')
    write_once(folder/'evaluation_manifest.json',manifest)
    return manifest


def choose_mlr(evaluations,lock):
    legal=[]
    for result in evaluations:
        c=result['candidate']
        if c not in lock['candidates']:raise ValueError('Comparison contains a candidate outside the immutable shortlist')
        if result['model']=='mlr':
            if c['candidate_set']!='deployable27' or c['k']==10 or not 14<=len(result['features'])<=27:
                raise ValueError('Illegal MLR candidate')
            if not set(result['features'])<=set(DEPLOYABLE_FEATURES):raise ValueError('Forbidden deployment feature')
            legal.append(result)
    if len(legal)!=len(lock['candidates']):raise ValueError('Incomplete locked MLR comparison')
    return min(legal,key=lambda r:(r['metrics']['rmse'],r['candidate']['cv_rmse_std'],
                -r['candidate']['feature_jaccard_mean'],len(r['features']),r['candidate']['compute_seconds'],r['candidate']['experiment_id']))


def decide(folder=FOLDER):
    folder=Path(folder)
    lock=read_checkpoint(folder/'candidate_lock.json')
    evaluation=read_checkpoint(folder/'evaluation_manifest.json')
    if evaluation['lock_sha256']!=sha(folder/'candidate_lock.json'):raise ValueError('Selection lock mismatch')
    winner=choose_mlr(evaluation['evaluations'],lock)
    record=dict(status='MLR_SCHEME_SELECTED_FINAL_FIT_NOT_RUN',decided_at_utc=now(),
                planned_model_version=lock['protocol']['planned_model_version'],model_type='sklearn.linear_model.LinearRegression',
                method=winner['candidate']['method'],k=winner['candidate']['k'],features=winner['features'],feature_count=len(winner['features']),
                candidate_experiment_id=winner['candidate']['experiment_id'],validation_metrics=winner['metrics'],
                lock_sha256=sha(folder/'candidate_lock.json'),evaluation_manifest_sha256=sha(folder/'evaluation_manifest.json'),
                source_train_artifact_sha256=winner['train_artifact_sha256'],
                rule=lock['protocol']['final_mlr_order'],
                rationale='Minimum Validation RMSE among the four PRELOCKED legal MLR schemes. '
                          'CV stability, feature stability, size and cost are predeclared tie-breaks. '
                          'Ridge/Lasso are research comparisons only and cannot replace MLR. '
                          'Freeze this Training-selected feature list; later Final Fit must not rerun selection on Train+Validation.',
                final_fit='NOT_RUN',calibration='NOT_RUN',test_evaluation='NOT_RUN',website='NOT_RUN')
    write_once(folder/'mlr_decision.json',record)
    print(json.dumps(record,indent=2),flush=True)
    return record


def main():
    if sys.version_info[:2]!=(3,11) or Path(sys.prefix).resolve()!=ROOT/'.venv':raise RuntimeError('Official Python 3.11 .venv required')
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['lock','train','evaluate','decide'])
    parser.add_argument('--resume-validation',action='store_true')
    args=parser.parse_args()
    with run_lock(FOLDER):
        try:
            if args.phase=='lock':lock_candidates()
            elif args.phase=='train':train_candidates()
            elif args.phase=='evaluate':evaluate_validation(resume=args.resume_validation)
            else:decide()
        except BaseException as exc:
            write_once(FOLDER/('incident-'+now().replace(':','-')+'.json'),dict(at_utc=now(),phase=args.phase,error=repr(exc),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
