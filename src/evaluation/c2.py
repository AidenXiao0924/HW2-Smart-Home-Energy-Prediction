"""One-shot frozen Test inference. No predictive fitting or model selection."""
from contextlib import contextmanager
import json
import platform
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from src.training.c1 import ROOT, OUT as C1, MODEL, ARTIFACT, state_hash, decision
from src.training.b3_models import FrozenColumns
from src.training.audit import AuditedImputer, AuditedScaler, AuditedRegressor
from src.training.b3_io import DatasetGate, sha, write_once, now
from src.training.checkpoint import read_checkpoint
from src.training.telemetry import fit_counts
from src.export.reference import FEATURES, load_verified
from src.calibration.conformal import interval
from src.evaluation.metrics import regression_metrics

OUT = ROOT/'reports/evaluation/c2-mlr-1.0.0'
RUN_ID = 'c2-mlr-1.0.0-test-v1'


class TestGate(DatasetGate):
    __test__ = False
    def __init__(self, enabled=False):
        super().__init__(ROOT,'lock')
        self.phase='c2-test' if enabled else 'c2-no-dataset'
        self.allowed={ROOT/'data/processed/test.csv'} if enabled else set()


@contextmanager
def forbid_model_updates():
    from contextlib import ExitStack
    attempts=[]
    def blocked(*args,**kwargs):
        attempts.append('FORBIDDEN_MODEL_UPDATE')
        raise RuntimeError('Frozen Test evaluation forbids model updates')
    with ExitStack() as stack:
        for cls in [Pipeline,SimpleImputer,StandardScaler,LinearRegression,
                    FrozenColumns,AuditedImputer,AuditedScaler,AuditedRegressor]:
            for name in ['fit','fit_transform','partial_fit']:
                if hasattr(cls,name):stack.enter_context(patch.object(cls,name,blocked))
        yield attempts


def preflight():
    with TestGate() as gate:
        d, decision_hash=decision()
        manifest=read_checkpoint(C1/'manifest.json')
        if manifest['model_version']!='mlr-1.0.0' or manifest['test_read'] is not False:
            raise ValueError('Invalid C1 provenance')
        for name,expected in manifest['files'].items():
            if sha(ROOT/name)!=expected:raise ValueError('C1 artifact changed: '+name)
        a=load_verified(ROOT,ARTIFACT,C1/'manifest.json')
        calibration=read_checkpoint(C1/'calibration.json')
        start=read_checkpoint(C1/'run_start.json')
        for name,expected in start['source_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('C1 pinned source changed')
        metadata=json.loads((ROOT/'reports/data/split_manifest.json').read_text())
        if sha(ROOT/'reports/data/split_manifest.json')!=start['split_manifest_sha256']:
            raise ValueError('Split manifest changed')
        if a['dataset_version']!=metadata['raw_sha256']:
            raise ValueError('Dataset version mismatch')
        training=manifest['training']
        if (training['rows']!=15787 or training['decision_sha256']!=decision_hash
            or set(training['partitions'])!={'train','validation'}
            or training['feature_selection_rerun'] is not False
            or calibration['fit_calls'] or calibration['pipeline_sha256']!=sha(MODEL)
            or calibration['state_before_sha256']!=calibration['state_after_sha256']):
            raise ValueError('Invalid training/calibration isolation')
        for name in ['train','validation']:
            if training['partitions'][name]!=metadata['partitions'][name]:
                raise ValueError('Fit partition mismatch')
        if calibration['partition']!=metadata['partitions']['calibration']:
            raise ValueError('Calibration partition mismatch')
        if not training['end']<calibration['start']<=calibration['end']<metadata['partitions']['test']['start']:
            raise ValueError('Chronological overlap')
        pi=a['prediction_interval']
        if (pi['q']!=185.3571214361994 or pi['q']!=calibration['q']
            or pi['alpha']!=0.05 or pi['n']!=986 or pi['k']!=938
            or a['calibration_metadata']['record_sha256']!=sha(C1/'calibration.json')):
            raise ValueError('Frozen conformal mismatch')
        pipeline=joblib.load(MODEL)
        if (type(pipeline.named_steps['estimator'].estimator_) is not LinearRegression
            or list(pipeline.named_steps['features'].features)!=FEATURES
            or list(pipeline.named_steps['features'].expected)!=FEATURES):
            raise ValueError('Unexpected estimator or feature order')
        reg=pipeline.named_steps['estimator'].estimator_
        if reg.intercept_!=a['intercept'] or not np.array_equal(reg.coef_,a['coefficients']):
            raise ValueError('Pipeline/JSON coefficients mismatch')
        for name,key,attr in [('imputer','statistics','statistics_'),('scaler','mean','mean_'),('scaler','scale','scale_')]:
            if not np.array_equal(getattr(pipeline.named_steps[name],attr),a['preprocessing'][name][key]):
                raise ValueError('Pipeline/JSON preprocessing mismatch')
        if any(step.fit_boundary_['rows']!=15787 or step.fit_boundary_['end']!=training['end']
               for step in pipeline.named_steps.values()):
            raise ValueError('Unexpected pipeline fit boundary')
        hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                [MODEL,ARTIFACT,C1/'manifest.json',C1/'calibration.json',
                 ROOT/'reports/experiments/b3-validation-v1/mlr_decision.json']}
    # Hash verification reads bytes only; no CSV parsing or Target interpretation.
    with TestGate(True) as hash_gate:
        actual_hash=sha(ROOT/'data/processed/test.csv')
        if actual_hash!=metadata['partitions']['test']['sha256']:
            raise ValueError('Test CSV hash mismatch; no evaluation permitted')
    evidence=dict(verified_at_utc=now(),model_version='mlr-1.0.0',hashes=hashes,
        dataset_version=metadata['raw_sha256'],test_partition=metadata['partitions']['test'],
        test_sha256=actual_hash,preflight_data_reads=gate.reads,hash_only_reads=hash_gate.reads,
        test_target_parsed=False,q=pi['q'],feature_order=FEATURES,
        c1_manifest_sha256=sha(C1/'manifest.json'))
    return pipeline,a,evidence


def coverage_audit(frame):
    valid=frame.interval_valid.to_numpy(dtype=bool)
    covered=valid & (frame.actual.to_numpy()>=frame.lower.to_numpy()) & (frame.actual.to_numpy()<=frame.upper.to_numpy())
    widths=(frame.upper-frame.lower).to_numpy()[valid]
    return dict(total_rows=len(frame),valid_intervals=int(valid.sum()),invalid_intervals=int((~valid).sum()),
        covered_count=int(covered.sum()),coverage_valid=float(covered.sum()/valid.sum()) if valid.any() else None,
        coverage_all=float(covered.mean()),average_width_valid=float(widths.mean()) if len(widths) else None,
        median_width_valid=float(np.median(widths)) if len(widths) else None,
        below_lower_count_valid=int((valid & (frame.actual.to_numpy()<frame.lower.to_numpy())).sum()),
        above_upper_count_valid=int((valid & (frame.actual.to_numpy()>frame.upper.to_numpy())).sum()),
        negative_prediction_count=int((frame.prediction<0).sum()),
        negative_warning_count=int(frame.warnings.str.contains('NEGATIVE_ESTIMATE_WARNING',regex=False).sum()),
        invalid_warning_count=int(frame.warnings.str.contains('INVALID_INTERVAL',regex=False).sum()),
        below_above_denominator='VALID_INTERVALS_ONLY; coverage_all counts invalid intervals as not covered',
        coverage_scope='OBSERVED_TEST_NOT_FUTURE_GUARANTEE')


def evaluate():
    if (OUT/'evaluation_manifest.json').exists():
        m=read_checkpoint(OUT/'evaluation_manifest.json')
        for name,value in m['files'].items():
            if sha(ROOT/name)!=value:raise ValueError('Completed evaluation artifact modified')
        return m
    if (OUT/'first_evaluation.json').exists():
        raise RuntimeError('First Test attempt already recorded. Preserve incident; no silent reevaluation.')
    with forbid_model_updates(),fit_counts() as preflight_fits:
        pipeline,artifact,evidence=preflight()
    if preflight_fits['counts']:raise ValueError('Unexpected preflight model update')
    write_once(OUT/'first_evaluation.json',dict(run_id=RUN_ID,started_at_utc=now(),
        preflight=evidence,policy='ONE_SHOT; no retry/overwrite; reports may use saved predictions only',
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                       [Path(__file__),ROOT/'src/evaluation/metrics.py',ROOT/'src/calibration/conformal.py']}))
    try:
        before=state_hash(pipeline)
        with TestGate(True) as gate,forbid_model_updates() as attempts,fit_counts() as telemetry,threadpool_limits(limits=1):
            frame=pd.read_csv(ROOT/'data/processed/test.csv')
            dates=pd.to_datetime(frame.date,format='%Y-%m-%d %H:%M:%S')
            expected=evidence['test_partition']
            if (len(frame)!=2962 or len(frame)!=expected['rows'] or str(dates.iloc[0])!=expected['start']
                or str(dates.iloc[-1])!=expected['end'] or not dates.is_unique or not dates.is_monotonic_increasing
                or not np.array_equal(frame.Hour.to_numpy(),dates.dt.hour.to_numpy())):
                raise ValueError('Unexpected Test chronology/schema')
            X=frame.loc[:,FEATURES].copy();X.index=pd.DatetimeIndex(dates)
            predictions=pipeline.predict(X)  # Exactly one official Test prediction call.
            actual=frame.Appliances.to_numpy(dtype=float)
            metrics=regression_metrics(actual,predictions,p=15)
            intervals=[interval(float(p),artifact['prediction_interval']['q']) for p in predictions]
            saved=pd.DataFrame(dict(timestamp=dates.dt.strftime('%Y-%m-%d %H:%M:%S'),actual=actual,
                prediction=predictions,residual=actual-predictions,absolute_error=np.abs(actual-predictions),
                lower=[i['lower'] for i in intervals],upper=[i['upper'] for i in intervals],
                interval_valid=[i['interval_valid'] for i in intervals],warnings=['|'.join(i['warnings']) for i in intervals]))
            with (OUT/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
                saved.to_csv(stream,index=False,lineterminator='\n',float_format='%.17g')
            # Frozen transforms only. Store design so diagnostics never reopen Test.
            design=pipeline[:-1].transform(X)
            design.index.name='timestamp'
            with (OUT/'test_design.csv').open('x',encoding='utf-8',newline='') as stream:
                design.to_csv(stream,lineterminator='\n',float_format='%.17g')
            audit=coverage_audit(saved)
            if telemetry['counts'] or attempts or state_hash(pipeline)!=before:
                raise ValueError('Frozen model altered during Test')
        after={name:sha(ROOT/name) for name in evidence['hashes']}
        if after!=evidence['hashes']:raise ValueError('C1 artifacts changed')
        # Verify saved full-precision predictions, without another model call or Test read.
        restored=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip',keep_default_na=False)
        if regression_metrics(restored.actual,restored.prediction,15)!=metrics or coverage_audit(restored)!=audit:
            raise ValueError('Persisted prediction recomputation mismatch')
        write_once(OUT/'test_metrics.json',dict(run_id=RUN_ID,split='TEST',model_version='mlr-1.0.0',
            metrics=metrics,prediction_interval=audit,q=artifact['prediction_interval']['q']))
        with (OUT/'test_metrics.csv').open('x',encoding='utf-8',newline='') as stream:
            pd.DataFrame([dict(run_id=RUN_ID,split='TEST',**metrics,**audit)]).to_csv(stream,index=False)
        files=[OUT/n for n in ['first_evaluation.json','predictions.csv','test_design.csv','test_metrics.json','test_metrics.csv']]
        manifest=dict(status='COMPLETE',run_id=RUN_ID,completed_at_utc=now(),model_version='mlr-1.0.0',
            files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files},
            preflight=evidence,hashes_after=after,state_before=before,state_after=state_hash(pipeline),
            fit_counts=dict(telemetry['counts']),forbidden_update_attempts=attempts,test_prediction_calls=1,
            test_data_reads=gate.reads,metrics=metrics,prediction_interval=audit,q=artifact['prediction_interval']['q'],
            python=platform.python_version(),test_used_for_selection=False)
        write_once(OUT/'evaluation_manifest.json',manifest)
        return manifest
    except Exception as error:
        write_once(OUT/('incident-'+now().replace(':','-')+'.json'),dict(run_id=RUN_ID,error=repr(error),
            at_utc=now(),policy='Preserve first attempt. No automatic Test reevaluation.'))
        raise


if __name__=='__main__':print(json.dumps(evaluate(),indent=2))
