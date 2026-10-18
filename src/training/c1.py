"""C1 only: fixed Train+Validation fit, then frozen Calibration. Never Test."""
import io
import json
import hashlib
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.training.b3_models import FrozenColumns
from src.training.audit import processed_estimator
from src.training.b3_io import DatasetGate, sha, write_once, now
from src.training.checkpoint import read_checkpoint, digest
from src.training.telemetry import fit_counts
from src.export.reference import FEATURES, predict, validate_artifact
from src.calibration.conformal import quantile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/experiments/c1-mlr-1.0.0'
MODEL = ROOT/'models/offline/mlr-1.0.0/pipeline.joblib'
ARTIFACT = ROOT/'artifacts/model/mlr-1.0.0/model.json'
FIXTURES = ROOT/'artifacts/parity/mlr-1.0.0/fixtures.json'
LIMITATION = ('Nominal 95% split conformal intervals assume exchangeable calibration and prediction scores. '
              'Chronological time-series dependence and distribution shift may violate exchangeability; '
              'no strict 95% coverage guarantee applies to future time segments. Nonnegative lower clipping '
              'changes interval interpretation. Calibration coverage is in-sample, not Test coverage.')


class C1Gate(DatasetGate):
    def __init__(self, phase):
        super().__init__(ROOT, 'train')
        self.phase = phase
        names = {'fit':['train','validation'], 'calibration':['calibration'], 'none':[]}[phase]
        self.allowed = {ROOT/f'data/processed/{name}.csv' for name in names}


def decision():
    folder = ROOT/'reports/experiments/b3-validation-v1'
    d = read_checkpoint(folder/'mlr_decision.json')
    lock = read_checkpoint(folder/'candidate_lock.json')
    if (d['planned_model_version'] != 'mlr-1.0.0' or d['features'] != FEATURES
        or d['method'] != 'forward_sfs' or d['k'] != 15 or d['feature_count'] != 15
        or d['model_type'] != 'sklearn.linear_model.LinearRegression'
        or d['lock_sha256'] != sha(folder/'candidate_lock.json')
        or d['evaluation_manifest_sha256'] != sha(folder/'evaluation_manifest.json')):
        raise ValueError('B3 frozen decision/version/hash mismatch')
    for name, expected in lock['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Historical B3 source modified: '+name)
    return d, sha(folder/'mlr_decision.json')


def load_partition(name, metadata):
    path = ROOT/f'data/processed/{name}.csv'
    m = metadata['partitions'][name]
    if sha(path) != m['sha256']:
        raise ValueError('Partition hash mismatch')
    frame = pd.read_csv(path)
    frame['date'] = pd.to_datetime(frame.date, format='%Y-%m-%d %H:%M:%S')
    frame = frame.set_index('date')
    if (len(frame) != m['rows'] or str(frame.index[0]) != m['start']
        or str(frame.index[-1]) != m['end'] or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing or frame.index.tz is not None):
        raise ValueError('Partition chronology mismatch')
    if not np.isfinite(frame.Appliances.to_numpy()).all():
        raise ValueError('Invalid target')
    if not np.array_equal(frame.Hour.to_numpy(), frame.index.hour):
        raise ValueError('Hour/calendar mismatch')
    return frame


def state_hash(pipeline):
    stream = io.BytesIO()
    joblib.dump(pipeline, stream, compress=0)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def run():
    if (OUT/'manifest.json').exists():
        manifest = read_checkpoint(OUT/'manifest.json')
        for name, value in manifest['files'].items():
            if sha(ROOT/name) != value: raise ValueError('Completed artifact modified')
        return manifest
    d, decision_hash = decision()
    metadata = json.loads((ROOT/'reports/data/split_manifest.json').read_text())
    # Exclusive start marker prevents silent refit/overwrite after a partial failure.
    write_once(OUT/'run_start.json', dict(started_at_utc=now(), decision_sha256=decision_hash,
        model_version='mlr-1.0.0', source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
        for p in (ROOT/'src').rglob('*.py')}, split_manifest_sha256=sha(ROOT/'reports/data/split_manifest.json')))
    with C1Gate('fit') as fit_gate, threadpool_limits(limits=1), fit_counts() as fits:
        train = load_partition('train', metadata)
        validation = load_partition('validation', metadata)
        if train.index[-1] >= validation.index[0]: raise ValueError('Overlapping fit partitions')
        frame = pd.concat([train,validation])
        X, y = frame.loc[:,FEATURES], frame.Appliances
        pipeline = Pipeline([('features',FrozenColumns(tuple(FEATURES),tuple(FEATURES))),
                             *processed_estimator(LinearRegression()).steps])
        pipeline.fit(X,y)
        model = pipeline.named_steps['estimator'].estimator_
        if type(model) is not LinearRegression: raise ValueError('Wrong estimator')
        MODEL.parent.mkdir(parents=True,exist_ok=True)
        with MODEL.open('xb') as stream: joblib.dump(pipeline,stream)
        if not np.array_equal(joblib.load(MODEL).predict(X.iloc[:10]),pipeline.predict(X.iloc[:10])):
            raise ValueError('Serialization mismatch')
        training = dict(rows=len(frame),start=str(frame.index[0]),end=str(frame.index[-1]),
            partitions={n:metadata['partitions'][n] for n in ['train','validation']},
            python=platform.python_version(), sklearn=__import__('sklearn').__version__,
            numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__,
            fits=dict(fits['counts']), data_reads=fit_gate.reads, decision_sha256=decision_hash,
            feature_selection_rerun=False,
            component_boundaries={n:{k:v for k,v in step.fit_boundary_.items() if k!='timestamps'}
                                  for n,step in pipeline.named_steps.items()})
    before = state_hash(pipeline)
    model_hash = sha(MODEL)
    write_once(OUT/'final_fit.json',dict(training=training,pipeline_sha256=model_hash,
        intercept=float(model.intercept_),coefficients=model.coef_.tolist(),features=FEATURES))
    with C1Gate('calibration') as calibration_gate, fit_counts() as calibration_fits:
        cal = load_partition('calibration',metadata)
        if frame.index[-1] >= cal.index[0]: raise ValueError('Calibration overlaps fit')
        predictions = pipeline.predict(cal.loc[:,FEATURES])
        scores = np.abs(cal.Appliances.to_numpy()-predictions)
        conformal = quantile(scores)
        if calibration_fits['counts'] or state_hash(pipeline)!=before or sha(MODEL)!=model_hash:
            raise ValueError('Calibration changed fitted state')
        calibration = dict(**conformal,start=str(cal.index[0]),end=str(cal.index[-1]),
            partition=metadata['partitions']['calibration'],data_reads=calibration_gate.reads,
            fit_calls=dict(calibration_fits['counts']),state_before_sha256=before,
            state_after_sha256=state_hash(pipeline),pipeline_sha256=model_hash,limitations=LIMITATION,
            residual_summary=dict(min=float(scores.min()),median=float(np.median(scores)),
                mean=float(scores.mean()),max=float(scores.max())),
            residuals=[dict(date=str(t),actual=float(y),predicted=float(p),absolute_residual=float(s))
                       for t,y,p,s in zip(cal.index,cal.Appliances,predictions,scores)])
        write_once(OUT/'calibration.json',calibration)
        imputer = pipeline.named_steps['imputer']
        scaler = pipeline.named_steps['scaler']
        artifact = dict(schema_version='1.0.0',model_version='mlr-1.0.0',
            dataset_version=metadata['raw_sha256'],model_type='LinearRegression',
            prediction_task='contemporaneous_estimation',target=dict(name='Appliances',unit='Wh'),
            feature_order=FEATURES,features=[dict(name=f,type='number',unit=(
                'degC' if f.startswith('T') else '%' if f.startswith('RH') else
                'm/s' if f=='Windspeed' else 'km' if f=='Visibility' else 'hour')) for f in FEATURES],
            intercept=float(model.intercept_),coefficients=model.coef_.tolist(),
            preprocessing=dict(imputer=dict(strategy='median',statistics=imputer.statistics_.tolist()),
                scaler=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),variance=scaler.var_.tolist()),
                time_features=dict(format='YYYY-MM-DD HH:mm:ss',timezone='source-local naive; no conversion',
                    Hour='integer hour 0-23',DayOfWeek='Monday=0, Sunday=6',IsWeekend='DayOfWeek>=5')),
            prediction_interval=dict(**conformal,method='split_conformal_absolute_residual',
                lower_rule='max(0,predicted-q)',upper_rule='predicted+q',
                negative_warning='NEGATIVE_ESTIMATE_WARNING',invalid_warning='INVALID_INTERVAL',limitations=LIMITATION),
            calibration_metadata=dict(rows=len(cal),start=str(cal.index[0]),end=str(cal.index[-1]),
                partition_sha256=metadata['partitions']['calibration']['sha256'],
                record_sha256=sha(OUT/'calibration.json')),
            training_metadata=training,
            input_contract=dict(required_columns=FEATURES[:-1]+['date'],missing_cells='null only; missing columns rejected',
                nonfinite='reject NaN/Infinity, numeric strings, booleans; JSON null is imputed',
                ordering='object key order ignored; reorder by feature_order; positional arrays rejected',
                precision='IEEE754 binary64, no rounding before prediction; parity tolerance 1e-6*max(1,abs(reference))'))
        validate_artifact(artifact)
        ARTIFACT.parent.mkdir(parents=True,exist_ok=True)
        with ARTIFACT.open('x',encoding='utf-8') as stream:
            json.dump(artifact,stream,indent=2,allow_nan=False)
        fixture_rows=[]
        for i in [0,1,2,10,50,100,200,500,985]:
            row={f:float(cal.iloc[i][f]) for f in FEATURES}
            row['date']=str(cal.index[i])
            fixture_rows.append(dict(kind='real_calibration_row',input=row,expected=predict(artifact,row)))
        row=dict(fixture_rows[0]['input']);row['T1']=None
        fixture_rows.append(dict(kind='real_row_with_explicit_missing_cell',input=row,expected=predict(artifact,row)))
        reference=np.array([predict(artifact,dict({f:float(r[f]) for f in FEATURES},date=str(t)))['predicted']
                            for t,r in cal.iterrows()])
        if not np.allclose(reference,predictions,rtol=1e-6,atol=1e-6):
            raise ValueError('JSON reference and Pipeline mismatch')
        FIXTURES.parent.mkdir(parents=True,exist_ok=True)
        with FIXTURES.open('x',encoding='utf-8') as stream:
            json.dump(dict(model_version='mlr-1.0.0',artifact_sha256=sha(ARTIFACT),
                javascript_execution='NOT_RUN_C1_FIXTURES_ONLY',tolerance='1e-6*max(1,abs(reference))',
                fixtures=fixture_rows),stream,indent=2,allow_nan=False)
    files=[MODEL,ARTIFACT,FIXTURES,OUT/'run_start.json',OUT/'final_fit.json',OUT/'calibration.json']
    manifest=dict(status='FROZEN_FINAL_FIT_AND_CALIBRATION_COMPLETE',model_version='mlr-1.0.0',
        created_at_utc=now(),files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files},
        training=training,calibration={k:v for k,v in calibration.items() if k!='residuals'},
        reference_max_abs_error=float(np.max(np.abs(reference-predictions))),
        test_read=False,test_evaluation='NOT_RUN',website='NOT_RUN',javascript_parity='FIXTURES_PREPARED_NOT_EXECUTED')
    write_once(OUT/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
