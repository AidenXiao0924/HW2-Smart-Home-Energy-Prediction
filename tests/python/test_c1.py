from copy import deepcopy
import json
import math
import joblib
import numpy as np
import pandas as pd
import pytest
from src.training.c1 import ROOT, OUT, MODEL, ARTIFACT, FIXTURES, C1Gate, decision, state_hash, run
from src.training.b3_io import sha
from src.training.checkpoint import read_checkpoint
from src.export.reference import FEATURES, predict, validate_artifact, load_verified
from src.calibration.conformal import quantile, interval


@pytest.fixture
def artifact():
    return load_verified(ROOT, ARTIFACT, OUT/'manifest.json')


def test_frozen_order_and_decision(artifact):
    d, value = decision()
    assert d['features'] == artifact['feature_order'] == FEATURES
    assert value == artifact['training_metadata']['decision_sha256']
    assert len(FEATURES) == 15 and not {'lights','rv1','rv2'} & set(FEATURES)


@pytest.mark.parametrize('phase,name', [('fit','calibration'),('fit','test'),('calibration','train'),
    ('calibration','validation'),('calibration','test'),('none','test'),('fit','raw')])
def test_phase_read_boundary(phase,name):
    path = ROOT/('data/raw/energydata_complete.csv' if name=='raw' else f'data/processed/{name}.csv')
    with C1Gate(phase), pytest.raises(PermissionError):
        path.read_bytes()


def test_fit_statistics_come_only_from_train_validation(artifact):
    with C1Gate('fit'):
        frame = pd.concat([pd.read_csv(ROOT/f'data/processed/{name}.csv') for name in ['train','validation']])
    X = frame[FEATURES].to_numpy()
    statistics = np.nanmedian(X,axis=0)
    filled = np.where(np.isnan(X), statistics, X)
    np.testing.assert_allclose(statistics,artifact['preprocessing']['imputer']['statistics'])
    np.testing.assert_allclose(filled.mean(axis=0),artifact['preprocessing']['scaler']['mean'])
    np.testing.assert_allclose(filled.std(axis=0),artifact['preprocessing']['scaler']['scale'])
    training = artifact['training_metadata']
    assert training['rows'] == 15787
    assert training['feature_selection_rerun'] is False
    assert training['fits'] == dict(frozen_columns=1,imputer=1,scaler=1,estimator=1)
    assert all(b['rows']==15787 and b['end']=='2016-04-30 08:00:00'
               for b in training['component_boundaries'].values())


def test_calibration_order_statistic_and_unchanged_model():
    c = read_checkpoint(OUT/'calibration.json')
    scores = [abs(r['actual']-r['predicted']) for r in c['residuals']]
    assert quantile(scores) == {k:c[k] for k in ['n','k','q','alpha']}
    assert c['n']==986 and c['k']==938
    assert c['fit_calls']=={}
    assert c['state_before_sha256']==c['state_after_sha256']==sha(MODEL)
    assert c['start'] > '2016-04-30 08:00:00'


def test_finite_quantile_not_interpolated():
    result=quantile(range(20))
    assert result['k']==20 and result['q']==19
    with pytest.raises(ValueError,match='Insufficient'):quantile([1,2])


@pytest.mark.parametrize('scores', [[],[float('nan')],[-1],[float('inf')]])
def test_quantile_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):quantile(scores)


def test_negative_warning_invalid_interval_and_zero_lower():
    assert interval(-2,5)==dict(predicted=-2,lower=0,upper=3,interval_valid=True,warnings=['NEGATIVE_ESTIMATE_WARNING'])
    invalid=interval(-6,5)
    assert invalid['predicted']==-6 and invalid['lower']==0 and invalid['upper']==-1
    assert not invalid['interval_valid'] and 'INVALID_INTERVAL' in invalid['warnings']
    assert interval(2,5)['lower']==0


def test_pipeline_json_serialization_and_frozen_inference(artifact,monkeypatch):
    from sklearn.linear_model import LinearRegression
    from src.training.audit import AuditedImputer,AuditedScaler,AuditedRegressor
    def forbidden(*args,**kwargs):raise AssertionError('Inference cannot Fit')
    for cls in [LinearRegression,AuditedImputer,AuditedScaler,AuditedRegressor]:
        monkeypatch.setattr(cls,'fit',forbidden)
    fixtures=json.loads(FIXTURES.read_text())
    with C1Gate('none'):
        pipeline=joblib.load(MODEL)
        before=state_hash(pipeline)
        for fixture in fixtures['fixtures']:
            row=fixture['input']
            X=pd.DataFrame([{f:np.nan if row[f] is None else row[f] for f in FEATURES}],
                index=pd.DatetimeIndex([row['date']]))
            expected=float(pipeline.predict(X)[0])
            output=predict(json.loads(json.dumps(artifact,allow_nan=False)),row)
            assert abs(output['predicted']-expected)<=1e-6*max(1,abs(expected))
            assert output==fixture['expected']
        assert state_hash(pipeline)==before


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),'12',True])
def test_nonfinite_or_non_numeric_input_rejected(artifact,bad):
    row=json.loads(FIXTURES.read_text())['fixtures'][0]['input'];row['T1']=bad
    with pytest.raises(ValueError):predict(artifact,row)


def test_missing_columns_calendar_and_object_order(artifact):
    row=json.loads(FIXTURES.read_text())['fixtures'][0]['input']
    assert predict(artifact,row)==predict(artifact,dict(reversed(list(row.items()))))
    missing=dict(row);del missing['T1']
    with pytest.raises(ValueError):predict(artifact,missing)
    bad=dict(row);bad['Hour']=24
    with pytest.raises(ValueError):predict(artifact,bad)
    bad=dict(row);bad['date']='2016-04-30T08:10:00Z'
    with pytest.raises(ValueError):predict(artifact,bad)


@pytest.mark.parametrize('field,value', [('schema_version','2.0.0'),('model_version','other'),
    ('coefficients',[1]),('intercept',float('nan')),('feature_order',list(reversed(FEATURES)))])
def test_invalid_artifact_blocked(artifact,field,value):
    artifact[field]=value
    with pytest.raises(ValueError):validate_artifact(artifact)


def test_manifest_hashes_and_no_test_reads():
    with C1Gate('none'):
        m=read_checkpoint(OUT/'manifest.json')
        for path,value in m['files'].items():assert sha(ROOT/path)==value
        assert m['test_read'] is False and m['test_evaluation']=='NOT_RUN'
        assert {r['path'] for r in m['training']['data_reads']}=={'data/processed/train.csv','data/processed/validation.csv'}
        assert {r['path'] for r in m['calibration']['data_reads']}=={'data/processed/calibration.csv'}
        assert run()==m  # Idempotent completed run, no refit or data read.


def test_manifest_tamper_rejected(artifact,tmp_path):
    fake=tmp_path/'model.json';fake.write_text(json.dumps(artifact))
    with pytest.raises(ValueError,match='hash'):
        load_verified(tmp_path,fake,OUT/'manifest.json')
