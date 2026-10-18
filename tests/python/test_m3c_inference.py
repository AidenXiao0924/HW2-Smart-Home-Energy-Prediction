"""Saved unit-fixture parity audit only. No raw Dataset/model fitting/Test evaluation."""
import hashlib
import json
from pathlib import Path
from src.export.reference import predict
from src.training.c1 import C1Gate
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
PUBLIC='artifacts/deployment/mlr-1.0.0/'

def test_saved_js_parity_meets_stricter_absolute_threshold():
    p=read('reports/tests/m3c/parity.json')
    assert p['status']=='PASS' and p['tolerance_absolute_wh']==1e-6
    assert p['original_python_fixtures']==10 and p['supplemental_python_cases']==7
    assert p['interval_function_cases']==2 and p['original_invalid_cases']==4
    assert all(v<=1e-6 for v in p['max_absolute_errors'].values())
    assert all(r['status']=='PASS' for r in p['rows']+p['rejected'])

def test_original_c1_c3_fixtures_and_model_bytes_unchanged():
    manifest=read(PUBLIC+'manifest.json');f=read(PUBLIC+'parity_fixtures.json')
    for name in ['model.json','parity_fixtures.json']:
        assert sha(PUBLIC+name)==manifest['files'][name]['sha256']
        assert sha('web/public/mlr-1.0.0/'+name)==sha(PUBLIC+name)
    assert sha(f['source']['path'])==f['source']['sha256']
    assert f['javascript_execution']=='NOT_RUN'  # historical C3 marker intentionally immutable

def test_supplemental_cases_are_labelled_and_match_python_reference():
    extra=read('reports/tests/m3c/supplemental-python-reference.json')
    model=read(PUBLIC+'model.json')
    assert extra['origin']=='M3C_SUPPLEMENTAL_REFERENCE_INFERENCE_ONLY_NOT_RESEARCH_METRICS'
    assert extra['raw_dataset_reads']==[] and extra['model_fits']==extra['formal_test_predictions']==0
    with C1Gate('none'):
        for f in extra['cases']:
            assert f['source']=='M3C_SUPPLEMENTAL_CONTRACT_CASE_NOT_EXISTING_C1_FIXTURE'
            assert predict(model,f['input'])==f['expected']

def test_each_saved_js_value_warning_and_validity_matches_source():
    p=read('reports/tests/m3c/parity.json')
    original=read(PUBLIC+'parity_fixtures.json')
    for i,f in enumerate(original['fixtures']):
        r=p['rows'][i]
        assert r['input']==f['input'] and r['expected']==f['expected']
    for r in p['rows']:
        assert r['actual']['warnings']==r['expected']['warnings']
        assert r['actual']['interval_valid']==r['expected']['interval_valid']
        for key in ['predicted','lower','upper']:
            assert abs(r['actual'][key]-r['expected'][key])<=1e-6

def test_negative_point_and_invalid_interval_audit():
    p=read('reports/tests/m3c/parity.json')
    negative=next(r for r in p['rows'] if r['name']=='negative_point_valid_interval')['actual']
    invalid=next(r for r in p['rows'] if r['name']=='negative_point_invalid_interval')['actual']
    assert negative['predicted']<0 and negative['interval_valid']
    assert invalid['upper']<invalid['lower'] and not invalid['interval_valid']
    assert invalid['warnings']==['NEGATIVE_ESTIMATE_WARNING','INVALID_INTERVAL']

def test_frozen_finalfit_calibration_and_formal_test_unchanged():
    c2=read('reports/evaluation/c2-mlr-1.0.0/evaluation_manifest.json')['payload']
    for name,expected in c2['hashes_after'].items():assert sha(name)==expected
    for name,expected in c2['files'].items():assert sha(name)==expected
    assert c2['test_prediction_calls']==1

def test_current_public_allowlist_stays_static_without_csv_backend():
    m=read('reports/web/m4a3/public-assets.json' if (ROOT/'reports/web/m4a3/public-assets.json').exists() else 'reports/web/m4a1/public-assets.json' if (ROOT/'reports/web/m4a1/public-assets.json').exists() else 'reports/web/m4a/public-assets.json' if (ROOT/'reports/web/m4a/public-assets.json').exists() else 'reports/web/m3d/public-assets.json' if (ROOT/'reports/web/m3d/public-assets.json').exists() else 'reports/web/m3c/public-assets.json')
    for f in m['files']:
        assert sha('web/'+f['path'])==f['sha256']
        assert not f['path'].endswith(('.joblib','.py','.ipynb'))
        assert not f['path'].endswith('.csv') or f['path']=='public/test-demo-v1/demo.csv'  # M4-A.1 explicitly authorizes bounded genuine Test Demo
    assert m['assignment_ready'] is False and m['deployment']=='DEPLOYMENT_PENDING'
