"""Independent read-only external CSV QA: TRAIN prefix only, never formal Test evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
from src.export.reference import predict
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def test_genuine_training_prefix_provenance_and_reference_predictions():
    source=read('reports/tests/m3d/fixtures/manifest.json')
    assert source['source']=='data/processed/train.csv' and source['formal_test_rows_read']==0
    assert sha(source['source'])==source['source_sha256']
    for item in source['files']:assert sha(item['path'])==item['sha256']
    saved=read('reports/tests/m3d/actual-mode.json');model=read('artifacts/deployment/mlr-1.0.0/model.json')
    with (ROOT/'reports/tests/m3d/fixtures/actual.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(saved['results'])==32
    for i,(row,result) in enumerate(zip(rows,saved['results'])):
        named={k:v if k=='date' else float(v) for k,v in row.items()}
        expected=predict(model,named)
        assert result['row_number']==i+1 and result['actual']==float(row['Appliances'])
        for k in ['lower','upper']:assert abs(result[k]-expected[k])<=1e-6
        assert abs(result['predicted']-expected['predicted'])<=1e-6

def test_external_metrics_independently_recomputed_not_test_metrics():
    saved=read('reports/tests/m3d/actual-mode.json');rows=saved['results'];m=saved['evaluation'];n=len(rows)
    errors=[r['actual']-r['predicted'] for r in rows];mean=sum(r['actual'] for r in rows)/n
    expected=dict(mae=sum(abs(e) for e in errors)/n,mse=sum(e*e for e in errors)/n)
    expected['rmse']=math.sqrt(expected['mse']);expected['r2']=1-sum(e*e for e in errors)/sum((r['actual']-mean)**2 for r in rows)
    expected['adjusted_r2']=1-(1-expected['r2'])*(n-1)/(n-15-1)
    assert m['scope']=='EXTERNAL_USER_CSV_NOT_OFFICIAL_TEST'
    for key,value in expected.items():assert m[key]==pytest.approx(value,rel=1e-12,abs=1e-10)
    assert m['coverage_all']==sum(r['lower']<=r['actual']<=r['upper'] for r in rows)/n

def test_no_actual_mode_is_na_not_fabricated_accuracy():
    m=read('reports/tests/m3d/no-actual-mode.json')['evaluation']
    assert m['n']==0
    assert all(m[k] is None for k in ['mae','mse','rmse','r2','adjusted_r2','coverage_all','coverage_valid'])

def test_error_extension_schema_is_valid_without_modifying_c3():
    schema=read('web/public/contracts/csv-errors-1.1.0.schema.json');Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(dict(schema_version='1.1.0',status='INVALID_INPUT',errors=[dict(code='TIMEOUT',row=None,field=None,message='fixed')]))
    original=read('artifacts/deployment/mlr-1.0.0/schemas/external-error.schema.json')
    assert original['properties']['schema_version']['const']=='1.0.0'

def test_real_browser_cors_is_unmocked_and_does_not_rescore_research():
    report=read('reports/tests/m3d/real-network.json')
    assert not report['mocked'] and not report['security_bypass'] and report['inference_calls']==0
    assert report['failed']==0 and report['passed']==8
    assert len(report['probes'])==2
    for p in report['probes']:
        assert p['cross_origin'] and not p['mocked'] and p['source']['response_type']=='cors'
        assert p['bytes']<=10000000 and p['rows']<=20000 and p['inference_calls']==0

def test_20000_stress_is_labelled_repetition_and_max500_batch():
    report=read('reports/tests/m3d/stress.json')
    assert report['kind']=='REPEATED_REAL_TRAIN_ROW_CONTRACT_STRESS_NOT_NEW_DATASET'
    assert report['rows']==20000 and sum(report['batches'])==20000 and max(report['batches'])<=500

def test_download_preserves_negative_numbers_and_guards_text_formulas():
    with (ROOT/'reports/tests/m3d/boundary-export.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert float(rows[0]['Prediction_Wh'])<0 and not rows[0]['Prediction_Wh'].startswith("'")
    assert rows[0]['Original_note'].startswith("'=")
    assert rows[1]['Status']=='INVALID_INTERVAL' and rows[1]['Interval_Valid']=='false'

def test_frozen_engine_models_q_and_official_test_bytes_unchanged():
    prior=read('reports/web/m3c/public-assets.json')
    for f in prior['files']:
        if f['path'].startswith('js/inference/'):assert sha('web/'+f['path'])==f['sha256']
    c2=read('reports/evaluation/c2-mlr-1.0.0/evaluation_manifest.json')['payload']
    for p,h in {**c2['files'],**c2['hashes_after']}.items():assert sha(p)==h
    assert c2['test_prediction_calls']==1
