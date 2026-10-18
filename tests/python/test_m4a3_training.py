"""Independent TRAIN descriptive integrity checks, never re-evaluates Test."""
import json,hashlib,csv,math,runpy
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from jsonschema import Draft202012Validator
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
S=read('artifacts/training/train-v1/summary.json');R=read('artifacts/training/train-v1/ranges.json');X=pd.read_csv(ROOT/'data/processed/train.csv')[S['feature_order']]
def test_original_training_provenance_count_boundaries_and_engineering():
 d=pd.read_csv(ROOT/'data/processed/train.csv');p=S['provenance'];assert len(d)==p['rows']==13814;assert d.date.iloc[0]==p['start']=='2016-01-11 17:00:00';assert d.date.iloc[-1]==p['end']=='2016-04-16 15:10:00'
 assert sha(p['source'])==p['source_sha256']==read('reports/data/split_manifest.json')['partitions']['train']['sha256'];assert np.array_equal(pd.to_datetime(d.date).dt.hour,d.Hour);assert sha(p['engineering_source'])==p['engineering_sha256']
def test_histograms_exact_edges_counts_and_all_original_samples():
 for d in S['distributions']:
  values=X[d['feature']].dropna().to_numpy();counts,edges=np.histogram(values,bins=d['histogram']['edges']);assert counts.tolist()==d['histogram']['counts'];assert sum(counts)==d['valid_count']==13814-d['missing_count'];assert d['sample_count']==13814
  assert math.isclose(d['median'],float(np.median(values)),abs_tol=1e-12);assert math.isclose(d['mean'],float(np.mean(values)),abs_tol=1e-12)
def test_pearson_independent_numpy_symmetric_and_pairwise_policy():
 actual=np.array(S['correlation']['matrix']);np.testing.assert_allclose(actual,np.corrcoef(X.to_numpy().T),atol=1e-12);np.testing.assert_allclose(actual,actual.T,atol=1e-14);np.testing.assert_allclose(np.diag(actual),1);assert S['correlation']['pair_counts']==[[13814]*15 for _ in range(15)];assert 'no zero fill' in S['correlation']['missing_policy']
def test_training_ranges_exactly_train_not_train_validation_parameters():
 assert len(R['ranges'])==15
 for r in R['ranges']:assert r['min']==float(X[r['feature']].min());assert r['max']==float(X[r['feature']].max());assert r['valid_count']==13814
 assert R['provenance']['split']=='TRAIN' and R['provenance']['rows']!=15787
 assert R['feature_order']==read('artifacts/model/mlr-1.0.0/model.json')['feature_order']
def test_importance_existing_fold_mean_units_scope_and_original_sources():
 imp=S['importance'];assert imp['method']=='Permutation Importance' and imp['unit']=='Delta RMSE (Wh)' and imp['split']=='TRAIN_CV_INNER';scores=[]
 for f in imp['sources']:
  assert sha(f['source'])==f['sha256'];v=read(f['source'])['payload']['result'];assert v['candidate_set']=='deployable27' and v['k']==15;assert v['selector_contract']['method']=='permutation';assert v['train']['end']<=S['provenance']['end'];scores.append(v['selector_contract']['scores'])
 np.testing.assert_allclose(imp['mean'],np.mean(scores,axis=0),atol=1e-12);np.testing.assert_allclose(imp['std_outer_means'],np.std(scores,axis=0),atol=1e-12)
def test_exporter_no_fit_predict_and_no_other_split_reads():
 from sklearn.linear_model import LinearRegression
 original=pd.read_csv;seen=[]
 def only_train(path,*a,**k):
  assert Path(path).resolve()==(ROOT/'data/processed/train.csv').resolve();seen.append(str(path));return original(path,*a,**k)
 with patch.object(LinearRegression,'fit',side_effect=AssertionError('Forbidden Fit')),patch.object(LinearRegression,'predict',side_effect=AssertionError('Forbidden Predict')),patch('pandas.read_csv',side_effect=only_train):runpy.run_path(str(ROOT/'tools/export-training-analytics.py'))['export']()
 assert len(seen)==1
def test_all_new_json_schemas_and_public_hashes():
 schema=read('artifacts/training/train-v1/schema.json');Draft202012Validator.check_schema(schema);m=read('artifacts/training/train-v1/manifest.json');Draft202012Validator(schema['definitions']['manifest']).validate(m)
 for name,data in [('summary',S),('ranges',R)]:Draft202012Validator(schema['definitions'][name]).validate(data)
 for n,e in m['files'].items():assert sha('artifacts/training/train-v1/'+n)==e['sha256']==sha('web/public/training-v1/'+n)
def test_all_prior_frozen_research_and_candidates_unchanged():
 a=read('reports/web/m4a1/acceptance_manifest.json')
 for p,h in {**a['frozen_integrity']['verified_frozen_files'],**a['frozen_integrity']['c2_research_products']}.items():assert sha(p)==h
 for version in ['web-rc1','web-rc2']:
  assert sha('releases/'+version+'/release-manifest.json')==a['release_candidates'][version]['manifest_sha256']
  for f in read('releases/'+version+'/release-manifest.json')['files']:assert sha('releases/'+version+'/site/'+f['path'])==f['sha256']
 assert sha('output/pdf/HW2_CRISP_DM_Research_Report.pdf')==a['frozen_integrity']['academic_pdf_sha256']
 for p,h in a['frozen_integrity']['engine_unchanged_since_m3c'].items():assert sha('web/'+p)==h
