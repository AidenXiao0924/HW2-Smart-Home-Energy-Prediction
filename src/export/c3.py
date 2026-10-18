"""Consolidate frozen research. Only new statistical calculation is Train+Validation VIF."""
import argparse
from copy import deepcopy
import json
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.training.c1 import ROOT, OUT as C1, MODEL, ARTIFACT, FIXTURES, C1Gate, decision
from src.training.b3_io import sha, write_once, now
from src.training.checkpoint import read_checkpoint
from src.evaluation.c2 import OUT as C2, forbid_model_updates
from src.export.reference import FEATURES, validate_artifact
from src.export.c3_contracts import schemas, validate, public_safe, external_row_errors
from src.calibration.conformal import interval

PUBLIC=ROOT/'artifacts/deployment/mlr-1.0.0'
PRIVATE=ROOT/'reports/evaluation/c3-mlr-1.0.0'


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def csv(path):return pd.read_csv(path,float_precision='round_trip',keep_default_na=False)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def verify_sources():
    decision()
    paths=[ROOT/'reports/evaluation/b2b/research_manifest.json',ROOT/'reports/evaluation/b3/evaluation_manifest.json',
           C1/'manifest.json',C2/'evaluation_manifest.json',C2/'research_manifest.json']
    for path in paths:
        record=read_checkpoint(path) if path.parent in [C1,C2] else load_json(path)
        for key in ['files','inputs','outputs']:
            for name,expected in record.get(key,{}).items():
                if sha(ROOT/name)!=expected:raise ValueError('Source artifact mismatch: '+name)
    start=read_checkpoint(C1/'run_start.json')
    if sha(ROOT/'reports/data/split_manifest.json')!=start['split_manifest_sha256']:
        raise ValueError('Split manifest mismatch')
    for name,expected in start['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Historical C1 source modified')
    return paths


def training_vif(model,metadata):
    with C1Gate('fit') as gate:
        frames=[]
        for name in ['train','validation']:
            path=ROOT/f'data/processed/{name}.csv'
            if sha(path)!=metadata['partitions'][name]['sha256']:raise ValueError('VIF partition hash mismatch')
            frames.append(csv(path))
        frame=pd.concat(frames,ignore_index=True)
        if len(frame)!=15787:raise ValueError('Unexpected VIF rows')
        X=frame[FEATURES].to_numpy(dtype=float)
        prep=model['preprocessing']
        X=np.where(np.isnan(X),prep['imputer']['statistics'],X)
        X=(X-np.asarray(prep['scaler']['mean']))/np.asarray(prep['scaler']['scale'])
        if not np.isfinite(X).all():raise ValueError('Nonfinite VIF design')
        exog=np.column_stack([np.ones(len(X)),X])
        result=dict(split='TRAIN_VALIDATION',rows=len(frame),start=frame.date.iloc[0],end=frame.date.iloc[-1],
            feature_order=FEATURES,design='C1 frozen median imputation and standardization; with intercept',
            design_rank=int(np.linalg.matrix_rank(exog)),design_columns=exog.shape[1],
            vif=[dict(feature=f,vif=float(variance_inflation_factor(exog,i+1))) for i,f in enumerate(FEATURES)],
            input_hashes={n:metadata['partitions'][n]['sha256'] for n in ['train','validation']},
            data_reads=gate.reads,
            limitations='Auxiliary VIF regressions are diagnostics only. No predictive model fitting, feature selection or coefficient updates.')
    return result


def build(public=PUBLIC,private=PRIVATE):
    public,private=Path(public),Path(private)
    if (private/'consolidation_manifest.json').exists():
        m=read_checkpoint(private/'consolidation_manifest.json')
        for name,value in m['outputs'].items():
            if sha(ROOT/name)!=value:raise ValueError('Consolidated artifact changed')
        return m
    if public.exists() or (private/'run_start.json').exists():
        raise RuntimeError('Partial/versioned export exists: preserve evidence instead of overwriting')
    with C1Gate('none'),forbid_model_updates(),patch('joblib.load',side_effect=RuntimeError('C3 cannot load Python model')):
        manifest_paths=verify_sources()
        model=load_json(ARTIFACT);validate_artifact(model)
        metadata=load_json(ROOT/'reports/data/split_manifest.json')
        quality=load_json(ROOT/'reports/data/quality_report.json')
        attribution=load_json(ROOT/'configs/c3_attribution.json')
        cv=csv(ROOT/'reports/evaluation/b2b/feature_selection_results.csv');cv['k']=cv.k.astype(str)
        baseline=csv(ROOT/'reports/evaluation/b2b/baselines_summary.csv')
        validation=csv(ROOT/'reports/evaluation/b3/validation_metrics.csv')
        test=read_checkpoint(C2/'test_metrics.json')
        diagnostics=read_checkpoint(C2/'diagnostics.json')
        calibration=read_checkpoint(C1/'calibration.json')
        locked=read_checkpoint(ROOT/'reports/experiments/b3-validation-v1/candidate_lock.json')
        selected=read_checkpoint(ROOT/'reports/experiments/b3-validation-v1/mlr_decision.json')
        p=csv(C2/'predictions.csv')
        if quality['raw_sha256']!=model['dataset_version'] or metadata['raw_sha256']!=model['dataset_version']:
            raise ValueError('Dataset identity mismatch')
        if len(cv)!=50 or len(validation)!=12 or len(p)!=2962:raise ValueError('Incomplete source results')
    write_once(private/'run_start.json',dict(at_utc=now(),scope='CONSOLIDATION_ONLY',source_model_sha256=sha(ARTIFACT)))
    with forbid_model_updates(),patch('joblib.load',side_effect=RuntimeError('No model load')):
        vif=training_vif(model,metadata)
    write_once(private/'train_validation_vif.json',vif)
    pd.DataFrame(vif['vif']).to_csv(private/'train_validation_vif.csv',index=False)
    with C1Gate('none'):
        sources={}
        def source(path):
            path=Path(path);relative=str(path.relative_to(ROOT)).replace('\\','/')
            ref=dict(path=relative,sha256=sha(path));sources[relative]=ref['sha256'];return ref
        def block(split,path,values,transformation='Direct extraction of persisted results; no predictive model execution'):
            return dict(split=split,source=source(path),transformation=transformation,values=values)
        # Public model is a documented metadata projection, with identical inference parameters.
        deploy=deepcopy(model)
        deploy['training_metadata']={k:model['training_metadata'][k] for k in
            ['rows','start','end','partitions','python','sklearn','numpy','pandas','joblib','decision_sha256',
             'feature_selection_rerun','component_boundaries']}
        deploy['source_artifact_sha256']=sha(ARTIFACT)
        validate_artifact(deploy)
        public.mkdir(parents=True)
        dump(public/'model.json',deploy)
        schema_map=schemas()
        for name,value in schema_map.items():dump(public/f'schemas/{name}.schema.json',value)
        model_schema=load_json(ROOT/'artifacts/model/model.schema.json')
        dump(public/'schemas/model.schema.json',model_schema)
        validate(deploy,model_schema)
        units=dict(mae='Wh',mse='Wh^2',rmse='Wh',interval_width='Wh',r2='dimensionless',coverage='fraction')
        summary=dict(schema_version='1.0.0',model_version='mlr-1.0.0',units=units,
            dataset=block('ALL_RAW',ROOT/'reports/data/quality_report.json',
                {k:quality[k] for k in ['rows','columns','time_start','time_end','missing','duplicate_rows',
                'duplicate_timestamps','original_time_sorted','physical_range_violations','action','candidate_count','deployable_count','raw_sha256']}),
            splits=block('ALL_RAW',ROOT/'reports/data/split_manifest.json',metadata),
            feature_selection=block('TRAIN_CV',ROOT/'reports/evaluation/b2b/feature_selection_results.csv',
                dict(configurations=len(cv),outer_folds=250,methods=sorted(cv.method.unique().tolist()),
                    candidate_sets=['deployable27','research28'],k=['10','15','20','25','ALL'],
                    sd_definition='population across five outer folds; ddof=0',
                    research_only='K10 and all research28 configurations; ALL aliases preserve provenance')),
            training_cv=block('TRAIN_CV',ROOT/'reports/evaluation/b2b/baselines_summary.csv',
                dict(baselines=baseline.to_dict('records'),locked_candidates=locked['candidates'],
                    locked_candidates_source=source(ROOT/'reports/experiments/b3-validation-v1/candidate_lock.json'))),
            validation=block('VALIDATION',ROOT/'reports/evaluation/b3/validation_metrics.csv',
                dict(comparisons=validation.to_dict('records'),candidate_count=4,
                    protocol='Four locked schemes x three models; alpha search restricted to Training CV')),
            final_model=block('TRAIN_VALIDATION',C1/'final_fit.json',
                dict(model_version='mlr-1.0.0',estimator='LinearRegression',feature_order=FEATURES,
                    selection_method=selected['method'],k=selected['k'],training_rows=model['training_metadata']['rows'],
                    source_model_sha256=sha(ARTIFACT),public_model_sha256=sha(public/'model.json'),
                    decision_source=source(ROOT/'reports/experiments/b3-validation-v1/mlr_decision.json'))),
            calibration=block('CALIBRATION',C1/'calibration.json',
                {k:calibration[k] for k in ['n','k','q','alpha','start','end','residual_summary','limitations']}),
            test=block('TEST',C2/'test_metrics.json',dict(metrics=test['metrics'],prediction_interval=test['prediction_interval'],
                run_id=test['run_id'],q=test['q'],prediction_calls=read_checkpoint(C2/'evaluation_manifest.json')['test_prediction_calls'])),
            diagnostics=block('TEST',C2/'diagnostics.json',diagnostics),
            training_vif=block('TRAIN_VALIDATION',private/'train_validation_vif.json',
                {k:v for k,v in vif.items() if k!='data_reads'},'VIF auxiliary regressions on Train+Validation with C1 frozen transforms; no predictor fit'),
            attribution=attribution,limitations=[calibration['limitations'],diagnostics['limitations'],
                'High-energy Test peaks are often underestimated; residual Q-Q shows a pronounced right tail. '
                'Research CV, Validation and Test must remain distinctly labelled.'],
            lazy_chart_files=['charts/training_cv.json','charts/test_series.json','charts/test_residuals.json'])
        # CSV's empty alpha cell is represented explicitly as null, never a fabricated alpha.
        for row in summary['validation']['values']['comparisons']:
            if row['alpha']=='':row['alpha']=None
        validate(summary,schema_map['research-summary']);dump(public/'research_summary.json',summary)
        ref=source(C2/'predictions.csv')
        series=p.copy();series['hour']=pd.to_datetime(series.timestamp).dt.floor('h')
        series['covered']=(series.actual>=series.lower)&(series.actual<=series.upper)&series.interval_valid
        grouped=series.groupby('hour').agg(count=('actual','size'),actual_mean=('actual','mean'),
            actual_max=('actual','max'),prediction_mean=('prediction','mean'),lower_mean=('lower','mean'),
            upper_mean=('upper','mean'),covered_count=('covered','sum')).reset_index()
        grouped['timestamp']=grouped.pop('hour').dt.strftime('%Y-%m-%d %H:%M:%S')
        chart=dict(schema_version='1.0.0',model_version='mlr-1.0.0',split='TEST',source=ref,
            aggregation='source-local hourly; arithmetic means, actual maxima and exact counts',source_rows=len(p),rows=grouped.to_dict('records'))
        validate(chart,schema_map['test-series']);dump(public/'charts/test_series.json',chart)
        # Even temporal sample plus ALL negative estimates and ten largest absolute errors, labelled as sampled.
        indices=np.unique(np.r_[np.linspace(0,len(p)-1,400,dtype=int),np.flatnonzero(p.prediction<0),p.absolute_error.nlargest(10).index])
        scatter=p.iloc[indices][['timestamp','actual','prediction','residual','lower','upper','interval_valid','warnings']].to_dict('records')
        counts,edges=np.histogram(p.residual,bins=60)
        from scipy.stats import norm
        ordered=np.sort(p.residual.to_numpy());qq_indices=np.linspace(0,len(p)-1,101,dtype=int)
        chart=dict(schema_version='1.0.0',model_version='mlr-1.0.0',split='TEST',source=ref,source_rows=len(p),
            sampling='400 evenly spaced row indices union all negative estimates and top ten absolute errors; not representative for metric estimation',
            scatter=scatter,histogram=dict(edges=edges.tolist(),counts=counts.tolist()),
            qq=dict(theoretical=norm.ppf((qq_indices+.5)/len(p)).tolist(),ordered_residual=ordered[qq_indices].tolist()),
            residual_acf=diagnostics['residual_acf'])
        validate(chart,schema_map['test-residuals']);dump(public/'charts/test_residuals.json',chart)
        keys=['experiment_id','candidate_set','method','k','effective_k','cv_rmse_mean','cv_rmse_std','feature_jaccard_mean','compute_seconds','classification']
        chart=dict(schema_version='1.0.0',model_version='mlr-1.0.0',split='TRAIN_CV',
            source=source(ROOT/'reports/evaluation/b2b/feature_selection_results.csv'),
            aggregation='Five chronological outer folds; RMSE population SD (ddof=0)',configurations=cv[keys].to_dict('records'))
        validate(chart,schema_map['training-cv']);dump(public/'charts/training_cv.json',chart)
        original=load_json(FIXTURES)
        row=deepcopy(original['fixtures'][0]['input'])
        bads=[]
        for key,value,code in [('T1','invalid','NONFINITE_OR_TYPE'),('date','2016-02-30 08:10:00','INVALID_DATE'),
                               ('Hour',24,'TIME_FEATURE_MISMATCH')]:
            changed=deepcopy(row);changed[key]=value;bads.append(dict(input=changed,expected_code=code))
        changed=deepcopy(row);del changed['T1'];bads.append(dict(input=changed,expected_code='MISSING_COLUMN'))
        fixtures=dict(schema_version='1.0.0',model_version='mlr-1.0.0',source=source(FIXTURES),
            source_model_sha256=sha(ARTIFACT),public_model_sha256=sha(public/'model.json'),
            tolerance=original['tolerance'],javascript_execution='NOT_RUN',fixtures=original['fixtures'],
            interval_boundary_cases=[dict(kind='FUNCTION_CONTRACT_ONLY_NOT_MODEL_SCORE',prediction=value,
                q=model['prediction_interval']['q'],expected=interval(value,model['prediction_interval']['q']))
                for value in [-1.0,-model['prediction_interval']['q']-1]],invalid_input_cases=bads)
        validate(fixtures,schema_map['parity-fixtures']);dump(public/'parity_fixtures.json',fixtures)
        for item in bads:
            error=external_row_errors(item['input']);validate(error,schema_map['external-error'])
            if error['errors'][0]['code']!=item['expected_code']:raise ValueError('Incorrect input error fixture')
        dump(public/'attribution.json',attribution)
        # Explicit publish allowlist; nothing in models/, data/ or reports/ is copied.
        public_files=sorted(p for p in public.rglob('*') if p.is_file())
        for path in public_files:public_safe(load_json(path))
        homepage=['research_summary.json','attribution.json']
        budget=dict(homepage_json_bytes=sum((public/p).stat().st_size for p in homepage),
                    total_public_bytes=sum(p.stat().st_size for p in public_files),
                    raw_dataset_rows_bundled=0,scatter_rows=len(scatter),hourly_test_points=len(grouped))
        if budget['homepage_json_bytes']>100_000 or budget['total_public_bytes']>500_000:
            raise ValueError('Public data budget exceeded')
        deployment=dict(schema_version='1.0.0',model_version='mlr-1.0.0',source_model_sha256=sha(ARTIFACT),
            inference=dict(model='model.json',fixtures='parity_fixtures.json',python_backend_required=False,joblib_required=False),
            load_plan=dict(initial=homepage,lazy=summary['lazy_chart_files'],what_if=['model.json']),
            budgets=budget,files={str(p.relative_to(public)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size) for p in public_files},
            source_manifests=[source(p) for p in manifest_paths],
            publication_rule='Publish only this allowlisted folder; never publish repository root, data/, models/, reports/ or private upload data')
        public_safe(deployment);dump(public/'manifest.json',deployment)
        sources.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                        [ARTIFACT,FIXTURES,ROOT/'configs/c3_attribution.json',ROOT/'artifacts/model/model.schema.json']})
        outputs=[*public.rglob('*.json'),private/'train_validation_vif.json',private/'train_validation_vif.csv',private/'run_start.json']
        manifest=dict(status='COMPLETE',scope='CONSOLIDATION_NO_RETRAIN_NO_TEST_REEVALUATION',model_version='mlr-1.0.0',
            at_utc=now(),source_manifests=[str(p.relative_to(ROOT)).replace('\\','/') for p in manifest_paths],
            sources=sources,outputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in outputs},
            source_code={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),ROOT/'src/export/c3_contracts.py']},
            packages={n:version(n) for n in ['jsonschema','numpy','pandas','statsmodels']},
            budgets=budget,predictive_fit_calls=0,model_prediction_calls=0,test_dataset_reads=0,
            new_statistic='TRAIN_VALIDATION_VIF_ONLY',data_reads=vif['data_reads'])
        write_once(private/'consolidation_manifest.json',manifest)
        return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--public',type=Path,default=PUBLIC);parser.add_argument('--private',type=Path,default=PRIVATE)
    args=parser.parse_args();print(json.dumps(build(args.public,args.private),indent=2))
