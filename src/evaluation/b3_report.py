"""Validation evidence verification/reporting without reopening any dataset or refitting."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from src.training.b3 import ROOT,FOLDER,choose_mlr,source_snapshot
from src.training.b3_io import DatasetGate,sha
from src.training.checkpoint import read_checkpoint
from src.evaluation.metrics import regression_metrics


def verify(folder=FOLDER):
    folder=Path(folder)
    lock=read_checkpoint(folder/'candidate_lock.json')
    trained=read_checkpoint(folder/'training_manifest.json')
    evaluated=read_checkpoint(folder/'evaluation_manifest.json')
    ledger=read_checkpoint(folder/'validation_access.json')
    decision=read_checkpoint(folder/'mlr_decision.json')
    checks=dict(candidate_lock_source_unchanged=source_snapshot()==lock['source_sha256'],
                no_data_reads_during_preselection=lock['data_reads']==[],
                training_lock_hash=trained['lock_sha256']==sha(folder/'candidate_lock.json'),
                validation_training_manifest_hash=ledger['training_manifest_sha256']==sha(folder/'training_manifest.json'),
                evaluation_lock_hash=evaluated['lock_sha256']==sha(folder/'candidate_lock.json'),
                validation_fit_calls_empty=evaluated['validation_fit_calls']=={},
                twelve_fixed_comparisons=len(evaluated['evaluations'])==len(trained['models'])==12,
                alpha_grids_prelocked=lock['protocol']['ridge_alphas']==[.01,.1,1.,10.,100.,1000.] and lock['protocol']['lasso_alphas']==[.01,.1,1.,10.,100.],
                final_fit_not_run=decision['final_fit']=='NOT_RUN',
                decision_model_type=decision['model_type']=='sklearn.linear_model.LinearRegression')
    checks['runtime_data_whitelist']=all(r['path']=='data/processed/train.csv' for r in trained['data_reads']) and all(
        r['path'] in ['data/processed/train.csv','data/processed/validation.csv'] for r in evaluated['data_reads'])
    checks['candidate_count_four']=len(lock['candidates'])==4
    checks['no_illegal_candidate']=all(c['candidate_set']=='deployable27' and c['k'] in [15,20,25,'ALL'] for c in lock['candidates'])
    by_model={r['model_key']:r for r in trained['models']}
    actual_reference=None
    for result in evaluated['evaluations']:
        key=result['model_key']
        record=by_model[key]
        actual,pred=np.asarray(result['actual']),np.asarray(result['predictions'],dtype='<f8')
        if actual_reference is None:actual_reference=actual
        safe=np.array_equal(actual,actual_reference)
        safe &= record['features']==result['features'] and 14<=len(result['features'])<=27
        safe &= not set(result['features'])&{'lights','rv1','rv2','date','Appliances'}
        safe &= sha(ROOT/record['artifact_path'])==record['artifact_sha256']==result['train_artifact_sha256']
        safe &= hashlib.sha256(pred.tobytes()).hexdigest()==result['prediction_sha256']
        safe &= regression_metrics(actual,pred,len(result['features']),result['model']!='mlr')==result['metrics']
        safe &= record['fit_scope']['end']<result['validation_scope']['start']
        safe &= record['fit_scope']['rows']==13814 and result['validation_scope']['rows']==1973
        safe &= record['final_fit'] is False
        model=joblib.load(ROOT/record['artifact_path'])
        estimator=model.named_steps['estimator'].estimator_
        expected={'mlr':LinearRegression,'ridge':Ridge,'lasso':Lasso}[result['model']]
        safe &= type(estimator) is expected
        checks[key]=bool(safe)
    for candidate in lock['candidates']:
        directory=folder/'training'/candidate['experiment_id']
        tuning=read_checkpoint(directory/'tuning.json')
        if tuning['validation_used']:raise ValueError('Tuning used Validation')
        for fold in range(5):
            selection=read_checkpoint(directory/f'fold-{fold}-selection.json')
            models=read_checkpoint(directory/f'fold-{fold}-models.json')
            train_scope=selection['fit_scope']
            safe=models['training_scope']==train_scope
            for row in models['evaluations']:
                safe &= row['selection_checkpoint_sha256']==sha(directory/f'fold-{fold}-selection.json')
                safe &= row['features']==selection['features'] and train_scope['end']<row['cv_scope']['start']
                safe &= all(b['timestamp_sha256']==train_scope['timestamp_sha256'] for b in row['fit_components'].values())
            # Resolve each inner selection scope once to avoid repeating costly set construction.
            audit=selection['selector_audit'];allowed=set(train_scope['timestamps'])
            scope_sets={k:set(v['timestamps']) for k,v in audit['scopes'].items()}
            safe &= all(v<=allowed for v in scope_sets.values())
            for row in audit['records']:
                if 'train_scope' in row:
                    a,b=audit['scopes'][row['train_scope']],audit['scopes'][row['validation_scope']]
                    safe &= a['end']<b['start']
                    safe &= all(key==row['train_scope'] for key in row['components'].values())
            checks[candidate['experiment_id']+f'-tuning-fold-{fold}']=bool(safe)
        features=[r['features'] for r in trained['models'] if r['candidate']['experiment_id']==candidate['experiment_id']]
        checks[candidate['experiment_id']+'-same_features_three_models']=len(features)==3 and features[0]==features[1]==features[2]
        for model in ['ridge','lasso']:
            grid=[r for r in tuning['alpha_grid_summary'] if r['model']==model]
            checks[candidate['experiment_id']+'-'+model+'-alpha_selected_on_train']=tuning['chosen_alpha'][model]==min(grid,key=lambda r:(r['cv_rmse_mean'],r['alpha']))['alpha']
    winner=choose_mlr(evaluated['evaluations'],lock)
    checks['decision_matches_predeclared_rule']=decision['candidate_experiment_id']==winner['candidate']['experiment_id'] and decision['features']==winner['features']
    evidence=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,checked=len(checks),
                  candidate_lock_sha256=sha(folder/'candidate_lock.json'),evaluation_manifest_sha256=sha(folder/'evaluation_manifest.json'))
    if evidence['status']!='PASS':raise ValueError('B3 evidence verification failed')
    return evidence


def main():
    with DatasetGate(ROOT,'lock') as gate:
        evidence=verify()
        lock=read_checkpoint(FOLDER/'candidate_lock.json')
        evaluated=read_checkpoint(FOLDER/'evaluation_manifest.json')
        decision=read_checkpoint(FOLDER/'mlr_decision.json')
        rows=[];grid_rows=[]
        for row in evaluated['evaluations']:
            rows.append(dict(experiment_id=row['candidate']['experiment_id'],method=row['candidate']['method'],k=str(row['candidate']['k']),
                model=row['model'],alpha=row['alpha'],feature_count=len(row['features']),features_json=json.dumps(row['features']),
                **row['metrics'],artifact_sha256=row['train_artifact_sha256'],validation_sha256=row['validation_sha256'],
                negative_estimate_count=row['negative_estimate_count'],metric_split='VALIDATION_NOT_TEST'))
        for candidate in lock['candidates']:
            tune=read_checkpoint(FOLDER/'training'/candidate['experiment_id']/'tuning.json')
            grid_rows += [dict(experiment_id=candidate['experiment_id'],method=candidate['method'],k=str(candidate['k']),**r)
                          for r in tune['alpha_grid_summary']]
        out=ROOT/'reports/evaluation/b3';out.mkdir(parents=True,exist_ok=True)
        data=pd.DataFrame(rows)
        data.to_csv(out/'validation_metrics.csv',index=False,lineterminator='\n')
        pd.DataFrame(grid_rows).to_csv(out/'training_alpha_search.csv',index=False,lineterminator='\n')
        (out/'integrity.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
        labels=[c['method']+' / '+str(c['k']) for c in lock['candidates']]
        x=np.arange(4)
        for i,model in enumerate(['mlr','ridge','lasso']):
            values=[data[(data.experiment_id==c['experiment_id'])&(data.model==model)].rmse.iloc[0] for c in lock['candidates']]
            ax.bar(x+(i-1)*.25,values,width=.25,label=model.upper())
        ax.set(xticks=x,xticklabels=labels,ylabel='RMSE (Wh)',title='Locked candidate comparison | Validation, not Test')
        ax.legend();ax.grid(axis='y',alpha=.2)
        for ext in ['png','svg']:fig.savefig(out/('validation_rmse.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
        lines=['# B3 Validation evaluation and MLR scheme decision','',
               'Exactly four schemes were locked from Training CV before Validation was opened. '
               'All models fit 13,814 Training rows only; Validation has 1,973 rows. No Final Fit, Calibration or Test evaluation occurred.','',
               'Ridge/Lasso hyperparameters were chosen using five expanding Training CV folds. '
               'Each fold refit the feature selection scheme, including its nested preprocessing. '
               'Selection is shared only within that fold, while every alpha/model refits imputer/scaler. '
               'All three models use the same full-Training-selected feature list per scheme at Validation.','',
               '## Validation results','',
               '| Scheme | Model | Alpha | MAE Wh | MSE Wh² | RMSE Wh | R² | Adjusted R² |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            alpha='—' if r['alpha'] is None else str(r['alpha'])
            lines.append(f"| {r['method']} / {r['k']} | {r['model']} | {alpha} | {r['mae']:.4f} | {r['mse']:.4f} | {r['rmse']:.4f} | {r['r2']:.5f} | {r['adjusted_r2']:.5f} |")
        lines += ['', 'Traditional feature-count Adjusted R² for Ridge/Lasso is descriptive and is not an effective-degrees-of-freedom estimate.',
                  '', '## Decision','',f"Planned model version: `{decision['planned_model_version']}`. Selected: {decision['method']} / K={decision['k']}, {decision['feature_count']} features.",
                  '',decision['rationale'],'', 'Features: '+', '.join(decision['features']),
                  '', 'This is a scheme decision. Offline joblib artifacts are Training-only candidate fits, not the final Train+Validation model or frozen deployment artifacts.',
                  '', '![Validation comparison](validation_rmse.png)','',
                  f"Integrity: {evidence['checked']}/{evidence['checked']} checks true. Calibration/Test datasets were never opened by B3 phase code."]
        (out/'B3_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in FOLDER.rglob('*.json')}
        files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in out.iterdir() if p.is_file() and p.name!='evaluation_manifest.json'}
        manifest=dict(status='PASS',scope='VALIDATION_SELECTION_ONLY_FINAL_FIT_NOT_RUN',
                      locked_candidates=4,comparisons=12,planned_model_version=decision['planned_model_version'],
                      inputs=inputs,outputs=files,integrity=evidence,report_dataset_reads=gate.reads)
        (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(data[['method','k','model','alpha','mae','mse','rmse','r2','adjusted_r2']].to_string(index=False))
        print(json.dumps(decision,indent=2));print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
