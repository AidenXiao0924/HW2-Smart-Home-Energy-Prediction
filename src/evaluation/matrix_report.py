"""Export research tables/figures exclusively from verified formal Training CV checkpoints."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.training.checkpoint import read_checkpoint, digest
from src.training.matrix import DEFAULT_OUTPUT, ROOT, verify_matrix
from src.features.engineering import CANDIDATE_FEATURES

LABELS=dict(select_k_best='SelectKBest',rfe='RFE',lasso='Lasso selection',permutation='Permutation',forward_sfs='Forward SFS')
METHODS=list(LABELS)
KS=['10','15','20','25','ALL']


def summarize_rows(folds):
    rows=[]
    for (kind, experiment), group in folds.groupby(['kind','experiment_id'],sort=False):
        sets=[set(json.loads(v)) for v in group.selected_features_json]
        jaccard=[len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
        first=group.iloc[0]
        rows.append(dict(kind=kind,experiment_id=experiment,candidate_set=first.candidate_set,
                         candidate_count=int(first.candidate_count),method=first.method,k=first.k,
                         effective_k=int(first.effective_k),classification=first.classification,
                         research_only=bool(first.research_only),folds=len(group),
                         cv_mae_mean=float(group.mae.mean()),cv_mse_mean=float(group.mse.mean()),
                         cv_rmse_mean=float(group.rmse.mean()),cv_rmse_std=float(group.rmse.std(ddof=0)),
                         cv_r2_mean=float(group.r2.mean()),cv_adjusted_r2_mean=float(group.adjusted_r2.mean()),
                         feature_jaccard_mean=float(np.mean(jaccard)),compute_seconds=float(group.compute_seconds.sum()),
                         estimator_fit_calls=int(group.estimator_fit_calls.sum()),
                         computed_folds=int((group.execution=='COMPUTED').sum()),reused_folds=int((group.execution=='REUSED_ALL').sum()),
                         warnings=int(group.warning_count.sum())))
    return pd.DataFrame(rows)


def table(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)): return f'{v:.4f}'
        return str(v).replace('|','/')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def export_report(folder=DEFAULT_OUTPUT, output=None):
    folder=Path(folder)
    output=Path(output) if output else ROOT/'reports/evaluation/b2b'
    output.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'reports/figures/b2b'
    figures.mkdir(parents=True,exist_ok=True)
    integrity=verify_matrix(folder)
    manifest=read_checkpoint(folder/'manifest.json')
    records, frequencies, checkpoints=[],[],[]
    for task in manifest['tasks']:
        path=folder/'folds'/f"{task['key']}.json"
        payload=read_checkpoint(path)
        result=payload['result']
        records.append(dict(kind=task['kind'],experiment_id=task['experiment_id'],configuration_sha256=task['config_sha256'],
                            dataset_sha256=payload['dataset_sha256'],split_sha256=payload['split_sha256'],
                            train_csv_sha256=manifest['identity']['train_sha256'],candidate_set=task['config']['candidate_set'],
                            candidate_count=result['candidate_count'],method=task['config']['method'],k=str(task['config']['k']),
                            effective_k=len(result['selected_features']),fold=task['fold'],classification=task['classification'],
                            research_only=task['research_only'],train_rows=result['train']['rows'],train_start=result['train']['start'],
                            train_end=result['train']['end'],cv_rows=result['outer_validation']['rows'],
                            cv_start=result['outer_validation']['start'],cv_end=result['outer_validation']['end'],
                            selected_features_json=json.dumps(result['selected_features']),**result['metrics'],
                            compute_seconds=result['seconds'],execution=payload['execution'],
                            estimator_fit_calls=result['actual_fit_counts'].get('estimator',0),
                            imputer_fit_calls=result['actual_fit_counts'].get('imputer',0),
                            scaler_fit_calls=result['actual_fit_counts'].get('scaler',0),
                            selector_fit_calls=result['actual_fit_counts'].get('selector',0),
                            warning_count=len(result['warnings']),warnings_json=json.dumps(result['warnings']),
                            peak_sampled_worker_rss_mib=result['memory']['peak_sampled_rss'] if result['memory'] else None,
                            checkpoint_payload_sha256=digest(payload),checkpoint_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            reused_from=task['source_task'],prediction_sha256=result['prediction_sha256']))
        checkpoints.append(dict(task=task['key'],path=str(path.relative_to(ROOT)).replace('\\','/'),
                                payload_sha256=digest(payload),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                execution=payload['execution'],source=task['source_task']))
    folds=pd.DataFrame(records)
    summary=summarize_rows(folds)
    matrix=summary[summary.kind=='matrix'].copy()
    baselines=summary[summary.kind!='matrix'].copy()
    fold_matrix=folds[folds.kind=='matrix'].copy()
    for _, group in fold_matrix.groupby('experiment_id',sort=False):
        counter=Counter(f for value in group.selected_features_json for f in json.loads(value))
        for feature in CANDIDATE_FEATURES:
            frequencies.append(dict(experiment_id=group.experiment_id.iloc[0],candidate_set=group.candidate_set.iloc[0],
                                    method=group.method.iloc[0],k=group.k.iloc[0],feature=feature,
                                    selected_folds=counter[feature],selection_frequency=counter[feature]/5))
    paired=matrix.pivot(index=['method','k'],columns='candidate_set',values='cv_rmse_mean').reset_index()
    paired['delta_28_minus_27_wh']=paired.research28-paired.deployable27
    by_method=matrix.groupby('method').agg(mean_config_rmse=('cv_rmse_mean','mean'),
                 std_config_rmse=('cv_rmse_mean',lambda s:s.std(ddof=0)),mean_fold_rmse_std=('cv_rmse_std','mean'),
                 compute_seconds=('compute_seconds','sum'),feature_jaccard_mean=('feature_jaccard_mean','mean')).reset_index()
    by_k=matrix.groupby('k',sort=False).agg(mean_config_rmse=('cv_rmse_mean','mean'),
             std_config_rmse=('cv_rmse_mean',lambda s:s.std(ddof=0)),compute_seconds=('compute_seconds','sum'),
             feature_jaccard_mean=('feature_jaccard_mean','mean')).reset_index()
    by_candidate=matrix.groupby('candidate_set').agg(mean_config_rmse=('cv_rmse_mean','mean'),
                     min_config_rmse=('cv_rmse_mean','min'),max_config_rmse=('cv_rmse_mean','max')).reset_index()
    for name,data in [('feature_selection_results',matrix),('cv_fold_results',fold_matrix),('baseline_cv_folds',folds[folds.kind!='matrix']),
                      ('baselines_summary',baselines),('method_summary',by_method),('k_summary',by_k),
                      ('candidate_summary',by_candidate),('candidate_paired_comparison',paired),
                      ('selection_frequency',pd.DataFrame(frequencies))]:
        data.to_csv(output/f'{name}.csv',index=False,lineterminator='\n')
    paths=[]
    def save(fig,name):
        for suffix in ['png','svg']:
            path=figures/f'{name}.{suffix}'
            fig.savefig(path,dpi=160,bbox_inches='tight')
            paths.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ax,candidate in zip(axes,['deployable27','research28']):
        values=matrix[matrix.candidate_set==candidate].pivot(index='method',columns='k',values='cv_rmse_mean').reindex(index=METHODS,columns=KS)
        image=ax.imshow(values.to_numpy(),aspect='auto',cmap='viridis')
        ax.set(xticks=range(5),xticklabels=KS,yticks=range(5),yticklabels=[LABELS[m] for m in METHODS],
               xlabel='Number of selected features (K)',title=candidate+' | Training CV RMSE (Wh)')
        for i in range(5):
            for j in range(5): ax.text(j,i,f'{values.iloc[i,j]:.1f}',ha='center',va='center',color='white',fontsize=9)
        fig.colorbar(image,ax=ax,label='Mean fold RMSE (Wh)')
    save(fig,'feature_selection_rmse')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,candidate in zip(axes,['deployable27','research28']):
        for method in METHODS:
            group=matrix[(matrix.candidate_set==candidate)&(matrix.method==method)].set_index('k').reindex(KS)
            ax.plot(KS,group.feature_jaccard_mean,marker='o',label=LABELS[method])
        ax.set(xlabel='K',ylabel='Mean pairwise fold Jaccard',ylim=(0,1.05),title=candidate+' | Feature stability')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    save(fig,'feature_stability')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for method in METHODS:
        group=fold_matrix[(fold_matrix.candidate_set=='deployable27')&(fold_matrix.method==method)&(fold_matrix.k=='15')]
        ax.plot(group.fold+1,group.rmse,marker='o',label=LABELS[method])
    for kind,label in [('baseline_mean','Mean baseline'),('baseline_full_mlr','Full 27-feature MLR')]:
        group=folds[(folds.kind==kind)&(folds.candidate_set=='deployable27')]
        ax.plot(group.fold+1,group.rmse,linestyle='--',marker='s',label=label)
    ax.set(xlabel='Chronological outer fold',ylabel='RMSE (Wh)',title='Training CV | 27 candidates, selected K=15')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'cv_fold_rmse')
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    pairs=paired.sort_values(['method','k'])
    labels=[LABELS[r.method]+' / '+r.k for r in pairs.itertuples()]
    ax.bar(range(len(pairs)),pairs.delta_28_minus_27_wh)
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(xticks=range(len(pairs)),xticklabels=labels,ylabel='RMSE(28) - RMSE(27), Wh',
           title='Matched Training-CV configurations | Negative means lower RMSE with lights')
    plt.setp(ax.get_xticklabels(),rotation=65,ha='right',fontsize=8)
    save(fig,'candidate_comparison')
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    cost=by_method.set_index('method').reindex(METHODS)
    ax.bar([LABELS[m] for m in METHODS],cost.compute_seconds)
    ax.set(ylabel='Observed fold compute seconds',title='Actual computation charged to each method | ALL aliases cost zero fits')
    plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
    save(fig,'compute_cost')
    representative=matrix[(matrix.candidate_set=='deployable27')&(matrix.k=='15')].set_index('method').reindex(METHODS).reset_index()
    text=['# B2-B formal Training-only CV results','',
          '50 configurations / 250 matrix outer-fold results, plus 20 baseline fold results. '
          'All use the verified Training partition only. No external Validation, Calibration or Test selection/evaluation was performed.',
          '', 'RMSE standard deviation uses ddof=0 across five chronological folds. These are CV results, not Test performance or a selected deployment model.',
          '', '## Fixed comparison: 27 candidates, K=15','',
          table(representative[['method','cv_rmse_mean','cv_rmse_std','feature_jaccard_mean','compute_seconds']]),
          '', '## Mean baseline and full-feature MLR','',
          table(baselines[['kind','candidate_set','cv_rmse_mean','cv_rmse_std','cv_r2_mean','compute_seconds']]),
          '', '## Candidate-set aggregate (25 configurations each)','',table(by_candidate),
          '', '## K aggregate (10 configurations each)','',table(by_k),
          '', '## Method aggregate (10 configurations each)','',table(by_method),
          '', 'Aggregate means include K=ALL identities; they are descriptive matrix summaries, not independent selector rankings. '
          'K=ALL stability of 1 is structurally guaranteed. K=10 is RESEARCH_ONLY; all 28-feature configurations are research controls. '
          'Lasso exact-K can include zero coefficient ties; inspect persisted nonzero_count and coefficients. '
          'Feature frequency, coefficient magnitude and permutation importance are not causal effects.',
          '', '## Computation provenance','',
          f"Computed checkpoints: {integrity['computed']}; explicitly reused ALL checkpoints: {integrity['reused']}. "
          'Matrix-specific totals are 210 computed + 40 aliases; baselines add 10 actual Mean fits + 10 Full-MLR aliases. '
          'ALL aliases preserve direct source checkpoint hashes, predictions and metrics, and report zero actual fits/time. '
          'Timings are computational attribution, not a fair uncached benchmark between algorithms. '
          'Full regression fits were rerun separately for every non-ALL K, with no cross-K search reuse.',
          '', '## Figures','']
    text += [f'![{path.stem}](../../figures/b2b/{path.name})' for path in paths if path.suffix=='.png']
    report_path=output/'B2B_TRAINING_CV_REPORT.md'
    report_path.write_text('\n'.join(text)+'\n',encoding='utf-8')
    metrics_dir=ROOT/'artifacts/metrics'
    metrics_dir.mkdir(parents=True,exist_ok=True)
    artifact=metrics_dir/'feature_selection_cv.json'
    artifact.write_text(json.dumps(dict(schema_version='1.0.0',split='training_time_series_cv',
                          identity_hash=manifest['identity_hash'],std_ddof=0,
                          configurations=matrix.to_dict(orient='records'),baselines=baselines.to_dict(orient='records')),
                          indent=2,allow_nan=False),encoding='utf-8')
    files=list(output.glob('*.csv'))+[report_path,artifact]+paths
    published=dict(status='TRAINING_CV_COMPLETE_NOT_VALIDATION_SELECTED',identity_hash=manifest['identity_hash'],
                   experiment_manifest_path=str((folder/'manifest.json').relative_to(ROOT)).replace('\\','/'),
                   experiment_manifest_file_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
                   dataset_sha256=manifest['identity']['raw_sha256'],split_sha256=manifest['identity']['training_split_sha256'],
                   matrix_configs=50,matrix_folds=250,baseline_folds=20,
                   integrity=integrity,checkpoint_index=checkpoints,
                   files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'research_manifest.json').write_text(json.dumps(published,indent=2),encoding='utf-8')
    print(table(representative[['method','cv_rmse_mean','cv_rmse_std','feature_jaccard_mean','compute_seconds']]))
    print(table(baselines[['kind','candidate_set','cv_rmse_mean','cv_rmse_std','cv_r2_mean']]))
    print(table(by_k));print(table(by_candidate));print(table(by_method))
    print(json.dumps(dict(report=str(report_path),figures=len(paths),configs=len(matrix),folds=len(fold_matrix),
                         computed=integrity['computed'],reused=integrity['reused']),indent=2))
    return published


if __name__=='__main__': export_report()
