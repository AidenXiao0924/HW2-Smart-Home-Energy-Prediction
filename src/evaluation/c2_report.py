"""Diagnostics/reporting from persisted Test results, without model inference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from src.evaluation.c2 import ROOT, OUT, TestGate, coverage_audit
from src.evaluation.metrics import regression_metrics
from src.training.checkpoint import read_checkpoint
from src.training.b3_io import sha, write_once, now
from src.export.reference import FEATURES


def diagnose(predictions, design):
    residual=predictions.residual.to_numpy()
    X=design.loc[:,FEATURES].to_numpy()
    exog=sm.add_constant(X,has_constant='add')
    bp=het_breuschpagan(residual,exog,robust=True)
    vifs=[dict(feature=f,vif=float(variance_inflation_factor(exog,i+1))) for i,f in enumerate(FEATURES)]
    return dict(split='TEST',design='Frozen median imputation and StandardScaler; 15 features plus intercept',
        rows=len(predictions),durbin_watson=float(durbin_watson(residual)),
        breusch_pagan=dict(variant='Koenker robust=True',lm_statistic=float(bp[0]),lm_pvalue=float(bp[1]),
                          f_statistic=float(bp[2]),f_pvalue=float(bp[3])),
        vif=vifs,design_rank=int(np.linalg.matrix_rank(exog)),design_columns=exog.shape[1],
        residual_acf=acf(residual,nlags=40,fft=True).tolist(),
        limitations='DW and ACF describe ordered Test residual dependence. BP auxiliary OLS and VIF auxiliary '
        'regressions are statistical diagnostics, not predictive model training. BP p-values assume independent '
        'observations; serial dependence limits nominal interpretation. Q-Q is descriptive, not a normality guarantee. '
        'VIF includes an intercept, uses transformed TEST features, and cannot trigger reselection. '
        'Coefficients/correlations do not establish causal effects.')


def report():
    report_path=OUT/'research_manifest.json'
    if report_path.exists():
        m=read_checkpoint(report_path)
        for name,value in m['files'].items():
            if sha(ROOT/name)!=value:raise ValueError('Research output changed')
        return m
    with TestGate() as gate:
        m=read_checkpoint(OUT/'evaluation_manifest.json')
        for name,value in m['files'].items():
            if sha(ROOT/name)!=value:raise ValueError('Evaluation input changed')
        p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip',keep_default_na=False)
        X=pd.read_csv(OUT/'test_design.csv',float_precision='round_trip')
        if list(X.columns)!=['timestamp',*FEATURES] or not np.array_equal(X.timestamp,p.timestamp):
            raise ValueError('Diagnostic design not aligned')
        if regression_metrics(p.actual,p.prediction,15)!=m['metrics'] or coverage_audit(p)!=m['prediction_interval']:
            raise ValueError('Metrics not reproducible')
        d=diagnose(p,X)
        write_once(OUT/'diagnostics.json',d)
        pd.DataFrame(d['vif']).to_csv(OUT/'vif.csv',index=False)
        figures=OUT/'figures';figures.mkdir(exist_ok=True)
        def save(fig,name):
            for ext in ['png','svg']:fig.savefig(figures/f'{name}.{ext}',dpi=150,bbox_inches='tight')
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.scatter(p.actual,p.prediction,s=8,alpha=.35)
        lo=min(p.actual.min(),p.prediction.min());hi=max(p.actual.max(),p.prediction.max())
        ax.plot([lo,hi],[lo,hi],color='black',linestyle='--',label='Identity')
        ax.set(xlabel='Actual (Wh)',ylabel='Prediction (Wh)',title='Frozen MLR | Test');ax.legend();save(fig,'prediction_vs_actual')
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.scatter(p.prediction,p.residual,s=8,alpha=.35);ax.axhline(0,color='black',linestyle='--')
        ax.set(xlabel='Prediction (Wh)',ylabel='Actual - Prediction (Wh)',title='Test residual vs prediction');save(fig,'residual_vs_predicted')
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.hist(p.residual,bins=60);ax.set(xlabel='Actual - Prediction (Wh)',ylabel='Count',title='Test residual distribution');save(fig,'residual_distribution')
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        stats.probplot(p.residual,dist='norm',plot=ax);ax.set(title='Test residual Q-Q (normal reference)',ylabel='Ordered residual (Wh)');save(fig,'residual_qq')
        dates=pd.to_datetime(p.timestamp)
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        ax.plot(dates,p.actual,lw=.7,label='Actual');ax.plot(dates,p.prediction,lw=.7,label='Frozen estimate')
        ax.set(ylabel='Energy (Wh)',xlabel='Source-local date',title='All 2,962 Test rows');ax.legend();save(fig,'time_series_prediction')
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        valid=p.interval_valid.to_numpy(dtype=bool)
        ax.fill_between(dates,p.lower,p.upper,where=valid,alpha=.3,label='Nominal 95% PI (frozen q)')
        ax.plot(dates,p.actual,lw=.6,label='Actual');ax.plot(dates,p.prediction,lw=.7,label='Prediction')
        ax.set(ylabel='Energy (Wh)',xlabel='Source-local date',title='Observed Test intervals; no future coverage guarantee');ax.legend();save(fig,'prediction_interval')
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.stem(np.arange(len(d['residual_acf'])),d['residual_acf'])
        ax.set(xlabel='Lag (10-minute rows)',ylabel='Autocorrelation',title='Test residual ACF (descriptive)');save(fig,'residual_acf')
        metrics=m['metrics'];coverage=m['prediction_interval']
        text=['# C2 Frozen Test evaluation','',f"Run ID: {m['run_id']}",
            f"First evaluation UTC: {read_checkpoint(OUT/'first_evaluation.json')['started_at_utc']}",
            f"Completed UTC: {m['completed_at_utc']}",
            'Test: 2,962 chronological rows, 2016-05-07 04:30:00 through 2016-05-27 18:00:00.',
            'Model mlr-1.0.0. No retraining, selection or recalibration. Exactly one official Test pipeline.predict call.','',
            '## Metrics','',json.dumps(metrics,indent=2),'','## Interval audit','',json.dumps(coverage,indent=2),'',
            f"Frozen q={m['q']!r} Wh. Coverage is observed on this Test segment, not a strict future 95% guarantee.",
            'Widths and below/above counts use valid intervals; coverage_all uses all rows, with invalid intervals counted as uncovered.',
            'Point estimates are not clipped. The lower endpoint is clipped to zero.','',
            '## Statistical diagnostics','',json.dumps(d,indent=2),'',
            '## Freeze and reproducibility evidence','',
            f"Fit counters: {m['fit_counts']}; forbidden update attempts: {m['forbidden_update_attempts']}.",
            'C1 model, JSON, calibration record and decision hashes match before/after. Saved predictions exactly reproduce point metrics and interval audit.',
            'Preflight SHA checks read Test bytes only; the first-evaluation marker was saved before CSV Target parsing.',
            'All report computations use saved predictions/design; no Test file is reopened for reporting.','',
            '![Test prediction comparison](figures/prediction_vs_actual.png)','',
            '![Test interval audit](figures/prediction_interval.png)']
        (OUT/'C2_TEST_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
        files=[OUT/'evaluation_manifest.json',OUT/'diagnostics.json',OUT/'vif.csv',OUT/'C2_TEST_REPORT.md',*figures.iterdir()]
        result=dict(status='COMPLETE',run_id=m['run_id'],created_at_utc=now(),diagnostic_split='TEST',
            report_dataset_reads=gate.reads,report_model_prediction_calls=0,
            files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files})
        write_once(report_path,result)
        return result


if __name__=='__main__':print(json.dumps(report(),indent=2))
