"""Reproducible publication figures using only consolidated public JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.export.c3 import PUBLIC, PRIVATE, load_json
from src.training.c1 import C1Gate
from src.training.b3_io import write_once,sha,now


def generate():
    out=PRIVATE/'figures';manifest=PRIVATE/'chart_manifest.json'
    if manifest.exists():
        from src.training.checkpoint import read_checkpoint
        record=read_checkpoint(manifest)
        for path,value in record['files'].items():
            if sha(PRIVATE/path)!=value:raise ValueError('Figure changed')
        return record
    with C1Gate('none') as gate:
        summary=load_json(PUBLIC/'research_summary.json')
        series=load_json(PUBLIC/'charts/test_series.json')
        cv=load_json(PUBLIC/'charts/training_cv.json')
        residuals=load_json(PUBLIC/'charts/test_residuals.json')
        out.mkdir(exist_ok=True)
        def save(fig,name):
            for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=150,bbox_inches='tight')
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
        p=pd.DataFrame(series['rows']);dates=pd.to_datetime(p.timestamp)
        ax.plot(dates,p.actual_mean,label='Actual hourly mean')
        ax.plot(dates,p.prediction_mean,label='Estimate hourly mean')
        ax.plot(dates,p.actual_max,alpha=.4,lw=.7,label='Actual hourly maximum')
        ax.set(xlabel='Source-local time (hourly groups)',ylabel='Energy (Wh)',title='Test | Aggregated display; metrics use all original rows');ax.legend();save(fig,'test_hourly')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        data=pd.DataFrame(cv['configurations'])
        for (candidate,method),rows in data[data.k!='ALL'].groupby(['candidate_set','method']):
            rows=rows.sort_values('effective_k');ax.plot(rows.effective_k,rows.cv_rmse_mean,marker='o',label=f'{candidate}: {method}')
        ax.set(xlabel='K (K10 research only)',ylabel='Mean CV RMSE (Wh)',title='Training CV | 27/28 sets kept distinct');ax.legend(fontsize=7,ncols=2);save(fig,'training_cv')
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        train=summary['training_vif']['values']['vif'];test=summary['diagnostics']['values']['vif'];x=np.arange(15)
        ax.bar(x-.2,[r['vif'] for r in train],width=.4,label='Train+Validation')
        ax.bar(x+.2,[r['vif'] for r in test],width=.4,label='Test')
        ax.set(xticks=x,xticklabels=[r['feature'] for r in train],ylabel='VIF (intercept included)',title='Frozen 15-feature design | Diagnostic only');ax.tick_params(axis='x',rotation=55);ax.legend();save(fig,'vif_by_split')
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        edges=np.array(residuals['histogram']['edges']);counts=residuals['histogram']['counts']
        ax.bar(edges[:-1],counts,width=np.diff(edges),align='edge')
        ax.set(xlabel='Actual - Prediction (Wh)',ylabel='Count',title='Test | Full-row histogram, no model execution');save(fig,'test_residual_histogram')
        result=dict(at_utc=now(),inputs={str(p.relative_to(PUBLIC)):sha(p) for p in
            [PUBLIC/'research_summary.json',PUBLIC/'charts/test_series.json',PUBLIC/'charts/training_cv.json',PUBLIC/'charts/test_residuals.json']},
            files={str(p.relative_to(PRIVATE)).replace('\\','/'):sha(p) for p in out.iterdir()},
            dataset_reads=gate.reads,model_prediction_calls=0)
        write_once(manifest,result)
        return result


if __name__=='__main__':print(json.dumps(generate(),indent=2))
