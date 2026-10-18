import hashlib
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.training.audit import boundary, validate_xy, compact_audit
from src.training.pipeline import candidate_features, fit_pipeline
from src.evaluation.metrics import regression_metrics


def load_training(root):
    """Only opens the verified Train partition. Never opens held-out CSVs."""
    root = Path(root)
    manifest = json.loads((root / 'reports/data/split_manifest.json').read_text())
    path = root / 'data/processed/train.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['partitions']['train']['sha256']:
        raise ValueError('Training file hash differs from Part A manifest')
    frame = pd.read_csv(path)
    frame['date'] = pd.to_datetime(frame.date, format='%Y-%m-%d %H:%M:%S', errors='raise')
    frame = frame.set_index('date')
    if len(frame) != manifest['partitions']['train']['rows']:
        raise ValueError('Training count differs')
    boundary(frame)
    return frame, manifest


def run_cv(frame, config, allowed_train_index):
    """allowed_train_index must be the verified Training timestamps, not external partitions."""
    config.validate()
    if not frame.index.isin(allowed_train_index).all():
        raise ValueError('CV input contains timestamps outside verified Training')
    X, y = frame[candidate_features(config)], frame.Appliances
    validate_xy(X, y)
    folds = []
    started = time.perf_counter()
    for fold, (train, valid) in enumerate(TimeSeriesSplit(n_splits=5).split(X)):
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline = fit_pipeline(config, X.iloc[train], y.iloc[train])
            prediction = pipeline.predict(X.iloc[valid])
        selector = pipeline.named_steps['selector']
        p = 0 if config.model == 'mean' else len(selector.selected_features_)
        folds.append(dict(fold=fold, train=boundary(X.iloc[train]), validation=boundary(X.iloc[valid]),
                          selected_features=selector.selected_features_,
                          metrics=regression_metrics(y.iloc[valid], prediction, p,
                                                     config.model in ['ridge', 'lasso']),
                          fit_components={name: step.fit_boundary_ for name, step in pipeline.named_steps.items()},
                          selector_audit=compact_audit(selector.audit_),
                          warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                          seconds=time.perf_counter() - tick))
    rmse = [f['metrics']['rmse'] for f in folds]
    sets = [set(f['selected_features']) for f in folds]
    similarities = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return dict(schema_version='1.0.0', experiment_id=config.identifier, config=asdict(config),
                candidate_count=len(X.columns), effective_k=len(sets[0]),
                input=boundary(X), input_sha256=hashlib.sha256(
                    frame.to_csv(date_format='%Y-%m-%d %H:%M:%S', lineterminator='\n').encode()).hexdigest(),
                cv_type='TimeSeriesSplit', purpose='smoke_only_not_research_conclusion',
                folds=folds, cv_rmse_mean=float(np.mean(rmse)), cv_rmse_std=float(np.std(rmse)),
                feature_jaccard_mean=float(np.mean(similarities)), seconds=time.perf_counter() - started)
