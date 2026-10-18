"""Actual fit row identities, preserved through pandas-output preprocessing."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.training.telemetry import track_fit


def boundary(X):
    if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError('Fit requires a DataFrame with source-local DatetimeIndex')
    if X.index.tz is not None or X.index.hasnans or not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError('Fit timestamps must be naive, unique, valid and increasing')
    if X.empty or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError('Empty or infinite feature input')
    ids = X.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    return dict(rows=len(X), start=ids[0], end=ids[-1], timestamps=ids,
                timestamp_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())


def validate_xy(X, y):
    result = boundary(X)
    if not isinstance(y, pd.Series) or not y.index.equals(X.index):
        raise ValueError('Target must have exactly matching timestamp index')
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError('Target must be finite')
    return result


class AuditedImputer(SimpleImputer):
    def fit(self, X, y=None):
        track_fit('imputer', X)
        self.fit_boundary_ = boundary(X)
        return super().fit(X, y)


class AuditedScaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        track_fit('scaler', X)
        self.fit_boundary_ = boundary(X)
        return super().fit(X, y, sample_weight=sample_weight)


class AuditedRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        track_fit('estimator', X)
        self.fit_boundary_ = validate_xy(X, y)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def processed_estimator(estimator):
    # Empty columns are retained to keep the fixed feature contract across folds.
    return Pipeline([
        ('imputer', AuditedImputer(strategy='median', keep_empty_features=True).set_output(transform='pandas')),
        ('scaler', AuditedScaler().set_output(transform='pandas')),
        ('estimator', AuditedRegressor(estimator)),
    ])


def preprocessing_audit(pipeline):
    return {name: step.fit_boundary_ for name, step in pipeline.named_steps.items()}


def compact_audit(records):
    """Losslessly deduplicate repeated timestamp scopes in exported nested fit records."""
    scopes, exported = {}, []
    def reference(bounds):
        key = bounds['timestamp_sha256']
        if key in scopes and scopes[key] != bounds:
            raise ValueError('Conflicting audit scope')
        scopes[key] = bounds
        return key
    for record in records:
        item = dict(context=record['context'],
                    components={name: reference(bounds) for name, bounds in record['components'].items()})
        if 'train' in record:
            item['train_scope'] = reference(record['train'])
            ids = record['validation_timestamps']
            bounds = dict(rows=len(ids), start=ids[0], end=ids[-1], timestamps=ids,
                          timestamp_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
            item['validation_scope'] = reference(bounds)
        exported.append(item)
    return dict(scopes=scopes, records=exported)
