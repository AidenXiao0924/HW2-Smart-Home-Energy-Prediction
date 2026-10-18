"""Selectors receive RAW fold data so inner CV never sees outer-fitted transforms."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from src.training.audit import validate_xy, processed_estimator, preprocessing_audit
from src.training.telemetry import track_fit, progress

METHODS = ('select_k_best', 'rfe', 'forward_sfs', 'lasso', 'permutation')


class RFECountedRegressor(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        track_fit('estimator', X)
        track_fit('rfe_estimator', X)
        return super().fit(X, y, sample_weight=sample_weight)


class TemporalFeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, method='select_k_best', k=15, inner_splits=5, seed=42,
                 lasso_alpha=1.0, permutation_repeats=5, expected_features=None):
        self.method = method
        self.k = k
        self.inner_splits = inner_splits
        self.seed = seed
        self.lasso_alpha = lasso_alpha
        self.permutation_repeats = permutation_repeats
        self.expected_features = expected_features

    def fit(self, X, y):
        track_fit('selector', X)
        if self.expected_features is not None and list(X.columns) != list(self.expected_features):
            raise ValueError('Expected exact candidate feature schema and order')
        self.fit_boundary_ = validate_xy(X, y)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        count = self.n_features_in_ if self.k == 'ALL' else self.k
        if self.method not in METHODS or not isinstance(count, int) or not 1 <= count <= X.shape[1]:
            raise ValueError('Invalid feature selection method or K')
        self.audit_ = []
        self._audit_scope_cache = {}
        self._validation_index_cache = {}
        self.scores_ = None
        self.selection_path_ = []
        self.nonzero_count_ = None
        self.lasso_metadata_ = None
        if count == X.shape[1]:
            selected = list(range(count))
        elif self.method == 'forward_sfs':
            selected = self._forward(X, y, count)
        elif self.method == 'permutation':
            selected = self._permutation(X, y, count)
        else:
            selected = self._direct(X, y, count)
        self.selected_indices_ = np.asarray(sorted(selected), dtype=int)
        self.selected_features_ = self.feature_names_in_[self.selected_indices_].tolist()
        return self

    def _record(self, pipe, X, train, validation, context):
        # Identical inner-fold scopes recur for every SFS candidate. Share their row IDs in memory.
        components = {name: self._audit_scope_cache.setdefault(bounds['timestamp_sha256'], bounds)
                      for name, bounds in preprocessing_audit(pipe).items()}
        key = tuple(validation)
        if key not in self._validation_index_cache:
            self._validation_index_cache[key] = X.index[validation].strftime('%Y-%m-%d %H:%M:%S').tolist()
        self.audit_.append(dict(context=context, train=components['imputer'],
                                validation_timestamps=self._validation_index_cache[key],
                                components=components))

    def _direct(self, X, y, count):
        # These methods do not evaluate validation error. Their transforms use only this fit call's rows.
        estimator = Lasso(alpha=self.lasso_alpha, max_iter=50000, tol=1e-6,
                          selection='cyclic', random_state=self.seed) if self.method == 'lasso' else LinearRegression()
        pipe = processed_estimator(estimator)
        if self.method != 'lasso':
            pipe = pipe[:-1]
        pipe.fit(X, y)
        transformed = (pipe[:-1] if self.method == 'lasso' else pipe).transform(X)
        self.audit_.append(dict(context='selector_fit', components=preprocessing_audit(pipe)))
        if self.method == 'select_k_best':
            selector = SelectKBest(f_regression, k=count).fit(transformed, y)
            scores = selector.scores_
            self.scores_ = np.nan_to_num(scores, nan=0., posinf=np.finfo(float).max).tolist()
            return self._top(self.scores_, count)
        if self.method == 'rfe':
            self.selector_ = RFE(RFECountedRegressor(), n_features_to_select=count, step=1).fit(transformed, y)
            return np.flatnonzero(self.selector_.support_).tolist()
        self.scores_ = np.abs(pipe.named_steps['estimator'].estimator_.coef_).tolist()
        self.nonzero_count_ = int(np.count_nonzero(self.scores_))
        self.lasso_metadata_ = dict(alpha=self.lasso_alpha, nonzero_count=self.nonzero_count_,
                                    coefficients=pipe.named_steps['estimator'].estimator_.coef_.tolist(),
                                    scaling='StandardScaler after median imputation, fit only on selector training rows',
                                    scaler_mean=pipe.named_steps['scaler'].mean_.tolist(),
                                    scaler_scale=pipe.named_steps['scaler'].scale_.tolist(),
                                    imputer_statistics=pipe.named_steps['imputer'].statistics_.tolist(),
                                    n_iter=int(pipe.named_steps['estimator'].estimator_.n_iter_),
                                    dual_gap=float(pipe.named_steps['estimator'].estimator_.dual_gap_))
        # Exact-K comparison: rank absolute coefficients; zero ties use fixed source column order.
        return self._top(self.scores_, count)

    @staticmethod
    def _top(scores, count):
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:count]

    def _forward(self, X, y, count):
        selected = []
        folds = list(TimeSeriesSplit(n_splits=self.inner_splits).split(X))
        for step in range(count):
            candidates = []
            for index in range(X.shape[1]):
                if index in selected:
                    continue
                columns = sorted(selected + [index])
                errors = []
                for fold, (train, valid) in enumerate(folds):
                    pipe = processed_estimator(LinearRegression())
                    pipe.fit(X.iloc[train, columns], y.iloc[train])
                    prediction = pipe.predict(X.iloc[valid, columns])
                    errors.append(float(np.sqrt(mean_squared_error(y.iloc[valid], prediction))))
                    self._record(pipe, X, train, valid, dict(step=step, candidate=index, inner_fold=fold))
                candidates.append((float(np.mean(errors)), index))
            score, winner = min(candidates)  # deterministic index tie-break
            selected.append(winner)
            self.selection_path_.append(dict(feature=str(X.columns[winner]), mean_inner_rmse=score))
            progress(dict(event='sfs_step_complete', step=step + 1, requested_k=count,
                          selected_feature=str(X.columns[winner])))
        return selected

    def _permutation(self, X, y, count):
        rankings = []
        for fold, (train, valid) in enumerate(TimeSeriesSplit(n_splits=self.inner_splits).split(X)):
            pipe = processed_estimator(LinearRegression()).fit(X.iloc[train], y.iloc[train])
            # Permute raw validation features; preprocessing remains fixed at INNER train values.
            result = permutation_importance(pipe, X.iloc[valid], y.iloc[valid],
                                            scoring='neg_root_mean_squared_error',
                                            n_repeats=self.permutation_repeats,
                                            random_state=self.seed, n_jobs=1)
            rankings.append(result.importances_mean)
            self._record(pipe, X, train, valid, dict(inner_fold=fold, purpose='permutation_ranking'))
        self.scores_ = np.mean(rankings, axis=0).tolist()
        return self._top(self.scores_, count)

    def transform(self, X):
        check_is_fitted(self, 'selected_features_')
        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError('Feature schema/order differs from fit')
        return X.loc[:, self.selected_features_].copy()

    def get_support(self):
        check_is_fitted(self, 'selected_indices_')
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selected_features_')
        return np.asarray(self.selected_features_, dtype=object)
