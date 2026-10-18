"""Real Training data, plus explicit logic-only perturbations for leakage tests."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from src.features.selection import TemporalFeatureSelector, METHODS
from src.training.audit import processed_estimator
from src.training.config import ExperimentConfig
from src.training.pipeline import candidate_features, build_pipeline, fit_pipeline
from src.training.cv import load_training, run_cv
from src.evaluation.metrics import regression_metrics
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def training():
    frame, _ = load_training(ROOT)
    return frame


@pytest.fixture
def xy(training):
    config = ExperimentConfig()
    frame = training.iloc[:180]
    return frame[candidate_features(config)].copy(), frame.Appliances.copy()


@pytest.mark.parametrize('method', METHODS)
def test_all_five_selectors_real_data(method, xy):
    X, y = xy
    # Small 6-feature logic check to exercise Forward SFS without a large experiment.
    X = X.iloc[:, :6]
    selector = TemporalFeatureSelector(method=method, k=2, inner_splits=3, permutation_repeats=2)
    transformed = selector.fit_transform(X, y)
    assert transformed.shape == (len(X), 2)
    assert transformed.index.equals(X.index)
    assert selector.get_support().sum() == 2
    assert list(transformed.columns) == selector.get_feature_names_out().tolist()
    for record in selector.audit_:
        for bounds in record['components'].values():
            assert set(bounds['timestamps']) <= set(X.index.strftime('%Y-%m-%d %H:%M:%S'))
        if 'validation_timestamps' in record:
            train = record['train']['timestamps']
            valid = record['validation_timestamps']
            assert max(train) < min(valid)
            assert not set(train) & set(valid)
            for bounds in record['components'].values():
                assert bounds['timestamps'] == train


@pytest.mark.parametrize('method', ['forward_sfs', 'permutation'])
def test_nested_actual_preprocessing_fits(monkeypatch, method, xy):
    from src.training.audit import AuditedImputer, AuditedScaler
    X, y = xy
    X = X.iloc[:, :5].copy()
    X.iloc[-30:, :] = 1e8  # held-out sentinel must not alter earlier fitted statistics
    fits = []
    original = AuditedImputer.fit
    original_scale = AuditedScaler.fit
    def observed(self, data, target=None):
        fits.append(('imputer', data.index.copy()))
        return original(self, data, target)
    def observed_scale(self, data, target=None, sample_weight=None):
        fits.append(('scaler', data.index.copy()))
        return original_scale(self, data, target, sample_weight)
    monkeypatch.setattr(AuditedImputer, 'fit', observed)
    monkeypatch.setattr(AuditedScaler, 'fit', observed_scale)
    selector = TemporalFeatureSelector(method=method, k=2, inner_splits=3, permutation_repeats=1)
    selector.fit(X, y)
    allowed = {tuple(X.index[a]) for a, _ in TimeSeriesSplit(n_splits=3).split(X)}
    assert fits
    assert all(tuple(ids) in allowed for _, ids in fits)
    assert {name for name, _ in fits} == {'imputer', 'scaler'}


@pytest.mark.parametrize('method', METHODS)
def test_selector_reproducible(method, xy):
    X, y = xy
    X = X.iloc[:, :5]
    selector = TemporalFeatureSelector(method=method, k=2, inner_splits=2, permutation_repeats=1)
    first, second = clone(selector).fit(X, y), clone(selector).fit(X, y)
    assert first.selected_features_ == second.selected_features_
    if first.scores_ is not None:
        np.testing.assert_allclose(first.scores_, second.scores_, rtol=0, atol=0)


@pytest.mark.parametrize('k', [10, 15, 20, 25, 'ALL'])
@pytest.mark.parametrize('candidates', ['deployable27', 'research28'])
def test_configured_counts_and_exclusions(training, k, candidates):
    config = ExperimentConfig(k=k, candidate_set=candidates)
    X = training.iloc[:120][candidate_features(config)]
    model = fit_pipeline(config, X, training.Appliances.iloc[:120])
    selected = model.named_steps['selector'].selected_features_
    assert len(selected) == (X.shape[1] if k == 'ALL' else k)
    assert not set(selected) & {'rv1', 'rv2', 'date', 'Appliances'}
    assert ('lights' in X) == (candidates == 'research28')
    assert not config.deployment_eligible if k == 10 or candidates == 'research28' else config.deployment_eligible


def test_outer_cv_actual_fit_boundaries(monkeypatch, training):
    from src.training.audit import AuditedImputer, AuditedScaler, AuditedRegressor
    data = training.iloc[:180]
    events = []
    for cls in [AuditedImputer, AuditedScaler, AuditedRegressor, TemporalFeatureSelector]:
        original = cls.fit
        def observer(self, X, y=None, _original=original, _name=cls.__name__, **kw):
            events.append((_name, tuple(X.index)))
            return _original(self, X, y, **kw)
        monkeypatch.setattr(cls, 'fit', observer)
    config = ExperimentConfig(k=15)
    result = run_cv(data, config, training.index)
    allowed = {tuple(data.index[a]) for a, _ in TimeSeriesSplit(n_splits=5).split(data)}
    assert len(result['folds']) == 5
    assert all(ids in allowed for _, ids in events)
    assert {name for name, _ in events} == {'AuditedImputer', 'AuditedScaler', 'AuditedRegressor', 'TemporalFeatureSelector'}
    for fold in result['folds']:
        train, valid = fold['train']['timestamps'], fold['validation']['timestamps']
        assert max(train) < min(valid)
        assert not set(train) & set(valid)
        for bounds in fold['fit_components'].values():
            assert bounds['timestamps'] == train


def test_future_extremes_do_not_change_fitted_parameters(xy):
    X, y = xy
    train, valid = next(TimeSeriesSplit(n_splits=5).split(X))
    config = ExperimentConfig(k=15)
    first = fit_pipeline(config, X.iloc[train], y.iloc[train])
    changed_x, changed_y = X.copy(), y.copy()
    changed_x.iloc[valid] = 1e12
    changed_y.iloc[valid] = -1e12
    second = fit_pipeline(config, changed_x.iloc[train], changed_y.iloc[train])
    assert first.named_steps['selector'].selected_features_ == second.named_steps['selector'].selected_features_
    np.testing.assert_array_equal(first.named_steps['imputer'].statistics_, second.named_steps['imputer'].statistics_)
    np.testing.assert_array_equal(first.named_steps['scaler'].mean_, second.named_steps['scaler'].mean_)
    np.testing.assert_array_equal(first.predict(X.iloc[valid]), second.predict(X.iloc[valid]))


@pytest.mark.parametrize('model', ['mean', 'mlr', 'ridge', 'lasso'])
def test_model_interfaces_only(model, xy):
    X, y = xy
    config = ExperimentConfig(k='ALL', model=model)
    pipeline = fit_pipeline(config, X, y)
    prediction = pipeline.predict(X.iloc[-5:])
    assert np.isfinite(prediction).all()
    if model == 'mean': np.testing.assert_allclose(prediction, y.mean())


def test_all_missing_column_retained(xy):
    X, y = xy
    X.iloc[:, 0] = np.nan
    pipe = fit_pipeline(ExperimentConfig(k='ALL'), X, y)
    assert pipe.named_steps['estimator'].n_features_in_ == 27
    assert np.isfinite(pipe.predict(X)).all()


def test_fit_rejects_wrong_schema_order_time_target(xy, training):
    X, y = xy
    config = ExperimentConfig()
    for bad in [X.iloc[::-1], X.iloc[:, ::-1], X.assign(rv1=1)]:
        with pytest.raises(ValueError): fit_pipeline(config, bad, y)
    with pytest.raises(ValueError): fit_pipeline(config, X, y.iloc[::-1])
    with pytest.raises(ValueError): run_cv(training.tail(30), config, training.index[:180])


def test_pipeline_cloning_and_config_identity(xy):
    config = ExperimentConfig()
    assert config.identifier == replace(config).identifier
    assert config.identifier != replace(config, seed=43).identifier
    X, y = xy
    original = build_pipeline(config)
    assert not hasattr(original.named_steps['selector'], 'selected_features_')
    cloned = clone(original).fit(X, y)
    assert not hasattr(original.named_steps['selector'], 'selected_features_')
    assert len(cloned.named_steps['selector'].selected_features_) == 15
    with pytest.raises(ValueError): replace(config, outer_splits=4).validate()


def test_metrics_reference_and_undefined_cases():
    metric = regression_metrics([1., 2., 3.], [1., 3., 2.], p=1)
    assert metric['mse'] == pytest.approx(2 / 3)
    assert metric['rmse'] == pytest.approx(np.sqrt(2 / 3))
    assert metric['r2'] == pytest.approx(0)
    assert metric['adjusted_r2'] == pytest.approx(-1)
    assert regression_metrics([1., 1.], [1., 1.], p=1)['r2'] is None
    assert regression_metrics([1., 2.], [1., 2.], p=1)['adjusted_r2'] is None
    with pytest.raises(ValueError): regression_metrics([1.], [np.nan], p=1)


def test_training_loader_opens_only_train(monkeypatch):
    opened = []
    original = pd.read_csv
    def observe(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', observe)
    frame, _ = load_training(ROOT)
    assert opened == ['train.csv']
    assert len(frame) == 13814


def test_direct_pipeline_fit_cannot_bypass_candidate_guard(xy):
    X, y = xy
    pipeline = build_pipeline(ExperimentConfig())
    with pytest.raises(ValueError): pipeline.fit(X.assign(rv1=1), y)
    with pytest.raises(ValueError): pipeline.fit(X.iloc[:, ::-1], y)


def test_mean_never_runs_target_feature_selection(xy):
    X, y = xy
    pipeline = fit_pipeline(ExperimentConfig(method='forward_sfs', k=10, model='mean'), X, y)
    selector = pipeline.named_steps['selector']
    assert selector.audit_ == []
    assert len(selector.selected_features_) == 27
    np.testing.assert_allclose(pipeline.predict(X.head()), y.mean())


def test_imputer_and_scaler_statistics_match_only_fit_data(xy):
    X, y = xy
    train, valid = next(TimeSeriesSplit(n_splits=5).split(X))
    X.iloc[0, 0] = np.nan
    X.iloc[valid, :] = 1e9
    pipeline = fit_pipeline(ExperimentConfig(k='ALL'), X.iloc[train], y.iloc[train])
    median = X.iloc[train].median().to_numpy()
    np.testing.assert_allclose(pipeline.named_steps['imputer'].statistics_, median)
    imputed = X.iloc[train].fillna(X.iloc[train].median())
    np.testing.assert_allclose(pipeline.named_steps['scaler'].mean_, imputed.mean().to_numpy())


def test_nested_audit_export_lossless(xy):
    from src.training.audit import compact_audit
    X, y = xy
    selector = TemporalFeatureSelector(method='forward_sfs', k=2, inner_splits=2).fit(X.iloc[:, :4], y)
    compact = compact_audit(selector.audit_)
    assert len(compact['records']) == len(selector.audit_)
    assert len(compact['scopes']) <= 4
    for original, record in zip(selector.audit_, compact['records']):
        assert compact['scopes'][record['train_scope']] == original['train']
        assert compact['scopes'][record['validation_scope']]['timestamps'] == original['validation_timestamps']
        for component, key in record['components'].items():
            assert compact['scopes'][key] == original['components'][component]
