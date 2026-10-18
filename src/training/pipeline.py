from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from src.features.engineering import CANDIDATE_FEATURES, DEPLOYABLE_FEATURES
from src.features.selection import TemporalFeatureSelector
from src.training.audit import processed_estimator, validate_xy


def candidate_features(config):
    config.validate()
    return list(DEPLOYABLE_FEATURES if config.candidate_set == 'deployable27' else CANDIDATE_FEATURES)


def build_pipeline(config):
    config.validate()
    models = {'mean': DummyRegressor(strategy='mean'), 'mlr': LinearRegression(),
              'ridge': Ridge(alpha=config.alpha),
              'lasso': Lasso(alpha=config.alpha, max_iter=50000, tol=1e-6,
                             selection='cyclic', random_state=config.seed)}
    selector = TemporalFeatureSelector(method=config.method, k='ALL' if config.model == 'mean' else config.k,
                                       inner_splits=config.inner_splits, seed=config.seed,
                                       lasso_alpha=config.selector_lasso_alpha,
                                       permutation_repeats=config.permutation_repeats,
                                       expected_features=tuple(candidate_features(config)))
    # Selector must receive RAW X. It owns nested preprocessing for internal CV.
    # Final imputation/scaling fit only after selection, on the same outer training rows.
    return Pipeline([('selector', selector), *processed_estimator(models[config.model]).steps])


def fit_pipeline(config, X, y):
    validate_xy(X, y)
    if list(X.columns) != candidate_features(config):
        raise ValueError('Expected exact candidate schema; forbidden/extraneous columns rejected')
    return build_pipeline(config).fit(X, y)
