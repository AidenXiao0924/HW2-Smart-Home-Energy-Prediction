"""Importable fitted pipeline classes, safe to serialize across CLI/report processes."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.pipeline import Pipeline
from src.features.engineering import DEPLOYABLE_FEATURES
from src.training.audit import validate_xy,processed_estimator
from src.training.telemetry import track_fit


class FrozenColumns(TransformerMixin,BaseEstimator):
    def __init__(self, features, expected):
        self.features=features
        self.expected=expected
    def fit(self,X,y):
        self.fit_boundary_=validate_xy(X,y)
        if (tuple(X.columns)!=tuple(self.expected) or not 14<=len(self.features)<=27
            or len(set(self.features))!=len(self.features) or not set(self.features)<=set(DEPLOYABLE_FEATURES)):
            raise ValueError('Invalid fixed deployment feature scheme')
        track_fit('frozen_columns',X)
        self.feature_names_in_=np.asarray(X.columns,dtype=object)
        return self
    def transform(self,X):
        if tuple(X.columns)!=tuple(self.expected):raise ValueError('Input feature order differs')
        return X.loc[:,list(self.features)].copy()


def model_pipeline(features,model,alpha,protocol):
    estimators=dict(mlr=LinearRegression(),ridge=Ridge(alpha=alpha),
                    lasso=Lasso(alpha=alpha,max_iter=protocol['lasso_max_iter'],tol=protocol['lasso_tol'],
                                selection='cyclic',random_state=protocol['seed']))
    return Pipeline([('features',FrozenColumns(tuple(features),tuple(DEPLOYABLE_FEATURES))),
                     *processed_estimator(estimators[model]).steps])
