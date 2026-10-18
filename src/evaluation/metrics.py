import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted, p, regularized=False):
    y, prediction = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or len(y) == 0:
        raise ValueError('Nonempty matching vectors required')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or p < 0:
        raise ValueError('Finite vectors and nonnegative feature count required')
    mse = float(mean_squared_error(y, prediction))
    valid_r2 = len(y) > 1 and np.var(y) > 0
    r2 = float(r2_score(y, prediction)) if valid_r2 else None
    adjusted = 1 - (1 - r2) * (len(y) - 1) / (len(y) - p - 1) if valid_r2 and len(y) > p + 1 else None
    return dict(mae=float(mean_absolute_error(y, prediction)), mse=mse, rmse=float(np.sqrt(mse)),
                r2=r2, adjusted_r2=adjusted, n=len(y), p=p,
                adjusted_r2_note='Traditional feature-count formula; not effective degrees of freedom for Ridge/Lasso.'
                if regularized else 'Undefined if constant target, fewer than 2 rows, or n <= p+1.')
