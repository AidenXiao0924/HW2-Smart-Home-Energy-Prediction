import math
import numpy as np


def quantile(scores, alpha=0.05):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Residual scores must be finite, nonnegative and nonempty')
    if not 0 < alpha < 1:
        raise ValueError('Invalid alpha')
    n = len(values)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        raise ValueError('Insufficient calibration samples for finite order statistic')
    return dict(n=n, k=k, q=float(np.sort(values)[k-1]), alpha=alpha)


def interval(prediction, q):
    if not math.isfinite(prediction) or not math.isfinite(q) or q < 0:
        raise ValueError('Nonfinite prediction or invalid quantile')
    lower, upper = max(0.0, prediction-q), prediction+q
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError('Interval overflow')
    warnings = ['NEGATIVE_ESTIMATE_WARNING'] if prediction < 0 else []
    valid = upper >= lower
    if not valid:
        warnings.append('INVALID_INTERVAL')
    return dict(predicted=prediction, lower=lower, upper=upper,
                interval_valid=valid, warnings=warnings)
