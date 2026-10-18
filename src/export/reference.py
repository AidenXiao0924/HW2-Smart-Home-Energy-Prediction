"""Strict scalar binary64 reference for future JavaScript parity."""
from datetime import datetime
import math
import re
from src.calibration.conformal import interval

FEATURES = ['T1','RH_1','T3','RH_3','T4','T5','RH_5','T7','RH_8','T9','RH_9',
            'Windspeed','Visibility','Tdewpoint','Hour']


def validate_artifact(a):
    if a['schema_version'] != '1.0.0' or a['model_version'] != 'mlr-1.0.0':
        raise ValueError('Unsupported artifact version')
    if a['model_type'] != 'LinearRegression' or a['feature_order'] != FEATURES:
        raise ValueError('Invalid model or frozen feature order')
    arrays = [a['coefficients'], a['preprocessing']['imputer']['statistics'],
              a['preprocessing']['scaler']['mean'], a['preprocessing']['scaler']['scale']]
    if any(len(v) != len(FEATURES) for v in arrays):
        raise ValueError('Artifact vector length mismatch')
    for v in [*arrays, [a['intercept'], a['prediction_interval']['q']]]:
        if any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in v):
            raise ValueError('Artifact parameters must be finite numbers')
    if any(x <= 0 for x in arrays[3]) or a['prediction_interval']['q'] < 0:
        raise ValueError('Invalid scale or quantile')
    pi = a['prediction_interval']
    if pi['method'] != 'split_conformal_absolute_residual' or pi['alpha'] != 0.05:
        raise ValueError('Unsupported interval method')
    if pi['lower_rule'] != 'max(0,predicted-q)' or pi['upper_rule'] != 'predicted+q':
        raise ValueError('Unsupported interval rules')
    if a['preprocessing']['imputer']['strategy'] != 'median':
        raise ValueError('Unsupported imputation')
    if ([f['name'] for f in a['features']] != FEATURES or a['target'] != dict(name='Appliances',unit='Wh')
        or a['prediction_task'] != 'contemporaneous_estimation'
        or a['preprocessing']['time_features']['format'] != 'YYYY-MM-DD HH:mm:ss'):
        raise ValueError('Invalid feature metadata or time contract')
    return a


def load_verified(root, artifact_path, manifest_path):
    import json
    from pathlib import Path
    from src.training.checkpoint import read_checkpoint
    from src.training.b3_io import sha
    root, artifact_path = Path(root), Path(artifact_path)
    manifest = read_checkpoint(manifest_path)
    if manifest['model_version'] != 'mlr-1.0.0':
        raise ValueError('Manifest version mismatch')
    key = str(artifact_path.relative_to(root)).replace('\\','/')
    if manifest['files'].get(key) != sha(artifact_path):
        raise ValueError('Artifact hash mismatch')
    return validate_artifact(json.loads(artifact_path.read_text(encoding='utf-8')))


def ordered_values(row):
    if not isinstance(row, dict):
        raise ValueError('Named input object required')
    if 'date' not in row or not isinstance(row['date'],str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',row['date']):
        raise ValueError('Source-local date YYYY-MM-DD HH:mm:ss required')
    dt = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
    hour = dt.hour
    if 'Hour' in row and (isinstance(row['Hour'],bool) or not isinstance(row['Hour'],(int,float)) or row['Hour'] != hour):
        raise ValueError('Hour conflicts with source-local date')
    values = []
    for feature in FEATURES:
        if feature == 'Hour':
            values.append(hour)
            continue
        if feature not in row:
            raise ValueError('Missing required column: '+feature)
        value = row[feature]
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError('Only finite numbers or null accepted: '+feature)
        values.append(value)
    return values


def predict(a, row):
    validate_artifact(a)
    values = ordered_values(row)
    prep = a['preprocessing']
    result = float(a['intercept'])
    for i, value in enumerate(values):
        if value is None:
            value = prep['imputer']['statistics'][i]
        scaled = (value-prep['scaler']['mean'][i])/prep['scaler']['scale'][i]
        result += scaled*a['coefficients'][i]
    return interval(result, a['prediction_interval']['q'])
