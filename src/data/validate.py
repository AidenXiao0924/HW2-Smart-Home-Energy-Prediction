"""Strict raw schema and descriptive quality checks; never fits preprocessing."""
import csv
import numpy as np
import pandas as pd

SENSORS = [item for i in range(1, 10) for item in (f'T{i}', f'RH_{i}')]
WEATHER = ['T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
RAW_COLUMNS = ['date', 'Appliances', 'lights', *SENSORS, *WEATHER, 'rv1', 'rv2']
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def validate_frame(frame):
    if len(frame.columns) != len(set(frame.columns)) or set(frame.columns) != set(RAW_COLUMNS):
        raise ValueError('Raw schema must contain exactly the 29 specified unique columns')
    if frame.empty:
        raise ValueError('Empty dataset')
    result = frame.copy()
    strings = result['date'].astype('string')
    if not strings.str.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').fillna(False).all():
        raise ValueError('Invalid date format')
    result['date'] = pd.to_datetime(strings, format=DATE_FORMAT, errors='raise')
    for column in RAW_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors='raise')
        if np.isinf(result[column].to_numpy(dtype=float)).any():
            raise ValueError(f'Nonfinite numeric value: {column}')
    if result['Appliances'].isna().any():
        raise ValueError('Missing target')
    if result['date'].duplicated().any():
        raise ValueError('Duplicate timestamps: investigate before chronological splitting')
    return result


def read_raw(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV headers')
    return validate_frame(pd.read_csv(path))


def quality_report(frame):
    numeric = frame.drop(columns='date')
    bounds = {}
    for column in numeric:
        series = numeric[column]
        # Descriptive only; no removal/winsorization and no training threshold fitted.
        if column.startswith('RH_') or column == 'RH_out':
            count = ((series < 0) | (series > 100)).sum()
        elif column in ['Appliances', 'lights', 'Windspeed', 'Visibility', 'Press_mm_hg']:
            count = (series < 0).sum()
        else:
            count = 0
        bounds[column] = int(count)
    deltas = frame['date'].sort_values().diff().dropna()
    return dict(rows=len(frame), columns=len(frame.columns),
                time_start=str(frame.date.min()), time_end=str(frame.date.max()),
                missing={c: int(v) for c, v in frame.isna().sum().items()},
                duplicate_rows=int(frame.duplicated().sum()), duplicate_timestamps=int(frame.date.duplicated().sum()),
                original_time_sorted=bool(frame.date.is_monotonic_increasing),
                dtypes={c: str(v) for c, v in frame.dtypes.items()},
                physical_range_violations=bounds,
                physical_rule='RH in [0,100]; energy, wind, visibility and pressure nonnegative. No arbitrary temperature bound.',
                descriptive_statistics=numeric.describe().to_dict(),
                interval_seconds_counts={str(k.total_seconds()): int(v) for k, v in deltas.value_counts().items()},
                action='No rows dropped, imputed, scaled or selected; unusual values retained for research.')
