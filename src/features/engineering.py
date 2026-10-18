from src.data.validate import SENSORS, WEATHER

CANDIDATE_FEATURES = [*SENSORS, *WEATHER, 'Hour', 'DayOfWeek', 'IsWeekend', 'lights']
DEPLOYABLE_FEATURES = [c for c in CANDIDATE_FEATURES if c != 'lights']


def engineer(frame):
    """Use naive source-local calendar fields, never timezone conversion."""
    result = frame.copy()
    if result.date.dt.tz is not None:
        raise ValueError('Timezone-aware timestamps are not accepted')
    result['Hour'] = result.date.dt.hour
    result['DayOfWeek'] = result.date.dt.dayofweek
    result['IsWeekend'] = (result.DayOfWeek >= 5).astype(int)
    return result
