from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data.validate import read_raw, validate_frame, quality_report
from src.data.split import chronological_split, csv_bytes
from src.features.engineering import engineer, CANDIDATE_FEATURES, DEPLOYABLE_FEATURES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def real():
    # Formal dataset is mandatory: missing download is a failure, never a synthetic replacement.
    return read_raw(ROOT / 'data/raw/energydata_complete.csv')


def test_real_schema(real):
    assert len(real.columns) == 29
    assert len(real) > 0
    assert real.date.is_unique


def test_candidate_inventory(real):
    features = engineer(real)
    assert len(set(CANDIDATE_FEATURES)) == 28
    assert len(DEPLOYABLE_FEATURES) == 27
    assert not {'rv1', 'rv2', 'date', 'Appliances'} & set(CANDIDATE_FEATURES)
    assert set(CANDIDATE_FEATURES) <= set(features.columns)
    assert 'Hour' not in real.columns  # input was not mutated


def test_calendar_boundaries():
    # Logic-only boundary inputs, not a research dataset.
    frame = pd.DataFrame({'date': pd.to_datetime(['2026-10-16 23:59:59', '2026-10-17 00:00:00',
                                                 '2026-10-18 12:00:00', '2026-10-19 00:00:00'])})
    result = engineer(frame)
    assert result.Hour.tolist() == [23, 0, 12, 0]
    assert result.DayOfWeek.tolist() == [4, 5, 6, 0]
    assert result.IsWeekend.tolist() == [0, 1, 1, 0]


@pytest.mark.parametrize('defect', ['missing_column', 'extra_column', 'date_format', 'invalid_date',
                                     'duplicate_time', 'nonnumeric', 'infinite', 'missing_target'])
def test_invalid_schema(real, defect):
    frame = real.head(10).copy()
    frame['date'] = frame.date.dt.strftime('%Y-%m-%d %H:%M:%S')
    if defect == 'missing_column': frame = frame.drop(columns='T1')
    if defect == 'extra_column': frame['unknown'] = 1
    if defect == 'date_format': frame.loc[0, 'date'] = '2016-1-11 17:00:00'
    if defect == 'invalid_date': frame.loc[0, 'date'] = '2016-02-30 17:00:00'
    if defect == 'duplicate_time': frame.loc[1, 'date'] = frame.loc[0, 'date']
    if defect == 'nonnumeric':
        frame['T1'] = frame.T1.astype(object)
        frame.loc[0, 'T1'] = 'bad'
    if defect == 'infinite': frame.loc[0, 'T1'] = np.inf
    if defect == 'missing_target':
        frame['Appliances'] = frame.Appliances.astype(float)
        frame.loc[0, 'Appliances'] = np.nan
    with pytest.raises(ValueError): validate_frame(frame)


def test_quality_and_missing_sensor(real):
    frame = real.head(10).copy()
    frame.loc[0, 'T1'] = np.nan
    frame.loc[1, 'RH_1'] = 101.
    result = validate_frame(frame.assign(date=frame.date.dt.strftime('%Y-%m-%d %H:%M:%S')))
    report = quality_report(result)
    assert report['missing']['T1'] == 1
    assert report['physical_range_violations']['RH_1'] == 1
    assert pd.isna(result.loc[0, 'T1'])  # never globally imputed


def test_split_integrity_and_reproducibility(real):
    frame = engineer(real)
    parts, manifest = chronological_split(frame.iloc[::-1])
    again, metadata = chronological_split(frame)
    assert manifest == metadata
    n = len(real)
    sizes = [n * 70 // 100, n * 10 // 100, n * 5 // 100]
    assert [len(p) for p in parts.values()] == sizes + [n - sum(sizes)]
    timestamps = set()
    last = None
    for name, part in parts.items():
        assert not timestamps.intersection(part.date)
        assert part.date.is_monotonic_increasing
        assert last is None or last < part.date.iloc[0]
        assert csv_bytes(part) == csv_bytes(again[name])
        timestamps.update(part.date)
        last = part.date.iloc[-1]
    assert len(timestamps) == n


def test_split_rejects_duplicate_and_small(real):
    with pytest.raises(ValueError): chronological_split(pd.concat([real.head(10), real.head(1)]))
    with pytest.raises(ValueError): chronological_split(real.head(5))


def test_timezone_rejected(real):
    frame = real.head(2).copy()
    frame['date'] = frame.date.dt.tz_localize('UTC')
    with pytest.raises(ValueError): engineer(frame)


def test_duplicate_csv_headers(tmp_path):
    path = tmp_path / 'duplicate.csv'
    path.write_text('date,date\na,b\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate CSV headers'): read_raw(path)


def test_acquisition_kaggle_filename(monkeypatch, tmp_path):
    import io
    import zipfile
    import json
    from src.data.acquire import acquire
    data = (ROOT / 'data/raw/energydata_complete.csv').read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('KAG_energydata_complete.csv', data)
    class Response(io.BytesIO):
        url = 'https://example.test/mock-download'
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Response(stream.getvalue()))
    acquire(tmp_path)
    assert (tmp_path / 'data/raw/energydata_complete.csv').read_bytes() == data
    meta = json.loads((tmp_path / 'reports/data/provenance.json').read_text())
    assert meta['source'] == 'Kaggle'
    assert meta['archive_member'] == 'KAG_energydata_complete.csv'
    with pytest.raises(FileExistsError): acquire(tmp_path)
