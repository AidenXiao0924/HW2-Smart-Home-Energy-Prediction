"""Pre-Test contract checks: never read Test data here."""
import ast
from pathlib import Path
import pandas as pd
import pytest
import src.evaluation.c2 as c2
from src.evaluation.c2 import TestGate,coverage_audit,forbid_model_updates


def test_evaluation_source_has_no_fit_or_selection_calls():
    tree=ast.parse(Path(c2.__file__).read_text())
    forbidden={'fit','fit_transform','partial_fit','select_features','tune_scheme','quantile','GridSearchCV'}
    calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
           for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not forbidden.intersection(calls)


@pytest.mark.parametrize('enabled,name',[(False,'test'),(True,'train'),(True,'validation'),(True,'calibration')])
def test_phase_dataset_boundaries(enabled,name):
    with TestGate(enabled),pytest.raises(PermissionError):
        (c2.ROOT/f'data/processed/{name}.csv').read_bytes()


def test_guard_blocks_predictive_model_fit():
    from sklearn.linear_model import LinearRegression
    with forbid_model_updates() as attempts,pytest.raises(RuntimeError):
        getattr(LinearRegression(),'fit')([[1],[2]],[1,2])
    assert attempts==['FORBIDDEN_MODEL_UPDATE']


def test_invalid_intervals_do_not_inflate_all_row_coverage():
    frame=pd.DataFrame(dict(actual=[1,1,10],prediction=[1,-5,1],lower=[0,0,0],upper=[2,-1,2],
        interval_valid=[True,False,True],warnings=['','NEGATIVE_ESTIMATE_WARNING|INVALID_INTERVAL','']))
    result=coverage_audit(frame)
    assert result['coverage_valid']==.5 and result['coverage_all']==1/3
    assert result['valid_intervals']==2 and result['invalid_intervals']==1
    assert result['above_upper_count_valid']==1
    assert result['negative_warning_count']==1 and result['invalid_warning_count']==1


def test_prior_attempt_prevents_new_target_read(tmp_path,monkeypatch):
    (tmp_path/'first_evaluation.json').write_text('{}')
    monkeypatch.setattr(c2,'OUT',tmp_path)
    def blocked():raise AssertionError('Must not start preflight')
    monkeypatch.setattr(c2,'preflight',blocked)
    with TestGate(),pytest.raises(RuntimeError,match='already recorded'):c2.evaluate()


def test_bad_artifact_stops_before_test_hash(monkeypatch):
    original=c2.sha
    def changed(path):
        if Path(path)==c2.MODEL:return 'corrupt'
        return original(path)
    monkeypatch.setattr(c2,'sha',changed)
    with TestGate(),pytest.raises(ValueError,match='artifact changed'):c2.preflight()
