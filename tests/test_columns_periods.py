import pandas as pd
import pytest

from actuarialpy.columns import ensure_unique_keys, sum_columns, validate_columns
from actuarialpy.periods import add_duration_column, add_period_column, months_between


def test_column_helpers():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert sum_columns(df, ["a", "b"]).tolist() == [4, 6]
    with pytest.raises(ValueError):
        validate_columns(df, ["c"])


def test_unique_keys():
    df = pd.DataFrame({"a": [1, 1]})
    with pytest.raises(ValueError):
        ensure_unique_keys(df, "a")


def test_period_helpers():
    df = pd.DataFrame({"month": ["2026-01-15"], "effective": ["2026-01-01"]})
    out = add_period_column(df, "month", "Q", "quarter")
    assert str(out.loc[0, "quarter"]) == "2026Q1"
    dur = add_duration_column(df, "effective", "month")
    assert dur.loc[0, "duration_month"] == 1
    assert months_between("2025-01-01", "2026-03-01") == 14
