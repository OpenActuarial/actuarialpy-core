"""Tests for exposure measurement and age bases."""
import pandas as pd
import pytest

import actuarialpy as ap


# ----- age bases ----- #
def test_age_last_birthday():
    assert ap.age("1990-06-15", "2024-06-14", "last") == 33  # birthday not yet reached
    assert ap.age("1990-06-15", "2024-06-15", "last") == 34  # on the birthday


def test_age_exact_on_birthday():
    assert ap.age("1990-06-15", "2024-06-15", "exact") == pytest.approx(34.0)


def test_age_nearest_birthday():
    # ~0.75 into the age-year -> rounds up
    assert ap.age("1990-01-01", "2024-10-01", "nearest") == 35
    # ~0.25 into the age-year -> rounds down
    assert ap.age("1990-01-01", "2024-04-01", "nearest") == 34


def test_age_leap_day_birthday():
    assert ap.age("2000-02-29", "2024-02-29", "last") == 24
    # non-leap valuation year should not raise
    assert ap.age("2000-02-29", "2023-03-01", "last") == 23


def test_age_before_birth_raises():
    with pytest.raises(ValueError):
        ap.age("2000-01-01", "1999-01-01", "exact")


# ----- exposure years ----- #
def test_exposure_full_window():
    # in force across the whole 2023 study year (non-leap)
    assert ap.exposure_years("2020-01-01", "2025-01-01", "2023-01-01", "2024-01-01") == pytest.approx(1.0)


def test_exposure_partial_window():
    exp = ap.exposure_years("2023-07-01", "2026-01-01", "2023-01-01", "2024-01-01")
    assert exp == pytest.approx(184 / 365, rel=1e-9)


def test_exposure_no_overlap():
    assert ap.exposure_years("2025-01-01", "2026-01-01", "2023-01-01", "2024-01-01") == 0.0


def test_add_exposure_column():
    df = pd.DataFrame({
        "entry": ["2022-06-01", "2023-07-01"],
        "exit": ["2024-01-01", "2023-10-01"],
    })
    out = ap.add_exposure_column(df, "entry", "exit", "2023-01-01", "2024-01-01")
    # row 0: full 2023 year clipped -> 365/365 = 1.0
    assert out["exposure_years"].iloc[0] == pytest.approx(1.0)
    # row 1: Jul 1 to Oct 1 2023 = 92 days
    assert out["exposure_years"].iloc[1] == pytest.approx(92 / 365, rel=1e-9)


def test_add_exposure_column_missing_raises():
    df = pd.DataFrame({"entry": ["2023-01-01"]})
    with pytest.raises(ValueError):
        ap.add_exposure_column(df, "entry", "exit", "2023-01-01", "2024-01-01")
