"""Tests for consistent, readable column ordering in actual-vs-expected summaries."""

import pandas as pd

from actuarialpy.expected import summarize_actual_vs_expected


def _df():
    return pd.DataFrame({
        "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
        "segment": ["EHPI", "EHPI", "HIP", "HIP"],
        "actual_claims": [100, 120, 90, 110],
        "expected_claims": [95, 115, 100, 105],
        "member_months": [10, 11, 8, 9],
    })


def _ave(groupby="segment", **kw):
    return summarize_actual_vs_expected(
        _df(), groupby=groupby,
        actual_cols="actual_claims", expected_cols="expected_claims",
        exposure_cols=kw.pop("exposure_cols", "member_months"), **kw,
    )


def _adjacent(cols, a, b):
    return abs(cols.index(a) - cols.index(b)) == 1


def test_date_is_leftmost():
    cols = list(_ave(["segment", "month"]).columns)
    assert cols[0] == "month"


def test_exposure_sits_between_groups_and_amounts():
    cols = list(_ave(["segment", "month"]).columns)
    assert cols.index("member_months") < cols.index("actual")
    assert cols.index("segment") < cols.index("member_months")


def test_each_amount_next_to_its_per_member_months():
    cols = list(_ave().columns)
    assert _adjacent(cols, "actual", "actual_per_member_months")
    assert _adjacent(cols, "expected", "expected_per_member_months")


def test_variance_family_grouped_on_the_right():
    cols = list(_ave().columns)
    # variance, variance_per_member_months, variance_pct contiguous, after the expected block
    assert _adjacent(cols, "variance", "variance_per_member_months")
    assert _adjacent(cols, "variance_per_member_months", "variance_pct")
    assert cols.index("variance") > cols.index("expected")


def test_ratio_is_last():
    assert list(_ave().columns)[-1] == "actual_to_expected"


def test_components_precede_their_totals():
    cols = list(_ave().columns)
    assert cols.index("actual_claims") < cols.index("actual")


def test_no_exposure_drops_per_exposure_keeps_order():
    cols = list(_ave(exposure_cols=None).columns)
    assert not any("_per_" in c for c in cols)
    assert cols.index("variance") < cols.index("variance_pct") < cols.index("actual_to_expected")
    assert cols[-1] == "actual_to_expected"


def test_custom_names_are_ordered_too():
    cols = list(_ave(actual_name="paid", expected_name="forecast",
                     variance_name="diff", ae_name="paid_to_forecast").columns)
    assert _adjacent(cols, "paid", "paid_per_member_months")
    assert _adjacent(cols, "forecast", "forecast_per_member_months")
    assert cols[-1] == "paid_to_forecast"
    assert cols.index("diff") < cols.index("paid_to_forecast")
