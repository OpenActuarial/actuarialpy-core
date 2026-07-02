"""Tests for consistent, readable column ordering in experience summaries."""

import pandas as pd

from actuarialpy.experience import summarize_experience, summarize_views


def _df():
    return pd.DataFrame({
        "month": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
        "segment": ["EHPI", "HIP", "EHPI", "HIP"],
        "product": ["EPO", "PPO", "EPO", "PPO"],
        "inpatient": [50, 60, 55, 65],
        "outpatient": [30, 35, 33, 38],
        "premium": [120, 130, 125, 135],
        "member_months": [10, 12, 11, 13],
    })


def _summary(groupby, **kw):
    return summarize_experience(
        _df(), groupby=groupby,
        expense_cols=["inpatient", "outpatient"], revenue_cols="premium",
        exposure_cols=kw.pop("exposure_cols", "member_months"),
        profile=kw.pop("profile", "health"), **kw,
    )


def _adjacent(cols, a, b):
    return abs(cols.index(a) - cols.index(b)) == 1


def test_date_group_is_leftmost_regardless_of_groupby_order():
    cols = list(_summary(["segment", "month"]).columns)
    assert cols[0] == "month"  # moved ahead of segment


def test_total_is_adjacent_to_its_per_exposure():
    cols = list(_summary("segment").columns)
    assert _adjacent(cols, "total_expense", "total_expense_per_member_months")
    assert _adjacent(cols, "total_revenue", "total_revenue_per_member_months")


def test_ratio_is_last():
    assert list(_summary("segment").columns)[-1] == "mlr"


def test_expense_and_revenue_blocks_are_contiguous():
    cols = list(_summary("segment").columns)
    expense_block = ["inpatient", "outpatient", "total_expense", "total_expense_per_member_months"]
    revenue_block = ["premium", "total_revenue", "total_revenue_per_member_months"]
    e_idx = [cols.index(c) for c in expense_block]
    r_idx = [cols.index(c) for c in revenue_block]
    assert e_idx == list(range(min(e_idx), max(e_idx) + 1))  # no gaps
    assert r_idx == list(range(min(r_idx), max(r_idx) + 1))


def test_views_share_identical_metric_ordering():
    views = summarize_views(
        _df(),
        views={"by_month": ["segment", "month"], "by_segment": "segment",
               "by_product": ["product", "month"], "total": None},
        expense_cols=["inpatient", "outpatient"], revenue_cols="premium",
        exposure_cols="member_months", profile="health",
    )
    metrics = ["member_months", "inpatient", "outpatient", "total_expense",
               "total_expense_per_member_months", "premium", "total_revenue", "total_revenue_per_member_months", "mlr"]
    tails = {name: [c for c in v.columns if c in metrics] for name, v in views.items()}
    assert len({tuple(t) for t in tails.values()}) == 1  # all identical


def test_string_typed_month_still_moves_left():
    df = _df()
    df["month"] = df["month"].dt.strftime("%Y-%m")  # now a plain string column
    cols = list(summarize_experience(
        df, groupby=["segment", "month"],
        expense_cols="inpatient", revenue_cols="premium",
        exposure_cols="member_months", profile="health").columns)
    assert cols[0] == "month"  # detected by name


def test_non_date_groups_preserve_their_order():
    cols = list(_summary(["product", "segment"]).columns)
    assert cols.index("product") < cols.index("segment")


def test_no_exposure_means_no_per_exposure_and_ratio_last():
    cols = list(_summary("segment", exposure_cols=None).columns)
    assert not any("_per_" in c for c in cols)
    assert cols[-1] == "mlr"
    assert _adjacent(cols, "total_expense", "total_revenue") or "total_revenue" in cols


def test_multiple_exposures_grouped_in_their_blocks():
    df = _df()
    df["subscriber_months"] = [4, 5, 4, 6]
    cols = list(summarize_experience(
        df, groupby="segment",
        expense_cols="inpatient", revenue_cols="premium",
        exposure_cols=["member_months", "subscriber_months"], profile="health").columns)
    # both expense rates follow total_expense before revenue starts
    assert cols.index("total_expense") < cols.index("total_expense_per_member_months") < cols.index("total_expense_per_subscriber_months")
    assert cols.index("total_expense_per_subscriber_months") < cols.index("total_revenue")


def test_custom_total_names_are_ordered_too():
    cols = list(_summary("segment", total_expense_name="claims_total",
                          total_revenue_name="prem_total").columns)
    assert _adjacent(cols, "claims_total", "claims_total_per_member_months")
    assert _adjacent(cols, "prem_total", "prem_total_per_member_months")
