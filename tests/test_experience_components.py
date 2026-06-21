import pandas as pd
import pytest

from actuarialpy.components import component_driver_analysis, component_trend, summarize_components
from actuarialpy.experience import status_summary, summarize_experience, summarize_views


def sample_df():
    return pd.DataFrame({
        "group_id": ["G1", "G1", "G2"],
        "product": ["PPO", "PPO", "HMO"],
        "status": ["active", "active", "termed"],
        "paid_claims": [100, 200, 300],
        "ibnr": [10, 20, 30],
        "premium": [200, 400, 600],
        "member_months": [2, 4, 6],
        "inpatient": [50, 100, 150],
        "outpatient": [60, 120, 180],
        "year": [2025, 2026, 2026],
    })


def test_summarize_experience_health_profile():
    result = summarize_experience(
        sample_df(),
        groupby="product",
        expense_cols=["paid_claims", "ibnr"],
        revenue_cols="premium",
        exposure_cols="member_months",
        profile="health",
    )
    ppo = result[result["product"] == "PPO"].iloc[0]
    assert ppo["total_expense"] == 330
    assert ppo["total_revenue"] == 600
    assert ppo["mlr"] == 0.55
    assert ppo["expense_pmpm"] == 55
    assert ppo["revenue_pmpm"] == 100


def test_custom_summary_labels():
    result = summarize_experience(
        sample_df(),
        groupby="product",
        expense_cols=["paid_claims", "ibnr"],
        revenue_cols="premium",
        total_expense_name="total_claims",
        total_revenue_name="total_premium",
        ratio_name="mlr",
    )
    assert "total_claims" in result.columns
    assert "total_premium" in result.columns


def test_reject_id_as_exposure():
    with pytest.raises(ValueError):
        summarize_experience(sample_df(), expense_cols="paid_claims", revenue_cols="premium", exposure_cols="group_id")


def test_summarize_views_and_status():
    views = summarize_views(sample_df(), views={"overall": None, "product": "product"}, expense_cols=["paid_claims", "ibnr"], revenue_cols="premium", exposure_cols="member_months")
    assert set(views) == {"overall", "product"}
    stat = status_summary(sample_df(), status_col="status", entity_col="group_id", expense_cols=["paid_claims", "ibnr"], revenue_cols="premium")
    assert stat.loc[stat["status"] == "active", "entity_count"].iloc[0] == 1


def test_components():
    result = summarize_components(sample_df(), groupby="product", component_cols=["inpatient", "outpatient"], exposure_col="member_months")
    ppo = result[result["product"] == "PPO"].iloc[0]
    assert ppo["total_expense"] == 330
    assert ppo["inpatient_pmpm"] == 25
    assert ppo["inpatient_share"] == pytest.approx(150 / 330)


def test_component_driver_analysis():
    df = pd.DataFrame({
        "year": [2025, 2026],
        "inpatient": [100, 120],
        "outpatient": [100, 150],
        "member_months": [10, 10],
    })
    out = component_driver_analysis(
        df,
        period_col="year",
        prior_period=2025,
        current_period=2026,
        component_cols=["inpatient", "outpatient"],
        exposure_col="member_months",
    )
    assert out.loc[out["component"] == "outpatient", "trend"].iloc[0] == pytest.approx(0.5)
    alias = component_trend(
        df,
        period_col="year",
        prior_period=2025,
        current_period=2026,
        component_cols=["inpatient", "outpatient"],
        exposure_col="member_months",
    )
    assert len(alias) == 2
