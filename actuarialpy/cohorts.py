"""Cohort and duration summaries."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.experience import summarize_experience
from actuarialpy.periods import add_duration_column, add_period_column


def cohort_summary(
    df: pd.DataFrame,
    *,
    entity_col: str,
    date_col: str,
    start_date_col: str,
    duration_months: int = 12,
    groupby: str | Iterable[str] | None = None,
    expense_cols: str | Iterable[str],
    revenue_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    profile: str | None = None,
) -> pd.DataFrame:
    """Summarize each entity's first N months or cohort-duration window."""
    groups = as_list(groupby)
    validate_columns(df, [entity_col, date_col, start_date_col] + groups)
    temp = add_duration_column(df, start_date_col, date_col, duration_col="duration_month", one_based=True)
    temp = temp[(temp["duration_month"] >= 1) & (temp["duration_month"] <= duration_months)].copy()
    temp["first_month"] = pd.to_datetime(temp[start_date_col]).dt.to_period("M")
    temp["cohort_year"] = pd.to_datetime(temp[start_date_col]).dt.year
    return summarize_experience(
        temp,
        groupby=[entity_col, "first_month", "cohort_year"] + groups,
        expense_cols=expense_cols,
        revenue_cols=revenue_cols,
        exposure_cols=exposure_cols,
        profile=profile,
    )


def cohort_summary_by_period(
    cohort_df: pd.DataFrame,
    *,
    cohort_date_col: str = "first_month",
    freq: str = "Q",
    entity_col: str | None = None,
    expense_col: str = "total_expense",
    revenue_col: str = "total_revenue",
    exposure_cols: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Roll entity-level cohort summaries into cohort month/quarter/year buckets."""
    temp = cohort_df.copy()
    if cohort_date_col not in temp.columns:
        raise ValueError(f"Missing required column: {cohort_date_col}")
    if isinstance(temp[cohort_date_col].iloc[0], pd.Period):
        temp["cohort_period"] = temp[cohort_date_col].dt.asfreq(freq)
    else:
        temp = add_period_column(temp, cohort_date_col, freq, "cohort_period", copy=False)
    exposures = as_list(exposure_cols)
    summary = summarize_experience(
        temp,
        groupby="cohort_period",
        expense_cols=expense_col,
        revenue_cols=revenue_col,
        exposure_cols=exposures,
    )
    if entity_col:
        counts = temp.groupby("cohort_period", dropna=False)[entity_col].nunique().reset_index(name="entity_count")
        summary = counts.merge(summary, on="cohort_period", how="right")
    return summary


def duration_summary(
    df: pd.DataFrame,
    *,
    entity_col: str,
    date_col: str,
    start_date_col: str,
    expense_cols: str | Iterable[str],
    revenue_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    max_duration_month: int | None = None,
) -> pd.DataFrame:
    """Summarize experience by duration month since entity start."""
    temp = add_duration_column(df, start_date_col, date_col, duration_col="duration_month", one_based=True)
    temp = temp[temp["duration_month"] >= 1].copy()
    if max_duration_month is not None:
        temp = temp[temp["duration_month"] <= max_duration_month]
    return summarize_experience(
        temp,
        groupby="duration_month",
        expense_cols=expense_cols,
        revenue_cols=revenue_cols,
        exposure_cols=exposure_cols,
    )
