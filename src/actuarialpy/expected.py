"""Actual-versus-expected experience summaries."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from actuarialpy.columns import as_list, is_date_like, per_exposure_name, sum_columns, validate_columns
from actuarialpy.metrics import actual_to_expected as actual_to_expected_ratio, per_exposure, safe_divide


def _order_ave_columns(
    out: pd.DataFrame,
    *,
    groups: list[str],
    actuals: list[str],
    expecteds: list[str],
    exposures: list[str],
    actual_name: str,
    expected_name: str,
    ae_name: str,
    variance_name: str,
    variance_pct_name: str,
) -> pd.DataFrame:
    """Reorder actual-vs-expected summary columns into a consistent, readable layout.

    Order: date-like grouping columns, then other grouping columns, then exposure
    (volume), then the actual block (components, total, per-exposure rate), then the
    expected block, then the comparison metrics on the right -- variance and its
    per-exposure rate(s), variance percent, and finally the actual-to-expected ratio.
    Each total stays next to its own per-exposure rate, and the derived comparison
    metrics (variance, variance percent, ratio) are grouped together at the right.
    Unexpected columns are appended rather than dropped.
    """
    date_groups = [g for g in groups if is_date_like(out[g], g)]
    other_groups = [g for g in groups if g not in date_groups]
    actual_rates = [per_exposure_name(actual_name, e) for e in exposures]
    expected_rates = [per_exposure_name(expected_name, e) for e in exposures]
    variance_rates = [per_exposure_name(variance_name, e) for e in exposures]
    actual_block = list(actuals) + [actual_name] + actual_rates
    expected_block = list(expecteds) + [expected_name] + expected_rates
    variance_block = [variance_name] + variance_rates + [variance_pct_name]
    preferred = (
        date_groups + other_groups + list(exposures)
        + actual_block + expected_block + variance_block + [ae_name]
    )

    seen: set[str] = set()
    ordered: list[str] = []
    for col in preferred:
        if col in out.columns and col not in seen:
            seen.add(col)
            ordered.append(col)
    for col in out.columns:  # preserve anything not explicitly ordered
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return out[ordered]


def summarize_actual_vs_expected(
    df: pd.DataFrame,
    *,
    groupby: str | Iterable[str] | None = None,
    actual_cols: str | Iterable[str],
    expected_cols: str | Iterable[str],
    exposure_cols: str | Iterable[str] | None = None,
    actual_name: str = "actual",
    expected_name: str = "expected",
    ae_name: str = "actual_to_expected",
    variance_name: str = "variance",
    variance_pct_name: str = "variance_pct",
) -> pd.DataFrame:
    """Summarize actual-versus-expected results by optional grouping columns.

    Actual and expected amounts are aggregated before ratios are calculated.
    This makes the function suitable for claim costs, benefits, expenses,
    revenue, or any other actual-versus-expected measure.
    """
    groups = as_list(groupby)
    actuals = as_list(actual_cols)
    expecteds = as_list(expected_cols)
    exposures = as_list(exposure_cols)
    validate_columns(df, groups + actuals + expecteds + exposures)

    amount_cols = list(dict.fromkeys(actuals + expecteds + exposures))
    if groups:
        out = df[groups + amount_cols].groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
    else:
        out = pd.DataFrame({col: [df[col].sum()] for col in amount_cols})

    out[actual_name] = sum_columns(out, actuals)
    out[expected_name] = sum_columns(out, expecteds)
    out[ae_name] = actual_to_expected_ratio(out[actual_name], out[expected_name])
    out[variance_name] = out[actual_name] - out[expected_name]
    out[variance_pct_name] = safe_divide(out[variance_name], out[expected_name])

    for exposure in exposures:
        out[per_exposure_name(actual_name, exposure)] = per_exposure(out[actual_name], out[exposure])
        out[per_exposure_name(expected_name, exposure)] = per_exposure(out[expected_name], out[exposure])
        out[per_exposure_name(variance_name, exposure)] = per_exposure(out[variance_name], out[exposure])

    return _order_ave_columns(
        out,
        groups=groups,
        actuals=actuals,
        expecteds=expecteds,
        exposures=exposures,
        actual_name=actual_name,
        expected_name=expected_name,
        ae_name=ae_name,
        variance_name=variance_name,
        variance_pct_name=variance_pct_name,
    )
