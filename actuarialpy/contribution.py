"""Contribution and driver-analysis primitives."""

from __future__ import annotations

import pandas as pd

from actuarialpy.columns import as_list, validate_columns
from actuarialpy.metrics import safe_divide


def share_of_total(component, total):
    """Calculate component share of total."""
    return safe_divide(component, total)


def contribution_to_change(component_change, total_change):
    """Calculate component contribution to a total change."""
    return safe_divide(component_change, total_change)


def top_contributors(
    df: pd.DataFrame,
    amount_col: str,
    *,
    n: int = 10,
    ascending: bool = False,
    by_abs: bool = False,
) -> pd.DataFrame:
    """Return top contributors by signed or absolute amount."""
    validate_columns(df, [amount_col])
    result = df.copy()
    if by_abs:
        sort_col = "_abs_sort_amount"
        result[sort_col] = result[amount_col].abs()
        result = result.sort_values(sort_col, ascending=ascending).drop(columns=sort_col).head(n)
    else:
        result = result.sort_values(amount_col, ascending=ascending).head(n)
    return result.copy()


def component_contribution(
    df: pd.DataFrame,
    *,
    component_cols,
    total_col: str | None = None,
    prefix: str = "share",
    copy: bool = True,
) -> pd.DataFrame:
    """Add component share-of-total columns for a set of component columns."""
    components = as_list(component_cols)
    validate_columns(df, components)
    result = df.copy() if copy else df
    if total_col is None:
        total_col = "total_component"
        result[total_col] = result[components].sum(axis=1)
    else:
        validate_columns(result, [total_col])
    for col in components:
        result[f"{col}_{prefix}"] = share_of_total(result[col], result[total_col])
    return result
