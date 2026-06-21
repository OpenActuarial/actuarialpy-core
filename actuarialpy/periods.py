"""Date and period helper primitives."""

from __future__ import annotations

import pandas as pd


def to_period(values, freq: str):
    """Convert scalar or array-like date values to pandas Period values."""
    return pd.to_datetime(values).to_period(freq)


def add_period_column(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    period_col: str | None = None,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add a pandas Period column from a date column.

    Common frequencies include ``M``, ``Q``, and ``Y``.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required column: {date_col}")
    result = df.copy() if copy else df
    name = period_col or f"{date_col}_{freq.lower()}"
    result[name] = pd.to_datetime(result[date_col]).dt.to_period(freq)
    return result


def period_label(period, *, fmt: str | None = None) -> str:
    """Format a pandas Period or date-like value as a string label."""
    if isinstance(period, pd.Period):
        return str(period)
    value = pd.to_datetime(period)
    return value.strftime(fmt) if fmt else str(value.date())


def months_between(start, end) -> int:
    """Number of whole month boundaries between two date-like values."""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return (e.year - s.year) * 12 + (e.month - s.month)


def add_duration_column(
    df: pd.DataFrame,
    start_col: str,
    date_col: str,
    duration_col: str = "duration_month",
    *,
    one_based: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Add elapsed duration in months between a start date and an observation date."""
    for col in [start_col, date_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    result = df.copy() if copy else df
    start = pd.to_datetime(result[start_col])
    date = pd.to_datetime(result[date_col])
    duration = (date.dt.year - start.dt.year) * 12 + (date.dt.month - start.dt.month)
    if one_based:
        duration = duration + 1
    result[duration_col] = duration
    return result
