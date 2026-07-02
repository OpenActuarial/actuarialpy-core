r"""Explicit-weight aggregation for quantities that cannot be summed.

Additive amounts (claims, premium, exposure) roll up by summation, and
ratios of them roll up as ratios of sums -- that is
:func:`actuarialpy.summarize_experience`'s contract. Quantities that are
*already* rates or ratios at the row level (rate actions, trend
assumptions, persistency) cannot be summed and must be averaged with an
explicit weight:

.. math::

    \bar{x}_w = \frac{\sum_i w_i x_i}{\sum_i w_i}.

The weight is a **required** argument everywhere in this module. An
unweighted mean of rate actions silently equal-weights a small risk with a
large one; forcing the caller to name the weight (premium, exposure, ...)
makes that choice visible and reviewable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from actuarialpy.columns import as_list, validate_columns


def weighted_mean(values: Any, weights: Any, *, skipna: bool = False) -> float:
    """Weighted mean with validated, explicit weights.

    Parameters
    ----------
    values : array-like
        Row-level rates or ratios to average.
    weights : array-like
        Non-negative, finite weights, same length as ``values``, with a
        positive total.
    skipna : bool
        When True, pairs where the value is NaN are dropped before
        averaging. Default False: a NaN value propagates to the result, so
        missing data surfaces instead of silently shrinking the base.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError(
            f"values and weights must have the same shape, got {v.shape} and {w.shape}"
        )
    if v.ndim != 1:
        v = v.ravel()
        w = w.ravel()
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if skipna:
        keep = ~np.isnan(v)
        v, w = v[keep], w[keep]
    total = w.sum()
    if not total > 0:
        raise ValueError("weights must sum to a positive total")
    return float((v * w).sum() / total)


def weighted_summary(
    df: pd.DataFrame,
    *,
    value_cols: str | Iterable[str],
    weight_col: str,
    groupby: str | Iterable[str] | None = None,
    skipna: bool = False,
) -> pd.DataFrame:
    """Grouped weighted means of one or more value columns.

    Each value column ``x`` produces ``x_weighted`` =
    :math:`\\sum wx / \\sum w` per group; the weight total is reported as
    ``{weight_col}_total`` so the base of every average is visible.

    Typical use: premium-weighted rate actions by cohort, exposure-weighted
    persistency by segment.
    """
    values = as_list(value_cols)
    groups = as_list(groupby)
    validate_columns(df, groups + values + [weight_col])

    work = df[groups + values + [weight_col]].copy()
    w = work[weight_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{weight_col!r} must be finite")
    if np.any(w < 0):
        raise ValueError(f"{weight_col!r} must be non-negative")

    for col in values:
        mask = work[col].isna() if skipna else pd.Series(False, index=work.index)
        work[f"_wx_{col}"] = work[col].fillna(0.0) * work[weight_col]
        work[f"_w_{col}"] = work[weight_col].where(~mask, 0.0)

    if groups:
        agg = work.groupby(groups, dropna=False, as_index=False).sum(numeric_only=True)
    else:
        num_cols = [c for c in work.columns if c not in groups]
        agg = pd.DataFrame({col: [work[col].sum()] for col in num_cols})

    out = agg[groups].copy() if groups else pd.DataFrame(index=[0])
    for col in values:
        weight_sum = agg[f"_w_{col}"]
        if (weight_sum <= 0).any():
            raise ValueError(
                f"weights sum to zero for at least one group when averaging {col!r}"
            )
        out[f"{col}_weighted"] = agg[f"_wx_{col}"] / weight_sum
        if not skipna:
            nan_groups = (
                work.assign(_isna=work[col].isna())
                .groupby(groups, dropna=False)["_isna"]
                .any()
                .to_numpy()
                if groups
                else np.array([work[col].isna().any()])
            )
            out.loc[nan_groups, f"{col}_weighted"] = np.nan
    out[f"{weight_col}_total"] = agg[weight_col].to_numpy()
    return out.reset_index(drop=True)
