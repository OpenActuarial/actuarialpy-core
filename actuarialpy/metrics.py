"""Core actuarial metrics."""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np


def safe_divide(numerator: Any, denominator: Any) -> Any:
    """Safely divide numerator by denominator.

    Returns ``np.nan`` where the denominator is zero.

    Parameters
    ----------
    numerator:
        Scalar or array-like numerator.
    denominator:
        Scalar or array-like denominator.

    Returns
    -------
    Any
        Division result. Scalars return scalars; array-like inputs return NumPy arrays.

    Examples
    --------
    >>> safe_divide(10, 2)
    5.0
    >>> np.isnan(safe_divide(10, 0))
    True
    """
    if isinstance(numerator, Number) and isinstance(denominator, Number):
        return np.nan if denominator == 0 else numerator / denominator

    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)

    return np.divide(
        numerator_arr,
        denominator_arr,
        out=np.full_like(numerator_arr, np.nan, dtype=float),
        where=denominator_arr != 0,
    )


def loss_ratio(expenses: Any, revenue: Any) -> Any:
    """Calculate a loss ratio.

    The loss ratio is defined generally as expenses divided by revenue.

    Parameters
    ----------
    expenses:
        Claims, losses, benefits, or other expense amount.
    revenue:
        Premium, earned premium, revenue, or other denominator amount.

    Returns
    -------
    Any
        Loss ratio as a decimal.

    Examples
    --------
    >>> loss_ratio(850_000, 1_000_000)
    0.85
    """
    return safe_divide(expenses, revenue)
