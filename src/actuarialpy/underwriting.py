r"""Underwriting income statement: the two-tier margin view.

Underwriting results are reported in two tiers, generically across lines of
business. **Gross margin** is revenue less loss (claim / benefit) expense and
excludes operating expense -- which is also why operating expense never
enters a loss ratio. **Gain / (loss)** is gross margin less operating
expense: the underwriting result.

.. math::

    \text{total revenue} &= \textstyle\sum \text{revenue components
        (premium, refunds, recasts, ...)} \\
    \text{total loss}    &= \textstyle\sum \text{loss components
        (claims by category, benefits, ...)} \\
    \text{gross margin}  &= \text{total revenue} - \text{total loss} \\
    \text{gain / (loss)} &= \text{gross margin} - \text{total expense}

The three ratios mirror :func:`actuarialpy.loss_ratio`,
:func:`actuarialpy.expense_ratio`, and :func:`actuarialpy.combined_ratio`.
Component labels and ratio names are the caller's vocabulary: the library
only sums the components, and domain naming (a health shop's ``mlr``, a life
shop's ``benefit_ratio``) comes from the ``profile`` / ``labels`` options on
the output views -- never from the calculation itself.

Ratio conventions differ across shops -- and often across metrics on the same
exhibit -- so denominators here are **explicit parameters**, never
assumptions:

* ``loss_ratio``: total loss / *loss-ratio denominator*. Default
  ``"total_revenue"`` (revenue net of refunds and other offsets).
* ``expense_ratio``: total expense / *expense-ratio denominator*. Default
  ``"premium"``: the gross premium component named by ``premium_label``,
  before refunds.
* ``gain ratio``: gain / *gain denominator*. Default ``"total_revenue"``.

With mixed denominators the identity ``gain ratio = 1 - combined ratio``
holds only approximately; it is exact when every denominator is the same
series. :meth:`UnderwritingSummary.reconciliation` reports the difference so
the convention drift is visible instead of silent.

These are management / pricing metrics. Regulated ratio calculations (for
example, a rebate loss ratio prescribed by statute) have their own numerator
and denominator adjustments and are out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from actuarialpy.columns import as_list, per_exposure_name, sum_columns, validate_columns
from actuarialpy.metrics import per_exposure, safe_divide
from actuarialpy.profiles import apply_profile_labels, get_profile_defaults

_DENOMINATORS = ("total_revenue", "premium")

def _validate_denominator(value: str, name: str) -> str:
    if value not in _DENOMINATORS:
        raise ValueError(
            f"{name} must be one of {_DENOMINATORS}, got {value!r}"
        )
    return value


def _ratio_rename(profile: str | None, labels: Mapping[str, str] | None) -> dict[str, str]:
    """Output renames: the profile's ratio name for ``loss_ratio``, then labels."""
    rename: dict[str, str] = {}
    ratio_col = get_profile_defaults(profile).get("ratio_col")
    if ratio_col is not None and ratio_col != "loss_ratio":
        rename["loss_ratio"] = ratio_col
    rename.update(dict(labels or {}))
    return rename


@dataclass
class UnderwritingSummary:
    """Two-tier underwriting income statement for a single entity or period.

    Parameters
    ----------
    revenue : Mapping[str, float]
        Labeled revenue components (e.g. ``{"premium": ..., "refund": ...}``).
        Offsets such as refunds should be signed (negative). The library
        never interprets the labels; it only sums them.
    losses : Mapping[str, float]
        Labeled loss components -- claim or benefit expense by whatever
        categories the caller uses.
    expenses : Mapping[str, float] | float
        Operating expense, itemized or as a single amount. Default 0.
    exposure : float, optional
        Exposure units (member months, policy months, earned exposures, ...)
        for per-exposure figures. Required only when a ``*_per_exposure``
        property is accessed.
    premium_label : str
        Which revenue component is the gross premium, used when a
        denominator is ``"premium"``. Default ``"premium"``.
    loss_ratio_denominator, expense_ratio_denominator, gain_denominator : str
        ``"total_revenue"`` or ``"premium"``. Defaults follow the common
        exhibit convention: loss and gain ratios over total revenue, expense
        ratio over gross premium.

    Examples
    --------
    >>> uw = UnderwritingSummary(
    ...     revenue={"premium": 1_200_000.0, "refund": -4_000.0},
    ...     losses={"claims": 1_090_000.0},
    ...     expenses=110_000.0,
    ...     exposure=3_000.0,
    ... )
    >>> round(uw.gross_margin, 0)
    106000.0
    >>> round(uw.gain, 0)
    -4000.0
    """

    revenue: Mapping[str, float]
    losses: Mapping[str, float]
    expenses: Mapping[str, float] | float = 0.0
    exposure: float | None = None
    premium_label: str = "premium"
    loss_ratio_denominator: str = "total_revenue"
    expense_ratio_denominator: str = "premium"
    gain_denominator: str = "total_revenue"
    _expense_items: Mapping[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.revenue:
            raise ValueError("revenue must contain at least one component")
        if not self.losses:
            raise ValueError("losses must contain at least one component")
        if isinstance(self.expenses, Mapping):
            self._expense_items = dict(self.expenses)
        else:
            self._expense_items = {"expense": float(self.expenses)}
        for name in (
            "loss_ratio_denominator",
            "expense_ratio_denominator",
            "gain_denominator",
        ):
            _validate_denominator(getattr(self, name), name)
        uses_premium = "premium" in (
            self.loss_ratio_denominator,
            self.expense_ratio_denominator,
            self.gain_denominator,
        )
        if uses_premium and self.premium_label not in self.revenue:
            raise ValueError(
                f"premium_label {self.premium_label!r} is not a revenue "
                f"component; available: {sorted(self.revenue)}"
            )
        if self.exposure is not None and not self.exposure > 0:
            raise ValueError(
                f"exposure must be positive when provided, got {self.exposure!r}"
            )

    @classmethod
    def from_per_exposure(
        cls,
        *,
        revenue_per_exposure: Mapping[str, float],
        loss_per_exposure: Mapping[str, float],
        expense_per_exposure: Mapping[str, float] | float = 0.0,
        exposure: float,
        **kwargs: Any,
    ) -> "UnderwritingSummary":
        """Build a summary from per-exposure components and total exposure.

        Forecast exhibits are usually stated per exposure unit (PMPM in a
        health shop, per policy month in life); this converts each component
        to amounts by ``exposure`` so totals, per-exposure figures, and
        ratios all come from one set of inputs.
        """
        if not exposure > 0:
            raise ValueError(f"exposure must be positive, got {exposure!r}")
        units = float(exposure)
        if isinstance(expense_per_exposure, Mapping):
            expenses: Mapping[str, float] | float = {
                k: v * units for k, v in expense_per_exposure.items()
            }
        else:
            expenses = float(expense_per_exposure) * units
        return cls(
            revenue={k: v * units for k, v in revenue_per_exposure.items()},
            losses={k: v * units for k, v in loss_per_exposure.items()},
            expenses=expenses,
            exposure=units,
            **kwargs,
        )

    # ----- totals ----- #
    @property
    def total_revenue(self) -> float:
        return float(sum(self.revenue.values()))

    @property
    def total_loss(self) -> float:
        return float(sum(self.losses.values()))

    @property
    def total_expense(self) -> float:
        return float(sum(self._expense_items.values()))

    @property
    def gross_margin(self) -> float:
        """Tier one: total revenue less loss expense (operating expense excluded)."""
        return self.total_revenue - self.total_loss

    @property
    def gain(self) -> float:
        """Tier two: gross margin less operating expense."""
        return self.gross_margin - self.total_expense

    # ----- ratios (explicit denominators) ----- #
    def _denominator(self, which: str) -> float:
        if which == "total_revenue":
            return self.total_revenue
        return float(self.revenue[self.premium_label])

    @property
    def loss_ratio(self) -> float:
        """Loss expense over the ``loss_ratio_denominator``."""
        return float(
            safe_divide(self.total_loss, self._denominator(self.loss_ratio_denominator))
        )

    @property
    def expense_ratio(self) -> float:
        """Operating expense over the ``expense_ratio_denominator``."""
        return float(
            safe_divide(
                self.total_expense, self._denominator(self.expense_ratio_denominator)
            )
        )

    @property
    def combined_ratio(self) -> float:
        """Loss ratio plus expense ratio, each on its own denominator."""
        return self.loss_ratio + self.expense_ratio

    @property
    def gross_margin_ratio(self) -> float:
        """Gross margin over the ``loss_ratio_denominator`` (its complement)."""
        return float(
            safe_divide(
                self.gross_margin, self._denominator(self.loss_ratio_denominator)
            )
        )

    @property
    def gain_ratio(self) -> float:
        """Gain / (loss) over the ``gain_denominator``."""
        return float(
            safe_divide(self.gain, self._denominator(self.gain_denominator))
        )

    def reconciliation(self) -> float:
        """``gain_ratio - (1 - combined_ratio)``: the mixed-denominator gap.

        Zero when every denominator is the same series; otherwise the size
        of the drift introduced by quoting the loss, expense, and gain
        ratios over different bases. Useful as an exhibit footnote or a
        data-quality check.
        """
        return self.gain_ratio - (1.0 - self.combined_ratio)

    # ----- per exposure ----- #
    def _require_exposure(self) -> float:
        if self.exposure is None:
            raise ValueError(
                "exposure is required for per-exposure figures; pass it to "
                "the constructor or use from_per_exposure(...)"
            )
        return float(self.exposure)

    @property
    def revenue_per_exposure(self) -> float:
        return self.total_revenue / self._require_exposure()

    @property
    def loss_per_exposure(self) -> float:
        return self.total_loss / self._require_exposure()

    @property
    def expense_per_exposure(self) -> float:
        return self.total_expense / self._require_exposure()

    @property
    def gross_margin_per_exposure(self) -> float:
        return self.gross_margin / self._require_exposure()

    @property
    def gain_per_exposure(self) -> float:
        return self.gain / self._require_exposure()

    # ----- views ----- #
    def to_frame(
        self,
        *,
        profile: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """One tidy row of every total and ratio (per-exposure when given).

        ``profile`` renames only the loss-ratio column to the domain's ratio
        name (``"health"`` -> ``mlr``, ``"life"`` -> ``benefit_ratio``);
        ``labels`` renames any output column. Calculations are unaffected.
        """
        row: dict[str, float] = {
            "total_revenue": self.total_revenue,
            "total_loss": self.total_loss,
            "total_expense": self.total_expense,
            "gross_margin": self.gross_margin,
            "gain": self.gain,
            "loss_ratio": self.loss_ratio,
            "expense_ratio": self.expense_ratio,
            "combined_ratio": self.combined_ratio,
            "gross_margin_ratio": self.gross_margin_ratio,
            "gain_ratio": self.gain_ratio,
        }
        if self.exposure is not None:
            row["exposure"] = float(self.exposure)
            row["revenue_per_exposure"] = self.revenue_per_exposure
            row["loss_per_exposure"] = self.loss_per_exposure
            row["expense_per_exposure"] = self.expense_per_exposure
            row["gross_margin_per_exposure"] = self.gross_margin_per_exposure
            row["gain_per_exposure"] = self.gain_per_exposure
        frame = pd.DataFrame([row])
        return frame.rename(columns=_ratio_rename(profile, labels))

    def statement(
        self,
        *,
        profile: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> pd.Series:
        """Exhibit-shaped Series: components, subtotals, tiers, then ratios."""
        lines: dict[str, float] = {}
        for label, value in self.revenue.items():
            lines[label] = float(value)
        lines["total_revenue"] = self.total_revenue
        for label, value in self.losses.items():
            lines[label] = float(value)
        lines["total_loss"] = self.total_loss
        lines["loss_ratio"] = self.loss_ratio
        lines["gross_margin"] = self.gross_margin
        for label, value in self._expense_items.items():
            lines[label] = float(value)
        lines["total_expense"] = self.total_expense
        lines["expense_ratio"] = self.expense_ratio
        lines["gain"] = self.gain
        lines["gain_ratio"] = self.gain_ratio
        series = pd.Series(lines, name="statement")
        return series.rename(index=_ratio_rename(profile, labels))


def underwriting_summary(
    df: pd.DataFrame,
    *,
    groupby: str | Iterable[str] | None = None,
    revenue_cols: str | Iterable[str],
    loss_cols: str | Iterable[str],
    expense_cols: str | Iterable[str],
    exposure_col: str | None = None,
    premium_col: str | None = None,
    loss_ratio_denominator: str = "total_revenue",
    expense_ratio_denominator: str = "premium",
    gain_denominator: str = "total_revenue",
    profile: str | None = None,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Grouped two-tier underwriting summary from a tidy table.

    Component columns are **summed first** and every ratio is computed on the
    aggregated totals (ratio of sums, never an average of row-level ratios) --
    the same contract as :func:`actuarialpy.summarize_experience`.

    Parameters
    ----------
    df : pd.DataFrame
        One row per entity / period at whatever grain is being rolled up.
    groupby : str | Iterable[str], optional
        Grouping columns; omit for a single all-rows summary.
    revenue_cols, loss_cols, expense_cols : str | Iterable[str]
        Component columns for each tier. Revenue offsets (refunds) should be
        signed.
    exposure_col : str, optional
        Exposure column; adds ``{amount}_per_{exposure_col}`` output columns.
        Domain-style names (a health shop's ``_pmpm``) are applied via
        ``labels``, never inferred from the column name.
    premium_col : str, optional
        Gross premium column, required when any denominator is
        ``"premium"``.
    loss_ratio_denominator, expense_ratio_denominator, gain_denominator : str
        ``"total_revenue"`` or ``"premium"``; see the module docstring for
        the convention discussion.
    profile : str, optional
        Renames only the loss-ratio column to the domain's ratio name
        (``"health"`` -> ``mlr``, ``"life"`` -> ``benefit_ratio``).
    labels : dict, optional
        Explicit output column renames, applied after ``profile``.

    Returns
    -------
    pd.DataFrame
        Group keys, component sums, ``total_revenue``, ``total_loss``,
        ``total_expense``, ``gross_margin``, ``gain``, the three ratios plus
        ``gross_margin_ratio`` and ``gain_ratio``, and per-exposure columns
        when ``exposure_col`` is given.
    """
    groups = as_list(groupby)
    revenues = as_list(revenue_cols)
    losses = as_list(loss_cols)
    expenses = as_list(expense_cols)
    for name, value in (
        ("loss_ratio_denominator", loss_ratio_denominator),
        ("expense_ratio_denominator", expense_ratio_denominator),
        ("gain_denominator", gain_denominator),
    ):
        _validate_denominator(value, name)
    uses_premium = "premium" in (
        loss_ratio_denominator,
        expense_ratio_denominator,
        gain_denominator,
    )
    if uses_premium and premium_col is None:
        raise ValueError(
            'premium_col is required when any denominator is "premium"'
        )

    amount_cols = list(dict.fromkeys(revenues + losses + expenses))
    if premium_col is not None:
        validate_columns(df, [premium_col])
        if premium_col not in amount_cols:
            amount_cols.append(premium_col)
    exposures = [exposure_col] if exposure_col is not None else []
    validate_columns(df, groups + amount_cols + exposures)

    if groups:
        summary = (
            df[groups + amount_cols + exposures]
            .groupby(groups, dropna=False, as_index=False)
            .sum(numeric_only=True)
        )
    else:
        summary = pd.DataFrame(
            {col: [df[col].sum()] for col in amount_cols + exposures}
        )

    summary["total_revenue"] = sum_columns(summary, revenues)
    summary["total_loss"] = sum_columns(summary, losses)
    summary["total_expense"] = sum_columns(summary, expenses)
    summary["gross_margin"] = summary["total_revenue"] - summary["total_loss"]
    summary["gain"] = summary["gross_margin"] - summary["total_expense"]

    def _denom(which: str) -> pd.Series:
        if which == "total_revenue":
            return summary["total_revenue"]
        return summary[premium_col]

    summary["loss_ratio"] = safe_divide(
        summary["total_loss"], _denom(loss_ratio_denominator)
    )
    summary["expense_ratio"] = safe_divide(
        summary["total_expense"], _denom(expense_ratio_denominator)
    )
    summary["combined_ratio"] = summary["loss_ratio"] + summary["expense_ratio"]
    summary["gross_margin_ratio"] = safe_divide(
        summary["gross_margin"], _denom(loss_ratio_denominator)
    )
    summary["gain_ratio"] = safe_divide(summary["gain"], _denom(gain_denominator))

    ordered = (
        groups
        + [c for c in amount_cols if c in summary.columns]
        + exposures
        + [
            "total_revenue",
            "total_loss",
            "total_expense",
            "gross_margin",
            "gain",
            "loss_ratio",
            "expense_ratio",
            "combined_ratio",
            "gross_margin_ratio",
            "gain_ratio",
        ]
    )
    if exposure_col is not None:
        for amount, base in (
            ("total_revenue", "revenue"),
            ("total_loss", "loss"),
            ("total_expense", "expense"),
            ("gross_margin", "gross_margin"),
            ("gain", "gain"),
        ):
            name = per_exposure_name(base, exposure_col)
            summary[name] = per_exposure(summary[amount], summary[exposure_col])
            ordered.append(name)
    summary = summary[list(dict.fromkeys(ordered))]
    return apply_profile_labels(
        summary.rename(columns=_ratio_rename(profile, None)), labels=labels
    )
