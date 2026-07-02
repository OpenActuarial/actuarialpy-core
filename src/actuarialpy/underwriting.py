r"""Underwriting income statement: the two-tier margin view.

Group underwriting results are reported in two tiers. **Gross margin** (the
medical or benefit margin) is revenue less benefit expense, and excludes
administrative cost -- which is also why administrative expense never enters
a medical loss ratio. **Gain / (loss)** is gross margin less administrative
expense: the underwriting result.

.. math::

    \text{total revenue} &= \textstyle\sum \text{revenue components
        (premium, refunds, recasts, ...)} \\
    \text{total benefit} &= \textstyle\sum \text{benefit components
        (inpatient, outpatient, professional, pharmacy, ...)} \\
    \text{gross margin}  &= \text{total revenue} - \text{total benefit} \\
    \text{gain / (loss)} &= \text{gross margin} - \text{total admin}

Ratio conventions differ across shops -- and often across metrics on the same
exhibit -- so denominators here are **explicit parameters**, never
assumptions:

* ``mcr``: total benefit / *mcr denominator*. Default ``"total_revenue"``
  (premium net of refunds and other revenue offsets).
* ``aer``: total admin / *aer denominator*. Default ``"premium"``: the gross
  premium component named by ``premium_label``, before refunds.
* ``gain ratio``: gain / *gain denominator*. Default ``"total_revenue"``.

With mixed denominators the identity ``gain ratio = 1 - MCR - AER`` holds
only approximately; it is exact when every denominator is the same series.
:meth:`UnderwritingSummary.reconciliation` reports the difference so the
convention drift is visible instead of silent.

These are *management / pricing* metrics. The ACA rebate MLR is a separate
regulated calculation with its own numerator and denominator adjustments and
is out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from actuarialpy.columns import as_list, sum_columns, validate_columns
from actuarialpy.metrics import per_exposure, safe_divide

_DENOMINATORS = ("total_revenue", "premium")


def _validate_denominator(value: str, name: str) -> str:
    if value not in _DENOMINATORS:
        raise ValueError(
            f"{name} must be one of {_DENOMINATORS}, got {value!r}"
        )
    return value


@dataclass
class UnderwritingSummary:
    """Two-tier underwriting income statement for a single entity or period.

    Parameters
    ----------
    revenue : Mapping[str, float]
        Labeled revenue components in dollars (e.g. ``{"premium": ...,
        "refund": ...}``). Offsets such as refunds should be signed
        (negative). The library never interprets the labels; it only sums
        them -- component vocabulary stays with the caller.
    benefit : Mapping[str, float]
        Labeled benefit (medical / claims) expense components in dollars.
    admin : Mapping[str, float] | float
        Administrative expense, itemized or as a single amount. Default 0.
    member_months : float, optional
        Exposure used for per-member-per-month figures. Required only when a
        ``*_pmpm`` property is accessed.
    premium_label : str
        Which revenue component is the gross premium, used when a
        denominator is ``"premium"``. Default ``"premium"``.
    mcr_denominator, aer_denominator, gain_denominator : str
        ``"total_revenue"`` or ``"premium"``. Defaults follow the common
        exhibit convention: MCR and gain ratio over total revenue, AER over
        gross premium.

    Examples
    --------
    >>> uw = UnderwritingSummary(
    ...     revenue={"premium": 1_200_000.0, "refund": -4_000.0},
    ...     benefit={"inpatient": 500_000.0, "outpatient": 590_000.0},
    ...     admin=110_000.0,
    ...     member_months=3_000.0,
    ... )
    >>> round(uw.gross_margin, 0)
    106000.0
    >>> round(uw.gain, 0)
    -4000.0
    """

    revenue: Mapping[str, float]
    benefit: Mapping[str, float]
    admin: Mapping[str, float] | float = 0.0
    member_months: float | None = None
    premium_label: str = "premium"
    mcr_denominator: str = "total_revenue"
    aer_denominator: str = "premium"
    gain_denominator: str = "total_revenue"
    _admin_items: Mapping[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.revenue:
            raise ValueError("revenue must contain at least one component")
        if not self.benefit:
            raise ValueError("benefit must contain at least one component")
        if isinstance(self.admin, Mapping):
            self._admin_items = dict(self.admin)
        else:
            self._admin_items = {"admin": float(self.admin)}
        for name in ("mcr_denominator", "aer_denominator", "gain_denominator"):
            _validate_denominator(getattr(self, name), name)
        uses_premium = "premium" in (
            self.mcr_denominator,
            self.aer_denominator,
            self.gain_denominator,
        )
        if uses_premium and self.premium_label not in self.revenue:
            raise ValueError(
                f"premium_label {self.premium_label!r} is not a revenue "
                f"component; available: {sorted(self.revenue)}"
            )
        if self.member_months is not None and not self.member_months > 0:
            raise ValueError(
                f"member_months must be positive when provided, got "
                f"{self.member_months!r}"
            )

    @classmethod
    def from_pmpm(
        cls,
        *,
        revenue_pmpm: Mapping[str, float],
        benefit_pmpm: Mapping[str, float],
        admin_pmpm: Mapping[str, float] | float = 0.0,
        member_months: float,
        **kwargs: Any,
    ) -> "UnderwritingSummary":
        """Build a summary from PMPM components and member months.

        Forecast exhibits are usually stated PMPM; this dollarizes each
        component by ``member_months`` so totals, PMPMs, and ratios all come
        from one set of inputs.
        """
        if not member_months > 0:
            raise ValueError(
                f"member_months must be positive, got {member_months!r}"
            )
        mm = float(member_months)
        if isinstance(admin_pmpm, Mapping):
            admin: Mapping[str, float] | float = {
                k: v * mm for k, v in admin_pmpm.items()
            }
        else:
            admin = float(admin_pmpm) * mm
        return cls(
            revenue={k: v * mm for k, v in revenue_pmpm.items()},
            benefit={k: v * mm for k, v in benefit_pmpm.items()},
            admin=admin,
            member_months=mm,
            **kwargs,
        )

    # ----- totals ----- #
    @property
    def total_revenue(self) -> float:
        return float(sum(self.revenue.values()))

    @property
    def total_benefit(self) -> float:
        return float(sum(self.benefit.values()))

    @property
    def total_admin(self) -> float:
        return float(sum(self._admin_items.values()))

    @property
    def gross_margin(self) -> float:
        """Tier one: total revenue less benefit expense (admin excluded)."""
        return self.total_revenue - self.total_benefit

    @property
    def gain(self) -> float:
        """Tier two: gross margin less administrative expense."""
        return self.gross_margin - self.total_admin

    # ----- ratios (explicit denominators) ----- #
    def _denominator(self, which: str) -> float:
        if which == "total_revenue":
            return self.total_revenue
        return float(self.revenue[self.premium_label])

    @property
    def mcr(self) -> float:
        """Benefit expense over the ``mcr_denominator``."""
        return float(
            safe_divide(self.total_benefit, self._denominator(self.mcr_denominator))
        )

    @property
    def aer(self) -> float:
        """Administrative expense over the ``aer_denominator``."""
        return float(
            safe_divide(self.total_admin, self._denominator(self.aer_denominator))
        )

    @property
    def gross_margin_ratio(self) -> float:
        """Gross margin over the ``mcr_denominator`` (its complement)."""
        return float(
            safe_divide(self.gross_margin, self._denominator(self.mcr_denominator))
        )

    @property
    def gain_ratio(self) -> float:
        """Gain / (loss) over the ``gain_denominator``."""
        return float(
            safe_divide(self.gain, self._denominator(self.gain_denominator))
        )

    def reconciliation(self) -> float:
        """``gain_ratio - (1 - mcr - aer)``: the mixed-denominator gap.

        Zero when every denominator is the same series; otherwise the size
        of the drift introduced by quoting MCR, AER, and gain over different
        bases. Useful as an exhibit footnote or a data-quality check.
        """
        return self.gain_ratio - (1.0 - self.mcr - self.aer)

    # ----- per member per month ----- #
    def _require_member_months(self) -> float:
        if self.member_months is None:
            raise ValueError(
                "member_months is required for PMPM figures; pass it to the "
                "constructor or use from_pmpm(...)"
            )
        return float(self.member_months)

    @property
    def revenue_pmpm(self) -> float:
        return self.total_revenue / self._require_member_months()

    @property
    def benefit_pmpm(self) -> float:
        return self.total_benefit / self._require_member_months()

    @property
    def admin_pmpm(self) -> float:
        return self.total_admin / self._require_member_months()

    @property
    def gross_margin_pmpm(self) -> float:
        return self.gross_margin / self._require_member_months()

    @property
    def gain_pmpm(self) -> float:
        return self.gain / self._require_member_months()

    # ----- views ----- #
    def to_frame(self) -> pd.DataFrame:
        """One tidy row of every total and ratio (PMPMs when exposure given)."""
        row: dict[str, float] = {
            "total_revenue": self.total_revenue,
            "total_benefit": self.total_benefit,
            "total_admin": self.total_admin,
            "gross_margin": self.gross_margin,
            "gain": self.gain,
            "mcr": self.mcr,
            "aer": self.aer,
            "gross_margin_ratio": self.gross_margin_ratio,
            "gain_ratio": self.gain_ratio,
        }
        if self.member_months is not None:
            row["member_months"] = float(self.member_months)
            row["revenue_pmpm"] = self.revenue_pmpm
            row["benefit_pmpm"] = self.benefit_pmpm
            row["admin_pmpm"] = self.admin_pmpm
            row["gross_margin_pmpm"] = self.gross_margin_pmpm
            row["gain_pmpm"] = self.gain_pmpm
        return pd.DataFrame([row])

    def statement(self) -> pd.Series:
        """Exhibit-shaped Series: components, subtotals, tiers, then ratios."""
        lines: dict[str, float] = {}
        for label, value in self.revenue.items():
            lines[label] = float(value)
        lines["total_revenue"] = self.total_revenue
        for label, value in self.benefit.items():
            lines[label] = float(value)
        lines["total_benefit"] = self.total_benefit
        lines["mcr"] = self.mcr
        lines["gross_margin"] = self.gross_margin
        for label, value in self._admin_items.items():
            lines[label] = float(value)
        lines["total_admin"] = self.total_admin
        lines["aer"] = self.aer
        lines["gain"] = self.gain
        lines["gain_ratio"] = self.gain_ratio
        return pd.Series(lines, name="statement")


def underwriting_summary(
    df: pd.DataFrame,
    *,
    groupby: str | Iterable[str] | None = None,
    revenue_cols: str | Iterable[str],
    benefit_cols: str | Iterable[str],
    admin_cols: str | Iterable[str],
    exposure_col: str | None = None,
    premium_col: str | None = None,
    mcr_denominator: str = "total_revenue",
    aer_denominator: str = "premium",
    gain_denominator: str = "total_revenue",
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
    revenue_cols, benefit_cols, admin_cols : str | Iterable[str]
        Component columns for each tier. Revenue offsets (refunds) should be
        signed.
    exposure_col : str, optional
        Member-months column; adds ``*_pmpm`` output columns.
    premium_col : str, optional
        Gross premium column, required when any denominator is
        ``"premium"``.
    mcr_denominator, aer_denominator, gain_denominator : str
        ``"total_revenue"`` or ``"premium"``; see the module docstring for
        the convention discussion.

    Returns
    -------
    pd.DataFrame
        Group keys, component sums, ``total_revenue``, ``total_benefit``,
        ``total_admin``, ``gross_margin``, ``gain``, ``mcr``, ``aer``,
        ``gross_margin_ratio``, ``gain_ratio``, and PMPM columns when
        ``exposure_col`` is given.
    """
    groups = as_list(groupby)
    revenues = as_list(revenue_cols)
    benefits = as_list(benefit_cols)
    admins = as_list(admin_cols)
    for name in (
        ("mcr_denominator", mcr_denominator),
        ("aer_denominator", aer_denominator),
        ("gain_denominator", gain_denominator),
    ):
        _validate_denominator(name[1], name[0])
    uses_premium = "premium" in (mcr_denominator, aer_denominator, gain_denominator)
    if uses_premium and premium_col is None:
        raise ValueError(
            'premium_col is required when any denominator is "premium"'
        )

    amount_cols = list(dict.fromkeys(revenues + benefits + admins))
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
    summary["total_benefit"] = sum_columns(summary, benefits)
    summary["total_admin"] = sum_columns(summary, admins)
    summary["gross_margin"] = summary["total_revenue"] - summary["total_benefit"]
    summary["gain"] = summary["gross_margin"] - summary["total_admin"]

    def _denom(which: str) -> pd.Series:
        if which == "total_revenue":
            return summary["total_revenue"]
        return summary[premium_col]

    summary["mcr"] = safe_divide(summary["total_benefit"], _denom(mcr_denominator))
    summary["aer"] = safe_divide(summary["total_admin"], _denom(aer_denominator))
    summary["gross_margin_ratio"] = safe_divide(
        summary["gross_margin"], _denom(mcr_denominator)
    )
    summary["gain_ratio"] = safe_divide(summary["gain"], _denom(gain_denominator))

    ordered = (
        groups
        + [c for c in amount_cols if c in summary.columns]
        + exposures
        + [
            "total_revenue",
            "total_benefit",
            "total_admin",
            "gross_margin",
            "gain",
            "mcr",
            "aer",
            "gross_margin_ratio",
            "gain_ratio",
        ]
    )
    if exposure_col is not None:
        for amount, name in (
            ("total_revenue", "revenue_pmpm"),
            ("total_benefit", "benefit_pmpm"),
            ("total_admin", "admin_pmpm"),
            ("gross_margin", "gross_margin_pmpm"),
            ("gain", "gain_pmpm"),
        ):
            summary[name] = per_exposure(summary[amount], summary[exposure_col])
            ordered.append(name)
    return summary[list(dict.fromkeys(ordered))]
