r"""Financial mathematics: the time-value-of-money primitives.

Every reserve, premium, and valuation discounts cash flows, so this module is
the foundation the rest of the toolkit stands on. It covers interest-rate
fundamentals and their conversions, present/accumulated values, annuities-
certain, cash-flow analysis (NPV/IRR), loan amortization, discounting against a
spot curve, and day-count year fractions.

Notation: ``i`` is the effective annual rate, ``v = 1/(1+i)`` the discount
factor, ``d = i/(1+i)`` the effective rate of discount, and ``delta = ln(1+i)``
the force of interest. Nominal rates convertible ``m`` times per year are
``i^(m)`` and ``d^(m)``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DateLike = object


# --------------------------------------------------------------------------- #
# interest-rate fundamentals and conversions
# --------------------------------------------------------------------------- #
def _check_rate(i: float) -> float:
    i = float(i)
    if i <= -1.0:
        raise ValueError("the effective rate i must exceed -1.")
    return i


def discount_factor(i: float, t: float = 1.0) -> float:
    r"""Discount factor :math:`v^t = (1+i)^{-t}`."""
    return float((1.0 + _check_rate(i)) ** (-float(t)))


def accumulation_factor(i: float, t: float = 1.0) -> float:
    r"""Accumulation factor :math:`(1+i)^t`."""
    return float((1.0 + _check_rate(i)) ** float(t))


def effective_discount(i: float) -> float:
    r"""Effective rate of discount :math:`d = i/(1+i) = 1 - v`."""
    i = _check_rate(i)
    return float(i / (1.0 + i))


def force_of_interest(i: float) -> float:
    r"""Force of interest :math:`\delta = \ln(1+i)`."""
    return float(np.log1p(_check_rate(i)))


def rate_from_force(delta: float) -> float:
    r"""Effective rate from the force of interest: :math:`i = e^\delta - 1`."""
    return float(np.expm1(float(delta)))


def nominal_interest(i: float, m: int) -> float:
    r"""Nominal interest convertible ``m`` times: :math:`i^{(m)} = m[(1+i)^{1/m}-1]`."""
    m = _check_periods(m)
    i = _check_rate(i)
    return float(m * ((1.0 + i) ** (1.0 / m) - 1.0))


def nominal_discount(i: float, m: int) -> float:
    r"""Nominal discount convertible ``m`` times: :math:`d^{(m)} = m[1-v^{1/m}]`."""
    m = _check_periods(m)
    i = _check_rate(i)
    v = 1.0 / (1.0 + i)
    return float(m * (1.0 - v ** (1.0 / m)))


def rate_from_nominal_interest(nominal: float, m: int) -> float:
    r"""Effective rate from a nominal interest rate: :math:`(1+i^{(m)}/m)^m - 1`."""
    m = _check_periods(m)
    return float((1.0 + float(nominal) / m) ** m - 1.0)


def rate_from_nominal_discount(nominal: float, m: int) -> float:
    r"""Effective rate from a nominal discount rate: :math:`(1-d^{(m)}/m)^{-m} - 1`."""
    m = _check_periods(m)
    base = 1.0 - float(nominal) / m
    if base <= 0:
        raise ValueError("nominal discount too large for the given m.")
    return float(base ** (-m) - 1.0)


def _check_periods(m: int) -> int:
    m = int(m)
    if m <= 0:
        raise ValueError("the number of periods/conversions must be positive.")
    return m


# --------------------------------------------------------------------------- #
# present and future value
# --------------------------------------------------------------------------- #
def present_value(amount: float, i: float, t: float) -> float:
    """Present value of a single ``amount`` due in ``t`` years."""
    return float(amount) * discount_factor(i, t)


def future_value(amount: float, i: float, t: float) -> float:
    """Accumulated value of a single ``amount`` after ``t`` years."""
    return float(amount) * accumulation_factor(i, t)


# --------------------------------------------------------------------------- #
# annuities-certain
# --------------------------------------------------------------------------- #
def annuity_immediate(i: float, n: int) -> float:
    r"""Present value of an annuity-immediate :math:`a_{\overline{n}|}=(1-v^n)/i`."""
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n)
    v = 1.0 / (1.0 + i)
    return float((1.0 - v**n) / i)


def annuity_due(i: float, n: int) -> float:
    r"""Present value of an annuity-due :math:`\ddot a_{\overline{n}|}=(1-v^n)/d`."""
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n)
    return float(annuity_immediate(i, n) * (1.0 + i))


def accumulated_immediate(i: float, n: int) -> float:
    r"""Accumulated value of an annuity-immediate :math:`s_{\overline{n}|}`."""
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n)
    return float(((1.0 + i) ** n - 1.0) / i)


def accumulated_due(i: float, n: int) -> float:
    r"""Accumulated value of an annuity-due :math:`\ddot s_{\overline{n}|}`."""
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n)
    return float(accumulated_immediate(i, n) * (1.0 + i))


def perpetuity_immediate(i: float) -> float:
    r"""Present value of a perpetuity-immediate :math:`1/i`."""
    i = _check_rate(i)
    if i <= 0:
        raise ValueError("a perpetuity requires i > 0.")
    return float(1.0 / i)


def perpetuity_due(i: float) -> float:
    r"""Present value of a perpetuity-due :math:`1/d`."""
    i = _check_rate(i)
    if i <= 0:
        raise ValueError("a perpetuity requires i > 0.")
    return float(1.0 / effective_discount(i))


def deferred_annuity_immediate(i: float, n: int, defer: int) -> float:
    r"""Present value of an ``n``-year annuity-immediate deferred ``defer`` years."""
    if defer < 0:
        raise ValueError("defer must be non-negative.")
    return float(discount_factor(i, defer) * annuity_immediate(i, n))


def annuity_continuous(i: float, n: int) -> float:
    r"""Present value of a continuous annuity :math:`\bar a_{\overline{n}|}=(1-v^n)/\delta`."""
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n)
    v = 1.0 / (1.0 + i)
    return float((1.0 - v**n) / force_of_interest(i))


def annuity_immediate_mthly(i: float, n: int, m: int) -> float:
    r"""Present value of an ``m``-thly annuity-immediate :math:`a^{(m)}_{\overline{n}|}`."""
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n)
    v = 1.0 / (1.0 + i)
    return float((1.0 - v**n) / nominal_interest(i, m))


def increasing_annuity_immediate(i: float, n: int) -> float:
    r"""Present value of an increasing annuity :math:`(Ia)_{\overline{n}|}`.

    Payments of 1, 2, ..., n at times 1, ..., n.
    """
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n * (n + 1) / 2)
    v = 1.0 / (1.0 + i)
    return float((annuity_due(i, n) - n * v**n) / i)


def decreasing_annuity_immediate(i: float, n: int) -> float:
    r"""Present value of a decreasing annuity :math:`(Da)_{\overline{n}|}`.

    Payments of n, n-1, ..., 1 at times 1, ..., n.
    """
    i = _check_rate(i)
    n = _check_term(n)
    if i == 0:
        return float(n * (n + 1) / 2)
    return float((n - annuity_immediate(i, n)) / i)


def geometric_annuity_immediate(i: float, n: int, growth: float) -> float:
    r"""Present value of a geometrically increasing annuity-immediate.

    Payments :math:`1, (1+g), (1+g)^2, \ldots` at times :math:`1, \ldots, n`:

    .. math::
        \frac{1 - \left(\frac{1+g}{1+i}\right)^n}{i - g}, \qquad i \neq g.
    """
    i = _check_rate(i)
    n = _check_term(n)
    g = float(growth)
    if abs(i - g) < 1e-15:
        return float(n / (1.0 + i))
    ratio = (1.0 + g) / (1.0 + i)
    return float((1.0 - ratio**n) / (i - g))


def _check_term(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError("the term n must be non-negative.")
    return n


# --------------------------------------------------------------------------- #
# cash-flow analysis
# --------------------------------------------------------------------------- #
def net_present_value(
    rate: float,
    cashflows: Sequence[float],
    times: Sequence[float] | None = None,
) -> float:
    """Net present value of ``cashflows`` discounted at ``rate``.

    If ``times`` is omitted the cash flows are assumed to occur at times
    ``0, 1, 2, ...``.
    """
    rate = _check_rate(rate)
    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cf)) if times is None else np.asarray(times, dtype=float)
    if t.shape != cf.shape:
        raise ValueError("times and cashflows must have the same length.")
    return float(np.sum(cf * (1.0 + rate) ** (-t)))


def internal_rate_of_return(
    cashflows: Sequence[float],
    times: Sequence[float] | None = None,
    *,
    low: float = -0.9999,
    high: float = 1e6,
    tol: float = 1e-10,
) -> float:
    """Internal rate of return: the ``rate`` solving ``net_present_value == 0``.

    Uses a bracketed bisection over ``(low, high)``, which is robust for the
    usual single-sign-change cash-flow streams. Raises if no sign change is
    found in the search range (e.g. all-positive or all-negative flows).
    """
    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cf)) if times is None else np.asarray(times, dtype=float)
    if t.shape != cf.shape:
        raise ValueError("times and cashflows must have the same length.")

    def npv(r: float) -> float:
        return float(np.sum(cf * (1.0 + r) ** (-t)))

    # scan for a sign change on a log-spaced grid above -1
    grid = np.concatenate(
        [np.linspace(low, 1.0, 200), np.linspace(1.0, high, 200)[1:]]
    )
    vals = np.array([npv(r) for r in grid])
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if sign_change.size == 0:
        raise ValueError("no sign change in NPV over the search range; IRR not found.")

    a, b = grid[sign_change[0]], grid[sign_change[0] + 1]
    fa = npv(a)
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = npv(mid)
        if abs(fm) < tol or (b - a) < 1e-15:
            return float(mid)
        if np.sign(fm) == np.sign(fa):
            a, fa = mid, fm
        else:
            b = mid
    return float(0.5 * (a + b))


# --------------------------------------------------------------------------- #
# loans and amortization
# --------------------------------------------------------------------------- #
def level_payment(principal: float, i: float, n: int) -> float:
    r"""Level payment amortizing ``principal`` over ``n`` periods at rate ``i``.

    :math:`P = L / a_{\overline{n}|}`.
    """
    principal = float(principal)
    a = annuity_immediate(i, n)
    if a == 0:
        raise ValueError("cannot amortize over zero periods.")
    return float(principal / a)


def outstanding_balance(principal: float, i: float, n: int, t: int) -> float:
    """Prospective outstanding loan balance just after the ``t``-th payment."""
    if not 0 <= t <= n:
        raise ValueError("t must be between 0 and n.")
    payment = level_payment(principal, i, n)
    return float(payment * annuity_immediate(i, n - t))


def amortization_schedule(
    principal: float, i: float, n: int, payment: float | None = None
) -> pd.DataFrame:
    """Amortization schedule with the interest/principal split and balance.

    Returns one row per period with columns ``period``, ``payment``,
    ``interest``, ``principal``, and ``balance``.
    """
    i = _check_rate(i)
    n = _check_term(n)
    pay = level_payment(principal, i, n) if payment is None else float(payment)
    rows = []
    balance = float(principal)
    for period in range(1, n + 1):
        interest = balance * i
        principal_paid = pay - interest
        balance = balance - principal_paid
        rows.append(
            {
                "period": period,
                "payment": pay,
                "interest": interest,
                "principal": principal_paid,
                "balance": balance,
            }
        )
    return pd.DataFrame(rows, columns=["period", "payment", "interest", "principal", "balance"])


# --------------------------------------------------------------------------- #
# discounting against a spot curve
# --------------------------------------------------------------------------- #
def discount_factors(spot_rates: Sequence[float], times: Sequence[float]) -> np.ndarray:
    r"""Discount factors :math:`(1+s_t)^{-t}` from spot rates at ``times``."""
    s = np.asarray(spot_rates, dtype=float)
    t = np.asarray(times, dtype=float)
    if s.shape != t.shape:
        raise ValueError("spot_rates and times must have the same length.")
    if np.any(s <= -1.0):
        raise ValueError("spot rates must exceed -1.")
    return (1.0 + s) ** (-t)


def present_value_curve(
    cashflows: Sequence[float],
    spot_rates: Sequence[float],
    times: Sequence[float],
) -> float:
    """Present value of ``cashflows`` discounted on a spot-rate curve."""
    cf = np.asarray(cashflows, dtype=float)
    df = discount_factors(spot_rates, times)
    if cf.shape != df.shape:
        raise ValueError("cashflows, spot_rates, and times must have the same length.")
    return float(np.sum(cf * df))


# --------------------------------------------------------------------------- #
# day-count year fractions
# --------------------------------------------------------------------------- #
def year_fraction(start: DateLike, end: DateLike, convention: str = "actual/365") -> float:
    """Year fraction between two dates under a day-count convention.

    Supported conventions: ``"actual/365"``, ``"actual/360"``, ``"30/360"``
    (US/NASD), and ``"actual/actual"`` (ISDA).
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    conv = convention.lower().replace(" ", "")

    if conv in ("actual/365", "act/365", "actual/365fixed"):
        return (e - s).days / 365.0
    if conv in ("actual/360", "act/360"):
        return (e - s).days / 360.0
    if conv in ("30/360", "30u/360", "bond"):
        d1, d2 = min(s.day, 30), min(e.day, 30) if s.day >= 30 else e.day
        days = 360 * (e.year - s.year) + 30 * (e.month - s.month) + (d2 - d1)
        return days / 360.0
    if conv in ("actual/actual", "act/act", "actual/actualisda"):
        if e < s:
            return -year_fraction(end, start, convention)
        if s.year == e.year:
            denom = 366.0 if s.is_leap_year else 365.0
            return (e - s).days / denom
        total = 0.0
        # leading stub to year-end
        year_end = pd.Timestamp(year=s.year, month=12, day=31)
        denom = 366.0 if s.is_leap_year else 365.0
        total += ((year_end - s).days + 1) / denom
        # whole years between
        total += e.year - s.year - 1
        # trailing stub from year-start
        year_start = pd.Timestamp(year=e.year, month=1, day=1)
        denom = 366.0 if e.is_leap_year else 365.0
        total += (e - year_start).days / denom
        return float(total)
    raise ValueError(f"unknown day-count convention: {convention!r}.")
