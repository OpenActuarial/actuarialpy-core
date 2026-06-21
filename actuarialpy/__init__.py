"""ActuarialPy: primitive-based tools for actuarial analysis."""

from actuarialpy.metrics import (
    actual_to_expected,
    combined_ratio,
    expense_ratio,
    frequency,
    indicated_change,
    loss_ratio,
    medical_loss_ratio,
    pepm,
    per_exposure,
    pmpm,
    pspm,
    pure_premium,
    ratio,
    required_revenue,
    safe_divide,
    severity,
)

__all__ = [
    "actual_to_expected",
    "combined_ratio",
    "expense_ratio",
    "frequency",
    "indicated_change",
    "loss_ratio",
    "medical_loss_ratio",
    "pepm",
    "per_exposure",
    "pmpm",
    "pspm",
    "pure_premium",
    "ratio",
    "required_revenue",
    "safe_divide",
    "severity",
]

__version__ = "0.4.0"
