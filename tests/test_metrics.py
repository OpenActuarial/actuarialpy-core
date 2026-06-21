import numpy as np
import pytest

from actuarialpy import (
    actual_to_expected,
    combined_ratio,
    frequency,
    indicated_change,
    loss_ratio,
    pmpm,
    pure_premium,
    required_revenue,
    safe_divide,
    severity,
)


def test_safe_divide_scalar_zero():
    assert np.isnan(safe_divide(10, 0))


def test_basic_metrics():
    assert loss_ratio(85, 100) == 0.85
    assert pmpm(1_000, 2) == 500
    assert actual_to_expected(110, 100) == 1.10
    assert frequency(20, 100) == 0.2
    assert severity(1_000, 20) == 50
    assert pure_premium(1_000, 2) == 500
    assert combined_ratio(70, 20, 100) == 0.90


def test_rate_adequacy():
    assert required_revenue(80, 0.8) == 100
    assert indicated_change(110, 100) == pytest.approx(0.10)
