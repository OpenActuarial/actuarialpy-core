import numpy as np

from actuarialpy import loss_ratio, safe_divide


def test_safe_divide_scalar():
    assert safe_divide(10, 2) == 5


def test_safe_divide_zero_denominator():
    assert np.isnan(safe_divide(10, 0))


def test_loss_ratio_scalar():
    assert loss_ratio(850_000, 1_000_000) == 0.85


def test_loss_ratio_array():
    result = loss_ratio([50, 80, 100], [100, 100, 0])

    assert result[0] == 0.5
    assert result[1] == 0.8
    assert np.isnan(result[2])
