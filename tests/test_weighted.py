import numpy as np
import pandas as pd
import pytest

from actuarialpy import weighted_mean, weighted_summary


# --------------------------------------------------------------------------- #
# weighted_mean
# --------------------------------------------------------------------------- #
def test_weighted_mean_basic():
    assert weighted_mean([0.10, 0.20], [1.0, 3.0]) == pytest.approx(0.175)


def test_weighted_mean_is_not_equal_weighted():
    # a large case must dominate a small one
    actions = [0.30, 0.05]
    premium = [1_000.0, 99_000.0]
    result = weighted_mean(actions, premium)
    assert result == pytest.approx(0.0525)
    assert result != pytest.approx(np.mean(actions))


def test_weighted_mean_validates_weights():
    with pytest.raises(ValueError, match="same shape"):
        weighted_mean([0.1, 0.2], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        weighted_mean([0.1, 0.2], [1.0, -1.0])
    with pytest.raises(ValueError, match="finite"):
        weighted_mean([0.1, 0.2], [1.0, np.inf])
    with pytest.raises(ValueError, match="positive total"):
        weighted_mean([0.1, 0.2], [0.0, 0.0])


def test_weighted_mean_nan_semantics():
    # default: missing data surfaces
    assert np.isnan(weighted_mean([0.1, np.nan], [1.0, 1.0]))
    # opt-in: pair is dropped
    assert weighted_mean([0.1, np.nan], [1.0, 1.0], skipna=True) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="positive total"):
        weighted_mean([np.nan], [1.0], skipna=True)


# --------------------------------------------------------------------------- #
# weighted_summary
# --------------------------------------------------------------------------- #
def book():
    return pd.DataFrame(
        {
            "cohort": ["apr", "apr", "sep", "sep"],
            "rate_action": [0.20, 0.05, 0.10, 0.30],
            "persistency": [0.90, 0.60, 0.80, 0.50],
            "premium": [1_000.0, 9_000.0, 4_000.0, 1_000.0],
        }
    )


def test_weighted_summary_grouped():
    out = weighted_summary(
        book(),
        value_cols=["rate_action", "persistency"],
        weight_col="premium",
        groupby="cohort",
    )
    apr = out.loc[out["cohort"] == "apr"].iloc[0]
    assert apr["rate_action_weighted"] == pytest.approx(
        (0.20 * 1_000 + 0.05 * 9_000) / 10_000
    )
    assert apr["persistency_weighted"] == pytest.approx(
        (0.90 * 1_000 + 0.60 * 9_000) / 10_000
    )
    assert apr["premium_total"] == pytest.approx(10_000.0)


def test_weighted_summary_ungrouped():
    out = weighted_summary(book(), value_cols="rate_action", weight_col="premium")
    assert out.shape[0] == 1
    expected = (0.20 * 1_000 + 0.05 * 9_000 + 0.10 * 4_000 + 0.30 * 1_000) / 15_000
    assert out.loc[0, "rate_action_weighted"] == pytest.approx(expected)


def test_weighted_summary_nan_propagates_by_default():
    df = book()
    df.loc[0, "rate_action"] = np.nan
    out = weighted_summary(
        df, value_cols="rate_action", weight_col="premium", groupby="cohort"
    )
    apr = out.loc[out["cohort"] == "apr"].iloc[0]
    sep = out.loc[out["cohort"] == "sep"].iloc[0]
    assert np.isnan(apr["rate_action_weighted"])
    assert not np.isnan(sep["rate_action_weighted"])

    skipped = weighted_summary(
        df,
        value_cols="rate_action",
        weight_col="premium",
        groupby="cohort",
        skipna=True,
    )
    apr_skipped = skipped.loc[skipped["cohort"] == "apr"].iloc[0]
    assert apr_skipped["rate_action_weighted"] == pytest.approx(0.05)


def test_weighted_summary_rejects_bad_weights():
    df = book()
    df.loc[0, "premium"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        weighted_summary(df, value_cols="rate_action", weight_col="premium")
