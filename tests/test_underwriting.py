import numpy as np
import pandas as pd
import pytest

from actuarialpy import UnderwritingSummary, underwriting_summary


def make_summary(**overrides):
    kwargs = dict(
        revenue={"premium": 1_200_000.0, "refund": -12_000.0},
        losses={"claims_a": 480_000.0, "claims_b": 350_000.0, "claims_c": 210_000.0},
        expenses={"fixed": 60_000.0, "variable": 50_000.0},
        exposure=3_000.0,
    )
    kwargs.update(overrides)
    return UnderwritingSummary(**kwargs)


# --------------------------------------------------------------------------- #
# two-tier structure
# --------------------------------------------------------------------------- #
def test_two_tier_reconciles():
    uw = make_summary()
    assert uw.total_revenue == pytest.approx(1_188_000.0)
    assert uw.total_loss == pytest.approx(1_040_000.0)
    assert uw.total_expense == pytest.approx(110_000.0)
    assert uw.gross_margin == pytest.approx(uw.total_revenue - uw.total_loss)
    assert uw.gain == pytest.approx(uw.gross_margin - uw.total_expense)
    # operating expense never enters the loss tier
    assert uw.gross_margin == pytest.approx(148_000.0)
    assert uw.gain == pytest.approx(38_000.0)


def test_expenses_accept_scalar():
    uw = make_summary(expenses=110_000.0)
    assert uw.total_expense == pytest.approx(110_000.0)
    assert uw.gain == pytest.approx(38_000.0)


# --------------------------------------------------------------------------- #
# explicit denominators
# --------------------------------------------------------------------------- #
def test_default_denominators_mixed():
    uw = make_summary()
    # loss ratio over total revenue (net of refund), expense ratio over gross premium
    assert uw.loss_ratio == pytest.approx(1_040_000.0 / 1_188_000.0)
    assert uw.expense_ratio == pytest.approx(110_000.0 / 1_200_000.0)
    assert uw.combined_ratio == pytest.approx(uw.loss_ratio + uw.expense_ratio)
    assert uw.gain_ratio == pytest.approx(38_000.0 / 1_188_000.0)


def test_reconciliation_gap_zero_when_denominators_match():
    same = make_summary(
        loss_ratio_denominator="total_revenue",
        expense_ratio_denominator="total_revenue",
        gain_denominator="total_revenue",
    )
    assert same.reconciliation() == pytest.approx(0.0, abs=1e-12)

    mixed = make_summary()  # expense ratio over gross premium by default
    assert mixed.reconciliation() != pytest.approx(0.0, abs=1e-9)
    # the gap is exactly expense * (1/gross - 1/net): negative here because
    # the expense ratio over the larger gross base understates the combined ratio
    expected_gap = 110_000.0 * (1 / 1_200_000.0 - 1 / 1_188_000.0)
    assert mixed.reconciliation() == pytest.approx(expected_gap)


def test_premium_label_required_when_used():
    with pytest.raises(ValueError, match="premium_label"):
        make_summary(revenue={"written": 1_200_000.0})
    # fine when no denominator touches gross premium
    uw = make_summary(
        revenue={"written": 1_200_000.0},
        expense_ratio_denominator="total_revenue",
    )
    assert uw.expense_ratio == pytest.approx(110_000.0 / 1_200_000.0)


def test_invalid_denominator_rejected():
    with pytest.raises(ValueError, match="loss_ratio_denominator"):
        make_summary(loss_ratio_denominator="earned")


# --------------------------------------------------------------------------- #
# per-exposure figures and construction
# --------------------------------------------------------------------------- #
def test_per_exposure_requires_exposure():
    uw = make_summary(exposure=None)
    with pytest.raises(ValueError, match="exposure"):
        _ = uw.gain_per_exposure


def test_from_per_exposure_roundtrip():
    uw = UnderwritingSummary.from_per_exposure(
        revenue_per_exposure={"premium": 400.0, "refund": -4.0},
        loss_per_exposure={"claims": 340.0},
        expense_per_exposure=37.0,
        exposure=3_000.0,
    )
    assert uw.total_revenue == pytest.approx(396.0 * 3_000.0)
    assert uw.revenue_per_exposure == pytest.approx(396.0)
    assert uw.gross_margin_per_exposure == pytest.approx(56.0)
    assert uw.gain_per_exposure == pytest.approx(19.0)
    assert uw.gain == pytest.approx(19.0 * 3_000.0)


def test_statement_and_frame_views():
    uw = make_summary()
    stmt = uw.statement()
    order = list(stmt.index)
    assert order.index("total_revenue") < order.index("total_loss")
    assert order.index("gross_margin") < order.index("total_expense") < order.index("gain")
    frame = uw.to_frame()
    assert frame.shape[0] == 1
    for col in ("loss_ratio", "expense_ratio", "combined_ratio", "gain_ratio", "gain_per_exposure"):
        assert col in frame.columns


# --------------------------------------------------------------------------- #
# profile: domain naming lives in the output views, never the calculation
# --------------------------------------------------------------------------- #
def test_profile_renames_loss_ratio_only_in_views():
    uw = make_summary()
    health = uw.to_frame(profile="health")
    assert "mlr" in health.columns and "loss_ratio" not in health.columns
    assert health.loc[0, "mlr"] == pytest.approx(uw.loss_ratio)
    # other names stay generic unless labels rename them
    assert "expense_ratio" in health.columns
    relabeled = uw.to_frame(profile="health", labels={"expense_ratio": "aer"})
    assert "aer" in relabeled.columns
    life = uw.statement(profile="life")
    assert "benefit_ratio" in life.index and "loss_ratio" not in life.index
    with pytest.raises(ValueError, match="Unknown profile"):
        uw.to_frame(profile="dental")


# --------------------------------------------------------------------------- #
# grouped frame version: ratio of sums, never average of ratios
# --------------------------------------------------------------------------- #
def grouped_df(exposure_name="exposure"):
    return pd.DataFrame(
        {
            "cohort": ["apr", "apr", "sep"],
            "premium": [100_000.0, 900_000.0, 500_000.0],
            "refund": [0.0, -9_000.0, 0.0],
            "claims": [95_000.0, 720_000.0, 430_000.0],
            "expense": [9_000.0, 81_000.0, 45_000.0],
            exposure_name: [250.0, 2_200.0, 1_200.0],
        }
    )


def test_grouped_ratio_of_sums():
    out = underwriting_summary(
        grouped_df(),
        groupby="cohort",
        revenue_cols=["premium", "refund"],
        loss_cols="claims",
        expense_cols="expense",
        exposure_col="exposure",
        premium_col="premium",
    )
    apr = out.loc[out["cohort"] == "apr"].iloc[0]
    # ratio of sums
    assert apr["loss_ratio"] == pytest.approx(815_000.0 / 991_000.0)
    # explicitly not the average of row-level ratios
    row_average = np.mean([95_000.0 / 100_000.0, 720_000.0 / 891_000.0])
    assert apr["loss_ratio"] != pytest.approx(row_average)
    assert apr["expense_ratio"] == pytest.approx(90_000.0 / 1_000_000.0)
    assert apr["gain"] == pytest.approx(991_000.0 - 815_000.0 - 90_000.0)
    assert apr["gain_per_exposure"] == pytest.approx(apr["gain"] / 2_450.0)


def test_grouped_per_exposure_names_follow_the_column():
    # every exposure column gets the mechanical {amount}_per_{column} form ...
    out = underwriting_summary(
        grouped_df("member_months"),
        groupby="cohort",
        revenue_cols=["premium", "refund"],
        loss_cols="claims",
        expense_cols="expense",
        exposure_col="member_months",
        premium_col="premium",
    )
    assert {"revenue_per_member_months", "loss_per_member_months", "gain_per_member_months"} <= set(out.columns)
    # ... whatever the column is called
    generic = underwriting_summary(
        grouped_df("car_years"),
        groupby="cohort",
        revenue_cols=["premium", "refund"],
        loss_cols="claims",
        expense_cols="expense",
        exposure_col="car_years",
        premium_col="premium",
    )
    assert "gain_per_car_years" in generic.columns
    # domain names are opt-in via labels, never inferred from the column
    labeled = underwriting_summary(
        grouped_df("member_months"),
        groupby="cohort",
        revenue_cols=["premium", "refund"],
        loss_cols="claims",
        expense_cols="expense",
        exposure_col="member_months",
        premium_col="premium",
        labels={"gain_per_member_months": "gain_pmpm"},
    )
    assert "gain_pmpm" in labeled.columns


def test_grouped_profile_and_all_rows():
    out = underwriting_summary(
        grouped_df(),
        revenue_cols=["premium", "refund"],
        loss_cols="claims",
        expense_cols="expense",
        premium_col="premium",
        profile="health",
    )
    assert out.shape[0] == 1
    assert "mlr" in out.columns and "loss_ratio" not in out.columns
    assert out.loc[0, "total_revenue"] == pytest.approx(1_491_000.0)


def test_grouped_requires_premium_col_for_premium_denominator():
    with pytest.raises(ValueError, match="premium_col"):
        underwriting_summary(
            grouped_df(),
            revenue_cols=["premium", "refund"],
            loss_cols="claims",
            expense_cols="expense",
        )
    out = underwriting_summary(
        grouped_df(),
        revenue_cols=["premium", "refund"],
        loss_cols="claims",
        expense_cols="expense",
        expense_ratio_denominator="total_revenue",
    )
    assert "expense_ratio" in out.columns
