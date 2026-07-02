import numpy as np
import pandas as pd
import pytest

from actuarialpy import UnderwritingSummary, underwriting_summary


def make_summary(**overrides):
    kwargs = dict(
        revenue={"premium": 1_200_000.0, "refund": -12_000.0},
        benefit={"inpatient": 480_000.0, "outpatient": 350_000.0, "pharmacy": 210_000.0},
        admin={"fixed": 60_000.0, "variable": 50_000.0},
        member_months=3_000.0,
    )
    kwargs.update(overrides)
    return UnderwritingSummary(**kwargs)


# --------------------------------------------------------------------------- #
# two-tier structure
# --------------------------------------------------------------------------- #
def test_two_tier_reconciles():
    uw = make_summary()
    assert uw.total_revenue == pytest.approx(1_188_000.0)
    assert uw.total_benefit == pytest.approx(1_040_000.0)
    assert uw.total_admin == pytest.approx(110_000.0)
    assert uw.gross_margin == pytest.approx(uw.total_revenue - uw.total_benefit)
    assert uw.gain == pytest.approx(uw.gross_margin - uw.total_admin)
    # admin never enters the benefit tier
    assert uw.gross_margin == pytest.approx(148_000.0)
    assert uw.gain == pytest.approx(38_000.0)


def test_admin_accepts_scalar():
    uw = make_summary(admin=110_000.0)
    assert uw.total_admin == pytest.approx(110_000.0)
    assert uw.gain == pytest.approx(38_000.0)


# --------------------------------------------------------------------------- #
# explicit denominators
# --------------------------------------------------------------------------- #
def test_default_denominators_mixed():
    uw = make_summary()
    # MCR over total revenue (net of refund), AER over gross premium
    assert uw.mcr == pytest.approx(1_040_000.0 / 1_188_000.0)
    assert uw.aer == pytest.approx(110_000.0 / 1_200_000.0)
    assert uw.gain_ratio == pytest.approx(38_000.0 / 1_188_000.0)


def test_reconciliation_gap_zero_when_denominators_match():
    same = make_summary(
        mcr_denominator="total_revenue",
        aer_denominator="total_revenue",
        gain_denominator="total_revenue",
    )
    assert same.reconciliation() == pytest.approx(0.0, abs=1e-12)

    mixed = make_summary()  # AER over gross premium by default
    assert mixed.reconciliation() != pytest.approx(0.0, abs=1e-9)
    # the gap is exactly admin * (1/gross - 1/net): negative here because
    # AER over the larger gross base understates 1 - MCR - AER
    expected_gap = 110_000.0 * (1 / 1_200_000.0 - 1 / 1_188_000.0)
    assert mixed.reconciliation() == pytest.approx(expected_gap)


def test_premium_label_required_when_used():
    with pytest.raises(ValueError, match="premium_label"):
        make_summary(revenue={"written": 1_200_000.0})
    # fine when no denominator touches gross premium
    uw = make_summary(
        revenue={"written": 1_200_000.0},
        aer_denominator="total_revenue",
    )
    assert uw.aer == pytest.approx(110_000.0 / 1_200_000.0)


def test_invalid_denominator_rejected():
    with pytest.raises(ValueError, match="mcr_denominator"):
        make_summary(mcr_denominator="earned")


# --------------------------------------------------------------------------- #
# PMPM and construction
# --------------------------------------------------------------------------- #
def test_pmpm_requires_member_months():
    uw = make_summary(member_months=None)
    with pytest.raises(ValueError, match="member_months"):
        _ = uw.gain_pmpm


def test_from_pmpm_roundtrip():
    uw = UnderwritingSummary.from_pmpm(
        revenue_pmpm={"premium": 400.0, "refund": -4.0},
        benefit_pmpm={"medical": 340.0},
        admin_pmpm=37.0,
        member_months=3_000.0,
    )
    assert uw.total_revenue == pytest.approx(396.0 * 3_000.0)
    assert uw.revenue_pmpm == pytest.approx(396.0)
    assert uw.gross_margin_pmpm == pytest.approx(56.0)
    assert uw.gain_pmpm == pytest.approx(19.0)
    assert uw.gain == pytest.approx(19.0 * 3_000.0)


def test_statement_and_frame_views():
    uw = make_summary()
    stmt = uw.statement()
    order = list(stmt.index)
    assert order.index("total_revenue") < order.index("total_benefit")
    assert order.index("gross_margin") < order.index("total_admin") < order.index("gain")
    frame = uw.to_frame()
    assert frame.shape[0] == 1
    for col in ("mcr", "aer", "gain_ratio", "gain_pmpm"):
        assert col in frame.columns


# --------------------------------------------------------------------------- #
# grouped frame version: ratio of sums, never average of ratios
# --------------------------------------------------------------------------- #
def grouped_df():
    return pd.DataFrame(
        {
            "cohort": ["apr", "apr", "sep"],
            "premium": [100_000.0, 900_000.0, 500_000.0],
            "refund": [0.0, -9_000.0, 0.0],
            "medical": [95_000.0, 720_000.0, 430_000.0],
            "admin": [9_000.0, 81_000.0, 45_000.0],
            "member_months": [250.0, 2_200.0, 1_200.0],
        }
    )


def test_grouped_ratio_of_sums():
    out = underwriting_summary(
        grouped_df(),
        groupby="cohort",
        revenue_cols=["premium", "refund"],
        benefit_cols="medical",
        admin_cols="admin",
        exposure_col="member_months",
        premium_col="premium",
    )
    apr = out.loc[out["cohort"] == "apr"].iloc[0]
    # ratio of sums
    assert apr["mcr"] == pytest.approx(815_000.0 / 991_000.0)
    # explicitly not the average of row-level MLRs
    row_average = np.mean([95_000.0 / 100_000.0, 720_000.0 / 891_000.0])
    assert apr["mcr"] != pytest.approx(row_average)
    assert apr["aer"] == pytest.approx(90_000.0 / 1_000_000.0)
    assert apr["gain"] == pytest.approx(991_000.0 - 815_000.0 - 90_000.0)
    assert apr["gain_pmpm"] == pytest.approx(apr["gain"] / 2_450.0)


def test_grouped_all_rows_when_no_groupby():
    out = underwriting_summary(
        grouped_df(),
        revenue_cols=["premium", "refund"],
        benefit_cols="medical",
        admin_cols="admin",
        premium_col="premium",
    )
    assert out.shape[0] == 1
    assert out.loc[0, "total_revenue"] == pytest.approx(1_491_000.0)


def test_grouped_requires_premium_col_for_premium_denominator():
    with pytest.raises(ValueError, match="premium_col"):
        underwriting_summary(
            grouped_df(),
            revenue_cols=["premium", "refund"],
            benefit_cols="medical",
            admin_cols="admin",
        )
    # switching every denominator to total_revenue lifts the requirement
    out = underwriting_summary(
        grouped_df(),
        revenue_cols=["premium", "refund"],
        benefit_cols="medical",
        admin_cols="admin",
        aer_denominator="total_revenue",
    )
    assert "aer" in out.columns
