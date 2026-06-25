"""Tests for frequency-severity summary and PMPM trend decomposition."""

import numpy as np
import pandas as pd
import pytest

from actuarialpy import decompose_pmpm_trend, frequency_severity_summary


def _book(freq_per_mm, sev, mm=1000, n_months=12, **extra):
    rows = []
    for i in range(n_months):
        cnt = freq_per_mm * mm
        row = dict(month=i, claim_count=cnt, claims=cnt * sev, member_months=mm)
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def _fs(df, **kw):
    return frequency_severity_summary(df, count_col="claim_count", loss_col="claims",
                                      exposure_col="member_months", **kw)


def _dec(prior, current, **kw):
    return decompose_pmpm_trend(prior, current, count_col="claim_count", loss_col="claims",
                                exposure_col="member_months", **kw)


def test_pmpm_equals_frequency_times_severity():
    row = _fs(_book(0.40, 250.0)).iloc[0]
    assert row["pmpm"] == pytest.approx(row["frequency"] * row["severity"])
    assert row["pmpm"] == pytest.approx(100.0)


def test_util_per_1000_annualized():
    row = _fs(_book(0.40, 250.0)).iloc[0]
    assert row["util_per_1000"] == pytest.approx(row["frequency"] * 12 * 1000)


def test_summary_groupby():
    df = pd.concat([_book(0.40, 250.0, plan="A"), _book(0.30, 300.0, plan="B")], ignore_index=True)
    out = _fs(df, groupby="plan")
    assert sorted(out["plan"]) == ["A", "B"]
    assert (out["pmpm"] == out["frequency"] * out["severity"]).all()


def test_multiplicative_decomposition_exact():
    d = _dec(_book(0.40, 250.0), _book(0.42, 262.5)).iloc[0]
    assert d["util_trend"] * d["cost_trend"] == pytest.approx(d["pmpm_trend"])
    assert d["util_trend"] == pytest.approx(1.05)
    assert d["cost_trend"] == pytest.approx(1.05)
    assert d["pmpm_trend"] == pytest.approx(1.1025)


def test_additive_decomposition_exact():
    d = _dec(_book(0.40, 250.0), _book(0.42, 262.5)).iloc[0]
    assert d["util_effect"] + d["cost_effect"] == pytest.approx(d["pmpm_change"])
    assert d["pmpm_change"] == pytest.approx(10.25)


def test_pure_utilization_change():
    d = _dec(_book(0.40, 250.0), _book(0.48, 250.0)).iloc[0]   # only frequency moves
    assert d["cost_trend"] == pytest.approx(1.0)
    assert d["util_trend"] == pytest.approx(1.2)
    assert d["cost_effect"] == pytest.approx(0.0)


def test_pure_cost_change():
    d = _dec(_book(0.40, 250.0), _book(0.40, 275.0)).iloc[0]   # only severity moves
    assert d["util_trend"] == pytest.approx(1.0)
    assert d["cost_trend"] == pytest.approx(1.1)
    assert d["util_effect"] == pytest.approx(0.0)


def test_decompose_grouped_outer_join():
    pri = pd.concat([_book(0.40, 250.0, plan="A"), _book(0.30, 300.0, plan="B")], ignore_index=True)
    cur = pd.concat([_book(0.42, 262.5, plan="A"), _book(0.30, 300.0, plan="C")], ignore_index=True)
    out = _dec(pri, cur, on="plan")
    assert set(out["plan"]) == {"A", "B", "C"}   # outer join keeps one-sided plans
    a = out[out["plan"] == "A"].iloc[0]
    assert a["pmpm_trend"] == pytest.approx(1.1025)


def test_output_leads_with_pmpm_and_trends():
    cols = list(_dec(_book(0.40, 250.0), _book(0.42, 262.5)).columns)
    assert cols[:5] == ["pmpm_prior", "pmpm_current", "pmpm_trend", "util_trend", "cost_trend"]


# --- three-way (utilization x unit cost x mix) via mix_by -------------------

def _cells(spec):
    """spec: {cell_value: (member_months, count, loss)} -> one row per cell."""
    return pd.DataFrame(
        [dict(cell=k, member_months=mm, claim_count=n, claims=a) for k, (mm, n, a) in spec.items()]
    )


def _mix(prior, current, **kw):
    return decompose_pmpm_trend(prior, current, count_col="claim_count", loss_col="claims",
                                exposure_col="member_months", mix_by="cell", **kw)


# the worked example: util ~+5.0%, unit cost ~+4.6%, mix ~+8.8%, pmpm ~+19.5%
_PRIOR = _cells({"Healthy": (70000.0, 28000.0, 14_000_000.0),
                 "Chronic": (30000.0, 24000.0, 21_600_000.0)})
_CURR = _cells({"Healthy": (64000.0, 26880.0, 13_977_600.0),
                "Chronic": (36000.0, 30240.0, 28_576_800.0)})


def test_three_way_multiplicative_reconciles():
    d = _mix(_PRIOR, _CURR).iloc[0]
    assert d["util_trend"] * d["cost_trend"] * d["mix_trend"] == pytest.approx(d["pmpm_trend"])


def test_three_way_additive_reconciles():
    d = _mix(_PRIOR, _CURR).iloc[0]
    assert d["util_effect"] + d["cost_effect"] + d["mix_effect"] == pytest.approx(d["pmpm_change"])


def test_three_way_known_values():
    d = _mix(_PRIOR, _CURR).iloc[0]
    assert d["util_trend"] == pytest.approx(1.0499, abs=1e-3)
    assert d["cost_trend"] == pytest.approx(1.0463, abs=1e-3)
    assert d["mix_trend"] == pytest.approx(1.0881, abs=1e-3)
    assert d["pmpm_trend"] == pytest.approx(1.19535, abs=1e-4)


def test_mix_by_none_is_two_way_unchanged():
    two = _dec(_PRIOR, _CURR)
    assert "mix_trend" not in two.columns          # no third component
    assert "mix_effect" not in two.columns
    row = two.iloc[0]
    assert row["util_trend"] * row["cost_trend"] == pytest.approx(row["pmpm_trend"])


def test_three_way_differs_from_book_wide_two_way():
    # the whole point: the 2-way blames util/cost for the population shift
    two = _dec(_PRIOR, _CURR).iloc[0]
    three = _mix(_PRIOR, _CURR).iloc[0]
    assert two["pmpm_trend"] == pytest.approx(three["pmpm_trend"])      # same total
    assert two["util_trend"] > three["util_trend"] + 0.03              # book-wide util overstated
    assert two["cost_trend"] > three["cost_trend"] + 0.03              # book-wide cost overstated


def test_pure_mix_shift_isolated():
    # within-cell util & unit cost identical both periods; only weights move
    prior = _cells({"A": (70000.0, 28000.0, 14_000_000.0),    # u=0.40, c=500
                    "B": (30000.0, 24000.0, 21_600_000.0)})   # u=0.80, c=900
    current = _cells({"A": (50000.0, 20000.0, 10_000_000.0),  # same u,c
                      "B": (50000.0, 40000.0, 36_000_000.0)})
    d = _mix(prior, current).iloc[0]
    assert d["util_trend"] == pytest.approx(1.0)
    assert d["cost_trend"] == pytest.approx(1.0)
    assert d["mix_trend"] == pytest.approx(d["pmpm_trend"])            # all trend is mix
    assert d["util_effect"] == pytest.approx(0.0, abs=1e-9)
    assert d["cost_effect"] == pytest.approx(0.0, abs=1e-9)


def test_single_mix_cell_gives_zero_mix_and_matches_two_way():
    # one cell -> no composition to shift; reduces to the two-way factors
    prior = _cells({"only": (1000.0, 400.0, 100_000.0)})
    current = _cells({"only": (1000.0, 420.0, 110_250.0)})
    d = _mix(prior, current).iloc[0]
    two = _dec(prior, current).iloc[0]
    assert d["mix_trend"] == pytest.approx(1.0)
    assert d["util_trend"] == pytest.approx(two["util_trend"])
    assert d["cost_trend"] == pytest.approx(two["cost_trend"])


# 2x2 product x age fixture (within-cell util & cost fixed; only the joint mix moves)
def _pa(weights):
    u = {("A", "Y"): 0.40, ("A", "O"): 0.60, ("B", "Y"): 0.50, ("B", "O"): 0.80}
    c = {("A", "Y"): 500.0, ("A", "O"): 700.0, ("B", "Y"): 600.0, ("B", "O"): 900.0}
    rows = []
    for (p, a), share in weights.items():
        mm = share * 100000.0
        n = u[(p, a)] * mm
        rows.append(dict(product=p, age=a, member_months=mm, claim_count=n, claims=c[(p, a)] * n))
    return pd.DataFrame(rows)


_PA_PRIOR = _pa({("A", "Y"): 0.35, ("A", "O"): 0.15, ("B", "Y"): 0.30, ("B", "O"): 0.20})
_PA_CURR = _pa({("A", "Y"): 0.25, ("A", "O"): 0.15, ("B", "Y"): 0.25, ("B", "O"): 0.35})


def _padec(mix_by, **kw):
    return decompose_pmpm_trend(_PA_PRIOR, _PA_CURR, count_col="claim_count", loss_col="claims",
                                exposure_col="member_months", mix_by=mix_by, **kw)


def test_mix_by_list_cross_reconciles_and_captures_all_composition():
    d = _padec(["product", "age"]).iloc[0]
    assert d["util_trend"] * d["cost_trend"] * d["mix_trend"] == pytest.approx(d["pmpm_trend"])
    # within-cell util & cost are fixed in the fixture -> all movement is mix
    assert d["util_trend"] == pytest.approx(1.0)
    assert d["cost_trend"] == pytest.approx(1.0)
    assert d["mix_trend"] == pytest.approx(d["pmpm_trend"])


def test_cross_mix_is_not_sum_of_marginal_mixes():
    cross = _padec(["product", "age"]).iloc[0]["mix_trend"]
    prod = _padec("product").iloc[0]["mix_trend"]
    age = _padec("age").iloc[0]["mix_trend"]
    # association between product and age means logs don't add up
    assert np.log(cross) != pytest.approx(np.log(prod) + np.log(age), abs=1e-3)
    # but each single-dimension run still reconciles
    for one in ("product", "age"):
        d = _padec(one).iloc[0]
        assert d["util_trend"] * d["cost_trend"] * d["mix_trend"] == pytest.approx(d["pmpm_trend"])


def test_on_and_mix_by_compose():
    # report by line of business (on), measure mix across product within each line
    pri = pd.concat([_PRIOR.assign(lob="IP"), _PRIOR.assign(lob="OP")], ignore_index=True)
    cur = pd.concat([_CURR.assign(lob="IP"), _CURR.assign(lob="OP")], ignore_index=True)
    out = decompose_pmpm_trend(pri, cur, count_col="claim_count", loss_col="claims",
                               exposure_col="member_months", on="lob", mix_by="cell")
    assert set(out["lob"]) == {"IP", "OP"}
    for _, r in out.iterrows():
        assert r["util_trend"] * r["cost_trend"] * r["mix_trend"] == pytest.approx(r["pmpm_trend"])


def test_mix_by_requires_positive_cells():
    prior = _cells({"A": (1000.0, 400.0, 100_000.0), "B": (500.0, 200.0, 60_000.0)})
    current = _cells({"A": (1000.0, 0.0, 0.0), "B": (500.0, 210.0, 63_000.0)})  # A has no claims now
    with pytest.raises(ValueError, match="positive"):
        _mix(prior, current)


def test_three_way_reconciles_on_random_books():
    rng = np.random.default_rng(0)
    for _ in range(25):
        k = int(rng.integers(2, 6))
        def period():
            mm = rng.uniform(500, 5000, k)
            u = rng.uniform(0.1, 1.5, k)
            cost = rng.uniform(100, 1500, k)
            n = u * mm
            return pd.DataFrame(dict(cell=[f"c{i}" for i in range(k)],
                                     member_months=mm, claim_count=n, claims=cost * n))
        d = _mix(period(), period()).iloc[0]
        assert d["util_trend"] * d["cost_trend"] * d["mix_trend"] == pytest.approx(d["pmpm_trend"])
        assert d["util_effect"] + d["cost_effect"] + d["mix_effect"] == pytest.approx(d["pmpm_change"])
