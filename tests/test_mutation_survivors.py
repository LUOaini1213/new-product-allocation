"""Tests added for the mutants that survived the first mutation run (results/mutation.json, first_run).

Each test names the mutant in scripts/mutants.py that it kills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from npalloc import backtest as B, bass, cleaning, launches as L, metrics as M
from npalloc.allocation import Economics, Rules, greedy_allocate, proportional_allocate

from test_cleaning import _orders
from test_launches import prod_frame


def test_cancellation_with_the_same_timestamp_is_matched():
    """c3_same_timestamp: 'on or before' includes an order stamped at the same minute as its cancellation."""
    orders = _orders([(1.0, "11111", 12, "2010-01-05 10:00")])
    cancels = _orders([(1.0, "11111", -12, "2010-01-05 10:00")])
    kept, n = cleaning.remove_matched_cancellations(orders, cancels, window_days=30)
    assert n == 1 and len(kept) == 0


def test_succession_threshold_is_inclusive_and_needs_a_phased_out_code():
    """l5_jaccard_edge and l5_no_upper_gap."""
    prod = prod_frame(
        [
            ("40001", "2010-01-05", "2010-07-20", "LARGE RED RETROSPOT MUG"),
            ("40010", "2010-07-25", "2011-06-01", "LARGE RED RETROSPOT MUG NEW"),
            ("41001", "2010-01-05", "2011-06-01", "SMALL BLUE SPOT BOWL"),          # same range, still on sale a year later
            ("41010", "2010-07-25", "2011-06-01", "SMALL BLUE SPOT BOWL NEW"),
        ]
    )
    price = pd.Series(2.0, index=prod.index)
    out = L.flag_succession(prod, pd.Index(["40010", "41010"]), price, 0.8, 1.25, 56)
    assert out.loc["40010", "succession"]                     # Jaccard is exactly 4/5 = the threshold
    assert not out.loc["41010", "succession"]                 # the look-alike was never phased out


def test_bass_cdf_hand_value_with_unequal_coefficients():
    """p3_cdf_ratio: p = 0.1, q = 0.3, t = 2 -> (1 - e^-0.8) / (1 + 3 e^-0.8)."""
    e = np.exp(-0.8)
    assert bass.bass_cdf(2, 0.1, 0.3) == pytest.approx((1 - e) / (1 + 3 * e))
    assert bass.bass_cdf(2, 0.1, 0.3) == pytest.approx(0.2345290, abs=1e-6)


def test_wape_is_scaled_by_actuals():
    """m1_wape_denominator: errors 10 + 10 over actual 40 (the forecasts sum to 60)."""
    assert M.wape(np.array([10.0, 30.0]), np.array([20.0, 40.0])) == pytest.approx(0.5)


def test_greedy_keeps_a_case_at_the_dc_when_indifferent():
    """a4_keep_tie: a case worth exactly the DC value is not shipped."""
    # salvage 0.5, DC 0.75: marginal = 10 * (0.5 + 0.5 P) ; P = 0.5 gives exactly 7.5 = DC value
    scen = np.array([[1], [1], [3], [3]])
    e = Economics(price=10.0, salvage=0.5, dc_value=0.75, ship_cost=np.array([0.0]))
    r = Rules(pack=1, min_cases=np.array([0]), priority_order=(), max_share=1.0)
    assert greedy_allocate(scen, 5, r, e).tolist() == [1]
    assert greedy_allocate(scen, 5, r, e, ship_all=True).tolist() == [5]


def test_proportional_repair_tie_breaks():
    """a5_donor_tie and a5_destination_tie."""
    # 8 cases at 4/4/0: channel 2 needs one case; both donors have the same slack -> the higher index gives
    r = Rules(pack=1, min_cases=np.array([0, 0, 1]), priority_order=(2,), max_share=1.0)
    assert proportional_allocate(np.array([0.5, 0.5, 0.0]), 8, r).tolist() == [4, 3, 1]
    # 10 cases at 6/2/2 with a cap of 5: the extra case goes to the lower index of the two equal gaps
    r = Rules(pack=1, min_cases=np.array([0, 0, 0]), priority_order=(), max_share=0.5)
    assert proportional_allocate(np.array([0.6, 0.2, 0.2]), 10, r).tolist() == [5, 3, 2]


def test_stock_is_never_below_one_case():
    """b1_zero_stock: a forecast of zero still gets one case."""
    assert B.stock_units(0.0, 1.0, 12) == 12


def test_post_launch_choice_uses_exactly_the_validated_rows():
    """post_selection_on_all_rows: the three validated launches (rows 3-5) are forecast exactly by their priors;
    the first three rows have no sales after week 2 and must not enter the choice or the residuals."""
    from npalloc import pipeline as P
    from test_forecast import launch_data

    weekly = np.array([[4, 0, 0, 0], [8, 0, 0, 0], [12, 0, 0, 0], [16, 0, 32, 16], [20, 0, 40, 20], [24, 0, 48, 24]])
    data = launch_data([(f"L{i}", "X", "other", 1.0, f"2010-06-{i + 1:02d}", 0) for i in range(6)], weekly=weekly)
    pm = P.fit_post_model(data, 2, 4, np.array([64.0, 80.0, 96.0]), np.array([0.1, -0.1, 0.0]), np.array([3, 4, 5]))
    assert pm.selected == "prelaunch_only"
    assert np.allclose(pm.residuals, 0.0)
