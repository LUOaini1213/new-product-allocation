"""Hand-checked cases for the allocation rules A1-A5 and the MIP/greedy/brute-force cross-checks."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from npalloc.allocation import (
    Economics, Rules, case_bounds, exceed_prob, expected_value, greedy_allocate, largest_remainder,
    mip_allocate, pack_marginals, proportional_allocate, unit_marginals,
)


def econ(price=10.0, salvage=0.3, dc=0.5, ship=(0.0, 0.0)):
    return Economics(price=price, salvage=salvage, dc_value=dc, ship_cost=np.array(ship, dtype=float))


def rules(pack=1, mins=(0, 0), order=(), max_share=1.0):
    return Rules(pack=pack, min_cases=np.array(mins), priority_order=tuple(order), max_share=max_share)


def test_exceed_prob_hand_case():
    scen = np.array([[0, 5], [2, 5], [2, 5], [4, 5]])
    p = exceed_prob(scen, 5)
    # channel 0: P(D>=1)=3/4, P(D>=2)=3/4, P(D>=3)=1/4, P(D>=4)=1/4, P(D>=5)=0
    assert p[0].tolist() == [0.75, 0.75, 0.25, 0.25, 0.0]
    assert p[1].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_exceed_prob_with_on_hand_start():
    scen = np.array([[0, 5], [2, 5], [2, 5], [4, 5]])
    p = exceed_prob(scen, 2, start=np.array([2, 4]))
    assert p[0].tolist() == [0.25, 0.25]      # P(D>=3), P(D>=4)
    assert p[1].tolist() == [1.0, 0.0]        # P(D>=5), P(D>=6)


def test_unit_and_pack_marginals_hand_case():
    e = econ(price=10.0, salvage=0.3, ship=(0.1, 0.0))
    prob = np.array([[1.0, 0.5, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    um = unit_marginals(prob, e)
    # channel 0: 10*(0.3-0.1+0.7*P) -> 9, 5.5, 2, 2 ; channel 1: 10*(0.3+0.35)=6.5
    assert np.allclose(um, [[9.0, 5.5, 2.0, 2.0], [6.5, 6.5, 6.5, 6.5]])
    assert np.allclose(pack_marginals(um, 2), [[14.5, 4.0], [13.0, 13.0]])
    assert pack_marginals(um, 3).shape == (2, 1)


def test_expected_value_hand_case():
    # one channel, demand 2 or 6 with equal odds, ship 4 of 5 units, price 10
    scen = np.array([[2], [6]])
    e = Economics(price=10.0, salvage=0.3, dc_value=0.5, ship_cost=np.array([0.1]))
    # sales: 2 or 4 -> mean 3 -> 30 ; leftover: 2 or 0 -> mean 1 -> 0.3*10 = 3
    # shipping: 0.1*10*4 = 4 ; DC: 0.5*10*1 = 5  => 30 + 3 - 4 + 5 = 34
    assert expected_value(np.array([4]), scen, 5, e) == pytest.approx(34.0)
    # with 1 unit already on hand and 3 more shipped: same sales, shipping only on 3, DC keeps 2
    assert expected_value(np.array([3]), scen, 5, e, onhand=np.array([1])) == pytest.approx(30 + 3 - 3 + 10)


def test_case_bounds_minimums_in_priority_order():
    r = rules(pack=6, mins=(2, 1, 0), order=(0, 1), max_share=1.0)
    lo, up = case_bounds(60, r)
    assert lo.tolist() == [2, 1, 0] and up.tolist() == [10, 10, 10]
    lo, _ = case_bounds(12, r)            # two cases only: channel 0 takes both
    assert lo.tolist() == [2, 0, 0]
    lo, _ = case_bounds(17, r)            # 17 units = 2 whole cases
    assert lo.tolist() == [2, 0, 0]
    lo, _ = case_bounds(18, r)
    assert lo.tolist() == [2, 1, 0]
    r2 = rules(pack=6, mins=(2, 1, 0), order=(1, 0))
    lo, _ = case_bounds(12, r2)           # channel 1 is served first
    assert lo.tolist() == [1, 1, 0]


def test_case_bounds_max_share():
    r = rules(pack=5, mins=(0, 0), max_share=0.9)
    _, up = case_bounds(100, r)
    assert up.tolist() == [18, 18]        # 90 units = 18 cases
    _, up = case_bounds(40, r)            # 36 units -> 7 cases
    assert up.tolist() == [7, 7]
    _, up = case_bounds(5, r)             # a single case may always be shipped
    assert up.tolist() == [1, 1]
    r = rules(pack=5, mins=(3, 0), order=(0,), max_share=0.5)
    lo, up = case_bounds(20, r)           # cap 2 cases, but the minimum of 3 wins for channel 0
    assert lo.tolist() == [3, 0] and up.tolist() == [3, 2]


def test_greedy_hand_case():
    # two channels, unit packs, no shipping cost, keep-at-DC value 0.5
    # channel 0 demand: 1,1,3,3 -> P(>=1)=1, P(>=2)=.5, P(>=3)=.5, P(>=4)=0
    # channel 1 demand: 0,2,2,2 -> P(>=1)=.75, P(>=2)=.75, P(>=3)=0
    # marginal = 10*(0.3+0.7P): ch0 10, 6.5, 6.5, 3 ; ch1 8.25, 8.25, 3 ; DC keeps 5
    scen = np.array([[1, 0], [1, 2], [3, 2], [3, 2]])
    e = econ()
    assert greedy_allocate(scen, 3, rules(), e).tolist() == [1, 2]
    assert greedy_allocate(scen, 5, rules(), e).tolist() == [3, 2]
    assert greedy_allocate(scen, 9, rules(), e).tolist() == [3, 2]            # the rest stays at the DC
    assert greedy_allocate(scen, 7, rules(), e, ship_all=True).tolist() == [5, 2]  # all ties at 3.0 go to the lower index
    assert greedy_allocate(scen, 6, rules(), e, ship_all=True).tolist() == [4, 2]


def test_greedy_respects_rules():
    scen = np.array([[1, 0], [1, 2], [3, 2], [3, 2]])
    e = econ()
    r = rules(pack=2, mins=(0, 2), order=(1,), max_share=1.0)
    x = greedy_allocate(scen, 6, r, e)
    assert x.tolist() == [2, 4]            # channel 1 forced to 2 cases; the case left goes to channel 0 (16.5 > 10)
    r = rules(pack=1, mins=(0, 0), max_share=0.5)
    assert greedy_allocate(scen, 4, r, e).tolist() == [2, 2]
    assert all(v % 3 == 0 for v in greedy_allocate(scen, 10, rules(pack=3), e))


def test_greedy_with_on_hand_stock():
    # channel 0 already holds 3 units (its whole demand): further units are worth 3 < 5, so only channel 1 is served
    scen = np.array([[1, 0], [1, 2], [3, 2], [3, 2]])
    x = greedy_allocate(scen, 4, rules(), econ(), onhand=np.array([3, 0]), lower=np.array([0, 0]), upper=np.array([4, 4]))
    assert x.tolist() == [0, 2]


def test_greedy_rejects_infeasible_lower_bound():
    scen = np.array([[1, 0]])
    with pytest.raises(ValueError):
        greedy_allocate(scen, 2, rules(), econ(), lower=np.array([2, 1]), upper=np.array([5, 5]))


def brute_force(scen, q, r, e, ship_all=False):
    lo, up = case_bounds(q, r)
    budget = q // r.pack
    best, best_x = -np.inf, None
    for cases in itertools.product(*[range(l, u + 1) for l, u in zip(lo, up)]):
        if sum(cases) > budget or (ship_all and sum(cases) != min(budget, int(up.sum()))):
            continue
        x = np.array(cases) * r.pack
        v = expected_value(x, scen, q, e)
        if v > best + 1e-12:
            best, best_x = v, x
    return best, best_x


@pytest.mark.parametrize("seed", range(12))
def test_greedy_equals_brute_force_and_mip(seed):
    rng = np.random.default_rng(seed)
    n_ch = 3
    scen = rng.poisson(rng.uniform(1, 8, size=n_ch), size=(15, n_ch)) * rng.integers(0, 2, size=(15, n_ch))
    pack = int(rng.choice([1, 2, 3]))
    q = int(rng.integers(2, 9)) * pack
    r = Rules(pack=pack, min_cases=np.array([1, 0, int(rng.integers(0, 2))]), priority_order=(0, 2),
              max_share=float(rng.choice([0.6, 0.8, 1.0])))
    e = Economics(price=float(rng.uniform(1, 20)), salvage=float(rng.uniform(0.0, 0.5)), dc_value=float(rng.uniform(0.2, 0.6)),
                  ship_cost=rng.uniform(0.0, 0.2, size=n_ch))
    x = greedy_allocate(scen, q, r, e)
    v_greedy = expected_value(x, scen, q, e)
    v_bf, _ = brute_force(scen, q, r, e)
    x_mip, v_mip = mip_allocate(scen, q, r, e)
    assert v_greedy == pytest.approx(v_bf, abs=1e-9)
    assert v_mip == pytest.approx(v_bf, abs=1e-6)
    assert expected_value(x_mip, scen, q, e) == pytest.approx(v_mip, abs=1e-6)
    lo, up = case_bounds(q, r)
    for alloc in (x, x_mip):
        assert (alloc % pack == 0).all() and alloc.sum() <= q
        assert (alloc // pack >= lo).all() and (alloc // pack <= up).all()
    x_all = greedy_allocate(scen, q, r, e, ship_all=True)
    v_all, _ = brute_force(scen, q, r, e, ship_all=True)
    assert expected_value(x_all, scen, q, e) == pytest.approx(v_all, abs=1e-9)
    assert x_all.sum() == min(q // pack, int(up.sum())) * pack


def test_mip_hand_case_matches_greedy():
    scen = np.array([[1, 0], [1, 2], [3, 2], [3, 2]])
    x, v = mip_allocate(scen, 5, rules(), econ())
    assert x.tolist() == [3, 2]
    # sales means 2 and 1.5 -> 35 ; leftover means 1 and 0.5 -> 4.5 ; nothing at DC
    assert v == pytest.approx(39.5)


def test_largest_remainder():
    assert largest_remainder(np.array([0.5, 0.3, 0.2]), 10).tolist() == [5, 3, 2]
    assert largest_remainder(np.array([0.6, 0.3, 0.1]), 5).tolist() == [3, 2, 0]   # quotas 3, 1.5, 0.5: tie -> lower index
    assert largest_remainder(np.array([1, 1, 1]), 4).tolist() == [2, 1, 1]
    assert largest_remainder(np.array([0, 0]), 3).tolist() == [2, 1]
    assert largest_remainder(np.array([0.5, 0.5]), 0).tolist() == [0, 0]
    assert largest_remainder(np.array([0.82, 0.03, 0.15]), 7).sum() == 7


def test_proportional_allocate_and_repair():
    shares = np.array([0.82, 0.03, 0.15])
    r = Rules(pack=6, min_cases=np.array([1, 1, 0]), priority_order=(0, 1), max_share=1.0)
    # 10 cases: quotas 8.2, 0.3, 1.5 -> 8, 0, 2 ; channel 1 needs one case, taken from the donor with most slack (channel 0)
    assert proportional_allocate(shares, 60, r).tolist() == [42, 6, 12]
    r = Rules(pack=6, min_cases=np.array([0, 0, 0]), priority_order=(), max_share=0.7)
    # cap 7 cases: 8, 0, 2 -> the extra case goes to the channel with the largest unmet quota (channel 1: 0.3 vs -0.5)
    assert proportional_allocate(shares, 60, r).tolist() == [42, 6, 12]
    r = Rules(pack=1, min_cases=np.array([0, 0, 0]), priority_order=(), max_share=1.0)
    assert proportional_allocate(shares, 100, r).tolist() == [82, 3, 15]
    assert proportional_allocate(shares, 60, Rules(6, np.array([0, 0, 0]), (), 1.0)).sum() == 60
