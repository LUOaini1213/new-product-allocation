"""Channel allocation of a limited stock under demand uncertainty (rules A1-A7).

Model (per launch, per decision). Stock is already bought, so its cost is sunk.
For channel c with x_c units shipped and random demand D_c over the horizon:

    value(x) = sum_c p * E[min(D_c, x_c)]            (sales)
             + sum_c v * p * E[(x_c - D_c)^+]         (channel leftover)
             - sum_c s_c * p * x_c                     (shipping)
             + v_dc * p * (Q - sum_c x_c)              (stock kept at the DC)

Demand uncertainty is a set of equally likely scenarios D[s, c]. The marginal
value of the k-th unit in channel c is p * (v - s_c + (1 - v) * P(D_c >= k)),
which never increases in k, so the expected value is separable and concave in
whole cases. Greedy marginal allocation from the lower bounds is therefore
optimal (classic resource-allocation result); ``mip_allocate`` solves the same
scenario problem as a mixed-integer programme so the two can be checked
against each other.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Economics:
    price: float                 # p, GBP per unit
    salvage: float               # v, leftover in a channel at horizon end, fraction of p
    dc_value: float              # v_dc, unit kept at the DC, fraction of p
    ship_cost: np.ndarray        # s_c, shipping cost per unit to channel c, fraction of p


@dataclass(frozen=True)
class Rules:
    pack: int                    # A1 case-pack: allocations are whole cases
    min_cases: np.ndarray        # A2 minimum cases per channel (0 = not a priority channel)
    priority_order: tuple        # A2 order in which minimums are served when stock is short
    max_share: float             # A3 max share of the stock Q per channel (1.0 = off)


def case_bounds(q_units: int, rules: Rules) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds in cases for an allocation of ``q_units``.

    A2: minimum cases are granted channel by channel in ``priority_order`` while
    whole cases remain; a channel whose minimum does not fit gets what is left
    in whole cases and later channels get nothing.
    A3: no channel may receive more than max_share * Q, rounded down to whole
    cases, but at least one case is always allowed (otherwise a one-case stock
    could not be shipped at all).
    """
    n_ch = len(rules.min_cases)
    total_cases = q_units // rules.pack
    lower = np.zeros(n_ch, dtype=np.int64)
    left = total_cases
    for c in rules.priority_order:
        give = min(int(rules.min_cases[c]), left)
        lower[c] = give
        left -= give
    cap = max(1, int(np.floor(rules.max_share * q_units / rules.pack + 1e-9)))
    upper = np.full(n_ch, cap, dtype=np.int64)
    upper = np.maximum(upper, lower)
    return lower, upper


def exceed_prob(scen: np.ndarray, max_units: int, start: np.ndarray | None = None) -> np.ndarray:
    """P(D_c >= start_c + k) for k = 1..max_units, from equally likely scenarios.

    scen has shape (S, C). Returns an array of shape (C, max_units).
    """
    s_count, n_ch = scen.shape
    start = np.zeros(n_ch, dtype=np.int64) if start is None else np.asarray(start, dtype=np.int64)
    out = np.zeros((n_ch, max_units))
    ks = np.arange(1, max_units + 1)
    for c in range(n_ch):
        d = np.sort(scen[:, c])
        # number of scenarios with D >= start + k  ==  S - (number with D < start + k)
        out[c] = (s_count - np.searchsorted(d, start[c] + ks, side="left")) / s_count
    return out


def unit_marginals(prob_ge: np.ndarray, econ: Economics) -> np.ndarray:
    """Marginal value (GBP) of the k-th extra unit in each channel, shape (C, K)."""
    v = econ.salvage
    return econ.price * ((v - econ.ship_cost)[:, None] + (1.0 - v) * prob_ge)


def pack_marginals(unit_m: np.ndarray, pack: int) -> np.ndarray:
    """Sum unit marginals over consecutive cases: shape (C, K // pack)."""
    n_ch, k = unit_m.shape
    n_cases = k // pack
    return unit_m[:, : n_cases * pack].reshape(n_ch, n_cases, pack).sum(axis=2)


def expected_value(x: np.ndarray, scen: np.ndarray, q_units: int, econ: Economics, onhand: np.ndarray | None = None) -> float:
    """Expected value of shipping x (on top of onhand) when q_units are available now."""
    x = np.asarray(x, dtype=float)
    held = x if onhand is None else x + np.asarray(onhand, dtype=float)
    sales = np.minimum(scen, held[None, :])
    left = held[None, :] - sales
    per = econ.price * (sales + econ.salvage * left).mean(axis=0)
    ship = econ.price * econ.ship_cost * x
    dc = econ.price * econ.dc_value * (q_units - x.sum())
    return float(per.sum() - ship.sum() + dc)


def greedy_allocate(
    scen: np.ndarray,
    q_units: int,
    rules: Rules,
    econ: Economics,
    onhand: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    ship_all: bool = False,
) -> np.ndarray:
    """A4: optimal allocation by marginal analysis, in whole cases.

    Starts every channel at its lower bound, then repeatedly ships the case
    with the largest marginal value as long as a whole case is left, the
    channel is below its upper bound and the case is worth more than keeping
    it at the DC (``ship_all`` drops the last condition). Ties go to the
    lower channel index.
    """
    n_ch = scen.shape[1]
    if lower is None or upper is None:
        lo, up = case_bounds(q_units, rules)
        lower = lo if lower is None else lower
        upper = up if upper is None else upper
    pack = rules.pack
    budget = q_units // pack
    cases = np.array(lower, dtype=np.int64).copy()
    budget -= int(cases.sum())
    if budget < 0:
        raise ValueError("lower bounds exceed the stock")
    max_cases = int(np.max(upper)) if n_ch else 0
    start = np.zeros(n_ch, dtype=np.int64) if onhand is None else np.asarray(onhand, dtype=np.int64)
    prob = exceed_prob(scen, max_cases * pack, start)
    pm = pack_marginals(unit_marginals(prob, econ), pack)
    keep = econ.price * econ.dc_value * pack
    heap = []
    for c in range(n_ch):
        j = cases[c]
        if j < upper[c]:
            heapq.heappush(heap, (-pm[c, j], c))
    while budget > 0 and heap:
        neg, c = heapq.heappop(heap)
        if not ship_all and -neg <= keep:
            break
        cases[c] += 1
        budget -= 1
        j = cases[c]
        if j < upper[c]:
            heapq.heappush(heap, (-pm[c, j], c))
    return cases * pack


def mip_allocate(
    scen: np.ndarray,
    q_units: int,
    rules: Rules,
    econ: Economics,
    time_limit: float = 60.0,
) -> tuple[np.ndarray, float]:
    """A4 cross-check: the same scenario problem as a mixed-integer programme.

    Variables: n_c (integer cases) and s_cs (sales in channel c, scenario s).
    max  sum_c pack*n_c*p*(v - s_c - v_dc) + (1/S) sum_cs p*(1-v)*s_cs + p*v_dc*Q
    s.t. s_cs <= pack*n_c,  0 <= s_cs <= D_cs,  sum_c pack*n_c <= Q,
         lower_c <= n_c <= upper_c (rules A2/A3), n_c integer (rule A1).
    Solved with HiGHS through scipy.optimize.milp.
    """
    s_count, n_ch = scen.shape
    pack = rules.pack
    lower, upper = case_bounds(q_units, rules)
    p, v, vdc = econ.price, econ.salvage, econ.dc_value
    n_s = n_ch * s_count
    c_obj = np.concatenate(
        [
            -(pack * p * (v - econ.ship_cost - vdc)),
            -np.full(n_s, p * (1.0 - v) / s_count),
        ]
    )
    # s_cs - pack * n_c <= 0 ; variable order: n_0..n_{C-1}, then s ordered by (c, s)
    rows = np.arange(n_s)
    s_cols = n_ch + rows
    n_cols = np.repeat(np.arange(n_ch), s_count)
    a1 = sparse.csr_matrix(
        (np.concatenate([np.ones(n_s), -pack * np.ones(n_s)]), (np.concatenate([rows, rows]), np.concatenate([s_cols, n_cols]))),
        shape=(n_s, n_ch + n_s),
    )
    a2 = sparse.csr_matrix((pack * np.ones(n_ch), (np.zeros(n_ch, dtype=int), np.arange(n_ch))), shape=(1, n_ch + n_s))
    cons = [
        LinearConstraint(a1, -np.inf, 0.0),
        LinearConstraint(a2, -np.inf, float(q_units)),
    ]
    lb = np.concatenate([lower.astype(float), np.zeros(n_s)])
    ub = np.concatenate([upper.astype(float), scen.T.reshape(-1).astype(float)])
    integrality = np.concatenate([np.ones(n_ch), np.zeros(n_s)])
    res = milp(
        c_obj,
        constraints=cons,
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if not res.success:
        raise RuntimeError(f"MIP failed: {res.message}")
    cases = np.round(res.x[:n_ch]).astype(np.int64)
    value = float(-res.fun + p * vdc * q_units)
    return cases * pack, value


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Split ``total`` whole units in proportion to ``weights`` (Hamilton method).

    Remainders are ranked from largest to smallest; ties go to the lower index.
    All-zero weights split equally.
    """
    w = np.asarray(weights, dtype=float)
    if total <= 0:
        return np.zeros(len(w), dtype=np.int64)
    if w.sum() <= 0:
        w = np.ones(len(w))
    quota = w / w.sum() * total
    base = np.floor(quota + 1e-12).astype(np.int64)
    rem = quota - base
    left = total - int(base.sum())
    order = sorted(range(len(w)), key=lambda i: (-rem[i], i))
    for i in order[:left]:
        base[i] += 1
    return base


def proportional_allocate(shares: np.ndarray, q_units: int, rules: Rules) -> np.ndarray:
    """A5 baseline: ship all whole cases in proportion to historical shares, then
    repair to satisfy the minimum (A2) and maximum-share (A3) rules.

    Repair: a priority channel below its minimum takes cases one at a time from
    the channel with the most cases above its own lower bound (ties: highest
    index); a channel above its cap passes the excess, one case at a time, to
    the channel with the largest unmet proportional quota that is below its
    cap (ties: lowest index).
    """
    lower, upper = case_bounds(q_units, rules)
    total_cases = q_units // rules.pack
    cases = largest_remainder(shares, total_cases)
    quota = np.asarray(shares, dtype=float)
    quota = quota / quota.sum() * total_cases if quota.sum() > 0 else np.full(len(quota), total_cases / len(quota))
    for c in rules.priority_order:
        while cases[c] < lower[c]:
            slack = cases - lower
            slack[c] = -1
            donor = int(np.flatnonzero(slack == slack.max())[-1])
            if slack[donor] <= 0:
                break
            cases[donor] -= 1
            cases[c] += 1
    for c in range(len(cases)):
        while cases[c] > upper[c]:
            room = cases < upper
            if not room.any():
                break
            gap = np.where(room, quota - cases, -np.inf)
            dest = int(np.flatnonzero(gap == gap.max())[0])
            cases[c] -= 1
            cases[dest] += 1
    return cases * rules.pack
