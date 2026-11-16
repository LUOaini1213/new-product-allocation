"""Week-by-week replay of a launch against the demand that actually occurred (rule S1).

Each week, every channel sells min(on-hand, demand); the rest of that week's
demand is lost (no backorders). An optional review at the end of week k may
ship whole cases from the DC reserve; the review callback only sees the sales
recorded so far, never the demand, so it cannot peek at lost sales or at the
future.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .allocation import Economics

ReviewFn = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class Outcome:
    shipped: np.ndarray
    sales: np.ndarray
    lost: np.ndarray
    left_channel: np.ndarray
    left_dc: int
    value: float

    @property
    def demand(self) -> np.ndarray:
        return self.sales + self.lost


def outcome_value(sales: np.ndarray, left_channel: np.ndarray, left_dc: int, shipped: np.ndarray, econ: Economics) -> float:
    p = econ.price
    return float(
        p * sales.sum()
        + econ.salvage * p * left_channel.sum()
        + econ.dc_value * p * left_dc
        - p * float(np.dot(econ.ship_cost, shipped))
    )


def replay(
    demand: np.ndarray,
    q_units: int,
    first_shipment: np.ndarray,
    econ: Economics,
    review_week: int | None = None,
    review: ReviewFn | None = None,
) -> Outcome:
    """Replay weekly demand (shape W x C) for a stock of q_units.

    ``first_shipment`` leaves the DC at week 0. If ``review`` is given it is
    called once after week ``review_week`` with (sales so far, shape k x C;
    on-hand per channel; units left at the DC) and returns extra units per
    channel, which arrive before the next week starts.
    """
    demand = np.asarray(demand, dtype=np.int64)
    n_weeks, n_ch = demand.shape
    shipped = np.asarray(first_shipment, dtype=np.int64).copy()
    if shipped.sum() > q_units or (shipped < 0).any():
        raise ValueError("first shipment exceeds the stock")
    onhand = shipped.copy()
    dc = int(q_units - shipped.sum())
    sales_hist = np.zeros((n_weeks, n_ch), dtype=np.int64)
    lost = np.zeros(n_ch, dtype=np.int64)
    for w in range(n_weeks):
        s = np.minimum(onhand, demand[w])
        sales_hist[w] = s
        onhand -= s
        lost += demand[w] - s
        if review is not None and review_week is not None and w + 1 == review_week:
            extra = np.asarray(review(sales_hist[: w + 1].copy(), onhand.copy(), dc), dtype=np.int64)
            if (extra < 0).any() or extra.sum() > dc:
                raise ValueError("review shipped more than the DC holds")
            onhand += extra
            shipped += extra
            dc -= int(extra.sum())
    sales = sales_hist.sum(axis=0)
    return Outcome(
        shipped=shipped,
        sales=sales,
        lost=lost,
        left_channel=onhand,
        left_dc=dc,
        value=outcome_value(sales, onhand, dc, shipped, econ),
    )
