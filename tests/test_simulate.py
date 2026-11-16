"""Hand-checked weekly replay (rule S1)."""
from __future__ import annotations

import numpy as np
import pytest

from npalloc.allocation import Economics
from npalloc.simulate import outcome_value, replay

ECON = Economics(price=10.0, salvage=0.3, dc_value=0.5, ship_cost=np.array([0.0, 0.1]))
DEMAND = np.array([[3, 1], [4, 0], [2, 2]])


def test_single_shipment():
    out = replay(DEMAND, 12, np.array([6, 4]), ECON)
    # channel 0: sells 3, 3 (1 lost), 0 (2 lost) ; channel 1: sells 1, 0, 2 -> 1 left
    assert out.sales.tolist() == [6, 3]
    assert out.lost.tolist() == [3, 0]
    assert out.left_channel.tolist() == [0, 1]
    assert out.left_dc == 2
    assert out.demand.tolist() == [9, 3]
    # 10*9 sales + 0.3*10*1 + 0.5*10*2 - 10*0.1*4 = 90 + 3 + 10 - 4
    assert out.value == pytest.approx(99.0)


def test_review_ships_from_dc_and_sees_only_sales():
    seen = {}

    def review(sales_hist, onhand, dc):
        seen["hist"], seen["onhand"], seen["dc"] = sales_hist, onhand, dc
        return np.array([2, 0])

    out = replay(DEMAND, 12, np.array([6, 4]), ECON, review_week=2, review=review)
    assert seen["hist"].tolist() == [[3, 1], [3, 0]]      # week-2 demand of 4 shows up as sales of 3
    assert seen["onhand"].tolist() == [0, 3] and seen["dc"] == 2
    assert out.sales.tolist() == [8, 3] and out.lost.tolist() == [1, 0]
    assert out.shipped.tolist() == [8, 4] and out.left_dc == 0
    assert out.value == pytest.approx(110 + 3 - 4)


def test_replay_rejects_overshipping():
    with pytest.raises(ValueError):
        replay(DEMAND, 5, np.array([6, 4]), ECON)
    with pytest.raises(ValueError):
        replay(DEMAND, 12, np.array([6, -1]), ECON)
    with pytest.raises(ValueError):
        replay(DEMAND, 12, np.array([6, 4]), ECON, review_week=1, review=lambda h, o, d: np.array([3, 0]))
    with pytest.raises(ValueError):
        replay(DEMAND, 12, np.array([6, 4]), ECON, review_week=1, review=lambda h, o, d: np.array([-1, 1]))


def test_review_not_called_without_week():
    called = []
    replay(DEMAND, 12, np.array([6, 4]), ECON, review_week=None, review=lambda h, o, d: called.append(1) or np.array([0, 0]))
    assert called == []


def test_outcome_value_components():
    v = outcome_value(np.array([5, 0]), np.array([0, 2]), 3, np.array([5, 2]), ECON)
    assert v == pytest.approx(50 + 6 + 15 - 2)
