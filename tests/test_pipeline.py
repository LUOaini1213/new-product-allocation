"""Hand-checked cases for the evaluation rules E1-E2 and the back-test rules B1-B4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from npalloc import backtest as B, forecast as F, pipeline as P
from npalloc.allocation import Economics, case_bounds
from npalloc.simulate import Outcome

from test_forecast import launch_data


def dated(days, totals=None):
    totals = totals or [10] * len(days)
    return launch_data([(f"L{i}", f"ITEM {i}", "other", 1.0, d, t) for i, (d, t) in enumerate(zip(days, totals))])


def test_split_has_no_leakage():
    data = dated(["2010-10-08", "2010-10-09", "2010-12-31", "2011-01-01", "2011-01-31", "2011-02-01"])
    train, test = P.split(data, pd.Timestamp("2011-01-01"), 12)
    # 2010-10-08 + 84 days = 2010-12-31 (observed) ; 2010-10-09 + 84 days = 2011-01-01 (allowed: ends as the origin starts)
    assert train.tolist() == [True, True, False, False, False, False]
    assert test.tolist() == [False, False, False, True, True, False]
    train, _ = P.split(data, pd.Timestamp("2010-12-31"), 12)
    assert train.tolist() == [True, False, False, False, False, False]
    assert not (train & test).any()


def test_launch_curve_summary():
    data = launch_data([("A", "X", "c", 1.0, "2010-06-01", 0), ("B", "Y", "c", 1.0, "2010-06-02", 0)],
                       weekly=[[5, 5, 0, 0], [1, 0, 0, 3]])
    got = P.launch_curve_summary(data)
    assert got["launches"] == 2
    assert got["mean_weekly_share"] == [0.375, 0.25, 0.0, 0.375]        # mean of (.5,.5,0,0) and (.25,0,0,.75)
    assert got["mean_cumulative_share"] == [0.375, 0.625, 0.625, 1.0]


def test_origins_cover_every_test_month(params):
    data = dated(["2010-06-01", "2011-01-15", "2011-03-02"])
    got = [o.strftime("%Y-%m-%d") for o in P.origins(data, params)]
    assert got == ["2011-01-01", "2011-02-01", "2011-03-01"]


def test_select_by_cv():
    assert P.select_by_cv({"a": 1.2, "b": 0.9, "c": 1.0}) == "b"
    assert P.select_by_cv({"a": 1.0, "b": 1.0}) == "a"


def test_forward_blocks_hand_case():
    got = F.forward_blocks(12, 6, 3)
    assert [(tr.tolist(), va.tolist()) for tr, va in got] == [
        (list(range(6)), [6, 7]), (list(range(8)), [8, 9]), (list(range(10)), [10, 11]),
    ]
    assert len(F.forward_blocks(12, 6, 5)) == 1


def test_forward_residuals_use_only_earlier_rows():
    data = dated([f"2010-06-{d:02d}" for d in range(1, 9)], [10, 20, 30, 40, 50, 60, 70, 80])

    class Memoriser(F.Method):
        def fit(self, train):
            self.known = dict(zip(train.meta.index, train.total))
            self.seen_max = max(train.meta["launch_day"])
            return self

        def predict(self, test):
            assert min(test.meta["launch_day"]) > self.seen_max        # validation rows are strictly later
            return np.array([self.known.get(c, 0.0) for c in test.meta.index], dtype=float)

    res, rows = F.forward_log_residuals(Memoriser(), data, blocks=4, min_train_blocks=2)
    assert rows.tolist() == [4, 5, 6, 7]
    assert np.allclose(res, np.log1p([50, 60, 70, 80]))
    res, rows = F.forward_log_residuals(F.GlobalMedian(), data, blocks=4, min_train_blocks=2)
    # medians of the first 4 rows (25) and of the first 6 rows (35)
    assert np.allclose(res, np.log1p([50, 60, 70, 80]) - np.log1p([25, 25, 35, 35]))


def test_history_shares(tmp_path, params):
    d = tmp_path / "data" / "derived"
    d.mkdir(parents=True)
    rows = ["month,channel,qty", "2009-12,United Kingdom,1000", "2010-01,United Kingdom,60", "2010-01,EIRE,20",
            "2010-12,France,20", "2011-01,Germany,500"]
    (d / "channel_month.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    h = P.load_history_shares(tmp_path, params, pd.Timestamp("2011-01-01"))
    # the 12 whole months before the origin are 2010-01 .. 2010-12: UK 60, EIRE 20, France 20
    assert h.tolist() == [0.6, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0]


def test_post_model_predictions():
    pm = P.PostModel(k=2, p=0.1, q=0.2, ratio=1.5, weight=1.0, selected="bass", residuals=np.zeros(1))
    sold, prior = np.array([40.0, 10.0]), np.array([100.0, 5.0])
    assert pm.predict("prelaunch_only", sold, prior, 12).tolist() == [60.0, 0.0]
    assert pm.predict("run_rate", sold, prior, 12).tolist() == [200.0, 50.0]
    assert pm.predict("curve_ratio", sold, prior, 12).tolist() == [60.0, 15.0]
    assert np.allclose(pm.predict("bass_bayes", sold, prior, 12), pm.predict("bass", sold, prior, 12))
    with pytest.raises(KeyError):
        pm.predict("nope", sold, prior, 12)


def test_fit_post_model_selects_on_validated_rows_only():
    # every launch sells s in week 1-2 and exactly 3s afterwards -> curve_ratio is exact on the validated rows
    weekly = np.array([[s, 0, 2 * s, s] for s in (4, 8, 12, 16, 20, 24)])
    data = launch_data([(f"L{i}", "X", "other", 1.0, f"2010-06-{i + 1:02d}", 0) for i in range(6)], weekly=weekly)
    rows = np.array([3, 4, 5])
    prior = np.array([1.0, 1.0, 1.0])
    pm = P.fit_post_model(data, 2, 4, prior, np.array([0.5, -0.5, 0.2]), rows)
    assert pm.ratio == 3.0 and pm.selected == "curve_ratio"
    assert np.allclose(pm.residuals, 0.0) and len(pm.residuals) == 3
    assert 0.0 < pm.weight <= 1.0


def test_load_launch_data_matches_extract(root, params):
    data = P.load_launch_data(root, params)
    wk = pd.read_csv(root / "data" / "derived" / "launch_weekly.csv", dtype={"code": str})
    assert data.weekly_channel.shape == (len(data.meta), 12, 7)
    assert int(data.total.sum()) == int(wk["qty"].sum())
    code = data.meta.index[0]
    assert int(data.total[0]) == int(wk.loc[wk["code"] == code, "qty"].sum())
    assert (data.weekly[:, 0] > 0).all()            # week 1 contains the first sale by construction


# ------------------------------------------------------------------ back-test helpers


def test_stock_units():
    assert B.stock_units(100.0, 1.0, 12) == 108      # 8.33 cases -> 9
    assert B.stock_units(96.0, 1.0, 12) == 96
    assert B.stock_units(100.0, 0.5, 12) == 60
    assert B.stock_units(0.4, 1.0, 12) == 12         # never less than one case
    assert B.stock_units(100.0, 1.5, 1) == 150


def test_residual_quantiles_and_scenarios():
    res = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(B.residual_quantiles(res, 2), [-0.5, 0.5])
    shares = np.array([[0.5, 0.5], [1.0, 0.0]])
    scen = B.demand_scenarios(99.0, np.zeros(3), shares, 4, np.random.default_rng(0))
    # zero residuals: every total is 99; shares alternate between the two analogue rows
    assert sorted(map(tuple, scen.tolist())) == [(50, 50), (50, 50), (99, 0), (99, 0)]
    scen = B.demand_scenarios(99.0, np.array([np.log(0.5), np.log(2.0)]), np.array([[1.0, 0.0]]), 2, np.random.default_rng(0))
    # quantiles 0.25 and 0.75 of the two residuals: log(0.5)+0.25*log 4 and log(0.5)+0.75*log 4
    assert scen[:, 0].tolist() == [int(np.rint(100 * 0.5 * 4 ** 0.25 - 1)), int(np.rint(100 * 0.5 * 4 ** 0.75 - 1))]
    assert (scen[:, 1] == 0).all()


def test_smoothed_shares():
    prior = np.array([0.8, 0.2])
    got = B.smoothed_shares(np.array([[0, 5], [80, 20], [0, 0]]), prior, 20)
    assert np.allclose(got, [[16 / 25, 9 / 25], [96 / 120, 24 / 120], [0.8, 0.2]])
    assert np.allclose(got.sum(axis=1), 1.0)


def test_first_stage_units():
    assert B.first_stage_units(120, 0.3, 12) == 84       # 10 cases -> 7 first
    assert B.first_stage_units(36, 0.3, 12) == 36        # 3 cases -> ceil(2.1) = 3
    assert B.first_stage_units(12, 0.3, 12) == 12
    assert B.first_stage_units(120, 0.0, 12) == 120


def test_make_rules_and_economics(params):
    names = ["United Kingdom", "EIRE", "Germany", "France", "Netherlands", "Other Europe", "Rest of World"]
    r = B.make_rules(6, params, names)
    assert r.pack == 6 and r.min_cases.tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert r.priority_order == (0, 1) and r.max_share == 0.9
    flipped = {"allocation": dict(params["allocation"], priority_channels=[{"name": "EIRE", "min_cases": 2}, {"name": "United Kingdom", "min_cases": 1}])}
    r2 = B.make_rules(6, flipped, names)
    assert r2.priority_order == (1, 0) and r2.min_cases.tolist() == [1, 2, 0, 0, 0, 0, 0]
    r = B.make_rules(6, params, names, use_min=False, use_max=False)
    assert r.min_cases.sum() == 0 and r.priority_order == () and r.max_share == 1.0
    e = B.make_economics(2.5, params, names)
    assert e.price == 2.5 and e.salvage == 0.3 and e.dc_value == 0.5
    assert e.ship_cost.tolist() == [0.02, 0.06, 0.08, 0.08, 0.08, 0.1, 0.15]
    e = B.make_economics(2.5, params, names, salvage=0.1, ship_scale=2.0)
    assert e.salvage == 0.1 and e.ship_cost[0] == 0.04


def outcome(sales, lost, left, dc, shipped, value):
    return Outcome(np.array(shipped), np.array(sales), np.array(lost), np.array(left), dc, value)


def test_aggregate_hand_case():
    outs = {
        "proportional": [outcome([6, 2], [2, 0], [0, 2], 0, [6, 4], 70.0), outcome([5, 0], [0, 0], [3, 2], 0, [8, 2], 30.0)],
        "newsvendor": [outcome([8, 2], [0, 0], [0, 0], 0, [8, 2], 100.0), outcome([5, 0], [0, 0], [1, 0], 4, [6, 0], 50.0)],
        "oracle": [outcome([8, 2], [0, 0], [0, 0], 0, [8, 2], 100.0), outcome([5, 0], [0, 0], [0, 0], 5, [5, 0], 100.0)],
    }
    idx = np.array([[0, 1], [0, 0], [1, 1]])
    res = B.aggregate(outs, np.array([10.0, 4.0]), idx)
    p, n = res["proportional"], res["newsvendor"]
    assert p["fill_rate"] == pytest.approx(13 / 15) and n["fill_rate"] == 1.0
    assert p["lost_units"] == 2.0 and p["lost_sales_value"] == 20.0
    assert p["leftover_channel_units"] == 7.0 and p["leftover_dc_units"] == 0.0
    assert p["stranded_share_of_stock"] == pytest.approx(7 / 20) and p["leftover_share_of_stock"] == pytest.approx(7 / 20)
    assert n["leftover_dc_units"] == 4.0 and n["stranded_share_of_stock"] == pytest.approx(1 / 20)
    assert n["leftover_share_of_stock"] == pytest.approx(5 / 20)
    assert p["value"] == 100.0 and p["value_per_launch"] == 50.0
    assert n["value_gain_vs_proportional"] == pytest.approx(0.5)
    # bootstrap rows: (0,1) -> 50/100 ; (0,0) -> 60/140 ; (1,1) -> 40/60
    assert n["value_gain_ci"][0] == pytest.approx(np.quantile([0.5, 60 / 140, 40 / 60], 0.025))
    assert n["value_gain_significant"] is True
    assert n["launches_better"] == 2 and n["launches_worse"] == 0
    assert n["fill_rate_gain_vs_proportional"] == pytest.approx(2 / 15)
    assert n["stranded_change_vs_proportional"] == pytest.approx(-6 / 7)
    assert p["value_share_of_oracle"] == pytest.approx(0.5) and n["value_share_of_oracle"] == pytest.approx(0.75)
    assert "value_gain_vs_proportional" not in p


def test_run_launch_invariants(params):
    names = ["United Kingdom", "EIRE", "Germany", "France", "Netherlands", "Other Europe", "Rest of World"]
    rng = np.random.default_rng(5)
    weekly = rng.poisson([30, 2, 1, 1, 1, 2, 1], size=(12, 7))
    hist = np.array([0.82, 0.03, 0.02, 0.025, 0.035, 0.055, 0.015])
    rows = B.smoothed_shares(rng.poisson([300, 20, 10, 10, 10, 20, 10], size=(20, 7)), hist, 20)
    pm = P.PostModel(k=4, p=0.08, q=1e-6, ratio=1.4, weight=0.8, selected="curve_ratio", residuals=np.sort(rng.normal(0, 0.6, 80)))
    om = P.OriginModel(pd.Timestamp("2011-01-01"), None, np.array([0]), "analogues_k10", np.sort(rng.normal(0, 1.0, 100)), None, {4: pm})
    rules = B.make_rules(6, params, names)
    econ = B.make_economics(2.0, params, names)
    out, extra = B.run_launch(weekly, 300.0, om, rows, hist, rules, econ, params, 1.0, 7, B.POLICIES)
    q = extra["stock"]
    assert q == 300
    lo, up = case_bounds(q, rules)
    for pol, o in out.items():
        assert (o.sales + o.lost == weekly.sum(axis=0)).all(), pol
        assert o.shipped.sum() + o.left_dc == q, pol
        assert (o.sales + o.left_channel == o.shipped).all(), pol
        assert (o.shipped % 6 == 0).all() and (o.shipped // 6 <= up).all(), pol
        assert (o.shipped // 6 >= lo).all(), pol
    for pol in ("proportional", "newsvendor_ship_all", "proportional_two_stage"):
        assert out[pol].left_dc == 0, pol
    assert extra["first"]["newsvendor_two_stage"].sum() <= B.first_stage_units(q, 0.3, 6)
    assert extra["first"]["proportional_two_stage"].sum() == B.first_stage_units(q, 0.3, 6)
    # hindsight is at least as good as every single-shipment policy under the same rules
    for pol in ("proportional", "newsvendor_ship_all", "newsvendor"):
        assert out["oracle"].value >= out[pol].value - 1e-9
    again, _ = B.run_launch(weekly, 300.0, om, rows, hist, rules, econ, params, 1.0, 7, B.POLICIES)
    assert all(again[p].value == out[p].value for p in out)
