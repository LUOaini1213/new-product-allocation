"""Hand-checked cases for the forecasting rules F1-F6, the Bass updates P1-P4 and the metrics M1-M3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from npalloc import bass, forecast as F, metrics as M


def launch_data(rows, weekly=None):
    """rows: (code, desc, category, price, launch_day, total). Units are put in week 1, channel 0 unless ``weekly``."""
    meta = pd.DataFrame(rows, columns=["code", "norm_desc", "category", "launch_price", "launch_day", "total"]).set_index("code")
    meta["launch_day"] = pd.to_datetime(meta["launch_day"])
    arr = np.zeros((len(meta), 4, 2), dtype=np.int64)
    if weekly is None:
        arr[:, 0, 0] = meta["total"].to_numpy()
    else:
        arr[:, :, 0] = np.asarray(weekly)
    return F.LaunchData(meta.drop(columns="total"), arr)


TRAIN = launch_data(
    [
        ("A", "RED MUG", "kitchen", 2.0, "2010-06-01", 10),
        ("B", "BLUE MUG", "kitchen", 2.0, "2010-06-02", 30),
        ("C", "GREEN MUG", "kitchen", 8.0, "2010-06-03", 20),
        ("D", "PAPER BAG", "bags", 1.0, "2010-06-04", 100),
        ("E", "JUTE BAG", "bags", 1.0, "2010-06-05", 300),
    ]
)
TEST = launch_data(
    [
        ("X", "RED MUG LARGE", "kitchen", 2.0, "2010-12-01", 0),
        ("Y", "PAPER BAG", "bags", 1.0, "2010-12-02", 0),
        ("Z", "ZINC POT", "garden", 3.0, "2010-12-03", 0),
    ]
)


def test_launch_data_views():
    d = launch_data([("A", "X", "c", 1.0, "2010-06-01", 0)], weekly=[[1, 2, 3, 4]])
    d.weekly_channel[0, :, 1] = [10, 0, 0, 0]
    assert d.weekly.tolist() == [[11, 2, 3, 4]]
    assert d.channel.tolist() == [[10, 10]]
    assert d.total.tolist() == [20]
    assert len(TRAIN.take(np.array([True, False, True, False, False]))) == 2
    assert TRAIN.take([4, 0]).meta.index.tolist() == ["E", "A"]


def test_global_baselines():
    assert F.GlobalMean().fit(TRAIN).predict(TEST).tolist() == [92.0, 92.0, 92.0]      # (10+30+20+100+300)/5
    assert F.GlobalMedian().fit(TRAIN).predict(TEST).tolist() == [30.0, 30.0, 30.0]


def test_category_baselines_with_fallback():
    m = F.CategoryCurve("mean", min_count=3).fit(TRAIN)
    # kitchen has 3 launches (mean 20); bags has only 2 -> global mean 92; garden unseen -> global
    assert m.predict(TEST).tolist() == [20.0, 92.0, 92.0]
    m = F.CategoryCurve("median", min_count=2).fit(TRAIN)
    assert m.predict(TEST).tolist() == [20.0, 200.0, 30.0]
    assert m.name == "category_median"


def test_analogue_distance_and_forecast():
    idx = F.AnalogueIndex(price_weight=0.25).fit(TRAIN.meta)
    d = idx.distances(TEST.meta)
    assert d[1, 3] == pytest.approx(0.0, abs=1e-9)             # identical text and price
    assert d[0, 0] < d[0, 1] < d[0, 3]                         # RED MUG closest to RED MUG LARGE; bags share no token
    # no shared token: distance = 1 + 0.25 * |log 3 - log 1|
    assert d[2, 3] == pytest.approx(1.0 + 0.25 * np.log(3.0))
    assert F.Analogue(1, 0.25).fit(TRAIN).predict(TEST)[:2].tolist() == [10.0, 100.0]
    assert F.Analogue(1, 0.25).name == "nearest_analogue" and F.Analogue(3, 0.25).name == "analogues_k3"
    # k=3 for X: the three mugs (the price gap of C costs 0.25*log 4 = 0.35, still closer than any bag) -> median 20
    assert F.Analogue(3, 0.25).fit(TRAIN).predict(TEST)[0] == 20.0
    # with a heavy price weight the expensive mug C drops out for a bag: {A, B, D or E} -> median 30
    assert F.Analogue(3, 5.0).fit(TRAIN).predict(TEST)[0] == 30.0


def test_self_distances_exclude_self():
    idx = F.AnalogueIndex(0.25).fit(TRAIN.meta)
    d = idx.self_distances()
    assert np.isinf(np.diag(d)).all()
    assert F.nearest(d, 1)[:, 0].tolist() != [0, 1, 2, 3, 4]


def test_nearest_ties_go_to_lower_index():
    d = np.array([[0.5, 0.2, 0.2, 0.9]])
    assert F.nearest(d, 2).tolist() == [[1, 2]]
    assert F.nearest(d, 10).shape == (1, 4)


def test_attribute_features():
    meta = pd.DataFrame(
        {
            "norm_desc": ["SET OF 6 RED CHRISTMAS BAUBLES", "PLAIN TIN"],
            "category": ["christmas", "storage"],
            "launch_price": [np.e, 1.0],
            "launch_day": pd.to_datetime(["2010-12-20", "2010-12-26"]),
        },
        index=["A", "B"],
    )
    x = F.attribute_features(meta, ["christmas", "storage", "other"])
    assert x.loc["A"].tolist() == [1.0, 0.0, 12.0, 5.0, 6.0, 6.0, 1.0, 1.0]
    assert x.loc["B"].tolist() == [0.0, 1.0, 12.0, 364.0, 2.0, 1.0, 0.0, 0.0]
    assert F.bundle_size("PACK OF 12 TISSUES") == 12 and F.bundle_size("BOX OF NOTHING") == 1
    assert F.days_to_christmas(pd.Timestamp("2011-12-25")) == 0


def test_gradient_boosting_learns_price_effect_and_is_deterministic():
    rng = np.random.default_rng(0)
    n = 300
    price = rng.choice([1.0, 10.0], size=n)
    total = np.where(price < 5, 400, 20) + rng.integers(0, 5, size=n)
    rows = [(f"P{i}", f"ITEM {i % 7}", "other", price[i], "2010-06-01", int(total[i])) for i in range(n)]
    data = launch_data(rows)
    test = launch_data([("T1", "ITEM 3", "other", 1.0, "2010-12-01", 0), ("T2", "ITEM 3", "other", 10.0, "2010-12-01", 0)])
    g = F.GradientBoosting(["other"], k=5, price_weight=0.25, seed=1).fit(data)
    p1 = g.predict(test)
    assert 300 < p1[0] < 500 and 10 < p1[1] < 40
    p2 = F.GradientBoosting(["other"], k=5, price_weight=0.25, seed=1).fit(data).predict(test)
    assert p1.tolist() == p2.tolist()
    lo = F.GradientBoosting(["other"], 5, 0.25, 1, quantile=0.1).fit(data).predict(test)
    hi = F.GradientBoosting(["other"], 5, 0.25, 1, quantile=0.9).fit(data).predict(test)
    assert (lo <= hi + 1e-9).all()


def test_cv_residuals_are_out_of_fold():
    class Memoriser(F.Method):
        """Predicts the training total of a known code exactly, otherwise 0: any leak gives a zero residual."""

        def fit(self, train):
            self.known = dict(zip(train.meta.index, train.total))
            return self

        def predict(self, test):
            return np.array([self.known.get(c, 0.0) for c in test.meta.index], dtype=float)

    res = F.cv_log_residuals(Memoriser(), TRAIN, folds=5, seed=3)
    assert np.allclose(res, np.log1p(TRAIN.total))
    res2 = F.cv_log_residuals(F.GlobalMedian(), TRAIN, folds=5, seed=3)
    # leave-one-out medians: without A -> 65, B -> 60, C -> 65, D -> 25, E -> 25
    want = np.log1p([10, 30, 20, 100, 300]) - np.log1p([65, 60, 65, 25, 25])
    assert np.allclose(res2, want)


def test_conformal_quantiles_and_interval():
    res = np.arange(1, 20, dtype=float) / 10.0            # 0.1 .. 1.9, n = 19
    lo, hi = F.conformal_quantiles(res, 0.8)
    # ranks floor(20*0.1)=2 and ceil(20*0.9)=18 -> 0.2 and 1.8
    assert (lo, hi) == (pytest.approx(0.2), pytest.approx(1.8))
    lo, hi = F.conformal_quantiles(np.array([-1.0, 0.0, 2.0]), 0.8)
    assert (lo, hi) == (-1.0, 2.0)                        # ranks clipped to the sample range
    low, high = F.interval(np.array([9.0]), np.log(0.5), np.log(2.0))
    assert low[0] == pytest.approx(4.0) and high[0] == pytest.approx(19.0)
    low, _ = F.interval(np.array([0.0]), -1.0, 1.0)
    assert low[0] == 0.0


def test_make_methods_names(params):
    names = [m.name for m in F.make_methods(params)]
    assert names == [
        "global_mean_curve", "global_median", "category_mean", "category_median",
        "nearest_analogue", "analogues_k10", "gradient_boosting",
    ]


# ------------------------------------------------------------------ Bass


def test_bass_cdf_values():
    # p = q = 0.5, t = 1: (1 - e^-1) / (1 + e^-1)
    assert bass.bass_cdf(1, 0.5, 0.5) == pytest.approx((1 - np.exp(-1)) / (1 + np.exp(-1)))
    assert bass.bass_cdf(0, 0.1, 0.3) == 0.0
    f = bass.bass_cdf(np.arange(1, 30), 0.05, 0.4)
    assert (np.diff(f) > 0).all() and f[-1] < 1.0


def test_fit_shape_recovers_known_curve():
    t = np.arange(1, 13)
    f = bass.bass_cdf(t, 0.08, 0.3)
    cum = np.vstack([f / f[-1], f / f[-1]])
    p, q = bass.fit_shape(cum, 12)
    g = bass.bass_cdf(t, p, q)
    assert np.allclose(g / g[-1], f / f[-1], atol=1e-4)


def test_post_launch_updates_hand_cases():
    assert bass.run_rate_remaining(np.array([20.0]), 2, 12).tolist() == [100.0]
    weekly = np.array([[10, 10, 30, 10], [5, 5, 10, 0], [0, 0, 7, 7], [4, 0, 2, 2]])
    # ratios rest/first for k=2: 40/20=2, 10/10=1, (first=0 skipped), 4/4=1 -> median 1
    assert bass.curve_ratio(weekly, 2) == 1.0
    p, q = 0.1, 0.2
    fk, fh = bass.bass_cdf(2, p, q), bass.bass_cdf(12, p, q)
    assert bass.bass_remaining(np.array([50.0]), 2, 12, p, q)[0] == pytest.approx(50.0 * (fh - fk) / fk)
    assert bass.precision_weight(1.0, 3.0) == 0.25
    assert bass.precision_weight(0.0, 0.0) == 0.5
    # weight 1 -> pure sales-based Bass ; weight 0 -> prior only, rescaled to the weeks left
    sold, prior = np.array([50.0]), np.array([400.0])
    assert bass.bass_bayes_remaining(sold, prior, 2, 12, p, q, 1.0)[0] == pytest.approx(50.0 * (fh - fk) / fk)
    assert bass.bass_bayes_remaining(sold, prior, 2, 12, p, q, 0.0)[0] == pytest.approx(400.0 * (fh - fk) / fh)
    mid = bass.bass_bayes_remaining(sold, prior, 2, 12, p, q, 0.5)[0]
    assert mid == pytest.approx(np.sqrt(50.0 / fk * 400.0 / fh) * (fh - fk))
    # zero sales are floored at one unit rather than sending the estimate to zero
    assert bass.bass_bayes_remaining(np.array([0.0]), prior, 2, 12, p, q, 1.0)[0] == pytest.approx((fh - fk) / fk)


# --------------------------------------------------------------- metrics


def test_metrics_hand_cases():
    a, f = np.array([10.0, 0.0, 30.0]), np.array([20.0, 5.0, 15.0])
    assert M.wape(a, f) == pytest.approx(30 / 40)
    assert M.bias(a, f) == pytest.approx(0.0)
    assert M.bias(a, f + 4) == pytest.approx(12 / 40)
    want = (abs(np.log(21 / 11)) + abs(np.log(6)) + abs(np.log(16 / 31))) / 3
    assert M.male(a, f) == pytest.approx(want)


def test_bootstrap_ci_is_deterministic_and_brackets():
    x = np.arange(100, dtype=float)
    idx = M.bootstrap_indices(100, 500, seed=7)
    assert idx.shape == (500, 100) and idx.min() >= 0 and idx.max() <= 99
    lo, hi = M.bootstrap_ci(lambda r: x[r].mean(), idx)
    assert lo < 49.5 < hi and hi - lo < 25
    assert (lo, hi) == M.bootstrap_ci(lambda r: x[r].mean(), M.bootstrap_indices(100, 500, seed=7))
    lo90, hi90 = M.bootstrap_ci(lambda r: x[r].mean(), idx, level=0.5)
    assert lo < lo90 < hi90 < hi


def test_wilson_interval():
    lo, hi = M.wilson(80, 100)
    assert lo == pytest.approx(0.7112, abs=1e-4) and hi == pytest.approx(0.8666, abs=1e-4)
    assert M.wilson(0, 0) == (0.0, 1.0)
    lo, hi = M.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)
