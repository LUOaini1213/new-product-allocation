"""Pre-launch forecasts of a new product's demand over its first H weeks (rules F1-F6).

Every method follows the same contract: ``fit(train)`` sees only launches whose
whole horizon was observed before the forecast origin, ``predict(test)`` sees
only attributes known at launch (description, category, launch price, launch
date). Forecasts are point forecasts of total units in weeks 1..H.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

COLOUR_WORDS = frozenset(
    "RED PINK BLUE GREEN WHITE BLACK IVORY CREAM GOLD SILVER PURPLE ORANGE YELLOW BROWN GREY TURQUOISE".split()
)
XMAS_WORDS = frozenset("CHRISTMAS XMAS SANTA REINDEER SNOWFLAKE ADVENT NATIVITY".split())
BUNDLE_RE = re.compile(r"\b(?:SET|PACK|BOX|BAG|TIN)S? OF (\d+)\b")


@dataclass
class LaunchData:
    """Launch attributes plus realised demand arrays aligned on ``meta`` rows."""

    meta: pd.DataFrame                  # one row per launch, index = code
    weekly_channel: np.ndarray          # (n, H, C) units

    @property
    def weekly(self) -> np.ndarray:
        return self.weekly_channel.sum(axis=2)

    @property
    def channel(self) -> np.ndarray:
        return self.weekly_channel.sum(axis=1)

    @property
    def total(self) -> np.ndarray:
        return self.weekly_channel.sum(axis=(1, 2))

    def __len__(self) -> int:
        return len(self.meta)

    def take(self, idx) -> "LaunchData":
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return LaunchData(self.meta.iloc[idx], self.weekly_channel[idx])


# ---------------------------------------------------------------- features


def bundle_size(desc: str) -> int:
    m = BUNDLE_RE.search(desc)
    return int(m.group(1)) if m else 1


def days_to_christmas(day: pd.Timestamp) -> int:
    xmas = pd.Timestamp(year=day.year, month=12, day=25)
    if day > xmas:
        xmas = pd.Timestamp(year=day.year + 1, month=12, day=25)
    return int((xmas - day).days)


def attribute_features(meta: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """F5 features known at launch (no sales information)."""
    toks = meta["norm_desc"].str.split()
    cat_index = {c: i for i, c in enumerate(categories)}
    return pd.DataFrame(
        {
            "log_price": np.log(meta["launch_price"].astype(float)),
            "category": meta["category"].map(cat_index).astype(float),
            "month": meta["launch_day"].dt.month.astype(float),
            "days_to_xmas": meta["launch_day"].map(days_to_christmas).astype(float),
            "n_tokens": toks.map(len).astype(float),
            "bundle": meta["norm_desc"].map(bundle_size).astype(float),
            "colour": toks.map(lambda t: float(any(w in COLOUR_WORDS for w in t))),
            "xmas": toks.map(lambda t: float(any(w in XMAS_WORDS for w in t))),
        },
        index=meta.index,
    )


class AnalogueIndex:
    """F4: similarity between launches = text cosine (TF-IDF, fit on training
    descriptions only) minus a price penalty: d = (1 - cos) + w * |log p_i - log p_j|.
    """

    def __init__(self, price_weight: float):
        self.price_weight = price_weight

    def fit(self, meta: pd.DataFrame) -> "AnalogueIndex":
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        self.x = self.vec.fit_transform(meta["norm_desc"].tolist())
        self.logp = np.log(meta["launch_price"].to_numpy(dtype=float))
        return self

    def distances(self, meta: pd.DataFrame) -> np.ndarray:
        q = self.vec.transform(meta["norm_desc"].tolist())
        cos = (q @ self.x.T).toarray()
        logp = np.log(meta["launch_price"].to_numpy(dtype=float))
        return (1.0 - cos) + self.price_weight * np.abs(logp[:, None] - self.logp[None, :])

    def self_distances(self) -> np.ndarray:
        cos = (self.x @ self.x.T).toarray()
        d = (1.0 - cos) + self.price_weight * np.abs(self.logp[:, None] - self.logp[None, :])
        np.fill_diagonal(d, np.inf)
        return d


def nearest(d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances per row; ties go to the lower index."""
    k = min(k, d.shape[1])
    return np.argsort(d, axis=1, kind="stable")[:, :k]


# ------------------------------------------------------------------ methods


class Method:
    name = "method"

    def fit(self, train: LaunchData) -> "Method":
        raise NotImplementedError

    def predict(self, test: LaunchData) -> np.ndarray:
        raise NotImplementedError


class GlobalMean(Method):
    """F1: mean training launch curve; the forecast is the sum of that curve."""

    name = "global_mean_curve"

    def fit(self, train):
        self.curve = train.weekly.mean(axis=0)
        return self

    def predict(self, test):
        return np.full(len(test), float(self.curve.sum()))


class GlobalMedian(Method):
    """F1: median of training totals."""

    name = "global_median"

    def fit(self, train):
        self.v = float(np.median(train.total))
        return self

    def predict(self, test):
        return np.full(len(test), self.v)


class CategoryCurve(Method):
    """F2: category mean curve (or median total); categories with fewer than
    ``min_count`` training launches fall back to the global value."""

    def __init__(self, stat: str, min_count: int):
        self.stat = stat
        self.min_count = min_count
        self.name = f"category_{stat}"

    def _agg(self, x: np.ndarray) -> float:
        return float(np.mean(x) if self.stat == "mean" else np.median(x))

    def fit(self, train):
        tot = pd.Series(train.total, index=train.meta.index)
        cats = train.meta["category"]
        self.glob = self._agg(tot.to_numpy())
        self.by_cat = {}
        for c, grp in tot.groupby(cats):
            if len(grp) >= self.min_count:
                self.by_cat[c] = self._agg(grp.to_numpy())
        return self

    def predict(self, test):
        return np.array([self.by_cat.get(c, self.glob) for c in test.meta["category"]], dtype=float)


class Analogue(Method):
    """F4: median total of the k most similar earlier launches (k=1: nearest analogue)."""

    def __init__(self, k: int, price_weight: float):
        self.k = k
        self.price_weight = price_weight
        self.name = "nearest_analogue" if k == 1 else f"analogues_k{k}"

    def fit(self, train):
        self.index = AnalogueIndex(self.price_weight).fit(train.meta)
        self.y = train.total.astype(float)
        return self

    def predict(self, test):
        nn = nearest(self.index.distances(test.meta), self.k)
        return np.median(self.y[nn], axis=1)


class GradientBoosting(Method):
    """F5: gradient boosting on log(1+units) from launch attributes plus two
    analogue features (median log units of the k nearest training launches and
    the distance to the nearest one; leave-one-out on the training rows)."""

    name = "gradient_boosting"

    def __init__(self, categories: list[str], k: int, price_weight: float, seed: int, quantile: float | None = None):
        self.categories = categories
        self.k = k
        self.price_weight = price_weight
        self.seed = seed
        self.quantile = quantile

    def _model(self):
        kw = dict(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=self.seed,
            categorical_features=[1],
        )
        if self.quantile is not None:
            return HistGradientBoostingRegressor(loss="quantile", quantile=self.quantile, **kw)
        return HistGradientBoostingRegressor(**kw)

    def _features(self, meta: pd.DataFrame, d: np.ndarray) -> np.ndarray:
        x = attribute_features(meta, self.categories)
        nn = nearest(d, self.k)
        x["knn_log_median"] = np.median(self.logy[nn], axis=1)
        x["knn_dist1"] = np.take_along_axis(d, nn[:, :1], axis=1)[:, 0]
        return x.to_numpy(dtype=float)

    def fit(self, train):
        self.index = AnalogueIndex(self.price_weight).fit(train.meta)
        self.logy = np.log1p(train.total.astype(float))
        x = self._features(train.meta, self.index.self_distances())
        self.model = self._model().fit(x, self.logy)
        return self

    def predict_log(self, test) -> np.ndarray:
        x = self._features(test.meta, self.index.distances(test.meta))
        return self.model.predict(x)

    def predict(self, test):
        return np.maximum(np.expm1(self.predict_log(test)), 0.0)


def make_methods(params: dict) -> list[Method]:
    f = params["forecast"]
    cats = [c["name"] for c in params["categories"]["rules"]] + [params["categories"]["fallback"]]
    return [
        GlobalMean(),
        GlobalMedian(),
        CategoryCurve("mean", f["category_min_count"]),
        CategoryCurve("median", f["category_min_count"]),
        Analogue(1, f["price_weight"]),
        Analogue(f["analogue_k"], f["price_weight"]),
        GradientBoosting(cats, f["analogue_k"], f["price_weight"], f["seed"]),
    ]


# ------------------------------------------------------- calibration and PIs


def cv_log_residuals(method: Method, train: LaunchData, folds: int, seed: int) -> np.ndarray:
    """F6: out-of-fold residuals log(1+y) - log(1+yhat) inside the training set."""
    n = len(train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % folds
    res = np.empty(n)
    for f in range(folds):
        te = fold_of == f
        m = method.fit(train.take(~te))
        res[te] = np.log1p(train.total[te]) - np.log1p(m.predict(train.take(te)))
    return res


def conformal_quantiles(residuals: np.ndarray, coverage: float) -> tuple[float, float]:
    """F6: split-conformal residual quantiles for a central interval.

    Uses the finite-sample levels floor((n+1)a)/n and ceil((n+1)(1-a))/n with
    a = (1 - coverage)/2, clipped to the sample range.
    """
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    a = (1.0 - coverage) / 2.0
    lo_rank = int(np.floor((n + 1) * a + 1e-9))   # 1-based rank; 1e-9 guards 20 * 0.1 = 1.999...
    hi_rank = int(np.ceil((n + 1) * (1.0 - a) - 1e-9))
    lo = r[max(lo_rank, 1) - 1]
    hi = r[min(hi_rank, n) - 1]
    return float(lo), float(hi)


def interval(point: np.ndarray, q_lo: float, q_hi: float) -> tuple[np.ndarray, np.ndarray]:
    lp = np.log1p(np.asarray(point, dtype=float))
    return np.maximum(np.expm1(lp + q_lo), 0.0), np.expm1(lp + q_hi)
