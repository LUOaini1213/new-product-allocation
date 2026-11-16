"""Bass diffusion curve and post-launch updates (rules P1-P4).

Cumulative sales after t weeks: N(t) = m * F(t) with
    F(t) = (1 - exp(-(p + q) t)) / (1 + (q / p) exp(-(p + q) t)).
The shape (p, q) is fitted once per forecast origin on the mean normalised
cumulative curve of the training launches; after k weeks of sales the scale m
is updated, either from the sales alone or by precision-weighting the sales
with the pre-launch forecast.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def bass_cdf(t, p: float, q: float):
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return (1.0 - e) / (1.0 + (q / p) * e)


def fit_shape(cum_share: np.ndarray, horizon: int) -> tuple[float, float]:
    """Least-squares (p, q) so that F(t)/F(H) matches the mean normalised curve.

    cum_share has shape (n, H): cumulative units after week t divided by the
    H-week total, for each training launch.
    """
    target = np.asarray(cum_share, dtype=float).mean(axis=0)
    t = np.arange(1, horizon + 1)

    def resid(theta):
        p, q = np.exp(theta)
        f = bass_cdf(t, p, q)
        return f / f[-1] - target

    best = None
    for p0, q0 in [(0.03, 0.38), (0.2, 0.1), (0.5, 0.01), (0.05, 1.0)]:
        sol = least_squares(resid, x0=np.log([p0, q0]), bounds=(np.log([1e-4, 1e-6]), np.log([5.0, 5.0])))
        if best is None or sol.cost < best.cost:
            best = sol
    p, q = np.exp(best.x)
    return float(p), float(q)


def run_rate_remaining(sold_k: np.ndarray, k: int, horizon: int) -> np.ndarray:
    """P1: average weekly sales so far times the weeks left."""
    return np.asarray(sold_k, dtype=float) / k * (horizon - k)


def curve_ratio(train_weekly: np.ndarray, k: int) -> float:
    """P2: median over training launches of (units in weeks k+1..H) / (units in weeks 1..k)."""
    first = train_weekly[:, :k].sum(axis=1)
    rest = train_weekly[:, k:].sum(axis=1)
    ok = first > 0
    return float(np.median(rest[ok] / first[ok]))


def bass_remaining(sold_k: np.ndarray, k: int, horizon: int, p: float, q: float) -> np.ndarray:
    """P3: scale m = sales_k / F(k); remaining = m * (F(H) - F(k))."""
    fk, fh = bass_cdf(k, p, q), bass_cdf(horizon, p, q)
    return np.asarray(sold_k, dtype=float) * (fh - fk) / fk


def precision_weight(var_prior: float, var_data: float) -> float:
    """P4: weight on the sales-based estimate when combining two log-normal estimates."""
    if var_prior <= 0 and var_data <= 0:
        return 0.5
    return float(var_prior / (var_prior + var_data))


def bass_bayes_remaining(
    sold_k: np.ndarray, prior_total: np.ndarray, k: int, horizon: int, p: float, q: float, w: float
) -> np.ndarray:
    """P4: log m_post = w log(sales_k / F(k)) + (1 - w) log(prior_total / F(H));
    remaining = m_post * (F(H) - F(k)).

    Both sales_k and prior_total are floored at one unit before taking logs.
    """
    fk, fh = bass_cdf(k, p, q), bass_cdf(horizon, p, q)
    m_data = np.maximum(np.asarray(sold_k, dtype=float), 1.0) / fk
    m_prior = np.maximum(np.asarray(prior_total, dtype=float), 1.0) / fh
    m_post = np.exp(w * np.log(m_data) + (1.0 - w) * np.log(m_prior))
    return m_post * (fh - fk)
