"""Error metrics and bootstrap intervals (rules M1-M3)."""
from __future__ import annotations

import numpy as np


def wape(actual, forecast) -> float:
    """M1: sum |F - A| / sum A."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.abs(f - a).sum() / a.sum())


def male(actual, forecast) -> float:
    """M1: mean |log(1+F) - log(1+A)| (scale-free, every launch weighs the same)."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(np.log1p(f) - np.log1p(a))))


def bias(actual, forecast) -> float:
    """M1: sum (F - A) / sum A (positive = over-forecast)."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float((f - a).sum() / a.sum())


METRICS = {"wape": wape, "male": male, "bias": bias}


def bootstrap_indices(n: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(reps, n))


def bootstrap_ci(stat, idx: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """M2: percentile interval of ``stat(rows)`` over bootstrap row samples ``idx``."""
    vals = np.array([stat(rows) for rows in idx])
    a = (1.0 - level) / 2.0
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1.0 - a))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """M3: Wilson score interval for a proportion (used for PI coverage)."""
    if n == 0:
        return 0.0, 1.0
    ph = successes / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)
