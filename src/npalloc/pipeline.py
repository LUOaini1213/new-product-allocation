"""Rolling-origin evaluation of the forecasts (rules E1-E3) on the committed extracts.

For each calendar month M from ``test_start`` on (the *origin*):
  train = launches whose whole horizon ended on or before the first day of M,
  test  = launches first sold in M.
Nothing about a test launch other than its launch attributes is used before
its launch, and nothing about any launch whose horizon was still running at
the origin is used at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bass, forecast as F, metrics as M
from .categories import category_names
from .channels import channel_names

POST_METHODS = ["prelaunch_only", "run_rate", "curve_ratio", "bass", "bass_bayes"]


def load_launch_data(root: Path, params: dict) -> F.LaunchData:
    horizon = params["data"]["horizon_weeks"]
    names = channel_names(params)
    meta = pd.read_csv(root / "data" / "derived" / "launches.csv", dtype={"code": str}, keep_default_na=False)
    meta["launch_day"] = pd.to_datetime(meta["launch_day"])
    meta = meta.set_index("code")
    if not meta["launch_day"].is_monotonic_increasing:
        raise ValueError("launches.csv must be sorted by launch day (forward-chaining relies on it)")
    wk = pd.read_csv(root / "data" / "derived" / "launch_weekly.csv", dtype={"code": str})
    arr = np.zeros((len(meta), horizon, len(names)), dtype=np.int64)
    pos = {c: i for i, c in enumerate(meta.index)}
    cpos = {c: i for i, c in enumerate(names)}
    for code, week, ch, qty in zip(wk["code"], wk["week"], wk["channel"], wk["qty"]):
        arr[pos[code], week - 1, cpos[ch]] += qty
    return F.LaunchData(meta, arr)


def load_history_shares(root: Path, params: dict, origin: pd.Timestamp) -> np.ndarray:
    """A5: unit share per channel over the ``history_months`` whole months before the origin."""
    names = channel_names(params)
    cm = pd.read_csv(root / "data" / "derived" / "channel_month.csv")
    months = pd.to_datetime(cm["month"] + "-01")
    lo = origin - pd.DateOffset(months=params["allocation"]["history_months"])
    sel = cm[(months >= lo) & (months < origin)]
    units = sel.groupby("channel")["qty"].sum().reindex(names).fillna(0).to_numpy(dtype=float)
    return units / units.sum()


def launch_curve_summary(data: F.LaunchData) -> dict:
    """L6: average launch curve = mean over launches of each week's share of the launch's own total."""
    weekly = data.weekly.astype(float)
    share = weekly / weekly.sum(axis=1, keepdims=True)
    mean_share = share.mean(axis=0)
    return {
        "launches": int(len(data)),
        "mean_weekly_share": [float(x) for x in mean_share],
        "mean_cumulative_share": [float(x) for x in np.cumsum(mean_share)],
    }


def origins(data: F.LaunchData, params: dict) -> list[pd.Timestamp]:
    start = pd.Timestamp(params["forecast"]["test_start"])
    last = data.meta["launch_day"].max()
    out = []
    o = start
    while o <= last:
        out.append(o)
        o = o + pd.DateOffset(months=1)
    return out


def split(data: F.LaunchData, origin: pd.Timestamp, horizon_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """E1: boolean masks (train, test) for one origin."""
    day = data.meta["launch_day"]
    train = (day + pd.Timedelta(days=7 * horizon_weeks) <= origin).to_numpy()
    test = ((day >= origin) & (day < origin + pd.DateOffset(months=1))).to_numpy()
    return train, test


def select_by_cv(cv_male: dict[str, float]) -> str:
    """E2: the method with the lowest validation error on the training launches
    (ties: first in the dictionary order)."""
    best = None
    for name, v in cv_male.items():
        if best is None or v < cv_male[best]:
            best = name
    return best


@dataclass
class PostModel:
    k: int
    p: float
    q: float
    ratio: float
    weight: float
    selected: str
    residuals: np.ndarray          # log1p(actual remaining) - log1p(forecast), training launches

    def predict(self, method: str, sold_k: np.ndarray, prior_total: np.ndarray, horizon: int) -> np.ndarray:
        sold_k = np.asarray(sold_k, dtype=float)
        prior_total = np.asarray(prior_total, dtype=float)
        if method == "prelaunch_only":
            return np.maximum(prior_total - sold_k, 0.0)
        if method == "run_rate":
            return bass.run_rate_remaining(sold_k, self.k, horizon)
        if method == "curve_ratio":
            return sold_k * self.ratio
        if method == "bass":
            return bass.bass_remaining(sold_k, self.k, horizon, self.p, self.q)
        if method == "bass_bayes":
            return bass.bass_bayes_remaining(sold_k, prior_total, self.k, horizon, self.p, self.q, self.weight)
        raise KeyError(method)


def fit_post_model(train: F.LaunchData, k: int, horizon: int, prior_val: np.ndarray, prior_res: np.ndarray, rows: np.ndarray) -> PostModel:
    """Fit everything the post-launch updates need from training launches only.

    ``rows`` are the training launches that have a forward-chaining pre-launch
    forecast ``prior_val`` (with residuals ``prior_res``); the method choice and
    the calibration residuals use those launches only.
    """
    weekly = train.weekly.astype(float)
    total = weekly.sum(axis=1)
    cum = np.cumsum(weekly, axis=1) / total[:, None]
    p, q = bass.fit_shape(cum, horizon)
    ratio = bass.curve_ratio(weekly, k)
    sold = weekly[:, :k].sum(axis=1)
    fk, fh = bass.bass_cdf(k, p, q), bass.bass_cdf(horizon, p, q)
    data_err = np.log(np.maximum(sold, 1.0) * fh / fk) - np.log(np.maximum(total, 1.0))
    w = bass.precision_weight(float(np.var(prior_res)), float(np.var(data_err)))
    model = PostModel(k, p, q, ratio, w, "curve_ratio", np.zeros(1))
    rem, sold_v = (total - sold)[rows], sold[rows]
    errs = {m: M.male(rem, model.predict(m, sold_v, prior_val, horizon)) for m in POST_METHODS}
    model.selected = select_by_cv(errs)
    model.residuals = np.sort(np.log1p(rem) - np.log1p(model.predict(model.selected, sold_v, prior_val, horizon)))
    return model


@dataclass
class OriginModel:
    origin: pd.Timestamp
    train: F.LaunchData
    test_idx: np.ndarray
    selected: str
    residuals: np.ndarray                # sorted forward-chaining log residuals of the selected method
    index: F.AnalogueIndex               # fit on training descriptions/prices
    post: dict[int, PostModel] = field(default_factory=dict)


def evaluate_forecasts(data: F.LaunchData, params: dict, progress=None) -> tuple[dict, pd.DataFrame, list[OriginModel]]:
    """Run the rolling-origin evaluation. Returns (results dict, per-launch table, origin models)."""
    fp = params["forecast"]
    horizon = params["data"]["horizon_weeks"]
    cov = fp["pi_coverage"]
    cats = category_names(params)
    method_names = [m.name for m in F.make_methods(params)]
    rows = []
    models: list[OriginModel] = []
    origin_log = []
    for origin in origins(data, params):
        tr_mask, te_mask = split(data, origin, horizon)
        if te_mask.sum() == 0:
            continue
        train, test = data.take(tr_mask), data.take(te_mask)
        if progress:
            progress(f"origin {origin:%Y-%m}: train {len(train)}, test {len(test)}")
        preds, pis, val_male, val_res, cv_male, cv_res = {}, {}, {}, {}, {}, {}
        val_rows = None
        for m in F.make_methods(params):
            res, val_rows = F.forward_log_residuals(m, train, fp["forward_blocks"], fp["forward_min_train_blocks"])
            val_res[m.name] = res
            val_male[m.name] = float(np.mean(np.abs(res)))
            cv_res[m.name] = F.cv_log_residuals(m, train, fp["cv_folds"], fp["seed"])
            cv_male[m.name] = float(np.mean(np.abs(cv_res[m.name])))
            m.fit(train)
            preds[m.name] = m.predict(test)
            q_lo, q_hi = F.conformal_quantiles(res, cov)
            pis[m.name] = F.interval(preds[m.name], q_lo, q_hi)
        a = (1.0 - cov) / 2.0
        gq = []
        for qt in (a, 1.0 - a):
            g = F.GradientBoosting(cats, fp["analogue_k"], fp["price_weight"], fp["seed"], quantile=qt).fit(train)
            gq.append(np.maximum(np.expm1(g.predict_log(test)), 0.0))
        pis["gradient_boosting_quantile"] = (np.minimum(gq[0], gq[1]), np.maximum(gq[0], gq[1]))
        selected = select_by_cv(val_male)
        preds["selected"] = preds[selected]
        pis["selected"] = pis[selected]
        # recorded comparison: what random-fold cross-validation would have chosen and promised
        sel_cv = select_by_cv(cv_male)
        preds["selected_random_cv"] = preds[sel_cv]
        pis["selected_random_cv"] = F.interval(preds[sel_cv], *F.conformal_quantiles(cv_res[sel_cv], cov))
        om = OriginModel(
            origin, train, np.flatnonzero(te_mask), selected, np.sort(val_res[selected]),
            F.AnalogueIndex(fp["price_weight"]).fit(train.meta),
        )
        prior_val = np.expm1(np.log1p(train.total[val_rows]) - val_res[selected])
        post_pred = {}
        for k in fp["post_launch_weeks"]:
            pm = fit_post_model(train, k, horizon, prior_val, val_res[selected], val_rows)
            om.post[k] = pm
            sold = test.weekly[:, :k].sum(axis=1)
            for name in POST_METHODS:
                post_pred[(k, name)] = pm.predict(name, sold, preds["selected"], horizon)
            post_pred[(k, "selected")] = post_pred[(k, pm.selected)]
        models.append(om)
        origin_log.append(
            {
                "origin": origin.strftime("%Y-%m-%d"), "train": int(len(train)), "test": int(len(test)),
                "selected": selected, "validation_male": val_male, "validated_launches": int(len(val_rows)),
                "selected_random_cv": sel_cv, "random_cv_male": cv_male,
                "post": {str(k): {"p": pm.p, "q": pm.q, "ratio": pm.ratio, "weight": pm.weight, "selected": pm.selected}
                         for k, pm in om.post.items()},
            }
        )
        for j, code in enumerate(test.meta.index):
            row = {"code": code, "origin": origin.strftime("%Y-%m-%d"), "actual": int(test.total[j]),
                   "selected_method": selected, "selected_method_random_cv": sel_cv}
            for name in method_names + ["selected", "selected_random_cv"]:
                row[f"f_{name}"] = float(preds[name][j])
            for name, (lo, hi) in pis.items():
                row[f"lo_{name}"] = float(lo[j])
                row[f"hi_{name}"] = float(hi[j])
            for k in fp["post_launch_weeks"]:
                row[f"sold_{k}"] = int(test.weekly[j, :k].sum())
                for name in POST_METHODS + ["selected"]:
                    row[f"r{k}_{name}"] = float(post_pred[(k, name)][j])
            rows.append(row)
    table = pd.DataFrame(rows)
    results = summarise_forecasts(table, params, method_names)
    results["origins"] = origin_log
    results["launch_curve"] = launch_curve_summary(data)
    shapes = [pm for o in origin_log for pm in o["post"].values()]
    results["bass_shape"] = {
        "p_min": min(s["p"] for s in shapes), "p_max": max(s["p"] for s in shapes),
        "q_min": min(s["q"] for s in shapes), "q_max": max(s["q"] for s in shapes),
        "q_lower_bound": bass.Q_LOWER_BOUND,
        "q_at_lower_bound": bool(max(s["q"] for s in shapes) <= bass.Q_LOWER_BOUND * 1.01),
        "weight_min": min(s["weight"] for s in shapes), "weight_max": max(s["weight"] for s in shapes),
    }
    return results, table, models


def _metric_block(actual: np.ndarray, fc: dict[str, np.ndarray], idx: np.ndarray, level: float) -> dict:
    out = {}
    for name, f in fc.items():
        block = {}
        for mname, fn in M.METRICS.items():
            lo, hi = M.bootstrap_ci(lambda r, fn=fn, f=f: fn(actual[r], f[r]), idx, level)
            block[mname] = fn(actual, f)
            block[f"{mname}_ci"] = [lo, hi]
        out[name] = block
    return out


def _paired(actual: np.ndarray, f_a: np.ndarray, f_b: np.ndarray, idx: np.ndarray, level: float) -> dict:
    """Difference metric(a) - metric(b) with a paired bootstrap interval (negative = a is better)."""
    out = {}
    for mname in ("wape", "male"):
        fn = M.METRICS[mname]
        lo, hi = M.bootstrap_ci(lambda r: fn(actual[r], f_a[r]) - fn(actual[r], f_b[r]), idx, level)
        out[mname] = fn(actual, f_a) - fn(actual, f_b)
        out[f"{mname}_ci"] = [lo, hi]
        out[f"{mname}_significant"] = bool(hi < 0 or lo > 0)
    return out


def summarise_forecasts(table: pd.DataFrame, params: dict, method_names: list[str]) -> dict:
    fp = params["forecast"]
    actual = table["actual"].to_numpy(dtype=float)
    n = len(table)
    idx = M.bootstrap_indices(n, fp["bootstrap_reps"], fp["seed"])
    fc = {name: table[f"f_{name}"].to_numpy() for name in method_names + ["selected", "selected_random_cv"]}
    level = fp["ci_level"]
    pre = _metric_block(actual, fc, idx, level)
    simple = [m for m in method_names if not m.startswith("gradient_boosting")]
    best_simple = min(simple, key=lambda m: pre[m]["male"])
    best_simple_wape = min(simple, key=lambda m: pre[m]["wape"])
    comparisons = {
        "best_simple_by_male": best_simple,
        "best_simple_by_wape": best_simple_wape,
        "gradient_boosting_vs_best_simple_by_male": _paired(actual, fc["gradient_boosting"], fc[best_simple], idx, level),
        "gradient_boosting_vs_best_simple_by_wape": _paired(actual, fc["gradient_boosting"], fc[best_simple_wape], idx, level),
        "gradient_boosting_vs_global_median": _paired(actual, fc["gradient_boosting"], fc["global_median"], idx, level),
        "calendar_vs_no_calendar_boosting": _paired(actual, fc["gradient_boosting_calendar"], fc["gradient_boosting"], idx, level),
        "selected_vs_selected_random_cv": _paired(actual, fc["selected"], fc["selected_random_cv"], idx, level),
        "selected_vs_best_simple_by_male": _paired(actual, fc["selected"], fc[best_simple], idx, level),
        "analogues_vs_global_median": _paired(actual, fc[f"analogues_k{fp['analogue_k']}"], fc["global_median"], idx, level),
        "nearest_analogue_vs_global_median": _paired(actual, fc["nearest_analogue"], fc["global_median"], idx, level),
        "selected_vs_global_median": _paired(actual, fc["selected"], fc["global_median"], idx, level),
    }
    coverage = {}
    pi_names = [c[3:] for c in table.columns if c.startswith("lo_")]
    for name in pi_names:
        lo, hi = table[f"lo_{name}"].to_numpy(), table[f"hi_{name}"].to_numpy()
        inside = int(((actual >= lo) & (actual <= hi)).sum())
        w_lo, w_hi = M.wilson(inside, n)
        coverage[name] = {
            "nominal": fp["pi_coverage"], "covered": inside, "n": n, "coverage": inside / n,
            "wilson_ci": [w_lo, w_hi],
            "nominal_inside_ci": bool(w_lo <= fp["pi_coverage"] <= w_hi),
            "median_width_ratio": float(np.median((1.0 + hi) / (1.0 + lo))),
        }
    post = {}
    for k in fp["post_launch_weeks"]:
        rem = actual - table[f"sold_{k}"].to_numpy(dtype=float)
        fck = {name: table[f"r{k}_{name}"].to_numpy() for name in POST_METHODS + ["selected"]}
        block = _metric_block(rem, fck, idx, level)
        best_simple_k = min(("run_rate", "curve_ratio"), key=lambda m: block[m]["male"])
        post[str(k)] = {
            "methods": block,
            "best_simple_by_male": best_simple_k,
            "bass_vs_best_simple": _paired(rem, fck["bass"], fck[best_simple_k], idx, level),
            "bass_bayes_vs_best_simple": _paired(rem, fck["bass_bayes"], fck[best_simple_k], idx, level),
            "bass_bayes_vs_prelaunch_only": _paired(rem, fck["bass_bayes"], fck["prelaunch_only"], idx, level),
            "remaining_units": float(rem.sum()),
        }
    sel_counts = table.groupby("selected_method").size()
    sel_counts_cv = table.groupby("selected_method_random_cv").size()
    return {
        "params": fp,
        "horizon_weeks": params["data"]["horizon_weeks"],
        "origins_count": int(table["origin"].nunique()),
        "test_launches": n,
        "test_units": float(actual.sum()),
        "first_origin": table["origin"].min(),
        "last_origin": table["origin"].max(),
        "pre_launch": pre,
        "comparisons": comparisons,
        "interval_coverage": coverage,
        "post_launch": post,
        "selected_method_launches": {k: int(v) for k, v in sel_counts.items()},
        "selected_method_launches_random_cv": {k: int(v) for k, v in sel_counts_cv.items()},
    }
