"""Back-test of the allocation policies on held-out launches (rules B1-B4).

Every decision uses only what was known when it was taken: the pre-launch
forecast and its out-of-fold residuals, the analogue launches in the training
set, the channel history before the origin and, at the review, the sales
recorded so far. The outcome is then computed against the demand that
actually occurred.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import forecast as F, metrics as M
from .allocation import (
    Economics, Rules, case_bounds, expected_value, greedy_allocate, largest_remainder, mip_allocate,
    proportional_allocate,
)
from .channels import channel_names
from .pipeline import OriginModel, load_history_shares
from .simulate import replay

POLICIES = ["proportional", "newsvendor_ship_all", "newsvendor", "proportional_two_stage", "newsvendor_two_stage", "oracle"]


def make_rules(pack: int, params: dict, names: list[str], use_min: bool = True, use_max: bool = True) -> Rules:
    ap = params["allocation"]
    mins = np.zeros(len(names), dtype=np.int64)
    order = []
    if use_min:
        for ch, v in ap["priority_min_cases"].items():
            mins[names.index(ch)] = int(v)
            order.append(names.index(ch))
    return Rules(pack=int(pack), min_cases=mins, priority_order=tuple(order), max_share=float(ap["max_share"]) if use_max else 1.0)


def make_economics(price: float, params: dict, names: list[str], salvage: float | None = None, ship_scale: float = 1.0,
                   dc_value: float | None = None) -> Economics:
    ap = params["allocation"]
    ship = np.array([ap["ship_cost"][n] for n in names], dtype=float) * ship_scale
    return Economics(
        price=float(price), salvage=float(ap["salvage"] if salvage is None else salvage),
        dc_value=float(ap["dc_value"] if dc_value is None else dc_value), ship_cost=ship,
    )


def stock_units(forecast: float, ratio: float, pack: int) -> int:
    """B1: stock = ratio x forecast, rounded up to whole cases, at least one case."""
    cases = int(np.ceil(ratio * forecast / pack - 1e-9))
    return max(cases, 1) * pack


def residual_quantiles(sorted_res: np.ndarray, n: int) -> np.ndarray:
    """B2: n equally likely residuals = the (s + 0.5)/n quantiles of the calibration residuals."""
    return np.quantile(sorted_res, (np.arange(n) + 0.5) / n)


def smoothed_shares(channel_units: np.ndarray, prior: np.ndarray, prior_units: float) -> np.ndarray:
    """B2: (units_c + a * prior_c) / (total + a): launches with few units lean on the prior."""
    u = np.asarray(channel_units, dtype=float)
    return (u + prior_units * prior) / (u.sum(axis=-1, keepdims=True) + prior_units)


def demand_scenarios(point: float, sorted_res: np.ndarray, share_rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """B2: scenario s = total_s x share_s, rounded to whole units.

    total_s = expm1(log1p(point) + residual quantile s); the share vectors are the
    analogue launches' smoothed channel shares, cycled and randomly paired with the totals.
    """
    totals = np.maximum(np.expm1(np.log1p(point) + residual_quantiles(sorted_res, n)), 0.0)
    pick = rng.permutation(n) % len(share_rows)
    return np.rint(totals[:, None] * share_rows[pick]).astype(np.int64)


def first_stage_units(q_units: int, reserve_fraction: float, pack: int) -> int:
    """B3: ship ceil((1 - reserve) x cases) cases first; the rest waits at the DC."""
    cases = q_units // pack
    return int(np.ceil((1.0 - reserve_fraction) * cases - 1e-9)) * pack


def run_launch(
    weekly_demand: np.ndarray, point: float, om: OriginModel, analogue_rows: np.ndarray, hist: np.ndarray,
    rules: Rules, econ: Economics, params: dict, stock_ratio: float, seed: int, policies: list[str],
) -> tuple[dict, dict]:
    ap = params["allocation"]
    horizon = params["data"]["horizon_weeks"]
    n_scen = ap["scenarios"]
    q = stock_units(point, stock_ratio, rules.pack)
    rng = np.random.default_rng(seed)
    scen = demand_scenarios(point, om.residuals, analogue_rows, n_scen, rng)
    out, extra = {}, {"stock": q, "scen": scen}
    k = ap["review_week"]
    prior_share = analogue_rows.mean(axis=0)
    lower_q, upper_q = case_bounds(q, rules)

    def nv_review(sales_hist, onhand, dc):
        pm = om.post[k]
        sold_c = sales_hist.sum(axis=0)
        rem_hat = float(pm.predict(pm.selected, np.array([sold_c.sum()]), np.array([point]), horizon)[0])
        r = np.random.default_rng(seed + 1)
        totals = np.maximum(np.expm1(np.log1p(rem_hat) + residual_quantiles(pm.residuals, n_scen)), 0.0)
        shares = r.dirichlet(sold_c + ap["share_prior_units"] * prior_share + 1e-9, size=n_scen)
        rem_scen = np.rint(r.permutation(totals)[:, None] * shares).astype(np.int64)
        shipped_cases = (first["newsvendor_two_stage"] // rules.pack)
        return greedy_allocate(
            rem_scen, dc, rules, econ, onhand=onhand,
            lower=np.zeros(len(onhand), dtype=np.int64), upper=np.maximum(upper_q - shipped_cases, 0),
        )

    def prop_review(sales_hist, onhand, dc):
        w = sales_hist.sum(axis=0) + ap["share_prior_units"] * hist
        return largest_remainder(w, dc // rules.pack) * rules.pack

    first = {}
    for pol in policies:
        review, rw = None, None
        if pol == "proportional":
            x = proportional_allocate(hist, q, rules)
        elif pol == "newsvendor_ship_all":
            x = greedy_allocate(scen, q, rules, econ, ship_all=True)
        elif pol == "newsvendor":
            x = greedy_allocate(scen, q, rules, econ)
        elif pol == "oracle":
            x = greedy_allocate(weekly_demand.sum(axis=0)[None, :], q, rules, econ)
        elif pol == "proportional_two_stage":
            x = proportional_allocate(hist, first_stage_units(q, ap["reserve_fraction"], rules.pack), rules)
            review, rw = prop_review, k
        elif pol == "newsvendor_two_stage":
            q1 = first_stage_units(q, ap["reserve_fraction"], rules.pack)
            lo1, _ = case_bounds(q1, rules)
            x = greedy_allocate(scen, q1, rules, econ, lower=lo1, upper=np.maximum(upper_q, lo1))
            review, rw = nv_review, k
        else:
            raise KeyError(pol)
        first[pol] = x
        out[pol] = replay(weekly_demand, q, x, econ, review_week=rw, review=review)
    extra["first"] = first
    return out, extra


def aggregate(outcomes: dict[str, list], price: np.ndarray, idx: np.ndarray, level: float = 0.95, reference: str = "proportional") -> dict:
    """B4: fill rate, lost sales, leftover and value per policy, with paired bootstrap intervals vs the reference."""
    arr = {}
    for pol, outs in outcomes.items():
        arr[pol] = {
            "sales": np.array([o.sales.sum() for o in outs], dtype=float),
            "lost": np.array([o.lost.sum() for o in outs], dtype=float),
            "demand": np.array([o.demand.sum() for o in outs], dtype=float),
            "left_channel": np.array([o.left_channel.sum() for o in outs], dtype=float),
            "left_dc": np.array([o.left_dc for o in outs], dtype=float),
            "shipped": np.array([o.shipped.sum() for o in outs], dtype=float),
            "value": np.array([o.value for o in outs], dtype=float),
        }
    res = {}
    ref = arr[reference]
    for pol, a in arr.items():
        stock = a["shipped"] + a["left_dc"]
        block = {
            "fill_rate": float(a["sales"].sum() / a["demand"].sum()),
            "lost_units": float(a["lost"].sum()),
            "lost_sales_value": float((a["lost"] * price).sum()),
            "leftover_channel_units": float(a["left_channel"].sum()),
            "leftover_dc_units": float(a["left_dc"].sum()),
            "leftover_share_of_stock": float((a["left_channel"].sum() + a["left_dc"].sum()) / stock.sum()),
            "stranded_share_of_stock": float(a["left_channel"].sum() / stock.sum()),
            "shipped_units": float(a["shipped"].sum()),
            "value": float(a["value"].sum()),
            "value_per_launch": float(a["value"].mean()),
        }
        if pol != reference:
            dv = a["value"] - ref["value"]
            lo, hi = M.bootstrap_ci(lambda r: dv[r].sum() / ref["value"][r].sum(), idx, level)
            block["value_gain_vs_proportional"] = float(dv.sum() / ref["value"].sum())
            block["value_gain_ci"] = [lo, hi]
            block["value_gain_significant"] = bool(lo > 0 or hi < 0)
            block["launches_better"] = int((dv > 1e-9).sum())
            block["launches_worse"] = int((dv < -1e-9).sum())
            lo, hi = M.bootstrap_ci(
                lambda r: a["sales"][r].sum() / a["demand"][r].sum() - ref["sales"][r].sum() / ref["demand"][r].sum(), idx, level
            )
            block["fill_rate_gain_vs_proportional"] = block["fill_rate"] - float(ref["sales"].sum() / ref["demand"].sum())
            block["fill_rate_gain_ci"] = [lo, hi]
            ds = a["left_channel"] - ref["left_channel"]
            lo, hi = M.bootstrap_ci(lambda r: ds[r].sum() / ref["left_channel"][r].sum(), idx, level)
            block["stranded_change_vs_proportional"] = float(ds.sum() / ref["left_channel"].sum())
            block["stranded_change_ci"] = [lo, hi]
        res[pol] = block
    if "oracle" in arr:
        for pol in res:
            res[pol]["value_share_of_oracle"] = float(arr[pol]["value"].sum() / arr["oracle"]["value"].sum())
    return res


def run(data: F.LaunchData, models: list[OriginModel], table: pd.DataFrame, params: dict, root: Path, progress=None):
    ap, fp = params["allocation"], params["forecast"]
    names = channel_names(params)
    point_of = dict(zip(table["code"], table["f_selected"]))
    jobs = []
    for om in models:
        hist = load_history_shares(root, params, om.origin)
        test = data.take(om.test_idx)
        d = om.index.distances(test.meta)
        nn = F.nearest(d, ap["share_analogues"])
        tr_channel = om.train.channel
        for j, code in enumerate(test.meta.index):
            rows = smoothed_shares(tr_channel[nn[j]], hist, ap["share_prior_units"])
            jobs.append((code, om, test.weekly_channel[j], float(point_of[code]), rows, hist,
                         int(test.meta["pack"].iloc[j]), float(test.meta["launch_price"].iloc[j]), int(om.test_idx[j])))
    n = len(jobs)
    idx = M.bootstrap_indices(n, fp["bootstrap_reps"], fp["seed"] + 1)
    price = np.array([j[7] for j in jobs])

    def run_config(policies, stock_ratio=None, salvage=None, ship_scale=1.0, dc_value=None, use_min=True, use_max=True, keep=False):
        outcomes = {p: [] for p in policies}
        extras = []
        for code, om, wk, point, rows, hist, pack, pr, gi in jobs:
            rules = make_rules(pack, params, names, use_min, use_max)
            econ = make_economics(pr, params, names, salvage, ship_scale, dc_value)
            out, extra = run_launch(wk, point, om, rows, hist, rules, econ, params,
                                    ap["stock_ratio"] if stock_ratio is None else stock_ratio, fp["seed"] + gi, policies)
            for p in policies:
                outcomes[p].append(out[p])
            if keep:
                extras.append((rules, econ, extra))
        return outcomes, extras

    if progress:
        progress(f"allocation back-test on {n} launches")
    outcomes, extras = run_config(POLICIES, keep=True)
    level = fp["ci_level"]
    main = aggregate(outcomes, price, idx, level)
    # calibration of the scenario model: value promised by the scenarios vs value realised
    for pol in ("proportional", "newsvendor"):
        promised = sum(expected_value(e[2]["first"][pol], e[2]["scen"], e[2]["stock"], e[1]) for e in extras)
        main[pol]["model_expected_value"] = float(promised)
        main[pol]["realised_over_expected"] = float(main[pol]["value"] / promised)

    # A4 cross-check: MIP vs marginal allocation on every held-out launch
    if progress:
        progress("MIP cross-check")
    t0 = time.time()
    gaps, same_x = [], 0
    for rules, econ, extra in extras:
        scen, q = extra["scen"], extra["stock"]
        x_g = extra["first"]["newsvendor"]
        v_g = expected_value(x_g, scen, q, econ)
        x_m, v_m = mip_allocate(scen, q, rules, econ)
        gaps.append(abs(v_m - v_g) / max(1.0, abs(v_m)))
        same_x += int(np.array_equal(x_g, x_m))
    gaps = np.array(gaps)
    mip_check = {
        "launches": n,
        "scenarios": ap["scenarios"],
        "tolerance": 1e-6,
        "objective_agree": int((gaps <= 1e-6).sum()),
        "max_relative_gap": float(gaps.max()),
        "identical_allocation": same_x,
        "seconds": round(time.time() - t0, 1),
    }

    stock = np.array([e[2]["stock"] for e in extras], dtype=float)
    demand = np.array([o.demand.sum() for o in outcomes["proportional"]], dtype=float)
    context = {
        "launches": n,
        "stock_units": float(stock.sum()),
        "demand_units": float(demand.sum()),
        "launches_stock_below_demand": int((stock < demand).sum()),
        "mean_first_shipment_share": {
            pol: {nm: float(np.mean([e[2]["first"][pol][c] / max(e[2]["stock"], 1) for e in extras])) for c, nm in enumerate(names)}
            for pol in ("proportional", "newsvendor", "oracle")
        },
        "demand_share": {nm: float(sum(o.demand[c] for o in outcomes["proportional"]) / demand.sum()) for c, nm in enumerate(names)},
    }

    if progress:
        progress("rule cost and sensitivity")
    rule_cost = {}
    base_val = main["newsvendor"]["value"]
    for label, um, ux in (("no_rules", False, False), ("minimum_only", True, False), ("max_share_only", False, True)):
        o, _ = run_config(["proportional", "newsvendor"], use_min=um, use_max=ux)
        v = float(sum(x.value for x in o["newsvendor"]))
        rule_cost[label] = {"newsvendor_value": v, "proportional_value": float(sum(x.value for x in o["proportional"]))}
    free = rule_cost["no_rules"]["newsvendor_value"]
    rule_cost["both_rules"] = {"newsvendor_value": base_val, "proportional_value": main["proportional"]["value"]}
    for label in ("minimum_only", "max_share_only", "both_rules"):
        rule_cost[label]["cost_vs_no_rules"] = float((free - rule_cost[label]["newsvendor_value"]) / free)
        rule_cost[label]["cost_per_launch"] = float((free - rule_cost[label]["newsvendor_value"]) / n)

    sens = []
    sp = ap["sensitivity"]
    configs = [("stock_ratio", v, dict(stock_ratio=v)) for v in sp["stock_ratio"]]
    configs += [("salvage", v, dict(salvage=v)) for v in sp["salvage"]]
    configs += [("dc_value", v, dict(dc_value=v)) for v in sp["dc_value"]]
    configs += [("ship_cost_scale", v, dict(ship_scale=v)) for v in sp["ship_cost_scale"]]
    pols = ["proportional", "newsvendor_ship_all", "newsvendor", "newsvendor_two_stage"]
    for pname, val, kw in configs:
        o, _ = run_config(pols, **kw)
        agg = aggregate(o, price, idx, level)
        sens.append({"parameter": pname, "value": val, **{
            pol: {k2: agg[pol][k2] for k2 in ("fill_rate", "stranded_share_of_stock", "value", "value_gain_vs_proportional", "value_gain_ci") if k2 in agg[pol]}
            for pol in pols}})

    per_launch = pd.DataFrame({"code": [j[0] for j in jobs], "stock": stock.astype(int), "pack": [j[6] for j in jobs]})
    for pol in POLICIES:
        per_launch[f"value_{pol}"] = [o.value for o in outcomes[pol]]
        per_launch[f"sales_{pol}"] = [int(o.sales.sum()) for o in outcomes[pol]]
        per_launch[f"left_{pol}"] = [int(o.left_channel.sum() + o.left_dc) for o in outcomes[pol]]
    result = {
        "params": ap,
        "channels": names,
        "context": context,
        "policies": main,
        "mip_check": mip_check,
        "rule_cost": rule_cost,
        "sensitivity": sens,
    }
    return result, per_launch
