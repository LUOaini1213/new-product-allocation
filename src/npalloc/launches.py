"""Launch identification (rules L1-L5) and launch curves (rule L6).

A *launch* is a product code whose first demand line falls after the burn-in
period, whose full horizon is observed, and that is not a re-code or a colour
variant of a product that was already on sale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .textutil import jaccard, normalise_description, tokens


def modal_value(values: pd.Series):
    """Most frequent value; ties broken by the smallest value (deterministic)."""
    counts = values.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def product_table(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per product code: first/last demand day, modal description, base code."""
    lines = lines.assign(day=lines["date"].dt.normalize(), norm_desc=lines["description"].map(normalise_description))
    g = lines.groupby("code")
    prod = pd.DataFrame(
        {
            "first_day": g["day"].min(),
            "last_day": g["day"].max(),
            "n_lines": g.size(),
        }
    )
    nd = lines[lines["norm_desc"] != ""]
    prod["norm_desc"] = nd.groupby("code")["norm_desc"].agg(modal_value).reindex(prod.index).fillna("")
    prod["description"] = nd.groupby("code")["description"].agg(modal_value).reindex(prod.index).fillna("")
    prod["base"] = prod.index.str[:5]
    return prod


def week_index(day: pd.Series, launch_day: pd.Series) -> pd.Series:
    """L6: week 1 = launch day + 0..6 days, week 2 = +7..13 days, ..."""
    return (day - launch_day).dt.days // 7 + 1


def flag_window(prod: pd.DataFrame, burn_in_end: pd.Timestamp, data_end_excl: pd.Timestamp, horizon_weeks: int) -> pd.DataFrame:
    """L1 (after burn-in) and L2 (full horizon observed)."""
    out = pd.DataFrame(index=prod.index)
    out["after_burn_in"] = prod["first_day"] >= burn_in_end
    out["full_horizon"] = prod["first_day"] + pd.Timedelta(days=7 * horizon_weeks) <= data_end_excl
    return out


def flag_variants(prod: pd.DataFrame, colaunch_days: int) -> pd.Series:
    """L3: another code with the same five-digit base was first sold more than
    ``colaunch_days`` earlier (a new colour/size of an existing product).

    Siblings whose first sales fall within ``colaunch_days`` of the family's
    first sale are a co-launch and stay eligible.
    """
    earliest_base = prod.groupby("base")["first_day"].transform("min")
    return (prod["first_day"] - earliest_base).dt.days > colaunch_days


def flag_same_description(prod: pd.DataFrame) -> pd.Series:
    """L4: the modal description equals that of a different code sold strictly earlier."""
    named = prod[prod["norm_desc"] != ""]
    earliest_desc = named.groupby("norm_desc")["first_day"].transform("min")
    flag = named["first_day"] > earliest_desc
    return flag.reindex(prod.index).fillna(False).astype(bool)


def flag_succession(
    prod: pd.DataFrame,
    candidates: pd.Index,
    price: pd.Series,
    jaccard_min: float,
    price_ratio_max: float,
    gap_days: int,
) -> pd.DataFrame:
    """L5: a probable re-code of an older code that was phased out around the launch.

    Older code e matches candidate c when e was first sold before c,
    e's last sale is within +-gap_days of c's first sale, the prices differ by
    at most a factor price_ratio_max, and the description token sets have
    Jaccard similarity >= jaccard_min.
    """
    tok = {code: tokens(d) for code, d in prod["norm_desc"].items()}
    first = prod["first_day"]
    last = prod["last_day"]
    logp = np.log(price.reindex(prod.index).astype(float))
    gap = pd.Timedelta(days=gap_days)
    max_log = np.log(price_ratio_max)
    rows = []
    for c in candidates:
        lc = first[c]
        pool = prod.index[(first < lc) & (last >= lc - gap) & (last <= lc + gap)]
        if np.isfinite(logp[c]):
            pool = pool[np.abs(logp[pool].to_numpy() - logp[c]) <= max_log + 1e-12]
        best, best_j = None, 0.0
        for e in pool:
            j = jaccard(tok[c], tok[e])
            if j > best_j or (j == best_j and best is not None and e < best):
                best, best_j = e, j
        hit = best is not None and best_j >= jaccard_min
        rows.append((c, hit, best if hit else "", best_j if hit else 0.0))
    out = pd.DataFrame(rows, columns=["code", "succession", "succession_of", "succession_jaccard"]).set_index("code")
    return out


def launch_attributes(lines: pd.DataFrame, launch_day: pd.Series, horizon_weeks: int) -> pd.DataFrame:
    """Launch price (median unit price in week 1) and case-pack proxy (modal line quantity)."""
    sub = lines[lines["code"].isin(launch_day.index)].copy()
    sub["week"] = week_index(sub["date"].dt.normalize(), sub["code"].map(launch_day))
    sub = sub[(sub["week"] >= 1) & (sub["week"] <= horizon_weeks)]
    w1 = sub[sub["week"] == 1]
    out = pd.DataFrame(index=launch_day.index)
    out["launch_price"] = w1.groupby("code")["price"].median().reindex(out.index)
    out["pack"] = sub.groupby("code")["qty"].agg(modal_value).reindex(out.index)
    out["lines_12w"] = sub.groupby("code").size().reindex(out.index).fillna(0).astype(int)
    out["units_12w"] = sub.groupby("code")["qty"].sum().reindex(out.index).fillna(0).astype(int)
    return out


def weekly_curves(lines: pd.DataFrame, launch_day: pd.Series, horizon_weeks: int) -> pd.DataFrame:
    """Long table code, week (1..H), channel, qty for the given launches (non-zero cells only)."""
    sub = lines[lines["code"].isin(launch_day.index)].copy()
    sub["week"] = week_index(sub["date"].dt.normalize(), sub["code"].map(launch_day))
    sub = sub[(sub["week"] >= 1) & (sub["week"] <= horizon_weeks)]
    cur = sub.groupby(["code", "week", "channel"], as_index=False)["qty"].sum()
    return cur.sort_values(["code", "week", "channel"]).reset_index(drop=True)
