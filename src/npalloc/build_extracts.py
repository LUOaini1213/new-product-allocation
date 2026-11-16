"""Raw rows -> cleaned demand lines -> launches -> committed extracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import cleaning, launches as L
from .categories import categorise
from .channels import assign_channels, channel_names
from .io import read_json, write_csv, write_json


def identify_launches(lines: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply rules L1-L5. Returns (launch table, rejected candidates, funnel counts)."""
    d = params["data"]
    horizon = d["horizon_weeks"]
    prod = L.product_table(lines)
    win = L.flag_window(prod, pd.Timestamp(d["burn_in_end"]), pd.Timestamp(d["data_end_exclusive"]), horizon)
    cand = win["after_burn_in"] & win["full_horizon"]
    variant = L.flag_variants(prod, d["colaunch_days"])
    same = L.flag_same_description(prod)
    price_all = lines.groupby("code")["price"].median()
    open_codes = prod.index[cand & ~variant & ~same]
    succ = L.flag_succession(
        prod, open_codes, price_all,
        d["succession_jaccard_min"], d["succession_price_ratio_max"], d["succession_gap_days"],
    )
    succ_flag = pd.Series(prod.index.isin(succ.index[succ["succession"].astype(bool)]), index=prod.index)
    genuine = cand & ~variant & ~same & ~succ_flag
    funnel = {
        "products": int(len(prod)),
        "first_sale_in_burn_in": int((~win["after_burn_in"]).sum()),
        "first_sale_after_burn_in": int(win["after_burn_in"].sum()),
        "horizon_not_observed": int((win["after_burn_in"] & ~win["full_horizon"]).sum()),
        "candidates": int(cand.sum()),
        "rejected_variant": int((cand & variant).sum()),
        "rejected_same_description": int((cand & ~variant & same).sum()),
        "rejected_succession": int((cand & ~variant & ~same & succ_flag).sum()),
        "launches": int(genuine.sum()),
    }
    launch_day = prod.loc[genuine, "first_day"]
    att = L.launch_attributes(lines, launch_day, horizon)
    table = pd.DataFrame(
        {
            "code": launch_day.index,
            "launch_day": launch_day.dt.strftime("%Y-%m-%d").to_numpy(),
            "description": prod.loc[genuine, "description"].to_numpy(),
            "norm_desc": prod.loc[genuine, "norm_desc"].to_numpy(),
            "launch_price": att["launch_price"].round(4).to_numpy(),
            "pack": att["pack"].astype(int).to_numpy(),
        }
    )
    table["category"] = [categorise(x, params) for x in table["norm_desc"]]
    table = table.sort_values(["launch_day", "code"]).reset_index(drop=True)
    rej = prod[cand & ~genuine]
    reason = pd.Series("succession", index=rej.index)
    reason[same.reindex(rej.index)] = "same_description"
    reason[variant.reindex(rej.index)] = "variant"
    rejected = pd.DataFrame(
        {
            "code": rej.index,
            "first_day": rej["first_day"].dt.strftime("%Y-%m-%d").to_numpy(),
            "norm_desc": rej["norm_desc"].to_numpy(),
            "reason": reason.to_numpy(),
        }
    ).sort_values("code").reset_index(drop=True)
    return table, rejected, funnel


def build_all(rows: pd.DataFrame, params: dict, root: Path) -> None:
    d = params["data"]
    lines, stats = cleaning.clean_lines(rows, d["cancel_window_days"])
    lines["channel"] = assign_channels(lines, params)
    names = channel_names(params)
    table, rejected, funnel = identify_launches(lines, params)
    launch_day = pd.to_datetime(table.set_index("code")["launch_day"])
    weekly = L.weekly_curves(lines, launch_day, d["horizon_weeks"])
    lines["month"] = lines["date"].dt.strftime("%Y-%m")
    chan_month = lines.groupby(["month", "channel"], as_index=False)["qty"].sum()
    chan_month = chan_month.sort_values(["month", "channel"]).reset_index(drop=True)
    derived = root / "data" / "derived"
    write_csv(derived / "launches.csv", table)
    write_csv(derived / "launch_weekly.csv", weekly)
    write_csv(derived / "channel_month.csv", chan_month)
    write_csv(derived / "candidates_rejected.csv", rejected)

    units = lines.groupby("channel")["qty"].sum().reindex(names).fillna(0)
    revenue = (lines["qty"] * lines["price"]).groupby(lines["channel"]).sum().reindex(names).fillna(0)
    hashes = read_json(root / "data" / "hashes.json")
    summary = {
        "source": {k: hashes[k] for k in ("dataset", "landing_page", "source_url", "retrieved", "licence", "citation")},
        "sha256": {k: v["sha256"] for k, v in hashes["files"].items()},
        "cleaning": stats,
        "first_line": lines["date"].min().strftime("%Y-%m-%d"),
        "last_line": lines["date"].max().strftime("%Y-%m-%d"),
        "countries": int(lines["country"].nunique()),
        "channels": len(names),
        "channel_unit_share": {n: float(units[n] / units.sum()) for n in names},
        "channel_revenue_share": {n: float(revenue[n] / revenue.sum()) for n in names},
    }
    write_json(root / "results" / "data_summary.json", summary)
    by_month = table["launch_day"].str[:7].value_counts().sort_index()
    by_cat = table["category"].value_counts().sort_index()
    total = weekly.groupby("code")["qty"].sum().reindex(table["code"]).to_numpy()
    launches_json = {
        "params": d,
        "funnel": funnel,
        "launches_by_month": {k: int(v) for k, v in by_month.items()},
        "launches_by_category": {k: int(v) for k, v in by_cat.items()},
        "units_12w": {
            "p10": float(pd.Series(total).quantile(0.1)),
            "median": float(pd.Series(total).median()),
            "mean": float(total.mean()),
            "p90": float(pd.Series(total).quantile(0.9)),
            "max": int(total.max()),
        },
        "median_launch_price": float(table["launch_price"].median()),
    }
    write_json(root / "results" / "launches.json", launches_json)
