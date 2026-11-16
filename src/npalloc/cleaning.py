"""Cleaning rules for the raw invoice lines (rules C1-C5 in rules.md).

Every rule is a small pure function so that it can be checked by hand in
tests/test_cleaning.py.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# C1: a sellable product has a five-digit code with an optional one- or two-letter
# variant suffix (e.g. 85123A). Postage, fees, adjustments, samples, gift vouchers
# and test codes (POST, DOT, M, C2, BANK CHARGES, AMAZONFEE, gift_0001_10, TEST001,
# DCGS0058 ...) do not match.
PRODUCT_CODE_RE = re.compile(r"^[0-9]{5}[A-Z]{0,2}$")


def normalise_code(code: object) -> str:
    return str(code).strip().upper()


def is_product_code(code: object) -> bool:
    return bool(PRODUCT_CODE_RE.match(normalise_code(code)))


def is_cancellation(invoice: object) -> bool:
    """C2: invoice numbers starting with 'C' are cancellations (UCI variable notes)."""
    return str(invoice).strip().upper().startswith("C")


def dedupe_sheet_overlap(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """C0: the two workbook sheets overlap in time; keep each overlapping row once.

    Rows of sheet 0 dated on or after the first timestamp of sheet 1 are
    dropped, but only after checking that the overlapping rows are the same
    multiset in both sheets. Returns (frame, number of rows dropped).
    """
    if raw["sheet"].nunique() < 2:
        return raw.copy(), 0
    start1 = raw.loc[raw["sheet"] == 1, "InvoiceDate"].min()
    end0 = raw.loc[raw["sheet"] == 0, "InvoiceDate"].max()
    a = raw[(raw["sheet"] == 0) & (raw["InvoiceDate"] >= start1)]
    b = raw[(raw["sheet"] == 1) & (raw["InvoiceDate"] <= end0)]
    if len(a) == 0:
        return raw.copy(), 0
    cols = [c for c in raw.columns if c != "sheet"]

    def key(df: pd.DataFrame) -> pd.Series:
        return df[cols].astype(str).agg("|".join, axis=1).value_counts().sort_index()

    ka, kb = key(a), key(b)
    if not ka.equals(kb):
        raise ValueError("sheet overlap is not identical; refusing to drop rows")
    return raw.drop(index=a.index).reset_index(drop=True), int(len(a))


def remove_matched_cancellations(
    orders: pd.DataFrame, cancels: pd.DataFrame, window_days: int
) -> tuple[pd.DataFrame, int]:
    """C3: drop each order line that a later cancellation reverses.

    A cancellation (customer, code, -q, t) removes the latest not-yet-removed
    order line (customer, code, q, t0) with t - window_days <= t0 <= t.
    Lines without a customer id cannot be matched and are kept.
    Returns (kept orders, number of order lines removed).
    """
    o = orders.reset_index(drop=True)
    pending: dict[tuple, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
    has_cust = o["customer"].notna()
    for idx, cust, code, qty, ts in zip(
        o.index[has_cust], o.loc[has_cust, "customer"], o.loc[has_cust, "code"],
        o.loc[has_cust, "qty"], o.loc[has_cust, "date"],
    ):
        pending[(cust, code, int(qty))].append((ts, idx))
    for k in pending:
        pending[k].sort()
    removed: set[int] = set()
    window = pd.Timedelta(days=window_days)
    c = cancels[cancels["customer"].notna()].sort_values("date", kind="mergesort")
    for cust, code, qty, ts in zip(c["customer"], c["code"], c["qty"], c["date"]):
        key = (cust, code, int(-qty))
        cands = pending.get(key)
        if not cands:
            continue
        # latest unmatched order on or before the cancellation, within the window
        for j in range(len(cands) - 1, -1, -1):
            t0, idx = cands[j]
            if t0 > ts:
                continue
            if t0 < ts - window:
                break
            if idx not in removed:
                removed.add(idx)
                del cands[j]
                break
    kept = o.drop(index=sorted(removed)).reset_index(drop=True)
    return kept, len(removed)


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "invoice": raw["Invoice"].astype(str).str.strip(),
            "code": raw["StockCode"].map(normalise_code),
            "description": raw["Description"].fillna("").astype(str),
            "qty": pd.to_numeric(raw["Quantity"], errors="coerce").fillna(0).astype(np.int64),
            "date": pd.to_datetime(raw["InvoiceDate"]),
            "price": pd.to_numeric(raw["Price"], errors="coerce").fillna(0.0),
            "customer": pd.to_numeric(raw["Customer ID"], errors="coerce"),
            "country": raw["Country"].fillna("Unspecified").astype(str).str.strip(),
        }
    )
    return out


def clean_lines(raw: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, dict]:
    """Apply C0-C4 and return demand lines (positive, priced, product, not reversed)."""
    stats: dict = {"raw_rows": int(len(raw))}
    raw, n_overlap = dedupe_sheet_overlap(raw)
    stats["overlap_rows_dropped"] = n_overlap
    std = standardise(raw)
    cancel_mask = std["invoice"].map(is_cancellation)
    product_mask = std["code"].map(is_product_code)
    stats["cancellation_rows"] = int(cancel_mask.sum())
    stats["non_product_rows"] = int((~product_mask).sum())
    cancels = std[cancel_mask & product_mask & (std["qty"] < 0)]
    # C4: an order line needs a positive quantity and a positive price
    orders = std[~cancel_mask & product_mask & (std["qty"] > 0) & (std["price"] > 0)]
    stats["dropped_nonpositive_rows"] = int((~cancel_mask & product_mask).sum() - len(orders))
    kept, n_removed = remove_matched_cancellations(orders, cancels, window_days)
    stats["orders_reversed_by_cancellation"] = n_removed
    stats["demand_lines"] = int(len(kept))
    return kept, stats
