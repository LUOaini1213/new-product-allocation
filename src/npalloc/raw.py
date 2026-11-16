"""Read the raw workbook once and cache it as a compressed CSV in data/interim/.

The workbook has two sheets ("Year 2009-2010", "Year 2010-2011"). The cached
file keeps every raw row plus the sheet it came from; all cleaning happens in
:mod:`npalloc.cleaning` so that every rule is unit-tested.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def read_workbook(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        engine="openpyxl",
    )
    frames = []
    for i, (name, df) in enumerate(sheets.items()):
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"sheet {name!r} lacks columns {missing}")
        df = df[RAW_COLUMNS].copy()
        df["sheet"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def cache_workbook(xlsx: Path, out_csv: Path) -> pd.DataFrame:
    df = read_workbook(xlsx)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
    return df


def load_cached(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        parse_dates=["InvoiceDate"],
        compression="gzip",
    )
    return df
