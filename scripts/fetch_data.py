"""Download the UCI Online Retail II archive into data/raw/ and verify it.

Usage (from the repository root):
    python -I scripts/fetch_data.py

The archive and the extracted workbook stay in data/raw/ (git-ignored). Only
their SHA-256 hashes, stored in data/hashes.json, are committed. Nothing in
data/raw/ is ever executed; the workbook is only parsed as data by
``python -I -m npalloc.prepare``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HASHES = ROOT / "data" / "hashes.json"
ZIP_NAME = "online_retail_ii.zip"
XLSX_NAME = "online_retail_II.xlsx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    meta = json.loads(HASHES.read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / ZIP_NAME
    want_zip = meta["files"][ZIP_NAME]["sha256"]
    if not zip_path.exists() or sha256(zip_path) != want_zip:
        print(f"downloading {meta['source_url']}")
        req = urllib.request.Request(meta["source_url"], headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=300) as resp, zip_path.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    got_zip = sha256(zip_path)
    if got_zip != want_zip:
        print(f"zip hash mismatch: {got_zip} != {want_zip}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if XLSX_NAME not in names:
            print(f"unexpected archive members: {names}", file=sys.stderr)
            return 1
        zf.extract(XLSX_NAME, RAW)
    got_xlsx = sha256(RAW / XLSX_NAME)
    want_xlsx = meta["files"][XLSX_NAME]["sha256"]
    if got_xlsx != want_xlsx:
        print(f"xlsx hash mismatch: {got_xlsx} != {want_xlsx}", file=sys.stderr)
        return 1
    print("ok: archive and workbook match data/hashes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
