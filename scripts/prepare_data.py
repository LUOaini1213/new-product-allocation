"""Build data/interim/ (ignored) and the committed extracts in data/derived/.

Usage (from the repository root, after scripts/fetch_data.py):
    python -I scripts/prepare_data.py

Steps
1. parse the workbook once into data/interim/raw_rows.csv.gz;
2. apply the cleaning rules (C0-C4) in memory;
3. identify launches and write the small committed extracts to data/derived/
   plus results/data_summary.json and results/launches.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npalloc import build_extracts, raw  # noqa: E402
from npalloc.config import load_params  # noqa: E402


def main() -> int:
    params = load_params(ROOT / "config" / "params.json")
    xlsx = ROOT / "data" / "raw" / "online_retail_II.xlsx"
    cache = ROOT / "data" / "interim" / "raw_rows.csv.gz"
    t0 = time.time()
    if not cache.exists():
        print("parsing workbook (a few minutes) ...", flush=True)
        raw.cache_workbook(xlsx, cache)
        print(f"  cached in {time.time() - t0:.0f}s", flush=True)
    rows = raw.load_cached(cache)
    build_extracts.build_all(rows, params, ROOT)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
