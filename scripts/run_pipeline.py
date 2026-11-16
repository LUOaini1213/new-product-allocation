"""Re-run the forecasting evaluation and the allocation back-test from data/derived/.

Usage (from the repository root):
    python scripts/run_pipeline.py            # writes results/forecast.json, results/allocation.json, results/per_launch.csv
Needs no network and no raw data.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npalloc import pipeline  # noqa: E402
from npalloc.config import load_params  # noqa: E402
from npalloc.io import write_csv, write_json  # noqa: E402


def main() -> int:
    t0 = time.time()
    params = load_params(ROOT / "config" / "params.json")
    data = pipeline.load_launch_data(ROOT, params)
    say = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    fc, table, models = pipeline.evaluate_forecasts(data, params, progress=say)
    write_json(ROOT / "results" / "forecast.json", fc)
    if "--forecast-only" in sys.argv:
        write_csv(ROOT / "results" / "per_launch.csv", table.round(4))
        return 0
    from npalloc import backtest

    alloc, per_launch = backtest.run(data, models, table, params, ROOT, progress=say)
    write_json(ROOT / "results" / "allocation.json", alloc)
    write_csv(ROOT / "results" / "per_launch.csv", table.merge(per_launch, on="code").round(4))
    say("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
