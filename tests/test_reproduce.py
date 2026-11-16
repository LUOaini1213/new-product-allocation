"""Slow reproduction check: re-run the whole pipeline from data/derived and compare with results/.

Not part of the default run (see ``addopts`` in pyproject.toml). Run with:
    python -m pytest -m slow tests/test_reproduce.py
"""
from __future__ import annotations

import json
import math

import pytest

from npalloc import backtest, pipeline


def assert_close(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            if k == "seconds":
                continue
            assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, path


@pytest.mark.slow
def test_pipeline_reproduces_committed_results(root, params):
    data = pipeline.load_launch_data(root, params)
    fc, table, models = pipeline.evaluate_forecasts(data, params)
    alloc, _ = backtest.run(data, models, table, params, root)
    want_fc = json.loads((root / "results" / "forecast.json").read_text(encoding="utf-8"))
    want_al = json.loads((root / "results" / "allocation.json").read_text(encoding="utf-8"))
    assert_close(json.loads(json.dumps(fc, default=float)), want_fc, "forecast")
    assert_close(json.loads(json.dumps(alloc, default=float)), want_al, "allocation")
