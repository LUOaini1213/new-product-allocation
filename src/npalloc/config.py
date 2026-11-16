"""Load the decision-rule parameters from config/params.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = ROOT / "config" / "params.json"


def load_params(path: Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_PARAMS
    return json.loads(p.read_text(encoding="utf-8"))
