"""Regenerate README.md, docs/rules.md and docs/decision_brief.md from results/*.json.

Usage (from the repository root):
    python scripts/make_docs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npalloc import numcheck, report  # noqa: E402
from npalloc.config import load_params  # noqa: E402


def main() -> int:
    params = load_params(ROOT / "config" / "params.json")
    docs = report.render_all(ROOT, params)
    numbers, strings = numcheck.collect_results(ROOT / "results")
    status = 0
    for rel, text in docs.items():
        bad = numcheck.unverified(text, numbers, strings)
        if bad:
            print(f"{rel}: numbers not found in results: {bad}", file=sys.stderr)
            status = 1
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {rel} ({len(text.splitlines())} lines)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
