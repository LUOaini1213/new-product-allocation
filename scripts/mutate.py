"""Mutation testing with hand-written mutants (scripts/mutants.py).

Usage (from the repository root):
    python scripts/mutate.py --label first     # before adding tests for survivors
    python scripts/mutate.py --label final     # after

The repository is copied to .mutation_work/copy (git-ignored). The clean copy
must pass the suite; then the control mutant must be killed; only then are
the mutants applied one at a time. A mutant is killed when pytest fails.
Each run is stored under its label in results/mutation.json; the other
label's record is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mutants import CONTROL, MUTANTS  # noqa: E402

WORK = ROOT / ".mutation_work" / "copy"
COPY = ["src", "tests", "config", "data/derived", "data/hashes.json", "results", "docs", "scripts", "README.md",
        "LICENSE", "pyproject.toml", "requirements.txt", ".gitignore", ".github"]
OUT = ROOT / "results" / "mutation.json"
# the test files most likely to notice a mutant in each module; they run first so that a kill is quick.
# A mutant that passes them is still run against the whole suite before it is called a survivor.
TARGETS = {
    "cleaning.py": ["tests/test_cleaning.py", "tests/test_mutation_survivors.py"],
    "launches.py": ["tests/test_launches.py", "tests/test_mutation_survivors.py"],
    "textutil.py": ["tests/test_launches.py"],
    "channels.py": ["tests/test_launches.py"],
    "categories.py": ["tests/test_launches.py"],
    "forecast.py": ["tests/test_forecast.py", "tests/test_pipeline.py"],
    "bass.py": ["tests/test_forecast.py", "tests/test_pipeline.py", "tests/test_mutation_survivors.py"],
    "metrics.py": ["tests/test_forecast.py", "tests/test_pipeline.py", "tests/test_mutation_survivors.py"],
    "allocation.py": ["tests/test_allocation.py", "tests/test_mutation_survivors.py"],
    "simulate.py": ["tests/test_simulate.py"],
    "pipeline.py": ["tests/test_pipeline.py", "tests/test_mutation_survivors.py"],
    "backtest.py": ["tests/test_pipeline.py", "tests/test_mutation_survivors.py"],
    "numcheck.py": ["tests/test_docs.py"],
}


def fresh_copy() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    for rel in COPY:
        src, dst = ROOT / rel, WORK / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def run_tests(extra: list[str], timeout: int = 1800) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *extra]
    try:
        proc = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return proc.returncode == 0, tail[0]


def apply(mutant) -> str:
    mid, fname, old, new, _ = mutant
    path = WORK / "src" / "npalloc" / fname
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"mutant {mid}: target text occurs {text.count(old)} times in {fname}, expected exactly 1")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    return text


def restore(mutant, original: str) -> None:
    (WORK / "src" / "npalloc" / mutant[1]).write_text(original, encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["first", "final"], required=True)
    ap.add_argument("--only", nargs="*", help="mutant ids to run (debugging; the record is not written)")
    args = ap.parse_args()
    t0 = time.time()
    fresh_copy()
    ok, tail = run_tests([])
    print(f"clean suite: {tail}", flush=True)
    if not ok:
        print("the clean copy does not pass; fix the tests first", file=sys.stderr)
        return 1
    m = re.search(r"(\d+) passed", tail)
    tests_passed = int(m.group(1)) if m else 0

    original = apply(CONTROL)
    ok, tail = run_tests([])
    restore(CONTROL, original)
    print(f"control mutant: {'SURVIVED' if ok else 'killed'} ({tail})", flush=True)
    if ok:
        print("the control mutant survived: the harness or the suite is broken", file=sys.stderr)
        return 1

    todo = [x for x in MUTANTS if not args.only or x[0] in args.only]
    killed, survived = [], []
    for i, mutant in enumerate(todo, 1):
        original = apply(mutant)
        ok, tail = run_tests(TARGETS[mutant[1]])
        if ok:
            ok, tail = run_tests([])
        restore(mutant, original)
        (survived if ok else killed).append({"id": mutant[0], "file": mutant[1], "rule": mutant[4]})
        print(f"[{i}/{len(todo)}] {mutant[0]:32s} {'SURVIVED' if ok else 'killed'}  ({tail})", flush=True)
    ok, tail = run_tests([])
    if not ok:
        print("the copy does not pass after restoring every mutant", file=sys.stderr)
        return 1
    record = {
        "date": date.today().isoformat(),
        "tests_passed": tests_passed,
        "control_killed": True,
        "mutants": len(todo),
        "killed": len(killed),
        "survived": len(survived),
        "survivors": survived,
        "seconds": round(time.time() - t0),
    }
    print(json.dumps({k: v for k, v in record.items() if k != "survivors"}), flush=True)
    for s in survived:
        print("  survivor:", s["id"], s["rule"])
    if args.only:
        return 0
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data["tool"] = "scripts/mutate.py (hand-written mutants in scripts/mutants.py; a mutant is killed when pytest fails)"
    data["control"] = {"id": CONTROL[0], "description": CONTROL[4], "killed": True}
    data[f"{args.label}_run"] = record
    if "final_run" not in data:
        data["final_run"] = record
    data["tests_passed"] = data["final_run"]["tests_passed"]
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
