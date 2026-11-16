"""Check that every number printed in a generated document comes from results/*.json.

A number in the text is accepted when some numeric value v in the results
(numeric leaves, plus numbers embedded in result strings) satisfies, at the
number of decimals d shown in the text,
    round(|v|, d) == |shown|          or, for a number followed by % or pp,
    round(100 * |v|, d) == |shown|.
ISO dates and 64-character hashes are compared as whole strings.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FENCE_RE = re.compile(r"```.*?```", re.S)
CODE_RE = re.compile(r"`[^`\n]*`")
URL_RE = re.compile(r"https?://[^\s)>\]]+")
DATE_RE = re.compile(r"(?<![0-9A-Za-z])\d{4}-\d{2}-\d{2}(?![0-9A-Za-z])")
HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")
# a number not glued to a letter, underscore or another number part (so "A1", "k10", "v2" are not numbers)
NUM_RE = re.compile(r"(?<![0-9A-Za-z_.])[-+−]?\d[\d,]*(?:\.\d+)?(?![0-9A-Za-z_])(\s?(?:%|pp\b))?")
STR_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def strip_code(text: str) -> str:
    """Remove fenced blocks, inline code and URLs (commands, file names and links are not claims)."""
    return URL_RE.sub(" ", CODE_RE.sub(" ", FENCE_RE.sub(" ", text)))


def collect_results(results_dir: Path) -> tuple[list[float], set[str]]:
    """All numbers and all strings found anywhere in results/*.json."""
    numbers: list[float] = []
    strings: set[str] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            numbers.append(float(o))
        elif isinstance(o, str):
            strings.add(o)
            numbers.extend(float(m) for m in STR_NUM_RE.findall(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                strings.add(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for path in sorted(Path(results_dir).glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return numbers, strings


def parse_tokens(text: str) -> list[tuple[str, float, int, bool]]:
    """(token, absolute value, decimals shown, is_percent) for every number in the text."""
    out = []
    for m in NUM_RE.finditer(text):
        tok = m.group(0)
        pct = m.group(1) is not None
        core = tok[: len(tok) - len(m.group(1))] if pct else tok
        core = core.replace(",", "").lstrip("+-−")
        decimals = len(core.split(".")[1]) if "." in core else 0
        out.append((tok.strip(), float(core), decimals, pct))
    return out


def matches(value: float, decimals: int, pct: bool, numbers: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) * 1e-6 + 1e-12
    for v in numbers:
        a = abs(v)
        if abs(round(a, decimals) - value) <= tol:
            return True
        if pct and abs(round(100.0 * a, decimals) - value) <= tol:
            return True
    return False


def unverified(text: str, numbers: list[float], strings: set[str]) -> list[str]:
    """Numbers, dates and hashes in ``text`` that cannot be traced to the results."""
    body = strip_code(text)
    bad = []
    blob = "\n".join(strings)
    for d in DATE_RE.findall(body):
        if d not in blob:
            bad.append(d)
    for h in HASH_RE.findall(body):
        if h not in blob:
            bad.append(h)
    body = HASH_RE.sub(" ", DATE_RE.sub(" ", body))
    for tok, value, decimals, pct in parse_tokens(body):
        if not matches(value, decimals, pct, numbers):
            bad.append(tok)
    return bad
