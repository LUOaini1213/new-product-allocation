"""Description normalisation shared by the launch checks, categories and analogues."""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalise_description(text: object) -> str:
    """Upper-case, replace every run of non-alphanumerics by one space, strip."""
    s = "" if text is None else str(text)
    return _NON_ALNUM.sub(" ", s.upper()).strip()


def tokens(text: object) -> frozenset[str]:
    norm = normalise_description(text)
    return frozenset(norm.split()) if norm else frozenset()


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
