"""Product category from description keywords (rule G1 in rules.md)."""
from __future__ import annotations

from .textutil import tokens


def category_names(params: dict) -> list[str]:
    return [r["name"] for r in params["categories"]["rules"]] + [params["categories"]["fallback"]]


def categorise(description: object, params: dict) -> str:
    """First rule (in file order) with a keyword among the description's tokens."""
    toks = tokens(description)
    for rule in params["categories"]["rules"]:
        if toks.intersection(rule["keywords"]):
            return rule["name"]
    return params["categories"]["fallback"]
