"""Sales channels = country groups (rule CH1 in rules.md)."""
from __future__ import annotations

import pandas as pd


def channel_names(params: dict) -> list[str]:
    return [c["name"] for c in params["channels"]["groups"]]


def country_to_channel(country: object, params: dict) -> str:
    """Map a country to its channel; anything not listed goes to the fallback channel."""
    c = str(country).strip()
    for group in params["channels"]["groups"]:
        if c in group["countries"]:
            return group["name"]
    return params["channels"]["fallback"]


def assign_channels(lines: pd.DataFrame, params: dict) -> pd.Series:
    mapping = {c: country_to_channel(c, params) for c in lines["country"].unique()}
    return lines["country"].map(mapping)
