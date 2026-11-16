"""Hand-checked cases for launch identification (L1-L5), curves (L6), channels (CH1) and categories (G1)."""
from __future__ import annotations

import pandas as pd
import pytest

from npalloc import launches as L
from npalloc.build_extracts import identify_launches
from npalloc.categories import categorise, category_names
from npalloc.channels import channel_names, country_to_channel
from npalloc.textutil import jaccard, normalise_description, tokens


def lines(rows):
    df = pd.DataFrame(rows, columns=["code", "description", "qty", "date", "price", "country"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_normalise_and_jaccard():
    assert normalise_description("  Set/6 red-spot  mugs. ") == "SET 6 RED SPOT MUGS"
    assert normalise_description(None) == ""
    assert tokens("RED RED MUG") == frozenset({"RED", "MUG"})
    assert jaccard(frozenset("ABC"), frozenset("BCD")) == pytest.approx(0.5)
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_modal_value_tie_goes_to_smallest():
    assert L.modal_value(pd.Series([6, 12, 12, 6, 1])) == 6
    assert L.modal_value(pd.Series(["B", "A"])) == "A"


def test_week_index_boundaries():
    launch = pd.Series(pd.to_datetime(["2010-06-01"] * 4))
    day = pd.Series(pd.to_datetime(["2010-06-01", "2010-06-07", "2010-06-08", "2010-08-23"]))
    assert L.week_index(day, launch).tolist() == [1, 1, 2, 12]


def prod_frame(rows):
    df = pd.DataFrame(rows, columns=["code", "first_day", "last_day", "norm_desc"]).set_index("code")
    df["first_day"] = pd.to_datetime(df["first_day"])
    df["last_day"] = pd.to_datetime(df["last_day"])
    df["base"] = df.index.str[:5]
    return df


def test_flag_window_burn_in_and_horizon():
    prod = prod_frame(
        [
            ("10001", "2010-05-31", "2011-01-01", "A"),   # last day of burn-in -> existing product
            ("10002", "2010-06-01", "2011-01-01", "B"),   # first day after burn-in -> candidate
            ("10003", "2011-09-17", "2011-12-01", "C"),   # 84 days later = 2011-12-10 -> horizon just observed
            ("10004", "2011-09-18", "2011-12-01", "D"),   # one day too late
        ]
    )
    w = L.flag_window(prod, pd.Timestamp("2010-06-01"), pd.Timestamp("2011-12-10"), 12)
    assert w["after_burn_in"].tolist() == [False, True, True, True]
    assert w["full_horizon"].tolist() == [True, True, True, False]


def test_flag_variants_colaunch_window():
    prod = prod_frame(
        [
            ("20001A", "2010-01-05", "2011-01-01", "MUG RED"),
            ("20001B", "2010-07-01", "2011-01-01", "MUG BLUE"),    # new colour of an old product
            ("20002A", "2010-07-01", "2011-01-01", "BAG RED"),
            ("20002B", "2010-07-29", "2011-01-01", "BAG BLUE"),    # 28 days after its sibling -> co-launch
            ("20002C", "2010-07-30", "2011-01-01", "BAG PINK"),    # 29 days -> variant
            ("20003", "2010-07-01", "2011-01-01", "TIN"),
        ]
    )
    assert L.flag_variants(prod, 28).tolist() == [False, True, False, False, True, False]


def test_flag_same_description():
    prod = prod_frame(
        [
            ("30001", "2010-01-05", "2010-06-30", "GINGHAM HEART DECORATION"),
            ("30002", "2010-07-16", "2011-01-01", "GINGHAM HEART DECORATION"),   # re-code of 30001
            ("30003", "2010-07-16", "2011-01-01", "ANOTHER THING"),
            ("30004", "2010-07-16", "2011-01-01", "ANOTHER THING"),              # same day: neither is earlier
            ("30005", "2010-08-01", "2011-01-01", ""),                           # no description: never flagged
            ("30006", "2010-01-01", "2011-01-01", ""),
        ]
    )
    assert L.flag_same_description(prod).tolist() == [False, True, False, False, False, False]


def test_flag_succession():
    prod = prod_frame(
        [
            ("40001", "2010-01-05", "2010-07-20", "LARGE RED RETROSPOT MUG"),      # phased out around the launch
            ("40002", "2010-01-05", "2011-06-01", "LARGE RED RETROSPOT MUG GIFT"), # still selling: no succession
            ("40003", "2010-01-05", "2010-03-01", "LARGE RED RETROSPOT MUG BOXED"),# ended long before
            ("40010", "2010-07-25", "2011-06-01", "LARGE RED RETROSPOT MUG NEW"),  # candidate: 4/5 tokens shared
            ("40011", "2010-07-25", "2011-06-01", "SMALL BLUE BOWL"),              # candidate: nothing similar
        ]
    )
    price = pd.Series({"40001": 2.0, "40002": 2.0, "40003": 2.0, "40010": 2.4, "40011": 2.0})
    out = L.flag_succession(prod, pd.Index(["40010", "40011"]), price, 0.75, 1.25, 56)
    assert out.loc["40010", "succession"] and out.loc["40010", "succession_of"] == "40001"
    assert out.loc["40010", "succession_jaccard"] == pytest.approx(0.8)
    assert not out.loc["40011", "succession"]
    # price more than 25% apart -> not a re-code
    out = L.flag_succession(prod, pd.Index(["40010"]), price.replace(2.4, 2.6), 0.75, 1.25, 56)
    assert not out.loc["40010", "succession"]
    # similarity threshold above 0.8 -> no hit
    out = L.flag_succession(prod, pd.Index(["40010"]), price, 0.81, 1.25, 56)
    assert not out.loc["40010", "succession"]
    # a 4-day gap window excludes the old code that ended 5 days before
    out = L.flag_succession(prod, pd.Index(["40010"]), price, 0.75, 1.25, 4)
    assert not out.loc["40010", "succession"]


LINES = lines(
    [
        ("50001", "OLD MUG", 5, "2010-01-10 09:00", 1.0, "United Kingdom"),
        ("50002", "NEW TIN", 12, "2010-06-10 09:00", 2.0, "United Kingdom"),
        ("50002", "NEW TIN", 12, "2010-06-10 15:00", 2.5, "France"),
        ("50002", "NEW TIN", 6, "2010-06-16 23:00", 2.5, "EIRE"),          # day 6 -> week 1
        ("50002", "NEW TIN", 4, "2010-06-17 00:10", 2.5, "Japan"),         # day 7 -> week 2
        ("50002", "new tin", 12, "2010-09-01 10:00", 2.5, "United Kingdom"),  # day 83 -> week 12
        ("50002", "NEW TIN", 99, "2010-09-02 10:00", 2.5, "United Kingdom"),  # day 84 -> outside the horizon
        ("50003", "LATE BAG", 3, "2011-11-20 10:00", 4.0, "United Kingdom"),  # horizon not observed
        ("50001B", "OLD MUG BLUE", 2, "2010-07-01 10:00", 1.0, "United Kingdom"),  # variant of 50001
        ("50004", "OLD MUG", 2, "2010-07-02 10:00", 1.0, "United Kingdom"),        # same description as 50001
    ]
)


def test_launch_attributes_and_curves(params):
    from npalloc.channels import assign_channels

    df = LINES.copy()
    df["channel"] = assign_channels(df, params)
    launch_day = pd.Series({"50002": pd.Timestamp("2010-06-10")})
    att = L.launch_attributes(df, launch_day, 12)
    assert att.loc["50002", "launch_price"] == pytest.approx(2.5)     # median of 2.0, 2.5, 2.5 in week 1
    assert att.loc["50002", "pack"] == 12                             # modal line quantity in weeks 1-12
    assert att.loc["50002", "units_12w"] == 46 and att.loc["50002", "lines_12w"] == 5
    cur = L.weekly_curves(df, launch_day, 12)
    got = {(r.week, r.channel): r.qty for r in cur.itertuples()}
    assert got == {
        (1, "EIRE"): 6, (1, "France"): 12, (1, "United Kingdom"): 12,
        (2, "Rest of World"): 4, (12, "United Kingdom"): 12,
    }


def test_identify_launches_funnel(params):
    table, rejected, funnel = identify_launches(LINES, params)
    assert table["code"].tolist() == ["50002"]
    assert table.loc[0, "launch_day"] == "2010-06-10"
    assert table.loc[0, "category"] == "storage"
    assert funnel == {
        "products": 5, "first_sale_in_burn_in": 1, "first_sale_after_burn_in": 4, "horizon_not_observed": 1,
        "candidates": 3, "rejected_variant": 1, "rejected_same_description": 1, "rejected_succession": 0, "launches": 1,
    }
    assert dict(zip(rejected["code"], rejected["reason"])) == {"50001B": "variant", "50004": "same_description"}


def test_channel_mapping(params):
    assert channel_names(params) == [
        "United Kingdom", "EIRE", "Germany", "France", "Netherlands", "Other Europe", "Rest of World",
    ]
    assert country_to_channel("United Kingdom", params) == "United Kingdom"
    assert country_to_channel(" EIRE ", params) == "EIRE"
    assert country_to_channel("Denmark", params) == "Other Europe"
    assert country_to_channel("Channel Islands", params) == "Other Europe"
    assert country_to_channel("Australia", params) == "Rest of World"
    assert country_to_channel("Unspecified", params) == "Rest of World"
    listed = [c for g in params["channels"]["groups"] for c in g["countries"]]
    assert len(listed) == len(set(listed))


def test_categories(params):
    assert categorise("SET OF 3 CAKE TINS PANTRY DESIGN", params) == "kitchen"      # CAKE precedes TINS in rule order
    assert categorise("CHRISTMAS CARD ROBIN", params) == "christmas"                 # christmas is tested first
    assert categorise("JUMBO BAG VINTAGE LEAF", params) == "bags"
    assert categorise("T-LIGHT HOLDER HANGING", params) == "lighting"
    assert categorise("HEART OF WICKER SMALL", params) == "decor"
    assert categorise("ZINC HERB GARDEN CONTAINER", params) == "garden"
    assert categorise("SILVER NECKLACE", params) == "jewellery"
    assert categorise("POTTING SHED SOMETHING", params) == "other"                   # POT must match as a whole token
    assert category_names(params)[-1] == "other" and len(category_names(params)) == 11
