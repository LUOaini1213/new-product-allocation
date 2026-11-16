"""Hand-checked cases for the cleaning rules C0-C4."""
from __future__ import annotations

import pandas as pd
import pytest

from npalloc import cleaning


@pytest.mark.parametrize(
    "code,ok",
    [
        ("85123A", True),
        ("22423", True),
        ("79323LP", True),
        ("15056bl", True),     # lower-case suffix is normalised to upper case
        (" 21232 ", True),
        ("POST", False),
        ("DOT", False),
        ("M", False),
        ("C2", False),
        ("BANK CHARGES", False),
        ("AMAZONFEE", False),
        ("gift_0001_10", False),
        ("TEST001", False),
        ("DCGS0058", False),
        ("1234", False),
        ("123456", False),
        ("85123ABC", False),
    ],
)
def test_product_code_rule(code, ok):
    assert cleaning.is_product_code(code) is ok


def test_cancellation_prefix():
    assert cleaning.is_cancellation("C489449")
    assert cleaning.is_cancellation("c489449")
    assert not cleaning.is_cancellation("489449")
    assert not cleaning.is_cancellation("A506401")


def _orders(rows):
    return pd.DataFrame(rows, columns=["customer", "code", "qty", "date"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_matched_cancellation_removes_latest_order_in_window():
    orders = _orders(
        [
            (1.0, "11111", 12, "2010-01-01"),   # 0 older identical order
            (1.0, "11111", 12, "2010-01-10"),   # 1 latest before the cancel -> removed
            (1.0, "11111", 12, "2010-01-20"),   # 2 after the cancel -> kept
            (1.0, "11111", 6, "2010-01-09"),    # 3 different quantity -> kept
            (2.0, "11111", 12, "2010-01-09"),   # 4 different customer -> kept
        ]
    )
    cancels = _orders([(1.0, "11111", -12, "2010-01-15")])
    kept, n = cleaning.remove_matched_cancellations(orders, cancels, window_days=30)
    assert n == 1
    assert sorted(kept["date"].dt.day.tolist()) == [1, 9, 9, 20]


def test_cancellation_outside_window_is_not_matched():
    orders = _orders([(1.0, "11111", 12, "2010-01-01")])
    cancels = _orders([(1.0, "11111", -12, "2010-02-01")])   # 31 days later
    kept, n = cleaning.remove_matched_cancellations(orders, cancels, window_days=30)
    assert n == 0 and len(kept) == 1
    kept, n = cleaning.remove_matched_cancellations(orders, cancels, window_days=31)
    assert n == 1 and len(kept) == 0


def test_two_cancellations_remove_two_orders_and_no_more():
    orders = _orders([(1.0, "11111", 12, "2010-01-01"), (1.0, "11111", 12, "2010-01-02")])
    cancels = _orders([(1.0, "11111", -12, "2010-01-05"), (1.0, "11111", -12, "2010-01-06"), (1.0, "11111", -12, "2010-01-07")])
    kept, n = cleaning.remove_matched_cancellations(orders, cancels, window_days=30)
    assert n == 2 and len(kept) == 0


def test_orders_without_customer_are_never_matched():
    orders = _orders([(None, "11111", 12, "2010-01-01")])
    cancels = _orders([(None, "11111", -12, "2010-01-02")])
    kept, n = cleaning.remove_matched_cancellations(orders, cancels, window_days=30)
    assert n == 0 and len(kept) == 1


def _raw(rows):
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "sheet"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_sheet_overlap_is_dropped_once():
    a = ("500001", "11111", "MUG", 6, "2010-12-02 10:00", 1.0, 1.0, "France")
    b = ("400001", "22222", "BAG", 2, "2010-11-30 10:00", 2.0, 2.0, "EIRE")
    c = ("500002", "33333", "TIN", 1, "2010-12-05 10:00", 3.0, 3.0, "Germany")
    raw = _raw([b + (0,), a + (0,), a + (1,), c + (1,)])
    out, n = cleaning.dedupe_sheet_overlap(raw)
    assert n == 1
    assert len(out) == 3
    assert out["Invoice"].tolist().count("500001") == 1


def test_sheet_overlap_mismatch_raises():
    a = ("500001", "11111", "MUG", 6, "2010-12-02 10:00", 1.0, 1.0, "France")
    a2 = ("500001", "11111", "MUG", 7, "2010-12-02 10:00", 1.0, 1.0, "France")
    raw = _raw([a + (0,), a2 + (1,)])
    with pytest.raises(ValueError):
        cleaning.dedupe_sheet_overlap(raw)


def test_clean_lines_end_to_end():
    rows = [
        ("500001", "11111", "MUG", 12, "2010-01-02 10:00", 1.0, 7.0, "France", 0),      # kept
        ("C500002", "11111", "MUG", -12, "2010-01-03 10:00", 1.0, 7.0, "France", 0),    # reverses the order above
        ("500003", "22222", "BAG", 5, "2010-01-04 10:00", 2.0, 8.0, "EIRE", 0),         # kept
        ("500004", "POST", "POSTAGE", 1, "2010-01-04 10:00", 18.0, 8.0, "EIRE", 0),     # not a product
        ("500005", "33333", "TIN", -3, "2010-01-05 10:00", 0.0, None, "United Kingdom", 0),  # stock adjustment
        ("500006", "44444", "SAMPLE", 2, "2010-01-05 10:00", 0.0, 9.0, "United Kingdom", 0),  # zero price
    ]
    kept, stats = cleaning.clean_lines(_raw(rows), window_days=30)
    assert kept["code"].tolist() == ["22222"]
    assert stats["orders_reversed_by_cancellation"] == 1
    assert stats["non_product_rows"] == 1
    assert stats["cancellation_rows"] == 1
    assert stats["dropped_nonpositive_rows"] == 2
    assert stats["demand_lines"] == 1
