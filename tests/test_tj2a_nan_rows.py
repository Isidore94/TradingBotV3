"""TJ-2A follow-up (lead, 2026-09-20): a union-index NaN row is not a bar.

THE LIVE FAILURE. A multi-ticker yfinance frame carries ONE index - the union
of every symbol's timestamps - so a symbol with no print at a stamp gets an
all-NaN row there. `_rows_from_download` did `int(row.get("Volume", 0) or 0)`;
NaN is truthy, `int(nan)` raises, and the `except` around the whole symbol
threw away every GOOD bar the name had. On 2026-09-20 16:05 the desk logged
`cannot convert float NaN to integer` for about eighty names in one start and
each of them lost its session tape.

THE RULE. Missing data is uncertainty, never a bar (plan.md sec 5): a row with
any NaN price is SKIPPED, a NaN volume on a priced row is 0, a symbol whose
rows are all NaN is simply absent, and one bad name never costs another its
bars.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

NAN = float("nan")
STAMPS = pd.to_datetime(
    ["2026-09-18 09:30", "2026-09-18 09:35", "2026-09-18 09:40"]
).tz_localize("America/New_York")


def _frame(per_symbol: dict[str, list[tuple[float, float, float, float, float]]]):
    """A yfinance-shaped frame: columns are (field, symbol), one shared index."""
    columns = {}
    for symbol, rows in per_symbol.items():
        for position, field in enumerate(("Open", "High", "Low", "Close", "Volume")):
            columns[(field, symbol)] = [row[position] for row in rows]
    frame = pd.DataFrame(columns, index=STAMPS)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


GOOD = (100.0, 101.0, 99.5, 100.5, 1200.0)
HOLE = (NAN, NAN, NAN, NAN, NAN)


def test_a_union_index_hole_costs_the_symbol_one_row_and_not_its_tape(caplog):
    bars = importlib.import_module("day_review_bars")
    download = _frame({"AAON": [GOOD, HOLE, GOOD], "SPY": [GOOD, GOOD, GOOD]})

    with caplog.at_level(logging.INFO):
        rows = bars._rows_from_download(download, ("AAON", "SPY"))

    assert len(rows["SPY"]) == 3
    # Before the fix AAON was absent altogether: the NaN volume raised and the
    # except dropped the two good bars with it.
    assert [row["close"] for row in rows["AAON"]] == [100.5, 100.5]
    assert all(row["volume"] == 1200 for row in rows["AAON"])
    assert "cannot convert" not in caplog.text
    assert "were unavailable" not in caplog.text


def test_a_priced_row_with_no_volume_is_kept_at_zero_volume():
    bars = importlib.import_module("day_review_bars")
    thin = (100.0, 101.0, 99.5, 100.5, NAN)
    rows = bars._rows_from_download(
        _frame({"AAON": [thin, GOOD, GOOD], "SPY": [GOOD, GOOD, GOOD]}), ("AAON", "SPY")
    )

    assert [row["volume"] for row in rows["AAON"]] == [0, 1200, 1200]


def test_a_row_with_one_missing_price_is_never_stored_as_a_bar():
    """`float(nan)` does not raise, so a half-priced row used to be written to
    the session file as a candle with a NaN high."""
    bars = importlib.import_module("day_review_bars")
    half = (100.0, NAN, 99.5, 100.5, 900.0)
    rows = bars._rows_from_download(
        _frame({"AAON": [GOOD, half, GOOD], "SPY": [GOOD, GOOD, GOOD]}), ("AAON", "SPY")
    )

    assert len(rows["AAON"]) == 2
    for row in rows["AAON"]:
        assert all(row[field] == row[field] for field in ("open", "high", "low", "close"))


def test_a_name_with_no_prints_at_all_is_absent_and_quiet(caplog):
    """An option symbol or a junk name Yahoo cannot serve comes back all-NaN.
    It is simply not there - no traceback per name in the trader's console."""
    bars = importlib.import_module("day_review_bars")
    download = _frame({"DRAM261016C00070000": [HOLE, HOLE, HOLE], "SPY": [GOOD, GOOD, GOOD]})

    with caplog.at_level(logging.INFO):
        rows = bars._rows_from_download(download, ("DRAM261016C00070000", "SPY"))

    assert "DRAM261016C00070000" not in rows
    assert len(rows["SPY"]) == 3
    assert "Traceback" not in caplog.text and "were unavailable" not in caplog.text
