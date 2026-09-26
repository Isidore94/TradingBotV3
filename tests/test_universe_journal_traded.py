"""S15 item 8 - the journal's traded names join the scan universe as a labelled `journal_traded` source.

The trader's wins (SPCX, DRAM) are measured on the same ruler as the scan only if the scan sees them.
Stock trades only, the latest trade's direction picks the side list, most recent first, capped.
Nothing already in the universe moves: a name the screen or the trader's include files put there
keeps its list and its side, and the write floor is judged on the screen alone.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import universe_builder as ub  # noqa: E402

TODAY = date(2026, 9, 26)


def _journal(path: Path, trades) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (trade_id TEXT, symbol TEXT, security_type TEXT, direction TEXT,"
                 " trade_date TEXT, opened_at TEXT)")
    for index, (symbol, kind, direction, day) in enumerate(trades):
        conn.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
                     (f"t{index}", symbol, kind, direction, day, f"{day}T10:00:00-04:00"))
    conn.commit()
    conn.close()
    return path


def test_stock_trades_only_latest_direction_wins(tmp_path):
    path = _journal(tmp_path / "j.sqlite3", [
        ("DRAM", "STK", "SHORT", "2026-06-01"),
        ("DRAM", "STK", "LONG", "2026-09-20"),
        ("spcx", "STK", "LONG", "2026-09-10"),
        ("DRAM260618P00055000", "OPT", "SHORT", "2026-09-21"),
        ("USD.CAD", "CASH", "LONG", "2026-09-21"),
        ("PLTR", "STK", "SHORT", "2026-09-01"),
        ("OLD", "STK", "LONG", "2025-08-26"),  # outside the 365-day lookback
        ("", "STK", "LONG", "2026-09-01"),
    ])
    assert ub.journal_traded_symbols(db_path=path, today=TODAY) == {
        "DRAM": "LONG", "SPCX": "LONG", "PLTR": "SHORT"}


def test_the_cap_keeps_the_most_recently_traded(tmp_path):
    path = _journal(tmp_path / "j.sqlite3", [
        ("AAA", "STK", "LONG", "2026-09-01"),
        ("BBB", "STK", "LONG", "2026-09-03"),
        ("CCC", "STK", "SHORT", "2026-09-02"),
    ])
    assert ub.journal_traded_symbols(db_path=path, today=TODAY, limit=2) == {"BBB": "LONG", "CCC": "SHORT"}
    assert ub.JOURNAL_TRADED_MAX_SYMBOLS == 100 and ub.JOURNAL_TRADED_LOOKBACK_DAYS == 365


def test_no_journal_is_no_names_and_the_journal_is_never_written(tmp_path):
    assert ub.journal_traded_symbols(db_path=tmp_path / "missing.sqlite3", today=TODAY) == {}
    assert not (tmp_path / "missing.sqlite3").exists()
    path = _journal(tmp_path / "j.sqlite3", [("DRAM", "STK", "LONG", "2026-09-20")])
    before = path.stat().st_mtime_ns
    ub.journal_traded_symbols(db_path=path, today=TODAY)
    assert path.stat().st_mtime_ns == before
    empty = tmp_path / "empty.sqlite3"
    sqlite3.connect(empty).close()
    assert ub.journal_traded_symbols(db_path=empty, today=TODAY) == {}


# ---------------------------------------------------------------------------
# build_universe
# ---------------------------------------------------------------------------
def _metrics(symbols, *, long_side=True):
    count = len(symbols)
    return pd.DataFrame({
        "symbol": symbols, "last_price": [50.0] * count, "avg_volume_20d": [5e6] * count,
        "dollar_volume_20d": [2.5e8] * count, "sma_50": [40.0] * count, "sma_100": [40.0] * count,
        "sma_200": [40.0] * count, "above_sma_50": [long_side] * count, "above_sma_100": [long_side] * count,
        "above_sma_200": [long_side] * count, "below_sma_50": [not long_side] * count,
        "below_sma_100": [not long_side] * count, "below_sma_200": [not long_side] * count,
    })


@pytest.fixture()
def home(tmp_path):
    files = {name: tmp_path / f"universe_{name}.txt" for name in ("all", "longs", "shorts")}
    with ExitStack() as stack:
        for attr, value in (("UNIVERSE_ALL_FILE", files["all"]), ("UNIVERSE_LONGS_FILE", files["longs"]),
                            ("UNIVERSE_SHORTS_FILE", files["shorts"]),
                            ("UNIVERSE_METADATA_FILE", tmp_path / "universe_metadata.csv")):
            stack.enter_context(patch.object(ub, attr, value))
        stack.enter_context(patch.object(
            ub, "UNIVERSE_INCLUDE_FILES", {name: tmp_path / f"universe_include_{name}.txt" for name in files}))
        stack.enter_context(patch.object(ub, "_universe_ledger_path", lambda: tmp_path / "ledger.jsonl"))
        stack.enter_context(patch.object(ub, "_snapshot_universe_lists", lambda: ""))
        yield tmp_path, files


def _build(metrics, journal=None, **kwargs):
    history = pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])
    with patch.object(ub, "fetch_all_listed_symbols", return_value=["AAPL"]), \
            patch.object(ub, "fetch_optionable_symbols", return_value=["AAPL"]), \
            patch.object(ub, "fetch_price_history", return_value=history), \
            patch.object(ub, "compute_universe_metrics", return_value=metrics), \
            patch.object(ub, "fetch_market_caps", return_value={}), \
            patch.object(ub, "journal_traded_symbols", return_value=journal or {}):
        return ub.build_universe(write_outputs=True, **kwargs)


def _read(path):
    return path.read_text(encoding="utf-8").split()


def test_existing_universe_names_are_unchanged_and_journal_names_are_added(home):
    root, files = home
    (root / "universe_include_all.txt").write_text("TYPED\n", encoding="utf-8")
    screen = [f"S{index:03d}" for index in range(30)]
    plain = _build(_metrics(screen), force=True)  # a 30-name universe is under the 500 floor
    plain_lists = {name: _read(path) for name, path in files.items()}
    # S000 is a screen long the trader last traded SHORT: it keeps its list and its side.
    result = _build(_metrics(screen), journal={"SPCX": "LONG", "DRAM": "SHORT", "S000": "SHORT", "TYPED": "LONG"},
                    force=True)
    lists = {name: _read(path) for name, path in files.items()}
    for name in files:
        assert set(plain_lists[name]) <= set(lists[name]), name
    assert set(lists["all"]) - set(plain_lists["all"]) == {"SPCX", "DRAM"}
    assert set(lists["longs"]) - set(plain_lists["longs"]) == {"SPCX"}
    assert set(lists["shorts"]) - set(plain_lists["shorts"]) == {"DRAM"}
    assert "S000" not in lists["shorts"] and "TYPED" not in lists["longs"]
    assert result["journal_traded"] == ["DRAM", "SPCX"]
    assert set(plain["all"]) <= set(result["all"])
    ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["stages"][ub.JOURNAL_TRADED_SOURCE] == 2


def test_the_write_floor_is_judged_on_the_screen_alone(home):
    _root, files = home
    files["all"].write_text("\n".join(f"P{index:04d}" for index in range(1487)) + "\n", encoding="utf-8")
    journal = {f"J{index:03d}": "LONG" for index in range(100)}
    with pytest.raises(RuntimeError, match="floor"):
        # 690 screened + 100 journal names would clear the 743 floor; the screen alone does not.
        _build(_metrics([f"S{index:04d}" for index in range(690)]), journal=journal)
    assert len(_read(files["all"])) == 1487


def test_a_journal_read_failure_never_stops_the_rebuild(home):
    _root, files = home
    history = pd.DataFrame(columns=["symbol", "datetime", "close", "volume"])
    with patch.object(ub, "fetch_all_listed_symbols", return_value=["AAPL"]), \
            patch.object(ub, "fetch_optionable_symbols", return_value=["AAPL"]), \
            patch.object(ub, "fetch_price_history", return_value=history), \
            patch.object(ub, "compute_universe_metrics", return_value=_metrics(["AAA", "BBB"])), \
            patch.object(ub, "fetch_market_caps", return_value={}), \
            patch.object(ub, "journal_traded_symbols", side_effect=sqlite3.DatabaseError("locked")):
        result = ub.build_universe(write_outputs=True)
    assert result["all"] == ["AAA", "BBB"] and result["journal_traded"] == []
    assert _read(files["all"]) == ["AAA", "BBB"]
