"""P18 E: the desk's M5 bar publisher (completed bars only, one writer, never raises) and its readers."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import bars_pack, tilt_pack  # noqa: E402
from ui.services import m5_bar_publisher as pub  # noqa: E402

ET = ZoneInfo("America/New_York")


def _bot_bars(symbol, count, *, first=datetime(2026, 9, 30, 9, 30), step=1.0):
    """The bot's chart dicts: naive market-local (ET here) ``dt``, OHLCV."""
    out = []
    for i in range(count):
        base = 100.0 + step * i
        out.append({"dt": first + timedelta(minutes=5 * i), "open": base, "high": base + 1, "low": base - 0.5,
                    "close": base + 0.5, "volume": 1000.0 * (i + 1)})
    return out


class FakeBot:
    def __init__(self, bars, *, fail=(), delay=0.0):
        self.bars, self.fail, self.calls, self.delay = bars, set(fail), [], delay

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls.append((symbol, max_sessions))
        if self.delay:
            import time

            time.sleep(self.delay)
        if symbol in self.fail:
            raise RuntimeError("proxy said no")
        return list(self.bars.get(symbol, []))


def _publisher(tmp_path, bot, now, symbols=("SPY", "ALL")):
    clock = {"now": now}
    p = pub.M5BarPublisher(lambda: bot, symbols=lambda: list(symbols), directory=tmp_path / "m5_bars",
                           latest=tmp_path / "m5_bars" / "m5_latest.json", market_tz=ET,
                           now=lambda: clock["now"], gap_seconds=0)
    return p, clock


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_completed_bars_only_never_the_forming_one(tmp_path):
    bot = FakeBot({"SPY": _bot_bars("SPY", 7)})  # 09:30 .. 10:00; at 10:03 the 10:00 bar is forming
    p, _clock = _publisher(tmp_path, bot, datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc), symbols=("SPY",))
    result = p.run_once()
    assert result["appended"] == 6 and bot.calls == [("SPY", 2)]
    rows = _lines(tmp_path / "m5_bars" / "2026-09-30.jsonl")
    assert [r["start"][11:16] for r in rows] == ["09:30", "09:35", "09:40", "09:45", "09:50", "09:55"]
    assert all(r["source"] == "desk_publisher" and r["start"].endswith("-04:00") for r in rows)
    latest = json.loads((tmp_path / "m5_bars" / "m5_latest.json").read_text(encoding="utf-8"))
    spy = latest["symbols"]["SPY"]
    assert latest["schema"] == "m5_latest_v1" and spy["bar"]["start"][11:16] == "09:55"
    assert spy["vwap_complete"] is True and spy["session_bars"] == 6
    bars = _bot_bars("SPY", 6)
    want = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in bars) / sum(b["volume"] for b in bars)
    assert abs(spy["session_vwap"] - want) < 1e-9


def test_a_bar_is_written_once_across_passes_and_restarts(tmp_path):
    bot = FakeBot({"SPY": _bot_bars("SPY", 7)})
    p, clock = _publisher(tmp_path, bot, datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc), symbols=("SPY",))
    p.run_once()
    assert p.run_once()["appended"] == 0
    clock["now"] = datetime(2026, 9, 30, 14, 6, tzinfo=timezone.utc)  # the 10:00 bar has closed
    assert p.run_once()["appended"] == 1
    again, _ = _publisher(tmp_path, bot, datetime(2026, 9, 30, 14, 6, tzinfo=timezone.utc), symbols=("SPY",))
    assert again.run_once()["appended"] == 0  # a restarted desk seeds from m5_latest.json
    assert len(_lines(tmp_path / "m5_bars" / "2026-09-30.jsonl")) == 7


def test_a_failing_symbol_or_bot_never_raises_and_keeps_the_last_good_file(tmp_path):
    bot = FakeBot({"ALL": _bot_bars("ALL", 4)}, fail={"SPY"})
    p, _ = _publisher(tmp_path, bot, datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc))
    assert p.run_once()["appended"] == 4
    good = (tmp_path / "m5_bars" / "m5_latest.json").read_text(encoding="utf-8")

    class Broken:
        def m5_chart_bars(self, *_a, **_k):
            return [{"dt": object(), "open": "x"}]

    p._bot = lambda: Broken()
    p.run_once()
    assert "ALL" in json.loads((tmp_path / "m5_bars" / "m5_latest.json").read_text(encoding="utf-8"))["symbols"]

    def boom(*_a, **_k):
        raise OSError("disk gone")

    import ui.services.m5_bar_publisher as module

    original = module.atomic_write_json
    module.atomic_write_json = boom
    try:
        p._bot = lambda: FakeBot({"ALL": _bot_bars("ALL", 5)})
        assert p.run_once() == {"failed": True}
    finally:
        module.atomic_write_json = original
    assert (tmp_path / "m5_bars" / "m5_latest.json").read_text(encoding="utf-8") == good
    p._bot = lambda: None
    assert p.run_once() == {"skipped": "no bot"}


def test_a_slow_proxy_halves_the_batch_and_the_list_is_walked_round_robin(tmp_path):
    names = [f"S{i}" for i in range(120)]
    bot = FakeBot({})
    p, _ = _publisher(tmp_path, bot, datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc), symbols=names)
    p.run_once()
    assert [c[0] for c in bot.calls] == names[:50]
    p.run_once()
    assert [c[0] for c in bot.calls[50:]] == names[50:100]
    p.run_once()
    assert [c[0] for c in bot.calls[100:]] == names[100:] + names[:30]
    slow = FakeBot({}, delay=0.21)
    p._bot = lambda: slow
    p.batch = 12
    p.run_once()
    assert p.batch == 10 and p.last_call_ms > pub.SLOW_CALL_MS  # halved, floored at MIN_BATCH


def test_tick_runs_only_in_session_and_one_pass_at_a_time(tmp_path):
    bot = FakeBot({"SPY": _bot_bars("SPY", 3)})
    p, clock = _publisher(tmp_path, bot, datetime(2026, 9, 30, 2, 0, tzinfo=timezone.utc))
    assert p.tick() is False and bot.calls == []  # 22:00 ET
    clock["now"] = datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc)  # Saturday
    assert p.tick() is False
    clock["now"] = datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc)
    gate = threading.Event()
    p._run_locked = lambda: (gate.wait(5), p._busy.release())
    assert p.tick() is True and p.tick() is False  # the first pass is still running
    gate.set()


def test_bars_pack_prefers_the_publisher_files_and_labels_the_source(tmp_path):
    bot = FakeBot({"ALL": _bot_bars("ALL", 7)})
    p, _ = _publisher(tmp_path, bot, bars_pack.FIXTURE_NOW, symbols=("ALL",))
    p.run_once()
    files = sorted((tmp_path / "m5_bars").glob("????-??-??.jsonl"))
    rows = bars_pack.preferred_bars("ALL", files, lambda: [])
    assert len(rows) == 6 and {r["source"] for r in rows} == {"desk_publisher"}
    src = bars_pack.Sources(bars=lambda sym: bars_pack.preferred_bars(sym, files, lambda: []),
                            market_tz=lambda: ET)
    pack = bars_pack.build("ALL", n=3, now=bars_pack.FIXTURE_NOW, sources=src)
    asof = pack.rows[0]
    assert asof["source"] == "desk_publisher" and asof["text"].endswith("from the desk's M5 publisher")
    assert "3 min ago" in asof["text"]
    # A name the publisher has not written falls back to the spool tee, labelled as such.
    spool = tmp_path / "segment-x.jsonl"
    spool.write_text("\n".join(json.dumps({"dataset": "bar_m5", "row": row}) for row in bars_pack.fixture_rows())
                     + "\n", encoding="utf-8")
    fallback = bars_pack.preferred_bars("ALL", [], lambda: [spool])
    assert fallback and {r["source"] for r in fallback} == {"spool"}
    spool_pack = bars_pack.build("ALL", n=3, now=bars_pack.FIXTURE_NOW, sources=bars_pack.Sources(
        bars=lambda sym: bars_pack.preferred_bars(sym, [], lambda: [spool]), market_tz=lambda: ZoneInfo(
            "America/Los_Angeles")))
    assert spool_pack.rows[0]["text"].endswith("from the research spool tee")


def _journal(path, legs, trades):
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE trade_legs (leg_id INTEGER, trade_id TEXT, side TEXT, role TEXT, timestamp TEXT,"
        " quantity REAL, price REAL);"
        "CREATE TABLE trades (trade_id TEXT, account_number TEXT, symbol TEXT, security_type TEXT, direction TEXT,"
        " status TEXT, opened_at TEXT, closed_at TEXT, quantity_opened REAL, quantity_closed REAL,"
        " average_entry_price REAL, average_exit_price REAL, net_pnl REAL, net_pnl_usd REAL);"
        "CREATE TABLE trade_annotations (trade_id TEXT, planned_stop REAL, planned_risk REAL);")
    conn.executemany("INSERT INTO trade_legs VALUES (?,?,?,?,?,?,?)", legs)
    conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", trades)
    conn.commit()
    conn.close()
    return path


def test_tilt_chase_an_open_into_three_green_bars_after_a_loss(tmp_path):
    legs = [(1, "L1", "SELL", "OPEN", "2026-09-30T09:35:00-04:00", 10, 150.0),
            (2, "L1", "BUY", "CLOSE", "2026-09-30T09:45:00-04:00", 10, 152.0),
            (3, "C1", "BUY", "OPEN", "2026-09-30T10:00:30-04:00", 10, 106.0),
            (4, "C2", "SELL", "OPEN", "2026-09-30T10:01:00-04:00", 10, 106.0)]
    trades = [("L1", "M1", "AMD", "STK", "SHORT", "CLOSED", "2026-09-30T09:35:00-04:00",
               "2026-09-30T09:45:00-04:00", 10, 10, 150.0, 152.0, -20.0, -20.0),
              ("C1", "M1", "ALL", "STK", "LONG", "OPEN", "2026-09-30T10:00:30-04:00", "", 10, 0, 106.0, 0, 0, 0),
              ("C2", "M1", "ALL", "STK", "SHORT", "OPEN", "2026-09-30T10:01:00-04:00", "", 10, 0, 106.0, 0, 0, 0)]
    db = _journal(tmp_path / "j.sqlite3", legs, trades)
    bot = FakeBot({"ALL": _bot_bars("ALL", 7)})  # all green 09:30..10:00
    p, _ = _publisher(tmp_path, bot, datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc), symbols=("ALL",))
    p.run_once()
    now = datetime(2026, 9, 30, 14, 3, tzinfo=timezone.utc)
    bars_for = tilt_pack.publisher_bars(now.astimezone(ET).date(), tmp_path / "m5_bars")
    pack = tilt_pack.build(now=now, journal=db, bars_for=bars_for)
    chase = [row for row in pack.rows if row.get("kind") == "chase"]
    assert [row["id"] for row in chase] == ["tilt:chase:ALL:100030"]  # the long; never the short into green bars
    assert "after 3 green M5 bars in a row (09:45-10:00 ET)" in chase[0]["text"]
    assert "losing close on AMD" in chase[0]["text"] and chase[0]["legs"] == [2, 3]
    assert chase[0] in tilt_pack.observations(pack)
    # No bar files: no chase row, never a guess.
    empty = tilt_pack.build(now=now, journal=db, bars_for=lambda symbol: [])
    assert not [row for row in empty.rows if row.get("kind") == "chase"]
