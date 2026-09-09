"""SN5 / SN6 (trader, 2026-09-08): the M5 scanner breathes, and scans the trader's picks first.

Measured on the desk 2026-09-08: `Thread-4 (run_strategy)` held 0.62 of a core
in hour 13 and 71-88% per minute at the close while the GUI thread got
0.10-0.15; 13,031 GUI stalls over 50 ms. The fast lane scanned 258 names A to Z
with the trader's own Focus names mixed among 107 auto-adopted ones.

Two pacing-only edits to `bounce_bot_lib/legacy.py`, a detector file, under the
trader's "Go" of 2026-09-08 for exactly these seams:

* **SN5** - after each symbol's compute in the fast lane and the two main-sweep
  loops, the scanner waits `SYMBOL_BREATH_SECONDS` on the STOP EVENT (never
  `time.sleep`), so shutdown latency is unchanged and the GUI gets the lock.
* **SN6** - the fast lane scans the trader's own Focus names first, then the
  auto-adopted ones (`focus_auto_picks.json` markers), then the sweep. The SET
  is unchanged; every name still scans every cycle.

Nothing detected, scored, stored or alerted changes; the golden fixtures and the
whole BounceBot suite must pass unchanged.
"""

from __future__ import annotations

import inspect
import json
import sys
import threading
import time
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ------------------------------------------------------------ the marker read


def _write_markers(tmp_path: Path, picks: dict) -> Path:
    longs = tmp_path / "focus_longs.txt"
    longs.write_text("", encoding="utf-8")
    (tmp_path / "focus_auto_picks.json").write_text(
        json.dumps({"market_date": "2026-09-08", "picks": picks}), encoding="utf-8"
    )
    return longs


def test_load_auto_pick_symbols_reads_todays_markers_only(tmp_path):
    from focus_picks import load_auto_pick_symbols

    longs = _write_markers(
        tmp_path,
        {
            "AAPL|long": {"session_date": "2026-09-08", "source": "strength_board"},
            "msft|short": {"session_date": "2026-09-08"},
            "TSLA|long": {"session_date": "2026-09-05"},  # yesterday's marker: not trusted
            "NVDA|long": "not-a-marker",
        },
    )
    got = load_auto_pick_symbols(focus_longs_path=longs, today=date(2026, 9, 8))
    assert got == {"AAPL", "MSFT"}


def test_load_auto_pick_symbols_is_empty_when_the_file_is_missing_or_malformed(tmp_path):
    from focus_picks import load_auto_pick_symbols

    longs = tmp_path / "focus_longs.txt"
    longs.write_text("", encoding="utf-8")
    assert load_auto_pick_symbols(focus_longs_path=longs, today=date(2026, 9, 8)) == set()
    (tmp_path / "focus_auto_picks.json").write_text("{not json", encoding="utf-8")
    assert load_auto_pick_symbols(focus_longs_path=longs, today=date(2026, 9, 8)) == set()


def test_the_store_and_the_engine_read_the_same_markers(tmp_path):
    """One reader: the store's `_load_auto_picks` and the engine's view agree."""
    from focus_picks import FocusPickStore, load_auto_pick_symbols

    longs = _write_markers(
        tmp_path,
        {
            "AAPL|long": {"session_date": date.today().isoformat()},
            "TSLA|long": {"session_date": "2000-01-01"},
        },
    )
    (tmp_path / "focus_shorts.txt").write_text("", encoding="utf-8")
    store = FocusPickStore(
        focus_longs_path=longs,
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )
    assert set(store.auto_pick_markers()) == {"AAPL|long"}
    assert load_auto_pick_symbols(focus_longs_path=longs) == {"AAPL"}


# ------------------------------------------------------------------- the bot


@pytest.fixture
def bot():
    """A BounceBot with nothing but the seams under test."""
    from bounce_bot_lib.legacy import BounceBot

    made = BounceBot.__new__(BounceBot)
    made._stop_event = threading.Event()
    made.atr_cache = {}
    made.human_focus_map = {"long": set(), "short": set()}
    made.human_focus_auto_symbols = set()
    made.scanned = []
    made.breaths = 0
    made.is_scanning_enabled = lambda: True
    made.get_scan_symbol_set = lambda: set(made.atr_cache)
    made.request_and_detect_bounce = lambda symbol, **kw: made.scanned.append(symbol)
    for name in (
        "check_orb_break_setups",
        "check_ema8_grind_setups",
        "check_lrsi_cross_setups",
        "check_confluence_setups",
        "check_orb_first_candle_setups",
    ):
        setattr(made, name, lambda symbols=None: None)
    return made


# SN6 -------------------------------------------------------------------------


def test_fast_lane_order_is_trader_picks_then_auto_picks_and_the_set_is_unchanged(bot):
    bot.human_focus_auto_symbols = {"AAPL", "ZZZ", "MMM"}
    ordered = bot._fast_lane_order({"ZZZ", "aapl", "BBB", "MMM", "AAA", "NVDA"})
    assert ordered == ["AAA", "BBB", "NVDA", "AAPL", "MMM", "ZZZ"]
    assert set(ordered) == {"AAA", "BBB", "NVDA", "AAPL", "MMM", "ZZZ"}
    assert len(ordered) == 6


def test_fast_lane_order_treats_every_name_as_the_traders_without_markers(bot):
    """A lost marker file promotes names, never drops or demotes one (R2)."""
    del bot.human_focus_auto_symbols
    assert bot._fast_lane_order({"B", "A", "C"}) == ["A", "B", "C"]


def test_the_fast_lane_scans_trader_picks_first_then_auto_picks(bot):
    bot.atr_cache = {sym: 1.0 for sym in ("AAPL", "BBB", "MMM", "AAA", "NVDA", "ZZZ")}
    bot.human_focus_map = {"long": {"AAPL", "BBB", "MMM"}, "short": {"AAA", "NVDA", "ZZZ"}}
    bot.human_focus_auto_symbols = {"AAPL", "ZZZ", "MMM"}
    processed = bot._scan_human_focus_fast_lane({"ema_8"})
    assert bot.scanned == ["AAA", "BBB", "NVDA", "AAPL", "MMM", "ZZZ"]
    assert processed == {"AAA", "BBB", "NVDA", "AAPL", "MMM", "ZZZ"}


def test_load_human_focus_picks_reads_the_auto_pick_symbols(bot, monkeypatch):
    import bounce_bot_lib.legacy as legacy

    monkeypatch.setattr(legacy, "load_focus_map", lambda: {"long": {"aapl", "bbb"}, "short": set()})
    monkeypatch.setattr(legacy, "load_auto_pick_symbols", lambda: {"aapl"})
    bot.load_human_focus_picks()
    assert bot.human_focus_map == {"long": {"AAPL", "BBB"}, "short": set()}
    assert bot.human_focus_auto_symbols == {"AAPL"}


def test_a_failed_marker_read_leaves_every_name_the_traders(bot, monkeypatch):
    import bounce_bot_lib.legacy as legacy

    def boom():
        raise OSError("sidecar unreadable")

    monkeypatch.setattr(legacy, "load_focus_map", lambda: {"long": {"AAPL"}, "short": set()})
    monkeypatch.setattr(legacy, "load_auto_pick_symbols", boom)
    bot.load_human_focus_picks()
    assert bot.human_focus_map["long"] == {"AAPL"}
    assert bot.human_focus_auto_symbols == set()


# SN5 -------------------------------------------------------------------------


def test_breathe_waits_on_the_stop_event_for_the_declared_breath(bot, monkeypatch):
    import bounce_bot_lib.legacy as legacy

    waits = []
    bot._stop_event = type("E", (), {"wait": lambda self, secs: waits.append(secs) or False})()
    monkeypatch.setattr(legacy.time, "sleep", lambda *_: pytest.fail("SN5 must never time.sleep"))
    bot._breathe()
    assert waits == [legacy.SYMBOL_BREATH_SECONDS]
    assert legacy.SYMBOL_BREATH_SECONDS == 0.02


def test_a_set_stop_event_makes_the_breath_free(bot):
    """Shutdown latency is unchanged: a set event returns at once."""
    bot._stop_event.set()
    started = time.perf_counter()
    for _ in range(200):
        bot._breathe()
    assert time.perf_counter() - started < 0.5


def test_the_fast_lane_breathes_once_per_scanned_symbol(bot):
    bot._breathe = lambda: setattr(bot, "breaths", bot.breaths + 1)
    bot.atr_cache = {"AAA": 1.0, "BBB": 1.0, "CCC": None}
    bot.human_focus_map = {"long": {"AAA", "BBB", "CCC"}, "short": set()}
    bot._scan_human_focus_fast_lane({"ema_8"})
    assert bot.scanned == ["AAA", "BBB"]
    assert bot.breaths == 2  # CCC had no ATR: not scanned, no breath


def test_the_main_sweep_breathes_after_each_symbols_compute():
    """Both sweep loops in `run_strategy` breathe right after the detect call.

    Source-level on purpose: `run_strategy` is the live loop and cannot be
    entered in a test without IB, so the seam is pinned where it is written.
    """
    from bounce_bot_lib.legacy import BounceBot

    source = inspect.getsource(BounceBot.run_strategy)
    head = source.index("# 1) Prioritize strongest/weakest names first.")
    tail = source.index("# Keep EOD outcome tracking alive")
    sweep = source[head:tail]
    assert sweep.count("self.request_and_detect_bounce(") == 2
    assert sweep.count("self._breathe()") == 2
    for loop in sweep.split("# 2) Then scan all remaining symbols")[:2]:
        assert loop.index("self.request_and_detect_bounce(") < loop.index("self._breathe()")


def test_the_scan_cycle_clock_still_never_sleeps():
    """SN5 lives in the bot, not the clock: `ScanCycleClock` measures and decides nothing."""
    import bounce_bot_lib.legacy as legacy

    clock_source = inspect.getsource(legacy.ScanCycleClock)
    for forbidden in ("sleep", "wait(", "start(", "Thread"):
        assert forbidden not in clock_source
