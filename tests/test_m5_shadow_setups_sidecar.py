"""S7 shadow sidecar: the capture pass, its records, its de-duplication and its isolation."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = ROOT_DIR / "tests"
for _path in (SCRIPTS_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import m5_shadow_setups as sidecar  # noqa: E402
from test_s7_shadow_parity_golden import SEEDED_NOW, seeded_tapes  # noqa: E402

LOCAL = ZoneInfo("America/Los_Angeles")
NOW = SEEDED_NOW.replace(tzinfo=LOCAL)
REQUIRED = {"schema", "shadow_only", "event_id", "engine", "symbol", "side", "session", "bar_time",
            "bar_close", "level", "entry", "stop", "risk_per_share", "details", "observed_at"}


def _cache(tapes=None):
    tapes = tapes if tapes is not None else seeded_tapes()
    return {f"{name}|5 D|5 mins": bars for name, bars in tapes.items()}


def _capture(path, now=NOW):
    return sidecar.ShadowSetupsCapture(path=path, tz=LOCAL, clock=lambda _zone: now)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_m5_series_reads_only_the_five_minute_key():
    # The plain symbol key can hold any bar size (the bot setdefaults it), so it is unknown.
    bars = [{"dt": datetime(2026, 8, 21, 6, 30)}]
    cache = {"AAA|5 D|5 mins": bars, "AAA": ["other"], "BBB": bars, "CCC|1 D|1 day": bars, "DDD|5 D|5 mins": []}
    assert sidecar.m5_series(cache) == {"AAA": bars}


def test_a_pass_writes_one_record_per_event_with_what_the_bracket_needs(tmp_path):
    path = tmp_path / "m5_shadow_setups.jsonl"
    written = _capture(path).run_pass(_cache())
    rows = _lines(path)
    assert written == len(rows) > 0
    engines = {row["engine"] for row in rows}
    assert {"pd_level_break_hold", "m5_compression_break", "trendline_break"} <= engines
    for row in rows:
        assert REQUIRED <= set(row)
        assert row["schema"] == sidecar.SCHEMA and row["shadow_only"] is True
        start = datetime.fromisoformat(row["bar_time"])
        assert start.utcoffset() in (timedelta(hours=-4), timedelta(hours=-5))
        assert datetime.fromisoformat(row["bar_close"]) == start + timedelta(minutes=5)
        assert row["side"] in ("long", "short")
        assert row["risk_per_share"] > 0
        if row["side"] == "long":
            assert row["stop"] < row["entry"]
        else:
            assert row["stop"] > row["entry"]
        assert row["event_id"].startswith(f"s7:{row['engine']}:{row['symbol']}:{row['side']}:")
        # Completed bars only: no event bar closes after the pass clock.
        assert datetime.fromisoformat(row["bar_close"]) <= NOW


def test_the_bots_ibbar_objects_give_the_same_records_as_dicts(tmp_path):
    from bounce_bot_lib.legacy import IbBar

    as_ib = {name: [IbBar(**bar) for bar in bars] for name, bars in seeded_tapes().items()}
    _capture(tmp_path / "ib.jsonl").run_pass(_cache(as_ib))
    _capture(tmp_path / "dict.jsonl").run_pass(_cache())
    ib_rows, dict_rows = _lines(tmp_path / "ib.jsonl"), _lines(tmp_path / "dict.jsonl")
    assert ib_rows and ib_rows == dict_rows


def test_passes_never_write_an_event_twice_even_across_restarts(tmp_path):
    path = tmp_path / "m5_shadow_setups.jsonl"
    first = _capture(path)
    count = first.run_pass(_cache())
    assert first.run_pass(_cache()) == 0
    assert _capture(path).run_pass(_cache()) == 0  # a fresh object reads the ids on disk
    assert len(_lines(path)) == count


def test_the_vwap_reclaim_needs_spy_for_its_environment(tmp_path):
    tapes = seeded_tapes()
    with_spy = _capture(tmp_path / "a.jsonl")
    with_spy.run_pass(_cache(tapes))
    without = _capture(tmp_path / "b.jsonl")
    without.run_pass(_cache({name: bars for name, bars in tapes.items() if name != "SPY"}))
    rows = _lines(tmp_path / "b.jsonl")
    assert all(row["engine"] != "vwap_reclaim_after_flush" for row in rows)


def _spy_strong_open():
    """SPY flat at 500 on 08-20, then 08-21 opening +0.8%: the champion's early read is bullish_strong."""
    bars = []
    for day, price in ((datetime(2026, 8, 20, 6, 30), 500.0), (datetime(2026, 8, 21, 6, 30), 504.0)):
        for index in range(78):
            bars.append({"dt": day + timedelta(minutes=5 * index), "open": price, "high": price + 0.1,
                         "low": price - 0.1, "close": price, "volume": 1000.0})
    return bars


def test_a_reclaim_is_recorded_when_spy_says_bullish_strong_at_that_bar(tmp_path):
    tapes = seeded_tapes()
    tapes["SPY"] = _spy_strong_open()
    path = tmp_path / "m5_shadow_setups.jsonl"
    _capture(path).run_pass(_cache(tapes))
    reclaims = [row for row in _lines(path) if row["engine"] == "vwap_reclaim_after_flush"]
    assert reclaims and all(row["side"] == "long" for row in reclaims)
    assert all(row["details"]["environment"] == "bullish_strong" for row in reclaims)
    assert "FLUSH_RECLAIM" in {row["symbol"] for row in reclaims}


def test_the_environment_is_the_champions_auto_read_at_the_bar():
    from bounce_bot_lib.legacy import _auto_market_regime_stats

    spy = seeded_tapes()["SPY"]
    reader = sidecar.spy_environment_reader(spy, now=NOW, tz=LOCAL)
    at = datetime(2026, 8, 21, 11, 0, tzinfo=ZoneInfo("America/New_York"))  # 08:00 local
    today = [sidecar._as_ib(bar) for bar in spy if datetime(2026, 8, 21) <= bar["dt"] <= datetime(2026, 8, 21, 8, 0)]
    prior_close = [bar for bar in spy if bar["dt"] < datetime(2026, 8, 21)][-1]["close"]
    assert reader(at) == _auto_market_regime_stats(today, prior_close)["env_key"]
    # A bar before any SPY history is unknown.
    assert reader(datetime(2026, 8, 19, 10, 0, tzinfo=ZoneInfo("America/New_York"))) is None


def test_submit_runs_on_the_worker_and_never_raises(tmp_path):
    path = tmp_path / "m5_shadow_setups.jsonl"
    capture = _capture(path)

    class Bot:
        latest_bars = _cache()

    assert capture.submit(Bot()) is True
    assert capture.wait_idle(20.0)
    capture.close()
    assert capture.rows_written == len(_lines(path)) > 0


def test_the_process_proxy_cache_is_read_on_the_worker(tmp_path):
    path = tmp_path / "m5_shadow_setups.jsonl"
    capture = _capture(path)

    class Proxy:
        is_process_proxy = True

        @property
        def latest_bars(self):
            return _cache()

    assert capture.submit(Proxy()) is True
    assert capture.wait_idle(20.0)
    capture.close()
    assert len(_lines(path)) > 0


def test_a_failed_write_loses_the_events_never_the_desk(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    capture = _capture(blocked)  # a directory cannot be appended to

    class Bot:
        latest_bars = _cache()

    assert capture.submit(Bot()) is True
    assert capture.wait_idle(20.0)
    capture.close()
    assert capture.rows_written == 0
    assert capture.last_error


def test_the_switch_turns_it_off(tmp_path, monkeypatch):
    monkeypatch.setenv(sidecar.ENABLED_ENV, "0")

    class Bot:
        latest_bars = _cache()

    assert _capture(tmp_path / "x.jsonl").submit(Bot()) is False


def test_nothing_live_reads_the_shadow_engines_or_the_sidecar():
    """Shadow only: the engines and the sidecar are reached from the capture and its one owner."""
    pattern = re.compile(r"shadow_setup_events|M5_SHADOW_SETUPS_FILE|m5_shadow_setups|ShadowSetupEvent"
                         r"|pd_level_break_hold_events|vwap_reclaim_after_flush_events"
                         r"|compression_break_events|trendline_break_events")
    allowed = {"m5_signal_engines.py", "m5_shadow_setups.py", "project_paths.py", "bounce_service.py"}
    hits = sorted(
        str(path.relative_to(SCRIPTS_DIR))
        for path in SCRIPTS_DIR.rglob("*.py")
        if path.name not in allowed and pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    )
    assert hits == []


@pytest.mark.parametrize("name", ["legacy.py", "learning.py"])
def test_the_detector_does_not_import_the_shadow_engines(name):
    text = (SCRIPTS_DIR / "bounce_bot_lib" / name).read_text(encoding="utf-8")
    assert "shadow_setup_events" not in text and "m5_shadow_setups" not in text
