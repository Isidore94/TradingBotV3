"""Packet WS-FC1, the builder's own additions.

Three behaviours the red tests do not reach, each one a refusal the guard makes
on the trader's behalf:

* a refresh in which EVERY row is refused leaves the last verified cache file
  alone rather than replacing it with an empty one (missing data is
  uncertainty, never confirmation);
* a row whose session date cannot be read is dropped and counted, never stored
  and never assumed complete;
* the repair widens its refetch window to REACH the bad session, because a
  fixed ten-day window answers "the provider has no bar for that session" for
  a row dated months ago - which measures the request, not the provider.

Nothing here weakens `tests/test_ws_fc1_daily_bar_cache.py`; it only adds.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
from master_avwap_lib import daily_bar_cache, legacy  # noqa: E402

ET = ZoneInfo("America/New_York")


def _sessions_ending(day: date, count: int) -> list[date]:
    out: list[date] = []
    cursor = day
    while len(out) < count:
        if market_calendar.is_session(cursor):
            out.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(out))


def _latest_session() -> date:
    cursor = date.today()
    while not market_calendar.is_session(cursor):
        cursor -= timedelta(days=1)
    return cursor


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return legacy._set_daily_bar_source(frame, legacy.DAILY_BAR_SOURCE_YAHOO)


def _ordinary_row(day, base: float) -> dict:
    return {
        "datetime": pd.Timestamp(day) if day is not None else None,
        "open": base,
        "high": base + 0.60,
        "low": base - 0.55,
        "close": base + 0.20,
        "volume": 1_200_000.0,
    }


def _partial_row(day: date) -> dict:
    return {
        "datetime": pd.Timestamp(day),
        "open": 71.87,
        "high": 71.805,
        "low": 70.97,
        "close": 71.23,
        "volume": 460_375.0,
    }


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    directory = tmp_path / "machine_cache" / "daily_bars"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(legacy, "DAILY_BARS_CACHE_DIR", directory)
    legacy._DAILY_BAR_FRAME_CACHE.clear()
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
    yield directory
    legacy._DAILY_BAR_FRAME_CACHE.clear()
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()


def test_a_refresh_that_is_entirely_refused_leaves_the_cache_file_alone(
    cache_dir, monkeypatch
):
    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    monkeypatch.setattr(
        daily_bar_cache,
        "market_now",
        lambda: datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11),
    )

    legacy._write_cached_daily_bar_frame("TEST", _frame([_ordinary_row(previous, 70.0)]))
    path = cache_dir / "TEST.csv"
    before = path.read_bytes()

    # Only today's forming bar this time: nothing writable at all.
    legacy._write_cached_daily_bar_frame("TEST", _frame([_partial_row(today)]))

    assert path.read_bytes() == before, (
        "an all-refused refresh replaced the last verified cache file"
    )
    cached = legacy._load_cached_daily_bar_frame("TEST")
    assert not cached.empty, "the in-process frame cache was emptied by a refused refresh"


def test_a_row_whose_session_date_cannot_be_read_is_dropped_and_counted():
    day = _latest_session()
    previous, _today = _sessions_ending(day, 2)
    now = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(hour=16, minute=30)

    frame = _frame([_ordinary_row(previous, 70.0), _ordinary_row(None, 71.0)])
    kept, counts = daily_bar_cache.filter_writable_rows(frame, symbol="TEST", now=now)

    assert int(counts.fetched) == 2
    assert int(counts.kept) == 1
    assert int(counts.forming_dropped) == 0
    assert int(counts.invalid_dropped) == 1
    assert len(kept) == 1


def test_the_repair_widens_its_refetch_window_to_reach_an_old_bad_session(
    cache_dir, monkeypatch, capsys
):
    """PRKS's bad row is dated 2026-07-07. A fixed ten-day window cannot see it."""
    day = _latest_session()
    history = _sessions_ending(day, 90)
    bad_day = history[0]  # ~90 sessions back, far outside a ten-day window

    # The bad row is the LAST row of the file but is dated long ago: this is the
    # MCW / TERN / PRKS shape, a symbol the scan stopped refreshing.
    older = _sessions_ending(bad_day, 20)[:-1]
    rows = [_ordinary_row(one, 45.0 + index * 0.01) for index, one in enumerate(older)]
    rows.append(_partial_row(bad_day))
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")
    frame["source"] = legacy.DAILY_BAR_SOURCE_YAHOO
    frame["volume_unit"] = legacy.DAILY_BAR_UNIT_SHARES
    path = cache_dir / "OLD.csv"
    frame.to_csv(path, index=False)

    windows: list[int] = []

    def _fetch(symbol, days):
        windows.append(int(days))
        return legacy._normalize_daily_bar_frame(
            _frame([_ordinary_row(bad_day, 46.7)])
        )

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _fetch)

    assert daily_bar_cache.main(["repair", "--cache-dir", str(cache_dir)]) == 0

    assert windows, "the repair never refetched the bad session"
    span = (date.today() - bad_day).days
    assert windows[0] >= span, (
        f"the refetch window {windows[0]} cannot reach a session {span} days back"
    )
    report = capsys.readouterr().out
    assert "OLD" in report and bad_day.isoformat() in report
