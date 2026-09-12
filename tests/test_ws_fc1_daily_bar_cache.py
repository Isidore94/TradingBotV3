"""Packet WS-FC1 - a forming candle never reaches the daily-bar cache.

The defect these tests exist for (verified read-only against the live cache on
2026-09-12): of 1,988 files under
``%LOCALAPPDATA%\\TradingBotV3\\machine_cache\\daily_bars``, **66 end in a candle
that breaks ``low <= open, close <= high``** - e.g. ``ADC 2026-09-11
O=71.870 H=71.805 L=70.970 C=71.230``. That signature is a FORMING session bar
that the scan wrote into the cache mid-session and that nothing replaced. Every
reader of the cache (the D1 indicators, the band history, the SMA floors, the
session-horizon outcomes) reads it as a real close, against ``plan.md`` sec 5:
*state transitions use completed bars only; a forming bar is a labelled preview.*

The writer seam is ``master_avwap_lib.legacy._write_cached_daily_bar_frame``
(``legacy.py:3234``), reached from ``fetch_daily_bars`` (``legacy.py:18890``)
after ``_merge_daily_bar_frames``. Neither the merge nor the write consults
``scripts/completed_bars.py`` or the candle invariant, so whatever Yahoo returned
at 11:00 is what lands on disk.

--------------------------------------------------------------------------
The contract these red tests define, for the builder
--------------------------------------------------------------------------
A new module ``scripts/master_avwap_lib/daily_bar_cache.py`` (the packet asks for
the filter, the invariant check, the counters and the repair CLI to live there so
the ``legacy.py`` diff stays a few lines - the trader's FC1 prompt is the yes for
the CACHE WRITER seam only):

* ``market_now() -> datetime`` - the ONE clock hook. Aware. Every test below
  freezes it with ``monkeypatch.setattr``; production may spell it
  ``get_market_local_now()``. It is aware on purpose: the close is judged by
  ``astimezone``, never by stripping the offset.
* ``FORMING_DROPPED_COUNTER == "daily_bars_forming_dropped"`` and
  ``INVALID_DROPPED_COUNTER == "daily_bars_invalid_dropped"`` - the run-manifest
  counter names.
* ``PROTECTED_DATA_ROOT`` - module constant, ``C:\\TradingBotData``, pointed at a
  scratch tree by the refusal test (the ``tracker_execution_compare.py:56``
  pattern) so the refusal is proven without going near the live store.
* ``DropCounts`` with integer ``fetched`` / ``kept`` / ``forming_dropped`` /
  ``invalid_dropped``, reconciling exactly.
* ``filter_writable_rows(frame, *, symbol="", now=None) -> (frame, DropCounts)``
  - ``now`` defaults to ``market_now()``.
* ``main(argv) -> int`` for ``python -m master_avwap_lib.daily_bar_cache repair
  [--apply] [--cache-dir PATH]``; dry run by default; prints
  ``project_paths.DATA_DIR`` and the cache directory before anything else;
  refuses a target under ``PROTECTED_DATA_ROOT``; refetches through
  ``legacy.fetch_daily_bars_from_yahoo`` (the desk's pinned daily source);
  writes temp-and-rename.

``legacy._write_cached_daily_bar_frame`` routes its frame through
``filter_writable_rows``, logs one DEBUG line per dropped row naming the symbol
and the session date, and adds the two counts to the active
``diagnostics.ManifestRecorder``. One INFO line per scan names both counters and
their totals.

Nothing here weakens: a test may be added, never relaxed.
"""

from __future__ import annotations

import logging
import os
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
import project_paths  # noqa: E402
from master_avwap_lib import legacy  # noqa: E402

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

# The AEE row named in the packet, and the ADC row measured on 2026-09-12: a
# forming session bar whose `open` sits outside [low, high] because the day's
# range had not yet grown to contain it.
PARTIAL_OPEN = 71.87
PARTIAL_HIGH = 71.805
PARTIAL_LOW = 70.97
PARTIAL_CLOSE = 71.23

# What the same session looks like once it has closed.
COMPLETE_OPEN = 71.87
COMPLETE_HIGH = 72.41
COMPLETE_LOW = 70.97
COMPLETE_CLOSE = 71.98


# ---------------------------------------------------------------------------
# helpers - real session dates, real file shape
# ---------------------------------------------------------------------------
def _sessions_ending(day: date, count: int) -> list[date]:
    """``count`` real exchange sessions, oldest first, ending on ``day``.

    Built from ``market_calendar.is_session`` rather than ``bdate_range`` so a
    holiday never sneaks a non-session date into a frame that claims to be
    sessions - the cache only ever holds sessions.
    """
    out: list[date] = []
    cursor = day
    while len(out) < count:
        if market_calendar.is_session(cursor):
            out.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(out))


def _latest_session(today: date | None = None) -> date:
    cursor = today or date.today()
    while not market_calendar.is_session(cursor):
        cursor -= timedelta(days=1)
    return cursor


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return legacy._set_daily_bar_source(frame, legacy.DAILY_BAR_SOURCE_YAHOO)


def _ordinary_row(day: date, base: float) -> dict:
    return {
        "datetime": pd.Timestamp(day),
        "open": base,
        "high": base + 0.60,
        "low": base - 0.55,
        "close": base + 0.20,
        "volume": 1_400_000.0,
    }


def _partial_row(day: date) -> dict:
    """Today's bar the way Yahoo hands it back while the session is open."""
    return {
        "datetime": pd.Timestamp(day),
        "open": PARTIAL_OPEN,
        "high": PARTIAL_HIGH,
        "low": PARTIAL_LOW,
        "close": PARTIAL_CLOSE,
        "volume": 460_375.0,
    }


def _completed_row(day: date) -> dict:
    return {
        "datetime": pd.Timestamp(day),
        "open": COMPLETE_OPEN,
        "high": COMPLETE_HIGH,
        "low": COMPLETE_LOW,
        "close": COMPLETE_CLOSE,
        "volume": 1_605_900.0,
    }


def _read_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    return frame


def _dates_in(frame: pd.DataFrame) -> list[date]:
    return [pd.Timestamp(value).date() for value in frame["datetime"]]


def _breaks_the_candle_invariant(row) -> bool:
    return not (
        float(row["low"]) <= float(row["open"]) <= float(row["high"])
        and float(row["low"]) <= float(row["close"]) <= float(row["high"])
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")
    frame["source"] = legacy.DAILY_BAR_SOURCE_YAHOO
    frame["volume_unit"] = legacy.DAILY_BAR_UNIT_SHARES
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_bar_cache():
    """The module the packet asks for. Imported here so a missing module is a
    clean red failure on every test rather than a collection error."""
    from master_avwap_lib import daily_bar_cache as module

    return module


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point every spelling of the cache directory at a scratch tree.

    `legacy.DAILY_BARS_CACHE_DIR` is the global `_daily_bar_cache_file` reads
    (`legacy.py:2872`); the other two are patched with `raising=False` so the
    new module may resolve it either way.
    """
    directory = tmp_path / "machine_cache" / "daily_bars"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(legacy, "DAILY_BARS_CACHE_DIR", directory)
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", directory, raising=False)
    try:
        from master_avwap_lib import daily_bar_cache as module

        monkeypatch.setattr(module, "DAILY_BARS_CACHE_DIR", directory, raising=False)
    except ImportError:
        pass
    legacy._DAILY_BAR_FRAME_CACHE.clear()
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
    legacy._DAILY_BAR_LIVE_FAILURE_AT.clear()
    yield directory
    legacy._DAILY_BAR_FRAME_CACHE.clear()
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
    legacy._DAILY_BAR_LIVE_FAILURE_AT.clear()


@pytest.fixture
def freeze_clock(daily_bar_cache, monkeypatch):
    def _freeze(moment: datetime):
        monkeypatch.setattr(daily_bar_cache, "market_now", lambda: moment)
        return moment

    return _freeze


def _age_the_cache_file(path: Path) -> None:
    """Make the refresh actually run: `_daily_bar_cache_is_recent` short-circuits
    anything touched inside 30 minutes (`legacy.py:1892`)."""
    old = (datetime.now() - timedelta(hours=3)).timestamp()
    os.utime(path, (old, old))
    legacy._DAILY_BAR_CACHE_TOUCHED_AT.clear()
    legacy._DAILY_BAR_FRAME_CACHE.clear()


# ===========================================================================
# 1. the writer seam
# ===========================================================================
def test_a_forming_bar_for_the_open_session_is_not_written_to_the_cache(
    cache_dir, freeze_clock
):
    """11:00 ET on a live session: today's bar has not happened yet."""
    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    freeze_clock(datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11))

    frame = _frame([_ordinary_row(previous, 70.0), _partial_row(today)])
    legacy._write_cached_daily_bar_frame("TEST", frame)

    written = _read_cache(cache_dir / "TEST.csv")
    assert _dates_in(written) == [previous], (
        "the forming session bar was written into the daily-bar cache"
    )
    # The in-process frame cache is a reader too, and it must not hold what the
    # file refused.
    cached = legacy._load_cached_daily_bar_frame("TEST")
    assert today not in _dates_in(cached)


def test_a_completed_session_bar_is_written_to_the_cache(cache_dir, freeze_clock):
    """16:30 ET: the same session is over, so the bar is real."""
    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    freeze_clock(
        datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=16, minute=30)
    )

    frame = _frame([_ordinary_row(previous, 70.0), _completed_row(today)])
    legacy._write_cached_daily_bar_frame("TEST", frame)

    written = _read_cache(cache_dir / "TEST.csv")
    assert _dates_in(written) == [previous, today]
    assert float(written.iloc[-1]["close"]) == pytest.approx(COMPLETE_CLOSE)


def test_a_row_whose_open_sits_outside_its_range_is_dropped(cache_dir, freeze_clock):
    """The candle invariant, on a session that closed days ago.

    This is the MCW 2026-06-08 / TERN 2026-05-15 shape: long complete, still
    impossible. Completion alone does not make a candle real.
    """
    day = _latest_session()
    older, stale_bad, newer = _sessions_ending(day, 3)
    freeze_clock(
        datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(hour=16, minute=30)
    )

    frame = _frame(
        [
            _ordinary_row(older, 70.0),
            _partial_row(stale_bad),
            _ordinary_row(newer, 72.0),
        ]
    )
    legacy._write_cached_daily_bar_frame("TEST", frame)

    written = _read_cache(cache_dir / "TEST.csv")
    assert _dates_in(written) == [older, newer], (
        "a completed row that breaks low <= open, close <= high was kept"
    )


def test_a_row_that_is_both_forming_and_invalid_is_counted_once_as_forming(
    cache_dir, daily_bar_cache, freeze_clock
):
    """The real bad row is both. It may only be counted once, or the manifest
    cannot reconcile. Forming is the CAUSE, so forming is the count."""
    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    now = freeze_clock(
        datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11)
    )

    frame = _frame([_ordinary_row(previous, 70.0), _partial_row(today)])
    kept, counts = daily_bar_cache.filter_writable_rows(frame, symbol="TEST", now=now)

    assert int(counts.fetched) == 2
    assert int(counts.kept) == 1
    assert int(counts.forming_dropped) == 1
    assert int(counts.invalid_dropped) == 0
    assert _dates_in(kept) == [previous]


def test_the_kept_and_dropped_counts_reconcile_with_what_was_fetched(
    cache_dir, daily_bar_cache, freeze_clock
):
    """Five rows in: three keepers, one forming, one long-closed impossible."""
    day = _latest_session()
    a, b, c, d, today = _sessions_ending(day, 5)
    now = freeze_clock(
        datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11)
    )

    frame = _frame(
        [
            _ordinary_row(a, 68.0),
            _partial_row(b),  # completed session, impossible candle
            _ordinary_row(c, 70.0),
            _ordinary_row(d, 71.0),
            _partial_row(today),  # the open session
        ]
    )
    kept, counts = daily_bar_cache.filter_writable_rows(frame, symbol="TEST", now=now)

    assert int(counts.fetched) == 5
    assert int(counts.kept) == 3
    assert int(counts.forming_dropped) == 1
    assert int(counts.invalid_dropped) == 1
    assert (
        int(counts.kept) + int(counts.forming_dropped) + int(counts.invalid_dropped)
        == int(counts.fetched)
    )
    assert len(kept) == 3


def test_every_dropped_row_is_logged_at_debug_with_its_symbol_and_date(
    cache_dir, freeze_clock, caplog
):
    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    freeze_clock(datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11))

    frame = _frame([_ordinary_row(previous, 70.0), _partial_row(today)])
    with caplog.at_level(logging.DEBUG):
        legacy._write_cached_daily_bar_frame("TEST", frame)

    said = [
        record.getMessage()
        for record in caplog.records
        if "TEST" in record.getMessage() and today.isoformat() in record.getMessage()
    ]
    assert said, (
        "a dropped daily-bar row must be logged at DEBUG with its symbol and date; "
        f"saw {[r.getMessage() for r in caplog.records]!r}"
    )


def test_the_close_is_judged_in_exchange_time_not_the_traders_wall_clock(
    cache_dir, freeze_clock
):
    """13:05 Pacific IS 16:05 in New York - the bar is complete.

    `scripts/completed_bars.py` says the conversion is `astimezone` and never
    `replace(tzinfo=None)`. An implementation that compares the trader's 13:05
    against a 16:00 close number would call this session open and throw away a
    finished bar - missing data manufactured out of a timezone.
    """
    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    freeze_clock(
        datetime.combine(today, datetime.min.time(), tzinfo=PACIFIC).replace(
            hour=13, minute=5
        )
    )

    frame = _frame([_ordinary_row(previous, 70.0), _completed_row(today)])
    legacy._write_cached_daily_bar_frame("TEST", frame)

    written = _read_cache(cache_dir / "TEST.csv")
    assert _dates_in(written) == [previous, today]


# ===========================================================================
# 2. the real scan path, and the manifest
# ===========================================================================
def test_the_scan_path_drops_the_partial_before_the_close_and_takes_the_real_bar_after(
    cache_dir, freeze_clock, monkeypatch
):
    """`fetch_daily_bars` end to end (`legacy.py:18849`), twice on one day.

    Leg 1 is the defect exactly as it happened: a mid-session refresh whose
    Yahoo frame ends in today's partial bar, on top of a cache that already
    holds yesterday's leftover partial. Neither may reach the file.
    Leg 2 is the same symbol after the close: now the session bar is real.
    """
    day = _latest_session()
    history = _sessions_ending(day, 200)
    previous, today = history[-2], history[-1]

    seeded = [_ordinary_row(d, 65.0 + index * 0.01) for index, d in enumerate(history[:-1])]
    seeded.append(_partial_row(today))  # what the old code left behind
    path = cache_dir / "TEST.csv"
    _write_csv(path, seeded)
    _age_the_cache_file(path)

    def _yahoo_mid_session(symbol, days):
        rows = [_ordinary_row(d, 70.0) for d in history[-3:-1]]
        rows.append(_partial_row(today))
        return legacy._normalize_daily_bar_frame(_frame(rows))

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _yahoo_mid_session)
    freeze_clock(datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11))
    legacy.fetch_daily_bars(None, "TEST", 60)

    mid = _read_cache(path)
    assert today not in _dates_in(mid), (
        "a mid-session refresh left the forming bar in the daily-bar cache"
    )
    assert _dates_in(mid)[-1] == previous
    assert not mid.apply(_breaks_the_candle_invariant, axis=1).any()

    def _yahoo_after_close(symbol, days):
        rows = [_ordinary_row(d, 70.0) for d in history[-3:-1]]
        rows.append(_completed_row(today))
        return legacy._normalize_daily_bar_frame(_frame(rows))

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _yahoo_after_close)
    _age_the_cache_file(path)
    freeze_clock(
        datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=16, minute=30)
    )
    legacy.fetch_daily_bars(None, "TEST", 60)

    after = _read_cache(path)
    assert _dates_in(after)[-1] == today
    assert float(after.iloc[-1]["high"]) == pytest.approx(COMPLETE_HIGH)
    assert not after.apply(_breaks_the_candle_invariant, axis=1).any()


def test_the_scan_manifest_carries_both_drop_counters(
    cache_dir, daily_bar_cache, freeze_clock, tmp_path, monkeypatch
):
    """`run_master` is the manifest wrapper (`runner.py:3058`); the counts have
    to survive the whole way out to the saved JSON."""
    import diagnostics.run_manifest as rm
    from diagnostics.run_manifest import load_recent_manifests
    from master_avwap_lib import runner

    manifest_dir = tmp_path / "run_manifests"
    monkeypatch.setattr(rm, "default_manifest_dir", lambda: manifest_dir)

    day = _latest_session()
    older, stale_bad, previous, today = _sessions_ending(day, 4)
    freeze_clock(datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11))

    def _fake_scan(**kwargs):
        legacy._write_cached_daily_bar_frame(
            "TEST",
            _frame(
                [
                    _ordinary_row(older, 69.0),
                    _partial_row(stale_bad),
                    _ordinary_row(previous, 70.0),
                    _partial_row(today),
                ]
            ),
        )
        return {}

    monkeypatch.setattr(runner, "_run_master_impl", _fake_scan)
    runner.run_master()

    counters = load_recent_manifests(manifest_dir, limit=1)[0]["counters"]
    assert daily_bar_cache.FORMING_DROPPED_COUNTER == "daily_bars_forming_dropped"
    assert daily_bar_cache.INVALID_DROPPED_COUNTER == "daily_bars_invalid_dropped"
    assert counters.get("daily_bars_forming_dropped") == 1
    assert counters.get("daily_bars_invalid_dropped") == 1


def test_the_scan_logs_one_line_with_the_drop_totals(
    cache_dir, freeze_clock, tmp_path, monkeypatch, caplog
):
    """The gate reads `trading_bot.log`, so the totals are one greppable line
    naming both counters - not a number the reader has to guess at."""
    import diagnostics.run_manifest as rm
    from master_avwap_lib import runner

    monkeypatch.setattr(rm, "default_manifest_dir", lambda: tmp_path / "run_manifests")

    day = _latest_session()
    previous, today = _sessions_ending(day, 2)
    freeze_clock(datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=11))

    def _fake_scan(**kwargs):
        legacy._write_cached_daily_bar_frame(
            "TEST", _frame([_ordinary_row(previous, 70.0), _partial_row(today)])
        )
        return {}

    monkeypatch.setattr(runner, "_run_master_impl", _fake_scan)
    with caplog.at_level(logging.INFO):
        runner.run_master()

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.INFO
        and "daily_bars_forming_dropped" in record.getMessage()
        and "daily_bars_invalid_dropped" in record.getMessage()
    ]
    assert len(lines) == 1, (
        "exactly one scan-level line must carry both daily-bar drop totals; "
        f"saw {lines!r}"
    )
    assert "1" in lines[0]


# ===========================================================================
# 3. the repair CLI
# ===========================================================================
def _seed_repairable_file(cache_dir: Path, symbol: str, bad_day: date, history: list[date]):
    rows = [_ordinary_row(d, 65.0 + index * 0.01) for index, d in enumerate(history)]
    rows.append(_partial_row(bad_day))
    path = cache_dir / f"{symbol}.csv"
    _write_csv(path, rows)
    return path


def _refetch_stub(day: date, seen: list[str]):
    def _fetch(symbol, days):
        seen.append(symbol)
        return legacy._normalize_daily_bar_frame(_frame([_completed_row(day)]))

    return _fetch


def test_the_repair_replaces_an_invalid_last_row_with_the_refetched_session_bar(
    cache_dir, daily_bar_cache, monkeypatch, capsys
):
    day = _latest_session()
    history = _sessions_ending(day, 6)
    bad_day, earlier = history[-1], history[:-1]
    path = _seed_repairable_file(cache_dir, "TEST", bad_day, earlier)

    seen: list[str] = []
    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _refetch_stub(bad_day, seen))

    exit_code = daily_bar_cache.main(["repair", "--apply", "--cache-dir", str(cache_dir)])
    assert exit_code == 0

    assert seen == ["TEST"], "the repair must refetch through the pinned Yahoo path"
    written = _read_cache(path)
    assert _dates_in(written)[-1] == bad_day
    assert float(written.iloc[-1]["high"]) == pytest.approx(COMPLETE_HIGH)
    assert not written.apply(_breaks_the_candle_invariant, axis=1).any()

    report = capsys.readouterr().out
    assert "TEST" in report and bad_day.isoformat() in report
    assert str(PARTIAL_HIGH) in report, "the report must show the old row"
    assert str(COMPLETE_HIGH) in report, "the report must show the new row"


def test_the_repair_leaves_an_invalid_interior_row_alone(
    cache_dir, daily_bar_cache, monkeypatch, capsys
):
    """Only the LAST row is the forming-bar signature. An interior oddity is a
    data question this tool does not get to answer."""
    day = _latest_session()
    history = _sessions_ending(day, 6)
    rows = [_ordinary_row(d, 65.0 + index * 0.01) for index, d in enumerate(history)]
    rows[2] = _partial_row(history[2])
    path = cache_dir / "TEST.csv"
    _write_csv(path, rows)
    before = path.read_bytes()

    def _must_not_fetch(symbol, days):  # pragma: no cover - the assertion is the point
        raise AssertionError("the repair refetched a file it had no business touching")

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _must_not_fetch)

    assert daily_bar_cache.main(["repair", "--apply", "--cache-dir", str(cache_dir)]) == 0
    assert path.read_bytes() == before


def test_the_repair_leaves_a_valid_file_untouched(cache_dir, daily_bar_cache, monkeypatch):
    day = _latest_session()
    history = _sessions_ending(day, 6)
    rows = [_ordinary_row(d, 65.0 + index * 0.01) for index, d in enumerate(history)]
    path = cache_dir / "TEST.csv"
    _write_csv(path, rows)
    before = path.read_bytes()

    def _must_not_fetch(symbol, days):  # pragma: no cover
        raise AssertionError("a healthy cache file was refetched")

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _must_not_fetch)

    assert daily_bar_cache.main(["repair", "--apply", "--cache-dir", str(cache_dir)]) == 0
    assert path.read_bytes() == before


def test_the_repair_dry_run_writes_nothing_and_still_reports(
    cache_dir, daily_bar_cache, monkeypatch, capsys
):
    day = _latest_session()
    history = _sessions_ending(day, 6)
    bad_day, earlier = history[-1], history[:-1]
    path = _seed_repairable_file(cache_dir, "TEST", bad_day, earlier)
    before = path.read_bytes()

    seen: list[str] = []
    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _refetch_stub(bad_day, seen))

    assert daily_bar_cache.main(["repair", "--cache-dir", str(cache_dir)]) == 0

    assert path.read_bytes() == before, "the dry run wrote to the cache"
    assert not list(cache_dir.glob("*.tmp"))
    report = capsys.readouterr().out
    assert "TEST" in report and bad_day.isoformat() in report


def test_the_repair_apply_writes_through_a_temp_file_and_renames(
    cache_dir, daily_bar_cache, monkeypatch
):
    """A half-written cache file is the corruption this tool exists to remove."""
    day = _latest_session()
    history = _sessions_ending(day, 6)
    bad_day, earlier = history[-1], history[:-1]
    path = _seed_repairable_file(cache_dir, "TEST", bad_day, earlier)

    seen: list[str] = []
    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _refetch_stub(bad_day, seen))

    renames: list[tuple[str, str]] = []
    real_replace = os.replace

    def _spy(src, dst, *args, **kwargs):
        renames.append((str(src), str(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _spy)

    assert daily_bar_cache.main(["repair", "--apply", "--cache-dir", str(cache_dir)]) == 0

    landed = [pair for pair in renames if Path(pair[1]) == path]
    assert landed, f"the repaired file was not renamed into place; saw {renames!r}"
    source = Path(landed[-1][0])
    assert source != path
    assert source.parent == path.parent, "the temp file must sit beside its target"
    assert not list(cache_dir.glob("*.tmp"))


def test_the_repair_refuses_a_cache_directory_under_the_protected_data_root(
    cache_dir, daily_bar_cache, monkeypatch, tmp_path
):
    """`PROTECTED_DATA_ROOT` is pointed at the scratch tree so the refusal is
    proven without ever addressing `C:\\TradingBotData` (the 2026-09-05 rule)."""
    day = _latest_session()
    history = _sessions_ending(day, 6)
    bad_day, earlier = history[-1], history[:-1]
    path = _seed_repairable_file(cache_dir, "TEST", bad_day, earlier)
    before = path.read_bytes()

    def _must_not_fetch(symbol, days):  # pragma: no cover
        raise AssertionError("the repair ran against a protected target")

    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _must_not_fetch)
    monkeypatch.setattr(daily_bar_cache, "PROTECTED_DATA_ROOT", tmp_path)

    assert daily_bar_cache.main(["repair", "--apply", "--cache-dir", str(cache_dir)]) != 0
    assert path.read_bytes() == before


def test_the_repair_prints_the_data_dir_and_the_cache_directory_first(
    cache_dir, daily_bar_cache, monkeypatch, capsys
):
    day = _latest_session()
    history = _sessions_ending(day, 6)
    bad_day, earlier = history[-1], history[:-1]
    _seed_repairable_file(cache_dir, "TEST", bad_day, earlier)

    seen: list[str] = []
    monkeypatch.setattr(legacy, "fetch_daily_bars_from_yahoo", _refetch_stub(bad_day, seen))

    daily_bar_cache.main(["repair", "--cache-dir", str(cache_dir)])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert str(project_paths.DATA_DIR) in lines[0]
    assert str(cache_dir) in lines[1]
