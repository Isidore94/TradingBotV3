"""Packet WS-10A - last scan, latest input bar and shown report are three clocks.

Written by the TESTER before any fix exists, on ``claude/ws-10a-scan-freshness``
off ``origin/claude/wishlist-sweep-2026-09-12`` at ``e01aede4`` (2026-09-12).
Every test in this file is RED on that commit; the failures are recorded in the
commit message.

The trader's report (WISHLIST 10A): *"the strongest Master AVWAP updates seem to
arrive in the final hour or at EOD, even when the app says it updated earlier"*,
and *"a recent file timestamp proves neither fresh input bars nor good
discovery"*. This packet is half (1) of the split the brief asks for - truthful
freshness and publication. Half (2), discovery coverage, is what the replay CLI
measures rather than fixes.

What exists on ``e01aede4``
---------------------------
* ``runner.run_master`` (``runner.py:3058``) already wraps the scan in a
  ``diagnostics.ManifestRecorder`` - phases, provider counters and WS-FC1's two
  daily-bar drop counters. That manifest lands in the machine-local diagnostics
  folder, is keyed by run id, and says nothing about input-bar freshness.
* ``run_result`` is an IN-MEMORY dict (``runner.py:2314``, returned ``:3011``).
  **There is no scan manifest file in the shared home today**, so nothing on the
  desk can separate "the scan ran at 12:31" from "its newest input bar was
  Thursday's" from "the report you are looking at was written at 12:31".
* The Setups panel shows ``Last run: <stamp>`` (``master_avwap_panel.py:1369``,
  written only by ``_on_scan_finished``) and ``Setups as of <date>``
  (``_apply_data_as_of``, ``:1216``). Neither is the scan's own record: after a
  FAILED scan the panel keeps showing the old report with no "stale" label at
  all, which is precisely the complaint.

The contract these red tests define, for the builder
----------------------------------------------------
**1. ``scripts/master_avwap_lib/scan_manifest.py``** - a new module (so the
``legacy.py`` ask-first diff stays at zero; this packet carries no yes for it).

* ``market_now() -> datetime`` - the ONE clock hook, AWARE, market-local. Every
  test here freezes it. Production may spell it ``get_market_local_now()``.
* ``read_manifest(path=None) -> dict | None`` - ``None`` when there is no file
  or it cannot be parsed. Missing data is uncertainty, never a guess.
* ``freshness_line(manifest, *, report_mtime=None) -> str`` - the one-line strip
  text. **It renders each timestamp in the offset the stamp itself carries and
  never re-converts it** (the repo's "attach, never strip" rule), so a manifest
  written at 12:31 New York prints 12:31 on a Pacific desk.
* ``scan_reports_dir() -> Path`` - ``project_paths.get_diagnostics_dir() /
  "scan_reports"``. The desk keeps no dated report copies today, so the packet's
  parenthesis applies: the smallest dated copy per scan is ADDED here.
* ``REPORT_COPY_RETENTION_SESSIONS == 30`` - the packet says "a 30-day cap"; it
  is counted as the 30 most recent distinct dates that have copies, so a long
  weekend cannot silently shorten the window. (Recorded as a deliberate reading
  of the packet, not a premise it stated.)
* ``record_scan(...)`` - the writer ``runner.run_master`` calls on BOTH paths.
  No test calls it directly: every manifest below is produced by driving
  ``runner.run_master``.

``project_paths`` gains two constants, in the shared home beside the other
``master_avwap_*`` runtime json:

* ``MASTER_AVWAP_SCAN_MANIFEST_FILE`` -> ``master_avwap_scan_manifest.json``
* ``MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE`` ->
  ``master_avwap_scan_manifest_history.jsonl``

The manifest payload (temp-and-rename):

.. code-block:: text

    run_id                     str, the ManifestRecorder's run id, so the two
                               manifests join
    status                     "ok" | "partial" | "failed"
    started_at / finished_at   aware market-local ISO-8601, offset ALWAYS present
    universe_size              int - symbols the scan set out to evaluate
                               (`run_result["universe_size"]`, a key the runner
                               adds; the scan already knows it)
    symbols_fetched            int - len(run_result["daily_frames_by_symbol"])
    daily_bar_source_counts    {source: symbol count}, read through
                               `legacy._get_daily_bar_source(frame)`
    latest_input_bar_session   "YYYY-MM-DD" - the newest COMPLETED daily bar any
                               symbol carried, or None
    preview_bar_used           bool - some symbol's newest row is dated a session
                               that was not complete at `finished_at`
    daily_bars_forming_dropped / daily_bars_invalid_dropped
                               ints, from WS-FC1's `daily_bar_cache.run_totals()`
    outputs                    [{"name", "path", "rows"}] - `priority_setups`
                               (rows = len(run_result["priority_rows"])),
                               `theta_puts` (rows = put rows + PCS rows),
                               `d1_watchlist` (rows = d1_watchlist_symbol_count)
    error                      present on the failed path only

``status`` is ``partial`` when the scan RETURNED but fetched fewer symbols than
its universe, ``failed`` when it raised, ``ok`` only when it fetched its whole
universe. A failed scan writes the manifest and **touches no output file** - the
last good report keeps its bytes and its mtime and is labelled stale by the
strip instead.

Every scan appends exactly ONE line to the history JSONL (append-only; an
earlier line is never rewritten) and writes one dated copy under
``scan_reports_dir()`` named ``<YYYY-MM-DD>_<HHMM>.json`` carrying ``run_id``,
``status``, ``finished_at``, ``latest_input_bar_session``, ``preview_bar_used``
and ``rows`` of ``{"symbol", "side", "bucket"}`` (``bucket`` is the row's
``priority_bucket``).

**2. The strip.** Exactly these strings - the packet quotes two of them:

.. code-block:: text

    Scan ok 12:31 · inputs through Thu 09-10 (D1 complete) · shown: 12:31 report
    Scan ok 11:05 · inputs through Wed 11-25 · inputs: today preview · shown: 11:05 report
    Scan FAILED 13:02 · showing 12:31 report (stale)
    Scan partial 12:31 (940 of 1097 symbols) · inputs through Thu 09-10 (D1 complete) · shown: 12:31 report
    Scan: not recorded yet · no report

The separator is ``·`` (U+00B7) with a space either side, clocks are ``%H:%M``
and the input session is ``%a %m-%d``. The Setups panel puts the line in its
status row on the EXISTING ``refresh_from_reports`` path, and the same line
reaches the System Health strip as a check whose id is
``master_scan_freshness``. The manifest is read on that refresh path and NEVER
on paint.

**3. ``scripts/master_avwap_lib/scan_replay.py``** -
``python -m master_avwap_lib.scan_replay --symbol NVDA --session 2026-09-10``,
run from ``scripts/``, read-only, ``main(argv=None) -> int``. It prints
``project_paths.DATA_DIR`` first (the 2026-09-05 scratch-script rule), then one
line per checkpoint in this order, from the dated copies of that session:

* ``CHECKPOINTS == ("open", "midday", "final hour", "close")``
* market-local windows on the session, which partition the whole day:
  open ``< 11:00``, midday ``[11:00, 15:00)``, final hour ``[15:00, 16:00)``,
  close ``>= 16:00``. The LAST snapshot inside a window is the one reported -
  what stood at the end of that checkpoint.
* a window with no copy prints ``no recorded snapshot`` and NOTHING else: a
  reconstruction is what the brief forbids.
* the summary names first eligibility (the first checkpoint whose report carried
  the name in ANY bucket) and first publication (the first checkpoint that
  carried it in ``favorite_setup``), and the delay in minutes between them.

Nothing here weakens: a test may be added, never relaxed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
from master_avwap_lib import legacy  # noqa: E402

ET = ZoneInfo("America/New_York")

#: The fixture session. 2026-09-11 is a Friday; the newest COMPLETED session at
#: any time that day is Thursday 2026-09-10. (The packet's example line says
#: "Thu 09-11"; 2026-09-11 is a Friday, so the label below is the corrected one.)
SCAN_DAY = date(2026, 9, 11)
LAST_COMPLETE_SESSION = date(2026, 9, 10)
PRIOR_SESSION = date(2026, 9, 9)

#: Two clocks inside one scan, so `started_at` and `finished_at` cannot be the
#: same number by accident.
STARTED_AT = datetime(2026, 9, 11, 12, 5, 41, tzinfo=ET)
FINISHED_AT = datetime(2026, 9, 11, 12, 31, 9, tzinfo=ET)

#: The early-close fixture: the Friday after Thanksgiving 2026 closes at 13:00
#: ET. `market_calendar` deliberately does not model early closes and judges
#: every session against 16:00 (WS-FC1's module docstring), which is
#: conservative in the only direction that matters.
EARLY_CLOSE_DAY = date(2026, 11, 27)
EARLY_CLOSE_PRIOR_SESSION = date(2026, 11, 25)


# ---------------------------------------------------------------------------
# helpers - the frame shape the scan really carries
# ---------------------------------------------------------------------------
def _row(day: date, base: float) -> dict:
    return {
        "datetime": pd.Timestamp(day),
        "open": base,
        "high": base + 0.60,
        "low": base - 0.55,
        "close": base + 0.20,
        "volume": 1_400_000.0,
    }


def _forming_row(day: date) -> dict:
    """Today's bar the way Yahoo hands it back while the session is open."""
    return {
        "datetime": pd.Timestamp(day),
        "open": 71.87,
        "high": 71.805,
        "low": 70.97,
        "close": 71.23,
        "volume": 460_375.0,
    }


def _frame(rows: list[dict], source: str) -> pd.DataFrame:
    """`runner.py:879-886`: one DataFrame per symbol with a `datetime` column
    and the source declared as a frame attribute."""
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return legacy._set_daily_bar_source(frame, source)


def _priority_row(symbol: str, side: str, bucket: str) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "priority_bucket": bucket,
        "priority_score": 71.2,
        "setup_family": "avwap_reclaim",
    }


def _run_result(
    *,
    frames: dict[str, pd.DataFrame] | None = None,
    universe_size: int | None = None,
    priority_rows: list[dict] | None = None,
) -> dict:
    """A `run_result` shaped like `runner.py:2314`'s, carrying only the keys the
    manifest reads. `universe_size` and `priority_rows` are the two the runner
    has to add; everything else is already there today."""
    if frames is None:
        frames = {
            "NVDA": _frame(
                [_row(PRIOR_SESSION, 180.0), _row(LAST_COMPLETE_SESSION, 182.0)],
                legacy.DAILY_BAR_SOURCE_YAHOO,
            ),
            "AMD": _frame(
                [_row(PRIOR_SESSION, 150.0), _row(LAST_COMPLETE_SESSION, 151.0)],
                legacy.DAILY_BAR_SOURCE_YAHOO,
            ),
            "MU": _frame(
                [_row(PRIOR_SESSION, 98.0), _row(LAST_COMPLETE_SESSION, 99.0)],
                legacy.DAILY_BAR_SOURCE_CACHE,
            ),
        }
    if priority_rows is None:
        priority_rows = [
            _priority_row("NVDA", "LONG", "favorite_setup"),
            _priority_row("AMD", "SHORT", "near_favorite_zone"),
            _priority_row("MU", "LONG", "watch"),
        ]
    tracked = [
        row
        for row in priority_rows
        if row["priority_bucket"] in {"favorite_setup", "near_favorite_zone"}
    ]
    return {
        "watchlist_label": "test lists",
        "universe_size": len(frames) if universe_size is None else int(universe_size),
        "daily_frames_by_symbol": dict(frames),
        "priority_rows": list(priority_rows),
        "tracked_rows": tracked,
        "theta_put_rows": [{"symbol": "DRAM"}, {"symbol": "AMD"}],
        "theta_pcs_rows": [{"symbol": "NVDA"}],
        "d1_watchlist_symbol_count": 4,
        "setup_tracker_updated": False,
        "setup_tracker_allowed": False,
        "setup_tracker_skip_reason": "",
    }


def _aware(value) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    assert parsed.tzinfo is not None, f"{value!r} carries no timezone offset"
    assert parsed.utcoffset() is not None
    return parsed


def _write_report(text: str) -> Path:
    path = Path(project_paths.MASTER_AVWAP_PRIORITY_SETUPS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def scan_manifest():
    """The module the packet asks for. Imported inside the fixture so a missing
    module is a clean red failure per test instead of a collection error."""
    from master_avwap_lib import scan_manifest as module

    return module


@pytest.fixture
def scan_replay():
    from master_avwap_lib import scan_replay as module

    return module


@pytest.fixture
def clean_home(tmp_path, monkeypatch):
    """No manifest, no history, no dated copies, and the diagnostics run
    manifests in `tmp_path` so nothing here reaches a live store.

    `conftest.py` has already pointed `TRADINGBOTV3_DATA_DIR` and
    `LOCALAPPDATA` at temp directories; this only clears what a previous test
    in this file may have written.
    """
    import diagnostics.run_manifest as rm

    monkeypatch.setattr(rm, "default_manifest_dir", lambda: tmp_path / "run_manifests")
    assert "TradingBotData" not in str(project_paths.DATA_DIR), (
        f"refusing to run against the live home folder {project_paths.DATA_DIR}"
    )

    def _wipe() -> None:
        for name in (
            "MASTER_AVWAP_SCAN_MANIFEST_FILE",
            "MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE",
        ):
            path = getattr(project_paths, name, None)
            if path is not None and Path(path).exists():
                Path(path).unlink()
        reports = Path(project_paths.get_diagnostics_dir()) / "scan_reports"
        if reports.exists():
            for child in reports.iterdir():
                child.unlink()

    _wipe()
    yield
    _wipe()


@pytest.fixture
def freeze_clock(monkeypatch, scan_manifest):
    """Drive the scan's clock from a queue: the first call is the start, every
    later call is the finish. One frozen value would let `started_at ==
    finished_at` pass a test about three separate clocks."""
    from master_avwap_lib import daily_bar_cache

    def _freeze(*moments: datetime):
        queue = list(moments)

        def _now() -> datetime:
            value = queue.pop(0) if len(queue) > 1 else queue[0]
            assert value.tzinfo is not None
            return value

        monkeypatch.setattr(scan_manifest, "market_now", _now)
        monkeypatch.setattr(daily_bar_cache, "market_now", lambda: moments[-1])
        return _now

    return _freeze


def _run_scan(monkeypatch, result_or_error, run_id: str = "master_scan-ws10a"):
    """Drive the REAL publication path: `runner.run_master`, the wrapper that
    owns both the success and the failure branch."""
    from master_avwap_lib import runner

    monkeypatch.setenv("TRADINGBOT_RUN_ID", run_id)
    monkeypatch.setenv("TRADINGBOT_RUN_TRIGGER", "Auto Pilot swing scan (12:00)")

    if isinstance(result_or_error, BaseException):

        def _impl(**kwargs):
            raise result_or_error

    else:

        def _impl(**kwargs):
            return result_or_error

    monkeypatch.setattr(runner, "_run_master_impl", _impl)
    return runner


def _manifest() -> dict:
    path = Path(project_paths.MASTER_AVWAP_SCAN_MANIFEST_FILE)
    assert path.exists(), f"the scan wrote no manifest at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _history_lines() -> list[str]:
    path = Path(project_paths.MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE)
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ===========================================================================
# 1 - the manifest file
# ===========================================================================
def test_the_paths_of_the_scan_manifest_and_its_history_are_named_in_project_paths(clean_home):
    """The packet names both files and says they live in the shared home under a
    `project_paths` constant, addressed by name and never by a literal."""
    manifest = getattr(project_paths, "MASTER_AVWAP_SCAN_MANIFEST_FILE", None)
    history = getattr(project_paths, "MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE", None)
    assert manifest is not None, "project_paths.MASTER_AVWAP_SCAN_MANIFEST_FILE is missing"
    assert history is not None, (
        "project_paths.MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE is missing"
    )
    assert Path(manifest).name == "master_avwap_scan_manifest.json"
    assert Path(history).name == "master_avwap_scan_manifest_history.jsonl"
    for path in (manifest, history):
        assert str(project_paths.DATA_DIR) in str(Path(path)), (
            f"{path} is not in the shared home {project_paths.DATA_DIR}"
        )


def test_a_completed_scan_writes_the_manifest_with_the_three_clocks_and_the_counts(
    clean_home, freeze_clock, monkeypatch
):
    """The three clocks the trader has to be able to separate: when the scan ran
    (`started_at` / `finished_at`), how fresh its INPUTS were
    (`latest_input_bar_session`), and what it published (`outputs`).

    The fixture makes `latest_input_bar_session` a number a wrong formula gets
    wrong: the scan finishes at 12:31 on Friday 2026-09-11, and the newest
    COMPLETED bar any symbol carries is Thursday 2026-09-10. A manifest that
    stamps "today" or the scan's own date fails here.
    """
    freeze_clock(STARTED_AT, FINISHED_AT)
    runner = _run_scan(monkeypatch, _run_result(), run_id="master_scan-ws10a-ok")
    runner.run_master()

    manifest = _manifest()
    assert manifest["status"] == "ok"
    assert manifest["run_id"] == "master_scan-ws10a-ok"
    assert _aware(manifest["started_at"]) == STARTED_AT
    assert _aware(manifest["finished_at"]) == FINISHED_AT
    assert manifest["latest_input_bar_session"] == "2026-09-10"
    assert manifest["preview_bar_used"] is False
    assert manifest["universe_size"] == 3
    assert manifest["symbols_fetched"] == 3
    assert manifest["daily_bar_source_counts"] == {
        legacy.DAILY_BAR_SOURCE_YAHOO: 2,
        legacy.DAILY_BAR_SOURCE_CACHE: 1,
    }
    assert manifest["daily_bars_forming_dropped"] == 0
    assert manifest["daily_bars_invalid_dropped"] == 0

    outputs = {entry["name"]: entry for entry in manifest["outputs"]}
    assert set(outputs) == {"priority_setups", "theta_puts", "d1_watchlist"}
    assert outputs["priority_setups"]["path"] == str(
        project_paths.MASTER_AVWAP_PRIORITY_SETUPS_FILE
    )
    # 3 priority rows, not the 2 TRACKED ones: the report holds what the scan
    # published, and a manifest that counts the tracked subset is wrong by one.
    assert outputs["priority_setups"]["rows"] == 3
    assert outputs["theta_puts"]["rows"] == 3
    assert outputs["d1_watchlist"]["rows"] == 4


def test_a_scan_that_fetched_fewer_symbols_than_its_universe_is_partial_not_ok(
    clean_home, freeze_clock, monkeypatch
):
    """WISHLIST 10A asks for a partial-universe test by name. A scan that
    evaluated 3 of 7 names published a third of a report; calling that `ok`
    is the untruthful freshness the packet exists to end."""
    freeze_clock(STARTED_AT, FINISHED_AT)
    runner = _run_scan(
        monkeypatch,
        _run_result(universe_size=7),
        run_id="master_scan-ws10a-partial",
    )
    runner.run_master()

    manifest = _manifest()
    assert manifest["status"] == "partial"
    assert manifest["universe_size"] == 7
    assert manifest["symbols_fetched"] == 3


def test_a_failed_scan_writes_failed_and_leaves_the_last_good_report_bytes_untouched(
    clean_home, freeze_clock, monkeypatch
):
    """*"Preserve the last good report on failure, labelled stale."* The bytes
    AND the mtime: a republished identical file is a new timestamp, and a new
    timestamp is exactly the false freshness signal the trader reported."""
    report = _write_report("NVDA LONG favorite_setup 88.4\nAMD SHORT near 71.2\n")
    os.utime(report, (1_757_000_000, 1_757_000_000))
    before_bytes = report.read_bytes()
    before_mtime = report.stat().st_mtime

    freeze_clock(STARTED_AT, datetime(2026, 9, 11, 13, 2, 44, tzinfo=ET))
    runner = _run_scan(
        monkeypatch,
        RuntimeError("yahoo daily bars unavailable"),
        run_id="master_scan-ws10a-failed",
    )
    with pytest.raises(RuntimeError, match="yahoo daily bars unavailable"):
        runner.run_master()

    manifest = _manifest()
    assert manifest["status"] == "failed"
    assert "yahoo daily bars unavailable" in str(manifest.get("error", ""))
    assert _aware(manifest["finished_at"]) == datetime(2026, 9, 11, 13, 2, 44, tzinfo=ET)
    assert manifest["outputs"] == []
    assert manifest["latest_input_bar_session"] is None

    assert report.read_bytes() == before_bytes, "the failed scan rewrote the last good report"
    assert report.stat().st_mtime == pytest.approx(before_mtime, abs=1.0), (
        "the failed scan touched the last good report's mtime"
    )


def test_a_forming_bar_preview_input_sets_preview_bar_used(
    clean_home, freeze_clock, monkeypatch
):
    """The same scan, one symbol carrying today's FORMING bar.

    `latest_input_bar_session` must still be Thursday - a forming bar is a
    labelled preview, never a state transition (plan.md sec 5) - and
    `preview_bar_used` must say the preview was in the inputs. A manifest that
    takes `max(last bar date)` reports Friday and fails both assertions.
    """
    frames = {
        "NVDA": _frame(
            [
                _row(PRIOR_SESSION, 180.0),
                _row(LAST_COMPLETE_SESSION, 182.0),
                _forming_row(SCAN_DAY),
            ],
            legacy.DAILY_BAR_SOURCE_YAHOO,
        ),
        "AMD": _frame(
            [_row(PRIOR_SESSION, 150.0), _row(LAST_COMPLETE_SESSION, 151.0)],
            legacy.DAILY_BAR_SOURCE_YAHOO,
        ),
    }
    freeze_clock(STARTED_AT, FINISHED_AT)
    runner = _run_scan(
        monkeypatch,
        _run_result(frames=frames),
        run_id="master_scan-ws10a-preview",
    )
    runner.run_master()

    manifest = _manifest()
    assert manifest["preview_bar_used"] is True
    assert manifest["latest_input_bar_session"] == "2026-09-10"


def test_an_early_close_session_is_complete_only_after_the_regular_close(
    clean_home, freeze_clock, monkeypatch
):
    """2026-11-27 is the half day after Thanksgiving (13:00 ET close), and
    2026-11-26 is the holiday, so the session before it is Wednesday 11-25.

    At 14:00 the bar for 11-27 has stopped moving but `market_calendar` does not
    model early closes and judges every session against 16:00 (WS-FC1's rule).
    Conservative is the right direction: the manifest calls 11-27 a preview and
    the newest COMPLETE session 11-25. After 16:05 the same bar is complete.
    A manifest that compares dates instead of session closes says `2026-11-27`
    at 14:00 and fails the first half.
    """
    from master_avwap_lib import runner

    frames = {
        "NVDA": _frame(
            [
                _row(date(2026, 11, 24), 180.0),
                _row(EARLY_CLOSE_PRIOR_SESSION, 182.0),
                _forming_row(EARLY_CLOSE_DAY),
            ],
            legacy.DAILY_BAR_SOURCE_YAHOO,
        )
    }

    freeze_clock(
        datetime(2026, 11, 27, 13, 40, tzinfo=ET),
        datetime(2026, 11, 27, 14, 0, tzinfo=ET),
    )
    _run_scan(monkeypatch, _run_result(frames=frames), run_id="master_scan-ws10a-half-1")
    runner.run_master()

    midday = _manifest()
    assert midday["latest_input_bar_session"] == "2026-11-25"
    assert midday["preview_bar_used"] is True

    freeze_clock(
        datetime(2026, 11, 27, 16, 4, tzinfo=ET),
        datetime(2026, 11, 27, 16, 5, tzinfo=ET),
    )
    _run_scan(monkeypatch, _run_result(frames=frames), run_id="master_scan-ws10a-half-2")
    runner.run_master()

    after_close = _manifest()
    assert after_close["latest_input_bar_session"] == "2026-11-27"
    assert after_close["preview_bar_used"] is False


def test_every_manifest_timestamp_carries_a_timezone_offset(
    clean_home, freeze_clock, monkeypatch
):
    """*"timestamps carry explicit timezones"* (plan.md sec 5). Every stamp in
    the manifest, in its history line and in the dated copy parses aware and
    keeps the offset it was written with - the 2026-09-05 sidecar defect was a
    naive stamp compared against an aware one."""
    freeze_clock(STARTED_AT, FINISHED_AT)
    runner = _run_scan(monkeypatch, _run_result(), run_id="master_scan-ws10a-tz")
    runner.run_master()

    manifest = _manifest()
    for field in ("started_at", "finished_at"):
        assert _aware(manifest[field]).utcoffset() == FINISHED_AT.utcoffset()

    line = json.loads(_history_lines()[-1])
    assert _aware(line["finished_at"]).utcoffset() == FINISHED_AT.utcoffset()

    from master_avwap_lib import scan_manifest as module

    copies = sorted(Path(module.scan_reports_dir()).glob("*.json"))
    assert copies, "the scan kept no dated report copy"
    copy = json.loads(copies[-1].read_text(encoding="utf-8"))
    assert _aware(copy["finished_at"]).utcoffset() == FINISHED_AT.utcoffset()


def test_every_scan_appends_exactly_one_history_line_and_never_rewrites_an_earlier_one(
    clean_home, freeze_clock, monkeypatch
):
    """The replay walks this file, so it is append-only: two scans leave two
    lines and the first line is byte-identical afterwards."""
    freeze_clock(STARTED_AT, FINISHED_AT)
    runner = _run_scan(monkeypatch, _run_result(), run_id="master_scan-ws10a-h1")
    runner.run_master()

    first_lines = _history_lines()
    assert len(first_lines) == 1
    first = first_lines[0]

    freeze_clock(
        datetime(2026, 9, 11, 13, 0, tzinfo=ET),
        datetime(2026, 9, 11, 13, 2, 44, tzinfo=ET),
    )
    _run_scan(
        monkeypatch,
        RuntimeError("yahoo daily bars unavailable"),
        run_id="master_scan-ws10a-h2",
    )
    with pytest.raises(RuntimeError):
        runner.run_master()

    lines = _history_lines()
    assert len(lines) == 2, "each scan appends exactly one history line"
    assert lines[0] == first, "an earlier history line was rewritten"

    second = json.loads(lines[1])
    assert second["status"] == "failed"
    assert second["run_id"] == "master_scan-ws10a-h2"


def test_the_scan_keeps_a_dated_report_copy_capped_at_thirty_dates(
    clean_home, freeze_clock, monkeypatch, scan_manifest
):
    """The desk keeps no historic snapshot of the priority report, so the replay
    has nothing to read - the packet's parenthesis. The smallest dated copy per
    scan is added here, and the cap is 30 distinct dates: 34 seeded dates plus
    today's scan leaves exactly 30, the 5 oldest deleted.
    """
    reports = Path(scan_manifest.scan_reports_dir())
    reports.mkdir(parents=True, exist_ok=True)
    seeded = []
    for offset in range(1, 35):
        day = SCAN_DAY - timedelta(days=offset)
        path = reports / f"{day.isoformat()}_1600.json"
        path.write_text(json.dumps({"run_id": f"seed-{offset}", "rows": []}), encoding="utf-8")
        seeded.append(path)

    freeze_clock(STARTED_AT, FINISHED_AT)
    runner = _run_scan(monkeypatch, _run_result(), run_id="master_scan-ws10a-cap")
    runner.run_master()

    assert scan_manifest.REPORT_COPY_RETENTION_SESSIONS == 30
    kept = sorted(path.name for path in reports.glob("*.json"))
    assert len(kept) == 30, f"expected 30 dated copies after the cap, saw {kept}"
    assert "2026-09-11_1231.json" in kept, "this scan's own dated copy was not kept"
    for path in seeded[-5:]:
        assert not path.exists(), f"{path.name} is older than the cap and was kept"

    copy = json.loads((reports / "2026-09-11_1231.json").read_text(encoding="utf-8"))
    assert copy["run_id"] == "master_scan-ws10a-cap"
    assert copy["latest_input_bar_session"] == "2026-09-10"
    assert {row["symbol"]: row["bucket"] for row in copy["rows"]} == {
        "NVDA": "favorite_setup",
        "AMD": "near_favorite_zone",
        "MU": "watch",
    }


# ===========================================================================
# 2 - the truthful strip
# ===========================================================================
def _ok_manifest() -> dict:
    return {
        "run_id": "master_scan-ws10a-ok",
        "status": "ok",
        "started_at": STARTED_AT.isoformat(),
        "finished_at": FINISHED_AT.isoformat(),
        "universe_size": 3,
        "symbols_fetched": 3,
        "daily_bar_source_counts": {legacy.DAILY_BAR_SOURCE_YAHOO: 3},
        "latest_input_bar_session": LAST_COMPLETE_SESSION.isoformat(),
        "preview_bar_used": False,
        "daily_bars_forming_dropped": 0,
        "daily_bars_invalid_dropped": 0,
        "outputs": [
            {
                "name": "priority_setups",
                "path": str(project_paths.MASTER_AVWAP_PRIORITY_SETUPS_FILE),
                "rows": 3,
            }
        ],
    }


def test_the_strip_says_scan_ok_with_the_three_clocks(scan_manifest):
    """The packet's own line. Three clocks in one sentence: the scan finished at
    12:31, its inputs ran through Thursday's COMPLETE daily bar, and the report
    on screen is the 12:31 one."""
    line = scan_manifest.freshness_line(_ok_manifest(), report_mtime=FINISHED_AT)
    assert line == (
        "Scan ok 12:31 · inputs through Thu 09-10 (D1 complete) · shown: 12:31 report"
    )


def test_the_strip_says_failed_and_names_the_stale_report_it_is_still_showing(
    scan_manifest,
):
    """The packet's second line. The failure does not blank the screen and does
    not pretend the 13:02 attempt produced what is on it: the trader is told,
    in one line, that what they are reading is the 12:31 report and it is
    stale."""
    manifest = {
        "run_id": "master_scan-ws10a-failed",
        "status": "failed",
        "started_at": datetime(2026, 9, 11, 13, 0, tzinfo=ET).isoformat(),
        "finished_at": datetime(2026, 9, 11, 13, 2, 44, tzinfo=ET).isoformat(),
        "error": "RuntimeError('yahoo daily bars unavailable')",
        "universe_size": 0,
        "symbols_fetched": 0,
        "daily_bar_source_counts": {},
        "latest_input_bar_session": None,
        "preview_bar_used": False,
        "outputs": [],
    }
    line = scan_manifest.freshness_line(manifest, report_mtime=FINISHED_AT)
    assert line == "Scan FAILED 13:02 · showing 12:31 report (stale)"


def test_the_strip_says_today_preview_when_the_inputs_carried_a_forming_bar(
    scan_manifest,
):
    """A preview-bar scan never claims "(D1 complete)". The manifest below is
    the half-day one: inputs complete through Wednesday, today's bar a preview.
    """
    manifest = _ok_manifest()
    manifest["started_at"] = datetime(2026, 11, 27, 10, 50, tzinfo=ET).isoformat()
    manifest["finished_at"] = datetime(2026, 11, 27, 11, 5, tzinfo=ET).isoformat()
    manifest["latest_input_bar_session"] = EARLY_CLOSE_PRIOR_SESSION.isoformat()
    manifest["preview_bar_used"] = True

    line = scan_manifest.freshness_line(
        manifest, report_mtime=datetime(2026, 11, 27, 11, 5, tzinfo=ET)
    )
    assert line == (
        "Scan ok 11:05 · inputs through Wed 11-25 · inputs: today preview "
        "· shown: 11:05 report"
    )
    assert "(D1 complete)" not in line


def test_a_partial_scan_says_how_much_of_its_universe_it_reached(scan_manifest):
    """"Partial" with no number is not truthful either: the trader needs the
    two counts to judge whether an absent name means anything."""
    manifest = _ok_manifest()
    manifest["status"] = "partial"
    manifest["universe_size"] = 1097
    manifest["symbols_fetched"] = 940

    line = scan_manifest.freshness_line(manifest, report_mtime=FINISHED_AT)
    assert line == (
        "Scan partial 12:31 (940 of 1097 symbols) · inputs through Thu 09-10 "
        "(D1 complete) · shown: 12:31 report"
    )


def test_no_manifest_yet_says_so_instead_of_guessing(scan_manifest, clean_home):
    """Missing data is uncertainty, never confirmation. Before the first scan of
    a new build there is no manifest, and the strip must not fall back to a file
    mtime - a recent mtime is the very signal the trader says proves nothing."""
    assert scan_manifest.read_manifest() is None
    assert scan_manifest.freshness_line(None) == "Scan: not recorded yet · no report"
    assert scan_manifest.freshness_line(None, report_mtime=FINISHED_AT) == (
        "Scan: not recorded yet · shown: 12:31 report"
    )


# ---------------------------------------------------------------------------
# the strip, on the real surfaces
# ---------------------------------------------------------------------------
def _write_manifest(payload: dict) -> Path:
    path = Path(project_paths.MASTER_AVWAP_SCAN_MANIFEST_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pump(app, predicate, timeout: float = 5.0) -> bool:
    """Let the panel finish whatever it started off the Qt thread.

    The packet requires the manifest to be read OFF the Qt thread, so the label
    may be set by a queued signal rather than inside `refresh_from_reports`.
    This waits for either shape.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    app.processEvents()
    return predicate()


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.mark.qt
def test_the_setups_panel_status_row_shows_the_scan_line_after_a_report_refresh(
    qt_app, clean_home
):
    """The REAL path: the panel's existing `refresh_from_reports`, which the
    report watcher and the scheduler both call. On `e01aede4` the panel shows
    `Last run: never` and nothing about the inputs, so no label carries this
    sentence.

    The `shown:` clock depends on the report file's mtime, which is the
    machine's; the three pure tests above pin its formatting. What is asserted
    here is the manifest-derived part, which is machine-independent.
    """
    from PySide6.QtWidgets import QLabel

    from ui.panels.master_avwap_panel import MasterAvwapPanel

    _write_manifest(_ok_manifest())
    _write_report("NVDA LONG favorite_setup 88.4\n")

    expected = "Scan ok 12:31 · inputs through Thu 09-10 (D1 complete)"
    panel = MasterAvwapPanel()
    try:
        panel.resize(1600, 900)
        panel.show()
        qt_app.processEvents()
        panel.refresh_from_reports()

        def _shows() -> bool:
            return any(
                expected in label.text() for label in panel.findChildren(QLabel)
            )

        assert _pump(qt_app, _shows), (
            "no label in the Setups panel carries the scan freshness line; saw "
            + repr(
                [
                    label.text()
                    for label in panel.findChildren(QLabel)
                    if label.text().strip()
                ]
            )
        )
        shown = [
            label.text() for label in panel.findChildren(QLabel) if expected in label.text()
        ][0]
        assert "shown: " in shown, f"the line names no shown-report clock: {shown!r}"
    finally:
        panel.deleteLater()
        qt_app.processEvents()


@pytest.mark.qt
def test_the_manifest_is_read_on_the_refresh_path_and_never_on_paint(
    qt_app, clean_home, monkeypatch, scan_manifest
):
    """*"never `json.loads` on paint"*. The refresh path reads the manifest; a
    repaint of the table and the whole panel must read nothing."""
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    _write_manifest(_ok_manifest())
    _write_report("NVDA LONG favorite_setup 88.4\n")

    calls: list[int] = []
    real = scan_manifest.read_manifest

    def _counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scan_manifest, "read_manifest", _counted)

    panel = MasterAvwapPanel()
    try:
        panel.resize(1600, 900)
        panel.show()
        qt_app.processEvents()
        panel.refresh_from_reports()
        assert _pump(qt_app, lambda: bool(calls)), (
            "the refresh path never read the scan manifest"
        )

        calls.clear()
        for _ in range(5):
            panel.table.viewport().update()
            panel.update()
            qt_app.processEvents()
        assert calls == [], (
            f"the manifest was re-read {len(calls)} time(s) while painting"
        )
    finally:
        panel.deleteLater()
        qt_app.processEvents()


def test_the_system_health_strip_carries_the_same_scan_freshness_line(
    clean_home, tmp_path, scan_manifest
):
    """The packet puts the line on both surfaces, and one builder makes it, so
    System Health must print the SAME sentence rather than a second opinion."""
    import operations_audit

    from market_session import get_market_local_timezone

    _write_manifest(_ok_manifest())
    report = _write_report("NVDA LONG favorite_setup 88.4\n")
    # The `shown` clock is the report file's mtime read in MARKET-LOCAL time
    # (Pacific on this desk), which is the conversion the contract names. It is
    # computed here the same way so the assertion is about the sentence, not
    # about whose timezone the test process happens to be in.
    local_tz, _name = get_market_local_timezone()
    mtime = datetime.fromtimestamp(report.stat().st_mtime, tz=local_tz)
    expected = scan_manifest.freshness_line(_ok_manifest(), report_mtime=mtime)

    payload = operations_audit.build_operations_audit(
        now=datetime(2026, 9, 11, 13, 0),
        diagnostics_dir=tmp_path,
        review_capture=False,
    )
    checks = {str(check.get("id")): check for check in payload["checks"]}
    assert "master_scan_freshness" in checks, (
        "System Health emits no master_scan_freshness check; saw " + repr(sorted(checks))
    )
    summary = str(checks["master_scan_freshness"]["summary"])
    assert summary.startswith(
        "Scan ok 12:31 · inputs through Thu 09-10 (D1 complete)"
    ), f"System Health wrote its own sentence: {summary!r}"
    assert summary == expected


# ===========================================================================
# 3 - the replay CLI
# ===========================================================================
REPLAY_SESSION = date(2026, 9, 10)
#: The three checkpoints that WERE recorded, and the close that was not. The
#: delay the trader is asking about is 12:30 -> 15:05 = 155 minutes.
REPLAY_SNAPSHOTS = [
    (
        datetime(2026, 9, 10, 9, 58, tzinfo=ET),
        "2026-09-09",
        [{"symbol": "AMD", "side": "LONG", "bucket": "favorite_setup"}],
    ),
    (
        datetime(2026, 9, 10, 12, 30, tzinfo=ET),
        "2026-09-09",
        [
            {"symbol": "AMD", "side": "LONG", "bucket": "favorite_setup"},
            {"symbol": "NVDA", "side": "LONG", "bucket": "near_favorite_zone"},
        ],
    ),
    (
        datetime(2026, 9, 10, 15, 5, tzinfo=ET),
        "2026-09-09",
        [
            {"symbol": "AMD", "side": "LONG", "bucket": "favorite_setup"},
            {"symbol": "NVDA", "side": "LONG", "bucket": "favorite_setup"},
        ],
    ),
]


@pytest.fixture
def replay_history(clean_home, scan_manifest):
    """A recorded day: three dated copies and three history lines, no close."""
    reports = Path(scan_manifest.scan_reports_dir())
    reports.mkdir(parents=True, exist_ok=True)
    history = Path(project_paths.MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE)
    history.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, (finished, inputs, rows) in enumerate(REPLAY_SNAPSHOTS):
        run_id = f"master_scan-replay-{index}"
        payload = {
            "run_id": run_id,
            "status": "ok",
            "finished_at": finished.isoformat(),
            "latest_input_bar_session": inputs,
            "preview_bar_used": True,
            "rows": rows,
        }
        name = f"{finished.date().isoformat()}_{finished.strftime('%H%M')}.json"
        (reports / name).write_text(json.dumps(payload), encoding="utf-8")
        lines.append(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "ok",
                    "started_at": (finished - timedelta(minutes=22)).isoformat(),
                    "finished_at": finished.isoformat(),
                    "latest_input_bar_session": inputs,
                    "preview_bar_used": True,
                    "universe_size": 1097,
                    "symbols_fetched": 1097,
                }
            )
        )
    history.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reports


def test_the_replay_prints_the_four_checkpoints_for_the_name_it_was_asked_about(
    replay_history, scan_replay, capsys
):
    """The trader's question, answered from what was RECORDED: was NVDA in the
    report at the open, at midday, in the final hour and at the close, and what
    bucket was it in each time.

    The fixture is the trader's own complaint in miniature: NVDA is absent at
    the open, near-zone at midday and only a favorite in the final hour.
    """
    assert scan_replay.CHECKPOINTS == ("open", "midday", "final hour", "close")

    exit_code = scan_replay.main(["--symbol", "NVDA", "--session", "2026-09-10"])
    assert exit_code == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert str(project_paths.DATA_DIR) in lines[0], (
        "a read-only CLI prints the home folder it resolved first"
    )

    def _line_for(checkpoint: str) -> str:
        matches = [line for line in lines if line.strip().startswith(checkpoint)]
        assert matches, f"no {checkpoint!r} line in {lines!r}"
        return matches[0]

    open_line = _line_for("open")
    assert "09:58" in open_line
    assert "not in report" in open_line
    assert "2026-09-09" in open_line

    midday = _line_for("midday")
    assert "12:30" in midday
    assert "in report" in midday and "not in report" not in midday
    assert "near_favorite_zone" in midday

    final_hour = _line_for("final hour")
    assert "15:05" in final_hour
    assert "favorite_setup" in final_hour

    assert _line_for("close").strip() == "close · no recorded snapshot"


def test_the_replay_reports_the_delay_between_first_eligibility_and_first_publication(
    replay_history, scan_replay, capsys
):
    """The number the brief asks for: *"record first eligibility, first
    publication, delay"*. NVDA was eligible at 12:30 and published at 15:05, so
    the delay is 155 minutes - not 0 (which is what reading one clock twice
    gives) and not 307 (the open-to-final-hour span)."""
    scan_replay.main(["--symbol", "NVDA", "--session", "2026-09-10"])
    out = capsys.readouterr().out

    assert "first eligible 12:30" in out
    assert "first published 15:05" in out
    assert "delay 155 min" in out


def test_the_replay_says_no_recorded_snapshot_instead_of_reconstructing_one(
    clean_home, scan_replay, capsys
):
    """*"If historic intermediate snapshots do not exist, ... label the missing
    proof rather than reconstructing certainty."* With no history at all, every
    checkpoint says so and the summary refuses to name a delay."""
    exit_code = scan_replay.main(["--symbol", "NVDA", "--session", "2026-09-10"])
    assert exit_code == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    for checkpoint in ("open", "midday", "final hour", "close"):
        matches = [line for line in lines if line.strip().startswith(checkpoint)]
        assert matches, f"no {checkpoint!r} line in {lines!r}"
        assert matches[0].strip() == f"{checkpoint} · no recorded snapshot"

    joined = "\n".join(lines)
    assert "delay unmeasured" in joined
    assert re.search(r"delay\s+\d+\s*min", joined) is None, (
        "a delay was computed from a day with no recorded snapshot"
    )
