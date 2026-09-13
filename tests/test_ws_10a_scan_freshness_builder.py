"""Packet WS-10A - the three rules the tester's file could not reach.

Added by the BUILDER (2026-09-13) on top of
``tests/test_ws_10a_scan_freshness.py``. Nothing here weakens or replaces one of
the tester's assertions; each of these pins a behaviour their fixtures could not
observe, and each was proven to fail before the code that satisfies it existed
(the proof is in the commit message).

1. The tester drives ``runner.run_master`` with ``_run_master_impl`` replaced,
   so the two keys the REAL scan has to add to ``run_result`` are supplied by
   the fixture and never by production. A scan whose payload omits them writes
   a manifest saying ``0 of 0 symbols`` and ``priority_setups: 0 rows``
   forever - green tests, useless manifest.
2. The checkpoint windows in the replay are the MARKET's clock, not the stamp's
   own. On this Pacific desk a 09:58 stamp is 12:58 in New York, which is
   midday; filing it under "open" would answer the trader's "did it arrive in
   the final hour?" with the wrong checkpoint every single day.
3. The manifest's input-bar freshness asks the exchange calendar ONCE per scan.
   The calendar was 84% of a GIL sample on 2026-09-03; a completeness question
   asked per symbol (let alone per row) over 1,097 frames would put it back.
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

from master_avwap_lib import scan_manifest, scan_replay  # noqa: E402

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")


def test_the_real_scan_payload_carries_the_universe_size_and_the_published_rows(
    monkeypatch,
):
    """The manifest's ``partial`` verdict and its ``priority_setups`` row count
    are read off ``run_result``, so the SCAN has to put them there.

    Driven through the no-symbols early return, which is the one branch of
    ``_run_master_impl`` a test can reach without a market data feed - and the
    branch most likely to be forgotten, because it hand-builds its own payload
    dict instead of falling through to the main one.
    """
    from master_avwap_lib import runner

    monkeypatch.setattr(runner, "load_tickers_from_paths", lambda *a, **k: [])
    monkeypatch.setattr(
        runner, "append_master_avwap_d1_watchlist_symbols", lambda longs, shorts: ([], [], 0)
    )
    monkeypatch.setattr(runner, "load_theta_long_symbols", lambda *a, **k: [])
    monkeypatch.setattr(runner, "write_theta_put_report", lambda *a, **k: None)

    result = runner._run_master_impl()

    assert "universe_size" in result, (
        "the scan payload carries no universe_size, so every manifest would say "
        "'0 of 0 symbols' and no short run could ever be called partial"
    )
    assert "priority_rows" in result, (
        "the scan payload carries no priority_rows, so the manifest's "
        "priority_setups output would always report 0 rows"
    )
    assert result["universe_size"] == 0
    assert result["priority_rows"] == []


def test_the_replay_files_a_snapshot_by_the_markets_clock_not_the_stamps_own(
    tmp_path,
):
    """A Pacific 09:58 stamp is 12:58 in New York, so it is MIDDAY.

    The strip renders stamps in the offset they carry ("when did the scan run"
    is a question about the desk). The checkpoints are the other question - and
    answering "was it in the report at the open?" with a scan that finished
    three hours into the session would make the trader's own complaint
    unfalsifiable.
    """
    import json

    session = date(2026, 9, 10)
    stamp = datetime(2026, 9, 10, 9, 58, tzinfo=PT)
    (tmp_path / f"{session.isoformat()}_0958.json").write_text(
        json.dumps(
            {
                "run_id": "pacific",
                "status": "ok",
                "finished_at": stamp.isoformat(),
                "latest_input_bar_session": "2026-09-09",
                "preview_bar_used": False,
                "rows": [{"symbol": "NVDA", "side": "LONG", "bucket": "favorite_setup"}],
            }
        ),
        encoding="utf-8",
    )

    snapshots = scan_replay.load_snapshots(session, reports_dir=tmp_path)
    assert len(snapshots) == 1
    assert snapshots[0]["_checkpoint"] == "midday", (
        "a 09:58 Pacific stamp was filed under the wrong checkpoint: it is "
        "12:58 on the exchange's clock"
    )

    lines = scan_replay.build_report("NVDA", session, reports_dir=tmp_path)
    midday = [line for line in lines if line.startswith("midday")][0]
    assert "12:58" in midday, f"the replay printed the desk's clock, not the market's: {midday!r}"
    assert [line for line in lines if line.startswith("open")][0].strip() == (
        "open · no recorded snapshot"
    )


def test_the_input_bar_freshness_asks_the_exchange_calendar_once_per_scan(monkeypatch):
    """One calendar call for the whole scan, not one per symbol.

    ``market_calendar`` is memoized but the walk is not free, and this runs over
    every symbol the scan fetched. The 2026-09-03 freeze was an uncached
    calendar at 84% of the GIL samples; a manifest is not allowed to reintroduce
    it.
    """
    from master_avwap_lib import daily_bar_cache

    calls: list[object] = []
    real = daily_bar_cache.last_completed_session

    def _counted(now=None):
        calls.append(now)
        return real(now)

    monkeypatch.setattr(daily_bar_cache, "last_completed_session", _counted)

    finished = datetime(2026, 9, 11, 12, 31, tzinfo=ET)
    frames = {}
    for index in range(40):
        days = [date(2026, 9, 9), date(2026, 9, 10)]
        frames[f"SYM{index}"] = pd.DataFrame(
            {
                "datetime": pd.to_datetime([pd.Timestamp(day) for day in days]),
                "close": [10.0, 11.0],
            }
        )

    session, preview = scan_manifest.input_bar_freshness(frames, now=finished)

    assert session == "2026-09-10"
    assert preview is False
    assert len(calls) == 1, (
        f"the exchange calendar was asked {len(calls)} times for one scan's freshness"
    )


def test_a_scan_with_no_frames_reports_no_session_rather_than_guessing_one():
    """Missing data is uncertainty. A scan that fetched nothing has no input-bar
    session, and the strip says so instead of naming yesterday."""
    session, preview = scan_manifest.input_bar_freshness(
        {}, now=datetime(2026, 9, 11, 12, 31, tzinfo=ET)
    )
    assert session is None
    assert preview is False

    manifest = {
        "status": "ok",
        "finished_at": datetime(2026, 9, 11, 12, 31, tzinfo=ET).isoformat(),
        "latest_input_bar_session": None,
        "preview_bar_used": False,
        "universe_size": 0,
        "symbols_fetched": 0,
    }
    line = scan_manifest.freshness_line(manifest)
    assert "inputs: no completed session recorded" in line
    assert "(D1 complete)" not in line


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_the_dated_copy_cap_counts_dates_and_not_files(tmp_path, monkeypatch, offset):
    """Six copies on one busy day must not evict five other days.

    The packet says "a 30-day cap"; counting FILES would let an active Friday
    push out the whole week before it, and the replay's window would silently
    shorten exactly when the trader has the most to look at.
    """
    monkeypatch.setattr(scan_manifest, "scan_reports_dir", lambda: tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = date(2026, 9, 11)
    for day_offset in range(scan_manifest.REPORT_COPY_RETENTION_SESSIONS):
        day = base - timedelta(days=day_offset)
        for hour in range(offset):
            (tmp_path / f"{day.isoformat()}_1{hour}00.json").write_text("{}", encoding="utf-8")

    scan_manifest._prune_report_copies(tmp_path)

    kept_dates = {path.name.split("_", 1)[0] for path in tmp_path.glob("*.json")}
    assert len(kept_dates) == scan_manifest.REPORT_COPY_RETENTION_SESSIONS, (
        f"the cap evicted whole dates because it counted files: kept {sorted(kept_dates)}"
    )
