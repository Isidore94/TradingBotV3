"""Packet Q5 - the daily pick scorecard leaves the Qt thread.

Process review 2026-09-04, performance: `_score_todays_picks` materialised two
runtime CSVs (335 MB + 308 MB) on the calling thread and `ui_stalls.jsonl`
recorded 15,739 ms at 13:00:44 PT attributed to it. `picks_scored_at` was
written BEFORE scoring, so a failure was never retried. Every red test here
failed on `6b74165`.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import autopilot_core as core  # noqa: E402
from ui.services import autopilot_service as module  # noqa: E402
from ui.services.autopilot_service import AutopilotService  # noqa: E402

TODAY = "2026-09-04"
NOW = datetime(2026, 9, 4, 13, 5, 0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate(date, symbol, event_id, *, confirmed="True"):
    return {
        "trade_date": date, "symbol": symbol, "direction": "long", "event_id": event_id,
        "confirmed": confirmed, "bounce_types": "ema_15",
    }


def _outcome(date, event_id, *, close_r="1.0", mfe_r="2.0"):
    return {
        "trade_date": date, "event_id": event_id, "event_type": "final",
        "close_r": close_r, "mfe_r": mfe_r, "stop_hit": "False", "symbol": event_id.split("_")[0],
        "direction": "long",
    }


@pytest.fixture
def stores(tmp_path, monkeypatch):
    picks = tmp_path / "autopilot_picks.csv"
    candidates = tmp_path / "candidates.csv"
    outcomes = tmp_path / "outcomes.csv"
    scorecard = tmp_path / "scorecard.csv"
    _write_csv(picks, [
        {"date": TODAY, "side": "long", "symbol": "AAA", "source": "bot", "why": "x"},
        {"date": "2026-09-03", "side": "long", "symbol": "OLD", "source": "bot", "why": "x"},
    ])
    _write_csv(candidates, [
        _candidate("2026-09-02", "OLD", "OLD_long_20260902_09_35_00_ema_15"),
        _candidate("2026-09-03", "OLD", "OLD_long_20260903_09_35_00_ema_15"),
        _candidate(TODAY, "AAA", "AAA_long_20260904_09_35_00_ema_15"),
        _candidate(TODAY, "BBB", "BBB_long_20260904_09_40_00_ema_15"),
    ])
    _write_csv(outcomes, [
        _outcome("2026-09-02", "OLD_long_20260902_09_35_00_ema_15"),
        _outcome("2026-09-03", "OLD_long_20260903_09_35_00_ema_15"),
        _outcome(TODAY, "AAA_long_20260904_09_35_00_ema_15", close_r="0.5", mfe_r="1.5"),
        _outcome(TODAY, "AAA_long_20260904_09_35_00_ema_15", close_r="0.7", mfe_r="1.9"),
        _outcome(TODAY, "BBB_long_20260904_09_40_00_ema_15"),
    ])
    monkeypatch.setattr(module, "AUTOPILOT_PICKS_FILE", picks)
    monkeypatch.setattr(module, "INTRADAY_BOUNCE_CANDIDATES_FILE", candidates)
    monkeypatch.setattr(module, "INTRADAY_BOUNCE_OUTCOMES_FILE", outcomes)
    monkeypatch.setattr(module, "AUTOPILOT_SCORECARD_FILE", scorecard)
    monkeypatch.setattr(core, "last_completed_session_close", lambda now, *a, **k: NOW.replace(hour=13, minute=0))
    return {"picks": picks, "candidates": candidates, "outcomes": outcomes, "scorecard": scorecard}


def _service(state=None):
    service = AutopilotService.__new__(AutopilotService)
    service._state = dict(state or {"date": TODAY})
    service._logged = []
    service._log = service._logged.append  # type: ignore[method-assign]
    service._save_state = lambda: None  # type: ignore[method-assign]
    service._scorecard_line = "last good line"
    service._scorecard_running = False
    service._snapshot_manual_picks = lambda now: None  # type: ignore[method-assign]
    return service


def _join_scorecard_threads(timeout=10.0):
    for thread in threading.enumerate():
        if thread.name == "autopilot-scorecard":
            thread.join(timeout)


def _old_way_lines(stores) -> list[str]:
    """What 6b74165 computed: materialise both files, filter after."""
    with stores["picks"].open("r", newline="", encoding="utf-8") as handle:
        picks = [row for row in csv.DictReader(handle) if row.get("date") == TODAY]
    with stores["candidates"].open("r", newline="", encoding="utf-8") as handle:
        candidates = [row for row in csv.DictReader(handle) if row.get("trade_date") == TODAY]
    ids = {row["event_id"] for row in candidates}
    with stores["outcomes"].open("r", newline="", encoding="utf-8") as handle:
        outcomes = [row for row in csv.DictReader(handle) if row.get("event_id") in ids]
    lines = []
    for group, group_picks in sorted(core.group_picks_by_source(picks).items()):
        scorecard = core.score_autopilot_picks(group_picks, candidates, outcomes)
        lines.append(core.format_scorecard_line(scorecard, label=core.PICK_GROUP_LABELS.get(group, group)))
    return lines


# ---------------------------------------------------------------------------
# Q5.1 - one owned worker; the Qt thread reads no CSV
# ---------------------------------------------------------------------------


def test_the_calling_thread_never_opens_either_csv(stores, monkeypatch):
    opened: list[tuple[str, str]] = []
    real_open = Path.open
    watched = {str(stores["candidates"]), str(stores["outcomes"])}

    def _spy(self, *args, **kwargs):
        if str(self) in watched:
            opened.append((str(self), threading.current_thread().name))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _spy)
    service = _service()
    caller = threading.current_thread().name

    service._maybe_score_picks_daily(NOW)
    _join_scorecard_threads()

    assert opened, "the scorecard read both files"
    assert all(name != caller for _path, name in opened), opened
    assert {name for _p, name in opened} == {"autopilot-scorecard"}
    assert service._state.get("picks_scored_at")


def test_two_triggers_produce_one_worker_and_one_scorecard_append(stores, monkeypatch):
    gate = threading.Event()
    started: list[str] = []
    real_body = AutopilotService._score_picks_now

    def _slow_body(self, now):
        started.append(threading.current_thread().name)
        gate.wait(5)
        return real_body(self, now)

    monkeypatch.setattr(AutopilotService, "_score_picks_now", _slow_body)
    service = _service()

    service._maybe_score_picks_daily(NOW)
    service._maybe_score_picks_daily(NOW)
    gate.set()
    _join_scorecard_threads()

    assert started == ["autopilot-scorecard"], "a second trigger while running is a no-op"
    with stores["scorecard"].open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, "one row per pick group, appended once"


def test_a_failure_keeps_last_good_and_leaves_picks_scored_at_unset(stores, monkeypatch):
    service = _service()

    def _boom(now):
        raise RuntimeError("the picks file is locked")

    service._snapshot_manual_picks = _boom  # type: ignore[method-assign]

    service._maybe_score_picks_daily(NOW)
    _join_scorecard_threads()

    assert service._state.get("picks_scored_at") is None, "a failed run must be retried"
    assert service._scorecard_line == "last good line"
    assert service._state.get("scorecard_attempts_today") == {"date": TODAY, "count": 1}
    assert not stores["scorecard"].exists(), "no partial scorecard row"


def test_the_third_failure_marks_the_day_and_a_fourth_trigger_starts_nothing(stores, monkeypatch):
    service = _service()
    calls: list[int] = []

    def _boom(now):
        calls.append(1)
        raise RuntimeError("still locked")

    service._snapshot_manual_picks = _boom  # type: ignore[method-assign]

    for _ in range(3):
        service._maybe_score_picks_daily(NOW)
        _join_scorecard_threads()
    assert len(calls) == 3
    assert service._state.get("picks_scoring_failed_at")
    assert service._state.get("picks_scored_at") is None

    service._maybe_score_picks_daily(NOW)
    _join_scorecard_threads()
    assert len(calls) == 3, "a day marked failed spins no more workers"


def test_a_success_after_a_failure_clears_the_attempt_counter(stores, monkeypatch):
    service = _service()
    state = {"n": 0}

    def _flaky(now):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("once")

    service._snapshot_manual_picks = _flaky  # type: ignore[method-assign]

    service._maybe_score_picks_daily(NOW)
    _join_scorecard_threads()
    assert service._state.get("scorecard_attempts_today", {}).get("count") == 1

    service._maybe_score_picks_daily(NOW)
    _join_scorecard_threads()
    assert service._state.get("picks_scored_at")
    assert "scorecard_attempts_today" not in service._state
    assert service._scorecard_line != "last good line"


def test_the_wrapup_path_runs_the_body_inline_and_never_nests_a_thread(stores, monkeypatch):
    service = _service()
    seen: list[str] = []
    real_body = AutopilotService._score_picks_now

    def _spy(self, now):
        seen.append(threading.current_thread().name)
        return real_body(self, now)

    monkeypatch.setattr(AutopilotService, "_score_picks_now", _spy)
    threading.current_thread().name = "autopilot-wrapup-test"
    try:
        service._score_picks_inline(NOW)
    finally:
        threading.current_thread().name = "MainThread"

    assert seen == ["autopilot-wrapup-test"]
    assert service._state.get("picks_scored_at")


# ---------------------------------------------------------------------------
# Q5.2 - narrow reads, same answer
# ---------------------------------------------------------------------------


def test_the_streamed_reader_keeps_only_todays_rows_and_matches_the_old_answer(stores):
    candidates, outcomes = core.read_scorecard_inputs(
        stores["candidates"], stores["outcomes"], TODAY
    )
    assert {row["trade_date"] for row in candidates} == {TODAY}
    assert len(candidates) == 2
    assert len(outcomes) == 3
    assert all(row["trade_date"] == TODAY for row in outcomes)

    service = _service()
    lines = service._score_todays_picks(TODAY)
    assert lines == _old_way_lines(stores)


def test_a_missing_outcome_file_is_an_empty_answer_and_a_locked_one_is_a_failure(stores, tmp_path):
    candidates, outcomes = core.read_scorecard_inputs(
        stores["candidates"], tmp_path / "nope.csv", TODAY
    )
    assert len(candidates) == 2 and outcomes == []

    with pytest.raises(OSError):
        core.read_scorecard_inputs(stores["candidates"], tmp_path, TODAY)  # a directory
