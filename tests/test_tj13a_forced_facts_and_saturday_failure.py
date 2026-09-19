"""TJ-13A fix round, points 3 and 4 (lead answers, 2026-09-19).

**Point 3 - a forced daytime `daily_digest` still writes its FACTS.**
`daily_digest` is `uses_model=True` because its second artifact is narrated, so
TJ-13A item 1 made a forced daytime run of it record `skipped` - and that took
away the deterministic fact pack, which is seconds of work and calls no model.
`run_daily_digest` already has the switch for this (`narrate: bool = True`), so
the slot declares the keyword arguments that make its own `run` model-free and
a forced daytime run uses them. The lead's condition was "only if a clean
existing switch exists; do not invent a split" - it exists.

**Point 4 - `weekly_synthesis` runs tonight for the first time in 476 ledger
rows.** It has never executed on the live ledger, so the thing to pin is not
what it produces but that it cannot take the night with it: an arbitrary
exception out of the last slot on the Saturday slate leaves every earlier
slot's ledger row exactly as written, and the run exits as an ordinary job
failure rather than a crash.

NO MODEL IS CALLED HERE.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
SATURDAY_AFTERNOON = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)
SATURDAY_NIGHT = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)

LIVE_START = "01:00"
LIVE_END = "09:00"


def _settings(**values):
    from ai_jobs import store

    return mock.patch.object(
        store._paths(),
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


def _no_session_block(monkeypatch):
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")


def _frozen_clock(monkeypatch, moment):
    """Hold the runner's BETWEEN-JOBS clock still.

    `_run_slots_locked` re-reads `window.market_now()` with no argument after
    every slot, deliberately - sec 6.1 says a long job that crosses the window
    end stops gracefully. That is correct and this packet did not touch it, but
    it means a multi-slot test whose `now=` is only honoured by the first slot
    measures the wall clock for the rest. Freezing it is what makes a slate
    test about the SLATE.
    """
    from ai_jobs import window

    real = window.market_now
    monkeypatch.setattr(window, "market_now", lambda now=None: real(moment))


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# point 3 - the facts survive a forced daytime run
# ---------------------------------------------------------------------------


def _slate_with_spies(calls):
    """The REAL slate with every `run` replaced by a recorder.

    `dataclasses.replace` keeps every other field, which is the point: the
    model-free keywords are declared BESIDE `run` rather than as a second
    callable, so a spy that replaces `run` receives them too and no test can
    accidentally reach the real job through them.
    """
    from ai_jobs import runner

    def _record(name):
        def _run(**kwargs):
            calls.append((name, dict(kwargs)))
            return {"reason": f"{name} ran"}

        return _run

    return [replace(slot, run=_record(slot.name)) for slot in runner.default_slots()]


def test_a_forced_daytime_digest_writes_the_facts_without_the_narration(tmp_path, monkeypatch):
    """The deterministic half is seconds of work and calls no model."""
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    calls: list[tuple[str, dict]] = []
    led = tmp_path / "ledger.jsonl"

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            report = runner.run_slots(
                _slate_with_spies(calls),
                now=SATURDAY_AFTERNOON,
                force=True,
                only="daily_digest",
                ledger_path=led,
            )

    assert [name for name, _kwargs in calls] == ["daily_digest"]
    assert calls[0][1]["narrate"] is False, "no model call by day"
    # The row says what it left out, so a reader of the ledger is never told
    # this was the night's full digest.
    row = _rows(led)[0]
    assert row["status"] != ledger.STATUS_SKIPPED
    assert "narration" in row["reason"]
    assert "daily_digest ran" in row["reason"]
    assert report.results


def test_a_forced_daytime_summary_still_starts_nothing(tmp_path, monkeypatch):
    """Guard: the carve-out is the SWITCH, not the flag.

    `ai_summary` declares no model-free keywords, because there is no summary
    without a model. A forced daytime run of it must still record `skipped` -
    otherwise point 3 would have quietly reopened item 1.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    calls: list[tuple[str, dict]] = []
    led = tmp_path / "ledger.jsonl"

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            report = runner.run_slots(
                _slate_with_spies(calls),
                now=SATURDAY_AFTERNOON,
                force=True,
                only="ai_summary",
                ledger_path=led,
            )

    assert calls == []
    assert [row["status"] for row in report.results] == [ledger.STATUS_SKIPPED]


def test_a_night_run_of_the_digest_still_narrates(tmp_path, monkeypatch):
    """The carve-out is for a FORCED DAYTIME run and nothing else."""
    from ai_jobs import runner, store

    _no_session_block(monkeypatch)
    calls: list[tuple[str, dict]] = []
    led = tmp_path / "ledger.jsonl"

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            runner.run_slots(
                _slate_with_spies(calls),
                now=SATURDAY_NIGHT,
                only="daily_digest",
                ledger_path=led,
            )

    assert [name for name, _kwargs in calls] == ["daily_digest"]
    assert "narrate" not in calls[0][1], "the nightly call is unchanged"


# ---------------------------------------------------------------------------
# point 4 - the first weekly_synthesis cannot take the night with it
# ---------------------------------------------------------------------------


def test_a_failing_weekly_synthesis_leaves_every_earlier_row_intact(tmp_path, monkeypatch):
    """It has never run in 476 ledger rows. Tonight it runs unattended.

    So: the last slot on the Saturday slate raises something nobody predicted,
    and the night must be exactly as complete as it was a moment before, with
    one ordinary `failed` row appended. A crash out of `run_slots` would lose
    the report and, with it, the runner's own account of the night.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    _frozen_clock(monkeypatch, SATURDAY_NIGHT)
    led = tmp_path / "ledger.jsonl"
    ran: list[str] = []

    slate = runner.slots_for("saturday")
    assert slate[-1].name == "weekly_synthesis", "the Saturday slate ends with it"

    class _Unpredicted(Exception):
        """A class no caller of this slot has ever seen."""

    def _boom(**kwargs):
        raise _Unpredicted("something nobody predicted")

    def _ok(name):
        def _run(**kwargs):
            ran.append(name)
            return {"reason": f"{name} ran", "outputs": [f"{name}.json"]}

        return _run

    spied = [
        replace(slot, run=_boom if slot.name == "weekly_synthesis" else _ok(slot.name))
        for slot in slate
    ]

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            report = runner.run_slots(spied, now=SATURDAY_NIGHT, ledger_path=led)

    rows = _rows(led)
    by_job = {row["job"]: row for row in rows}

    # Every earlier slot ran and kept its row, outputs and all.
    assert "journal_import" in ran and "ai_summary" in ran
    assert by_job["journal_import"]["status"] == ledger.STATUS_OK
    assert by_job["ai_summary"]["outputs"] == ["ai_summary.json"]
    assert by_job["daily_digest"]["status"] == ledger.STATUS_OK

    # ...and the failure is one ordinary row, named, with its error.
    failure = by_job["weekly_synthesis"]
    assert failure["status"] == ledger.STATUS_FAILED
    assert "something nobody predicted" in failure["error"]
    assert report.failed == 1
    assert report.ran >= 2
