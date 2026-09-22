"""AI-R3: the overnight sweep uses the canonical BounceBot finalizer safely."""

from __future__ import annotations

import threading
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_outcome_sweep_factory_has_no_constructor_or_broker_side_effects(monkeypatch, tmp_path):
    """The after-close job must use the existing finalizer without starting a scanner.

    A missing checkpoint is an honest empty state.  The factory is deliberately
    driven rather than a hand-built object, because an ``__init__`` call here
    would create IB clients and maintenance threads before the job could decide
    whether there is anything to finalize.
    """
    from bounce_bot_lib import legacy

    checkpoint = tmp_path / "pending_bounce_outcomes.json"
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)

    def constructor_called(*_args, **_kwargs):  # pragma: no cover - the assertion is the guard
        raise AssertionError("the outcome sweep factory called BounceBot.__init__")

    monkeypatch.setattr(legacy.BounceBot, "__init__", constructor_called)

    bot = legacy.BounceBot.for_outcome_sweep()

    assert isinstance(bot, legacy.BounceBot)
    assert bot.pending_bounce_outcomes == {}
    assert bot._finalized_outcome_ids() == {}
    assert bot._finalizing_outcome_ids() == {}
    assert isinstance(bot._pending_lock, type(threading.RLock()))
    assert not checkpoint.exists(), "constructing an empty sweep wrote a checkpoint"


def test_outcome_sweep_factory_refuses_a_bad_checkpoint_without_quarantining_or_rewriting_it(
    monkeypatch, tmp_path
):
    """A scanner may recover a checkpoint; the finalizer must not guess from one."""
    from bounce_bot_lib import legacy

    checkpoint = tmp_path / "pending_bounce_outcomes.json"
    checkpoint.write_text("{not json", encoding="utf-8")
    before = checkpoint.read_bytes()
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)
    monkeypatch.setattr(
        legacy.BounceBot,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("the factory called BounceBot.__init__"),
    )

    with pytest.raises(ValueError):
        legacy.BounceBot.for_outcome_sweep()

    assert checkpoint.read_bytes() == before
    assert list(tmp_path.glob("*.corrupt-*.json")) == []


def test_outcome_sweep_factory_refuses_malformed_state_sections(monkeypatch, tmp_path):
    from bounce_bot_lib import legacy

    checkpoint = tmp_path / "pending_bounce_outcomes.json"
    checkpoint.write_text(json.dumps({"pending": [], "finalized": {}, "finalizing": {}}), encoding="utf-8")
    before = checkpoint.read_bytes()
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)

    with pytest.raises(ValueError, match="pending"):
        legacy.BounceBot.for_outcome_sweep()

    assert checkpoint.read_bytes() == before


def test_outcome_sweep_job_skips_when_disabled_or_the_canonical_sweep_is_too_early():
    """The nightly owner preserves the existing autorun and close+35 gates."""
    from ai_jobs.outcome_sweep import run_outcome_sweep

    calls: list[datetime] = []

    class Bot:
        def sweep_pending_bounce_outcomes(self, *, now, wait_for_scan_window):
            assert wait_for_scan_window is False
            calls.append(now)
            return {"deferred": "scan_window_open", "finalized": 0}

    disabled = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 18, 14, 0),
        bot_factory=Bot,
        autorun_enabled=False,
    )
    early = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 18, 13, 5),
        bot_factory=Bot,
        autorun_enabled=True,
    )
    early_close = run_outcome_sweep(
        session_date="2026-11-27",
        now=datetime(2026, 11, 27, 10, 20),
        bot_factory=Bot,
        autorun_enabled=True,
    )
    after_close = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 18, 14, 0),
        bot_factory=Bot,
        autorun_enabled=True,
    )
    monday_scan = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 21, 13, 5),
        bot_factory=Bot,
        autorun_enabled=True,
    )
    monday_after = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 21, 13, 36),
        bot_factory=Bot,
        autorun_enabled=True,
    )

    assert disabled["status"] == "skipped", disabled
    assert "disabled" in str(disabled["reason"]).lower()
    assert early["status"] == "skipped", early
    assert "close" in str(early["reason"]).lower() or "early" in str(early["reason"]).lower()
    assert early_close["status"] == "skipped", early_close
    assert after_close["status"] == "skipped", after_close
    assert monday_scan["status"] == "skipped", monday_scan
    assert monday_after["status"] == "skipped", monday_after
    assert len(calls) == 2
    assert all(value.tzinfo is not None for value in calls)


def test_outcome_sweep_allows_a_prior_session_before_the_current_session_opens():
    """An overnight run may finish Friday before Monday's scanner starts."""
    from ai_jobs.outcome_sweep import run_outcome_sweep

    calls = []

    class Bot:
        def sweep_pending_bounce_outcomes(self, *, now, wait_for_scan_window):
            assert wait_for_scan_window is False
            calls.append(now)
            return {"finalized": 0}

    outcome = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 21, 5, 0),
        bot_factory=Bot,
        autorun_enabled=True,
    )

    assert outcome["status"] == "ok", outcome
    assert len(calls) == 1 and calls[0].tzinfo is not None


def test_outcome_sweep_fails_when_recovery_cannot_resolve_a_prior_intent():
    """An ambiguous finalizing mark is a failed job even if the sweep is quiet."""
    from ai_jobs.outcome_sweep import run_outcome_sweep

    class Bot:
        def resolve_unfinished_finalizations(self):
            return {"unresolved": 1}

        def sweep_pending_bounce_outcomes(self, *, now, wait_for_scan_window):
            return {"finalized": 0, "failed": 0, "commit_failed": 0}

    outcome = run_outcome_sweep(
        session_date="2026-09-18",
        now=datetime(2026, 9, 18, 14, 0),
        bot_factory=Bot,
        autorun_enabled=True,
    )

    assert outcome["status"] == "failed", outcome
    assert "recovery_unresolved=1" in outcome["reason"]


def test_outcome_sweep_job_marks_a_commit_failure_as_failed_not_ok():
    """A CSV row without its durable checkpoint transaction is not a final."""
    from ai_jobs.outcome_sweep import run_outcome_sweep

    class Bot:
        def sweep_pending_bounce_outcomes(self, *, now, wait_for_scan_window):
            assert wait_for_scan_window is False
            return {
                "finalized": 0,
                "commit_failed": 1,
                "failed": 0,
                "by_terminal_kind": {"swept_measured": 1, "unmeasured": 1},
            }

    outcome = run_outcome_sweep(
        session_date="2026-09-21",
        now=datetime(2026, 9, 21, 14, 0),
        bot_factory=Bot,
        autorun_enabled=True,
    )

    assert outcome["status"] == "failed", outcome
    assert outcome["commit_failed"] == 1
    assert outcome["by_terminal_kind"] == {"swept_measured": 1, "unmeasured": 1}


def test_outcome_sweep_job_finalizes_measured_and_unmeasured_pending_rows_once_across_two_factories(
    monkeypatch, tmp_path
):
    """The job drives two real canonical sweep instances against one temp checkpoint."""
    from ai_jobs.outcome_sweep import run_outcome_sweep
    from bounce_bot_lib import legacy

    checkpoint = tmp_path / "state.json"
    outcomes = tmp_path / "outcomes.csv"
    measured = {
        "event_id": "measured", "symbol": "MEAS", "direction": "long",
        "trade_date": "2026-08-21", "entry_time": "2026-08-21T07:00:00",
        "entry_price": 100.0, "stop_price": 99.0, "risk_per_share": 1.0,
        "target_1r": 101.0, "target_2r": 102.0, "milestones_logged": [],
        "outcome_mode": "eod_hold", "context": {},
        "last_measured": {"bars": 3, "last_close": 100.5, "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2},
    }
    unmeasured = {**measured, "event_id": "unmeasured", "symbol": "NONE"}
    unmeasured.pop("last_measured")
    checkpoint.write_text(
        json.dumps({"pending": {"measured": measured, "unmeasured": unmeasured}, "finalized": {}, "finalizing": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOME_STATE_JSON", checkpoint)
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOMES_CSV", outcomes)
    monkeypatch.setattr(legacy.BounceBot, "_write_outcome_coverage", lambda *_args: None)
    monkeypatch.setattr(legacy.BounceBot, "_mirror_outcome_row_to_ledger", lambda *_args: None)

    first = run_outcome_sweep(
        session_date="2026-08-21", now=datetime(2026, 8, 21, 14, 30),
        bot_factory=legacy.BounceBot.for_outcome_sweep, autorun_enabled=True,
    )
    second = run_outcome_sweep(
        session_date="2026-08-21", now=datetime(2026, 8, 21, 14, 31),
        bot_factory=legacy.BounceBot.for_outcome_sweep, autorun_enabled=True,
    )

    assert first["status"] == "ok", first
    assert first["by_terminal_kind"] == {"measured_eod": 0, "measured_swept": 1, "unmeasured": 1}
    assert second["status"] == "ok", second
    assert second["finalized"] == 0 and second["already_finalized"] == 0
    with outcomes.open(newline="", encoding="utf-8") as handle:
        finals = [row for row in csv.DictReader(handle) if row.get("event_type") == "final"]
    assert [row["event_id"] for row in finals] == ["measured", "unmeasured"]
    assert {row["status"] for row in finals} == {"swept_measured", "unresolved"}


def test_outcome_sweep_job_refuses_weekend_and_a_calendar_failure_before_the_factory(monkeypatch):
    """A job must never name a non-session as covered or sweep on calendar doubt."""
    import market_calendar
    from ai_jobs.outcome_sweep import run_outcome_sweep

    def should_not_construct():
        pytest.fail("the outcome factory ran without a verified exchange session")

    weekend = run_outcome_sweep(
        session_date="2026-08-22", now=datetime(2026, 8, 22, 14, 30),
        bot_factory=should_not_construct, autorun_enabled=True,
    )
    monkeypatch.setattr(market_calendar, "is_session", lambda _day: (_ for _ in ()).throw(RuntimeError("calendar offline")))
    broken = run_outcome_sweep(
        session_date="2026-08-21", now=datetime(2026, 8, 21, 14, 30),
        bot_factory=should_not_construct, autorun_enabled=True,
    )

    assert weekend["status"] == "skipped", weekend
    assert "session" in str(weekend["reason"]).lower()
    assert broken["status"] == "failed", broken
    assert "calendar" in str(broken["reason"]).lower()
