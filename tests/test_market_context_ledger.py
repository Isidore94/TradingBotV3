"""R10.G - the machine's half of the day's record.

Audit C2: `market_environment_annotations.jsonl` **does not exist at all**, so
the regime the desk was operating under was unrecorded and unrecoverable. Ask
"what was the tape doing the week those setups failed?" and the only answer
available is a re-derivation from bars months later - which is a different
measurement from the one the desk actually acted on.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_context_ledger as mcl  # noqa: E402


# ==========================================================================
# regime shifts (C2)
# ==========================================================================
def test_an_auto_shift_and_a_manual_override_are_both_recorded_and_distinguishable():
    """The difference between what the machine thought and what the trader
    forced IS the agreement rate R10.H's timeline shows, and an agreement rate
    needs the disagreements."""
    auto = mcl.regime_shift_event(
        from_regime="neutral_chop", to_regime="bullish_strong",
        source=mcl.SOURCE_AUTO, session_date="2026-08-24",
    )
    manual = mcl.regime_shift_event(
        from_regime="bullish_strong", to_regime="bearish_weak",
        source=mcl.SOURCE_USER, session_date="2026-08-24",
    )

    assert auto["source"] == "auto" and manual["source"] == "user"
    assert auto["from_regime"] == "neutral_chop"
    assert manual["to_regime"] == "bearish_weak"


def test_an_unmeasurable_spy_move_is_absent_never_zero():
    row = mcl.regime_shift_event(
        from_regime="a", to_regime="b", source="auto", session_date="2026-08-24",
        spy_day_pct="",
    )
    assert row["spy_day_pct"] is None


def test_the_regime_shift_rides_the_one_setter(monkeypatch):
    """Emitted from where the change actually happens, so nothing can change
    the regime without the row."""
    from bounce_bot_lib import legacy

    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    import threading

    bot.market_environment_lock = threading.Lock()
    bot.market_environment = "neutral_chop"
    bot.market_environment_user_override = False
    bot._refresh_rrs_gui = lambda **kwargs: None
    recorded: list = []
    monkeypatch.setattr(
        legacy, "_record_regime_shift",
        lambda previous, env, source: recorded.append((previous, env, source)),
    )

    bot.set_market_environment("bullish_strong", source="auto")
    assert recorded == [("neutral_chop", "bullish_strong", "auto")]

    # Setting the same value again is not a shift and must not fabricate one.
    bot.set_market_environment("bullish_strong", source="auto")
    assert len(recorded) == 1


def test_a_ledger_failure_never_costs_the_regime_change(monkeypatch):
    """Evidence about a decision must not be able to cost the decision."""
    from bounce_bot_lib import legacy

    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    import threading

    bot.market_environment_lock = threading.Lock()
    bot.market_environment = "neutral_chop"
    bot.market_environment_user_override = False
    bot._refresh_rrs_gui = lambda **kwargs: None
    monkeypatch.setattr(
        "evidence_ledger.EvidenceLedger",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("ledger down")),
    )

    bot.set_market_environment("bearish_weak", source="auto")
    assert bot.get_market_environment() == "bearish_weak"


# ==========================================================================
# the daily context row
# ==========================================================================
def test_a_row_written_after_the_fact_says_so():
    """A row written on Tuesday about Monday is a different kind of evidence
    from one written while Monday was still happening."""
    row = mcl.daily_context_row(
        session_date="2026-08-21",
        measured={"spy_close": 500.0},
        completed_late=True,
        completed_at="2026-08-24T07:00:00-07:00",
    )

    assert row["completed_late"] is True
    assert "weaker evidence" in row["late_note"]


def test_a_timely_row_carries_no_late_note():
    row = mcl.daily_context_row(session_date="2026-08-21", measured={"spy_close": 500.0})
    assert row["completed_late"] is False
    assert "late_note" not in row


def test_only_measured_fields_reach_the_row():
    """Nothing fills a blank with a plausible number: a reader can tell "the
    desk did not measure this" from "the desk measured zero" (ground rule 6)."""
    row = mcl.daily_context_row(session_date="2026-08-21", measured={"spy_close": 500.0})
    assert row["spy_close"] == 500.0
    assert "spy_day_pct" not in row


def test_the_row_waits_for_close_plus_grace():
    close = datetime(2026, 8, 21, 16, 0, tzinfo=timezone.utc)
    assert mcl.context_due(close + timedelta(minutes=10), close) is False
    assert mcl.context_due(close + timedelta(minutes=40), close) is True


def test_a_naive_clock_is_normalized_by_attaching_never_stripping():
    """CLAUDE.md, `_gate_moment`: stripping the aware side ends the comparison
    error and keeps the wrong answer."""
    close = datetime(2026, 8, 21, 16, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 8, 21, 16, 40)
    assert mcl.context_due(naive, close) is True


def test_a_session_nobody_measured_gets_no_row():
    """Never fabricated. The gap IS the evidence."""
    sessions = [date(2026, 8, 19), date(2026, 8, 20), date(2026, 8, 21)]
    missing = mcl.missing_sessions({"2026-08-19"}, sessions, through=date(2026, 8, 21))
    assert missing == ["2026-08-20", "2026-08-21"]


def test_sessions_after_the_cutoff_are_not_reported_missing():
    sessions = [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24)]
    missing = mcl.missing_sessions(set(), sessions, through=date(2026, 8, 21))
    assert missing == ["2026-08-20", "2026-08-21"]


# ==========================================================================
# the calendar overlay
# ==========================================================================
def test_the_shipped_overlay_covers_the_current_year():
    overlay = mcl.load_calendar_overlay()
    coverage = mcl.calendar_coverage(overlay, today=date(2026, 8, 24))

    assert overlay["status"] == mcl.STATUS_OK
    assert coverage["covered"] is True
    assert coverage["status"] == "ok"


def test_an_uncovered_year_is_degraded_and_says_which_years_it_has():
    """Silently falling through to the computed rules would be the same failure
    the daily-bar store had: a value that looks right until the year it isn't."""
    overlay = mcl.load_calendar_overlay()
    coverage = mcl.calendar_coverage(overlay, today=date(2031, 3, 1))

    assert coverage["covered"] is False
    assert coverage["status"] == mcl.STATUS_DEGRADED
    assert "does not cover 2031" in coverage["note"]
    assert "2026" in coverage["note"]


def test_a_missing_overlay_is_degraded_not_silently_fine(tmp_path):
    overlay = mcl.load_calendar_overlay(tmp_path / "nope.json")
    coverage = mcl.calendar_coverage(overlay, today=date(2026, 8, 24))

    assert overlay["status"] == mcl.STATUS_ABSENT
    assert coverage["status"] == mcl.STATUS_DEGRADED
    assert "cannot know about an unscheduled close" in overlay["note"]


def test_an_unreadable_overlay_is_degraded_and_says_coverage_is_unknown(tmp_path):
    path = tmp_path / "market_calendar.json"
    path.write_text("{not json", encoding="utf-8")
    overlay = mcl.load_calendar_overlay(path)

    assert overlay["status"] == mcl.STATUS_DEGRADED
    assert "UNKNOWN" in overlay["note"]


def test_the_overlay_parses_holidays_and_early_closes():
    overlay = mcl.load_calendar_overlay()

    holidays = mcl.overlay_holidays(overlay, 2026)
    early = mcl.overlay_early_closes(overlay, 2026)

    assert date(2026, 7, 3) in early and early[date(2026, 7, 3)] == time(13, 0)
    assert holidays, "2026 declares holidays"
    assert all(isinstance(day, date) for day in holidays)


def test_an_unparseable_entry_is_skipped_not_guessed(tmp_path):
    path = tmp_path / "market_calendar.json"
    path.write_text(
        json.dumps(
            {
                "years": ["2026"],
                "holidays": {"2026": ["2026-01-01", "not-a-date"]},
                "early_closes": {"2026": {"2026-07-03": "13:00", "bad": "nope"}},
            }
        ),
        encoding="utf-8",
    )
    overlay = mcl.load_calendar_overlay(path)

    assert mcl.overlay_holidays(overlay, 2026) == {date(2026, 1, 1)}
    assert list(mcl.overlay_early_closes(overlay, 2026)) == [date(2026, 7, 3)]


def test_the_schemas_are_named_never_numbered():
    assert mcl.SCHEMA_DAILY_MARKET_CONTEXT == "daily_market_context_v1"
    assert mcl.SCHEMA_MARKET_REGIME_SHIFT == "market_regime_shift_v1"
