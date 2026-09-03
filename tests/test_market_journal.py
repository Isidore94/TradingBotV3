"""R10.H - the Market Journal store and its two surfaces.

R10.G records what the machine saw; this is what the trader saw, in their own
words. The two together are what lets a later reader understand a session
rather than re-derive it.

The load-bearing rule is decision record §5a: an entry written in the evening
about an AWAY session carries `session_date` = the session and `created_at` =
when it was actually written, tz-aware, **never backdated**. A reader weighing
"what did you think at the time?" needs to know it was not written at the time.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_journal as mj  # noqa: E402


# ==========================================================================
# the entry
# ==========================================================================
def test_an_entry_about_a_past_session_is_flagged_and_never_backdated():
    """Decision record §5a. On an AWAY day the trader writes that evening."""
    written = datetime(2026, 8, 24, 19, 30, tzinfo=timezone.utc)
    entry = mj.build_entry(
        text="Chop all day; I would have sat on my hands.",
        session_date="2026-08-21",
        now=written,
    )

    assert entry["session_date"] == "2026-08-21"
    assert entry["created_at"].startswith("2026-08-24")
    assert entry["written_after_the_session"] is True


def test_an_entry_written_DURING_the_session_is_not_flagged():
    """R4 A17 sharpened this: the boundary is the CLOSE, not the calendar date.

    The old version used "now", so on a machine whose clock happened to be past
    16:00 ET it was asserting the opposite of what it meant - and it could never
    have caught a 21:00 Pacific note claiming to have been written during a
    session that had already shut.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    entry = mj.build_entry(
        text="Opening drive held.",
        session_date="2026-09-02",
        now=datetime(2026, 9, 2, 11, 15, tzinfo=et),
    )
    assert entry["written_after_the_session"] is False


def test_an_entry_written_after_the_close_is_flagged_on_the_same_day():
    """A note typed at 21:00 Pacific is five hours after the market shut.

    Under the old calendar-date rule it read as written DURING the session,
    which is the one thing this field exists to deny.
    """
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("America/Los_Angeles")
    for hour in (21, 22):
        now = datetime(2026, 9, 2, hour, 0, tzinfo=pacific)
        session = mj.session_date_for(now)
        entry = mj.build_entry(text="late thought", session_date=session, now=now)

        assert session == "2026-09-02", hour
        assert entry["written_after_the_session"] is True, hour


def test_the_after_the_fact_flag_is_computed_never_claimed():
    """A caller cannot set it wrongly, because a caller cannot set it at all."""
    entry = mj.build_entry(
        text="x", session_date="2026-08-21", now=datetime(2026, 8, 24, tzinfo=timezone.utc)
    )
    assert "written_after_the_session" in entry
    with pytest.raises(TypeError):
        mj.build_entry(
            text="x", session_date="2026-08-21", written_after_the_session=False
        )


def test_both_stamps_are_timezone_aware():
    entry = mj.build_entry(text="x", session_date="2026-08-21")
    assert entry["created_at"].endswith("+00:00")


def test_an_empty_entry_is_refused_rather_than_stored():
    """A journal full of blanks makes the record look denser than the thinking
    behind it."""
    ok, reason = mj.is_publishable(mj.build_entry(text="   ", session_date="2026-08-21"))
    assert ok is False
    assert "not a thought" in reason


def test_an_unfiled_entry_is_refused():
    ok, reason = mj.is_publishable(mj.build_entry(text="something", session_date=""))
    assert ok is False
    assert "could never be read back" in reason


def test_an_unknown_timeframe_falls_back_rather_than_being_stored_raw():
    assert mj.build_entry(text="x", session_date="2026-08-21", timeframe="M3")["timeframe"] == "M5"
    assert mj.build_entry(text="x", session_date="2026-08-21", timeframe="d1")["timeframe"] == "D1"


# ==========================================================================
# corrections supersede; nothing is rewritten
# ==========================================================================
def test_a_correction_is_a_new_entry_naming_the_one_it_replaces():
    """Ground rule 5. A journal that can be quietly rewritten is not evidence
    about what anyone believed - it is evidence about what they will admit now."""
    original = mj.build_entry(text="I think this holds.", session_date="2026-08-21")
    fixed = mj.supersede(original, text="I thought this held; it did not.")

    assert fixed["supersedes"] == original["entry_id"]
    assert fixed["entry_id"] != original["entry_id"]
    assert fixed["session_date"] == original["session_date"]


def test_the_superseded_entry_is_hidden_on_read_but_still_on_disk():
    original = mj.build_entry(text="first", session_date="2026-08-21")
    fixed = mj.supersede(original, text="second")

    current = mj.resolve_entries([original, fixed])

    assert [row["text"] for row in current] == ["second"]
    # Both rows were handed in; only the view narrowed.
    assert len(mj.resolve_entries([original, fixed])) == 1


def test_entries_read_back_in_session_then_write_order():
    early = mj.build_entry(
        text="a", session_date="2026-08-20", now=datetime(2026, 8, 20, 14, tzinfo=timezone.utc)
    )
    late = mj.build_entry(
        text="b", session_date="2026-08-20", now=datetime(2026, 8, 20, 18, tzinfo=timezone.utc)
    )
    other = mj.build_entry(
        text="c", session_date="2026-08-21", now=datetime(2026, 8, 21, 14, tzinfo=timezone.utc)
    )
    assert [row["text"] for row in mj.resolve_entries([late, other, early])] == ["a", "b", "c"]


# ==========================================================================
# the agreement rate
# ==========================================================================
def test_a_session_the_trader_never_overrode_counts_as_agreement():
    shifts = [
        {"session_date": "2026-08-20", "source": "auto", "to_regime": "bullish_strong"},
        {"session_date": "2026-08-21", "source": "auto", "to_regime": "neutral_chop"},
    ]
    result = mj.agreement_rate(shifts)
    assert result["sessions_compared"] == 2
    assert result["rate"] == 1.0


def test_an_override_to_a_different_regime_is_a_disagreement():
    shifts = [
        {"session_date": "2026-08-20", "source": "auto", "to_regime": "bullish_strong"},
        {"session_date": "2026-08-20", "source": "user", "to_regime": "bearish_weak"},
        {"session_date": "2026-08-21", "source": "auto", "to_regime": "neutral_chop"},
    ]
    result = mj.agreement_rate(shifts)
    assert result["sessions_compared"] == 2
    assert result["sessions_agreed"] == 1
    assert result["rate"] == 0.5


def test_a_session_with_no_auto_read_is_not_counted_either_way():
    """There was nothing to agree with."""
    shifts = [{"session_date": "2026-08-20", "source": "user", "to_regime": "bearish_weak"}]
    assert mj.agreement_rate(shifts)["sessions_compared"] == 0


def test_no_comparable_session_is_unmeasured_not_a_hundred_percent():
    result = mj.agreement_rate([])
    assert result["rate"] is None
    assert "UNMEASURED" in result["note"]


def test_the_schema_is_named_never_numbered():
    assert mj.SCHEMA_MARKET_JOURNAL_ENTRY == "market_journal_entry_v1"


# ==========================================================================
# the one writer
# ==========================================================================
@pytest.fixture
def service(monkeypatch):
    pytest.importorskip("PySide6")
    from ui.services.market_journal_service import MarketJournalService

    written: list[dict] = []

    class _Stream:
        def append(self, event, **_kwargs):
            written.append(event)
            return dict(event)

        def read(self):
            return type("R", (), {"rows": tuple(written)})()

    instance = MarketJournalService()
    instance._ledger = _Stream()
    instance.written = written
    return instance


def test_the_service_refuses_an_empty_entry_without_raising(service):
    """Both hosts show the refusal in a status line; an exception here would
    turn "you typed nothing" into a traceback."""
    result = service.write_entry(text="  ", session_date="2026-08-21")

    assert result["ok"] is False
    assert service.written == []


def test_a_failed_write_is_reported_as_failed_never_as_saved(service, monkeypatch):
    """A capture that did not reach disk must never look like one that did -
    the trader would believe the record holds a thought it does not."""
    def _explode(event, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(service._ledger, "append", _explode)
    result = service.write_entry(text="real thought", session_date="2026-08-21")

    assert result["ok"] is False
    assert "disk full" in result["reason"]


def test_both_surfaces_write_through_the_same_service(service):
    service.write_entry(text="from the desk tab", session_date="2026-08-21", origin="desk_tab")
    service.write_entry(text="from the page", session_date="2026-08-21", origin="journal_page")

    origins = {row["origin"] for row in service.written}
    assert origins == {"desk_tab", "journal_page"}
    assert len(service.entries_for("2026-08-21")) == 2


def test_reading_filters_by_session(service):
    service.write_entry(text="a", session_date="2026-08-20")
    service.write_entry(text="b", session_date="2026-08-21")
    assert [row["text"] for row in service.entries_for("2026-08-21")] == ["b"]


def test_an_absent_day_context_is_reported_absent_not_invented(service, monkeypatch):
    """A session the desk never measured has no row, and inventing one here
    would defeat the point of never fabricating it in R10.G."""
    monkeypatch.setattr(
        "evidence_ledger.EvidenceLedger",
        lambda **kwargs: type("S", (), {"read": lambda _self: type("R", (), {"rows": ()})()})(),
    )
    result = service.day_context("2026-08-21")

    assert result["measured"] is False
    assert "did not measure it" in result["reason"]
