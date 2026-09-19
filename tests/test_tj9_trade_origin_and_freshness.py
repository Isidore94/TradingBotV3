"""TJ-9 items 5 and 6 - planned vs unplanned, and how fresh the fills are.

Written BEFORE the fix and red on `claude/tj9-forced-trade-labels`'s base commit
(`57151d0f`). The builder makes them pass and may only ADD.

WHAT IS PINNED
--------------
* `scripts/trade_origin.planned_state(trade, decisions, claims, focus_adds,
  armed)` is the packet's own signature. It is PURE - the packet gives it four
  already-loaded lanes and no page - so the lanes here are the shapes the real
  stores write, read by each store's OWN stamp key: `created_at` for an
  annotation (`ui/annotations/store.py:329`), `claim_at_utc` for a claimed pick
  (`claimed_picks.ROW_FIELDS`), `joined_at` for a Focus add
  (`focus_picks.py:1056`) and `armed_at` for an armed alert
  (`armed_alert_expiry.py:123`). A reader that only understands one of them
  fails three of these tests.
* **A date-only fill is `unmeasured`, never guessed** - even when a like sits
  in front of it. A Questrade statement row stamps midnight market-local
  (`journal_trade_shape.is_date_only`), and midnight is not a time a fill
  happens at, so "before the first fill" has no meaning for that trade.
* Every "before" case crosses a timezone: the decision is stamped in UTC and
  the fill in Pacific, arranged so comparing the raw ISO strings gives the
  WRONG answer.
* `fills_current_to` is asserted as an exact DATE built from a coverage ledger
  where the newest COVERED day is NOT the newest row - a `MAX(day)` that ignores
  status returns 2026-09-18 and fails.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj9_support import (  # noqa: E402
    REVIEWED,
    SESSION_TODAY,
    add_round_trip,
    mark_covered,
    new_store,
    slot_at,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

PACIFIC = ZoneInfo("America/Los_Angeles")

#: 07:15 Pacific written in UTC - sixteen minutes BEFORE the 07:31 Pacific fill,
#: and sixteen minutes AFTER it if the offsets are thrown away.
BEFORE_ENTRY = datetime(2026, 9, 11, 14, 15, tzinfo=timezone.utc).isoformat()
#: 11:00 Pacific, written in UTC. Comfortably after the entry.
AFTER_ENTRY = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc).isoformat()

FILLED_AT = "2026-09-11T07:31:00-07:00"
#: What a broker statement gives: a date and no clock time.
DATE_ONLY_FILL = "2026-09-11T00:00:00-07:00"


def _trade(opened: str = FILLED_AT, *, side: str = "LONG") -> dict:
    return {
        "trade_id": "T1",
        "symbol": "AAPL",
        "direction": side,
        "opened_at": opened,
        "trade_date": opened[:10],
    }


def _lane(kind: str, *, when: str, symbol: str = "AAPL", side: str = "LONG") -> dict:
    stamp_key = {
        "decisions": "created_at",
        "claims": "claim_at_utc",
        "focus_adds": "joined_at",
        "armed": "armed_at",
    }[kind]
    row = {"symbol": symbol, "side": side, stamp_key: when}
    if kind == "decisions":
        row["event_type"] = "like_claim"
    return row


def _lanes(**named):
    """The four positional lanes, with the named one filled."""
    return [named.get(name, []) for name in ("decisions", "claims", "focus_adds", "armed")]


# ---------------------------------------------------------------------------
# Item 5 - planned vs unplanned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", ["decisions", "claims", "focus_adds", "armed"])
def test_any_of_the_four_lanes_before_the_first_fill_makes_the_trade_planned(lane):
    """A like, a claim, a Focus add or an armed alert on that name and side,
    stamped before the first fill. Four lanes, four stamp keys, one verdict."""
    from trade_origin import planned_state

    rows = [_lane(lane, when=BEFORE_ENTRY)]
    assert planned_state(_trade(), *_lanes(**{lane: rows})) == "planned"


@pytest.mark.parametrize("lane", ["decisions", "claims", "focus_adds", "armed"])
def test_a_decision_stamped_after_the_first_fill_leaves_the_trade_unplanned(lane):
    """A trade from nowhere that the trader liked at 11:00 is still a trade from
    nowhere. Liking it afterwards is a different fact."""
    from trade_origin import planned_state

    rows = [_lane(lane, when=AFTER_ENTRY)]
    assert planned_state(_trade(), *_lanes(**{lane: rows})) == "unplanned"


def test_a_trade_with_nothing_in_front_of_it_is_unplanned():
    from trade_origin import planned_state

    assert planned_state(_trade(), [], [], [], []) == "unplanned"


def test_the_other_side_of_the_same_name_is_not_a_plan():
    """A LONG like says nothing about a SHORT entry - the same rule the claimed
    picks store keeps on `(symbol, side)`."""
    from trade_origin import planned_state

    rows = [_lane("claims", when=BEFORE_ENTRY, side="LONG")]
    assert planned_state(_trade(side="SHORT"), *_lanes(claims=rows)) == "unplanned"


def test_another_symbol_is_not_a_plan():
    from trade_origin import planned_state

    rows = [_lane("claims", when=BEFORE_ENTRY, symbol="MSFT")]
    assert planned_state(_trade(), *_lanes(claims=rows)) == "unplanned"


def test_a_date_only_fill_is_unmeasured_and_never_guessed():
    """The broker file is authoritative for money and BLIND TO TIME: it stamps
    midnight market-local. "Before the first fill" cannot be decided against a
    time that is not a time, so the answer is `unmeasured` even though a claim
    sits in front of it - and it is never `planned` and never `unplanned`."""
    from trade_origin import planned_state

    rows = [_lane("claims", when=BEFORE_ENTRY)]
    assert planned_state(_trade(DATE_ONLY_FILL), *_lanes(claims=rows)) == "unmeasured"
    assert planned_state(_trade(DATE_ONLY_FILL), [], [], [], []) == "unmeasured"


def test_a_trade_with_no_readable_first_fill_is_unmeasured():
    """Uncertainty is never a verdict. A trade whose opening stamp cannot be
    read is `unmeasured`, not `unplanned`."""
    from trade_origin import planned_state

    blank = {"trade_id": "T1", "symbol": "AAPL", "direction": "LONG", "opened_at": "", "trade_date": ""}
    assert planned_state(blank, *_lanes(claims=[_lane("claims", when=BEFORE_ENTRY)])) == "unmeasured"


def test_the_three_planned_states_are_named_constants():
    import trade_origin

    assert trade_origin.PLANNED == "planned"
    assert trade_origin.UNPLANNED == "unplanned"
    assert trade_origin.UNMEASURED == "unmeasured"
    assert trade_origin.PLANNED_STATES == ("planned", "unplanned", "unmeasured")


# ---------------------------------------------------------------------------
# Item 6 - journal freshness
# ---------------------------------------------------------------------------


def test_fills_current_to_names_the_last_session_with_verified_coverage(tmp_path):
    """The newest COVERED day, not the newest ROW. 2026-09-18 is present and
    FAILED, and 2026-09-19 is present and NO_SESSION; a `MAX(day)` that ignores
    status answers with one of those and is wrong about what the trader has."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    assert check.fills_current_to(store) is None, "no ledger is not a date"

    mark_covered(store, "2026-09-16")
    mark_covered(store, "2026-09-17")
    mark_covered(store, "2026-09-18", status="FAILED")
    mark_covered(store, "2026-09-19", status="NO_SESSION")

    assert check.fills_current_to(store) == date(2026, 9, 17)


def test_a_not_ready_task_carries_the_date_the_fills_are_current_to(tmp_path):
    """The task, not the widget, knows the date - so the AWAY digest and the
    Journal can print the same line without rebuilding it."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, "2026-09-10")
    # The reviewed session 2026-09-11 is NOT covered, so the check is not ready.
    task = check.build_task(store, SESSION_TODAY)

    assert task.journal_ready is False
    assert task.reason == check.REASON_NOT_READY
    assert task.fills_current_to == "2026-09-10"


def test_the_card_says_journal_not_ready_with_the_date_and_the_section_rides(tmp_path):
    """"The card SAYS so and the section rides to the next slot instead of
    asking nothing all day." Today the not-ready line names no date and the next
    slot wipes it."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, "2026-09-10")

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card.show_slot(slot_at(SESSION_TODAY, 9))
    card.set_trade_check(check.build_task(store, SESSION_TODAY), store=store)

    text = card.trade_check_label.text()
    assert f"{check.REASON_NOT_READY} - fills current to 2026-09-10" in text
    assert card.trade_check_label.isVisibleTo(card) is True

    card.show_slot(slot_at(SESSION_TODAY, 10))
    assert card.trade_check_label.isVisibleTo(card) is True, "not ready rides, it does not vanish"
    assert f"{check.REASON_NOT_READY} - fills current to 2026-09-10" in card.trade_check_label.text()


def test_a_ready_task_still_reports_how_fresh_the_fills_are(tmp_path):
    """Freshness is printed whether or not it is a problem (answer 27: the
    report says how fresh it is)."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    add_round_trip(store, "AAPL")

    task = check.build_task(store, SESSION_TODAY)

    assert task.journal_ready is True
    assert task.fills_current_to == REVIEWED
