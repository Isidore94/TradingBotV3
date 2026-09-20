"""TJ-14B review round - the three blockers, pinned where they broke.

BLOCKER 1. `_pre_card_journal_pull` (2 days) ran first and
`morning_import_retry` (3 days) second, in the SAME synchronous slot;
`JournalImportService.running` refused the second start and the retry stamped
`last_retry` anyway - so on a Monday whose Friday-night import had failed, the
three-day pull that reaches back to FRIDAY never ran. Rule: ONE card starts AT
MOST ONE import; when the morning catch-up is owed it goes FIRST and the
pre-card pull is skipped; `last_retry` is stamped only when an import STARTED.

BLOCKER 3. The pre-card pull fired on EVERY card, so the 07:00 and 08:00 cards
spent the whole day's cap and no fill after 11:00 ET was ever imported - while
the card went on asking for the `same_session` label this packet exists to make
reachable. Rule: the three attempts are RESERVED for the 09:00 card, the middle
card and the LAST card, read off the session's REAL slot list.

ITEM B. A corrupt persisted tally must be read as empty and rewritten clean,
and NOTHING in the pull path may cost TJ-9's forced trade section.

ITEM D. One card starts at most one import; a DATE-ONLY first fill is never
labelled `same_session`.

The fake import service honours the REAL `running` guard (`pull_recent_questrade`
returns False while an import is in flight), because that guard is what turned
two calls into one refusal. No broker, no model, no live store.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: E402,F401
from tj14b_support import (  # noqa: E402
    EARLY_CLOSE,
    REVIEWED,
    SESSION,
    add_round_trip,
    mark_covered,
    new_store,
    pacific,
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the desk seam is Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


class _RunningAwareService:
    """A stand-in that honours the REAL `running` guard.

    `JournalImportService.pull_recent_questrade` returns False while its
    `QThread` is alive. That refusal is the whole of blocker 1, so a fake that
    always returned True would have hidden it.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.running = False

    def pull_recent_questrade(self, days: int) -> bool:
        if self.running:
            return False
        self.calls.append(int(days))
        self.running = True
        return True

    def finish(self) -> None:
        """The import landed - what the hour between two cards does for real."""
        self.running = False

    def shutdown(self) -> None:
        pass


def _slots(session: date):
    from trade_mentor_schedule import slots_for_session

    return slots_for_session(session)


@pytest.fixture()
def desk(tmp_path, monkeypatch):
    import journal_store as journal_store_module
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)
    window = MainWindow(UiState(workspace_mode="workspace"))
    service = _RunningAwareService()
    monkeypatch.setattr(window, "_journal_import_service", lambda: service)
    window._journal_importer = service
    try:
        yield window, store, service
    finally:
        window.close()


def _run_session(window, service, session: date) -> list[tuple[int, int]]:
    """Drive every real card of `session` and record (hour, days) per START."""
    started: list[tuple[int, int]] = []
    for slot in _slots(session):
        before = len(service.calls)
        window._show_trade_mentor_prompt(slot)
        assert len(service.calls) - before <= 1, (
            f"the {slot.scheduled_at.hour:02d}:00 card started "
            f"{len(service.calls) - before} imports; a card starts at most one"
        )
        if len(service.calls) > before:
            started.append((slot.scheduled_at.hour, service.calls[-1]))
        # An hour passes between two cards, so the import that started has
        # landed by the time the next one is built.
        service.finish()
    return started


# ---------------------------------------------------------------------------
# blocker 3 - the pulls are spaced, not first-come
# ---------------------------------------------------------------------------


def test_the_days_pulls_are_reserved_for_three_spaced_cards(desk):
    """A full session carries six cards and exactly three may pull: 09:00, the
    middle card, and the last one. Before this the 07:00 and 08:00 cards spent
    the day and every afternoon fill went unimported."""
    import mentor_questions

    window, store, service = desk
    # The reviewed session HAS landed, so the morning catch-up is not owed and
    # this measures the pre-card schedule alone.
    mark_covered(store, REVIEWED)

    assert [slot.scheduled_at.hour for slot in _slots(SESSION)] == [7, 8, 9, 10, 11, 12]

    started = _run_session(window, service, SESSION)

    assert started == [
        (9, mentor_questions.PRE_CARD_PULL_DAYS),
        (11, mentor_questions.PRE_CARD_PULL_DAYS),
        (12, mentor_questions.PRE_CARD_PULL_DAYS),
    ], started
    assert len(started) == mentor_questions.PULLS_PER_DAY_CAP


def test_an_early_close_forfeits_the_middle_attempt_and_never_rolls_it_earlier(desk):
    """2026-11-27 closes at 10:00 Pacific and carries 07:00, 08:00, 09:00 and
    the 12:00 D1 slot. There is no card between 09:00 and the last, so that
    attempt is FORFEITED - the 07:00 card does not inherit it."""
    import mentor_questions

    window, store, service = desk
    mark_covered(store, "2026-11-25")

    assert [slot.scheduled_at.hour for slot in _slots(EARLY_CLOSE)] == [7, 8, 9, 12]

    started = _run_session(window, service, EARLY_CLOSE)

    assert started == [
        (9, mentor_questions.PRE_CARD_PULL_DAYS),
        (12, mentor_questions.PRE_CARD_PULL_DAYS),
    ], started


def test_the_reserved_cards_are_read_off_the_sessions_own_slot_list():
    """The schedule is a pure function of the session, so it is testable without
    a desk and an early close is not a special case in the caller."""
    import mentor_questions

    full = _slots(SESSION)
    short = _slots(EARLY_CLOSE)

    assert mentor_questions.pull_slot_ids(SESSION) == (
        full[2].slot_id,
        full[4].slot_id,
        full[5].slot_id,
    )
    assert mentor_questions.pull_slot_ids(EARLY_CLOSE) == (short[2].slot_id, short[3].slot_id)
    assert mentor_questions.pull_slot_ids(date(2026, 9, 19)) == (), "a weekend pulls nothing"


# ---------------------------------------------------------------------------
# blocker 1 - the morning catch-up goes first and is never pre-empted
# ---------------------------------------------------------------------------


def test_the_morning_catch_up_runs_and_is_never_pre_empted_by_the_pre_card_pull(desk):
    """The 09:00 card of a Monday whose Friday-night import failed: the THREE-day
    catch-up starts, and it is the only import that card makes."""
    import trade_mentor_trade_check as check

    window, store, service = desk
    # Nothing covered for the reviewed session - last night failed.
    mark_covered(store, "2026-09-09")

    nine = _slots(SESSION)[2]
    window._show_trade_mentor_prompt(nine)

    assert service.calls == [check.MORNING_RETRY_DAYS], (
        "the three-day catch-up did not run, or the two-day pre-card pull "
        "pre-empted it"
    )


def test_a_refused_catch_up_stays_owed_for_the_next_card(desk):
    """A start refused because something was already running did not do the pull
    this wanted, so the morning is not marked spent."""
    import trade_mentor_trade_check as check

    window, store, service = desk
    mark_covered(store, "2026-09-09")
    service.running = True  # something else is in flight

    window._show_trade_mentor_prompt(_slots(SESSION)[2])
    assert service.calls == [], "nothing could start"

    service.finish()
    window._show_trade_mentor_prompt(_slots(SESSION)[3])

    assert service.calls == [check.MORNING_RETRY_DAYS], "the catch-up was still owed"


def test_the_catch_up_does_not_spend_the_pre_card_cap(desk):
    """They answer different questions and reach different days, so a morning
    that needed a catch-up still gets its three spaced pre-card attempts."""
    import mentor_questions
    import trade_mentor_trade_check as check

    window, store, service = desk
    mark_covered(store, "2026-09-09")

    started = _run_session(window, service, SESSION)
    days = [entry[1] for entry in started]

    assert days.count(check.MORNING_RETRY_DAYS) == 1, "one catch-up a morning"
    assert (
        days.count(mentor_questions.PRE_CARD_PULL_DAYS) == mentor_questions.PULLS_PER_DAY_CAP
    ), started


# ---------------------------------------------------------------------------
# item B - nothing in the pull path costs the forced trade section
# ---------------------------------------------------------------------------


def test_a_corrupt_pull_tally_is_read_as_empty_and_rewritten_clean(desk):
    """`{"pulls": "three"}` is not a number of pulls. A corrupt state file must
    not be able to stop the desk pulling for a day - or to raise in a Qt slot."""
    window, store, service = desk
    mark_covered(store, REVIEWED)
    window.trade_mentor_service.set_pull_tally({"day": SESSION.isoformat(), "pulls": "three"})

    window._show_trade_mentor_prompt(_slots(SESSION)[2])

    assert service.calls, "a corrupt tally silenced the day's pulls"
    tally = window.trade_mentor_service.pull_tally()
    assert tally["pulls"] == 1 and tally["failures"] == 0
    assert tally["day"] == SESSION.isoformat()


def test_a_non_dict_tally_is_survivable(desk):
    window, store, service = desk
    mark_covered(store, REVIEWED)
    window.trade_mentor_service._pull_tally = "not a tally at all"

    window._show_trade_mentor_prompt(_slots(SESSION)[2])

    assert service.calls
    assert window.trade_mentor_service.pull_tally()["pulls"] == 1


def test_an_unimportable_registry_never_costs_the_forced_trade_section(desk, monkeypatch):
    """The trade section is the one thing on this card the trader may not skip.
    It used to be built AFTER the pull, so a raise in the pull path took it."""
    import builtins

    window, store, service = desk
    mark_covered(store, REVIEWED)
    add_round_trip(store, "AAPL", day=REVIEWED)

    real_import = builtins.__import__

    def _no_registry(name, *args, **kwargs):
        if name == "mentor_questions":
            raise ImportError("mentor_questions is gone")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_registry)
    window._show_trade_mentor_prompt(_slots(SESSION)[2])
    monkeypatch.undo()

    card = window.trading_panel.alert_center.chart_review.mentor_card
    assert card._answer_inputs, "the forced trade section was lost to the pull path"


def test_a_corrupt_tally_never_costs_the_forced_trade_section(desk):
    window, store, service = desk
    mark_covered(store, REVIEWED)
    add_round_trip(store, "AAPL", day=REVIEWED)
    window.trade_mentor_service.set_pull_tally({"pulls": "three", "failures": None})

    window._show_trade_mentor_prompt(_slots(SESSION)[2])

    card = window.trading_panel.alert_center.chart_review.mentor_card
    assert card._answer_inputs


# ---------------------------------------------------------------------------
# item D - a broker file is blind to time
# ---------------------------------------------------------------------------


def test_a_date_only_fill_is_never_labelled_same_session(tmp_path):
    """A statement importer writes every date-only fill at MIDNIGHT
    market-local. `trade_session` still names a real DATE for it, so a label
    typed at 11:00 that day would have read `same_session` - a claim that the
    trader labelled the trade before its outcome was known. There is no moment
    to have been before. It reads `recalled_after`, with the reason recorded.
    """
    import trade_mentor_trade_check as check
    import trade_origin

    store = new_store(tmp_path)
    today = SESSION.isoformat()
    mark_covered(store, today)
    trade_id = add_round_trip(store, "AAPL", day=today, entry_hour=7)
    # Rewrite the trade's own stamps the way a broker statement leaves them.
    with store.connection() as conn:
        conn.execute(
            "UPDATE trades SET opened_at = ?, closed_at = ? WHERE trade_id = ?",
            (f"{today}T00:00:00-04:00", f"{today}T00:00:00-04:00", trade_id),
        )

    rows = check.save_answers(
        store,
        trade_id,
        {"stop": {"state": check.ANSWER_NOT_REMEMBERED, "text": ""}},
        now=pacific(SESSION, 11),
    )

    payload = rows[0].get("payload") or {}
    assert trade_origin.first_fill_at(store.get_trade(trade_id)) is None
    assert payload["label_provenance"] == trade_origin.RECALLED_AFTER
    assert payload["recalled_after_session"] is True
    assert payload["label_provenance_reason"] == check.REASON_DATE_ONLY_FILL


def test_a_real_intraday_fill_still_reads_same_session(tmp_path):
    """The other side of the same rule: a fill with a real clock time on today's
    session, labelled today, is still the label made before the outcome."""
    import trade_mentor_trade_check as check
    import trade_origin

    store = new_store(tmp_path)
    today = SESSION.isoformat()
    mark_covered(store, today)
    trade_id = add_round_trip(store, "AAPL", day=today, entry_hour=7)

    rows = check.save_answers(
        store,
        trade_id,
        {"stop": {"state": check.ANSWER_NOT_REMEMBERED, "text": ""}},
        now=pacific(SESSION, 11),
    )

    payload = rows[0].get("payload") or {}
    assert payload["label_provenance"] == trade_origin.SAME_SESSION
    assert payload["recalled_after_session"] is False
    assert payload["label_provenance_reason"] == ""
