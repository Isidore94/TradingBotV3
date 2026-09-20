"""TJ-14B item 4 - one light journal pull before each card, and its cap.

RED BEFORE THE FIX. On `e8c04f88` the ONLY day-time pull is TJ-9's
`morning_import_retry`, which runs at most once a morning and only when the
PREVIOUS session's statement has not landed; an ordinary 11:00 card pulls
nothing at all. There is no per-day cap and no per-day failure count anywhere
in the repo (`grep -rn "failure cap\\|failures per day" scripts/` is empty), so
the policy this pins has to be built.

THE CONTRACT THESE PIN (plan.md 12.4 TJ-14 item 4)
--------------------------------------------------
* *"Before each card ONE light journal pull on a worker"* - the desk calls the
  import service, which owns its own `QThread` and is the single caller of the
  Questrade refresh chain. Nothing here refreshes a token.
* *"at most a fixed small number a day"* - fewer than the six cards a normal
  session carries, or the cap is not a cap.
* *"never if the day's failure cap is reached"*.
* *"A pull never blocks or delays a card."* A service that raises, refuses or
  is already running costs the card nothing.

The policy is a PURE function so it can be tested without a broker: the caller
persists the tally, exactly as TJ-9's `morning_import_retry` hands back
`last_retry`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

# TJ-14B gives the desk's day-time journal pulls ONE persisted per-day tally;
# this gives each test in this module its own day of it. See the module.
from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: E402,F401
from tj14b_support import (  # noqa: E402
    REVIEWED,
    SESSION,
    mark_covered,
    new_store,
    slot_at,
)

TODAY = SESSION.isoformat()
YESTERDAY = REVIEWED


class _Service:
    """A stand-in for `JournalImportService`. Never a broker, never a thread."""

    def __init__(self, *, started: bool = True, raises: bool = False) -> None:
        self.calls: list[int] = []
        self._started = started
        self._raises = raises

    def pull_recent_questrade(self, days: int) -> bool:
        self.calls.append(int(days))
        if self._raises:
            raise RuntimeError("Questrade refused the refresh")
        return self._started


def test_the_pre_card_pull_is_capped_below_a_full_session_of_cards():
    """A normal session carries six cards. A cap of six is no cap at all."""
    import mentor_questions
    from trade_mentor_schedule import slots_for_session

    assert len(slots_for_session(SESSION)) == 6
    assert 0 < mentor_questions.PULLS_PER_DAY_CAP < 6
    assert 0 < mentor_questions.PULL_FAILURES_PER_DAY_CAP <= mentor_questions.PULLS_PER_DAY_CAP


def test_every_card_of_a_session_together_spend_at_most_the_days_cap():
    """Six cards, one tally threaded through them, and the cap holds."""
    import mentor_questions

    service = _Service()
    tally: dict = {}
    for _card in range(6):
        outcome = mentor_questions.pre_card_pull(service, today=TODAY, tally=tally)
        tally = outcome["tally"]

    assert len(service.calls) == mentor_questions.PULLS_PER_DAY_CAP
    assert outcome["pulled"] is False
    assert "cap" in outcome["reason"].lower()


def test_a_failed_pull_is_counted_and_never_raises():
    """The chain is single-use. A failure has to be a number somebody can stop
    on, not an exception thrown over the card the trader is looking at."""
    import mentor_questions

    service = _Service(raises=True)

    outcome = mentor_questions.pre_card_pull(service, today=TODAY, tally={})

    assert outcome["pulled"] is False
    assert outcome["tally"]["failures"] == 1
    assert service.calls == [mentor_questions.PRE_CARD_PULL_DAYS]


def test_the_days_failures_stop_the_pull_even_with_pulls_left():
    """*"never if the day's failure cap is reached"*. The token chain is worth
    more than one more attempt at today's fills."""
    import mentor_questions

    service = _Service(raises=True)
    tally: dict = {}
    for _attempt in range(mentor_questions.PULL_FAILURES_PER_DAY_CAP):
        tally = mentor_questions.pre_card_pull(service, today=TODAY, tally=tally)["tally"]

    spent = len(service.calls)
    outcome = mentor_questions.pre_card_pull(service, today=TODAY, tally=tally)

    assert outcome["pulled"] is False
    assert len(service.calls) == spent, "a pull was attempted past the failure cap"
    assert "fail" in outcome["reason"].lower()


def test_a_new_day_resets_the_tally():
    """The cap is per DAY, so yesterday's spent attempts never silence today."""
    import mentor_questions

    service = _Service()
    tally: dict = {}
    for _card in range(6):
        tally = mentor_questions.pre_card_pull(service, today=YESTERDAY, tally=tally)["tally"]
    spent = len(service.calls)

    outcome = mentor_questions.pre_card_pull(service, today=TODAY, tally=tally)

    assert outcome["pulled"] is True
    assert len(service.calls) == spent + 1
    assert outcome["tally"]["day"] == TODAY


def test_a_service_already_running_spends_the_attempt_and_is_not_an_error():
    """The pull it is already doing is the pull this wanted - the same rule
    TJ-9's morning retry settled on."""
    import mentor_questions

    service = _Service(started=False)

    outcome = mentor_questions.pre_card_pull(service, today=TODAY, tally={})

    assert outcome["pulled"] is False
    assert outcome["tally"]["failures"] == 0
    assert outcome["tally"]["pulls"] == 1


# ---------------------------------------------------------------------------
# the live seam
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6", reason="the desk seam is Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture()
def desk(tmp_path, monkeypatch):
    import journal_store as journal_store_module
    from ui.app import MainWindow
    from ui.state import UiState

    QApplication.instance() or QApplication([])
    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    monkeypatch.setattr(journal_store_module, "JournalStore", lambda *a, **k: store)

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        yield window, store
    finally:
        window.close()


@pytest.mark.qt
def test_an_ordinary_card_pulls_the_journal_before_it_goes_up(desk, monkeypatch):
    """*"Before each card"* - not only the 09:00 one, and not only when last
    night failed. A fill from this morning is only askable if somebody looked."""
    window, _store = desk
    service = _Service()
    monkeypatch.setattr(window, "_journal_import_service", lambda: service)
    window._journal_importer = service

    window._show_trade_mentor_prompt(slot_at(SESSION, 11))

    assert service.calls, "the 11:00 card pulled nothing"


@pytest.mark.qt
def test_a_failed_pull_leaves_the_card_on_time(desk, monkeypatch):
    """The read is what the trader is being interrupted for. A broker that says
    no never costs it."""
    window, _store = desk
    service = _Service(raises=True)
    monkeypatch.setattr(window, "_journal_import_service", lambda: service)
    window._journal_importer = service

    window._show_trade_mentor_prompt(slot_at(SESSION, 11))

    card = window.trading_panel.alert_center.chart_review.mentor_card
    assert service.calls, "the failing pull was never attempted"
    assert card.isVisibleTo(card.parentWidget() or card)
    assert card._slot is not None and card._slot.scheduled_at.hour == 11
