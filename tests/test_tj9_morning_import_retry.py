"""TJ-9 item 6's retry - BUILDER-ADDED, on top of the tester's red suite.

The packet's lead decision: the ONE morning retry reuses
`ui/services/journal_import_service.JournalImportService` - its own `QThread`,
Questrade only, already the desk's single refresh-chain owner - and the builder
adds a test with a FAKE service. Nothing here touches a broker, a token or a
network: the fake records the call and nothing else, which is the whole point.

What is pinned:

* a READY journal retries nothing - the retry exists for the night that ended
  without an OK import, not for every card;
* at most ONCE per morning, keyed on the date the caller passes, so a card
  rebuilt at 10:00 does not queue a second pull behind the 09:00 one;
* a service already running spends the attempt rather than queueing;
* a service that RAISES never costs the card;
* the days asked for are `MORNING_RETRY_DAYS`, read by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _FakeImportService:
    """Records the pull. Never a broker, never a token, never a thread."""

    def __init__(self, *, started: bool = True, boom: bool = False) -> None:
        self.calls: list[int] = []
        self._started = started
        self._boom = boom

    def pull_recent_questrade(self, days: int) -> bool:
        self.calls.append(int(days))
        if self._boom:
            raise RuntimeError("the broker session is gone")
        return self._started


def _task(ready: bool, *, fills_current_to: str = "2026-09-10"):
    from trade_mentor_trade_check import REASON_NOT_READY, TradeCheckTask

    return TradeCheckTask(
        reviewed_session="2026-09-11",
        journal_ready=ready,
        reason="" if ready else REASON_NOT_READY,
        fills_current_to=fills_current_to,
    )


def test_a_not_ready_morning_pulls_questrade_once_through_the_existing_service():
    import trade_mentor_trade_check as check

    service = _FakeImportService()

    outcome = check.morning_import_retry(service, _task(False), today="2026-09-14")

    assert outcome["retried"] is True
    assert outcome["last_retry"] == "2026-09-14"
    assert service.calls == [check.MORNING_RETRY_DAYS]


def test_a_ready_journal_pulls_nothing():
    """The retry is for the night that failed, not for every card."""
    import trade_mentor_trade_check as check

    service = _FakeImportService()

    outcome = check.morning_import_retry(service, _task(True), today="2026-09-14")

    assert outcome["retried"] is False
    assert service.calls == []


def test_the_second_card_of_the_same_morning_does_not_pull_again():
    """"At most once per morning". The 10:00 card inherits the 09:00 card's
    answer rather than starting a second import behind it."""
    import trade_mentor_trade_check as check

    service = _FakeImportService()

    first = check.morning_import_retry(service, _task(False), today="2026-09-14")
    second = check.morning_import_retry(
        service, _task(False), today="2026-09-14", last_retry=first["last_retry"]
    )

    assert second["retried"] is False
    assert service.calls == [check.MORNING_RETRY_DAYS], "one pull, not two"

    # The NEXT morning is a new night's failure and does pull.
    third = check.morning_import_retry(
        service, _task(False), today="2026-09-15", last_retry=second["last_retry"]
    )
    assert third["retried"] is True
    assert len(service.calls) == 2


def test_an_import_already_running_leaves_the_morning_retry_still_owed():
    """AMENDED by TJ-14B's review (blocker 1, lead rule): `last_retry` is
    stamped ONLY when an import actually STARTED.

    It used to be stamped here too - "the attempt is spent either way". That was
    safe while this was the only day-time pull. It stopped being safe the moment
    a second pull could be in flight: TJ-14B's pre-card pull started first, this
    one was REFUSED by `JournalImportService.running`, and the morning was
    marked spent - so a Monday whose Friday-night import had failed never made
    the three-day pull that reaches back to Friday. A refused start did not do
    the pull this wanted, so the retry stays OWED for the next card.
    """
    import trade_mentor_trade_check as check

    service = _FakeImportService(started=False)

    outcome = check.morning_import_retry(service, _task(False), today="2026-09-14")

    assert outcome["retried"] is False
    assert outcome["last_retry"] == "", "a refused start does not spend the morning"
    assert "already running" in outcome["reason"]

    # And the next card makes the pull the refused one wanted.
    service = _FakeImportService()
    again = check.morning_import_retry(
        service, _task(False), today="2026-09-14", last_retry=outcome["last_retry"]
    )
    assert again["retried"] is True
    assert service.calls == [check.MORNING_RETRY_DAYS]


def test_away_pulls_nothing_at_this_seam_either():
    """TJ-14B review, item C: the AWAY refusal holds at BOTH seams."""
    import trade_mentor_trade_check as check

    service = _FakeImportService()

    outcome = check.morning_import_retry(
        service, _task(False), today="2026-09-14", auto_mode="AWAY"
    )

    assert outcome["retried"] is False
    assert service.calls == []
    assert outcome["last_retry"] == ""


def test_a_service_that_raises_never_costs_the_card():
    """A failed retry is a line in a log, never an exception into the prompt."""
    import trade_mentor_trade_check as check

    service = _FakeImportService(boom=True)

    outcome = check.morning_import_retry(service, _task(False), today="2026-09-14")

    assert outcome["retried"] is False
    assert outcome["last_retry"] == ""
    assert "broker session" in outcome["reason"]


def test_no_service_at_all_is_not_an_error():
    import trade_mentor_trade_check as check

    outcome = check.morning_import_retry(None, _task(False), today="2026-09-14")

    assert outcome["retried"] is False
    assert outcome["reason"] == "no import service"
