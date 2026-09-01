"""R4 section 6.3: display-only repetition and open-burst control.

Trader wording, 2026-08-14: *"I don't want to be constantly seeing the same
stocks over and over ... less spam and more quality ... I def find bangers
though."* The last clause is the constraint: this must reduce repeated ROWS
without weakening detection, evidence, History, armed-hit delivery, or the AWAY
push.

So every test here is written from one of two directions -- does a repeat stop
producing a new row, and does something that must never be folded still get
through. The second set matters more.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from alert_repetition import (  # noqa: E402
    ACTION_DIGEST,
    ACTION_ESCALATE,
    ACTION_FOLD,
    ACTION_NEW,
    OPEN_DIGEST_MINUTES,
    RepetitionLedger,
)

OPEN = datetime(2026, 8, 17, 6, 30)  # a Monday regular open, market-local


def _ledger(**kwargs) -> RepetitionLedger:
    kwargs.setdefault("session_open", OPEN)
    kwargs.setdefault("market_date", "2026-08-17")
    return RepetitionLedger(**kwargs)


def _consider(ledger, symbol="AAPL", side="LONG", *, now=None, **flags):
    return ledger.consider(
        symbol=symbol,
        side=side,
        tier=flags.pop("tier", "B"),
        is_proven=flags.pop("is_proven", False),
        privileged=flags.pop("privileged", False),
        now=now or OPEN + timedelta(hours=2),
        **flags,
    )


# --------------------------------------------------------------------------
# one row per symbol + side + day
# --------------------------------------------------------------------------
def test_the_first_sighting_is_a_new_row():
    decision = _consider(_ledger())
    assert decision.action == ACTION_NEW
    assert decision.repeat_count == 1


def test_a_repeat_folds_into_the_existing_row():
    ledger = _ledger()
    _consider(ledger)
    decision = _consider(ledger)
    assert decision.action == ACTION_FOLD
    assert decision.repeat_count == 2
    assert decision.sounds is False


def test_the_repeat_count_keeps_climbing():
    ledger = _ledger()
    for _ in range(5):
        decision = _consider(ledger)
    assert decision.repeat_count == 5


def test_the_first_seen_time_is_retained_across_repeats():
    ledger = _ledger()
    first = OPEN + timedelta(hours=1)
    _consider(ledger, now=first)
    decision = _consider(ledger, now=first + timedelta(minutes=40))
    assert decision.first_seen == first


def test_the_other_side_of_the_same_symbol_is_its_own_row():
    """A long and a short on one name are different trades, not a repeat."""
    ledger = _ledger()
    _consider(ledger, side="LONG")
    decision = _consider(ledger, side="SHORT")
    assert decision.action == ACTION_NEW


def test_a_different_symbol_is_its_own_row():
    ledger = _ledger()
    _consider(ledger, symbol="AAPL")
    assert _consider(ledger, symbol="NVDA").action == ACTION_NEW


def test_the_ledger_resets_on_a_new_market_date():
    """'Per market day' is the contract. Yesterday's sighting must not silence
    the first alert of a new session."""
    ledger = _ledger()
    _consider(ledger)
    ledger.set_market_date("2026-08-18")
    assert _consider(ledger).action == ACTION_NEW


# --------------------------------------------------------------------------
# escalation - the exhaustive list the trader confirmed 2026-08-16
# --------------------------------------------------------------------------
def test_a_strictly_higher_tier_escalates():
    ledger = _ledger()
    _consider(ledger, tier="B")
    decision = _consider(ledger, tier="A")
    assert decision.action == ACTION_ESCALATE
    assert decision.sounds is True


def test_an_equal_tier_does_not_escalate():
    ledger = _ledger()
    _consider(ledger, tier="A")
    assert _consider(ledger, tier="A").action == ACTION_FOLD


def test_a_lower_tier_does_not_escalate():
    """'Strictly higher' - a B after an A is the same name getting worse."""
    ledger = _ledger()
    _consider(ledger, tier="A")
    assert _consider(ledger, tier="B").action == ACTION_FOLD


def test_the_banger_escalation_is_gone():
    """BANGER retired 2026-09-01 (trader: "We can probably remove this because
    idk what it is"): a matcher with no producer, so this branch could never
    fire. The keyword is removed rather than ignored, so a caller still passing
    it is a loud error and not a silent no-op.

    Fail-before-fix: on the un-fixed code `consider` accepts `is_banger` and the
    second call escalates.
    """
    import pytest as _pytest

    ledger = _ledger()
    with _pytest.raises(TypeError):
        ledger.consider(symbol="AAPL", side="LONG", tier="B", is_banger=True)


def test_the_first_proven_escalates():
    ledger = _ledger()
    _consider(ledger, tier="B")
    assert _consider(ledger, tier="B", is_proven=True).action == ACTION_ESCALATE


def test_a_second_proven_does_not_escalate_again():
    ledger = _ledger()
    _consider(ledger, is_proven=True)
    _consider(ledger, is_proven=True)
    assert _consider(ledger, is_proven=True).action == ACTION_FOLD


def test_an_untiered_repeat_never_escalates_on_tier():
    """An alert with no tier must not read as 'higher than' a tiered one."""
    ledger = _ledger()
    _consider(ledger, tier="A")
    assert _consider(ledger, tier="").action == ACTION_FOLD


def test_a_tiered_alert_after_an_untiered_one_escalates():
    ledger = _ledger()
    _consider(ledger, tier="")
    assert _consider(ledger, tier="C").action == ACTION_ESCALATE


def test_escalation_raises_the_bar_for_the_next_one():
    """After escalating to A, another A folds. Otherwise every subsequent hit
    at the new high tier would re-sound."""
    ledger = _ledger()
    _consider(ledger, tier="B")
    assert _consider(ledger, tier="A").action == ACTION_ESCALATE
    assert _consider(ledger, tier="A").action == ACTION_FOLD
    assert _consider(ledger, tier="S").action == ACTION_ESCALATE


# --------------------------------------------------------------------------
# what must NEVER be folded
# --------------------------------------------------------------------------
def test_a_privileged_hit_is_never_folded():
    """Focus-privileged and trader-armed hits always surface and sound. The
    trader armed that exact condition; quieting it is the one failure this
    whole section is not allowed to cause."""
    ledger = _ledger()
    _consider(ledger, privileged=True)
    decision = _consider(ledger, privileged=True)
    assert decision.action == ACTION_NEW
    assert decision.sounds is True


def test_a_privileged_hit_survives_an_ordinary_row_already_existing():
    """The dangerous ordering: an ordinary alert lands first and creates the
    row, then the trader's own armed watch fires on the same name."""
    ledger = _ledger()
    _consider(ledger, privileged=False)
    assert _consider(ledger, privileged=True).action == ACTION_NEW


def test_a_privileged_hit_is_never_digested_at_the_open():
    ledger = _ledger()
    decision = _consider(ledger, privileged=True, now=OPEN + timedelta(minutes=3))
    assert decision.action == ACTION_NEW


def test_a_proven_alert_is_never_digested_at_the_open():
    ledger = _ledger()
    decision = _consider(ledger, is_proven=True, now=OPEN + timedelta(minutes=3))
    assert decision.action == ACTION_NEW


# --------------------------------------------------------------------------
# the open-burst digest - 30 minutes, trader-confirmed 2026-08-16
# --------------------------------------------------------------------------
def test_the_default_window_is_the_confirmed_thirty_minutes():
    assert OPEN_DIGEST_MINUTES == 30


def test_an_ordinary_alert_inside_the_window_is_digested():
    ledger = _ledger()
    decision = _consider(ledger, now=OPEN + timedelta(minutes=5))
    assert decision.action == ACTION_DIGEST
    assert decision.sounds is False


def test_an_ordinary_alert_after_the_window_is_a_normal_row():
    ledger = _ledger()
    assert _consider(ledger, now=OPEN + timedelta(minutes=31)).action == ACTION_NEW


def test_the_window_boundary_is_pinned_exactly():
    """Mutation check: an inclusive/exclusive flip or an off-by-one minute
    moves exactly this assertion."""
    ledger = _ledger()
    edge = OPEN + timedelta(minutes=OPEN_DIGEST_MINUTES)
    assert _consider(ledger, symbol="A", now=edge - timedelta(seconds=1)).action == ACTION_DIGEST
    assert _consider(ledger, symbol="B", now=edge).action == ACTION_NEW


def test_before_the_open_is_not_the_digest_window():
    """Pre-market alerts are not the open burst."""
    ledger = _ledger()
    assert _consider(ledger, now=OPEN - timedelta(minutes=10)).action == ACTION_NEW


def test_zero_disables_the_digest():
    ledger = _ledger(digest_minutes=0)
    assert _consider(ledger, now=OPEN + timedelta(minutes=1)).action == ACTION_NEW


def test_an_unknown_session_open_disables_the_digest():
    """Fail-open. If we cannot say when the session started we must not
    digest all day - that would be suppression by accident."""
    ledger = _ledger(session_open=None)
    assert _consider(ledger, now=OPEN + timedelta(minutes=1)).action == ACTION_NEW


def test_a_digested_alert_is_still_recorded_as_seen():
    """Digest contents stay reachable, and the row still exists: the next
    sighting of the same name folds into it rather than starting over."""
    ledger = _ledger()
    _consider(ledger, now=OPEN + timedelta(minutes=5))
    decision = _consider(ledger, now=OPEN + timedelta(minutes=40))
    assert decision.action == ACTION_FOLD
    assert decision.repeat_count == 2


def test_a_digested_alert_that_escalates_breaks_out_of_the_digest():
    ledger = _ledger()
    _consider(ledger, tier="B", now=OPEN + timedelta(minutes=5))
    decision = _consider(ledger, tier="S", now=OPEN + timedelta(minutes=10))
    assert decision.action == ACTION_ESCALATE
    assert decision.sounds is True


def test_the_digest_collects_what_it_folded():
    ledger = _ledger()
    _consider(ledger, symbol="AAPL", now=OPEN + timedelta(minutes=2))
    _consider(ledger, symbol="NVDA", now=OPEN + timedelta(minutes=3))
    assert set(ledger.digest_symbols()) == {"AAPL", "NVDA"}


def test_the_digest_empties_on_a_new_market_date():
    ledger = _ledger()
    _consider(ledger, now=OPEN + timedelta(minutes=2))
    ledger.set_market_date("2026-08-18")
    assert ledger.digest_symbols() == []
