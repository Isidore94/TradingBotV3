"""TJ-9 review round: blockers 2 and 3, both reproduced on live shapes.

BLOCKER 2 - `same_session` was computed from the CLOSE date. `trade_origin.
trade_session` read `trade_date` first, and `journal_store.rebuild_trades`
writes `trade_date = closed_at or opened_at`. The live SMPL trade opened
2026-08-27 and closed 2026-09-18; a label confirmed on the closing day read
`same_session` - the one thing `label_provenance` exists to make impossible -
and a label confirmed on the OPENING day read `recalled_after`. 116 of the
journal's 216 trades have an opened date that differs from `trade_date`, and
`journal_feed.save_annotation` writes this value today.

BLOCKER 3 - a `vetoed:<code>` provisional tag was offered as a one-click
confirmed SETUP. Reproduced on a copy of the live journal: APTV carried
`vetoed:too_extended_from_base` and the card offered it; one click wrote it
`confirmed`, where `journal_analytics`' "My setups" counts it.
`journal_analytics` prefixes a rejection *"so a rejection can never be mistaken
for an endorsement in a Tags column"*, and that sentence now has one definition
(`is_rejection_tag`) that both the writer and this reader use.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

PACIFIC = ZoneInfo("America/Los_Angeles")

#: The live SMPL shape: opened on one session, closed three weeks later, and
#: `trade_date` naming the CLOSE.
HELD_TRADE = {
    "trade_id": "SMPL-1",
    "symbol": "SMPL",
    "direction": "LONG",
    "opened_at": "2026-08-27T07:31:00-07:00",
    "closed_at": "2026-09-18T12:05:00-07:00",
    "trade_date": "2026-09-18",
}


# ---------------------------------------------------------------------------
# Blocker 2 - the session is the FIRST FILL's
# ---------------------------------------------------------------------------


def test_a_label_written_on_the_closing_day_of_a_held_trade_is_recalled_after():
    """Three weeks after the entry, the outcome is not just known - it is
    finished. Calling that `same_session` would make the provenance column say
    the opposite of what it is for."""
    from trade_origin import label_provenance

    verdict = label_provenance(
        HELD_TRADE, "avwap_breakout", (), datetime(2026, 9, 18, 13, 40, tzinfo=PACIFIC)
    )

    assert verdict == "recalled_after"


def test_a_label_written_on_the_day_the_held_trade_opened_is_same_session():
    """The other direction, and the one the trader actually earns: labelled the
    day they entered, before the trade had a result at all."""
    from trade_origin import label_provenance

    verdict = label_provenance(
        HELD_TRADE, "avwap_breakout", (), datetime(2026, 8, 27, 13, 40, tzinfo=PACIFIC)
    )

    assert verdict == "same_session"


def test_the_session_of_a_held_trade_is_the_first_fills_not_the_close():
    from trade_origin import trade_session

    assert trade_session(HELD_TRADE).isoformat() == "2026-08-27"


def test_a_trade_with_only_a_date_still_answers_from_that_date():
    """Uncertainty is not a verdict, but a trade with nothing but a date has
    one fact and it is used rather than thrown away."""
    from trade_origin import trade_session

    assert trade_session({"trade_id": "X", "trade_date": "2026-09-11"}).isoformat() == "2026-09-11"
    assert trade_session({"trade_id": "X"}) is None


def test_the_journal_page_dates_a_held_trade_by_its_entry(tmp_path, monkeypatch):
    """The real writer, on the shape that was wrong. `journal_feed` is what
    116 of the live trades would have gone through."""
    import datetime as datetime_module

    from journal_store import JournalStore
    from ui.services import journal_feed

    store = JournalStore(Path(tmp_path) / "journal.sqlite3")
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO trades(trade_id, symbol, direction, opened_at, closed_at, trade_date, "
            "status, broker, account_number, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "SMPL-1",
                "SMPL",
                "LONG",
                HELD_TRADE["opened_at"],
                HELD_TRADE["closed_at"],
                HELD_TRADE["trade_date"],
                "CLOSED",
                "MANUAL",
                "TJ9",
                "2026-09-18T12:05:00-07:00",
            ),
        )
    monkeypatch.setattr(journal_feed, "_STORE", store, raising=False)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)

    class _Frozen(datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102 - stdlib signature
            moment = datetime(2026, 9, 18, 13, 40, tzinfo=PACIFIC)
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(datetime_module, "datetime", _Frozen)
    journal_feed.save_annotation("SMPL-1", setup_tags="avwap_breakout", notes="")

    assert store.annotation_state("SMPL-1")["label_provenance"] == "recalled_after"


# ---------------------------------------------------------------------------
# Blocker 3 - a rejection is never a setup
# ---------------------------------------------------------------------------


def test_the_rejection_prefixes_have_one_definition_and_the_writer_uses_it():
    """The predicate lives beside `LINK_TAG_PREFIX`, in the module that writes
    the prefixes, so the reader cannot drift from the writer."""
    import journal_analytics

    assert journal_analytics.is_rejection_tag("vetoed:too_extended_from_base") is True
    assert journal_analytics.is_rejection_tag("vetoed") is True
    assert journal_analytics.is_rejection_tag("passed:thin,extended") is True
    assert journal_analytics.is_rejection_tag("passed") is True
    assert journal_analytics.is_rejection_tag("avwap_breakout") is False
    assert journal_analytics.is_rejection_tag("") is False
    assert journal_analytics.VETO_TAG_WORD in journal_analytics.REJECTION_TAG_WORDS
    assert journal_analytics.PASS_TAG_WORD in journal_analytics.REJECTION_TAG_WORDS


def test_a_vetoed_provisional_tag_is_never_offered_as_a_setup():
    """The APTV reproduction. The button read
    `Setup: vetoed:too_extended_from_base (confirm)`."""
    from trade_mentor_trade_check import setup_guess_for

    trade = {
        "trade_id": "APTV-1",
        "symbol": "APTV",
        "direction": "LONG",
        "opened_at": "2026-09-11T07:31:00-07:00",
        "trade_date": "2026-09-11",
        "setup_tags": "vetoed:too_extended_from_base",
        "tag_status": "provisional",
    }

    assert setup_guess_for(trade, ()) == ("", "")


def test_a_passed_tag_and_a_link_tag_are_never_offered_either():
    from trade_mentor_trade_check import setup_guess_for

    base = {
        "trade_id": "X",
        "symbol": "X",
        "direction": "LONG",
        "opened_at": "2026-09-11T07:31:00-07:00",
        "trade_date": "2026-09-11",
        "tag_status": "provisional",
    }

    assert setup_guess_for({**base, "setup_tags": "passed:thin,extended"}, ()) == ("", "")
    assert setup_guess_for({**base, "setup_tags": "link:focus_add"}, ()) == ("", "")


def test_a_tag_string_holding_several_tags_is_split_and_only_the_setup_survives():
    """The column is a LIST. A string holding a real setup AND a rejection must
    offer the setup alone, never the whole string."""
    from trade_mentor_trade_check import eligible_setup_names, setup_guess_for

    trade = {
        "trade_id": "X",
        "symbol": "X",
        "direction": "LONG",
        "opened_at": "2026-09-11T07:31:00-07:00",
        "trade_date": "2026-09-11",
        "setup_tags": "vetoed:too_extended_from_base; avwap_breakout",
        "tag_status": "provisional",
    }

    assert eligible_setup_names(trade["setup_tags"]) == ("avwap_breakout",)
    assert setup_guess_for(trade, ()) == ("avwap_breakout", "provisional")


def test_a_claimed_like_that_somehow_names_a_rejection_is_not_the_guess():
    """The claim lane gets the same filter. `claimed_setup_id` is validated on
    write, but the guess must not depend on a validation two stores away."""
    from trade_mentor_trade_check import setup_guess_for

    trade = {
        "trade_id": "X",
        "symbol": "AAPL",
        "direction": "LONG",
        "opened_at": "2026-09-11T07:31:00-07:00",
        "trade_date": "2026-09-11",
        "setup_tags": "",
        "tag_status": "provisional",
    }
    claims = [
        {
            "symbol": "AAPL",
            "side": "LONG",
            "claimed_setup_id": "vetoed:too_extended_from_base",
            "created_at": "2026-09-11T14:15:00+00:00",
        }
    ]

    assert setup_guess_for(trade, claims) == ("", "")


def test_the_writer_refuses_a_rejection_even_if_one_reaches_it(tmp_path):
    """The second line of defence. `confirm_setup` is what touches the
    trader-owned table, so the refusal is there too."""
    import trade_mentor_trade_check as check
    from journal_store import JournalStore

    store = JournalStore(Path(tmp_path) / "journal.sqlite3")
    question = check.TradeQuestion(
        trade_id="APTV-1",
        symbol="APTV",
        direction="LONG",
        missing=("setup",),
        setup_guess="vetoed:too_extended_from_base",
        setup_guess_lane="provisional",
        opened_at="2026-09-11T07:31:00-07:00",
        trade_date="2026-09-11",
    )

    result = check.confirm_setup(store, question, now=datetime(2026, 9, 14, 9, 3, tzinfo=PACIFIC))

    assert result["ok"] is False
    assert "not a setup name" in result["reason"]
    assert store.annotation_state("APTV-1")["setup_tags"] == ""
    assert store.annotation_state("APTV-1")["label_provenance"] == ""

    # And the same refusal when it arrives through the card's vocabulary list.
    passed = check.confirm_setup(
        store,
        check.TradeQuestion("T2", "X", "LONG", ("setup",), setup_guess="avwap_breakout"),
        now=datetime(2026, 9, 14, 9, 3, tzinfo=PACIFIC),
        setup="passed:thin",
    )
    assert passed["ok"] is False


def test_the_card_lists_the_setup_vocabulary_beside_the_confirm_button(tmp_path):
    """"a confirm button beside the vocabulary list". The list is the registry's
    own claim ids plus the setup documents' families, and it carries no
    rejection - so a wrong guess is CORRECTED here rather than confirmed."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    vocabulary = check.setup_vocabulary()
    assert "avwap_breakout" in vocabulary
    assert not any(check.is_rejection_or_link(name) for name in vocabulary)

    task = check.TradeCheckTask(
        reviewed_session="2026-09-11",
        journal_ready=True,
        trades=(
            check.TradeQuestion(
                trade_id="T1",
                symbol="AAPL",
                direction="LONG",
                missing=("setup",),
                setup_guess="opening_drive",
                setup_guess_lane="provisional",
                opened_at="2026-09-11T07:31:00-07:00",
                trade_date="2026-09-11",
            ),
        ),
        incomplete_total=1,
        fills_current_to="2026-09-11",
    )
    card = TradeMentorCard(drafts_path=Path(tmp_path) / "drafts.json")
    card.set_trade_check(task, store=None)

    box = card.setup_choice_box("T1")
    assert box is not None
    names = [box.itemData(index) for index in range(box.count())]
    assert names[0] == "opening_drive", "the guess is preselected, so one click is one click"
    assert "avwap_breakout" in names
    assert not any(check.is_rejection_or_link(name) for name in names)
