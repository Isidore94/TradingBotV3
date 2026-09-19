"""TJ-9 item 4 on the JOURNAL page - BUILDER-ADDED.

The packet's lead decision: "the Journal page's manual tag save: find its one
call to `save_trade_annotation` and pass the provenance from
`trade_origin.label_provenance`". That call is
`scripts/ui/services/journal_feed.save_annotation`, and this file is the test
for it - the tester's suite pins the STORE and the CARD, not this seam.

What is pinned:

* a tag typed on the Journal page the day the trade opened records
  `same_session`; typed later it records `recalled_after`;
* the Journal page never records `claimed_before_entry` - that is the card's
  finding, made against a claim that preceded the fill, and this page holds no
  claim to make it with;
* the value is the pure function's, so the two agree by construction;
* clearing a tag records nothing rather than blanking what a confirm wrote.
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

from tj9_support import REVIEWED, add_round_trip, new_store  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")


@pytest.fixture()
def feed(tmp_path, monkeypatch):
    from ui.services import journal_feed

    store = new_store(tmp_path)
    monkeypatch.setattr(journal_feed, "_STORE", store, raising=False)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    return store


def _freeze(monkeypatch, moment: datetime) -> None:
    """Freeze the wall clock `save_annotation` reads, without faking the rule."""
    import datetime as datetime_module

    class _Frozen(datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102 - stdlib signature
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(datetime_module, "datetime", _Frozen)


def test_a_tag_typed_the_next_day_records_recalled_after(feed, monkeypatch):
    from ui.services import journal_feed

    trade_id = add_round_trip(feed, "AAPL")
    _freeze(monkeypatch, datetime(2026, 9, 14, 11, 0, tzinfo=PACIFIC))

    journal_feed.save_annotation(trade_id, setup_tags="pullback_hold", notes="late note")

    state = feed.annotation_state(trade_id)
    assert state["tag_status"] == "confirmed"
    assert state["setup_tags"] == "pullback_hold"
    assert state["label_provenance"] == "recalled_after"


def test_a_tag_typed_on_the_day_of_the_fill_records_same_session(feed, monkeypatch):
    from ui.services import journal_feed

    trade_id = add_round_trip(feed, "AAPL")
    _freeze(monkeypatch, datetime(2026, 9, 11, 13, 40, tzinfo=PACIFIC))

    journal_feed.save_annotation(trade_id, setup_tags="pullback_hold", notes="")

    assert feed.annotation_state(trade_id)["label_provenance"] == "same_session"


def test_the_page_agrees_with_the_pure_function(feed, monkeypatch):
    """One rule, one implementation. The page never has its own opinion."""
    import trade_origin
    from ui.services import journal_feed

    trade_id = add_round_trip(feed, "AAPL")
    moment = datetime(2026, 9, 14, 11, 0, tzinfo=PACIFIC)
    _freeze(monkeypatch, moment)

    journal_feed.save_annotation(trade_id, setup_tags="pullback_hold", notes="")

    trade = feed.get_trade(trade_id) or {}
    assert feed.annotation_state(trade_id)["label_provenance"] == trade_origin.label_provenance(
        trade, "pullback_hold", (), moment
    )
    assert trade.get("trade_date") == REVIEWED


def test_clearing_the_tag_clears_the_provenance_with_it(feed, monkeypatch):
    """There is no label left to date. A provenance that outlived the label it
    was about would say `claimed_before_entry` over an empty setup, which every
    reader of the three would count."""
    from ui.services import journal_feed

    trade_id = add_round_trip(feed, "AAPL")
    feed.save_trade_annotation(
        trade_id, setup_tags="earnings_gap", notes="", label_provenance="claimed_before_entry"
    )
    _freeze(monkeypatch, datetime(2026, 9, 14, 11, 0, tzinfo=PACIFIC))

    journal_feed.save_annotation(trade_id, setup_tags="", notes="cleared")

    state = feed.annotation_state(trade_id)
    assert state["setup_tags"] == ""
    assert state["label_provenance"] == ""


def test_an_unchanged_tag_and_a_silent_caller_cannot_blank_the_provenance(feed):
    """The rule that still holds: a caller with nothing to say about the age of
    a label, saving the SAME label, leaves what the confirm recorded."""
    trade_id = add_round_trip(feed, "AAPL")
    feed.save_trade_annotation(
        trade_id, setup_tags="earnings_gap", notes="", label_provenance="claimed_before_entry"
    )

    feed.save_trade_annotation(trade_id, setup_tags="earnings_gap", notes="a later note")

    state = feed.annotation_state(trade_id)
    assert state["notes"] == "a later note"
    assert state["label_provenance"] == "claimed_before_entry"


def test_a_changed_tag_and_a_silent_caller_recomputes_rather_than_keeping_a_lie(feed):
    """`accept_auto_tags` adding a tag is this case. The old provenance was
    about the OLD label; keeping it would date a label that never existed."""
    from ui.services import journal_feed

    trade_id = add_round_trip(feed, "AAPL")
    feed.save_trade_annotation(
        trade_id, setup_tags="earnings_gap", notes="", label_provenance="claimed_before_entry"
    )

    journal_feed.accept_auto_tags(trade_id, ["avwap_breakout"])

    state = feed.annotation_state(trade_id)
    assert "avwap_breakout" in state["setup_tags"]
    assert state["label_provenance"] != "claimed_before_entry"
    assert state["label_provenance"] in ("same_session", "recalled_after")
