"""TJ-9 items 3 and 4 - one click for the setup, and a label that knows its age.

Written BEFORE the fix and red on `claude/tj9-forced-trade-labels`'s base commit
(`57151d0f`). The builder makes them pass and may only ADD.

WHAT IS PINNED
--------------
* **Lane order is asserted with BOTH lanes populated at once** (item 3). A trade
  that carries a provisional tag AND matched a claimed like must open on the
  LIKE's setup; an implementation that reads the provisional tag first passes a
  single-lane test and fails this one.
* **A provisional tag is not an answer.** Today `_FIELD_SOURCES["setup"]` reads
  `setup_tags` and `list_trades` joins the column whatever its `tag_status` is,
  so a machine guess silently retires the question it exists to ask. The trade
  row already carries `tag_status` (`journal_store.py:1907`), so nothing new has
  to be read to tell the two apart.
* **A guess nobody clicked writes NOTHING**, and a CONFIRMED tag is never
  offered a guess at all - the refusal that already lives in
  `apply_provisional_tags` is not weakened, it is simply never reached.
* **Provenance is decided from aware stamps with `astimezone`.** The
  `claimed_before_entry` case uses a like stamped in UTC against a fill stamped
  in Pacific, arranged so that a naive comparison of the two ISO strings gives
  the WRONG answer (`14:15` looks later than `07:31`, and is 16 minutes
  earlier).
* **An old annotation row has `label_provenance` PRESENT and EMPTY**, never
  absent and never NULL - the row is written through raw SQL without the column
  and the store is then reopened, which is what a live database does.

No test here asserts a vocabulary version, and every setup name is a plain
string the test itself chose.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
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
)

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

PACIFIC = ZoneInfo("America/Los_Angeles")

#: 07:15 Pacific on the reviewed session, written in UTC. The AAPL round trip's
#: first fill is 07:31 Pacific, so this is SIXTEEN MINUTES BEFORE it - and the
#: two ISO strings compare the other way round if the offsets are ignored.
LIKE_BEFORE_ENTRY = datetime(2026, 9, 11, 14, 15, tzinfo=timezone.utc)
#: 11:00 Pacific - after the first fill and after the exit.
LIKE_AFTER_ENTRY = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)


@pytest.fixture()
def annotations_file(tmp_path, monkeypatch):
    """A private annotation log. `preference_trade_outcomes` re-imports
    `TRADER_ANNOTATIONS_FILE` inside `_annotation_statements`, so patching the
    module attribute redirects the real read path rather than faking it."""
    import project_paths

    path = tmp_path / "trader_annotations.jsonl"
    monkeypatch.setattr(project_paths, "TRADER_ANNOTATIONS_FILE", path, raising=False)
    return path


def _record_claimed_like(path: Path, *, symbol: str, side: str, setup: str, when: datetime):
    from ui.annotations.store import EVENT_LIKE_CLAIM, record_annotation

    row = record_annotation(
        EVENT_LIKE_CLAIM,
        path=path,
        symbol=symbol,
        side=side,
        claimed_setup_id=setup,
        like_mode="claimed",
        session_date=REVIEWED,
        created_at=when,
    )
    assert row is not None, "the like did not reach the annotation log"
    return row


def _question_for(task, symbol: str):
    for question in task.trades:
        if question.symbol == symbol:
            return question
    raise AssertionError(f"{symbol} was not asked about: {task}")


# ---------------------------------------------------------------------------
# Item 3 - one click for the setup
# ---------------------------------------------------------------------------


def test_a_provisional_tag_is_not_an_answer_so_the_setup_is_still_asked(tmp_path, annotations_file):
    """33 of the trader's live trades carry a provisional tag and one carries a
    confirmed one. A provisional tag is the machine's guess, so the question
    stays open and the guess rides on it as a suggestion."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    assert store.apply_provisional_tags(trade_id, "opening_drive") is True

    question = _question_for(check.build_task(store, SESSION_TODAY), "AAPL")

    assert "setup" in question.missing, "a machine guess must not retire the question"
    assert question.setup_guess == "opening_drive"
    assert question.setup_guess_lane == "provisional"


def test_the_setup_guess_prefers_the_claimed_like_over_the_provisional_tag(
    tmp_path, annotations_file
):
    """Lane order, with both lanes loaded. The trader NAMED `earnings_gap` when
    they claimed the like sixteen minutes before their first fill; the bulk
    tagger later guessed `opening_drive`. The card opens on what the trader
    said."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    store.apply_provisional_tags(trade_id, "opening_drive")
    _record_claimed_like(
        annotations_file, symbol="AAPL", side="LONG", setup="earnings_gap", when=LIKE_BEFORE_ENTRY
    )

    question = _question_for(check.build_task(store, SESSION_TODAY), "AAPL")

    assert question.setup_guess == "earnings_gap"
    assert question.setup_guess_lane == "claimed_like"


def test_a_like_stamped_after_the_first_fill_is_not_the_guess(tmp_path, annotations_file):
    """"The like that preceded the trade" is the whole claim. A like clicked at
    11:00 about a trade opened at 07:31 did not name that entry, so the guess
    falls back to the provisional lane."""
    import trade_mentor_trade_check as check

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    store.apply_provisional_tags(trade_id, "opening_drive")
    _record_claimed_like(
        annotations_file, symbol="AAPL", side="LONG", setup="earnings_gap", when=LIKE_AFTER_ENTRY
    )

    question = _question_for(check.build_task(store, SESSION_TODAY), "AAPL")

    assert question.setup_guess == "opening_drive"
    assert question.setup_guess_lane == "provisional"


def test_a_guess_nobody_clicked_writes_nothing(tmp_path, annotations_file):
    """Showing the card is not a write. After the section is built and the
    trader walks away, the annotation is exactly what the bulk tagger left."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    store.apply_provisional_tags(trade_id, "opening_drive")
    _record_claimed_like(
        annotations_file, symbol="AAPL", side="LONG", setup="earnings_gap", when=LIKE_BEFORE_ENTRY
    )

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card.set_trade_check(check.build_task(store, SESSION_TODAY), store=store)
    button = card.setup_confirm_button(trade_id)
    assert button is not None and button.isEnabled() is True

    state = store.annotation_state(trade_id)
    assert state["tag_status"] == "provisional"
    assert state["setup_tags"] == "opening_drive"


def test_confirming_the_guess_writes_the_traders_own_confirmed_tag(tmp_path, annotations_file):
    """One click, and it is the TRADER's write through the Journal's own writer:
    `tag_status` becomes `confirmed` and the setup becomes the one they
    confirmed, not the one the bulk tagger parked."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    store.apply_provisional_tags(trade_id, "opening_drive")
    _record_claimed_like(
        annotations_file, symbol="AAPL", side="LONG", setup="earnings_gap", when=LIKE_BEFORE_ENTRY
    )

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card.set_trade_check(check.build_task(store, SESSION_TODAY), store=store)
    card.setup_confirm_button(trade_id).click()

    state = store.annotation_state(trade_id)
    assert state["tag_status"] == "confirmed"
    assert state["setup_tags"] == "earnings_gap"
    # The confirmed setup answers the question, so it stops being asked.
    assert "setup" not in _question_for(check.build_task(store, SESSION_TODAY), "AAPL").missing


def test_a_confirmed_tag_is_never_offered_or_overwritten_by_a_guess(tmp_path, annotations_file):
    """The trader already said `pullback_hold`. A claimed like naming something
    else does not reopen the question and cannot replace the answer."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    trade_id = add_round_trip(store, "AAPL")
    store.save_trade_annotation(trade_id, setup_tags="pullback_hold", notes="")
    _record_claimed_like(
        annotations_file, symbol="AAPL", side="LONG", setup="earnings_gap", when=LIKE_BEFORE_ENTRY
    )

    question = _question_for(check.build_task(store, SESSION_TODAY), "AAPL")
    assert "setup" not in question.missing
    assert question.setup_guess == ""
    assert question.setup_guess_lane == ""

    card = TradeMentorCard(drafts_path=tmp_path / "drafts.json")
    card.set_trade_check(check.build_task(store, SESSION_TODAY), store=store)
    assert card.setup_confirm_button(trade_id) is None

    state = store.annotation_state(trade_id)
    assert state["tag_status"] == "confirmed"
    assert state["setup_tags"] == "pullback_hold"


# ---------------------------------------------------------------------------
# Item 4 - label_provenance
# ---------------------------------------------------------------------------


def _trade(opened: str) -> dict:
    return {
        "trade_id": "T1",
        "symbol": "AAPL",
        "direction": "LONG",
        "opened_at": opened,
        "trade_date": opened[:10],
    }


def test_a_setup_the_trader_claimed_before_the_first_fill_is_claimed_before_entry():
    """The like is written in UTC and the fill in Pacific, arranged so the naive
    strings compare the wrong way: `14:15+00:00` is 07:15 Pacific, SIXTEEN
    MINUTES BEFORE `07:31-07:00`. Only `astimezone` gets this right."""
    from trade_origin import label_provenance

    claims = [
        {
            "symbol": "AAPL",
            "side": "LONG",
            "claimed_setup_id": "earnings_gap",
            "claim_at_utc": LIKE_BEFORE_ENTRY.isoformat(),
        }
    ]

    verdict = label_provenance(
        _trade("2026-09-11T07:31:00-07:00"),
        "earnings_gap",
        claims,
        datetime(2026, 9, 14, 9, 3, tzinfo=PACIFIC),
    )

    assert verdict == "claimed_before_entry"


def test_a_claim_that_named_a_different_setup_is_not_the_provenance_of_this_label():
    """The claim has to be a claim about THIS setup. A claim naming something
    else, however early, is somebody else's evidence."""
    from trade_origin import label_provenance

    claims = [
        {
            "symbol": "AAPL",
            "side": "LONG",
            "claimed_setup_id": "gap_fill",
            "claim_at_utc": LIKE_BEFORE_ENTRY.isoformat(),
        }
    ]

    verdict = label_provenance(
        _trade("2026-09-11T07:31:00-07:00"),
        "earnings_gap",
        claims,
        datetime(2026, 9, 14, 9, 3, tzinfo=PACIFIC),
    )

    assert verdict == "recalled_after"


def test_a_label_confirmed_on_the_session_the_trade_opened_is_same_session():
    """TJ-14 item 4 makes this reachable; the value is defined and tested now.
    Confirmed at 13:40 Pacific on the day of the fill - after the close, still
    the same session, and before the outcome has had a night to settle."""
    from trade_origin import label_provenance

    verdict = label_provenance(
        _trade("2026-09-11T07:31:00-07:00"),
        "earnings_gap",
        [],
        datetime(2026, 9, 11, 13, 40, tzinfo=PACIFIC),
    )

    assert verdict == "same_session"


def test_a_label_confirmed_the_next_morning_is_recalled_after():
    """The 09:00 card's own case, and the honest one: the label was made when
    the trade's outcome was already known."""
    from trade_origin import label_provenance

    verdict = label_provenance(
        _trade("2026-09-11T07:31:00-07:00"),
        "earnings_gap",
        [],
        datetime(2026, 9, 14, 9, 3, tzinfo=PACIFIC),
    )

    assert verdict == "recalled_after"


def test_the_three_provenance_values_are_named_constants():
    """Three strings, spelled once. A statistic that reports the three apart
    cannot be written against three literals scattered over four modules."""
    import trade_origin

    assert trade_origin.CLAIMED_BEFORE_ENTRY == "claimed_before_entry"
    assert trade_origin.SAME_SESSION == "same_session"
    assert trade_origin.RECALLED_AFTER == "recalled_after"
    assert trade_origin.LABEL_PROVENANCES == (
        "claimed_before_entry",
        "same_session",
        "recalled_after",
    )


def test_an_annotation_row_written_before_the_column_existed_reads_present_and_empty(tmp_path):
    """The live database has 185 annotation rows written before this column. The
    row here is inserted through raw SQL WITHOUT the column and the store is then
    reopened, which is exactly what the desk does on the next launch. The value
    must be the empty string - present and blank - never NULL and never absent,
    because every reader of the three provenances has to be able to say
    "unrecorded" without a schema check."""
    from journal_store import JournalStore

    db_path = Path(tmp_path) / "journal.sqlite3"
    store = JournalStore(db_path)
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO trade_annotations(trade_id, setup_tags, notes, updated_at, tag_status) "
            "VALUES(?, ?, ?, ?, ?)",
            ("OLD-TRADE", "pullback_hold", "took the level", "2026-08-01T10:00:00", "confirmed"),
        )

    reopened = JournalStore(db_path)
    with reopened.connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(trade_annotations)")}
        value = conn.execute(
            "SELECT label_provenance FROM trade_annotations WHERE trade_id = ?", ("OLD-TRADE",)
        ).fetchone()[0]

    assert "label_provenance" in columns
    assert value == "", "an old row is PRESENT and EMPTY, not NULL and not absent"
    assert reopened.annotation_state("OLD-TRADE")["label_provenance"] == ""


def test_the_journals_own_writer_records_the_provenance_of_a_confirmed_tag(tmp_path):
    """The write goes through `save_trade_annotation` - the Journal's own writer,
    which is what makes it the trader's act - and the provenance travels with
    it. Two saves of the same trade record the second provenance, because the
    second is what the trader just did."""
    store = new_store(tmp_path)
    trade_id = add_round_trip(store, "AAPL")

    store.save_trade_annotation(
        trade_id, setup_tags="earnings_gap", notes="", label_provenance="claimed_before_entry"
    )
    assert store.annotation_state(trade_id)["label_provenance"] == "claimed_before_entry"
    assert store.annotation_state(trade_id)["tag_status"] == "confirmed"

    store.save_trade_annotation(
        trade_id, setup_tags="pullback_hold", notes="", label_provenance="recalled_after"
    )
    assert store.annotation_state(trade_id)["label_provenance"] == "recalled_after"


def test_the_nine_oclock_confirm_stamps_recalled_after_on_the_written_row(
    tmp_path, annotations_file
):
    """End to end on the real click: the guess came from a like stamped before
    the first fill, so the row the confirm writes says `claimed_before_entry` -
    and the same click on a trade with no such claim would say
    `recalled_after`. The provenance is decided by the pure function, not by the
    button."""
    import trade_mentor_trade_check as check
    from ui.widgets.trade_mentor_card import TradeMentorCard

    store = new_store(tmp_path)
    mark_covered(store, REVIEWED)
    claimed_id = add_round_trip(store, "AAPL")
    plain_id = add_round_trip(store, "MSFT")
    store.apply_provisional_tags(plain_id, "opening_drive")
    _record_claimed_like(
        annotations_file, symbol="AAPL", side="LONG", setup="earnings_gap", when=LIKE_BEFORE_ENTRY
    )

    card = TradeMentorCard(
        drafts_path=tmp_path / "drafts.json",
        clock=lambda: datetime(2026, 9, 14, 9, 3, tzinfo=PACIFIC),
    )
    card.set_trade_check(check.build_task(store, SESSION_TODAY), store=store)
    card.setup_confirm_button(claimed_id).click()
    card.setup_confirm_button(plain_id).click()

    assert store.annotation_state(claimed_id)["label_provenance"] == "claimed_before_entry"
    assert store.annotation_state(plain_id)["label_provenance"] == "recalled_after"
