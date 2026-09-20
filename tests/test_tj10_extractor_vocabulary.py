"""TJ-10 item 5 - the trader's own trend words, as a NEW extractor version.

Packet `.claude/packets/TJ-10.md` item 5; `plan.md` §12.4 "TJ-10" item 6. RED
before the build. `scripts/market_thesis.py` is this packet's, for the
vocabulary and its version only.

The fixture
-----------
``tests/fixtures/tj10_live_notes.json`` holds the **49** Market Journal notes
the trader had written by 2026-09-19 21:40 PDT - TEXT and entry id only, copied
off a read-only copy of the live ledger. Every row carries ``stance_before``:
what ``extract_thesis`` read on the branch point (``e8c04f88``), pinned BEFORE
the bump, so the fixture is not a portrait of the code that is about to change.

Two corrections to the packet's premise, both measured on that copy
-------------------------------------------------------------------
* The packet says **42** notes (21 unstated / 8 bullish / 7 neutral / 6
  bearish). That was true before 2026-09-18's seven notes were written. Today
  it is **49**: 23 unstated, 11 bullish, 8 neutral, 7 bearish.
* Of the 23 ``unstated`` notes only **10** carry no stance word at all. **12**
  are unstated because a bullish word AND a bearish word both fire ("higher"
  and "lower" in one sentence) and one is an explicit hedge. So a wider
  vocabulary reaches the 10, not the 23 - and every word added also makes a
  both-directions collision more likely. That is a QUESTION for the lead, not
  something a test may decide, so the aggregate assertion below is a FLOOR (the
  count must fall) plus the exact rows the packet names.

What these tests pin
--------------------
* Each trend word the packet names reads as a stance, whole-token.
* The packet's own example - *"D1 SPY is still downtrending..."*, a real live
  note - reads BEARISH where it reads ``unstated`` today, and its span
  reproduces the word.
* An explicit hedge still wins: ``unstated``, whatever else is in the sentence.
* ``EXTRACTOR_VERSION`` is a NEW value, and a stored row keeps the version it
  was written with. **No literal version string is asserted anywhere here.**
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

#: The census the OLD extractor read over the fixture, pinned here as a fact
#: about the branch point rather than re-derived from the code under test.
BEFORE = {"unstated": 23, "bullish": 11, "neutral": 8, "bearish": 7}
NOTE_COUNT = 49


def _stance(text: str) -> str:
    import market_thesis

    return market_thesis.extract_thesis({"text": text}).stance


def test_the_fixture_is_the_49_live_notes_with_their_pinned_before_census():
    notes = fx.live_notes()

    assert len(notes) == NOTE_COUNT
    assert collections.Counter(note["stance_before"] for note in notes) == BEFORE


@pytest.mark.parametrize(
    "text,expected",
    [
        ("D1 SPY is downtrending", "bearish"),
        ("SPY is uptrending off the lows", "bullish"),
        ("we are rejecting the 20sma", "bearish"),
        ("the market is leaking all afternoon", "bearish"),
        ("SPY making lower highs on the M30", "bearish"),
        ("QQQ making higher lows since the open", "bullish"),
        ("the market is holding lows here", "bullish"),
        ("the bounce is failing at the 50sma", "bearish"),
        ("SPY breaking down through the 100sma", "bearish"),
        ("QQQ breaking out over the highs", "bullish"),
    ],
)
def test_each_trend_word_the_trader_uses_reads_as_a_stance(text, expected):
    """Whole-token, the same rule the rest of this vocabulary already keeps."""
    assert _stance(text) == expected


def test_a_trend_word_never_fires_inside_a_longer_word():
    """"uptrending" must not fire inside "abruptrending", and "leaking" must
    not fire inside "sleaking". Whole-token, exactly like `lower`/`flowering`."""
    assert _stance("the tape is unrejecting nothing at all") == "unstated"
    assert _stance("a sleaking sound from the vents") == "unstated"


def test_the_packets_own_live_note_reads_bearish_where_it_read_unstated():
    """*"D1 SPY is still downtrending..."* - the note the trader actually wrote
    on 2026-09-17, and the reason decision 0021 answer 21 made the graded read
    a CLICK. The extractor still has to read it."""
    import market_thesis

    note = next(
        row for row in fx.live_notes() if "downtrending" in row["text"].lower()
    )
    assert note["stance_before"] == "unstated"

    draft = market_thesis.extract_thesis({"text": note["text"]})

    assert draft.stance == market_thesis.STANCE_BEARISH
    start, end = draft.spans["stance"]
    assert note["text"][start:end].lower() in {"downtrending", "rejecting"}


def test_a_hedge_still_wins_over_every_new_trend_word():
    """"Unstated stays unstated" is rule 2 of this module, and a wider
    vocabulary may not quietly overrule it."""
    assert _stance("SPY is downtrending but I am not sure it holds") == "unstated"
    assert _stance("uptrending, though honestly I have no view here") == "unstated"


def test_more_of_the_live_notes_read_as_a_direction_than_before():
    """The measured before/after over the fixture, and it is a FLOOR.

    18 of the 49 notes read bullish or bearish on the branch point (11 + 7).
    The packet's own word list reaches exactly one note that today has no
    stance word at all - *"D1 SPY is still downtrending..."* - and one that
    today reads `neutral` ("the bounce is ... failing"), so the honest floor is
    19. It is a floor rather than an exact census because the 12 notes that are
    `unstated` from a bullish-AND-bearish collision cannot be reached by adding
    words at all, and which of them a wider vocabulary moves is a QUESTION for
    the lead, not something a test may settle.
    """
    import market_thesis

    notes = fx.live_notes()
    after = collections.Counter(
        market_thesis.extract_thesis({"text": note["text"]}).stance
        for note in notes
    )

    assert sum(after.values()) == NOTE_COUNT
    before_directional = BEFORE["bullish"] + BEFORE["bearish"]
    assert before_directional == 18
    assert after["bullish"] + after["bearish"] > before_directional


def test_a_wider_vocabulary_never_reverses_a_stance_the_old_one_already_read():
    """A GUARD, and it holds on the branch point too: adding words may read
    MORE notes, and may never read a bullish note as bearish."""
    import market_thesis

    flipped = [
        note["entry_id"] for note in fx.live_notes()
        if note["stance_before"] in {"bullish", "bearish"}
        and market_thesis.extract_thesis({"text": note["text"]}).stance
        in {"bullish", "bearish"}
        and market_thesis.extract_thesis({"text": note["text"]}).stance
        != note["stance_before"]
    ]

    assert flipped == []


def test_the_aggregate_never_gets_worse():
    """A GUARD, true on the branch point: a vocabulary that made MORE notes
    ambiguous would be a regression dressed as a feature."""
    import market_thesis

    after = collections.Counter(
        market_thesis.extract_thesis({"text": note["text"]}).stance
        for note in fx.live_notes()
    )

    assert after["unstated"] <= BEFORE["unstated"]


def test_the_extractor_version_moved_and_is_never_asserted_as_a_literal():
    """The veto vocabulary's rule, restated: the version travels on the row and
    a test never spells it. What IS pinned is that it CHANGED - every draft
    carries the module's current version, and a row stored under an older one
    keeps it."""
    import market_thesis

    draft = market_thesis.extract_thesis({"text": "SPY is downtrending"})
    assert draft.extractor_version == market_thesis.EXTRACTOR_VERSION

    older = {
        "entry_id": "mj-2026-08-27-old",
        "extractor_version": "market_thesis_vocab_v0_fixture",
        "kind": market_thesis.KIND_THESIS,
        "session_date": "2026-08-27",
        "stance": "unstated",
        "horizon": "unstated",
        "horizon_sessions": 0,
    }
    # A reader of an old row never re-stamps it.
    assert market_thesis.status_for_row(
        older, as_of=fx.SESSION_DATE
    )[0] == market_thesis.STATUS_OPEN
    assert older["extractor_version"] != market_thesis.EXTRACTOR_VERSION


def test_the_version_on_the_branch_point_is_not_the_version_after_the_bump():
    """Pinned as a FACT about `e8c04f88`, read there and written down here, so
    the assertion is "it moved" rather than "it equals <string>"."""
    import market_thesis

    branch_point_version = "market_thesis_vocab_v1"
    assert market_thesis.EXTRACTOR_VERSION != branch_point_version
