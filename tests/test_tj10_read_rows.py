"""TJ-10 item 1 - one read row per prediction, and the click always wins.

Packet `.claude/packets/TJ-10.md` item 1; `plan.md` §12.4 "TJ-10" items 1 and 4;
decision 0021 answers 21 and 29. RED before the build.

The contract these tests pin
----------------------------

``scripts/market_read_grades.py`` (new, PURE - no Qt, no model, no network, no
clock of its own).

``read_rows(entries, *, session) -> list[dict]``
    One row per prediction the session's non-machine entries carry, oldest
    first. Each row::

        {"read_id", "entry_id", "session", "stamp", "benchmark",
         "source", "direction", "horizon", "confidence", "because", "span"}

    * ``source`` is ``SOURCE_CLICK`` whenever ``market_journal.prediction_of``
      answers, and ``SOURCE_EXTRACTED`` only for an entry with NO click. A
      clicked row's direction, horizon, confidence and because are the stored
      click, byte for byte - the words are never consulted for a row that has
      one (decision 0021 answer 29).
    * ``stamp`` is ``mentor.responded_at`` when the row has one and
      ``created_at`` otherwise - the hour the trader ANSWERED, which is the
      rule ``day_review_markers._entry_stamp`` already uses (lead, 2026-09-19).
    * ``benchmark`` is ``DEFAULT_BENCHMARK`` ("SPY") unless the prediction
      names one of ``market_story.BENCHMARKS``.
    * ``span`` on an EXTRACTED row reproduces the stance word out of the
      entry's own text, exactly (``market_thesis``'s rule); a clicked row has
      no span, because a click is not a quotation.

``is_gradable(row) -> bool``
    ``False`` for ``no_view`` and for an ``unstated`` extraction. Those rows
    are RECORDED (the trader answered) and never graded right or wrong.

``pooled_accuracy(rows) -> dict``
    Raises ``PoolingError`` when the rows mix ``click`` with ``extracted``.
    Older extracted stances are never pooled with clicked predictions.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402


def test_a_clicked_prediction_is_the_read_and_the_words_are_never_consulted():
    """The trader clicked Down and typed a bullish sentence. Down is the read."""
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction="down",
        horizon="rest_of_day",
        timeframe="M5",
        text="strong tape, buyers holding the highs, bullish breakout",
        confidence="high",
        because="VXX is up with SPY",
    )

    rows = grades.read_rows([entry], session=fx.SESSION)

    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == grades.SOURCE_CLICK
    assert row["direction"] == "down"
    assert row["horizon"] == "rest_of_day"
    assert row["confidence"] == "high"
    assert row["because"] == "VXX is up with SPY"
    assert row["benchmark"] == grades.DEFAULT_BENCHMARK
    assert row["entry_id"] == entry["entry_id"]


def test_the_read_is_stamped_at_the_hour_it_was_answered_not_the_hour_it_was_asked():
    """A card scheduled at 07:00 and answered at 07:22 is a 07:22 read."""
    import market_read_grades as grades

    answered = fx.STAMP + timedelta(minutes=20)
    entry = fx.mentor_entry(
        direction="up", horizon="rest_of_day", timeframe="M5",
        created_at=fx.STAMP, responded_at=answered,
    )

    row = grades.read_rows([entry], session=fx.SESSION)[0]

    assert datetime.fromisoformat(row["stamp"]) == answered


@pytest.mark.parametrize("vintage", ["absent", "empty", "no_context", "context_v1"])
def test_every_older_row_vintage_is_labelled_extracted_and_never_a_click(vintage):
    """The four mentor shapes the live ledger actually holds. None is a click."""
    import market_read_grades as grades

    entry = fx.old_entry(vintage, text="the market is bearish, expecting lower")

    rows = grades.read_rows([entry], session=fx.SESSION)

    assert len(rows) == 1, f"{vintage} produced {len(rows)} rows"
    assert rows[0]["source"] == grades.SOURCE_EXTRACTED


def test_an_extracted_rows_span_reproduces_the_stance_word_out_of_the_trader_s_own_text():
    import market_read_grades as grades

    entry = fx.old_entry("empty", text="market is weak, expecting lower into the close")

    row = grades.read_rows([entry], session=fx.SESSION)[0]

    start, end = row["span"]
    assert entry["text"][start:end] == "weak"
    assert row["direction"] == "down"


def test_a_clicked_row_carries_no_span_because_a_click_is_not_a_quotation():
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction="up", horizon="rest_of_day", timeframe="M5",
        text="market is weak",
    )

    row = grades.read_rows([entry], session=fx.SESSION)[0]

    assert not row["span"]


def test_no_view_is_recorded_as_a_row_and_is_never_gradable():
    """A complete answer. It is kept, and it is never right or wrong."""
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction="no_view", horizon="rest_of_day", timeframe="M5", confidence="",
    )

    rows = grades.read_rows([entry], session=fx.SESSION)

    assert len(rows) == 1, "a No view answer must still be recorded"
    assert rows[0]["direction"] == "no_view"
    assert rows[0]["confidence"] == ""
    assert grades.is_gradable(rows[0]) is False


def test_an_unstated_extraction_is_recorded_and_is_never_gradable():
    import market_read_grades as grades

    entry = fx.old_entry("absent", text="I am not sure which way this goes today")

    rows = grades.read_rows([entry], session=fx.SESSION)

    assert len(rows) == 1
    assert grades.is_gradable(rows[0]) is False


def test_a_d1_click_grades_against_the_benchmark_the_prediction_names():
    """"opposite-benchmark notes never grade SPY" (plan.md TJ-10 item 1 tests)."""
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction="down", horizon="next_5_sessions", timeframe="D1",
        text="IWM looks heavy into next week",
    )

    row = grades.read_rows([entry], session=fx.SESSION)[0]

    assert row["benchmark"] == "IWM"


def test_pooling_a_clicked_row_with_an_extracted_one_raises():
    """Decision 0021 answer 29: an inferred stance is never pooled with a stated one."""
    import market_read_grades as grades

    clicked = grades.read_rows(
        [fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")],
        session=fx.SESSION,
    )
    extracted = grades.read_rows(
        [fx.old_entry("empty", text="market is weak into the close")],
        session=fx.SESSION,
    )

    # Each half on its own is fine.
    grades.pooled_accuracy(clicked)
    grades.pooled_accuracy(extracted)

    with pytest.raises(grades.PoolingError):
        grades.pooled_accuracy(clicked + extracted)


def test_the_same_entry_never_produces_both_a_click_and_an_extraction():
    """A card with a click AND words is ONE read, and the click is it."""
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction="chop", horizon="rest_of_day", timeframe="M5",
        text="SPY is weak and breaking down through the 50sma",
        confidence="low",
    )

    rows = grades.read_rows([entry], session=fx.SESSION)

    assert len(rows) == 1
    assert rows[0]["source"] == grades.SOURCE_CLICK
    assert rows[0]["direction"] == "chop"


def test_a_read_id_is_stable_and_distinguishes_two_calls_filed_in_one_second():
    """The D1 card files an M5 call and a D1 call with the same empty text."""
    import market_read_grades as grades

    m5 = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    d1 = fx.mentor_entry(direction="down", horizon="next_5_sessions", timeframe="D1")

    rows = grades.read_rows([m5, d1], session=fx.SESSION)

    ids = [row["read_id"] for row in rows]
    assert len(set(ids)) == 2, ids
    # Deterministic: the same entries read twice are the same two ids.
    assert [row["read_id"] for row in grades.read_rows([m5, d1], session=fx.SESSION)] == ids


def test_a_machine_row_is_never_a_read():
    """`is_machine_entry` is the ONE filter (TJ-1); an Auto Pilot flip said nothing."""
    import market_journal
    import market_read_grades as grades

    flip = market_journal.build_entry(
        text="Auto mode changed to AWAY",
        session_date=fx.SESSION,
        origin=market_journal.ORIGIN_AUTO_MODE_FLIP,
        now=fx.STAMP,
    )

    assert grades.read_rows([flip], session=fx.SESSION) == []
