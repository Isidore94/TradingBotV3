"""TJ-10 follow-up - three reviewer advisories on the congruence lines, fixed
before the merge (lead, 2026-09-20). RED against tip 14bcc894.

1. **An under-floor side mix is never a verdict.** Live 2026-09-17: a 2-of-2
   M5 side mix printed `disagrees` at full weight, and a chip or a later
   pooling packet keys on `verdict`, not the "too few to call" words beside
   it. Under `evidence_stats.MIN_REPORTABLE_N` the verdict is now
   `market_read_grades.VERDICT_TOO_FEW`; the counts and the floor note in the
   text are unchanged, and at or above the floor nothing changes.
2. **Each line keeps its own content on a both-ways day.** The reviewer's
   2026-09-17 shape: one extracted note read `up`, another read `down`, so
   `select_read` refuses to pick either one. Every line used to drop its own
   counts and print the bare contradiction sentence; now it says its own mix
   or label FIRST and the reason there is nothing to compare it with second.
3. **A pending five-session read never shows a forming close as its anchor.**
   While the decision session is still trading, `anchor_price` is `None` and
   the row stays `pending <date>` rather than reading a still-open bar.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")


def _decision(symbol: str, side: str, *, timeframe: str, verdict: str = "like"):
    return {
        "session_date": fx.SESSION, "symbol": symbol, "side": side,
        "category": "chart_review", "verdict": verdict, "source": "annotations",
        "timeframe": timeframe, "stamp": fx.STAMP.isoformat(),
        "capture_id": f"e-{symbol}-{timeframe}", "reason": "",
        "decision_session": fx.SESSION,
    }


def _extracted(text: str, *, timeframe: str = "D1", minutes: int = 0):
    return fx.old_entry(
        "empty", text=text, timeframe=timeframe,
        created_at=fx.STAMP + timedelta(minutes=minutes),
    )


# -- item 1: an under-floor picks verdict is `too_few`, never a verdict -----


def test_a_two_of_two_m5_side_mix_is_too_few_not_a_full_weight_disagrees():
    """The reviewer's live 2026-09-17 shape: 2 M5 likes, both SHORT, against
    an `up` read - the desk printed `disagrees` at full weight."""
    import market_read_grades as grader

    m5 = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    m5_read = grader.read_rows([m5], session=fx.SESSION)[0]
    decisions = [
        _decision("AAA", "SHORT", timeframe="M5"),
        _decision("BBB", "SHORT", timeframe="M5"),
    ]

    line = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, m5_read=m5_read, decisions=decisions,
        )
    }[grader.CONGRUENCE_M5_KIND]

    assert line["verdict"] == grader.VERDICT_TOO_FEW
    assert line["verdict"] != "disagrees"
    assert "2 of 2" in line["text"]
    assert "too few to call" in line["text"]
    assert "(n=2, under 30)" in line["text"]
    # The counts themselves are untouched by the floor.
    assert line["counts"] == {"long": 0, "short": 2, "not_today": 0}


def test_at_the_floor_the_picks_verdict_is_agrees_or_disagrees_as_before():
    """n == MIN_REPORTABLE_N (30): the old behaviour is unchanged."""
    import evidence_stats
    import market_read_grades as grader

    assert evidence_stats.MIN_REPORTABLE_N == 30

    m5 = fx.mentor_entry(direction="up", horizon="rest_of_day", timeframe="M5")
    m5_read = grader.read_rows([m5], session=fx.SESSION)[0]
    decisions = [
        _decision(f"SYM{i}", "LONG", timeframe="M5") for i in range(30)
    ]

    line = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, m5_read=m5_read, decisions=decisions,
        )
    }[grader.CONGRUENCE_M5_KIND]

    assert line["counts"]["long"] == 30
    assert line["verdict"] == "agrees"
    assert "too few to call" not in line["text"]


def test_a_d1_side_mix_below_the_floor_is_too_few_too():
    """Item 1 applies to BOTH picks lines, D1 as well as M5."""
    import market_read_grades as grader

    entry = fx.mentor_entry(direction="down", horizon="next_5_sessions", timeframe="D1")
    d1_read = grader.read_rows([entry], session=fx.SESSION)[0]
    decisions = [
        _decision("AAA", "LONG", timeframe="D1"),
        _decision("BBB", "LONG", timeframe="D1"),
        _decision("CCC", "LONG", timeframe="D1"),
    ]

    line = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, d1_read=d1_read, decisions=decisions,
        )
    }["picks_side_mix"]

    assert line["verdict"] == grader.VERDICT_TOO_FEW
    assert "3 of 3" in line["text"]
    assert "too few to call" in line["text"]


# -- item 2: each line keeps its own content on a both-ways day -------------


def test_the_reviewer_20260917_shape_every_line_says_its_own_content_first():
    """One extracted note up, one down; 59 LONG / 31 SHORT D1 likes and
    claims; the desk's label is `trending_down`. Every line names its own
    mix or label FIRST and the both-ways reason SECOND, and every verdict
    stays `unmeasured` - a both-ways day can never be agreement or
    disagreement."""
    import market_read_grades as grader

    rows = grader.read_rows(
        [
            _extracted("SPY is uptrending off the lows"),
            _extracted("D1 SPY is downtrending", minutes=120),
        ],
        session=fx.SESSION,
    )
    read, note = grader.select_read(rows, timeframe="D1")
    assert read is None
    assert note == "your notes read both ways (1 up, 1 down) - no single read to compare"

    decisions = [
        _decision(f"L{i}", "LONG", timeframe="D1") for i in range(59)
    ] + [
        _decision(f"S{i}", "SHORT", timeframe="D1") for i in range(31)
    ]

    lines = {
        row["kind"]: row for row in grader.congruence_lines(
            session=fx.SESSION, d1_read=None, d1_note=note,
            d1_label="trending_down", decisions=decisions,
        )
    }

    picks_text = lines["picks_side_mix"]["text"]
    assert picks_text.startswith(
        "59 of 90 D1 likes and claims were LONG (long 59, short 31)"
    )
    assert "both ways" in picks_text
    assert note in picks_text

    label_text = lines["desk_d1_label"]["text"]
    assert label_text.startswith("the desk reads trending_down")
    assert "trending_down" in label_text
    assert note in label_text

    assert lines["desk_d1_label"]["verdict"] == grader.UNMEASURED
    assert lines["picks_side_mix"]["verdict"] == grader.UNMEASURED
    assert lines["fills_bias"]["verdict"] == grader.UNMEASURED


# -- item 3: a pending five-session read never anchors on a forming close ---


def test_a_still_trading_decision_session_has_no_anchor_price_yet():
    """`_grade_five_sessions`, `now` still inside the decision session's own
    trading hours. `fx.SESSION` (2026-09-18) closes at 13:00 Pacific; 10:00
    is still open. A daily bar dated 09-18 with a canary close of 130.0 must
    never surface as the anchor - it is a forming bar, not a close."""
    import market_read_grades as grader

    row = fx.mentor_entry(
        direction="up", horizon="next_5_sessions", timeframe="D1",
    )
    read = grader.read_rows([row], session=fx.SESSION)[0]
    daily = fx.daily_bars({fx.SESSION: 130.0})

    grade = grader.grade_read(
        read, daily_bars=daily, atr=fx.atr_for(6.0, 3.0),
        now=datetime(2026, 9, 18, 10, 0, tzinfo=PACIFIC),
    )

    assert grade["anchor_price"] is None
    assert grade["verdict"].startswith(grader.PENDING_PREFIX)
    assert grade["verdict"] == f"pending {fx.SESSION}"
    assert grade["grader_gap"] == "the decision session has not closed yet"


def test_once_the_decision_session_closes_the_anchor_is_that_sessions_close():
    """The same read, graded again after `fx.SESSION` has actually closed:
    the anchor is now that session's own daily close, exactly as before this
    fix - the rule only cuts a FORMING bar, never a closed one."""
    import market_read_grades as grader

    row = fx.mentor_entry(
        direction="up", horizon="next_5_sessions", timeframe="D1",
    )
    read = grader.read_rows([row], session=fx.SESSION)[0]
    daily = fx.daily_bars({fx.SESSION: 100.0})

    grade = grader.grade_read(
        read, daily_bars=daily, atr=fx.atr_for(6.0, 3.0),
        now=fx.AFTER_THE_CLOSE,
    )

    assert grade["anchor_price"] == 100.0
    assert grade["anchor_at"] == fx.SESSION
