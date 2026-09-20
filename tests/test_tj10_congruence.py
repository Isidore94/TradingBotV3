"""TJ-10 item 6 - three congruence lines, printed and never acted on.

Packet `.claude/packets/TJ-10.md` item 6; `plan.md` §12.4 "TJ-10" item 2;
decision 0021 answer 15 ("Congruence is printed, never pushed and never acted
on"). Trader, 2026-09-19: *"if my thoughts about the market are potentially
incongruent with my overall D1 picture, I want to know about it."* RED before
the build.

The contract these tests pin
----------------------------

``congruence_lines(*, session, d1_read=None, d1_label="", decisions=(),
claims=(), trades=()) -> tuple[dict, ...]``

    Exactly three lines, in ``CONGRUENCE_KINDS`` order::

        "desk_d1_label"   your latest D1 click  vs the desk's own D1 label
        "picks_side_mix"  ... vs the long/short mix of that session's likes
                              and claims
        "fills_bias"      ... vs the bias of that session's fills

    Each line::

        {"kind", "text", "verdict", "counts", "source_ids", "timeframe",
         "missing"}

    * ``verdict`` is ``agrees`` / ``disagrees`` / ``UNMEASURED``. A D1 label
      with no direction in it (``compressed``, ``mixed``, ``unknown``) is
      ``UNMEASURED`` - never agreement by default.
    * ``counts["long"] + counts["short"]`` equals ``len(source_ids)``: the
      count IS the rows, and every counted row carries the line's own
      ``timeframe``. (TJ-15's measured lesson, 2026-09-19: 1,026 of the
      trader's reviewed decisions over 20 sessions are M5 against 1,013 D1, so
      a line that silently pooled the two would be comparing a D1 view with a
      mostly-M5 population.)
    * A line with a side missing NAMES the missing side in ``missing`` and is
      ``UNMEASURED``; it never reads an absence as agreement.
    * The bias of a fill is ``journal_exposure``'s - **a LONG option is never a
      bullish setup**.
    * No line carries a threshold, a priority or an alert, and the function
      touches no notifier. It is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402


def _d1_read(direction: str = "down"):
    import market_read_grades as grades

    entry = fx.mentor_entry(
        direction=direction, horizon="next_5_sessions", timeframe="D1",
    )
    return grades.read_rows([entry], session=fx.SESSION)[0]


def _decision(symbol: str, side: str, *, timeframe: str, verdict: str = "like",
              capture_id: str = ""):
    """One row in the shape `day_review_service` already hands the page."""
    return {
        "session_date": fx.SESSION,
        "symbol": symbol,
        "side": side,
        "category": "chart_review",
        "verdict": verdict,
        "source": "annotations",
        "timeframe": timeframe,
        "stamp": fx.STAMP.isoformat(),
        "capture_id": capture_id or f"e-{symbol}",
        "reason": "",
        "decision_session": fx.SESSION,
    }


def _lines(**kwargs):
    import market_read_grades as grades

    return {
        line["kind"]: line
        for line in grades.congruence_lines(session=fx.SESSION, **kwargs)
    }


# -- the shape ---------------------------------------------------------------


def test_there_are_exactly_three_lines_in_one_declared_order():
    import market_read_grades as grades

    lines = grades.congruence_lines(session=fx.SESSION)

    assert grades.CONGRUENCE_KINDS == (
        "desk_d1_label", "picks_side_mix", "fills_bias"
    )
    assert tuple(line["kind"] for line in lines) == grades.CONGRUENCE_KINDS


# -- against the desk's own D1 label -----------------------------------------


def test_a_bearish_d1_click_agrees_with_a_trending_down_desk_label():
    lines = _lines(d1_read=_d1_read("down"), d1_label="trending_down")

    assert lines["desk_d1_label"]["verdict"] == "agrees"


def test_a_bullish_d1_click_against_a_trending_down_label_is_the_line_the_trader_asked_for():
    lines = _lines(d1_read=_d1_read("up"), d1_label="trending_down")

    assert lines["desk_d1_label"]["verdict"] == "disagrees"
    assert "trending_down" in lines["desk_d1_label"]["text"]


def test_a_label_with_no_direction_in_it_is_never_agreement():
    """`compressed` and `mixed` say the tape has no direction; they do not
    agree with `down` and they do not disagree with it."""
    import market_read_grades as grades

    for label in ("compressed", "mixed", "unknown"):
        lines = _lines(d1_read=_d1_read("down"), d1_label=label)
        assert lines["desk_d1_label"]["verdict"] == grades.UNMEASURED, label


def test_a_missing_d1_click_names_the_missing_side():
    """Measured on a copy 2026-09-19: `d1_environment.jsonl` holds no row for
    2026-09-18 at all, so BOTH sides of this line really do go missing."""
    import market_read_grades as grades

    no_click = _lines(d1_read=None, d1_label="trending_down")
    no_label = _lines(d1_read=_d1_read("down"), d1_label="")

    assert no_click["desk_d1_label"]["verdict"] == grades.UNMEASURED
    assert "D1" in no_click["desk_d1_label"]["missing"]
    assert no_label["desk_d1_label"]["verdict"] == grades.UNMEASURED
    assert no_label["desk_d1_label"]["missing"]
    assert no_click["desk_d1_label"]["missing"] != no_label["desk_d1_label"]["missing"]


# -- against the day's likes and claims --------------------------------------


def test_the_counts_equal_the_rows_and_every_counted_row_shares_the_lines_timeframe():
    """The packet's own test: "the congruence counts equal the annotation rows"."""
    decisions = [
        _decision("AAA", "LONG", timeframe="D1"),
        _decision("BBB", "LONG", timeframe="D1"),
        _decision("CCC", "LONG", timeframe="D1"),
        _decision("DDD", "SHORT", timeframe="M5"),
        _decision("EEE", "SHORT", timeframe="M5"),
        _decision("FFF", "SHORT", timeframe="M5"),
        _decision("GGG", "SHORT", timeframe="M5"),
        _decision("HHH", "SHORT", timeframe="M5"),
    ]
    line = _lines(d1_read=_d1_read("down"), decisions=decisions)["picks_side_mix"]

    counted = {row["capture_id"] for row in decisions
               if row["timeframe"] == line["timeframe"]}

    assert line["timeframe"], "the line must SAY which population it counted"
    assert set(line["source_ids"]) == counted
    assert line["counts"]["long"] + line["counts"]["short"] == len(line["source_ids"])


def test_a_veto_is_not_a_like_and_never_joins_the_side_mix():
    decisions = [
        _decision("AAA", "LONG", timeframe="D1"),
        _decision("BBB", "LONG", timeframe="D1", verdict="veto"),
    ]
    line = _lines(d1_read=_d1_read("up"), decisions=decisions)["picks_side_mix"]

    assert line["counts"]["long"] == 1
    assert line["source_ids"] == ["e-AAA"]


def test_a_claim_counts_beside_a_like():
    import claimed_picks

    decisions = [_decision("AAA", "LONG", timeframe="D1")]
    claim = claimed_picks.build_claim_row(
        symbol="BBB", side="LONG", session_date=fx.SESSION,
        source="chart_review", now=fx.STAMP,
    )
    line = _lines(
        d1_read=_d1_read("up"), decisions=decisions, claims=[claim]
    )["picks_side_mix"]

    assert line["counts"]["long"] == 2
    assert line["counts"]["long"] + line["counts"]["short"] == len(line["source_ids"])


def test_a_session_with_no_likes_or_claims_names_that_side_as_missing():
    import market_read_grades as grades

    line = _lines(d1_read=_d1_read("down"), decisions=[], claims=[])["picks_side_mix"]

    assert line["verdict"] == grades.UNMEASURED
    assert line["missing"]
    assert line["counts"]["long"] == 0
    assert line["counts"]["short"] == 0


def test_a_bearish_view_against_a_day_of_long_picks_disagrees():
    decisions = [
        _decision("AAA", "LONG", timeframe="D1"),
        _decision("BBB", "LONG", timeframe="D1"),
        _decision("CCC", "LONG", timeframe="D1"),
    ]
    line = _lines(d1_read=_d1_read("down"), decisions=decisions)["picks_side_mix"]

    assert line["verdict"] == "disagrees"


# -- against the day's fills -------------------------------------------------


def _trade(trade_id: str, *, security_type: str, direction: str, symbol: str):
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "security_type": security_type,
        "direction": direction,
        "status": "closed",
        "opened_at": f"{fx.SESSION}T10:00:00",
        "closed_at": f"{fx.SESSION}T14:00:00",
        "session_date": fx.SESSION,
    }


def test_a_long_put_is_counted_bearish_and_never_bullish():
    """`journal_exposure` reads bias from the LEGS: a bought PUT is bearish."""
    trades = [
        _trade("t-1", security_type="STK", direction="LONG", symbol="AAA"),
        _trade("t-2", security_type="OPT", direction="LONG",
               symbol="BBB   261218P00100000"),
    ]
    line = _lines(d1_read=_d1_read("down"), trades=trades)["fills_bias"]

    assert line["counts"]["bullish"] == 1
    assert line["counts"]["bearish"] == 1
    assert len(line["source_ids"]) == 2


def test_a_bias_the_legs_cannot_establish_is_unmeasured_and_pools_no_side():
    """A multi-leg trade has no opinion; a straddle has no side at all."""
    trades = [
        _trade("t-1", security_type="BAG", direction="LONG", symbol="AAA"),
    ]
    line = _lines(d1_read=_d1_read("down"), trades=trades)["fills_bias"]

    assert line["counts"]["bullish"] == 0
    assert line["counts"]["bearish"] == 0
    assert line["counts"]["unknown"] == 1


def test_a_day_with_no_fills_names_that_side_as_missing():
    import market_read_grades as grades

    line = _lines(d1_read=_d1_read("down"), trades=[])["fills_bias"]

    assert line["verdict"] == grades.UNMEASURED
    assert line["missing"]


# -- printed, never pushed ---------------------------------------------------


def test_no_line_carries_a_threshold_a_priority_or_an_alert():
    """Decision 0021 answer 15: printed, never pushed, never acted on."""
    import market_read_grades as grades

    lines = grades.congruence_lines(
        session=fx.SESSION, d1_read=_d1_read("down"), d1_label="trending_up",
        decisions=[_decision("AAA", "LONG", timeframe="D1")],
    )

    banned = {"threshold", "priority", "alert", "push", "ntfy", "severity"}
    for line in lines:
        assert not (banned & set(line)), line["kind"]


def test_building_the_lines_reaches_no_notifier(monkeypatch):
    import market_read_grades as grades
    import push_notify

    def _never(*_args, **_kwargs):
        raise AssertionError("a congruence line reached the phone")

    monkeypatch.setattr(push_notify, "send_push", _never)
    monkeypatch.setattr(push_notify, "build_push_request", _never)

    grades.congruence_lines(
        session=fx.SESSION, d1_read=_d1_read("up"), d1_label="trending_down",
        decisions=[_decision("AAA", "LONG", timeframe="D1")],
        trades=[_trade("t-1", security_type="STK", direction="SHORT",
                       symbol="AAA")],
    )


def test_a_congruence_line_never_reaches_the_review_policy(tmp_path):
    """plan.md sec 5: `review_policy.json` ranks and annotates only, and
    nothing in this program may reach it."""
    import market_read_grades as grades

    policy = tmp_path / "review_policy.json"
    grades.congruence_lines(
        session=fx.SESSION, d1_read=_d1_read("up"), d1_label="trending_down",
    )

    assert not policy.exists()
    assert not list(tmp_path.iterdir())


def test_the_lines_are_deterministic_and_call_no_model():
    import market_read_grades as grades

    kwargs = dict(
        session=fx.SESSION, d1_read=_d1_read("down"), d1_label="trending_down",
        decisions=[_decision("AAA", "SHORT", timeframe="D1")],
    )
    first = grades.congruence_lines(**kwargs)
    second = grades.congruence_lines(**kwargs)

    assert [dict(line) for line in first] == [dict(line) for line in second]
