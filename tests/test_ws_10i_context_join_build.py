"""Packet WS-10I - the BUILDER's added tests (never a replacement for the tester's).

`tests/test_ws_10i_context_join.py` is the tester's file and is not touched.
These are the seams the packet names that the tester's file does not pin,
written after the build and proven to fail on the pre-change file the same way:
the Bot x Day extension of the By environment control, the My-trades cut by the
ENTRY context, and the panel reading the day population from its OWN reader.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: The same calendar the tester's fixtures use - Labor Day 2026-09-07 sits
#: between the two sessions every prior-session answer hinges on.
LABELS = {
    "2026-09-02": "mixed",
    "2026-09-03": "mixed",
    "2026-09-04": "trending_up",
    "2026-09-08": "compressed",
    "2026-09-09": "mixed",
    "2026-09-10": "trending_down",
    "2026-09-11": "compressed",
}


def _m5_rows(session: str, symbols=("AAPL", "MSFT")) -> list[dict]:
    rows: list[dict] = []
    for symbol, mfe in zip(symbols, (1.4, 0.6)):
        stamp = session.replace("-", "")
        event_id = f"{symbol}_long_{stamp}_10_35_00_h1_blue_after_red"
        context = json.dumps({"market_environment": "quiet"})
        rows.append(
            {
                "event_id": event_id, "event_type": "registered",
                "trade_date": session, "symbol": symbol, "direction": "long",
                "entry_time": f"{session} 10:35:00", "bars_elapsed": "0",
                "minutes_elapsed": "", "stop_hit": "", "mfe_r": "",
                "context_json": context,
            }
        )
        rows.append(
            {
                "event_id": event_id, "event_type": "update",
                "trade_date": session, "symbol": symbol, "direction": "long",
                "entry_time": f"{session} 10:35:00", "bars_elapsed": "9",
                "minutes_elapsed": "45", "stop_hit": "", "mfe_r": str(mfe),
                "context_json": context,
            }
        )
    return rows


def test_the_bot_day_page_gets_its_own_environment_section_and_its_own_statistic():
    """Bot x Day is re-cut too (item 3), by the D1 label of the session, and it
    is a SECOND section with a SECOND key - a day cell is never a row inside the
    swing cut, because the two do not share a statistic."""
    import held_run_score
    import research_results

    rows = _m5_rows("2026-09-09") + _m5_rows("2026-09-08", symbols=("NVDA", "AMD"))
    view = research_results.build_results_view(
        population="bot",
        horizon="day",
        window="recent",
        snapshot={},
        as_of=date(2026, 9, 11),
        environment_rows=rows,
        environment_labels=LABELS,
        environment_filter="mixed",
    )

    keys = [section.key for section in view.sections]
    assert research_results.DAY_ENVIRONMENT_SECTION_KEY in keys
    assert research_results.ENVIRONMENT_SECTION_KEY not in keys, (
        "the swing cut rendered on a day page - the two populations were pooled"
    )
    section = [
        one for one in view.sections
        if one.key == research_results.DAY_ENVIRONMENT_SECTION_KEY
    ][0]
    assert section.rows, "the chosen environment has rows and the page showed none"
    assert {row.values["environment"] for row in section.rows} == {"mixed"}
    assert {row.values["statistic_name"] for row in section.rows} == {
        held_run_score.HELD_RUN_STATISTIC_NAME
    }
    assert view.environment_basis == research_results.ENVIRONMENT_BASIS_OBSERVATION


def test_my_trades_is_cut_by_the_environment_known_at_the_entry():
    """The My-trades cut is the ENTRY context, so an 11:02 fill on 2026-09-08
    reads the PREVIOUS session's label - and the trade that entered a
    `compressed` session by that rule is the one the page keeps."""
    import research_results

    trades = [
        {"trade_id": "t-trending", "symbol": "NVDA", "status": "CLOSED",
         "security_type": "STK", "direction": "LONG",
         "opened_at": "2026-09-08 11:02:00", "closed_at": "2026-09-10 15:50:00",
         "net_pnl": 100.0, "fees": 1.0, "commission": -1.0,
         "planned_risk": "50", "tag_status": "confirmed",
         "setup_tags": "avwap-reclaim"},
        {"trade_id": "t-compressed", "symbol": "AMD", "status": "CLOSED",
         "security_type": "STK", "direction": "LONG",
         "opened_at": "2026-09-09 11:02:00", "closed_at": "2026-09-11 15:50:00",
         "net_pnl": 200.0, "fees": 1.0, "commission": -1.0,
         "planned_risk": "50", "tag_status": "confirmed",
         "setup_tags": "avwap-reclaim"},
    ]

    kept = research_results._trades_in_environment(trades, "compressed", LABELS)
    assert [trade["trade_id"] for trade in kept] == ["t-compressed"]

    # No cut is every trade, and an unknown label table cuts to nothing rather
    # than guessing - missing data is uncertainty, never a match.
    assert len(research_results._trades_in_environment(trades, "all", LABELS)) == 2
    assert research_results._trades_in_environment(trades, "compressed", {}) == []


def test_the_control_offers_the_whole_vocabulary_and_defaults_to_no_cut():
    """The choices are the RULE's labels, not the labels that happen to have
    rows this window: a choice list that shrank with the data would hide the
    environments that stopped appearing, which is itself the finding."""
    import research_results
    from indicators.d1_environment import LABELS as RULE_LABELS

    assert research_results.ENVIRONMENT_CHOICES[0] == research_results.ENVIRONMENT_ALL
    assert set(RULE_LABELS) <= set(research_results.ENVIRONMENT_CHOICES)
    assert "unknown" in research_results.ENVIRONMENT_CHOICES

    view = research_results.build_results_view(
        population="bot", horizon="swing", window="recent", snapshot={},
        as_of=date(2026, 9, 11),
    )
    assert view.environment_filter == research_results.ENVIRONMENT_ALL
    assert view.environment_choices == research_results.ENVIRONMENT_CHOICES
    assert "SPY" in view.environment_line


def test_the_day_page_reads_the_day_file_and_never_the_swing_one(monkeypatch):
    """Two populations, two readers. The Bot x Day page must not open the 10 MB
    swing tier file, and the Bot x Swing page must not stream the intraday
    outcome log."""
    from ui.panels import research_results_panel as panel

    swing_calls: list[object] = []
    day_calls: list[object] = []

    monkeypatch.setattr(
        panel, "_read_environment_rows", lambda as_of: swing_calls.append(as_of) or []
    )
    monkeypatch.setattr(
        panel,
        "_read_day_environment_rows",
        lambda as_of: (day_calls.append(as_of) or [], dict(LABELS)),
    )
    monkeypatch.setattr(panel, "read_persisted_snapshot", lambda: {})
    monkeypatch.setattr(panel, "load_trades", lambda: [])

    payload = panel._read(("bot", "day", "recent", "compressed"), "recent", None)
    assert len(day_calls) == 1
    assert swing_calls == []
    assert payload["view"].environment_filter == "compressed"
    assert payload["selection"] == ("bot", "day", "recent", "compressed")

    panel._read(("bot", "swing", "recent", "all"), "recent", None)
    assert len(swing_calls) == 1
    assert len(day_calls) == 1


def test_an_environment_the_page_cannot_answer_says_so_rather_than_showing_the_page(
):
    """A cut with no rows is an empty section and an unchanged page above it -
    never the uncut numbers under a cut heading."""
    import research_results

    rows = [
        {"observation_id": "obs-1", "scan_date": "2026-09-08", "symbol": "AAA",
         "setup_family": "avwap_reclaim", "side": "LONG", "win": "True",
         "side_return_pct": "1.0", "stale_horizon": "False",
         "d1_environment": "compressed"},
    ]
    view = research_results.build_results_view(
        population="bot", horizon="swing", window="recent", snapshot={},
        as_of=date(2026, 9, 11), environment_rows=rows,
        environment_filter="trending_down",
    )
    section = [
        one for one in view.sections
        if one.key == research_results.ENVIRONMENT_SECTION_KEY
    ][0]
    assert section.rows == ()
    assert "trending_down" in section.title
    assert section.stats["environment_filter"] == "trending_down"
    assert section.stats["rows_read"] == 0


def test_the_environment_cut_never_reaches_a_promotion_or_a_policy_store():
    """plan.md sec 5, at the SURFACE seam this time: the control filters a
    readout and writes nothing anywhere."""
    import inspect

    import research_results

    source = inspect.getsource(research_results.build_results_view)
    source += inspect.getsource(research_results.day_environment_section)
    source += inspect.getsource(research_results._trades_in_environment)
    for forbidden in ("open(", "write", "review_policy", "focus_pick"):
        assert forbidden not in source, f"the Results view reaches {forbidden}"


@pytest.mark.parametrize("chosen", ["compressed", "unknown", "all"])
def test_the_cut_is_idempotent_and_the_page_is_the_same_twice(chosen):
    """The worker re-runs this on every redraw."""
    import research_results

    rows = [
        {"observation_id": f"obs-{index}", "scan_date": "2026-09-08",
         "symbol": f"S{index}", "setup_family": "avwap_reclaim", "side": "LONG",
         "win": "True" if index % 2 else "False", "side_return_pct": "1.0",
         "stale_horizon": "False", "d1_environment": "compressed"}
        for index in range(8)
    ]
    kwargs = dict(
        population="bot", horizon="swing", window="recent", snapshot={},
        as_of=date(2026, 9, 11), environment_filter=chosen,
    )
    first = research_results.build_results_view(environment_rows=list(rows), **kwargs)
    second = research_results.build_results_view(environment_rows=list(rows), **kwargs)
    assert [one.title for one in first.sections] == [one.title for one in second.sections]
    assert [
        row.line for section in first.sections for row in section.rows
    ] == [row.line for section in second.sections for row in section.rows]


# ---------------------------------------------------------------------------
# item 3 - the Daily Recap WIRING (WS-DR landed on the sweep branch 2026-09-13)
# ---------------------------------------------------------------------------


def test_the_recap_rows_carry_both_context_labels():
    """The packet's item 3, second half: WS-DR's recap row now carries the
    OBSERVATION context on every row and the ENTRY context beside it on a
    matched trade - two labels, never one blended into the other, and never an
    entry label on an opportunity nobody took."""
    import daily_recap_reader

    # The store labels 2026-09-10 and 2026-09-09; 2026-09-08 is unlabelled.
    labels = {"2026-09-10": "bullish_strong", "2026-09-09": "neutral"}

    # An intraday decision on 2026-09-10 could not know 2026-09-10's own label.
    intraday = daily_recap_reader._context_labels("2026-09-10 10:35:00", "", labels)
    assert intraday["observation"] == "neutral"
    assert intraday["observation_certainty"] == "prior_session"
    assert intraday["entry"] == ""

    # A swing scan row IS decided on that session's completed bars.
    swing = daily_recap_reader._context_labels("2026-09-09", "", labels)
    assert swing["observation"] == "neutral"
    assert swing["observation_certainty"] == "session"

    # A matched fill carries the entry context too - 10:20 PT is 13:20 ET, which
    # is before the close, so it reads the previous session and is NOT flagged.
    matched = daily_recap_reader._context_labels(
        "2026-09-10T09:50:11-07:00", "2026-09-10T10:20:00-07:00", labels
    )
    assert matched["entry"] == "neutral"
    assert matched["entry_certainty"] == "prior_session"
    assert matched["entry_flagged"] is False

    # A broker file's date-only fill is flagged and takes the previous session.
    date_only = daily_recap_reader._context_labels(
        "2026-09-10T09:50:11-07:00", "2026-09-10 00:00:00", labels
    )
    assert date_only["entry_certainty"] == "date_only"
    assert date_only["entry_flagged"] is True

    # The row carries the five fields beside - never instead of - WS-DR's own
    # session label, which answers a different question.
    row = daily_recap_reader.RecapRow(
        symbol="AAPL", side="LONG", source="pick_feedback", category="m5",
        capture_id="pf-1", observed_at=None, measures={}, unavailable={},
        detail={}, pick_key=("2026-09-10", "AAPL", "LONG", "m5"),
        d1_environment="bullish_strong",
        **daily_recap_reader._both_contexts(matched),
    )
    assert row.observation_context == "neutral"
    assert row.entry_context == "neutral"
    assert row.d1_environment == "bullish_strong"


def test_the_recap_environment_cell_shows_two_labels_only_where_there_is_a_fill():
    """The panel prints `observed -> entered` on a matched row and the
    observation alone on an unmatched one, with the session's own label and the
    certainties in the tooltip."""
    from ui.panels.daily_recap_panel import DailyRecapPanel

    class _Row:
        d1_environment = "bullish_strong"
        observation_context = "neutral"
        observation_certainty = "prior_session"
        entry_context = "compressed"
        entry_certainty = "date_only"
        entry_flagged = True

    class _Unmatched(_Row):
        entry_context = ""
        entry_certainty = ""
        entry_flagged = False

    cell = DailyRecapPanel._cell
    text, tip = cell(None, _Row(), "Environment", None)
    assert text == "neutral → compressed"
    assert "prior_session" in tip and "date_only" in tip
    assert "bullish_strong" in tip, "the session's own label is still readable"
    assert "not known" in tip, "a date-only fill says so"

    alone, alone_tip = cell(None, _Unmatched(), "Environment", None)
    assert alone == "neutral"
    assert "→" not in alone
    assert "entered in" not in alone_tip
