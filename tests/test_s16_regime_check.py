"""S16 item 5: regime change alerts are questions, not calls.

When the machine labels disagree with the trader's regime for 3 consecutive
sessions, the Mentor asks "still a <regime>?" once. Unknown never counts.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
TYPED = datetime(2026, 9, 1, 9, 0, tzinfo=PACIFIC)

BEAR = ("lh_ll", "bearish_weak", False)
UP = ("hh_hl", "bullish_weak", False)
UNKNOWN = ("unknown", "unknown", None)


def _store(tmp_path):
    from journal_store import JournalStore

    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    store.append_structural_regime(start_date="2026-08-01", regime="bear_channel_lower_highs", entered_at=TYPED)
    return store


def _row(day, shape, symbol="SPY"):
    channel, d1, compressed = shape
    return {
        "session_date": day,
        "symbol": symbol,
        "timeframes": {"D1": d1, "W": "bullish_weak"},
        "structure": {"daily_channel": {"label": channel}, "atr": {"compressed": compressed}},
    }


def _table(*days_and_shapes):
    return [_row(day, shape) for day, shape in days_and_shapes]


def _checks(lane, session):
    import mentor_questions

    state = {"session": session, "structural_regime": lane, "answered": dict(lane["answered"])}
    return [s for s in mentor_questions.pending(state, slot=None).asked if s.kind == "regime_check"]


def test_the_mapping_is_one_function_and_unknown_is_never_disagreement():
    import structural_regime as sr

    assert sr.machine_agrees(_row("2026-09-21", BEAR), "bear_channel_lower_highs") is True
    assert sr.machine_agrees(_row("2026-09-21", UP), "bear_channel_lower_highs") is False
    assert sr.machine_agrees(_row("2026-09-21", UNKNOWN), "bear_channel_lower_highs") is None
    # A channel and a D1 that disagree with each other are unknown, not a call.
    assert sr.machine_regime(_row("2026-09-21", ("lh_ll", "bullish_strong", False))) is None
    assert sr.machine_agrees(_row("2026-09-21", UP), "") is None
    assert set(sr.MACHINE_TO_TRADER) == set(sr.MACHINE_WORDS)
    for regimes in sr.MACHINE_TO_TRADER.values():
        assert set(regimes) <= set(sr.VOCABULARY)


def test_three_disagreeing_sessions_ask_still_a_regime(tmp_path):
    import structural_regime as sr

    store = _store(tmp_path)
    two = _table(("2026-09-21", BEAR), ("2026-09-22", UP), ("2026-09-23", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-09-23", two)
    assert lane["disagreement"] is None
    assert _checks(lane, "2026-09-23") == []

    three = two + _table(("2026-09-24", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-09-24", three)
    asked = _checks(lane, "2026-09-24")
    assert len(asked) == 1
    subject = asked[0]
    assert subject.subject_id == "2026-09-22"
    assert subject.prompt.startswith("Still a bear channel, lower highs?")
    assert "3 sessions since 2026-09-22" in subject.prompt
    assert subject.options[0] == "still_bear_channel_lower_highs"


def test_an_unknown_session_breaks_the_count(tmp_path):
    import structural_regime as sr

    store = _store(tmp_path)
    table = _table(("2026-09-22", UP), ("2026-09-23", UNKNOWN), ("2026-09-24", UP), ("2026-09-25", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-09-25", table)
    assert lane["disagreement"] is None
    assert _checks(lane, "2026-09-25") == []


def test_it_is_asked_once_per_run_until_answered_or_the_labels_agree(tmp_path):
    import mentor_questions
    import structural_regime as sr

    store = _store(tmp_path)
    base = _table(("2026-09-22", UP), ("2026-09-23", UP), ("2026-09-24", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-09-24", base)
    first = _checks(lane, "2026-09-24")[0]

    # A fourth disagreeing session is the same run: the same one question.
    longer = base + _table(("2026-09-25", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-09-25", longer)
    assert [s.subject_id for s in _checks(lane, "2026-09-25")] == [first.subject_id]

    # The labels agree again: the run is closed and a carried question is dropped.
    agreed = longer + _table(("2026-09-28", BEAR))
    lane = sr.lane(store.list_structural_regime(), "2026-09-28", agreed)
    assert lane["disagreement"] is None
    assert f"regime_check:{first.subject_id}" in lane["answered"]
    state = {"session": "2026-09-28", "structural_regime": lane, "answered": lane["answered"], "carried": (first,)}
    assert not [s for s in mentor_questions.pending(state, slot=None).asked if s.kind == "regime_check"]

    # A new run of three after that asks again, about the new run.
    again = agreed + _table(("2026-09-29", UP), ("2026-09-30", UP), ("2026-10-01", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-10-01", again)
    assert [s.subject_id for s in _checks(lane, "2026-10-01")] == ["2026-09-29"]


def test_the_answer_keeps_the_run_and_resets_the_count(tmp_path):
    import mentor_questions
    import structural_regime as sr

    store = _store(tmp_path)
    table = _table(("2026-09-22", UP), ("2026-09-23", UP), ("2026-09-24", UP))
    lane = sr.lane(store.list_structural_regime(), "2026-09-24", table)
    subject = _checks(lane, "2026-09-24")[0]
    answered_at = datetime(2026, 9, 24, 10, 0, tzinfo=PACIFIC)
    outcome = mentor_questions.record_answer(
        subject, {"state": "still_bear_channel_lower_highs"}, store=store, now=answered_at
    )
    assert outcome["ok"] is True
    rows = store.list_structural_regime()
    assert rows[-1]["supersedes"] == rows[0]["segment_id"]
    assert sr.current_regime(rows, "2026-09-24")["start_date"] == "2026-08-01"

    # Still disagreeing the next two sessions: the answer reset the count, so no question.
    later = table + _table(("2026-09-25", UP), ("2026-09-28", UP))
    lane = sr.lane(rows, "2026-09-28", later)
    assert lane["disagreement"] is None
    assert f"regime_check:{subject.subject_id}" in lane["answered"]
    assert _checks(lane, "2026-09-28") == []


def test_the_lane_reads_the_regime_table_file(tmp_path):
    import market_regimes
    import structural_regime as sr

    store = _store(tmp_path)
    path = tmp_path / "market_regime_table.jsonl"
    market_regimes.append_rows(path, _table(("2026-09-22", UP), ("2026-09-23", UP), ("2026-09-24", UP)))
    lane = sr.load_lane(store, "2026-09-24", table_path=path)
    assert lane["disagreement"]["start"] == "2026-09-22"
    assert lane["disagreement"]["streak"] == 3
