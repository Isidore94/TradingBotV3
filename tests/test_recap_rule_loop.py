"""Day Recap coach: today's rule on the prep page, the desk chip, and the Mentor's reflection."""

from __future__ import annotations

import sys
import threading
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ET = timezone(timedelta(hours=-4))
SESSION = "2026-09-22"
NEXT = "2026-09-23"
EVENING = datetime(2026, 9, 22, 17, 30, tzinfo=ET)


@pytest.fixture()
def recap_file(tmp_path, monkeypatch):
    import project_paths

    path = tmp_path / "day_recap_events.jsonl"
    monkeypatch.setattr(project_paths, "DAY_RECAP_EVENTS_FILE", path)
    return path


def _rule(tag="hold_winners", text="hold winners to 1R", for_date=SESSION, streak=2):
    return {"for_date": for_date, "rule_id": "rc-1", "set_on": "2026-09-21", "text": text, "tag": tag, "streak": streak}


def _trade(trade_id, symbol="ZETA", *, opened="09:50", closed="11:00", net=50.0, cad=None, risk=None,
           entry=10.0, exit_=10.06, qty=100, side="LONG", status="CLOSED", day=SESSION):
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "direction": side,
        "status": status,
        "opened_at": f"{day}T{opened}:00-04:00" if opened else "",
        "closed_at": f"{day}T{closed}:00-04:00" if closed else "",
        "trade_date": day,
        "net_pnl": net,
        "net_pnl_cad": cad,
        "planned_risk": risk,
        "average_entry_price": entry,
        "average_exit_price": exit_,
        "quantity_opened": qty,
    }


# ---------------------------------------------------------------------------
# 1. the prep page / output
# ---------------------------------------------------------------------------
def test_prep_shows_todays_rule_with_its_streak(recap_file):
    import recap_rule_loop as loop
    import recap_store

    recap_store.record_rule(session_date="2026-09-21", text="hold winners to 1R", tag="hold_winners",
                            now=datetime(2026, 9, 21, 17, tzinfo=ET))
    recap_store.record_rule_check(session_date="2026-09-21", answer="yes", now=datetime(2026, 9, 21, 17, 5, tzinfo=ET))

    info = loop.rule_for_date(SESSION)
    assert loop.prep_line(info) == "Today's rule: hold winners to 1R (streak 1)"

    from market_prep.orchestrator import _todays_rule_line
    from market_prep.report_builder import build_daily_report_object

    line = _todays_rule_line(datetime(2026, 9, 22).date())
    report = build_daily_report_object(report_date=SESSION, todays_rule_line=line)
    lines = report["markdown"].splitlines()
    assert lines[0].startswith("# Daily Market Prep")
    assert lines[2] == "**Today's rule: hold winners to 1R (streak 1)**"


def test_prep_shows_nothing_when_there_is_no_rule(recap_file):
    import recap_rule_loop as loop
    from market_prep.orchestrator import _todays_rule_line
    from market_prep.report_builder import build_daily_report_object

    assert loop.rule_for_date(SESSION) is None
    assert loop.prep_line(None) == ""
    line = _todays_rule_line(datetime(2026, 9, 22).date())
    assert line == ""
    markdown = build_daily_report_object(report_date=SESSION, todays_rule_line=line)["markdown"]
    assert "rule" not in markdown.lower().split("## 1.")[0]


def test_a_rule_written_tonight_is_tomorrows_not_todays(recap_file):
    import recap_rule_loop as loop
    import recap_store

    recap_store.record_rule(session_date=SESSION, text="respect the stop", tag="respect_stop", now=EVENING)
    assert loop.rule_for_date(SESSION) is None
    assert loop.rule_for_date(NEXT)["text"] == "respect the stop"


# ---------------------------------------------------------------------------
# 2. the desk chip
# ---------------------------------------------------------------------------
@pytest.fixture()
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait_for(chip, qapp, timeout=5.0):
    import time

    got = []
    chip.ruleLoaded.connect(lambda info: got.append(info))
    deadline = time.monotonic() + timeout
    while not got and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return got


def test_the_chip_reads_off_the_qt_thread_and_shows_the_streak_in_its_tooltip(qapp):
    from ui.widgets.rule_chip import RuleChip

    seen = []

    def loader():
        seen.append(threading.get_ident())
        return _rule()

    chip = RuleChip(loader=loader, clock=lambda: EVENING)
    chip.start()
    got = _wait_for(chip, qapp)
    chip.shutdown()
    assert got and seen
    assert seen[0] != threading.get_ident()
    assert chip.text() == "Rule: hold winners to 1R"
    assert "Kept 2 session(s) in a row." in chip.toolTip()
    assert not chip.isHidden()


def test_the_chip_hides_without_a_rule_and_rereads_when_the_date_changes(qapp):
    from ui.widgets.rule_chip import RuleChip

    calls = []
    clock = {"now": EVENING}

    def loader():
        calls.append(clock["now"])
        return None

    chip = RuleChip(loader=loader, clock=lambda: clock["now"])
    chip.start()
    assert _wait_for(chip, qapp) == [None]
    assert chip.isHidden()
    chip._check_date()  # same market date: no read
    clock["now"] = EVENING + timedelta(hours=8)  # 01:30 ET next day
    chip._check_date()
    _wait_for(chip, qapp)
    chip.shutdown()
    assert len(calls) == 2


def test_the_chip_never_reads_a_file_on_construction(qapp, monkeypatch):
    import recap_rule_loop
    from ui.widgets.rule_chip import RuleChip

    def boom(*_a, **_k):
        raise AssertionError("read on construction")

    monkeypatch.setattr(recap_rule_loop, "load_today_rule", boom)
    chip = RuleChip()
    assert chip.isHidden() and chip.info() is None


def test_the_prep_page_banner_words_it_as_todays_rule(qapp):
    from ui.widgets.rule_chip import RuleChip

    banner = RuleChip(banner=True, loader=lambda: _rule(), clock=lambda: EVENING)
    banner.start()
    _wait_for(banner, qapp)
    banner.shutdown()
    assert banner.text() == "Today's rule: hold winners to 1R (streak 2)"


# ---------------------------------------------------------------------------
# 3. which closed trades the rule is checked on
# ---------------------------------------------------------------------------
def _rows(rule, trades, **kwargs):
    import recap_rule_loop as loop

    return loop.reflection_rows(rule, trades, session=SESSION, **kwargs)


def test_hold_winners_uses_r_when_planned_risk_is_known():
    rows = _rows(_rule(), [
        _trade("t1", "ZETA", net=60, cad=60, risk=100),   # +0.6R: asked
        _trade("t2", "BETA", net=150, cad=150, risk=100),  # +1.5R: kept
        _trade("t3", "GAMA", net=-40, cad=-40, risk=100),  # a loser: not this rule
    ])
    assert [row["trade_id"] for row in rows] == ["t1"]
    assert rows[0]["prompt"] == (
        "Your rule today: hold winners. You closed ZETA at +0.6R — kept it or broke it?"
    )


def test_hold_winners_falls_back_to_the_days_median_winner_pct():
    trades = [
        _trade("a", "AAA", exit_=10.05, closed="10:00"),  # +0.5%
        _trade("b", "BBB", exit_=10.20, closed="10:30"),  # +2.0%
        _trade("c", "CCC", exit_=10.30, closed="11:00"),  # +3.0%
    ]
    rows = _rows(_rule(), trades)
    assert [row["trade_id"] for row in rows] == ["a"]
    assert "under the day's median winner (+2.0%)" in rows[0]["prompt"]


def test_hold_winners_with_too_few_winners_and_no_risk_asks_nothing():
    assert _rows(_rule(), [_trade("a", exit_=10.05), _trade("b", exit_=10.30)]) == []


def test_respect_stop_asks_about_a_loss_beyond_planned_risk_only():
    rule = _rule("respect_stop", "respect the stop")
    rows = _rows(rule, [
        _trade("t1", "ZETA", net=-140, cad=-140, risk=100),
        _trade("t2", "BETA", net=-80, cad=-80, risk=100),
        _trade("t3", "GAMA", net=-300, cad=None, risk=100),  # R unknown
        _trade("t4", "DELT", net=-300, cad=-300, risk=None),  # risk unknown
    ])
    assert [row["trade_id"] for row in rows] == ["t1"]
    assert "-1.4R, past your planned risk" in rows[0]["prompt"]


def test_no_trade_first_15m_asks_about_an_entry_before_0945():
    rule = _rule("no_trade_first_15m", "no trades before 9:45")
    rows = _rows(rule, [
        _trade("t1", "ZETA", opened="09:38", closed="10:00"),
        _trade("t2", "BETA", opened="09:45", closed="10:10"),
        _trade("t3", "GAMA", opened="00:00", closed="10:20"),  # date-only stamp: unknown
    ])
    assert [row["trade_id"] for row in rows] == ["t1"]
    assert "You entered ZETA at 09:38" in rows[0]["prompt"]


def test_size_down_in_chop_needs_a_bigger_size_and_a_chop_reading_before_entry():
    rule = _rule("size_down_in_chop", "size down in chop")
    timeline = [(datetime(2026, 9, 22, 9, 35, tzinfo=ET), "neutral_chop"),
                (datetime(2026, 9, 22, 11, 0, tzinfo=ET), "bullish_weak")]
    trades = [
        _trade("big_chop", "ZETA", opened="10:00", closed="10:30", qty=500),   # $5,000 in chop
        _trade("small_chop", "BETA", opened="10:05", closed="10:40", qty=100),  # $1,000
        _trade("big_bull", "GAMA", opened="11:30", closed="12:00", qty=500),   # not chop
    ]
    rows = _rows(rule, trades, size_median=2000.0, regime_timeline=timeline)
    assert [row["trade_id"] for row in rows] == ["big_chop"]
    assert "neutral chop" in rows[0]["prompt"]
    # No regime seen before the entry, or no size baseline: unknown, no ask.
    assert _rows(rule, trades, size_median=2000.0, regime_timeline=[]) == []
    assert _rows(rule, trades, size_median=None, regime_timeline=timeline) == []


def test_size_baseline_uses_earlier_sessions_only_and_needs_five():
    import recap_rule_loop as loop
    from datetime import date

    earlier = [_trade(f"e{i}", qty=100 + i * 10, day="2026-09-1" + str(i)) for i in range(5)]
    today = [_trade("today", qty=9999)]
    assert loop.size_baseline(earlier + today, date(2026, 9, 22)) == pytest.approx(1200.0)
    assert loop.size_baseline(earlier[:4], date(2026, 9, 22)) is None


def test_unknown_data_and_other_tags_ask_nothing():
    losing = [_trade("t1", net=-500, cad=-500, risk=100)]
    assert _rows(_rule("other", "be calm"), losing) == []
    assert _rows(_rule("wait_for_confirmation"), losing) == []
    assert _rows(None, losing) == []
    # a rule for another day is not today's
    assert _rows(_rule("respect_stop", for_date=NEXT), losing) == []
    # an open trade, a trade closed another day, and a naive close stamp
    rule = _rule("respect_stop")
    open_trade = _trade("o", net=-500, cad=-500, risk=100, status="OPEN")
    yesterday = _trade("y", net=-500, cad=-500, risk=100, day="2026-09-21")
    naive = dict(_trade("n", net=-500, cad=-500, risk=100), closed_at=f"{SESSION}T11:00:00")
    assert _rows(rule, [open_trade, yesterday, naive]) == []


def test_at_most_three_a_day_the_first_closed_win():
    rule = _rule("respect_stop")
    trades = [
        _trade(f"t{i}", f"S{i}", closed=f"1{i}:00", net=-200, cad=-200, risk=100) for i in range(5)
    ]
    rows = _rows(rule, list(reversed(trades)) + [trades[0]])
    assert [row["trade_id"] for row in rows] == ["t0", "t1", "t2"]


# ---------------------------------------------------------------------------
# 4. the Mentor kind, its trigger, its cap and its reader
# ---------------------------------------------------------------------------
def _state(rows, answered=None):
    return {"session": SESSION, "auto_mode": "DESK", "rule_reflections": rows, "answered": answered or {}}


def _slot():
    return types.SimpleNamespace(slot_id=f"{SESSION}-1200", kind="m5", session=SESSION,
                                 scheduled_at=datetime(2026, 9, 22, 12, tzinfo=ET))


def test_the_kind_is_registered_awake_with_a_reader_that_reads_its_key():
    import mentor_questions
    import trade_mentor_trade_check as check

    kind = mentor_questions.kind_named("rule_reflection")
    assert kind.dormant_until == ""
    assert kind.cadence == mentor_questions.CADENCE_ONCE
    assert {"kept", "broke", "not_relevant", mentor_questions.STOP_ASKING, *check.ANSWER_STATES} <= set(kind.options)
    row = mentor_questions.consumer_report([kind])[0]
    assert row["imports"] and row["reads"], row["reason"]


def test_pending_asks_each_reflection_once_and_never_past_the_days_three():
    import mentor_questions

    rows = _rows(_rule("respect_stop"), [
        _trade("t1", "ZETA", net=-140, cad=-140, risk=100, closed="10:00"),
        _trade("t2", "BETA", net=-150, cad=-150, risk=100, closed="10:30"),
    ])
    asked = mentor_questions.pending(_state(rows), _slot()).asked
    reflections = [s for s in asked if s.kind == "rule_reflection"]
    assert [s.subject_id for s in reflections] == ["t1", "t2"]
    assert reflections[0].prompt.startswith("Your rule today: respect your stop. You closed ZETA")
    # t1 answered: only t2 left.
    answered = {"rule_reflection:t1": {"answered_at": SESSION}}
    asked = mentor_questions.pending(_state(rows, answered), _slot()).asked
    assert [s.subject_id for s in asked if s.kind == "rule_reflection"] == ["t2"]
    # Two OTHER trades already answered today: one slot of the three is left.
    answered = {"rule_reflection:x1": {"answered_at": SESSION}, "rule_reflection:x2": {"answered_at": SESSION}}
    asked = mentor_questions.pending(_state(rows, answered), _slot()).asked
    assert [s.subject_id for s in asked if s.kind == "rule_reflection"] == ["t1"]
    # AWAY asks nothing.
    away = dict(_state(rows), auto_mode="AWAY")
    assert mentor_questions.pending(away, _slot()).asked == ()


def test_an_answer_lands_where_the_day_record_reads_it():
    import day_session_record
    import mentor_questions

    rows = _rows(_rule(), [_trade("t1", "ZETA", net=60, cad=60, risk=100)])
    subject = [s for s in mentor_questions.pending(_state(rows), _slot()).asked if s.kind == "rule_reflection"][0]
    written = []

    class Store:
        def record_opportunity_event(self, **kwargs):
            written.append(kwargs)
            return {"payload": kwargs["payload"], **kwargs}

    result = mentor_questions.record_answer(
        subject, {"state": "broke"}, store=Store(), now=datetime(2026, 9, 22, 12, 5, tzinfo=ET)
    )
    assert result["ok"]
    assert written[0]["trade_id"] == "t1"
    reflections = day_session_record.rule_reflections({"t1": [{"payload": written[0]["payload"]}]})
    assert reflections[0]["answer"] == "broke"
    assert reflections[0]["rule_tag"] == "hold_winners"
    assert reflections[0]["symbol"] == "ZETA"
    recap = day_session_record._recap({"mentor_answers": {"t1": [{"payload": written[0]["payload"]}]}}, [])
    assert recap["rule_reflections"][0]["answer"] == "broke"


def test_the_desk_lane_uses_the_chips_rule_and_never_raises():
    pytest.importorskip("PySide6")
    from ui.app import MainWindow

    chip = types.SimpleNamespace(info=lambda: _rule("respect_stop"))
    host = types.SimpleNamespace(rule_chip=chip, _regime_timeline=[])
    trades = [_trade("t1", "ZETA", net=-140, cad=-140, risk=100)]
    rows = MainWindow._mentor_rule_lane(host, None, SESSION, trades)
    assert [row["trade_id"] for row in rows] == ["t1"]
    # No chip, no rule: nothing. A store that raises: nothing, no raise.
    assert MainWindow._mentor_rule_lane(types.SimpleNamespace(), None, SESSION, trades) == []

    class Broken:
        def list_trades(self, **_k):
            raise OSError("not mounted")

    host.rule_chip = types.SimpleNamespace(info=lambda: _rule("size_down_in_chop"))
    assert MainWindow._mentor_rule_lane(host, Broken(), SESSION, trades) == []


def test_the_desk_keeps_only_regime_changes_in_memory():
    pytest.importorskip("PySide6")
    from ui.app import MainWindow

    host = types.SimpleNamespace()
    at = datetime(2026, 9, 22, 9, 35, tzinfo=ET)
    MainWindow._note_regime_seen(host, "neutral_chop", now=at)
    MainWindow._note_regime_seen(host, "neutral_chop", now=at + timedelta(minutes=5))
    MainWindow._note_regime_seen(host, "", now=at + timedelta(minutes=6))
    MainWindow._note_regime_seen(host, "bullish_weak", now=at + timedelta(minutes=10))
    assert [label for _at, label in host._regime_timeline] == ["neutral_chop", "bullish_weak"]
