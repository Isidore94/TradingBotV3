"""S16 item 4 (Weekend Prep part): setups that have worked / are untested in THIS regime.

Facts only, from the published per-regime grades (`regime_grades.build_payload`).
Worked = graded B or better (n at the ladder floor) inside the current regime;
untested = no rows in the current regime. Read on the panel's existing
setup-research worker, formatted on the Qt thread.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import regime_grades  # noqa: E402

BEAR = "bear_channel_lower_highs"
BULL = "bull_run"


def _cell(grade, n, win=0.6, **extra):
    return {"grade": grade, "n": n, "wins": int(n * win), "win_rate": win, **extra}


def _payload(current=BEAR):
    return {
        "schema": regime_grades.SCHEMA,
        "current": (
            {"regime": current, "label": "bear channel", "start_date": "2026-08-14", "day_count": 44}
            if current else None
        ),
        "regimes": [BEAR, BULL],
        "labels": {BEAR: "bear channel", BULL: "bull run"},
        "swing": {
            "LONG|pullback|avwap_reclaim": {
                "side": "LONG", "bucket": "pullback", "family": "avwap_reclaim",
                "by_regime": {BEAR: _cell("B", 42), BULL: _cell("A", 80)},
            },
            "SHORT|breakdown|lower_high": {
                "side": "SHORT", "bucket": "breakdown", "family": "lower_high",
                "by_regime": {BEAR: _cell("A", 55, 0.52, grade_basis="tape", tape_win_rate=0.64, tape_n=40)},
            },
            "LONG|breakout|base_break": {
                "side": "LONG", "bucket": "breakout", "family": "base_break",
                "by_regime": {BULL: _cell("PROVEN", 120)},
            },
            "LONG|pullback|gap_fill": {
                "side": "LONG", "bucket": "pullback", "family": "gap_fill",
                "by_regime": {BEAR: _cell("C", 35, 0.51)},
            },
            "SHORT|breakdown|thin": {
                "side": "SHORT", "bucket": "breakdown", "family": "thin",
                "by_regime": {BEAR: _cell("New", 8, 0.9)},
            },
        },
        "daytrade": {
            "LONG|vwap_bounce": {
                "side": "LONG", "bounce_type": "vwap_bounce",
                "by_regime": {BEAR: _cell("PROVEN", 130)},
            },
            "SHORT|ema_reject": {
                "side": "SHORT", "bounce_type": "ema_reject",
                "by_regime": {BULL: _cell("B", 31)},
            },
        },
        "journal": {"LONG": {BEAR: {"n": 3, "wins": 2}}},
        "cohorts": {},
    }


def test_worked_and_untested_in_the_current_regime():
    view = regime_grades.regime_setups(_payload())
    worked = [(row["kind"], row["side"], row["setup"]) for row in view["worked"]]
    # Best grade first: PROVEN, A, B. C and New (n under the floor) are not "worked".
    assert worked == [
        ("day trade", "LONG", "vwap_bounce"),
        ("swing", "SHORT", "lower_high (breakdown)"),
        ("swing", "LONG", "avwap_reclaim (pullback)"),
    ]
    untested = [(row["kind"], row["side"], row["setup"]) for row in view["untested"]]
    assert untested == [
        ("swing", "LONG", "base_break (breakout)"),
        ("day trade", "SHORT", "ema_reject"),
    ]
    assert view["tested_not_worked"] == 2


def test_text_states_the_regime_and_the_short_basis():
    text = regime_grades.regime_setups_text(_payload())
    assert text.splitlines()[0].startswith("Regime now: bear channel since 2026-08-14 (day 44)")
    assert "Worked in this regime (B or better, n 30+):" in text
    # Shorts carry the vs-SPY rate beside the raw win; longs just the raw win.
    assert "SHORT lower_high (breakdown): A · vs SPY 64% · win 52% · n 55" in text
    assert "LONG avwap_reclaim (pullback): B · win 60% · n 42" in text
    assert "Untested in this regime (no rows yet):" in text
    assert "LONG base_break (breakout)" in text
    assert "2 more tested here, not B or better" in text
    # Nothing from another regime leaks into "worked".
    assert "PROVEN · win 60% · n 120" not in text


def test_unknown_regime_asks_the_trader_to_type_it():
    text = regime_grades.regime_setups_text(_payload(current=None))
    assert "type the regime in the Mentor first" in text
    assert "Worked" not in text
    assert "type the regime in the Mentor first" in regime_grades.regime_setups_text({})
    assert "type the regime in the Mentor first" in regime_grades.regime_setups_text(None)


def test_nothing_worked_and_nothing_untested_say_so():
    payload = _payload()
    payload["swing"] = {}
    payload["daytrade"] = {}
    text = regime_grades.regime_setups_text(payload)
    assert "Worked in this regime (B or better, n 30+): none yet" in text
    assert "Untested in this regime (no rows yet): none" in text


# ---------------------------------------------------------------------------
# the worker reads, the Qt thread formats
# ---------------------------------------------------------------------------


def _boom(*_args, **_kwargs):
    raise AssertionError("the Qt thread must not read this")


def test_worker_reads_the_persisted_regime_grades(monkeypatch):
    import slot_narration
    from ui.panels import weekend_prep_panel as module
    from ui.services import working_lately_service

    monkeypatch.setattr(slot_narration, "read_setup_research_narration", lambda: {"state": "present", "text": "ok"})
    monkeypatch.setattr(working_lately_service, "read_persisted_regime_grades", lambda *a, **k: _payload())
    read = module._read_setup_research_and_regime()
    assert read["text"] == "ok"
    assert read["regime_grades"]["current"]["regime"] == BEAR

    monkeypatch.setattr(working_lately_service, "read_persisted_regime_grades", _boom)
    read = module._read_setup_research_and_regime()
    assert read["text"] == "ok" and read["regime_grades"] == {}


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.qt
def test_week_ahead_prints_the_regime_setups(qapp, monkeypatch):
    from ui.panels import weekend_prep_panel as module
    from ui.services import working_lately_service

    panel = module.WeekendPrepPanel()
    monkeypatch.setattr(working_lately_service, "read_persisted_regime_grades", _boom)
    try:
        panel._on_setup_research_ready({"state": "present", "text": "R", "regime_grades": _payload()})
        view = panel.week_ahead.regime_setups_view.toPlainText()
        assert view == regime_grades.regime_setups_text(_payload())
        assert panel.week_ahead.setup_research_view.toPlainText() == "R"
        panel._on_setup_research_ready({"state": "present", "text": "R"})
        assert "type the regime in the Mentor first" in panel.week_ahead.regime_setups_view.toPlainText()
    finally:
        panel.shutdown()
        panel.deleteLater()


@pytest.mark.qt
def test_week_ahead_fits_2160_with_the_regime_card(qapp):
    from PySide6.QtCore import QPoint

    from ui.panels import weekend_prep_panel as module
    from ui.services.weekend_prep_service import STEP_IDS

    panel = module.WeekendPrepPanel()
    try:
        panel.resize(3456, 2160)
        panel.show()
        panel.rail.setCurrentRow(STEP_IDS.index("week_ahead"))
        panel._on_setup_research_ready({"state": "present", "text": "R", "regime_grades": _payload()})
        for _ in range(4):
            qapp.processEvents()
        assert panel.height() <= 2160
        view = panel.week_ahead.regime_setups_view
        assert view.isVisible()
        bottom = view.mapTo(panel, QPoint(0, view.height())).y()
        assert bottom <= 2160
    finally:
        panel.shutdown()
        panel.hide()
        panel.deleteLater()
