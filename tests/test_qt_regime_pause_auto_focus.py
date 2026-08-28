"""A with-trend "holding highs" row goes to M5 Focus, not the review chart.

Trader rule 2026-08-27, from the morning it came out of: 21 regime-pause charts
reviewed in nine minutes on a bullish open, twelve added to M5 Focus by hand,
74 charts waiting. `test_regime_pause_focus.py` proves the two-case rule; this
proves the Alert Center applies it at the one seam that matters - `add_alert` -
and that the placement costs nothing upstream: the feed row and the evidence
row are still written, and every case the rule does not name still charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """A DESK-mode panel on a bullish day with an empty Focus store."""
    from focus_picks import FocusPickStore
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    panel.focus_service = FocusService(store)
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(panel, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(panel, "_regime_pause_day_env", lambda: "bullish_weak", raising=False)
    # Since 2026-08-27 an intraday row the rule leaves alone lists in the M5
    # alert bar rather than queueing a chart; the tests read that list.
    panel._posted_to_bar = []
    panel.m5AlertPosted.connect(panel._posted_to_bar.append)
    yield panel, store
    panel.deleteLater()


def _regime_alert(symbol="FROG", side="LONG"):
    from ui.models.bounce import BounceAlert

    pattern = "pressing lows" if side == "SHORT" else "holding highs"
    word = "short" if side == "SHORT" else "long"
    return BounceAlert(
        time_text="07:09:19",
        symbol=symbol,
        side=side,
        trigger="M5 regime-pause watch · new HOD",
        timeframe="M5",
        tag="red" if side == "SHORT" else "green",
        raw_text=(
            f"REGIME PAUSE WATCH ({word}): SPY paused (-0.30% window) - "
            f"1 swing {word} still {pattern}: {symbol} (new HOD) (1 today). "
            "Recorded as swing-scan evidence, not an entry signal."
        ),
    )


def _events(panel) -> list[dict]:
    path = panel._review_events_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _auto_focus_rows(panel) -> list[dict]:
    return [row for row in _events(panel) if row["action"] == "regime_pause_auto_focus"]


def _posted_symbols(panel) -> list[str]:
    """What reached the M5 alert bar - where a row the rule did not place goes."""
    return [alert.symbol for alert in getattr(panel, "_posted_to_bar", [])]


def _queued_symbols(panel) -> list[str]:
    current = panel._current_review_alert
    return ([current.symbol] if current is not None else []) + [
        alert.symbol for alert in panel._review_queue
    ]


def test_a_long_holding_highs_on_a_bullish_day_lands_in_focus_and_skips_the_chart(desk):
    panel, store = desk
    alert = _regime_alert("FROG", "LONG")

    panel.add_alert(alert)

    assert store.focus_symbols("long", "m5") == ["FROG"]
    marker = store.auto_pick_marker("FROG", "long", "m5")
    assert marker is not None, "machine-placed, so 'Not today' must be able to reach it"
    assert marker["staged_at"] == "07:09:19"
    assert "bullish" in marker["reason"]
    assert _queued_symbols(panel) == [], "the decision is made; no chart to review"
    assert _posted_symbols(panel) == [], "and no line in the M5 bar either"
    # Upstream is untouched: the feed row and the evidence row are both there.
    assert panel._alerts and panel._alerts[0] is alert
    rows = _auto_focus_rows(panel)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "FROG"
    assert rows[0]["detail"] == {
        "env": "bullish_weak",
        "focus_side": "long",
        "outcome": "adopted",
    }


def test_a_short_on_a_bullish_day_still_charts(desk):
    """Counter-trend is not in the rule: it lists in the M5 bar for the
    trader to look at, exactly like any other intraday row."""
    panel, store = desk

    panel.add_alert(_regime_alert("MRK", "SHORT"))

    assert store.focus_symbols("short", "m5") == []
    assert store.focus_symbols("long", "m5") == []
    assert _posted_symbols(panel) == ["MRK"]
    assert _queued_symbols(panel) == []
    assert _auto_focus_rows(panel) == []


def test_a_short_pressing_lows_on_a_bearish_day_lands_in_focus_shorts(desk, monkeypatch):
    panel, store = desk
    monkeypatch.setattr(panel, "_regime_pause_day_env", lambda: "bearish_strong")

    panel.add_alert(_regime_alert("MRK", "SHORT"))

    assert store.focus_symbols("short", "m5") == ["MRK"]
    assert store.is_auto_adopted("MRK", "short", "m5")
    assert _queued_symbols(panel) == []


@pytest.mark.parametrize("env", ["", "neutral"])
def test_a_day_with_no_directional_read_admits_nothing(desk, monkeypatch, env):
    """Missing data is uncertainty, never confirmation."""
    panel, store = desk
    monkeypatch.setattr(panel, "_regime_pause_day_env", lambda: env)

    panel.add_alert(_regime_alert("FROG", "LONG"))

    assert store.focus_symbols("long", "m5") == []
    assert _posted_symbols(panel) == ["FROG"]
    assert _queued_symbols(panel) == []


def test_the_traders_own_focus_name_keeps_its_owner_and_its_chart(desk):
    """An unmarked entry is the trader's; the machine must not relabel it."""
    panel, store = desk
    store.add("FROG", "long", "m5")

    panel.add_alert(_regime_alert("FROG", "LONG"))

    assert store.auto_pick_marker("FROG", "long", "m5") is None
    assert _posted_symbols(panel) == ["FROG"], "it lists like any intraday row"
    assert _queued_symbols(panel) == []


def test_a_repeat_of_the_machines_own_placement_is_resolved_not_charted(desk):
    panel, store = desk
    panel.add_alert(_regime_alert("FROG", "LONG"))
    assert _queued_symbols(panel) == []

    panel.add_alert(_regime_alert("FROG", "LONG"))

    assert store.focus_symbols("long", "m5") == ["FROG"]
    assert _queued_symbols(panel) == []
    assert [row["detail"]["outcome"] for row in _auto_focus_rows(panel)] == [
        "adopted",
        "already_auto",
    ]


@pytest.mark.parametrize("mode", ["AWAY", "EVENING", "OFF"])
def test_only_desk_places(desk, monkeypatch, mode):
    """Nobody is present to prune in AWAY/EVENING, and OFF adopts nothing."""
    panel, store = desk
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: mode)

    panel.add_alert(_regime_alert("FROG", "LONG"))

    assert store.focus_symbols("long", "m5") == []
    assert store.auto_pick_marker("FROG", "long", "m5") is None
    assert _auto_focus_rows(panel) == []


def test_a_failed_placement_falls_open_onto_the_chart_queue(desk, monkeypatch):
    """The rule must never be the reason a chart went missing."""
    panel, store = desk

    def boom(*_args, **_kwargs):
        raise OSError("focus file locked")

    monkeypatch.setattr(store, "add", boom)

    panel.add_alert(_regime_alert("FROG", "LONG"))

    assert _posted_symbols(panel) == ["FROG"], "falls open onto the ordinary path"
    assert _queued_symbols(panel) == []
    assert _auto_focus_rows(panel) == []


def test_an_ordinary_alert_is_not_touched_by_the_rule(desk):
    from ui.models.bounce import BounceAlert

    panel, store = desk
    alert = BounceAlert.from_callback("[S-TIER] NVDA: Bounce confirmed (long)", "green")

    panel.add_alert(alert)

    assert store.focus_symbols("long", "m5") == []
    assert _posted_symbols(panel) == ["NVDA"]
    assert _queued_symbols(panel) == []
