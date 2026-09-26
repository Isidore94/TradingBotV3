"""B8: one `add_alert` must not re-walk the whole backing list three times.

The status line after every alert counted loud rows, Show-filter hides and
first-30 hides in three separate passes over up to 500 alerts, and every pass
asked the sector filter about every alert again. That made `add_alert` cost
grow with the feed (2 ms/alert at 100, 8 ms at 1,000). The status line now
comes from one pass: the sector answer is asked once per symbol and the
loudness of an alert (a fact of its own text) is remembered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SYMBOLS = [f"S{index:02d}" for index in range(10)]
TIERS = ("S", "A", "B", "C")


def _alert(index: int):
    from ui.models.bounce import BounceAlert

    symbol = SYMBOLS[index % len(SYMBOLS)]
    tier = TIERS[index % len(TIERS)]
    return BounceAlert(
        time_text="09:31:00" if index % 3 == 0 else "11:15:00",
        symbol=symbol,
        side="LONG" if index % 2 == 0 else "SHORT",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[{tier}-TIER] {symbol}: Bounce confirmed" + (" PROVEN" if index % 7 == 0 else ""),
        payload={"feedback": {"bounce_types": "ceetype" if index % 2 else "beetype"}},
    )


@pytest.fixture
def panel(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    import alert_repetition
    import alert_show_filter
    import project_paths
    import sector_exclusion
    from ui.panels import alert_center_panel as panel_mod
    from ui.panels.alert_center_panel import AlertCenterPanel

    settings: dict[str, object] = {}
    monkeypatch.setattr(
        project_paths, "get_local_setting", lambda key, default=None: settings.get(key, default)
    )
    monkeypatch.setattr(
        project_paths, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )
    longs = tmp_path / "longs.txt"
    shorts = tmp_path / "shorts.txt"
    longs.write_text("", encoding="utf-8")
    shorts.write_text("", encoding="utf-8")
    monkeypatch.setattr(project_paths, "LONGS_FILE", longs)
    monkeypatch.setattr(project_paths, "SHORTS_FILE", shorts)
    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: False
    )
    monkeypatch.setattr(panel_mod.QApplication, "beep", lambda: None)
    monkeypatch.setattr(sector_exclusion, "hide_enabled", lambda: True)
    alert_show_filter.clear_cache()
    made = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
    )
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: True)
    monkeypatch.setattr(made, "_auto_mode_now", lambda: "DESK")
    import setup_grades

    made.set_setup_grades(
        {
            "daytrade": [
                {"key": setup_grades.daytrade_key(kind, side), "bounce_type": kind, "side": side, "grade": grade}
                for kind, grade in (("beetype", "B"), ("ceetype", "C"))
                for side in ("LONG", "SHORT")
            ]
        }
    )
    made.first30_input.setChecked(True)
    assert made.show_filter_active()
    yield made
    made.deleteLater()
    alert_show_filter.clear_cache()


def test_one_add_alert_asks_the_sector_filter_once_per_symbol(panel, monkeypatch):
    import sector_exclusion

    for index in range(300):
        panel.add_alert(_alert(index))
    calls: list[str] = []
    monkeypatch.setattr(
        sector_exclusion, "symbol_is_excluded", lambda symbol: calls.append(symbol) or False
    )
    panel.add_alert(_alert(300))
    # One status pass, one sector answer per distinct symbol (+1 for the new alert).
    assert len(calls) <= len(SYMBOLS) + 1, len(calls)


def test_one_add_alert_does_not_rescore_every_alert_for_loudness(panel, monkeypatch):
    from ui.panels.alert_center import gates

    for index in range(300):
        panel.add_alert(_alert(index))
    calls: list = []
    original = gates.alert_is_loud
    monkeypatch.setattr(gates, "alert_is_loud", lambda alert: calls.append(alert) or original(alert))
    panel.add_alert(_alert(300))
    assert len(calls) <= 3, len(calls)


def test_the_status_line_numbers_are_unchanged(panel):
    """The one-pass status line says what the three separate counts say."""
    import alert_show_filter
    from ui.panels.alert_center.gates import alert_should_sound

    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)
    for index in range(120):
        panel.add_alert(_alert(index))
    loud = sum(
        1
        for alert in panel._alerts
        if alert_should_sound(alert, is_focus=panel._alert_has_focus_privilege(alert))
    )
    hidden = alert_show_filter.hidden_text(
        *panel.show_filter_hidden_counts(), first30=panel.show_filter_first30_count()
    )
    assert f"{len(panel._alerts)} live alert(s), {loud} loud;" in statuses[-1]
    assert hidden and hidden in statuses[-1]
