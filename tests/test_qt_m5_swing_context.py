"""The M5 bar marks alerts that sit on a D1 swing setup (trader, 2026-09-23).

``NVDA LONG VWAP bounce · D1 A ★``: the swing grade, and a star for a claimed
pick. Display only; with the prioritise switch on, swing-backed rows are drawn
first (claimed, then grade), and off, arrival order is unchanged.
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

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import BounceAlert  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _m5(symbol, side="LONG", *, at="07:09:19", trigger="[B-TIER] VWAP bounce"):
    return BounceAlert(
        time_text=at,
        symbol=symbol,
        side=side,
        trigger=trigger,
        timeframe="5m",
        tag="green",
        raw_text=f"{trigger} {symbol} ({side.lower()})",
    )


def _prioritise(monkeypatch, on: bool) -> None:
    import working_lately

    monkeypatch.setattr(working_lately, "prioritise_enabled", lambda: on)


def _texts(bar):
    return [bar.list.item(i).text() for i in range(bar.list.count())]


def _symbols(bar):
    return [a.symbol for a in bar.alerts()]


CONTEXT = {
    ("NVDA", "LONG"): {"grade": "A", "family": "avwap_bounce", "claimed": True},
    ("AMD", "LONG"): {"grade": "B", "family": "d1_wick", "claimed": False},
    ("TSLA", "SHORT"): {"grade": None, "family": "breakdown", "claimed": False},
}


def test_the_suffix_shows_on_a_swing_backed_row_and_in_the_tooltip():
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.set_swing_context(CONTEXT)
    bar.post(_m5("NVDA"))
    bar.post(_m5("TSLA", "SHORT"))
    bar.post(_m5("AMD", "SHORT"))  # AMD's setup is LONG: nothing
    rows = dict(zip(_symbols(bar), _texts(bar), strict=False))
    assert rows["NVDA"].endswith("VWAP bounce  · D1 A ★")
    assert rows["TSLA"].endswith("· D1 New")
    assert "D1" not in rows["AMD"]
    nvda = bar.list.item(_symbols(bar).index("NVDA"))
    assert nvda.toolTip().startswith("D1 setup: avwap_bounce, grade A, claimed")


def test_context_arriving_after_the_alert_rewrites_the_row_in_place():
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.post(_m5("NVDA"))
    item = bar.list.item(0)
    assert "D1" not in item.text()
    bar.set_swing_context(CONTEXT)
    assert bar.list.item(0) is item, "rewritten in place, never rebuilt"
    assert item.text().endswith("· D1 A ★")


def test_a_stale_or_empty_context_leaves_the_rows_clean():
    from ui.widgets.m5_alert_bar import M5AlertBar

    bar = M5AlertBar()
    bar.set_swing_context(CONTEXT)
    bar.post(_m5("NVDA"))
    bar.post(_m5("AMD"))
    bar.set_swing_context({("AMD", "LONG"): CONTEXT[("AMD", "LONG")]})
    rows = dict(zip(_symbols(bar), _texts(bar), strict=False))
    assert "D1" not in rows["NVDA"]
    assert rows["AMD"].endswith("· D1 B")
    bar.set_swing_context({})
    assert all("D1" not in text for text in _texts(bar))
    assert all("D1 setup" not in bar.list.item(i).toolTip() for i in range(bar.count()))


def test_prioritise_on_draws_swing_backed_rows_first(monkeypatch):
    from ui.widgets.m5_alert_bar import M5AlertBar

    _prioritise(monkeypatch, True)
    bar = M5AlertBar()
    for symbol in ("NVDA", "ZZZ", "AMD", "YYY"):
        bar.post(_m5(symbol))
    bar.post(_m5("TSLA", "SHORT"))
    assert _symbols(bar) == ["TSLA", "YYY", "AMD", "ZZZ", "NVDA"]
    bar.set_swing_context(CONTEXT)
    # Claimed first, then by grade (B before New), then arrival order.
    assert _symbols(bar) == ["NVDA", "AMD", "TSLA", "YYY", "ZZZ"]
    bar.set_swing_context({})
    assert _symbols(bar) == ["TSLA", "YYY", "AMD", "ZZZ", "NVDA"]


def test_prioritise_off_keeps_arrival_order(monkeypatch):
    from ui.widgets.m5_alert_bar import M5AlertBar

    _prioritise(monkeypatch, False)
    bar = M5AlertBar()
    for symbol in ("NVDA", "ZZZ", "AMD"):
        bar.post(_m5(symbol))
    bar.set_swing_context(CONTEXT)
    assert _symbols(bar) == ["AMD", "ZZZ", "NVDA"]
    assert _texts(bar)[2].endswith("· D1 A ★")


def test_the_desk_hands_the_setups_table_to_the_bar():
    from ui.models.setup import SetupRow
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        desk.m5_alert_bar.post(_m5("NVDA"))
        desk.m5_alert_bar.post(_m5("AMD", "SHORT"))
        model = desk.master_panel.model
        model.set_rows(
            [
                SetupRow(
                    symbol="NVDA",
                    side="LONG",
                    bucket="favorite_setup",
                    raw={"setup_family": "avwap_bounce"},
                ),
                SetupRow(
                    symbol="AMD",
                    side="LONG",
                    bucket="claimed_like",
                    raw={"claimed_setup_id": "d1_wick", "bucket_keys": ["claimed_like"]},
                ),
            ]
        )
        rows = dict(zip(_symbols(desk.m5_alert_bar), _texts(desk.m5_alert_bar), strict=False))
        assert rows["NVDA"].endswith("· D1 New"), "no grades loaded yet: New"
        assert "D1" not in rows["AMD"], "AMD's setup is LONG; the alert is SHORT"
        import setup_grades

        key = setup_grades.swing_key("LONG", "favorite_setup", "avwap_bounce")
        model._grades = {key: {"grade": "A"}}
        model.set_rows(model.rows())  # a claim/scan refresh resets the model
        nvda = desk.m5_alert_bar.list.item(_symbols(desk.m5_alert_bar).index("NVDA"))
        assert nvda.text().endswith("· D1 A")
        desk.m5_alert_bar.post(_m5("AMD", "LONG", at="07:15:00"))
        amd = desk.m5_alert_bar.list.item(
            [
                (a.symbol, a.side) for a in desk.m5_alert_bar.alerts()
            ].index(("AMD", "LONG"))
        )
        assert amd.text().endswith("★")
    finally:
        desk.shutdown()
        desk.close()
