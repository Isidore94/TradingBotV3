"""R4 section 6.3, through the real AlertCenterPanel.add_alert seam.

test_alert_repetition.py proves the decision logic. This file proves the panel
actually honours it -- and, more importantly, that the four things section 6.3.4
promises not to touch are still untouched: the backing alert list, the chart
review queue, History, and anything the trader armed.

The R8 review found all six of its blockers at seams the tests had bypassed.
These go through add_alert.
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

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from ui.panels.alert_center_panel import AlertCenterPanel

    p = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
    )
    # Never sound in a test run, and never reach the review-event file.
    monkeypatch.setattr(p, "_alerts_may_sound", lambda: False)
    yield p
    p.deleteLater()


def _alert(symbol="AAPL", side="LONG", *, tier="B", extra=""):
    from ui.models.bounce import BounceAlert

    text = f"[{tier}-TIER] {symbol}: Bounce confirmed{extra}" if tier else f"{symbol}: note{extra}"
    return BounceAlert(
        time_text="09:31:00",
        symbol=symbol,
        side=side,
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=text,
    )


def _rows(panel) -> int:
    """Feed rows currently on screen (the layout keeps one trailing stretch)."""
    return max(0, panel.feed_layout.count() - 1)


def _no_digest(panel, monkeypatch):
    """Put the clock well past the open so the digest never applies."""
    import alert_repetition

    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: False
    )


# --------------------------------------------------------------------------
# folding
# --------------------------------------------------------------------------
def test_a_repeat_does_not_add_a_second_row(panel, monkeypatch):
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert())
    before = _rows(panel)
    panel.add_alert(_alert())
    assert _rows(panel) == before


def test_a_repeat_shows_a_count_on_the_existing_row(panel, monkeypatch):
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert())
    panel.add_alert(_alert())
    item = panel._feed_rows[("AAPL", "LONG")]
    assert item.repeat_badge.isVisible() or item.repeat_badge.text() == "×2"
    assert item.repeat_badge.text() == "×2"


def test_a_folded_repeat_is_still_in_the_backing_list(panel, monkeypatch):
    """Section 6.3.4: History, the evidence streams and the AWAY push all read
    this list. Folding a ROW must never cost a RECORD."""
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert())
    panel.add_alert(_alert())
    panel.add_alert(_alert())
    assert len([a for a in panel._alerts if a.symbol == "AAPL"]) == 3


def test_a_folded_repeat_still_reaches_the_chart_review_queue(panel, monkeypatch):
    _no_digest(panel, monkeypatch)
    seen: list = []
    monkeypatch.setattr(panel, "_enqueue_review_alert", seen.append)
    panel.add_alert(_alert())
    panel.add_alert(_alert())
    assert len(seen) == 2


def test_the_other_side_gets_its_own_row(panel, monkeypatch):
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert(side="LONG"))
    before = _rows(panel)
    panel.add_alert(_alert(side="SHORT"))
    assert _rows(panel) == before + 1


def test_an_escalation_floats_a_row_again(panel, monkeypatch):
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert(tier="B"))
    before = _rows(panel)
    panel.add_alert(_alert(tier="S"))
    assert _rows(panel) == before + 1


# --------------------------------------------------------------------------
# what must never be folded
# --------------------------------------------------------------------------
def test_a_trader_armed_watch_hit_is_never_folded(panel, monkeypatch):
    """The rule this control is not allowed to break."""
    from ui.models.bounce import CHART_WATCH_TAG, BounceAlert

    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert())
    before = _rows(panel)
    armed = BounceAlert(
        time_text="09:40:00",
        symbol="AAPL",
        side="LONG",
        trigger="Chart watch hit",
        tag=CHART_WATCH_TAG,
        raw_text="AAPL: chart watch hit",
        payload={"chart_watch_kind": "vwap_reclaim"},
    )
    panel.add_alert(armed)
    assert _rows(panel) == before + 1


def test_a_focus_privileged_repeat_is_never_folded(panel, monkeypatch):
    _no_digest(panel, monkeypatch)
    monkeypatch.setattr(panel, "_alert_has_focus_privilege", lambda _a: True)
    panel.add_alert(_alert())
    before = _rows(panel)
    panel.add_alert(_alert())
    assert _rows(panel) == before + 1


# --------------------------------------------------------------------------
# the open-burst digest
# --------------------------------------------------------------------------
def test_open_burst_alerts_share_one_digest_row(panel, monkeypatch):
    import alert_repetition

    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: True
    )
    panel.add_alert(_alert(symbol="AAPL"))
    panel.add_alert(_alert(symbol="NVDA"))
    panel.add_alert(_alert(symbol="TSLA"))
    assert _rows(panel) == 1
    assert "3 name(s)" in panel._digest_row.text()
    for symbol in ("AAPL", "NVDA", "TSLA"):
        assert symbol in panel._digest_row.text()


def test_digested_alerts_are_all_still_recorded(panel, monkeypatch):
    import alert_repetition

    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: True
    )
    queued: list = []
    monkeypatch.setattr(panel, "_enqueue_review_alert", queued.append)
    panel.add_alert(_alert(symbol="AAPL"))
    panel.add_alert(_alert(symbol="NVDA"))
    assert len(panel._alerts) == 2
    assert len(queued) == 2


def test_a_banger_skips_the_digest(panel, monkeypatch):
    import alert_repetition

    monkeypatch.setattr(
        alert_repetition.RepetitionLedger, "_in_digest_window", lambda self, now: True
    )
    panel.add_alert(_alert(symbol="AAPL"))
    before = _rows(panel)
    panel.add_alert(_alert(symbol="NVDA", extra=" BANGER"))
    assert _rows(panel) == before + 1


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------
def test_a_broken_ledger_falls_back_to_a_plain_new_row(panel, monkeypatch):
    """Fail open. A presentation control must never cost the trader an alert."""

    def boom(*_a, **_k):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(panel, "_repetition_ledger", boom)
    panel.add_alert(_alert())
    panel.add_alert(_alert())
    assert _rows(panel) == 2
    assert len(panel._alerts) == 2


def test_a_rebuild_repoints_the_fold_registry_at_the_new_widgets(panel, monkeypatch):
    """A rebuild destroys every row widget and builds fresh ones. The registry
    must end up pointing at the NEW widgets - a stale entry would have the next
    repeat fold into a deleted object."""
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert())
    before = panel._feed_rows[("AAPL", "LONG")]
    panel._rebuild_feed()
    after = panel._feed_rows.get(("AAPL", "LONG"))
    assert after is not None and after is not before
    # And the entry is a widget that is really in the layout.
    live = {panel.feed_layout.itemAt(i).widget() for i in range(panel.feed_layout.count())}
    assert after in live


def test_a_repeat_after_a_rebuild_still_folds(panel, monkeypatch):
    """The behaviour the registry exists for, end to end across a rebuild."""
    _no_digest(panel, monkeypatch)
    panel.add_alert(_alert())
    panel._rebuild_feed()
    before = _rows(panel)
    panel.add_alert(_alert())
    assert _rows(panel) == before
    assert panel._feed_rows[("AAPL", "LONG")].repeat_badge.text() == "×2"
