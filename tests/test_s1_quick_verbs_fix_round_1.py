"""S1 fix round 1 - the three blockers the reviewer reproduced, plus A7.

Driven through the real panel, because every one of these is about what the
ALERT CENTER does with a signal the capture rail earned - none of them can be
seen from the rail or the chart pane alone.

* **Blocker 1.** A rail veto ended in the Alert Center writing its OWN veto row
  and opening its OWN note box. Before S1.2 that box appeared at the click; with
  the retire deferred it appeared AFTER the trader had already typed their note
  into the rail. The live decision stream shows one KKR veto of 2026-09-03 as
  THREE rows for this reason.
* **Blocker 2.** `like_advance` and the veto's `remove_today` were recorded off
  the deferred retire, so `dwell_ms` counted the trader typing and a desk closed
  between the click and Enter lost the decision entirely.
* **A7.** Two verdicts on one chart: the LAST one decided the route, so
  veto-then-like silently dropped `removeTodayRequested` and the vetoed symbol
  was never parked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _queue_mechanics_only(monkeypatch):
    """An ordinary intraday alert lists in the M5 bar instead of queueing a
    chart (trader rule 2026-08-27). These tests are about the chart."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    monkeypatch.setattr(
        AlertCenterPanel, "_is_m5_review_alert", staticmethod(lambda alert: False)
    )


def _rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _alert(symbol: str = "AAPL", side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="09:31:00",
        symbol=symbol,
        side=side,
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[B-TIER] {symbol}: Bounce confirmed",
    )


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """A real Alert Center whose every store is under `tmp_path`."""
    import pick_feedback
    from ui.annotations import verdicts
    from ui.panels.alert_center_panel import AlertCenterPanel

    annotations = tmp_path / "trader_annotations.jsonl"
    for name in ("record_like", "record_dislike", "record_not_today", "record_note_on"):
        real = getattr(verdicts, name)

        def _redirected(*args, _real=real, **kwargs):
            kwargs.setdefault("path", annotations)
            return _real(*args, **kwargs)

        monkeypatch.setattr(verdicts, name, _redirected)

    pick_feedback.clear_reviewed_today_cache()
    events = tmp_path / "events.jsonl"
    panel = AlertCenterPanel(review_events_path=events)
    panel.chart_review.capture_rail._annotations_path = annotations
    rail = panel.chart_review.capture_rail
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True}
    monkeypatch.setattr(
        panel.chart_review.snapshot, "set_symbol", lambda *a, **k: None
    )
    # A note box the trader never sees is still a note box: record every one.
    prompts: list = []
    monkeypatch.setattr(
        panel, "_prompt_for_not_today_note", lambda written: prompts.append(written)
    )
    yield panel, annotations, events, prompts
    panel.deleteLater()


def _actions(path: Path) -> list[str]:
    from review_events import load_review_events

    return [row["action"] for row in load_review_events(path)]


def _plain_reason(rail) -> str:
    vocabulary = rail._vocabulary
    assert vocabulary is not None
    for reason in vocabulary.reasons:
        if not reason.note_required:
            return reason.code
    raise AssertionError("every veto reason demands a note")


def _chart(panel, symbol: str = "AAPL") -> None:
    alert = _alert(symbol)
    panel._enqueue_review_alert(alert)
    QApplication.processEvents()
    assert panel.chart_review.alert is not None


def _veto(panel) -> None:
    rail = panel.chart_review.capture_rail
    rail.select_reason(_plain_reason(rail))


def _quick_like(panel) -> None:
    dict(panel.chart_review.capture_rail.action_shortcuts())["Alt+L"]()


def _finish(panel, text: str = "") -> None:
    rail = panel.chart_review.capture_rail
    if text:
        rail.note_input.setText(text)
    rail._settle_follow_up(write=bool(text))
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# BLOCKER 1 - a rail veto is ONE veto row and NO pop-up
# ---------------------------------------------------------------------------
def test_a_rail_veto_writes_one_veto_row_and_opens_no_box(desk):
    panel, annotations, _events, prompts = desk
    _chart(panel)

    _veto(panel)
    assert [row["event_type"] for row in _rows(annotations)] == ["veto"]
    assert prompts == [], "nothing may open at the click"

    _finish(panel, "rejecting a trendline")

    written = _rows(annotations)
    assert [row["event_type"] for row in written] == ["veto", "note"], (
        "the Alert Center must not add a second veto row behind the rail's - "
        f"got {[row['event_type'] for row in written]}"
    )
    assert written[1]["supersedes"] == written[0]["event_id"]
    assert prompts == [], (
        "and it must not open a note box after the trader has already typed one"
    )


def test_a_rail_veto_with_no_note_is_still_one_row_and_no_box(desk):
    panel, annotations, _events, prompts = desk
    _chart(panel)

    _veto(panel)
    _finish(panel)

    assert [row["event_type"] for row in _rows(annotations)] == ["veto"]
    assert prompts == []


def test_the_alert_centers_own_not_today_button_is_unchanged(desk):
    """It is not in the capture tab and the packet does not name it: it still
    writes its own uncoded veto row and still offers its own note box."""
    panel, annotations, events, prompts = desk
    _chart(panel)

    panel.chart_review.remove_today_button.click()
    QApplication.processEvents()

    written = _rows(annotations)
    assert [row["event_type"] for row in written] == ["veto"]
    assert written[0]["surface"] == "chart_review"
    assert len(prompts) == 1, "the direct button keeps its note box"
    assert "remove_today" in _actions(events)


# ---------------------------------------------------------------------------
# BLOCKER 2 - the decision is recorded on the CLICK
# ---------------------------------------------------------------------------
def test_like_advance_is_recorded_on_the_key_not_on_the_advance(desk):
    panel, _annotations, events, _prompts = desk
    _chart(panel)

    _quick_like(panel)

    assert _actions(events) == ["shown", "like_advance"], (
        "a desk closed between the click and Enter must still have the decision"
    )

    _finish(panel, "clean reclaim")
    assert _actions(events) == ["shown", "like_advance"], "and never a second one"


def test_the_vetos_remove_today_event_is_recorded_on_the_click_too(desk):
    panel, _annotations, events, _prompts = desk
    _chart(panel)

    _veto(panel)
    assert _actions(events) == ["shown", "remove_today"]

    _finish(panel)
    assert _actions(events) == ["shown", "remove_today"]


def test_the_recorded_dwell_does_not_include_the_typing_time(desk, monkeypatch):
    """`dwell_ms` is the denominator `review_learning` reads as "how long did
    the trader look at this chart". Typing about it is not looking at it."""
    from review_events import load_review_events

    panel, _annotations, events, _prompts = desk
    _chart(panel)

    # 17 while the chart is still waiting for the trader (i.e. at the click),
    # 9999 once they have finished typing. Which number lands in the row is
    # exactly the question, and it is asked of the real clock's caller rather
    # than of a call count.
    rail = panel.chart_review.capture_rail
    monkeypatch.setattr(
        panel,
        "_review_dwell_ms",
        lambda symbol: 17 if rail.follow_up_pending() else 9999,
    )

    _quick_like(panel)
    _finish(panel, "typed for a while")

    rows = [row for row in load_review_events(events) if row["action"] == "like_advance"]
    assert len(rows) == 1
    assert rows[0]["dwell_ms"] == 17, (
        f"the dwell must be measured at the click, got {rows[0]['dwell_ms']}"
    )


# ---------------------------------------------------------------------------
# A7 - two verdicts on one chart, both take effect
# ---------------------------------------------------------------------------
def test_a_veto_then_a_like_both_take_effect(desk):
    """A veto parks the name for the day; a like deliberately does not. Keeping
    only the last verb dropped the park - a judgement the trader made and the
    desk then ignored."""
    panel, annotations, events, _prompts = desk
    _chart(panel, "AAPL")

    _veto(panel)
    _quick_like(panel)
    assert [row["event_type"] for row in _rows(annotations)] == ["veto", "like_claim"]
    assert "AAPL" not in panel._ignored_symbols, "neither has moved the chart yet"

    _finish(panel)

    assert "AAPL" in panel._ignored_symbols, (
        "the veto's park must survive a later like on the same chart"
    )
    assert _actions(events) == ["shown", "remove_today", "like_advance"]


def test_a_like_then_a_veto_both_take_effect_too(desk):
    panel, annotations, events, _prompts = desk
    _chart(panel, "NVDA")

    _quick_like(panel)
    _veto(panel)
    _finish(panel)

    assert [row["event_type"] for row in _rows(annotations)] == ["like_claim", "veto"]
    assert "NVDA" in panel._ignored_symbols
    assert sorted(_actions(events)[1:]) == ["like_advance", "remove_today"]


def test_one_verb_still_fires_exactly_one_route(desk):
    """The set is not an excuse to fire both when only one was recorded."""
    panel, _annotations, events, _prompts = desk
    _chart(panel, "AMD")

    _quick_like(panel)
    _finish(panel)

    assert _actions(events) == ["shown", "like_advance"]
    assert "AMD" not in panel._ignored_symbols, "a like never parks the symbol"
