"""S1.1 - a capture verb is a quick button: it writes at once and opens nothing.

Trader, 2026-09-03: *"when I hit something in the capture tab such as veto, or
like and claim etc that is sufficient reason enough - these are quick buttons to
get a note in essentially and do NOT require a pop up note."*

Three routes still stop and ask today, and each one is driven here through the
handler the trader's click actually reaches:

* ``CaptureRail.commit_like`` REFUSES a claimed like whose why is empty
  (``_prompt_for_why``), so the click writes nothing at all;
* ``CaptureRail.prompt_quick_like`` - the route both quick-like BUTTONS use -
  opens ``QInputDialog.getMultiLineText`` and writes only after OK;
* ``MasterAvwapPanel._dislike_row`` - the setups table's ✕ - opens TWO modal
  ``QInputDialog``s and writes only after both.

The dialog constructors are replaced with a recorder that answers CANCEL, never
with one that answers OK: opening one at all is the failure, and a route that
still asks also still writes nothing, so it cannot pass on the second assertion
either.
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

from PySide6.QtWidgets import QApplication, QInputDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _no_cohort_writes(rail) -> None:
    """The cohort files are LIVE stores; this packet is not about them.

    ``_merge_cohort_safely`` already swallows everything, so replacing the three
    merges changes nothing the tests below observe - it only keeps a click in a
    test out of ``like_cohort_picks.csv``.
    """
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True}


@pytest.fixture
def pane(tmp_path):
    import pick_feedback
    from ui.widgets.alert_chart_review import AlertChartReview

    pick_feedback.clear_reviewed_today_cache()
    widget = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")
    _no_cohort_writes(widget.capture_rail)
    yield widget
    widget.deleteLater()


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


def _show(pane, monkeypatch, symbol: str = "AAPL", side: str = "LONG"):
    monkeypatch.setattr(pane.snapshot, "set_symbol", lambda *a, **k: None)
    pane.set_alert(_alert(symbol, side))


def _dialog_watch(monkeypatch) -> list[str]:
    """Record every modal question and answer it with CANCEL.

    Cancel rather than OK on purpose: a stub that answers would let a surviving
    dialog write the row anyway and the test would pass with the pop-up still
    there. Answering "the trader pressed Escape" means a route that still asks
    also still writes nothing, so both halves of the assertion fail together.
    """
    opened: list[str] = []

    def _record(name, answer):
        def _stub(*_args, **_kwargs):
            opened.append(name)
            return answer

        return staticmethod(_stub)

    monkeypatch.setattr(QInputDialog, "getMultiLineText", _record("getMultiLineText", ("", False)))
    monkeypatch.setattr(QInputDialog, "getText", _record("getText", ("", False)))
    monkeypatch.setattr(QInputDialog, "getItem", _record("getItem", ("", False)))
    return opened


# ---------------------------------------------------------------------------
# the claimed like
# ---------------------------------------------------------------------------
def test_a_claimed_like_with_no_why_is_written_not_refused(pane, monkeypatch, tmp_path):
    """The claim is the whole requirement; the why is the trader's option."""
    _show(pane, monkeypatch, "AAPL")
    rail = pane.capture_rail
    rail.setup_list.setCurrentRow(0)
    claim = rail.selected_setup_id()
    assert claim, "the rail offers at least one claim"
    rail.like_note_input.setText("")

    row = rail.commit_like()

    assert row is not None, "an empty why must not refuse the like"
    written = _rows(tmp_path / "trader_annotations.jsonl")
    assert len(written) == 1, written
    assert written[0]["event_type"] == "like_claim"
    assert written[0]["claimed_setup_id"] == claim
    assert written[0]["like_mode"] == "claimed"
    # Empty, not invented. `build_annotation` omits an empty note today
    # (store.py: `if note_text:`), so either shape reads as "no why".
    assert str(written[0].get("note", "")) == ""


def test_the_claim_list_gesture_writes_without_asking_for_a_why(pane, monkeypatch, tmp_path):
    """Double-click / the claim digit go through `select_setup` -> `commit_like`."""
    _show(pane, monkeypatch, "NVDA", "SHORT")
    rail = pane.capture_rail
    rail.like_note_input.setText("")

    rail._claim_picked(rail.setup_list.item(0))

    written = _rows(tmp_path / "trader_annotations.jsonl")
    assert len(written) == 1, (
        f"the gesture must write, got {written!r}; status={rail.status_text()!r}"
    )
    assert written[0]["symbol"] == "NVDA"
    assert written[0]["claimed_setup_id"]


# ---------------------------------------------------------------------------
# the quick like button
# ---------------------------------------------------------------------------
def test_the_rails_quick_like_button_opens_nothing_and_writes_at_once(
    pane, monkeypatch, tmp_path
):
    _show(pane, monkeypatch, "AMD")
    opened = _dialog_watch(monkeypatch)

    pane.capture_rail.quick_like_button.click()

    assert opened == [], "the quick-like button must not stop to ask"
    written = _rows(tmp_path / "trader_annotations.jsonl")
    assert len(written) == 1, written
    assert written[0]["event_type"] == "like_claim"
    assert written[0]["like_mode"] == "quick"
    assert "claimed_setup_id" not in written[0]


def test_the_charts_quick_like_button_opens_nothing_and_writes_at_once(
    pane, monkeypatch, tmp_path
):
    """The chart's verb row is the same verb; it must not keep the dialog."""
    _show(pane, monkeypatch, "AMD")
    opened = _dialog_watch(monkeypatch)

    pane.quick_like_button.click()

    assert opened == [], "the chart's quick-like button must not stop to ask"
    written = _rows(tmp_path / "trader_annotations.jsonl")
    assert len(written) == 1, written
    assert written[0]["like_mode"] == "quick"


# ---------------------------------------------------------------------------
# the Master AVWAP ✕
# ---------------------------------------------------------------------------
@pytest.fixture
def setups_panel(tmp_path, monkeypatch):
    from focus_picks import FocusPickStore
    from ui.annotations import verdicts
    from ui.models.setup import SetupRow
    from ui.panels.master_avwap_panel import MasterAvwapPanel
    from ui.services.focus_service import FocusService

    service = FocusService(
        FocusPickStore(
            focus_longs_path=tmp_path / "focus_longs.txt",
            focus_shorts_path=tmp_path / "focus_shorts.txt",
            longs_path=tmp_path / "longs.txt",
            shorts_path=tmp_path / "shorts.txt",
            membership_path=tmp_path / "focus_pick_membership.json",
        )
    )
    # The verdict annotation defaults to the LIVE trader_annotations.jsonl.
    annotations = tmp_path / "trader_annotations.jsonl"
    real_dislike = verdicts.record_dislike
    real_like = verdicts.record_like
    monkeypatch.setattr(
        verdicts,
        "record_dislike",
        lambda **kwargs: real_dislike(**{**kwargs, "path": annotations}),
    )
    monkeypatch.setattr(
        verdicts,
        "record_like",
        lambda **kwargs: real_like(**{**kwargs, "path": annotations}),
    )
    panel = MasterAvwapPanel(service, review_events_path=tmp_path / "events.jsonl")
    panel.set_rows(
        [
            SetupRow(
                symbol="LNG",
                side="LONG",
                score=245.0,
                bucket="favorite_setup",
                setup_tags=["AVWAP_BREAKOUT"],
                expected_r=0.85,
                raw={"setup_family": "avwap_breakout"},
            )
        ]
    )
    yield panel, annotations, tmp_path / "events.jsonl"
    panel.deleteLater()


def _cell(panel, key: str):
    column = next(
        index for index, (name, _label) in enumerate(panel.model.COLUMNS) if name == key
    )
    return panel.proxy.index(0, column)


def test_the_setups_x_opens_no_dialog_and_records_the_rejection_at_once(
    setups_panel, monkeypatch
):
    """The ✕ is a quick button: one click, one rejection, no picklist."""
    from review_events import load_review_events

    panel, annotations, events = setups_panel
    opened = _dialog_watch(monkeypatch)

    panel._on_table_clicked(_cell(panel, "dislike"))

    assert opened == [], "the setups ✕ must not open a picklist or a detail box"
    rows = load_review_events(events)
    assert [row["action"] for row in rows] == ["dislike"], rows
    detail = rows[0]["detail"]
    # The trader's own words are what this field carries, and they are typed on
    # the chart afterwards (S1.2), never coded by machine here.
    assert "reason" in detail, detail
    assert detail["reason"] == ""
    written = _rows(annotations)
    assert len(written) == 1, written
    assert written[0]["event_type"] == "veto"
    assert written[0]["symbol"] == "LNG"
