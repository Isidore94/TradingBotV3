"""S1.2 - the chart stays up until the trader has finished typing.

Trader, 2026-09-03: *"when I hit like or not today or anything, it should keep
the chart up UNTIL I finish typing."*

Today `AlertChartReview._on_captured` emits `removeTodayRequested` /
`likeAdvanceRequested` the instant the verb's row reaches disk, so a line typed
after the click lands in the NEXT chart's field. What must change is only the
RETIRE: the row is still written at once, and the same two signals still carry
the same alert - they are deferred until the trader presses Enter or Escape.

How these drive it, and what the builder is free to choose:

* the verdict row is read back off a temp `trader_annotations.jsonl`, not from a
  return value, so "it wrote" means it is on disk;
* the follow-up field is discovered through `capture_rail.focusWidget()` - the
  packet says the rail's inline note field takes focus, and Qt records that
  regardless of which field the builder picks;
* Enter and Escape are delivered to that field as real key events;
* the follow-up note's link is asserted as "one of this row's values IS the
  verdict row's `event_id`", never as a named key, because the packet forbids
  inventing a second opportunity id and `record_note_on` already carries the
  lineage in `supersedes`.
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

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)

_EDITORS = (QLineEdit, QPlainTextEdit, QTextEdit)


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


@pytest.fixture
def pane(tmp_path):
    import pick_feedback
    from ui.widgets.alert_chart_review import AlertChartReview

    pick_feedback.clear_reviewed_today_cache()
    widget = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")
    rail = widget.capture_rail
    # Live cohort files are not what this packet is about; the merge already
    # swallows every failure, so replacing it observes nothing and writes nothing.
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True}
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


class _Retires:
    """Both retire routes, counted together and separately."""

    def __init__(self, pane):
        self.removed: list = []
        self.advanced: list = []
        pane.removeTodayRequested.connect(self.removed.append)
        pane.likeAdvanceRequested.connect(self.advanced.append)

    def total(self) -> int:
        return len(self.removed) + len(self.advanced)


@pytest.fixture
def chart(pane, monkeypatch, tmp_path):
    monkeypatch.setattr(pane.snapshot, "set_symbol", lambda *a, **k: None)
    pane.set_alert(_alert("AAPL"))
    return pane, _Retires(pane), tmp_path / "trader_annotations.jsonl"


def _plain_reason(rail) -> str:
    """A veto reason that does NOT already demand a note of its own."""
    vocabulary = rail._vocabulary
    assert vocabulary is not None
    for reason in vocabulary.reasons:
        if not reason.note_required:
            return reason.code
    raise AssertionError("every veto reason demands a note")


def _veto(pane) -> None:
    """The trader's real gesture: the reason's digit, which commits."""
    rail = pane.capture_rail
    rail.select_reason(_plain_reason(rail))


def _quick_like_by_key(pane) -> None:
    """Exactly what Alt+L is bound to, read off the rail's own binding."""
    dict(pane.capture_rail.action_shortcuts())["Alt+L"]()


def _waiting_field(pane):
    field = pane.capture_rail.focusWidget()
    assert isinstance(field, _EDITORS), (
        "after a retiring verb the rail's inline note field must take focus, "
        f"got {field!r}"
    )
    return field


def _type(field, text: str) -> None:
    if isinstance(field, QLineEdit):
        field.setText(text)
    else:
        field.setPlainText(text)


def _key(field, key) -> None:
    QTest.keyClick(field, key)
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# the verb writes, and the chart stays
# ---------------------------------------------------------------------------
def test_a_veto_click_writes_the_row_and_leaves_the_chart_up(chart):
    pane, retires, annotations = chart

    _veto(pane)

    written = _rows(annotations)
    assert [row["event_type"] for row in written] == ["veto"], written
    assert pane.alert is not None and pane.alert.symbol == "AAPL"
    assert retires.total() == 0, (
        "the chart must stay up until the trader has finished typing"
    )


def test_the_waiting_chart_says_that_enter_moves_on(chart):
    pane, _retires, _annotations = chart

    _veto(pane)

    field = _waiting_field(pane)
    hint = " ".join(
        [
            field.placeholderText() if hasattr(field, "placeholderText") else "",
            pane.capture_rail.status_text(),
        ]
    ).lower()
    assert "enter" in hint, f"the waiting state must say how to advance, got {hint!r}"


def test_enter_with_a_typed_line_writes_one_follow_up_note_then_advances(chart):
    pane, retires, annotations = chart

    _veto(pane)
    verdict = _rows(annotations)[0]
    field = _waiting_field(pane)
    _type(field, "wick through the level, no volume")
    _key(field, Qt.Key.Key_Return)

    written = _rows(annotations)
    assert len(written) == 2, written
    note = written[1]
    assert note["event_type"] == "note"
    assert note["note"] == "wick through the level, no volume"
    assert note["symbol"] == "AAPL"
    # The verdict row it follows, named by the id that row ALREADY carries.
    assert verdict["event_id"] in [
        str(value) for value in note.values()
    ], f"the follow-up note must name {verdict['event_id']}, got {note!r}"
    assert retires.total() == 1, "Enter advances"
    assert retires.removed and retires.removed[0].symbol == "AAPL"


def test_enter_on_an_empty_field_advances_and_writes_nothing_extra(chart):
    pane, retires, annotations = chart

    _veto(pane)
    _key(_waiting_field(pane), Qt.Key.Key_Return)

    assert [row["event_type"] for row in _rows(annotations)] == ["veto"]
    assert retires.total() == 1


def test_escape_advances_and_writes_nothing_extra(chart):
    pane, retires, annotations = chart

    _veto(pane)
    field = _waiting_field(pane)
    _type(field, "started typing then changed my mind")
    _key(field, Qt.Key.Key_Escape)

    assert [row["event_type"] for row in _rows(annotations)] == ["veto"], (
        "Escape discards the line; the verdict itself already counted"
    )
    assert retires.total() == 1


def test_a_quick_like_by_key_waits_for_the_typed_line_too(chart):
    pane, retires, annotations = chart

    _quick_like_by_key(pane)

    assert [row["event_type"] for row in _rows(annotations)] == ["like_claim"]
    assert retires.total() == 0, "a like retires on Enter, not on the key"

    field = _waiting_field(pane)
    _type(field, "clean reclaim")
    _key(field, Qt.Key.Key_Return)

    written = _rows(annotations)
    assert [row["event_type"] for row in written] == ["like_claim", "note"], written
    assert written[0]["event_id"] in [str(value) for value in written[1].values()]
    assert retires.advanced and retires.advanced[0].symbol == "AAPL"
    assert retires.total() == 1


# ---------------------------------------------------------------------------
# what must NOT change
# ---------------------------------------------------------------------------
def test_a_note_and_a_pass_still_never_retire_the_chart(chart):
    pane, retires, annotations = chart
    rail = pane.capture_rail

    rail.note_input.setText("watching this one")
    rail.commit_note()
    assert retires.total() == 0

    code = next(iter(rail.pass_checkboxes))
    rail.pass_checkboxes[code].setChecked(True)
    rail.note_input.setText("spread too wide")
    rail.commit_pass()

    assert [row["event_type"] for row in _rows(annotations)] == ["note", "pass"]
    assert retires.total() == 0, "a note and a pass are written ABOUT this chart"


def test_charting_another_symbol_while_waiting_retires_with_nothing_extra(chart):
    """A click away is a pass, and it stays one - it is never a re-queue."""
    pane, retires, annotations = chart

    _veto(pane)
    field = _waiting_field(pane)
    _type(field, "half a thought")

    pane.set_alert(_alert("NVDA"))

    assert [row["event_type"] for row in _rows(annotations)] == ["veto"], (
        "walking away writes nothing extra"
    )
    assert retires.total() == 1
    assert retires.removed and retires.removed[0].symbol == "AAPL"


def test_two_verdicts_on_one_chart_both_land_before_the_advance(chart):
    pane, retires, annotations = chart

    _veto(pane)
    _quick_like_by_key(pane)

    written = _rows(annotations)
    assert [row["event_type"] for row in written] == ["veto", "like_claim"], written
    assert retires.total() == 0, "the second verb keeps waiting for Enter too"

    _key(_waiting_field(pane), Qt.Key.Key_Return)
    assert retires.total() >= 1


def test_a_follow_up_note_is_a_note_and_an_old_row_is_not_a_follow_up(chart):
    """Schema stays 1; the new key is additive and ABSENT means "not one".

    The old row here is written the way the file already holds them - every key
    the store writes, and none it does not - so "absent" is modelled rather than
    assumed. Nothing that counts verdicts may start counting a follow-up note.
    """
    import pick_feedback

    pane, _retires, annotations = chart

    _veto(pane)
    field = _waiting_field(pane)
    _type(field, "gapped away from the level")
    _key(field, Qt.Key.Key_Return)

    follow_up = _rows(annotations)[1]
    assert follow_up["event_type"] == "note"
    assert follow_up["schema_version"] == 1
    assert pick_feedback._ANNOTATION_DECISIONS == {"veto", "like_claim", "note"}

    old = dict(follow_up)
    # An OLD note row: the same shape, minus every key this packet added.
    for key in set(follow_up) - {
        "schema_version",
        "event_id",
        "event_type",
        "symbol",
        "session_date",
        "created_at",
        "source",
        "side",
        "note",
        "surface",
        "timeframe",
    }:
        old.pop(key, None)
    with Path(annotations).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(old) + "\n")

    from ui.annotations.store import load_annotations

    read_back = load_annotations(annotations)
    assert len(read_back) == 3, read_back
    new_keys = set(follow_up) - set(old)
    assert new_keys, "the follow-up note must carry a key an old note does not"
    assert not (new_keys & set(read_back[-1])), (
        "an old row carries none of the follow-up keys, so it is not a follow-up"
    )
