r"""TJ-7, live gate #151 - Day Review shows ONE mood line, off ONE payload.

Packet `TJ-5-6-7-13B.md`, TJ-7 correction 5 (the lead's EXTRA item, written by
the builder with its own fail-before-fix proof):

    *"Day Review shows ONE mood line read from the ONE payload the worker
    already builds (no new read on the Qt thread, no builder or model call from
    the page): the mood, the state tags and whether the plan was followed when
    recorded, and 'no mood recorded' when not - no percentage, n beside any
    count. A mood written after the session (`written_after_the_session`) is
    LABELLED as such on that line."*

TJ-1's rule is the one this must not break: **Day Review reads ONE payload on
ONE worker**, and the 476 MB stream it replaced is the reason. The mood line is
another projection of that payload - the section AND the formatted line are
built on the worker (`day_review_pack.mood_section` / `mood_statement`), and the
page pushes a string.

WHAT IS PINNED, AND WHY
-----------------------
* `PAYLOAD_KEYS` carries `mood`, present and EMPTY on a first paint, so every
  path hands the page the same shape (TJ-12's rule for `report_card`).
* The page does NOT call the builder. It is monkeypatched to explode; a render
  that reaches it fails.
* The honest first state is a COUNT of nothing: "no mood recorded yet", `n 0`,
  and no `%` anywhere - the live desk holds zero moods (tj7_support's read-only
  count), and a rate over zero clicks is not 0%.
* A mood typed in the evening is LABELLED on the line, never hidden: the
  partition a later reader needs is a label, never a deletion.

RED BEFORE THE FIX (proven by restoring `day_review_service.py` and
`day_review_panel.py` from `a65ef2e6`): `PAYLOAD_KEYS` holds no `mood`,
`read_day` writes none, and `DayReviewPanel` has no `mood_line`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Day Review page is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import tj7_support as fx  # noqa: E402

SESSION = fx.SESSION
NOW = datetime(2026, 9, 19, 7, 30)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------
def _wire(monkeypatch, entries):
    """`read_day` over plain objects. No live store is opened."""
    import chart_snapshot
    import claimed_picks
    import daily_recap_reader
    import day_review_bars
    import journal_store
    from ui.services.day_review_service import DayReviewService

    class _Store:
        def __init__(self, rows):
            self.rows = list(rows)

    class _Journal:
        def entries_about(self, _session):
            return [dict(row) for row in entries]

        def daily_story(self, _session):
            return None

        def theses_for(self, _session):
            return []

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store,
        "JournalStore",
        lambda *a, **k: type("_J", (), {"list_trades": lambda self: []})(),
    )
    monkeypatch.setattr(day_review_bars, "read_session_bars", lambda *a, **k: {})
    monkeypatch.setattr(day_review_bars, "session_is_closed", lambda *a, **k: True)
    monkeypatch.setattr(day_review_bars, "session_is_backfillable", lambda *a, **k: False)
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])

    service = DayReviewService(journal_service=_Journal())
    monkeypatch.setattr(service, "_read_recap", lambda *a, **k: object())
    monkeypatch.setattr(service, "_trades", lambda *a, **k: [])
    return service


def test_the_payload_declares_the_mood_present_and_empty():
    from ui.services.day_review_service import PAYLOAD_KEYS, empty_payload

    assert "mood" in PAYLOAD_KEYS
    blank = empty_payload(SESSION)
    assert "mood" in blank
    assert not blank["mood"], "a first paint has an EMPTY mood, never an invented one"


def test_the_worker_builds_the_section_and_the_line_for_the_page(monkeypatch):
    """One worker, one payload: the page never formats this itself."""
    morning = fx.row_with_a_mood(
        entry_id="mj-0730", stamp=fx.pacific(7, 30), score=2, state_tags=("rushed",)
    )
    close = fx.row_with_a_mood(
        entry_id="mj-1255",
        stamp=fx.pacific(12, 55),
        score=4,
        state_tags=("calm", "tired"),
        followed_plan="partly",
    )

    payload = _wire(monkeypatch, [morning, close]).read_day(SESSION, now=NOW)

    section = payload["mood"]
    assert section["n"] == 2, payload.get("error")
    assert [item["entry_id"] for item in section["recorded"]] == ["mj-0730", "mj-1255"]
    assert "2" in section["line"]
    assert "%" not in section["line"], "a count is not a rate"


def test_the_worker_carries_the_tags_and_the_plan_answer_onto_the_line(monkeypatch):
    row = fx.row_with_a_mood(
        entry_id="mj-1255",
        stamp=fx.pacific(12, 55),
        score=4,
        state_tags=("calm", "tired"),
        followed_plan="partly",
    )

    line = _wire(monkeypatch, [row]).read_day(SESSION, now=NOW)["mood"]["line"]

    assert "4" in line
    assert "calm" in line and "tired" in line
    assert "partly" in line
    assert "%" not in line


def test_a_mood_written_after_the_session_says_so_on_the_line(monkeypatch):
    """A mood typed in the evening is still the trader's mood: it is kept,
    counted and LABELLED. The label is the whole point."""
    evening = fx.row_with_a_mood(
        entry_id="mj-evening",
        stamp=fx.pacific(20, 30),
        score=1,
        state_tags=("tilted",),
        after_the_session=True,
    )

    payload = _wire(monkeypatch, [evening]).read_day(SESSION, now=NOW)

    assert payload["mood"]["recorded"][0]["written_after_the_session"] is True
    assert "after the session" in payload["mood"]["line"].lower()


def test_a_session_with_nothing_clicked_is_a_count_of_nothing(monkeypatch):
    """The live desk's own first state (tj7_support: zero moods today)."""
    import day_review_pack

    payload = _wire(monkeypatch, list(fx.three_kinds_of_absence())).read_day(SESSION, now=NOW)

    assert payload["mood"]["line"] == day_review_pack.MOOD_EMPTY_STATEMENT
    assert "no mood recorded yet" in payload["mood"]["line"].lower()
    assert "%" not in payload["mood"]["line"]
    assert fx.NO_MOOD_YET_N == 0


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
class _StubService:
    """Reads nothing. The page is what is under test."""

    def __init__(self, payload=None) -> None:
        self.payload = payload or {}
        self.reads = 0

    def read_day(self, session_date, **_kwargs):
        self.reads += 1
        body = dict(self.payload)
        body.setdefault("session_date", session_date)
        return body


@pytest.fixture()
def panel(qapp, monkeypatch):
    from ui.panels.day_review_panel import DayReviewPanel

    widget = DayReviewPanel(service=_StubService(), clock=lambda: NOW)
    monkeypatch.setattr(widget, "reload", lambda: None)
    yield widget
    try:
        widget.shutdown()
    except Exception:  # noqa: BLE001
        pass
    widget.deleteLater()
    qapp.processEvents()


def test_the_page_prints_the_worker_s_line_and_builds_nothing(panel, monkeypatch):
    """No new read on the Qt thread, and no builder call from the page."""
    import day_review_pack
    from ui.services.day_review_service import empty_payload

    def _never(*_args, **_kwargs):
        raise AssertionError("the page built a mood section on the Qt thread")

    monkeypatch.setattr(day_review_pack, "mood_section", _never)

    payload = empty_payload(SESSION)
    payload["mood"] = {
        "n": 1,
        "recorded": [],
        "line": "1 mood note(s) this session (n 1). Latest: 4 of 5.",
    }
    panel.render(payload)

    assert panel.mood_line.text() == payload["mood"]["line"]


def test_the_page_says_no_mood_recorded_when_the_payload_has_none(panel):
    import day_review_pack
    from ui.services.day_review_service import empty_payload

    panel.render(empty_payload(SESSION))

    said = panel.mood_line.text()
    assert said == day_review_pack.MOOD_EMPTY_STATEMENT
    assert "no mood recorded yet" in said.lower()
    assert "0" in said
    assert "%" not in said


def test_the_page_reads_the_one_payload_once(panel):
    """STATED GREEN GUARD - it passes on the un-fixed code and must go on
    passing (reviewer advisory 5, 2026-09-20: as first written it could not
    fail at all, so it is now an EXACT count through a spy).

    TJ-1's rule: Day Review reads ONE payload on ONE worker. `render` is
    formatting, so the number of reads it causes is exactly ZERO - a
    `_render_mood` that reached for `self.service.read_day` would fail this.
    """
    service = panel.service
    assert isinstance(service, _StubService)
    service.reads = 0

    panel.render(_payload_with_a_mood())
    panel.render(_payload_with_a_mood())

    assert service.reads == 0, "rendering asked the service for something"
    assert panel.mood_line.text() == _payload_with_a_mood()["mood"]["line"]


def test_the_payloads_mood_ids_are_the_packs_own_ids(monkeypatch):
    """Reviewer advisory 6: the payload mints with its OWN minter, so the ids
    have to be PROVEN equal to the pack's rather than assumed. Both go through
    `day_review_pack.mood_source_id`, which derives an id from the ROW and from
    nothing else - so a citation resolves against either.

    The written pack is deliberately NOT read by the worker: a mood typed in
    the evening arrives after the post-close pack was written, and a page that
    read the pack would show a stale line on the very save gate #151 asks the
    trader to make.
    """
    import day_review_pack
    import tj4_support as fx4

    moods = [
        fx.row_with_a_mood(entry_id="mj-a", stamp=fx.pacific(7, 30), score=2),
        fx.row_with_a_mood(entry_id="mj-b", stamp=fx.pacific(12, 55), score=4),
    ]
    payload = _wire(monkeypatch, moods).read_day(SESSION, now=NOW)

    inputs = fx4.pack_inputs()
    inputs["entries"] = list(inputs["entries"]) + moods
    pack = day_review_pack.build_pack(SESSION, now=fx4.AFTER_THE_CLOSE, **inputs)

    payload_ids = [item["source_id"] for item in payload["mood"]["recorded"]]
    pack_ids = [item["source_id"] for item in pack["mood"]["recorded"]]

    assert payload_ids == pack_ids, "the page and the night name the same row differently"
    for source_id in payload_ids:
        assert source_id in day_review_pack.allowed_source_ids(pack)


def _payload_with_a_mood():
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload["mood"] = {"n": 1, "recorded": [], "line": "1 mood note(s) this session (n 1)."}
    return payload
