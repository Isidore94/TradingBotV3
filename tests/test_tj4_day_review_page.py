r"""TJ-4 item 4 - the page shows the story, and Redo obeys the night. RED.

Packet `.claude/packets/TJ-4.md` item 4; `plan.md` §12.4 "TJ-4" change 4 and
TJ-13 item 5 ("A daytime request is queued for tonight (TJ-4's rule, now
general)"); TJ-1 item 3 (ONE read, on ONE worker, and the page computes nothing).

**NO MODEL IS CALLED HERE AND NO PROCESS IS SPAWNED.** The Redo launcher is an
injected seam, and the one test that touches the real default asserts the
COMMAND it would build, never runs it.

The contract these tests pin (the builder may ADD keys, never remove one)
------------------------------------------------------------------------

``scripts/ui/services/day_review_service.py``
    * ``PAYLOAD_KEYS`` and ``empty_payload`` gain ``day_story`` and ``d1_view``,
      both PRESENT and ``None``.
    * ``read_day`` fills them ON THE WORKER by READING the two verified files.
      It calls no model and writes nothing.
    * ``DayReviewService.build_pack_for(session_date)`` is the named seam the
      post-close tick calls, beside ``build_index_for`` / ``build_session_bars_for``
      / ``build_reads_for``, and ``_IndexBuildWorker`` calls it on the worker
      thread. A failure there costs the pack and nothing else.

``scripts/ui/panels/day_review_panel.py``
    * ``STORY_QUEUED_NOTE`` - what a daytime Redo says.
    * ``redo_command(session) -> list[str]`` - PURE; the argv `plan.md` change 4
      names.
    * ``launch_redo_process(session)`` - the default launcher.
    * ``DayReviewPanel(..., redo_launcher=None)``, ``panel.redo_story_button``,
      ``panel.redo_story()``, ``panel.story_body``, ``panel.d1_view_note``.
    * ``render`` shows the verified narration when there is one and keeps the
      deterministic facts under ``NO_STORY_YET`` when there is not. It formats
      and computes nothing.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import tj4_support as fx  # noqa: E402

SESSION = fx.SESSION
NOW = datetime(2026, 9, 19, 8, 0)

A_STORY = {
    "schema": "day_review_narration_v1",
    "session_date": SESSION,
    "inputs_hash": "pack-hash",
    "model": "local-test-medium",
    "narration": {
        "headline": "You called the afternoon up and it went up.",
        "what_happened": "SPY closed +2.00% and held over its session VWAP.",
        "what_you_thought": "At 07:02 you read breadth as better and called it up.",
        "were_you_right": [{
            "claim": "Rest of day: up",
            "source_id": "said:mj-1:prediction",
            "verdict": "right",
            "evidence_id": "read:rd-1",
        }],
        "chased_against_news": {"verdict": "unknown", "evidence_id": "read:rd-1"},
        "process": "One call, one note, two trades.",
        "sources": ["said:mj-1:prediction", "read:rd-1"],
    },
}

A_VIEW = {
    "schema": "d1_view_narration_v1",
    "inputs_hash": "d1-hash",
    "model": "local-test-medium",
    "narration": {
        "belief_now": "You have been leaning long the index since the gap held.",
        "open_theses": [{
            "claim": "The index grinds higher into month end",
            "since": "2026-09-11",
            "still_true": "unknown",
            "evidence_id": "said:mj-9:prediction",
        }],
        "sources": ["said:mj-9:prediction"],
    },
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def test_the_payload_declares_the_story_and_the_rolling_view_present_and_empty():
    from ui.services.day_review_service import PAYLOAD_KEYS, empty_payload

    assert "day_story" in PAYLOAD_KEYS
    assert "d1_view" in PAYLOAD_KEYS

    blank = empty_payload(SESSION)
    assert blank["day_story"] is None
    assert blank["d1_view"] is None


def test_read_day_reads_both_verified_files_on_the_worker_and_calls_no_model(
    tmp_path, monkeypatch
):
    """TJ-1 item 3: ONE read, on ONE worker. The page computes nothing.

    The story is a FILE the night already verified. Reading it here is a few
    kilobytes; building it here would be a 14 GB model load on the desk the
    trader is using.
    """
    import ai_summary
    import json as _json
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    (root / "narration").mkdir(parents=True)
    (root / "narration" / f"{SESSION}.json").write_text(
        _json.dumps(A_STORY), encoding="utf-8"
    )
    (root / "d1_view.json").write_text(_json.dumps(A_VIEW), encoding="utf-8")

    def _no_model(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a page read must not call a model")

    monkeypatch.setattr(ai_summary, "request_ai_summary", _no_model)

    service = _wire(monkeypatch)
    payload = service.read_day(SESSION, now=NOW)

    # A legacy story with no matching day pack is read but cannot be shown as
    # current facts. The page stays read-only and makes the missing stamp clear.
    assert payload["day_story"] is None
    assert payload["story_freshness"]["state"] in {"missing", "unread"}
    assert payload["d1_view"]["narration"]["open_theses"][0]["still_true"] == "unknown"


def test_a_story_written_for_another_session_never_appears_on_this_one(
    tmp_path, monkeypatch
):
    """One file per session. A page that fell back to "the newest story" would
    print Thursday's reading over Friday's tape.
    """
    import json as _json
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    (root / "narration").mkdir(parents=True)
    (root / "narration" / "2026-09-17.json").write_text(
        _json.dumps({**A_STORY, "session_date": "2026-09-17"}), encoding="utf-8"
    )

    service = _wire(monkeypatch)
    payload = service.read_day(SESSION, now=NOW)

    assert payload["day_story"] is None, payload["day_story"]


def _wire(monkeypatch):
    """`read_day` over plain dicts. No live store is opened (TJ-10's pattern)."""
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
            return []

        def daily_story(self, _session):
            return None

        def theses_for(self, _session):
            return []

    monkeypatch.setattr(daily_recap_reader, "_read_jsonl", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(daily_recap_reader, "_read_csv", lambda name, *a, **k: _Store([]))
    monkeypatch.setattr(claimed_picks, "load_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        journal_store, "JournalStore",
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


# ---------------------------------------------------------------------------
# the post-close seam
# ---------------------------------------------------------------------------


class _SeamService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def read_day(self, session_date, **_kwargs):
        return {"session_date": session_date}

    def build_index_for(self, session_date, **_kwargs):
        self.calls.append(("index", threading.get_ident()))
        return {}

    def build_session_bars_for(self, session_date, **_kwargs):
        self.calls.append(("bars", threading.get_ident()))
        return None

    def build_reads_for(self, session_date, **_kwargs):
        self.calls.append(("reads", threading.get_ident()))
        return []

    def build_pack_for(self, session_date, **_kwargs):
        self.calls.append(("pack", threading.get_ident()))
        return {}


def _drain(qapp, panel, *, timeout: float = 5.0) -> None:
    worker = panel._index_worker
    if worker is not None:
        worker.wait(int(timeout * 1000))
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and panel._index_worker is not None:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_the_post_close_tick_builds_the_pack_on_the_worker_after_the_grades(
    qapp, monkeypatch
):
    """plan TJ-4 change 1: the pack is built "by the post-close tick and by the
    nightly slot".

    AFTER the grades, because the pack's `reads` section IS what the grader just
    wrote; and on the WORKER, because the index build alone froze the desk for
    22.8 s from a 60-second timer slot (reviewer, 2026-09-17).
    """
    import daily_recap_schedule
    from ui.panels.day_review_panel import DayReviewPanel

    monkeypatch.setattr(daily_recap_schedule, "due_session", lambda *a, **k: None)
    monkeypatch.setattr(
        daily_recap_schedule, "post_close_due_session", lambda *a, **k: SESSION
    )

    service = _SeamService()
    panel = DayReviewPanel(service=service, clock=lambda: NOW)
    monkeypatch.setattr(panel, "reload", lambda: None)
    monkeypatch.setattr(panel, "show_session", lambda _session: None)
    slot_thread = threading.get_ident()
    try:
        assert panel.poll_auto_read() == SESSION
        _drain(qapp, panel)
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()

    kinds = [name for name, _thread in service.calls]
    assert "pack" in kinds, "the post-close tick never built the day pack"
    assert kinds.index("pack") > kinds.index("reads"), kinds
    pack_thread = next(thread for name, thread in service.calls if name == "pack")
    assert pack_thread != slot_thread, "the pack build ran on the Qt timer's thread"


def test_a_service_with_no_pack_seam_still_gets_its_index(qapp, monkeypatch):
    """`getattr`, like `build_reads_for` beside it: a host that hands this page
    a reader without the seam still gets the fast second open.
    """
    import daily_recap_schedule
    from ui.panels.day_review_panel import DayReviewPanel

    monkeypatch.setattr(daily_recap_schedule, "due_session", lambda *a, **k: None)
    monkeypatch.setattr(
        daily_recap_schedule, "post_close_due_session", lambda *a, **k: SESSION
    )

    service = _SeamService()
    service.build_pack_for = None  # present, and not callable

    panel = DayReviewPanel(service=service, clock=lambda: NOW)
    monkeypatch.setattr(panel, "reload", lambda: None)
    monkeypatch.setattr(panel, "show_session", lambda _session: None)
    try:
        assert panel.poll_auto_read() == SESSION
        _drain(qapp, panel)
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()

    assert "index" in [name for name, _thread in service.calls]


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


class _StubService:
    def __init__(self, payload=None) -> None:
        self.payload = payload or {}

    def read_day(self, session_date, **_kwargs):
        payload = dict(self.payload)
        payload.setdefault("session_date", session_date)
        return payload


@pytest.fixture
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


def _payload(**overrides):
    from ui.services.day_review_service import empty_payload

    base = empty_payload(SESSION)
    base.update(overrides)
    return base


def test_with_no_story_the_page_keeps_the_facts_and_says_so(panel):
    """"else the facts and 'no story yet'" - packet TJ-4 item 4. Unchanged."""
    from ui.panels.day_review_panel import NO_STORY_YET

    panel.render(_payload(story=fx.daily_story([fx.observation_only_entry()])))

    assert panel.story_note.text() == NO_STORY_YET
    assert "SPY" in panel.story_facts.text()
    assert not panel.story_body.text().strip()


def test_a_verified_story_leads_the_section_and_its_verdicts_are_quoted(panel):
    """The narration replaces the "no story yet" line, and every graded claim
    is printed with the verdict the READ ROW carried - the page grades nothing.
    """
    from ui.panels.day_review_panel import NO_STORY_YET

    panel.render(_payload(day_story=A_STORY))

    assert panel.story_note.text() != NO_STORY_YET
    assert panel.story_note.text() == A_STORY["narration"]["headline"]
    body = panel.story_body.text()
    assert "SPY closed +2.00%" in body
    assert "Rest of day: up" in body
    assert "right" in body
    assert "unknown" in body, "the chased verdict is printed, including `unknown`"


def test_the_rolling_view_prints_the_belief_and_every_open_thesis(panel):
    panel.render(_payload(d1_view=A_VIEW))

    text = panel.d1_view_note.text()
    assert "leaning long the index" in text
    assert "grinds higher into month end" in text
    assert "unknown" in text


def test_the_page_never_calls_the_grader_or_the_pack_builder(panel, monkeypatch):
    """TJ-10's rule, extended: the page FORMATS. It computes nothing.

    Every number on this section was measured on a worker or by the night.
    """
    import day_review_pack
    import market_read_grades

    for module, name in (
        (market_read_grades, "grade_read"),
        (market_read_grades, "congruence_lines"),
        (day_review_pack, "build_pack"),
    ):
        monkeypatch.setattr(module, name, _must_not_be_called, raising=False)

    panel.render(_payload(day_story=A_STORY, d1_view=A_VIEW))


def _must_not_be_called(*_args, **_kwargs):  # pragma: no cover - must never run
    raise AssertionError("the page computes nothing")


# ---------------------------------------------------------------------------
# Redo story
# ---------------------------------------------------------------------------


def test_the_redo_command_is_the_one_the_plan_names():
    """plan TJ-4 change 4: `run_ai_jobs.py --slot day_review_narration --force
    --session <date>`. PURE - it builds the argv and runs nothing.
    """
    from ui.panels.day_review_panel import redo_command

    argv = [str(part) for part in redo_command(SESSION)]

    assert any(part.endswith("run_ai_jobs.py") for part in argv), argv
    assert "--slot" in argv and argv[argv.index("--slot") + 1] == "day_review_narration"
    assert "--force" in argv
    assert "--session" in argv and argv[argv.index("--session") + 1] == SESSION


def test_the_cli_accepts_the_session_the_redo_names(monkeypatch):
    """The other end of the same command. Today `run_ai_jobs.py` has no
    `--session`, so the button's own argv would exit 2 before anything ran.

    `--status` returns before any slate is built, and it is stubbed, so this
    exercises the PARSER and starts no night.
    """
    import run_ai_jobs

    monkeypatch.setattr(run_ai_jobs, "_print_status", lambda *_a, **_k: 0)

    assert run_ai_jobs.main([
        "--status", "--slot", "day_review_narration", "--force", "--session", SESSION,
    ]) == 0


def test_a_redo_by_day_queues_for_tonight_and_starts_nothing(
    qapp, tmp_path, monkeypatch
):
    """plan TJ-13 item 5 / TJ-4 change 4: "A daytime request is queued for
    tonight". No local inference by day, from any door.
    """
    import day_review_pack
    import project_paths
    from ui.panels import day_review_panel as module
    from ui.panels.day_review_panel import STORY_QUEUED_NOTE, DayReviewPanel

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    monkeypatch.setattr(
        module.window, "launch_allowed",
        lambda *_a, **_k: (False, "outside the off-hours window (01:00-09:00 ET)"),
        raising=False,
    )

    launched: list[str] = []
    panel = DayReviewPanel(
        service=_StubService(), clock=lambda: NOW,
        redo_launcher=lambda session: launched.append(session),
    )
    monkeypatch.setattr(panel, "reload", lambda: None)
    try:
        panel.redo_story()

        assert launched == [], "a daytime Redo must start no process"
        assert day_review_pack.redo_requested(panel.session_date(), root=root) is True
        text = panel.status.text()
        assert STORY_QUEUED_NOTE.lower() in text.lower() or "tonight" in text.lower()
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()


def test_a_redo_at_night_starts_one_worker_process_and_writes_no_marker(
    qapp, tmp_path, monkeypatch
):
    """Inside the window the button does what it says - once, off the Qt thread,
    and in a PROCESS so a 14 GB load can never share the desk's own heap.
    """
    import day_review_pack
    import project_paths
    from ui.panels import day_review_panel as module
    from ui.panels.day_review_panel import DayReviewPanel

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    monkeypatch.setattr(
        module.window, "launch_allowed", lambda *_a, **_k: (True, "window open"),
        raising=False,
    )

    launched: list[str] = []
    panel = DayReviewPanel(
        service=_StubService(), clock=lambda: NOW,
        redo_launcher=lambda session: launched.append(session),
    )
    monkeypatch.setattr(panel, "reload", lambda: None)
    try:
        panel.redo_story()
        panel.redo_story()

        assert launched == [panel.session_date(), panel.session_date()]
        assert day_review_pack.redo_requested(panel.session_date(), root=root) is False
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()


def test_the_redo_button_exists_and_is_wired_to_the_slot(panel):
    """A verb the trader can reach, beside the story it redoes."""
    assert panel.redo_story_button.isEnabled()
    assert "redo" in panel.redo_story_button.text().lower()
