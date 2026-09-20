r"""TJ-4 review round 3 - one Redo at a time, and one rule for what it may name.

**BLOCKER (reviewer, 2026-09-20).** `redo_story()` built a NEW `_RedoPackWorker`
on every call and nothing tested for one in flight, so three impatient clicks
started three full `read_day` builds racing on one `pack.json` AND three
`run_ai_jobs.py` children:

```
BUILDS: [('2026-09-16','Dummy-3'), ('2026-09-16','Dummy-4'), ('2026-09-16','Dummy-5')]
PROCESSES STARTED: ['2026-09-16', '2026-09-16', '2026-09-16']
button enabled during the build? True
```

So: ONE redo in flight per panel, the button grey from the click until the
answer - EVERY answer, including a build that raised - and a session switch
mid-build neither releases it early nor strands it grey.

**ALSO (advisory 1):** the NIGHT branch now runs the same
`day_review_pack.validated_session` check the queue branch does, and
`run_ai_jobs.py --session` refuses a session that has not CLOSED. Today's
provisional pick used to launch a process and report the story as under way.

**NO MODEL IS CALLED AND NO PROCESS IS SPAWNED**: the launcher is injected and
the builder is a fake that blocks on an Event.
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

#: A Saturday morning: the picker's newest completed session is Friday 09-18.
SATURDAY = datetime(2026, 9, 19, 8, 0)
#: Monday morning, before the close: `Today` is offered and has NOT closed.
MONDAY_MORNING = datetime(2026, 9, 21, 9, 0)
SESSION = "2026-09-18"
UNCLOSED = "2026-09-21"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _GatedService:
    """A reader whose pack build BLOCKS until the test lets it through."""

    def __init__(self, gate, *, built=True, raises=False) -> None:
        self.calls: list[tuple[str, int]] = []
        # LEAD AMENDMENT 2026-09-20 (review round 4): set once a build has
        # STARTED on the worker thread, so a test can wait for that instead of
        # counting `calls` while the worker is still on its way to the append.
        self.started = threading.Event()
        self._gate = gate
        self._built = built
        self._raises = raises

    def read_day(self, session_date, **_kwargs):
        return {"session_date": session_date}

    def build_pack_for(self, session_date, **_kwargs):
        self.calls.append((str(session_date), threading.get_ident()))
        self.started.set()
        self._gate.wait(5.0)
        if self._raises:
            raise RuntimeError("the session's stores were unreadable")
        return {"schema": "day_review_pack_v1"} if self._built else None


def _panel(qapp, monkeypatch, service, *, launched, window_open, clock=SATURDAY):
    from ui.panels import day_review_panel as module
    from ui.panels.day_review_panel import DayReviewPanel

    monkeypatch.setattr(
        module.window, "launch_allowed",
        lambda *_a, **_k: (window_open, "window open" if window_open else "outside"),
        raising=False,
    )
    panel = DayReviewPanel(
        service=service,
        clock=lambda: clock,
        redo_launcher=lambda session: launched.append(session),
    )
    monkeypatch.setattr(panel, "reload", lambda: None)
    return panel


def _drain(qapp, panel, *, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and panel._redo_workers:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def _close(qapp, panel) -> None:
    try:
        panel.shutdown()
    except Exception:  # noqa: BLE001
        pass
    panel.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# one at a time
# ---------------------------------------------------------------------------


def test_three_clicks_start_one_build_and_one_process(qapp, tmp_path, monkeypatch):
    """The reviewer's reproduction. `build_pack_for` with no payload runs a FULL
    `read_day` - the walk-away build, the D1 bars, the grader - and then writes
    `pack.json`; three of those racing on one file, plus three child processes,
    came from one impatient double-click."""
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    gate = threading.Event()
    service = _GatedService(gate)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=True)
    try:
        panel.redo_story()
        panel.redo_story()
        panel.redo_story()

        assert panel.redo_story_button.isEnabled() is False
        assert "Building" in panel.status.text(), panel.status.text()
        gate.set()
        _drain(qapp, panel)

        assert [name for name, _thread in service.calls] == [SESSION], service.calls
        assert launched == [SESSION], launched
        assert list(root.rglob("redo_requested.json")) == []
        assert panel.redo_story_button.isEnabled() is True
    finally:
        gate.set()
        _close(qapp, panel)


def test_three_clicks_by_day_write_one_marker(qapp, tmp_path, monkeypatch):
    import day_review_pack
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    gate = threading.Event()
    service = _GatedService(gate)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=False)
    try:
        panel.redo_story()
        panel.redo_story()
        panel.redo_story()
        gate.set()
        _drain(qapp, panel)

        assert [name for name, _thread in service.calls] == [SESSION], service.calls
        assert launched == []
        assert [path.parent.name for path in root.rglob("redo_requested.json")] == [
            SESSION
        ]
        assert day_review_pack.redo_requested(SESSION, root=root) is True
    finally:
        gate.set()
        _close(qapp, panel)


def test_the_note_of_the_build_in_flight_is_left_alone(qapp, tmp_path, monkeypatch):
    """A second click is a no-op, not a second status line: the page must not
    tell the trader something new happened when nothing did."""
    import project_paths
    from ui.panels.day_review_panel import STORY_BUILDING_PACK_NOTE

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path / "d", raising=False)
    gate = threading.Event()
    service = _GatedService(gate)
    panel = _panel(qapp, monkeypatch, service, launched=[], window_open=False)
    try:
        panel.redo_story()
        during = panel.status.text()
        panel.redo_story()

        assert during == STORY_BUILDING_PACK_NOTE.format(session=SESSION)
        assert panel.status.text() == during
    finally:
        gate.set()
        _drain(qapp, panel)
        _close(qapp, panel)


def test_a_new_click_is_allowed_once_the_first_has_answered(qapp, tmp_path, monkeypatch):
    """One at a time is not once per session: the verb still works twice."""
    import project_paths

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path / "d", raising=False)
    gate = threading.Event()
    gate.set()
    service = _GatedService(gate)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=True)
    try:
        panel.redo_story()
        _drain(qapp, panel)
        panel.redo_story()
        _drain(qapp, panel)

        assert len(service.calls) == 2, service.calls
        assert launched == [SESSION, SESSION]
        assert panel.redo_story_button.isEnabled() is True
    finally:
        _close(qapp, panel)


# ---------------------------------------------------------------------------
# the button comes back from every ending
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "built, window_open, raises, clock",
    [
        (True, False, False, SATURDAY),        # queued for tonight
        (True, True, False, SATURDAY),         # launched now
        (False, False, False, SATURDAY),       # no pack could be built
        (True, True, False, MONDAY_MORNING),   # refused: the session is not closed
    ],
)
def test_the_button_comes_back_from_every_ending(
    qapp, tmp_path, monkeypatch, built, window_open, raises, clock
):
    import project_paths

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path / "d", raising=False)
    gate = threading.Event()
    service = _GatedService(gate, built=built, raises=raises)
    panel = _panel(
        qapp, monkeypatch, service, launched=[], window_open=window_open, clock=clock
    )
    if clock is MONDAY_MORNING:
        index = panel.session_picker.findData(UNCLOSED)
        assert index >= 0, "fixture drift: Today is not offered"
        panel.session_picker.setCurrentIndex(index)
    try:
        panel.redo_story()
        assert panel.redo_story_button.isEnabled() is False
        gate.set()
        _drain(qapp, panel)

        assert panel.redo_story_button.isEnabled() is True
        assert panel._redo_busy is False
    finally:
        gate.set()
        _close(qapp, panel)


def test_a_build_that_raises_re_enables_the_button_and_says_so(
    qapp, tmp_path, monkeypatch
):
    """Never grey for ever. A build that blew up is an ANSWER: it says no pack
    could be built and the verb comes back."""
    import day_review_pack
    import project_paths
    from ui.panels.day_review_panel import REDO_NO_PACK_NOTE

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    gate = threading.Event()
    gate.set()
    service = _GatedService(gate, raises=True)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=True)
    try:
        panel.redo_story()
        _drain(qapp, panel)

        assert panel.redo_story_button.isEnabled() is True
        assert panel.status.text() == REDO_NO_PACK_NOTE.format(session=SESSION)
        assert launched == []
        assert day_review_pack.redo_requested(SESSION, root=root) is False
    finally:
        _close(qapp, panel)


def test_a_launcher_that_raises_re_enables_the_button(qapp, tmp_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path / "d", raising=False)
    gate = threading.Event()
    gate.set()
    service = _GatedService(gate)

    from ui.panels import day_review_panel as module
    from ui.panels.day_review_panel import DayReviewPanel

    monkeypatch.setattr(
        module.window, "launch_allowed", lambda *_a, **_k: (True, "open"), raising=False
    )

    def _boom(_session):
        raise OSError("the child process could not be started")

    panel = DayReviewPanel(
        service=service, clock=lambda: SATURDAY, redo_launcher=_boom
    )
    monkeypatch.setattr(panel, "reload", lambda: None)
    try:
        panel.redo_story()
        _drain(qapp, panel)

        assert panel.redo_story_button.isEnabled() is True
        assert "could not be started" in panel.status.text()
    finally:
        _close(qapp, panel)


def test_a_session_switch_mid_build_neither_frees_nor_strands_the_button(
    qapp, tmp_path, monkeypatch
):
    """Switching the picker while a build runs must not let a second build
    through, and must not leave the verb grey once the first one answers."""
    import project_paths

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path / "d", raising=False)
    gate = threading.Event()
    service = _GatedService(gate)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=True)
    try:
        panel.redo_story()
        panel.show_session("2026-09-17")
        qapp.processEvents()

        assert panel.redo_story_button.isEnabled() is False
        panel.redo_story()
        # LEAD AMENDMENT 2026-09-20 (review round 4): this count raced the
        # worker thread (2 failures in 8 runs, reading 0 before the append).
        # Wait until the ONE build has started, then count: a second build
        # would have appended a second call by the time the gate opens, and
        # the exact-list assertion after the drain catches a late one.
        assert service.started.wait(5.0), "the first build never started"
        assert len(service.calls) == 1, service.calls

        gate.set()
        _drain(qapp, panel)

        assert panel.redo_story_button.isEnabled() is True
        # The build that ran is the session that was selected when it started.
        assert [name for name, _thread in service.calls] == [SESSION], service.calls
        assert launched == [SESSION]
    finally:
        gate.set()
        _close(qapp, panel)


# ---------------------------------------------------------------------------
# advisory 1 - one rule for what a redo may name
# ---------------------------------------------------------------------------


def test_a_session_that_has_not_closed_is_refused_at_night_too(
    qapp, tmp_path, monkeypatch
):
    """The night branch used to launch a process for the picker's provisional
    Today and report the story as under way (reviewer round 3)."""
    import day_review_pack
    import project_paths
    from ui.panels.day_review_panel import REDO_REFUSED_NOTE

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    gate = threading.Event()
    gate.set()
    service = _GatedService(gate)
    launched: list[str] = []
    panel = _panel(
        qapp, monkeypatch, service, launched=launched, window_open=True,
        clock=MONDAY_MORNING,
    )
    index = panel.session_picker.findData(UNCLOSED)
    assert index >= 0, "fixture drift: Today is not offered"
    panel.session_picker.setCurrentIndex(index)
    try:
        panel.redo_story()
        _drain(qapp, panel)

        assert launched == [], "a session with no closed tape must not be narrated"
        assert day_review_pack.redo_requested(UNCLOSED, root=root) is False
        assert panel.status.text().startswith(
            REDO_REFUSED_NOTE.format(session=UNCLOSED, reason="")[:20]
        ), panel.status.text()
        assert "closed" in panel.status.text()
    finally:
        _close(qapp, panel)


def test_the_cli_refuses_a_session_that_has_not_closed(monkeypatch, capsys):
    """One rule, both doors: `run_ai_jobs.py --session` checked `is_session`
    but not "closed", so the same provisional pick parsed and ran."""
    import market_calendar
    import run_ai_jobs
    from ai_jobs import runner

    monkeypatch.setattr(
        runner, "run_slots", lambda *a, **k: pytest.fail("nothing may run")
    )
    monkeypatch.setattr(
        market_calendar, "last_completed_session",
        lambda *_a, **_k: market_calendar.date.fromisoformat(SESSION),
        raising=False,
    )

    with pytest.raises(SystemExit) as refused:
        run_ai_jobs.main(["--slot", "day_review_narration", "--session", UNCLOSED])

    assert refused.value.code == 2
    complaint = capsys.readouterr().err
    assert "closed" in complaint, complaint
    assert "unrecognized" not in complaint.lower()


def test_the_cli_still_accepts_a_closed_session(monkeypatch):
    import run_ai_jobs

    monkeypatch.setattr(run_ai_jobs, "_print_status", lambda *_a, **_k: 0)

    assert run_ai_jobs.main(
        ["--status", "--slot", "day_review_narration", "--force", "--session", SESSION]
    ) == 0
