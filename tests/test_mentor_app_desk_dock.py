"""Dock the Mentor (trader 2026-10-01): the desk publishes its Mentor tab's spot, the app floats over it.

Desk side: ``MentorDockPublisher`` writes the window handle, the placeholder's global rect,
whether the tab is current and whether the desk is on screen. App side: ``DeskDock``
follows that file (fake file, fake clock, fake owner call here - no real window handles).
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the dock is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QTabWidget, QWidget  # noqa: E402


@pytest.fixture()
def app():
    return QApplication.instance() or QApplication([])


def _wait(app, ms=350):
    from PySide6.QtTest import QTest

    QTest.qWait(ms)
    app.processEvents()


# --------------------------------------------------------------------------- the desk
def test_the_publisher_writes_the_spot_on_tab_change(app, tmp_path):
    from ui.services.mentor_dock_publisher import MentorDockPublisher

    tabs = QTabWidget()
    setups, spot = QLabel("setups"), QWidget()
    tabs.addTab(setups, "Setups")
    tabs.addTab(spot, "Mentor")
    tabs.resize(600, 400)
    tabs.show()
    written: list[dict] = []
    publisher = MentorDockPublisher(tabs, spot, path=tmp_path / "dock.json", writer=lambda _p, d: written.append(d),
                                    clock=lambda: 1000.0)
    try:
        tabs.setCurrentWidget(spot)
        _wait(app)
        assert written, "a tab change publishes (debounced)"
        last = written[-1]
        assert last["tab_current"] is True and last["desk_visible"] is True
        assert last["rect"]["w"] == spot.width() and last["rect"]["h"] == spot.height() and last["rect"]["w"] > 0
        assert last["dpr"] > 0 and last["at"] == 1000.0 and "hwnd" in last
        tabs.setCurrentWidget(setups)
        _wait(app)
        assert written[-1]["tab_current"] is False, "leaving the tab is published so the Mentor hides"
        count = len(written)
        tabs.move(tabs.x() + 30, tabs.y() + 30)
        _wait(app)
        assert len(written) == count, "a desk move while the tab is hidden writes nothing"
    finally:
        publisher.shutdown()
        tabs.close()


def test_leaving_the_desk_page_hides_the_mentor(app, tmp_path):
    from PySide6.QtWidgets import QStackedWidget

    from ui.services.mentor_dock_publisher import MentorDockPublisher

    pages = QStackedWidget()
    tabs = QTabWidget()
    spot = QWidget()
    tabs.addTab(QLabel("setups"), "Setups")
    tabs.addTab(spot, "Mentor")
    journal = QLabel("journal")
    pages.addWidget(tabs)
    pages.addWidget(journal)
    pages.resize(600, 400)
    pages.show()
    written: list[dict] = []
    publisher = MentorDockPublisher(tabs, spot, path=tmp_path / "dock.json", writer=lambda _p, d: written.append(d),
                                    clock=lambda: 1000.0)
    try:
        tabs.setCurrentWidget(spot)
        _wait(app)
        assert written[-1]["tab_current"] is True
        pages.setCurrentWidget(journal)
        _wait(app)
        assert written[-1]["tab_current"] is False, "another desk page (Journal) hides the docked Mentor"
        pages.setCurrentWidget(tabs)
        _wait(app)
        assert written[-1]["tab_current"] is True, "coming back shows it again"
    finally:
        publisher.shutdown()
        pages.close()


def test_a_failed_write_never_raises(tmp_path):
    from ui.services.mentor_dock_publisher import read_dock_file, write_dock_file

    assert write_dock_file(tmp_path / "dock.json", {"rect": {"x": 1}}) is True
    assert read_dock_file(tmp_path / "dock.json") == {"rect": {"x": 1}}
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert write_dock_file(blocker / "sub" / "dock.json", {"rect": {}}) is False
    assert read_dock_file(tmp_path / "missing.json") is None


def test_the_desk_has_a_mentor_tab_that_publishes(app, monkeypatch):
    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        workspace = window.trading_panel.master_workspace
        names = [workspace.tabs.tabText(i) for i in range(workspace.tabs.count())]
        assert "Mentor" in names and names[0] == "Setups"
        written: list[dict] = []
        workspace.mentor_dock_publisher._writer = lambda _p, d: written.append(d)
        window.trading_panel.set_setups_visible(True)
        workspace.tabs.setCurrentWidget(workspace.mentor_placeholder)
        workspace.mentor_dock_publisher.publish_now()
        assert written and written[-1]["tab_current"] is True
        window.trading_panel.set_setups_visible(False)
        workspace.mentor_dock_publisher.publish_now()
        assert written[-1]["tab_current"] is False, "hiding the setups column hides the docked Mentor too"
        assert "Dock" in workspace.mentor_placeholder.hint.text()
    finally:
        window.close()
        window.deleteLater()


# --------------------------------------------------------------------------- the Mentor
class _World:
    def __init__(self, tmp_path):
        self.path = tmp_path / "mentor_dock.json"
        self.now = 1000.0
        self.alive = True
        self.owners: list[tuple[int, int]] = []
        self.seq = 1_000_000

    def publish(self, *, tab=True, visible=True, rect=(100, 200, 640, 480), hwnd=4242, at=None):
        payload = {
            "version": 1, "hwnd": hwnd, "rect": dict(zip("xywh", rect, strict=True)), "dpr": 1.0,
            "tab_current": tab, "desk_visible": visible, "at": self.now if at is None else at,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.seq += 1  # every publish is a new mtime, as a real write would be
        os.utime(self.path, (self.seq, self.seq))

    def dock(self, window):
        from mentor_app.desk_dock import DeskDock

        return DeskDock(
            window, path=self.path, set_owner=lambda own, owner: self.owners.append((own, owner)),
            desk_alive=lambda _hwnd: self.alive, clock=lambda: self.now,
        )


def test_the_window_follows_the_desks_spot_and_hides_off_tab(app, tmp_path):
    world = _World(tmp_path)
    win = QWidget()
    win.resize(500, 300)
    win.show()
    dock = world.dock(win)
    try:
        world.publish()
        dock.dock()
        assert dock.docked and win.isVisible()
        assert win.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert win.geometry() == QRect(100, 200, 640, 480)
        assert world.owners and world.owners[-1][1] == 4242, "the desk window owns the docked Mentor"
        world.publish(rect=(120, 220, 700, 500))
        dock.poll()
        assert win.geometry() == QRect(120, 220, 700, 500), "a desk move or resize moves the Mentor"
        world.publish(tab=False)
        dock.poll()
        assert not win.isVisible(), "hidden while the Mentor tab is not current"
        world.publish(visible=False)
        dock.poll()
        assert not win.isVisible(), "hidden while the desk is minimized"
        world.publish()
        dock.poll()
        assert win.isVisible()
    finally:
        dock.shutdown()
        win.close()


def test_a_stale_file_with_no_desk_undocks_with_a_note(app, tmp_path):
    world = _World(tmp_path)
    win = QWidget()
    win.setGeometry(50, 60, 500, 300)
    win.show()
    dock = world.dock(win)
    notes: list[str] = []
    dock.note.connect(notes.append)
    try:
        world.publish()
        dock.dock()
        world.now += 30
        dock.poll()
        assert dock.docked and win.isVisible(), "an old file is fine while the desk window lives"
        world.alive = False
        dock.poll()
        assert dock.docked and not win.isVisible(), "the desk is gone: hide at once"
        world.now += 11
        dock.poll()
        assert not dock.docked and notes and "undocked" in notes[-1]
        assert not (win.windowFlags() & Qt.WindowType.FramelessWindowHint), "the free window is back"
        assert win.isVisible() and win.geometry() == QRect(50, 60, 500, 300)
        assert world.owners[-1][1] == 0, "the owner is cleared"
    finally:
        dock.shutdown()
        win.close()


def test_a_missing_file_undocks_after_ten_seconds(app, tmp_path):
    world = _World(tmp_path)
    win = QWidget()
    win.show()
    dock = world.dock(win)
    try:
        dock.dock()
        assert dock.docked and not win.isVisible()
        world.now += 5
        dock.poll()
        assert dock.docked, "the desk may still be starting"
        world.now += 6
        dock.poll()
        assert not dock.docked
    finally:
        dock.shutdown()
        win.close()


@pytest.fixture()
def mentor_windows(app, tmp_path, monkeypatch):
    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    world = _World(tmp_path)
    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    windows: list = []

    def new_window():
        win = MentorWindow(store=store, queue=PrefetchQueue(), mentor_enabled=False, desk_dock=world.dock)
        windows.append(win)
        return win

    yield world, store, new_window
    for win in windows:
        win.shutdown()
        win.deleteLater()


def _drain_io(win):
    win._io.submit(lambda: None).result(timeout=5)
    QApplication.processEvents()


def test_the_dock_button_round_trips_and_is_remembered(mentor_windows):
    from mentor_app.desk_dock import STATE_KEY

    world, store, new_window = mentor_windows
    world.publish()
    win = new_window()
    win.show()
    assert win.dock_button.text() == "Dock" and win.dock_button.toolTip()
    win.dock_button.click()
    _drain_io(win)
    assert win.desk_dock.docked and win.dock_button.text() == "Undock"
    assert win.geometry() == QRect(100, 200, 640, 480)
    assert store.get_state(STATE_KEY) == "on"
    # A restart re-docks.
    again = new_window()
    again._load_dock_state()
    QApplication.processEvents()
    assert again.desk_dock.docked and again.dock_button.text() == "Undock"
    again.toggle_dock()
    _drain_io(again)
    assert not again.desk_dock.docked and again.dock_button.text() == "Dock"
    assert store.get_state(STATE_KEY) == "off"
    win.toggle_dock()
    _drain_io(win)
    assert not (win.windowFlags() & Qt.WindowType.FramelessWindowHint)


def test_the_owner_is_set_again_when_windows_drops_it(app, tmp_path):
    """Qt clears a Tool window's owner on show; the dock checks the real owner every poll."""
    from mentor_app.desk_dock import DeskDock

    world = _World(tmp_path)
    real = {"owner": 0}

    def set_owner(own, owner):
        world.owners.append((own, owner))
        real["owner"] = owner

    win = QWidget()
    win.show()
    dock = DeskDock(
        win, path=world.path, set_owner=set_owner, get_owner=lambda _own: real["owner"],
        desk_alive=lambda _hwnd: True, clock=lambda: world.now,
    )
    try:
        world.publish()
        dock.dock()
        assert real["owner"] == 4242
        real["owner"] = 0  # a hide/show (tab switch) wiped it
        dock.poll()
        assert real["owner"] == 4242, "a lost owner leaves the Mentor behind the desk, out of reach"
        sets = len(world.owners)
        dock.poll()
        assert len(world.owners) == sets, "an owner already right is not set again"
    finally:
        dock.shutdown()
        win.close()
