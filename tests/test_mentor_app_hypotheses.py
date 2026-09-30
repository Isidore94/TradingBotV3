"""Mentor app P11: /hypotheses shows the night's hypotheses with their cells; never a rule proposal."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import commands  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import hypothesis_pack  # noqa: E402


def test_the_hypotheses_command_parses():
    assert commands.handle("/hypotheses").action == "hypotheses"
    assert commands.handle("/hyp").action == "hypotheses"
    assert commands.handle("/hypotheses now").action == "error"
    assert "/hypotheses" in commands.HELP_TEXT


@pytest.fixture
def win(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    world = hypothesis_pack.write_fixture_world(tmp_path / "world")
    window = MentorWindow(
        store=MentorChatStore(world["chat"]), queue=PrefetchQueue(), news_queue=PrefetchQueue(),
        stream_post=lambda *a, **k: [], post=lambda *a, **k: {}, now=lambda: hypothesis_pack.FIXTURE_NOW,
        mentor_enabled=False,
    )
    window.permutation_history, window.permutation_report = world["history"], world["report_file"]
    yield window
    window.shutdown()
    window.deleteLater()


def test_hypotheses_card_shows_open_and_graded_with_their_cells(win):
    from PySide6.QtWidgets import QApplication

    win.send("/hypotheses")
    while win.queue.run_one():
        pass
    QApplication.processEvents()
    text = win.transcript.toPlainText()
    assert "Hypotheses (the night's queries into the shadow permutation grid)" in text
    assert "hyp:2026-09-28:1:cell" in text and "n=64" in text and "GONE" in text
    assert "a cell in the shadow grid; changes go through fixtures and the ladder" in text.lower()
