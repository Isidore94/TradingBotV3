r"""TJ-6 - the ideas card: honest when empty, one click at a time. RED.

`plan.md` §12.4 TJ-6 change 3: *"Card on Day Review and Week Review: the night's
ideas, Keep / Dismiss"*, and live gate #150: *"Dismiss hides one for good across
a restart; Keep on a program idea shows it under For WISHLIST"*.

THE EMPTY STATE IS THE FIRST THING THE TRADER SEES. The live home folder holds
no ideas store at all (`C:\TradingBotData`, listed read-only 2026-09-20), and
`scripts/ui/panels/day_review_panel.py:131` currently says *"Nothing yet - the
desk's AI starts speaking in TJ-6."* So the card must read as honest, not
broken: no ideas yet, ``kept 0 of 0``, a count beside every number and nothing
invented.

THE CLICK RULES (packet TJ-5 "CORRECTED" item 6, learned on another packet the
same week): a click that starts work is ONE at a time with its button disabled
until EVERY ending answers - including a raise - and the write it starts
validates its argument and fails closed. The write here reads a measurable over
`LATELY_SESSIONS` to freeze a baseline, so it belongs on a worker and not on the
Qt thread.

VERIFIED ON THIS BRANCH (1b9d77e0): there is no `scripts/ui/widgets/ideas_card.py`
and `day_review_panel._build_ideas` builds one static label, so every test here
fails on the import.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the ideas card is a PySide6 widget")

from PySide6.QtWidgets import QApplication  # noqa: E402

import tj6_support as fx  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _rows(*, kept: int = 0, total: int = 2):
    """Payload rows in the shape `day_review_service` hands the page."""
    out = []
    for index in range(total):
        row = {
            "idea_id": f"idea:2026-09-18:{index:012d}",
            "kind": "process" if index == 0 else "program",
            "text": f"Idea number {index}.",
            "evidence": ["2026-09-18/card:did_well"],
            "measurable": "veto_reason_real_miss_rate" if index == 0 else "",
            "seen_count": 1,
            "status": "kept" if index < kept else "",
        }
        out.append(row)
    return out


def _drain(qapp, card, *, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and card.is_writing():
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


class _GatedWriter:
    """A keep/dismiss writer that BLOCKS until the test lets it through."""

    def __init__(self, gate, *, raises: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.started = threading.Event()
        self._gate = gate
        self._raises = raises

    def __call__(self, idea_id: str, status: str):
        self.calls.append((str(idea_id), str(status)))
        self.started.set()
        self._gate.wait(5.0)
        if self._raises:
            raise RuntimeError("the state file could not be written")
        return {"idea_id": idea_id, "status": status}


def _card(qapp, **kwargs):
    from ui.widgets.ideas_card import IdeasCard

    return IdeasCard(**kwargs)


def _close(qapp, card) -> None:
    try:
        card.shutdown()
    except Exception:  # noqa: BLE001
        pass
    card.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# the empty state
# ---------------------------------------------------------------------------
def test_with_no_ideas_the_card_says_none_yet_and_kept_zero_of_zero(qapp):
    """Hand-counted: zero ideas, zero kept. The card says both numbers out loud
    and invents nothing - no rate, no "on track", no percentage of nothing."""
    card = _card(qapp)
    try:
        card.show_ideas([])
        summary = card.summary_text().lower()
        assert "0 of 0" in summary, card.summary_text()
        assert "kept" in summary
        assert "%" not in summary, "a percentage over nothing"
        assert card.rows == ()
        assert card.empty_note.isVisible() or card.empty_note.text().strip()
    finally:
        _close(qapp, card)


def test_the_summary_counts_kept_against_offered(qapp):
    """Hand-counted: 2 ideas, 1 of them kept -> "kept 1 of 2"."""
    card = _card(qapp)
    try:
        card.show_ideas(_rows(kept=1, total=2))
        assert "1 of 2" in card.summary_text(), card.summary_text()
        assert len(card.rows) == 2
    finally:
        _close(qapp, card)


def test_the_card_diffs_its_rows_rather_than_rebuilding_them(qapp):
    """CLAUDE.md: lists diff, never rebuild. Two renders of the same two ideas
    must leave the SAME two row widgets in place."""
    card = _card(qapp)
    try:
        card.show_ideas(_rows(total=2))
        first = card.rows
        card.show_ideas(_rows(total=2))
        assert card.rows == first, "the card rebuilt its rows"
    finally:
        _close(qapp, card)


# ---------------------------------------------------------------------------
# one click at a time
# ---------------------------------------------------------------------------
def test_three_clicks_start_one_write_and_the_buttons_go_grey(qapp):
    """The TJ-4 redo blocker in miniature.

    Hand-counted: three clicks -> ONE call to the writer. Every Keep and every
    Dismiss on the card is disabled while it runs, because the write reads a
    measurable over 20 sessions and a second one would race it on the same file.
    """
    gate = threading.Event()
    writer = _GatedWriter(gate)
    card = _card(qapp, writer=writer)
    try:
        card.show_ideas(_rows(total=2))
        card.rows[0].keep_button.click()
        card.rows[0].keep_button.click()
        card.rows[1].dismiss_button.click()

        assert writer.started.wait(5.0), "the write never reached the worker"
        assert card.is_writing() is True
        for row in card.rows:
            assert row.keep_button.isEnabled() is False
            assert row.dismiss_button.isEnabled() is False

        gate.set()
        _drain(qapp, card)

        assert writer.calls == [(card.rows[0].idea_id, "kept")], writer.calls
        assert card.is_writing() is False
        for row in card.rows:
            assert row.keep_button.isEnabled() is True
            assert row.dismiss_button.isEnabled() is True
    finally:
        gate.set()
        _close(qapp, card)


def test_a_write_that_raises_still_answers_and_says_so(qapp):
    """EVERY ending answers. A card left grey forever is a card the trader has
    to restart the desk to use again."""
    gate = threading.Event()
    writer = _GatedWriter(gate, raises=True)
    card = _card(qapp, writer=writer)
    try:
        card.show_ideas(_rows(total=1))
        card.rows[0].keep_button.click()
        assert writer.started.wait(5.0)
        gate.set()
        _drain(qapp, card)

        assert card.is_writing() is False
        assert card.rows[0].keep_button.isEnabled() is True
        assert card.status_text().strip(), "a failed write said nothing"
    finally:
        gate.set()
        _close(qapp, card)


def test_the_write_never_runs_on_the_qt_thread(qapp):
    """The baseline read walks `LATELY_SESSIONS` of evidence. On the Qt thread
    that is the 8.45 s freeze the Week Review worker exists for."""
    gate = threading.Event()
    seen: list[int] = []

    def _writer(idea_id, status):
        seen.append(threading.get_ident())
        gate.wait(5.0)
        return {"idea_id": idea_id, "status": status}

    card = _card(qapp, writer=_writer)
    try:
        card.show_ideas(_rows(total=1))
        card.rows[0].keep_button.click()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and not seen:
            qapp.processEvents()
            time.sleep(0.01)
        gate.set()
        _drain(qapp, card)
        assert seen and seen[0] != threading.get_ident(), "the write ran on the Qt thread"
    finally:
        gate.set()
        _close(qapp, card)


# ---------------------------------------------------------------------------
# what the click actually writes
# ---------------------------------------------------------------------------
def test_the_default_writer_is_the_stores_own_keep_and_dismiss(qapp, tmp_path, monkeypatch):
    """No injected writer: the card must call the ONE writer the state file has.

    Hand-counted: one Keep click on row 0 and one Dismiss on row 1 -> one call
    to each, with those rows' ids and nothing else.
    """
    from ai_jobs import improvement_ideas

    fx.install_stores(monkeypatch, tmp_path, write_the_week=False)
    kept: list[str] = []
    dismissed: list[str] = []
    monkeypatch.setattr(
        improvement_ideas, "keep_idea", lambda idea_id, **kwargs: kept.append(str(idea_id))
    )
    monkeypatch.setattr(
        improvement_ideas,
        "dismiss_idea",
        lambda idea_id, **kwargs: dismissed.append(str(idea_id)),
    )

    card = _card(qapp)
    try:
        card.show_ideas(_rows(total=2))
        card.rows[0].keep_button.click()
        _drain(qapp, card)
        card.rows[1].dismiss_button.click()
        _drain(qapp, card)

        assert kept == [card.rows[0].idea_id], kept
        assert dismissed == [card.rows[1].idea_id], dismissed
    finally:
        _close(qapp, card)


def test_a_kept_program_idea_is_shown_under_for_wishlist(qapp):
    """*"kept `program` ideas listed under 'For WISHLIST - copy' (the trader
    pastes; the AI never writes `WISHLIST.md`)"*.

    Hand-counted: 2 ideas, the `program` one kept -> exactly one line under the
    heading, and it is that idea's text.
    """
    card = _card(qapp)
    try:
        rows = _rows(total=2)
        rows[1]["status"] = "kept"
        card.show_ideas(rows)
        text = card.wishlist_text()
        assert "WISHLIST" in text
        assert rows[1]["text"] in text
        assert rows[0]["text"] not in text, "a process idea was offered for WISHLIST"
    finally:
        _close(qapp, card)
