"""The Focus board re-measured every chip's mover state on every redraw.

Measured 2026-08-25: 36 repeating stalls, 5.93 s total. `_refresh_all` is
called from a lot of places that have nothing to do with previous-day extremes
- a BounceBot alert arriving, an RS/RW snapshot landing, a side editor
changing - and each one walked every chip through
`AlertCenterPanel.mover_state`, which reads the D1 and M5 series and runs
`completed_session_bars` per symbol per side.

The fix is a memo, not a new measurement: resolve each (symbol, side) once per
mover-refresh cycle and hand the same answer to every chip that asks. The
invalidation point is the signal that says a new measurement exists -
`focusBreakStatesChanged` -> `refresh_mover_flags` - so the board is never
pinned to an answer older than the poll that produced it.

Scope note: this lives in `focus_picks_panel.py`, the CONSUMER. The natural
memo point would be `AlertCenterPanel._measure_mover_state`, but that file is
fenced under the file-scoped ask-first rule and the trader's pre-authorization
covers only the quick-journal symbols attachment. Same effect for this board,
no fenced edit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _CountingResolver:
    """Stands in for `AlertCenterPanel.mover_state` and counts every ask."""

    def __init__(self, answer: str = "OPEN") -> None:
        self.calls: list[tuple[str, str]] = []
        self.answer = answer

    def __call__(self, symbol: str, side: str) -> str:
        self.calls.append((symbol, side))
        return self.answer


def _panel(tmp_path):
    """A real panel over a tmp store - no writes to the trader's watchlists."""
    from focus_picks import FocusPickStore
    from ui.panels.focus_picks_panel import FocusPicksPanel
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )
    service = FocusService(store)
    service.add("AAA", "long", "swing")
    service.add("BBB", "short", "swing")
    return FocusPicksPanel(service)


def test_one_measurement_per_symbol_side_per_refresh_cycle(tmp_path):
    """Repeated redraws between polls must not re-walk the bars."""
    panel = _panel(tmp_path)
    resolver = _CountingResolver()
    panel.set_mover_source(resolver)

    resolver.calls.clear()
    # Three unrelated events that each trigger a full board redraw.
    panel._refresh_all()
    panel._refresh_all()
    panel._refresh_all()

    distinct = set(resolver.calls)
    assert len(resolver.calls) == len(distinct), (
        f"{len(resolver.calls)} measurements for {len(distinct)} distinct "
        "(symbol, side) pairs - the board is re-measuring on every redraw"
    )


def test_the_same_pair_asked_twice_in_one_cycle_is_measured_once(tmp_path):
    """Directly: two asks, one measurement, same answer."""
    panel = _panel(tmp_path)
    resolver = _CountingResolver("OPEN")
    panel.set_mover_source(resolver)
    resolver.calls.clear()

    # A symbol the board has not drawn, so the count is only what these two
    # asks cost - installing the source already measured the chips on screen.
    first = panel._mover_state_for("ZZZ", "long")
    second = panel._mover_state_for("ZZZ", "long")

    assert first == second == "OPEN"
    assert resolver.calls.count(("ZZZ", "long")) == 1, resolver.calls


def test_a_new_poll_re_measures_rather_than_serving_the_old_answer(tmp_path):
    """The memo must expire on the signal that says a new answer exists.

    Without this the board would latch its first reading for the life of the
    session - a stale flag is worse than a slow one, because it silently keeps
    asserting a break that has since closed.
    """
    panel = _panel(tmp_path)
    resolver = _CountingResolver("OPEN")
    panel.set_mover_source(resolver)

    assert panel._mover_state_for("AAA", "long") == "OPEN"
    resolver.answer = "CLOSED"

    # Nothing new has been measured yet, so the cached answer still stands.
    assert panel._mover_state_for("AAA", "long") == "OPEN"

    # The D1 poll landed: `focusBreakStatesChanged` -> `refresh_mover_flags`.
    panel.refresh_mover_flags()
    assert panel._mover_state_for("AAA", "long") == "CLOSED"


def test_installing_a_new_source_discards_the_old_answers(tmp_path):
    panel = _panel(tmp_path)
    first = _CountingResolver("OPEN")
    panel.set_mover_source(first)
    assert panel._mover_state_for("AAA", "long") == "OPEN"

    second = _CountingResolver("CLOSED")
    panel.set_mover_source(second)
    assert panel._mover_state_for("AAA", "long") == "CLOSED"


def test_a_failed_measurement_is_not_cached_as_an_answer(tmp_path):
    """A flag is decoration over a measurement; a failure is not a verdict.

    Caching the empty string a raising resolver produces would turn one
    transient miss into a flag that stays off until the next poll.
    """
    panel = _panel(tmp_path)

    state = {"boom": True}

    def flaky(symbol, side):
        if state["boom"]:
            raise RuntimeError("bars unavailable")
        return "OPEN"

    panel.set_mover_source(flaky)
    assert panel._mover_state_for("AAA", "long") == ""

    state["boom"] = False
    assert panel._mover_state_for("AAA", "long") == "OPEN", (
        "a failed measurement was cached as though it were an answer"
    )


def test_no_source_still_means_no_flag(tmp_path):
    """A bare panel with no desk behind it shows nothing, and does not raise."""
    panel = _panel(tmp_path)
    assert panel._mover_state_for("AAA", "long") == ""
