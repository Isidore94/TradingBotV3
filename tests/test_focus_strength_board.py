"""The Focus strength board beside the Alert Center tab stack.

Focus picks are pinned above the field, a pick ranking against its own thesis
is called out rather than hidden, and unranked Focus names are named rather
than silently dropped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui.models.focus_strength import (  # noqa: E402
    build_strength_board,
    focus_membership,
)


def _payload(**overrides) -> dict:
    payload = {
        "timeframe_key": "M5",
        "threshold": 0.5,
        "timestamp": "10:42:00",
        # (side, symbol, rrs) triples - the legacy list shape rrs_rows parses.
        "results": [
            ("RS", "NVDA", 1.8),
            ("RS", "AMD", 1.4),
            ("RS", "AVGO", 1.1),
            ("RS", "MU", 0.9),
            ("RW", "SOFI", -1.2),
            ("RW", "RIVN", -0.9),
            ("RW", "PLTR", -0.7),
            ("RS", "NOISE", 0.2),  # below threshold, must not rank
        ],
        "results_sector": [("RS", "AVGO", 2.4), ("RW", "HOOD", -1.5)],
        "results_industry": [("RS", "MU", 3.1)],
    }
    payload.update(overrides)
    return payload


def _focus(**overrides) -> dict:
    focus = {
        "swing": {"long": ["NVDA", "TSLA"], "short": []},
        "m5": {"long": ["AVGO"], "short": ["SOFI"]},
    }
    focus.update(overrides)
    return focus


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
def test_membership_marks_a_symbol_focused_in_both_categories():
    membership = focus_membership(
        {"swing": {"long": ["NVDA"]}, "m5": {"long": ["NVDA", "AMD"]}}
    )
    assert membership["NVDA"] == ("long", "both")
    assert membership["AMD"] == ("long", "m5")


def test_membership_ignores_malformed_entries():
    membership = focus_membership(
        {"swing": {"long": [" nvda ", "", None], "sideways": ["X"]}, "m5": "nope"}
    )
    assert membership == {"NVDA": ("long", "swing")}


# ---------------------------------------------------------------------------
# Board composition
# ---------------------------------------------------------------------------
def test_focus_names_are_pinned_out_of_the_field():
    board = build_strength_board(_payload(), _focus())
    focus_symbols = [row.symbol for row in board.focus]
    assert set(focus_symbols) == {"NVDA", "AVGO", "SOFI"}
    field = {row.symbol for row in board.strong} | {row.symbol for row in board.weak}
    assert field.isdisjoint(focus_symbols)
    assert "AMD" in field and "RIVN" in field


def test_a_symbol_is_reported_where_it_ranks_best_with_the_scope_named():
    board = build_strength_board(_payload(), _focus())
    avgo = next(row for row in board.focus if row.symbol == "AVGO")
    # #3 vs SPY but #1 vs its sector - the board shows the better read.
    assert (avgo.scope, avgo.rank) == ("Sector", 1)
    assert avgo.rank_text() == "#1 vs Sector"

    mu = next(row for row in board.strong if row.symbol == "MU")
    assert (mu.scope, mu.rank) == ("Industry", 1)


def test_one_symbol_never_occupies_two_rows():
    board = build_strength_board(_payload(), _focus())
    every = [row.symbol for row in board.focus + board.strong + board.weak]
    assert len(every) == len(set(every))


def test_a_pick_ranking_against_its_own_thesis_is_flagged_and_leads():
    """A long Focus pick sitting in relative weakness is the row to read first."""
    board = build_strength_board(
        _payload(results=[("RW", "NVDA", -1.4), ("RS", "AMD", 1.2)]),
        {"swing": {"long": ["NVDA"], "short": []}},
    )
    nvda = board.focus[0]
    assert nvda.symbol == "NVDA"
    assert nvda.aligned is False
    assert board.misaligned == [nvda]

    # A short pick in relative weakness agrees with its thesis.
    agreeing = build_strength_board(
        _payload(results=[("RW", "SOFI", -1.4)]), {"m5": {"short": ["SOFI"]}}
    )
    assert agreeing.focus[0].aligned is True


def test_misaligned_picks_sort_ahead_of_aligned_ones():
    board = build_strength_board(
        _payload(results=[("RS", "NVDA", 2.5), ("RW", "TSLA", -0.8)]),
        {"swing": {"long": ["NVDA", "TSLA"]}},
    )
    assert [row.symbol for row in board.focus] == ["TSLA", "NVDA"]


def test_unranked_focus_names_are_named_not_dropped():
    board = build_strength_board(_payload(), _focus())
    # TSLA is focused but appears in no scope's results.
    assert board.unranked_focus == ["TSLA"]


def test_below_threshold_rows_never_rank():
    board = build_strength_board(_payload(), {})
    assert "NOISE" not in {row.symbol for row in board.strong}


def test_limits_bound_each_lane():
    board = build_strength_board(_payload(), _focus(), focus_limit=2, field_limit=1)
    assert len(board.focus) == 2
    assert len(board.strong) == 1
    assert len(board.weak) == 1


def test_empty_payload_is_honestly_empty():
    board = build_strength_board(None, _focus())
    assert board.is_empty
    assert board.focus == []
    # Every focused name is unranked when there is no sweep at all.
    assert board.unranked_focus == ["AVGO", "NVDA", "SOFI", "TSLA"]


# ---------------------------------------------------------------------------
# Widget + Alert Center wiring (offscreen Qt)
# ---------------------------------------------------------------------------
def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class _StubFocusService:
    """Minimal stand-in: the board only needs the categories and a signal."""

    def __init__(self, focus):
        from PySide6.QtCore import QObject, Signal

        class _Emitter(QObject):
            focusChanged = Signal()

        self._emitter = _Emitter()
        self.focusChanged = self._emitter.focusChanged
        self._focus = focus

    def all_focus_by_category(self):
        return self._focus


def test_widget_renders_focus_names_and_survives_a_broken_store():
    if _qt_app() is None:
        return
    from ui.widgets.focus_strength_board import FocusStrengthBoard

    board = FocusStrengthBoard()
    board.set_focus_service(_StubFocusService(_focus()))
    board.update_snapshot(_payload())

    html = board.board.toHtml()
    assert "NVDA" in html and "YOUR FOCUS" in html.upper()
    assert "TSLA" in html  # the unranked-focus line
    assert board.current_board().focus

    text = board.board.toPlainText()
    # Focus lane above the field, and both fit in one render - a two-line
    # focus row used to push the field off the board entirely.
    assert text.index("YOUR FOCUS") < text.index("FIELD") < text.index("AMD")
    # One line per pick: marker + symbol, RRS, and where it ranks.
    assert "#1 SPY" in text and "· m5" in text
    assert "#1 SEC" in text  # scopes abbreviated so the line clears 170px
    # A glance board must never hide its numbers behind a horizontal drag.
    from PySide6.QtCore import Qt

    assert board.board.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    class _Broken(_StubFocusService):
        def all_focus_by_category(self):
            raise RuntimeError("store unavailable")

    board.set_focus_service(_Broken(_focus()))
    board.update_snapshot(_payload())
    # Degrades to a field-only board instead of taking the alert column down.
    assert board.current_board().focus == []


def test_alert_center_puts_the_board_beside_the_tab_stack(tmp_path):
    if _qt_app() is None:
        return
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    assert panel.tabs_row.count() == 2
    assert panel.tabs_row.widget(0) is panel.tabs
    assert panel.tabs_row.widget(1) is panel.focus_strength
    # The board is beside the tabs, not inside one, so it stays visible when
    # the trader switches to D1 Focus or Armed.
    assert panel.focus_strength.parent() is not panel.tabs
    assert panel.splitter.widget(1) is panel.tabs_row
    # Adding the board must not push the alert column past the 360px floor
    # the desk splitter gives it.
    assert panel.tabs.minimumWidth() + panel.focus_strength.minimumWidth() <= 360

    panel.focus_strength.update_snapshot(_payload())
    assert {row.symbol for row in panel.focus_strength.current_board().strong}


def test_board_updates_from_the_same_snapshot_signal_as_the_rrs_tab(tmp_path):
    if _qt_app() is None:
        return
    from PySide6.QtCore import QObject, Signal

    from ui.panels.alert_center_panel import AlertCenterPanel

    class _Service(QObject):
        alertReceived = Signal(object)
        rrsSnapshotChanged = Signal(object)
        statusChanged = Signal(str)

        def current_bot(self):
            return None

    panel = AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")
    service = _Service()
    panel.attach_service(service)
    service.rrsSnapshotChanged.emit(_payload())

    # One payload, both surfaces - no second service, thread, or request.
    assert {row.symbol for row in panel.focus_strength.current_board().strong}
    assert "NVDA" in panel.rrs_snapshot.board.toHtml()
