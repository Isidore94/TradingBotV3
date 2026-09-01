"""P2 item 3: the trader's own decisions, beside the bot's measurements.

`review_preference_state.json` has carried P(take | shown) per segment, and the
R of what was taken against the R of what was passed, since the review-learning
pass shipped. Its only surface was a text report. The Daytrade Tracker already
answers the same question about what the BOT measured, one tab per dimension,
so the other half of the loop belongs in the same tab strip.

Read-only over a file the review-learning pass writes. Nothing here reaches a
detector, score, alert, Focus list, review queue or `review_policy.json`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.panels import daytrade_tracker_panel as panel_module  # noqa: E402


def _state(**dimensions) -> dict:
    return {
        "schema": "review_learning_v1",
        "generated_at": "2026-09-01T11:10:48",
        "window_days": 90,
        "shown": 2606,
        "takes": 645,
        "overall_take_rate": 0.247,
        "dimensions": dimensions,
    }


def _segment(shown, take, take_rate, taken_r, taken_n, passed_r, passed_n) -> dict:
    return {
        "n": shown,
        "shown": shown,
        "take": take,
        "take_rate": take_rate,
        "take_rate_shrunk": take_rate,
        "taken": {"r_n": taken_n, "r_avg": taken_r, "fwd_n": 0, "fwd_avg_pct": None},
        "passed": {"r_n": passed_n, "r_avg": passed_r, "fwd_n": 0, "fwd_avg_pct": None},
    }


# ==========================================================================
# the projection
# ==========================================================================
def test_a_dimension_projects_every_column_the_tab_shows():
    """Fail-before-fix: `decision_rows` does not exist."""
    state = _state(
        bounce_type={"lrsi_cross_20": _segment(60, 17, 0.283, -0.376, 8, 0.962, 24)}
    )
    rows = panel_module.decision_rows(state, "bounce_type")

    assert len(rows) == 1
    row = rows[0]
    assert row["segment"] == "lrsi_cross_20"
    assert row["shown"] == 60
    assert row["take"] == 17
    assert row["take_rate"] == 0.283
    assert row["taken_r"] == -0.376
    assert row["taken_n"] == 8
    assert row["passed_r"] == 0.962
    assert row["passed_n"] == 24


def test_the_gap_is_taken_minus_passed_and_only_when_both_exist():
    """The one derived number on the tab, and it is a subtraction the trader
    would otherwise do by eye. It must never be a difference against an absent
    average - that would read as a measurement."""
    both = panel_module.decision_rows(
        _state(tier={"A": _segment(20, 5, 0.25, -0.376, 8, 0.962, 24)}), "tier"
    )[0]
    assert both["gap"] == pytest.approx(-1.338)

    one_side = panel_module.decision_rows(
        _state(tier={"A": _segment(20, 5, 0.25, None, 0, 0.962, 24)}), "tier"
    )[0]
    assert one_side["gap"] is None
    assert one_side["taken_r"] is None
    assert one_side["passed_r"] == 0.962


def test_a_segment_with_no_graded_outcome_is_listed_not_dropped():
    """"You saw 40 of these and none has a graded outcome yet" is a real
    answer. Dropping the row would hide it."""
    rows = panel_module.decision_rows(
        _state(tier={"D": _segment(40, 2, 0.05, None, 0, None, 0)}), "tier"
    )
    assert [row["segment"] for row in rows] == ["D"]
    assert rows[0]["shown"] == 40
    assert rows[0]["gap"] is None


def test_rows_order_by_how_often_the_segment_was_shown():
    state = _state(
        tier={
            "A": _segment(5, 1, 0.2, 0.1, 3, 0.2, 2),
            "B": _segment(50, 10, 0.2, 0.1, 3, 0.2, 2),
            "C": _segment(20, 4, 0.2, 0.1, 3, 0.2, 2),
        }
    )
    assert [r["segment"] for r in panel_module.decision_rows(state, "tier")] == ["B", "C", "A"]


def test_a_dimension_the_state_does_not_carry_is_empty_not_an_error():
    assert panel_module.decision_rows(_state(), "bounce_type") == []
    assert panel_module.decision_rows(None, "bounce_type") == []
    assert panel_module.decision_rows({"dimensions": "not a map"}, "tier") == []


# ==========================================================================
# the probation badge
# ==========================================================================
def test_the_probation_badge_is_set_membership_over_the_two_existing_dicts():
    """The R5 engines are on probation and the champions are not. The badge is
    which dict the bounce_type is in and nothing else - no threshold, no second
    list to maintain."""
    from bounce_bot_lib.legacy import BOUNCE_TYPE_DEFAULTS, M5_SIGNAL_TYPE_DEFAULTS

    probation = panel_module._probation_types()

    assert probation == frozenset(set(M5_SIGNAL_TYPE_DEFAULTS) - set(BOUNCE_TYPE_DEFAULTS))
    assert "lrsi_cross_20" in probation
    # A champion is never badged.
    for champion in list(BOUNCE_TYPE_DEFAULTS)[:5]:
        assert champion not in probation


def test_only_a_probation_bounce_type_carries_the_badge():
    state = _state(
        bounce_type={
            "lrsi_cross_20": _segment(60, 17, 0.283, -0.376, 8, 0.962, 24),
            "vwap": _segment(24, 12, 0.5, 0.488, 12, 0.076, 12),
        }
    )
    rows = panel_module.decision_rows(state, "bounce_type", panel_module._probation_types())
    badges = {row["segment"]: row["probation"] for row in rows}

    assert badges["lrsi_cross_20"] == "probation"
    assert badges["vwap"] == ""


def test_an_unreadable_taxonomy_badges_nothing_rather_than_guessing(monkeypatch):
    """The wrong direction to guess in would be calling a champion
    'probation'."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "bounce_bot_lib.legacy":
            raise ImportError("no taxonomy here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert panel_module._probation_types() == frozenset()


# ==========================================================================
# the panel: one tab per dimension, and no file read on the Qt thread
# ==========================================================================
def _settle(_panel, predicate, timeout=10.0):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


@pytest.fixture
def panel(qapp_guard):
    """A panel whose CONSTRUCTION read has already finished.

    `start_decisions_refresh` is single-flight by design, so a test that fires
    a second read while the constructor's is still running is silently ignored
    - fast enough to pass alone and racy under a loaded full-suite run. The
    button is re-enabled by `_on_decisions_loaded`, which is exactly "no read
    in flight".
    """
    made = panel_module.DaytradeTrackerPanel()
    assert _settle(made, lambda: made.decisions_button.isEnabled()), (
        "the construction read never finished"
    )
    yield made
    made.shutdown()
    made.deleteLater()


@pytest.fixture
def qapp_guard():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_the_panel_carries_one_tab_per_scoreboard_dimension(panel):
    labels = [panel.decisions_tabs.tabText(i) for i in range(panel.decisions_tabs.count())]
    assert labels == [label for _key, label in panel_module.DECISION_TABS]
    assert panel.tabs.tabText(panel.tabs.count() - 1) == "My Decisions"


def test_the_scoreboard_is_read_off_the_qt_thread(panel, monkeypatch):
    """The state file is a 34 KB JSON in the home folder. Reading it on the Qt
    thread is the drip these panels have been audited for twice."""
    import threading

    import review_learning

    seen: list[int] = []

    def recording(*args, **kwargs):
        seen.append(threading.get_ident())
        return _state(tier={"A": _segment(10, 3, 0.3, 0.5, 8, -0.2, 9)})

    monkeypatch.setattr(review_learning, "load_review_learning_state", recording)
    panel.start_decisions_refresh(rebuild=False)
    assert _settle(panel, lambda: bool(seen))
    assert seen and seen[0] != threading.main_thread().ident


def test_the_refresh_button_rebuilds_a_stale_scoreboard(panel, monkeypatch):
    """Exactly what `app.py` does at startup, in the same shape."""
    import review_learning

    calls: list[str] = []
    monkeypatch.setattr(
        review_learning,
        "refresh_review_learning_if_stale",
        lambda *a, **k: calls.append("rebuild"),
    )
    monkeypatch.setattr(
        review_learning, "load_review_learning_state", lambda *a, **k: _state()
    )

    panel.start_decisions_refresh(rebuild=True)
    assert _settle(panel, lambda: bool(calls))
    assert calls == ["rebuild"]


def test_construction_reads_but_never_rebuilds(qapp_guard, monkeypatch):
    """Opening the desk must not trigger a rebuild it did not ask for."""
    import review_learning

    calls: list[str] = []
    monkeypatch.setattr(
        review_learning,
        "refresh_review_learning_if_stale",
        lambda *a, **k: calls.append("rebuild"),
    )
    reads: list[str] = []
    monkeypatch.setattr(
        review_learning,
        "load_review_learning_state",
        lambda *a, **k: reads.append("read") or _state(),
    )

    made = panel_module.DaytradeTrackerPanel()
    try:
        assert _settle(made, lambda: bool(reads))
        assert calls == []
    finally:
        made.shutdown()
        made.deleteLater()


def test_an_unreadable_scoreboard_says_so_and_keeps_the_tabs(panel, monkeypatch):
    import review_learning

    def boom(*args, **kwargs):
        raise OSError("the scoreboard is unreadable")

    monkeypatch.setattr(review_learning, "load_review_learning_state", boom)
    panel.start_decisions_refresh(rebuild=False)
    assert _settle(panel, lambda: "unreadable" in panel.decisions_status.text())
    assert panel.decisions_tabs.count() == len(panel_module.DECISION_TABS)


def test_a_missing_scoreboard_reads_as_absent_not_as_no_decisions(panel, monkeypatch):
    import review_learning

    monkeypatch.setattr(review_learning, "load_review_learning_state", lambda *a, **k: None)
    panel.start_decisions_refresh(rebuild=False)
    assert _settle(panel, lambda: "absent measurement" in panel.decisions_status.text())


def test_the_worker_never_emits_into_a_deleted_panel(qapp_guard, monkeypatch):
    """A worker must not outlive the widget it was going to update.

    `shutdown` joins the thread, but deletion can win the race - and emitting
    into a deleted signal source raises `RuntimeError: Signal source has been
    deleted` out of a daemon thread. The full suite caught exactly that as an
    unhandled thread exception before this guard existed.

    Driven deterministically: the C++ object is destroyed first, then the
    worker body is run directly, which is the state the race produces.

    Fail-before-fix: without the guard this raises RuntimeError.
    """
    from PySide6.QtWidgets import QApplication

    import review_learning

    monkeypatch.setattr(review_learning, "load_review_learning_state", lambda *a, **k: _state())

    made = panel_module.DaytradeTrackerPanel()
    assert _settle(made, lambda: made.decisions_button.isEnabled())
    worker = made._decisions_worker

    # `shiboken6.delete` destroys the C++ object deterministically, which is
    # the state the race produces and `deleteLater` alone does not reach while
    # a Python reference survives.
    import shiboken6

    shiboken6.delete(made)
    QApplication.instance().processEvents()

    # There is nothing left to update; the payload is dropped, not raised.
    worker(False)
