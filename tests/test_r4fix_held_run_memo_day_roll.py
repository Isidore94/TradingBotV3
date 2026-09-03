"""R4 fix round 1, blocker 4 - the process memo froze A9's own fix after a day.

`_HELD_RUN_INDEX_MEMO` was set once and nothing reset it or the panel's copy.
Two things break on day 2 of uptime, and the desk is the always-on mini-PC:

* `_held_run_d1_symbols` is keyed by `trade_date`, so with no key for today every
  alert reads `d1_setup_present=False` again - precisely the state A9 was built
  to end (2,459 of 7,603 live episodes now True);
* the index is a 20-TRADING-SESSION window (`held_run_score.ROLLING_SESSIONS`)
  that never rolled, so the suffix stopped being "lately" while still claiming to
  be - against CLAUDE.md's rule that "lately" is ONE number counted in trading
  sessions.

The memo carries `built_for` now and expires on the day roll, rebuilt on the
worker at the first M5 alert of the new day.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp, tmp_path, monkeypatch):
    from ui.panels import alert_center_panel as module

    monkeypatch.setattr(module, "_HELD_RUN_INDEX_MEMO", None, raising=False)
    yield module.AlertCenterPanel(review_events_path=tmp_path / "events.jsonl")


def _alert(symbol="NVDA", *, trade_date):
    from ui.models.bounce import BounceAlert

    alert = BounceAlert(time_text="09:35:00", symbol=symbol, side="LONG", trigger="ema_15")
    alert.payload = {"feedback": {"bounce_types": "ema_15", "trade_date": trade_date,
                                  "market_environment": "trend_up"}}
    return alert


def test_the_memo_carries_the_day_it_was_built_for(panel, monkeypatch):
    from ui.panels import alert_center_panel as module

    monkeypatch.setattr(panel, "_held_run_day", lambda: "2026-09-02")
    monkeypatch.setattr(
        panel,
        "_held_run_index_worker",
        lambda: panel._on_held_run_index_loaded(
            {"index": {}, "d1": {"2026-09-02": {"NVDA"}}, "built_for": "2026-09-02"}
        ),
    )

    panel._on_held_run_index_loaded(
        {"index": {}, "d1": {"2026-09-02": {"NVDA"}}, "built_for": "2026-09-02"}
    )

    assert panel._held_run_built_for == "2026-09-02"
    assert module._HELD_RUN_INDEX_MEMO is None or isinstance(module._HELD_RUN_INDEX_MEMO, dict)


def test_a_memo_from_yesterday_is_rebuilt_rather_than_reused(panel, monkeypatch):
    """Day 2 of uptime. This is the whole blocker.

    The REAL `_ensure_held_run_index` runs - only the worker body is replaced,
    and the thread it starts is joined - so the test exercises the expiry rather
    than a copy of it.
    """
    from ui.panels import alert_center_panel as module

    monkeypatch.setattr(
        module,
        "_HELD_RUN_INDEX_MEMO",
        {"index": {}, "d1": {"2026-09-01": {"NVDA"}}, "built_for": "2026-09-01"},
        raising=False,
    )
    monkeypatch.setattr(panel, "_held_run_day", lambda: "2026-09-02")

    rebuilt: list[str] = []

    def _worker():
        rebuilt.append(panel._held_run_day())
        panel._heldRunIndexLoaded.emit(
            {"index": {}, "d1": {"2026-09-02": {"NVDA"}}, "built_for": "2026-09-02"}
        )

    monkeypatch.setattr(panel, "_held_run_index_worker", _worker)

    panel._ensure_held_run_index()
    if panel._held_run_thread is not None:
        panel._held_run_thread.join(timeout=5.0)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()

    assert rebuilt == ["2026-09-02"], "yesterday's memo was reused"
    assert panel._held_run_built_for == "2026-09-02"
    assert "2026-09-02" in panel._held_run_d1_symbols


def test_todays_memo_is_taken_without_starting_a_read(panel, monkeypatch):
    from ui.panels import alert_center_panel as module

    monkeypatch.setattr(
        module,
        "_HELD_RUN_INDEX_MEMO",
        {"index": {}, "d1": {"2026-09-02": {"AMD"}}, "built_for": "2026-09-02"},
        raising=False,
    )
    monkeypatch.setattr(panel, "_held_run_day", lambda: "2026-09-02")
    started: list[int] = []
    monkeypatch.setattr(panel, "_held_run_index_worker", lambda: started.append(1))

    panel._ensure_held_run_index()

    assert started == [], "a memo built for today must not start a 90 MB read"
    assert panel._held_run_d1_symbols == {"2026-09-02": {"AMD"}}
    assert panel._held_run_built_for == "2026-09-02"


def test_day_two_reads_todays_d1_symbols_on_a_real_alert(panel, monkeypatch):
    """The consequence the blocker names: `d1_setup_present` False again."""
    import held_run_score

    monkeypatch.setattr(panel, "_held_run_day", lambda: "2026-09-02")
    panel._on_held_run_index_loaded(
        {"index": {}, "d1": {"2026-09-02": {"NVDA"}}, "built_for": "2026-09-02"}
    )

    seen: list[bool] = []
    real = held_run_score.alert_cell

    def _spy(index, **kwargs):
        seen.append(bool(kwargs.get("d1_setup_present")))
        return real(index, **kwargs)

    monkeypatch.setattr(held_run_score, "alert_cell", _spy)
    panel._held_run_index = {("ema_15", "opening_drive", "trend_up", True): {}}

    panel._attach_held_run_suffix(_alert(trade_date="2026-09-02"))

    assert seen == [True], "today's D1 symbols were not in the map"


def test_the_expiry_is_not_merely_documented(panel):
    """`built_for` has to be READ somewhere, or the docstring is the only fix."""
    source = (ROOT / "scripts" / "ui" / "panels" / "alert_center_panel.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _ensure_held_run_index(", 1)[1].split("\n    def ", 1)[0]

    assert "_held_run_day()" in body
    assert 'memo.get("built_for")' in body
    assert "_held_run_built_for" in body
