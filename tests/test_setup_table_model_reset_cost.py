"""The Setup Tracker model resets only when its rows actually change.

2026-09-30: `_poll_longs_gate` and every Best-swing re-sort called `set_rows`
with the same rows, and each call was a full model reset: every row filtered
and sorted again through Qt->Python hops, the whole table repainted. With a
pure-Python worker busy, one reset blocked the desk for 3-13 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")

from ui.models.setup_table_model import SetupFilterProxyModel, SetupTableModel  # noqa: E402
from ui.models.setup import SetupRow  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _row(symbol: str, score: float) -> SetupRow:
    return SetupRow(symbol=symbol, side="LONG", score=score, bucket="favorite_setup")


def _count_resets(model) -> list[str]:
    seen: list[str] = []
    model.modelReset.connect(lambda: seen.append("reset"))
    return seen


def test_the_same_rows_in_the_same_order_do_not_reset(qapp):
    model = SetupTableModel()
    rows = [_row("NVDA", 90.0), _row("AMD", 80.0)]
    model.set_rows(rows)
    resets = _count_resets(model)
    model.set_rows(list(rows))
    assert resets == []
    assert model.rows() == rows


def test_a_reorder_or_a_new_row_still_resets(qapp):
    model = SetupTableModel()
    rows = [_row("NVDA", 90.0), _row("AMD", 80.0)]
    model.set_rows(rows)
    resets = _count_resets(model)
    model.set_rows(list(reversed(rows)))
    model.set_rows([_row("NVDA", 90.0), _row("AMD", 80.0)])  # equal values, new objects
    assert resets == ["reset", "reset"]


def test_the_proxy_filters_without_asking_the_model_for_an_index(qapp, monkeypatch):
    model = SetupTableModel()
    model.set_rows([_row("NVDA", 90.0), _row("AMD", 40.0)])
    proxy = SetupFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.set_filters(min_score=50.0)
    assert proxy.rowCount() == 1

    def boom(*args, **kwargs):
        raise AssertionError("filterAcceptsRow went through model.index()")

    monkeypatch.setattr(model, "index", boom)
    proxy.set_filters(min_score=0.0)
    assert proxy.rowCount() == 2
