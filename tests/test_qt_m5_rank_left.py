"""The M5 bar puts the alert's already-earned grade at the left edge."""

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
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.models.bounce import BounceAlert  # noqa: E402
from ui.widgets.m5_alert_bar import M5AlertBar  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    return QApplication.instance() or QApplication([])


def _alert(*, raw_text: str, at: str = "09:31:00") -> BounceAlert:
    """A real M5 alert whose grade exists only in its emitted raw text."""
    return BounceAlert(
        time_text=at,
        symbol="NVDA",
        side="LONG",
        trigger="new HOD",
        timeframe="M5",
        tag="green",
        raw_text=raw_text,
    )


@pytest.mark.parametrize("grade", ["S", "A", "B", "C", "D"])
def test_the_existing_tier_is_the_first_m5_row_text(grade):
    bar = M5AlertBar()
    bar.post(_alert(raw_text=f"[{grade}-TIER] NVDA: New HOD"))

    assert bar.list.item(0).text() == f"[{grade}]  09:31  ▲ NVDA  new HOD"


def test_proven_is_the_first_m5_grade_even_when_a_tier_is_also_present():
    bar = M5AlertBar()
    # The producer supplies both labels.  PROVEN is the top existing class.
    bar.post(_alert(raw_text="[A-TIER] PROVEN NVDA: New HOD"))

    assert bar.list.item(0).text() == "[PROVEN]  09:31  ▲ NVDA  new HOD"


def test_an_unrecognised_or_absent_grade_stays_explicitly_ungraded():
    bar = M5AlertBar()
    bar.post(_alert(raw_text="[Z-TIER] NVDA: New HOD"))

    assert bar.list.item(0).text() == "[—]  09:31  ▲ NVDA  new HOD"


def test_a_folded_repeat_replaces_its_left_grade_with_the_newest_alert_grade():
    bar = M5AlertBar()
    bar.post(_alert(raw_text="[C-TIER] NVDA: New HOD", at="09:30:00"))
    bar.post(_alert(raw_text="[A-TIER] NVDA: New HOD", at="09:35:00"))

    assert bar.count() == 1
    assert bar.list.item(0).text() == "[A]  09:35  ▲ NVDA  new HOD  ×2"


def test_the_left_grade_and_time_fit_in_the_visible_m5_row_prefix():
    bar = M5AlertBar()
    bar.resize(240, 180)
    bar.post(_alert(raw_text="[A-TIER] PROVEN NVDA: New HOD"))
    bar.show()
    QApplication.processEvents()

    visible_width = bar.list.viewport().width()
    prefix_width = bar.list.fontMetrics().horizontalAdvance("[PROVEN]  09:31")
    assert prefix_width <= visible_width
    assert bar.list.visualItemRect(bar.list.item(0)).left() == 0
    bar.close()
