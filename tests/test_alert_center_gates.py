"""The Alert Center split keeps every moved name reachable from the panel module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

GATE_NAMES = (
    "intraday_last_bucket_end",
    "_bar_close",
    "_d1_alert_prefix",
    "is_developing_d1_alert",
    "_is_feed_noise_alert",
    "is_ready_d1_alert",
    "d1_push_event",
    "extract_alert_tier",
    "is_proven_alert",
    "is_entry_assist_alert",
    "alert_passes_min_tier",
    "alert_is_loud",
    "alert_passes_feed_gate",
    "alert_should_sound",
    "favorite_category_for_alert",
    "favorite_origin_for_alert",
    "_TIER_RE",
    "_TIER_RANK",
    "_D1_READY_PREFIXES",
    "_D1_DEVELOPING_PREFIXES",
    "_D1_PUSH_LABELS",
    "_PROVEN_RE",
)


@pytest.mark.parametrize("name", GATE_NAMES)
def test_gate_lives_in_gates_and_panel_reexports_same_object(name):
    from ui.panels import alert_center_panel
    from ui.panels.alert_center import gates

    assert getattr(alert_center_panel, name) is getattr(gates, name)


def test_gate_functions_are_defined_in_gates_module():
    from ui.panels.alert_center import gates

    for name in GATE_NAMES:
        obj = getattr(gates, name)
        if callable(obj) and hasattr(obj, "__code__"):
            assert obj.__module__ == "ui.panels.alert_center.gates", name


def test_gates_module_imports_no_qt():
    code = (
        "import sys\n"
        "import ui.panels.alert_center.gates\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'shiboken6'))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT / "scripts"),
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_clickable_item_lives_in_items_and_panel_reexports_it():
    from ui.panels import alert_center_panel
    from ui.panels.alert_center import items

    assert items._ClickableItem.__module__ == "ui.panels.alert_center.items"
    assert alert_center_panel._ClickableItem is items._ClickableItem
