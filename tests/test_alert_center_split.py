"""A6 steps 2-3: the wall / any-bounce / H1 / pullback clusters are mixins `AlertCenterPanel` inherits.

Pure moves: each name lives on its mixin, not on the panel, and resolves to the
same object through the panel. Every mixin module is in the test patch helper's
list, so a test that freezes the Alert Center's `datetime` reaches moved code.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SPLIT = {
    "ui.panels.alert_center.wall": (
        "WallGateMixin",
        (
            "WALL_AUTO_ARM_CAP",
            "WALL_SMA_FOLLOW_UP_KINDS",
            "WALL_PULLBACK_TIMEFRAMES",
            "WALL_TRENDLINE_PROJECTION",
            "_wall_trendline_records",
            "_wall_trendlines_for",
            "wall_verdict",
            "wall_state",
            "_log_wall_decision",
            "_live_wall_symbols",
            "_save_wall_arms",
            "_is_wall_pullback_watch",
            "_ensure_wall_follow_up",
            "_wall_cap_reached",
            "_remember_wall_arm",
            "_ensure_wall_d1_events",
            "_ensure_wall_pullback",
            "_decline_wall_d1_event",
        ),
    ),
}


@pytest.mark.parametrize("module_name", sorted(SPLIT))
def test_cluster_is_a_mixin_the_panel_inherits(module_name):
    from ui.panels.alert_center_panel import AlertCenterPanel

    class_name, names = SPLIT[module_name]
    mixin = getattr(importlib.import_module(module_name), class_name)
    assert mixin in AlertCenterPanel.__mro__
    for name in names:
        assert name in mixin.__dict__, name
        assert name not in AlertCenterPanel.__dict__, name
        assert inspect.getattr_static(AlertCenterPanel, name) is mixin.__dict__[name], name


@pytest.mark.parametrize("module_name", sorted(SPLIT))
def test_the_patch_helper_reaches_every_mixin_module(module_name):
    from alert_center_support import MIXIN_MODULES

    assert module_name in MIXIN_MODULES
