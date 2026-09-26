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
    "ui.panels.alert_center.any_bounce": (
        "AnyBounceWatchMixin",
        (
            "any_bounce_armed_for",
            "_save_any_bounce_watches",
            "_toggle_any_bounce_watch",
            "arm_any_bounce_watch",
            "disarm_any_bounce_watch",
            "_zone_arms_unknown",
            "_any_bounce_levels_for",
            "_poll_any_bounce_watches",
        ),
    ),
    "ui.panels.alert_center.h1": (
        "H1RetesterMixin",
        (
            "H1_WATCH_M5_SESSIONS",
            "_h1_history_cache",
            "_h1_bars_for_watch",
            "_h1_warmup_bars",
            "_h1_watch_note",
            "_h1_refresh_failed",
            "_h1_warmup_counts",
            "_h1_warmup_note",
            "_h1_watches_due",
        ),
    ),
    "ui.panels.alert_center.pullback": (
        "PullbackWatchMixin",
        (
            "PULLBACK_TIMEFRAMES",
            "PULLBACK_SMA_TRIGGERS",
            "_intraday_history_cache",
            "_pullback_warmup_bars",
            "_pullback_triggers",
            "_pullback_timeframe_scope",
            "_pullback_uses_h1",
            "_pullback_sma_timeframes",
            "_pullback_bars_for_watch",
            "_pullback_timeframe_note",
            "_armed_watch_note",
            "_poll_pullback_watches",
            "_record_pullback_fires",
            "_is_auto_pullback_watch",
            "_pullback_judged_marks",
            "_pullback_request_marks",
            "_request_pullback_cache",
            "_pullback_cache_token",
            "_pullback_cache_snapshot",
            "_pullback_due",
            "_mark_pullback_judged",
            "pullback_fire_key",
            "_pullback_should_push",
            "_dispatch_pullback_sma_evaluation",
            "_pullback_d1_bars",
            "_pullback_episode_states",
            "_run_pullback_sma_evaluation",
            "_evaluate_one_pullback_job",
            "_pullback_mark_covers",
            "_on_pullback_fires",
            "_active_d1_claims",
            "_auto_pullback_sources",
            "_sweep_auto_pullback_watches",
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
