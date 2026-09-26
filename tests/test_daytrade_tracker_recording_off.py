"""S10a: the Daytrade Tracker says a family is no longer recorded."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_the_off_list_matches_what_the_sweep_skips():
    from bounce_bot_lib.legacy import H1_COLOR_TYPES_OFF
    from ui.panels.daytrade_tracker_panel import RECORDING_OFF_SINCE

    assert set(RECORDING_OFF_SINCE) == set(H1_COLOR_TYPES_OFF)


def test_the_blue_after_red_family_row_reads_off_since():
    from ui.panels.daytrade_tracker_panel import apply_champion_tier, recording_off_text

    state = {
        "segments": {
            "bounce_type": {
                "long|h1_blue_after_red": {"muted": True},
                "long|h1_ema10_bounce": {"proven": True},
            }
        }
    }
    rows = apply_champion_tier(
        [
            {"dimension": "bounce_type", "direction": "long", "segment": "h1_blue_after_red"},
            {"dimension": "bounce_type", "direction": "short", "segment": "h1_blue_after_red"},
            {"dimension": "bounce_type", "direction": "long", "segment": "h1_ema10_bounce"},
            {"dimension": "time_bucket", "direction": "long", "segment": "h1_blue_after_red"},
        ],
        state,
    )
    assert [row["champion_tier"] for row in rows] == [
        "off since 2026-09-26",
        "off since 2026-09-26",
        "PROVEN",
        "",
    ]
    assert recording_off_text("bounce_type", "h1_blue_after_red") == "off since 2026-09-26"
    assert recording_off_text("bounce_type", "h1_green_to_yellow") == ""
