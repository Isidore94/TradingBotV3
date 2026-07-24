"""The always-on sector/industry tape.

It renders the `group_strength` payload the Alert Center already receives.
`_group_strength_html` showed only the top-2 and bottom-2 per group type per
timeframe, so most of what the scan computed was discarded before it reached
the screen.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _groups():
    """Shape matches compute_group_strengths: group_key/etf/rrs/power_index."""
    return {
        "M5": {
            "sectors": [
                {"group_key": "tech", "etf": "XLK", "rrs": 1.8, "power_index": 0.9},
                {"group_key": "energy", "etf": "XLE", "rrs": -1.2, "power_index": -0.7},
            ],
            "industries": [
                {"group_key": "URA", "etf": "URA", "rrs": 2.1, "power_index": 1.4},
                {"group_key": "XBI", "etf": "XBI", "rrs": -0.6, "power_index": -0.3},
            ],
        },
        "H1": {
            "sectors": [
                {"group_key": "tech", "etf": "XLK", "rrs": 0.9, "power_index": 0.4},
                {"group_key": "energy", "etf": "XLE", "rrs": -0.1, "power_index": 0.0},
            ],
            "industries": [{"group_key": "URA", "etf": "URA", "rrs": 1.0, "power_index": 0.5}],
        },
        "D1": {
            "sectors": [
                {"group_key": "tech", "etf": "XLK", "rrs": -0.4, "power_index": -0.2},
                {"group_key": "energy", "etf": "XLE", "rrs": 0.9, "power_index": 0.5},
            ],
            "industries": [{"group_key": "URA", "etf": "URA", "rrs": -0.8, "power_index": -0.4}],
        },
    }


def test_chips_carry_every_timeframe_for_the_sparkline():
    from ui.widgets.group_tape_strip import build_chips

    chips = build_chips(_groups(), "sectors", 11)
    by_etf = {chip["etf"]: chip for chip in chips}
    assert set(by_etf) == {"XLK", "XLE"}
    # The sparkline is the whole point: D1 vs M5 in one glance.
    assert by_etf["XLK"]["spark"] == {"M5": 1.8, "H1": 0.9, "D1": -0.4}
    assert by_etf["XLE"]["spark"] == {"M5": -1.2, "H1": -0.1, "D1": 0.9}
    # Ranked by the M5 read - what is happening now.
    assert [chip["etf"] for chip in chips] == ["XLK", "XLE"]


def test_truncation_keeps_both_ends_not_just_the_leaders():
    """A head-of-list cut would hide every short candidate."""
    from ui.widgets.group_tape_strip import build_chips

    groups = {
        "M5": {
            "sectors": [
                {"etf": f"E{index:02d}", "rrs": 10.0 - index}
                for index in range(20)
            ],
            "industries": [],
        }
    }
    chips = build_chips(groups, "sectors", 4)
    names = [chip["etf"] for chip in chips]
    assert names[0] == "E00", "the strongest must survive"
    assert names[-1] == "E19", "the weakest must survive too"
    assert len(names) == 4


def test_rotation_callout_names_what_is_turning_and_what_is_fading():
    from ui.widgets.group_tape_strip import rotation_callout

    rotating_in, fading = rotation_callout(_groups())
    # Positive on M5 while still negative on D1 = today's move is not yet in
    # the daily read.
    assert "URA" in rotating_in
    # Negative on M5 while still positive on D1 = the mirror.
    assert "XLE" in fading


def test_rotation_callout_stays_silent_rather_than_inventing_a_call():
    from ui.widgets.group_tape_strip import rotation_callout

    assert rotation_callout({}) == ("", "")
    aligned = {
        "M5": {"sectors": [{"etf": "XLK", "rrs": 1.5}], "industries": []},
        "D1": {"sectors": [{"etf": "XLK", "rrs": 1.2}], "industries": []},
    }
    assert rotation_callout(aligned) == ("", "")


def test_strip_renders_chips_and_survives_an_empty_payload():
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups({"group_strength": _groups()})
    chips = strip.findChildren(GroupChip)
    assert {chip.etf for chip in chips} == {"XLK", "XLE", "URA", "XBI"}

    # Bot off / no cached ETF bars: explain, never render a blank bar.
    strip.update_groups({})
    assert strip.findChildren(GroupChip) == []
    assert "Waiting for BounceBot" in strip.callout.text()
    strip.close()


def test_a_chip_click_asks_for_that_etf_to_be_charted():
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups({"group_strength": _groups()})
    captured = []
    strip.symbolActivated.connect(captured.append)
    chip = next(c for c in strip.findChildren(GroupChip) if c.etf == "URA")
    chip.activated.emit(chip.etf)
    assert captured == ["URA"]
    strip.close()


def test_the_tape_is_wired_to_the_existing_rrs_payload_on_the_desk():
    """No new service: the desk feeds the tape off rrsSnapshotChanged."""
    from ui.panels.trading_desk import TradingDeskPanel
    from ui.widgets.group_tape_strip import GroupChip

    desk = TradingDeskPanel(workspace_mode="workspace")
    desk.bounce_panel.service.rrsSnapshotChanged.emit({"group_strength": _groups()})
    for _ in range(4):
        _app.processEvents()
    assert {chip.etf for chip in desk.group_tape.findChildren(GroupChip)} == {
        "XLK",
        "XLE",
        "URA",
        "XBI",
    }
    desk.close()
