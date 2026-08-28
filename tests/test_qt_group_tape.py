"""The always-on sector/industry tape.

Rebuilt 2026-08-27 (plan.md Phase 0.5 item 11). It used to render whatever
BounceBot's last RRS pass had left behind - 10 to 30 minutes stale, once 31
minutes late on a flip, and its one intraday number was a 60-minute window off
a 5-day fetch that reached across the overnight gap for the first hour. It now
renders `ui.services.group_tape_service`'s payload, and the sparkline reads
90 | 60 | 30 minutes off today's completed bars.

What is pinned here is the presentation half: the three windows, the blank a
window with too few bars must show instead of a zero, the ranking, the callout
that names the rotation AND the freshness, chips that diff instead of being
rebuilt, and the desk wiring - the tape on the new service, the RS Window tab
still on the old signal.
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
    """Shape matches `group_tape_service.build_group_tape`."""
    return {
        "30": {
            "sectors": [
                {"group_key": "technology", "etf": "XLK", "rrs": 1.8},
                {"group_key": "energy", "etf": "XLE", "rrs": -1.2},
            ],
            "industries": [
                {"group_key": "URA", "etf": "URA", "rrs": 2.1},
                {"group_key": "XBI", "etf": "XBI", "rrs": -0.6},
            ],
        },
        "60": {
            "sectors": [
                {"group_key": "technology", "etf": "XLK", "rrs": 0.9},
                {"group_key": "energy", "etf": "XLE", "rrs": -0.1},
            ],
            "industries": [{"group_key": "URA", "etf": "URA", "rrs": 1.0}],
        },
        "90": {
            "sectors": [
                {"group_key": "technology", "etf": "XLK", "rrs": -0.4},
                {"group_key": "energy", "etf": "XLE", "rrs": 0.9},
            ],
            "industries": [{"group_key": "URA", "etf": "URA", "rrs": -0.8}],
        },
    }


def _row_order(layout) -> list[str]:
    """The ETFs in a row, left to right, as the layout actually holds them."""
    from ui.widgets.group_tape_strip import GroupChip

    out = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, GroupChip):
            out.append(widget.etf)
    return out


def _payload(**extra):
    out = {
        "group_strength": _groups(),
        "as_of_text": "07:45",
        "as_of": "2026-08-27T07:45:00",
        "status": "58 of 60 groups measured on 90/60/30 min",
        "source": "yfinance",
    }
    out.update(extra)
    return out


def test_chips_carry_every_window_for_the_sparkline():
    from ui.widgets.group_tape_strip import build_chips

    chips = build_chips(_groups(), "sectors", 11)
    by_etf = {chip["etf"]: chip for chip in chips}
    assert set(by_etf) == {"XLK", "XLE"}
    # The sparkline is the whole point: 90 vs 30 in one glance.
    assert by_etf["XLK"]["spark"] == {"30": 1.8, "60": 0.9, "90": -0.4}
    assert by_etf["XLE"]["spark"] == {"30": -1.2, "60": -0.1, "90": 0.9}
    # Ranked by the 30-minute read - what is happening now.
    assert [chip["etf"] for chip in chips] == ["XLK", "XLE"]


def test_the_windows_agree_with_the_maths_module():
    """One order, stated in two places, so it is pinned in one test."""
    import group_rrs
    from ui.widgets import group_tape_strip

    assert group_tape_strip.SPARK_TIMEFRAMES == group_rrs.WINDOW_ORDER
    assert group_tape_strip.RANK_TIMEFRAME == "30"
    assert set(group_tape_strip.SPARK_TIMEFRAMES) == set(group_rrs.RRS_WINDOWS)


def test_truncation_keeps_both_ends_not_just_the_leaders():
    """A head-of-list cut would hide every short candidate."""
    from ui.widgets.group_tape_strip import build_chips

    groups = {
        "30": {
            "sectors": [{"etf": f"E{index:02d}", "rrs": 10.0 - index} for index in range(20)],
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
    # Up on the 30 while still down on the 90 = this half hour's move, not the
    # morning's.
    assert "URA" in rotating_in and "on 30" in rotating_in and "on 90" in rotating_in
    # Down on the 30 while still up on the 90 = the mirror.
    assert "XLE" in fading


def test_rotation_callout_stays_silent_rather_than_inventing_a_call():
    from ui.widgets.group_tape_strip import rotation_callout

    assert rotation_callout({}) == ("", "")
    aligned = {
        "30": {"sectors": [{"etf": "XLK", "rrs": 1.5}], "industries": []},
        "90": {"sectors": [{"etf": "XLK", "rrs": 1.2}], "industries": []},
    }
    assert rotation_callout(aligned) == ("", "")


def test_strip_renders_three_segments_and_blanks_an_unmeasured_window():
    """UNKNOWN draws nothing. A zero-height bar on the zero line would be
    indistinguishable from "exactly in line with SPY", which is a claim."""
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    chips = {chip.etf: chip for chip in strip.findChildren(GroupChip)}
    assert set(chips) == {"XLK", "XLE", "URA", "XBI"}

    assert chips["XLK"]._spark.values() == (-0.4, 0.9, 1.8), "90 | 60 | 30, in that order"
    # XBI only has a 30-minute read in the fixture: the other two are blank.
    assert chips["XBI"]._spark.values() == (None, None, -0.6)
    assert "not enough completed bars yet" in chips["XBI"]._spark.toolTip()
    assert _row_order(strip.sector_layout) == ["XLK", "XLE"]
    assert _row_order(strip.industry_layout) == ["URA", "XBI"]
    strip.close()


def test_a_chip_is_coloured_by_a_property_and_never_its_own_stylesheet():
    """The fluidity pass's rule: variants live in theme.qss, keyed on a
    dynamic property, so a re-render costs a property set and not a CSS
    parse per chip."""
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    chips = {chip.etf: chip for chip in strip.findChildren(GroupChip)}
    assert chips["XLK"].property("side") == "long"
    assert chips["XLE"].property("side") == "short"
    assert all(chip.styleSheet() == "" for chip in chips.values())

    # A flip re-labels the SAME widget rather than making a new one.
    flipped = _payload()
    for window in flipped["group_strength"].values():
        for row in window["sectors"]:
            row["rrs"] = -row["rrs"]
    strip.update_groups(flipped)
    assert chips["XLK"].property("side") == "short"
    strip.close()


def test_the_strip_diffs_its_chips_instead_of_rebuilding_them():
    """34 chips torn down and re-created every five minutes is the exact shape
    the 2026-08-21 fluidity pass measured and fixed elsewhere."""
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    first = {chip.etf: id(chip) for chip in strip.findChildren(GroupChip)}

    moved = _payload()
    moved["group_strength"]["30"]["sectors"] = [
        {"group_key": "energy", "etf": "XLE", "rrs": 3.0},
        {"group_key": "technology", "etf": "XLK", "rrs": -0.5},
    ]
    strip.update_groups(moved)
    again = {chip.etf: id(chip) for chip in strip.findChildren(GroupChip)}

    assert set(again) == set(first)
    assert again == first, "same widgets, re-labelled and re-ordered"
    assert _row_order(strip.sector_layout) == ["XLE", "XLK"], "the strongest 30 leads"
    strip.close()


def test_a_group_that_leaves_the_payload_leaves_the_strip():
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    assert "XBI" in {chip.etf for chip in strip.findChildren(GroupChip)}

    thinner = _payload()
    for window in thinner["group_strength"].values():
        window["industries"] = [row for row in window["industries"] if row["etf"] != "XBI"]
    strip.update_groups(thinner)
    _app.processEvents()
    live = {chip.etf for chip in strip.findChildren(GroupChip) if chip.parent() is not None}
    assert "XBI" not in live
    strip.close()


def test_the_callout_carries_the_as_of_and_the_status():
    """A stale or failed read has to be visible; a tape that looks current
    when its last refresh failed is worse than no tape."""
    from ui.widgets.group_tape_strip import GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    assert "as of 07:45" in strip.callout.text()
    assert "58 of 60 groups measured" in strip.callout.text()

    strip.set_status("Group tape 07:50:02: 58 of 60 groups measured · last refresh FAILED: boom")
    assert "last refresh FAILED: boom" in strip.callout.text()
    assert "as of 07:45" in strip.callout.text(), "the as-of survives a failed refresh"
    strip.close()


def test_strip_survives_an_empty_payload_and_explains_itself():
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups({})
    _app.processEvents()
    live = [c for c in strip.findChildren(GroupChip) if c.parent() is not None]
    assert live == []
    assert "90 | 60 | 30" in strip.callout.text()
    strip.close()


def test_a_chip_click_asks_for_that_etf_to_be_charted():
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    captured = []
    strip.symbolActivated.connect(captured.append)
    chip = next(c for c in strip.findChildren(GroupChip) if c.etf == "URA")
    chip.activated.emit(chip.etf)
    assert captured == ["URA"]
    strip.close()


def test_a_reused_chip_still_charts_the_group_it_now_shows():
    """A diffed widget keeps its `activated` connection; if the connection
    were re-made on every update the click would fire N times."""
    from ui.widgets.group_tape_strip import GroupChip, GroupTapeStrip

    strip = GroupTapeStrip()
    strip.update_groups(_payload())
    strip.update_groups(_payload())
    captured = []
    strip.symbolActivated.connect(captured.append)
    chip = next(c for c in strip.findChildren(GroupChip) if c.etf == "URA")
    chip.activated.emit(chip.etf)
    assert captured == ["URA"], "exactly once, not once per refresh"
    strip.close()


def test_the_tape_is_shown_on_the_desk_and_fed_by_its_own_service():
    """Trader decision 2026-08-27 hid it; the rebuild shows it again, on its
    own five-minute service rather than BounceBot's scan-cycle payload."""
    from ui.panels.trading_desk import TradingDeskPanel
    from ui.services.group_tape_service import GroupTapeService
    from ui.widgets.group_tape_strip import GroupChip

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        desk.show()
        assert isinstance(desk.group_tape_service, GroupTapeService)
        assert desk.group_tape.isVisibleTo(desk), "hidden 2026-08-27, shown again by the rebuild"

        desk.group_tape_service.tapeChanged.emit(_payload())
        for _ in range(4):
            _app.processEvents()
        assert {chip.etf for chip in desk.group_tape.findChildren(GroupChip)} == {
            "XLK",
            "XLE",
            "URA",
            "XBI",
        }
        desk.group_tape_service.statusChanged.emit("Group tape 07:50:02: all good")
        assert "all good" in desk.group_tape.callout.text()
    finally:
        desk.close()


def test_the_scan_cycle_payload_no_longer_reaches_the_tape():
    """The two answer different questions. Leaving both wired would make the
    tape flicker between a 30/60/90 read and a stale D1/H1/M5 one."""
    from ui.panels.trading_desk import TradingDeskPanel
    from ui.widgets.group_tape_strip import GroupChip

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        desk.bounce_panel.service.rrsSnapshotChanged.emit(
            {"group_strength": {"M5": {"sectors": [{"etf": "XLU", "rrs": 5.0}], "industries": []}}}
        )
        for _ in range(4):
            _app.processEvents()
        live = {c.etf for c in desk.group_tape.findChildren(GroupChip) if c.parent() is not None}
        assert live == set(), "BounceBot's payload must not paint the tape any more"
    finally:
        desk.close()


def test_the_rs_window_tab_still_receives_the_scan_cycle_payload():
    """Hard rule 7: the RS Window tab is untouched - it answers who led over
    the selected window at scan time, which the tape does not."""
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        seen = []
        desk.bounce_panel.service.rrsSnapshotChanged.connect(seen.append)
        snapshot = {"group_strength": {"M5": {"sectors": [], "industries": []}}}
        desk.bounce_panel.service.rrsSnapshotChanged.emit(snapshot)
        for _ in range(4):
            _app.processEvents()
        assert seen == [snapshot], "the signal still fires for its other consumers"
    finally:
        desk.close()


def test_the_desk_shuts_the_tape_service_down():
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    try:
        closed = []
        desk.group_tape_service.shutdown = lambda: closed.append(True)
        desk.shutdown()
        assert closed == [True], "an owned timer and worker must be released on close"
    finally:
        desk.close()
