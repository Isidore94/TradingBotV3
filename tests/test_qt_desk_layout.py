"""Trading Desk layout contract: the chart column leads and columns fit.

These lock in the two defects that made the desk unreadable. The charts the
trader works from measured 68px tall at 1640x980 and 105px at 2560x1440 - and
their share of the desk SHRANK as the window grew - because the Alert Center's
declared 5:2 split measured inverted. Separately, the setups table needed
1513px of columns in a 753px viewport, hiding half of itself (including the
sector/industry RS-RW readings) behind a horizontal scrollbar.
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


def _desk(width=1640, height=980):
    from ui.models.setup import SetupRow
    from ui.panels.trading_desk import TradingDeskPanel

    desk = TradingDeskPanel(workspace_mode="workspace")
    desk.resize(width, height)
    desk.show()
    desk.master_panel.set_rows(
        [
            SetupRow(
                symbol=f"SYM{index}",
                side="LONG",
                score=70.0 + index,
                bucket="favorite_setup",
                sector="Information Technology",
                industry="Semiconductors & Semiconductor Equipment",
                key_level="AVWAPE 101.25",
                expected_r=1.8,
            )
            for index in range(20)
        ]
    )
    for _ in range(6):
        _app.processEvents()
    return desk


def test_chart_column_leads_and_widens_with_the_desk():
    """The chart column takes the larger share, and grows on a bigger desk.

    The old 1:2 stretch sent every extra pixel 2:1 to the setups table, so the
    charts got relatively smaller the bigger the monitor.
    """
    narrow = _desk(1640, 980)
    narrow_sizes = narrow.desk_splitter.sizes()
    narrow_share = narrow_sizes[0] / sum(narrow_sizes)
    narrow.close()

    wide = _desk(2560, 1440)
    wide_sizes = wide.desk_splitter.sizes()
    wide_share = wide_sizes[0] / sum(wide_sizes)
    wide.close()

    assert narrow_share > 0.5, f"chart column should lead, got {narrow_share:.0%}"
    assert wide_share > narrow_share, (
        f"chart share must grow with the desk: {narrow_share:.0%} -> {wide_share:.0%}"
    )


def test_chart_pane_outgrows_the_alert_feed_in_its_column():
    """The declared 5:2 must actually measure that way, not invert."""
    desk = _desk()
    sizes = desk.alert_center.splitter.sizes()
    desk.close()
    assert sizes[0] > sizes[1], (
        f"chart pane must beat the feed tabs, measured {sizes[0]} vs {sizes[1]}"
    )


def test_compact_profile_never_hides_columns_behind_a_scrollbar():
    """Whatever the width, the compact profile fits its viewport."""
    for width in (1400, 1640, 2560):
        desk = _desk(width, 980)
        table = desk.master_panel.table
        model = desk.master_panel.model
        visible = sum(
            table.columnWidth(column)
            for column in range(model.columnCount())
            if not table.isColumnHidden(column)
        )
        viewport = table.viewport().width()
        desk.close()
        assert visible <= viewport, (
            f"at {width}px the table needs {visible}px in a {viewport}px viewport"
        )


def test_compact_profile_keeps_the_group_strength_readings():
    """Industry RS/RW is what the trader looks at; it must not be a casualty."""
    desk = _desk(1640, 980)
    table = desk.master_panel.table
    keys = [key for key, _label in desk.master_panel.model.COLUMNS]
    hidden = {
        key
        for column, key in enumerate(keys)
        if table.isColumnHidden(column)
    }
    desk.close()
    assert "d1_vs_industry" not in hidden
    assert "expected_r" not in hidden


def test_f9_expands_setups_to_the_full_desk_and_back():
    desk = _desk()
    before = desk.desk_splitter.sizes()

    assert desk.toggle_setups_expanded() is True
    for _ in range(4):
        _app.processEvents()
    assert not desk.alert_center.isVisibleTo(desk)
    assert desk.master_panel._column_profile == "full"

    assert desk.toggle_setups_expanded() is False
    for _ in range(4):
        _app.processEvents()
    assert desk.alert_center.isVisibleTo(desk)
    assert desk.desk_splitter.sizes() == before
    assert desk.master_panel._column_profile == "compact"
    desk.close()


def test_set_mode_is_idempotent_and_survives_a_settings_save():
    """A theme change must not reset the trader's dragged split.

    The guard has to check that the layout was actually built, not just the
    mode string: __init__ assigns workspace_mode before the first set_mode
    call, so guarding on the mode alone leaves the desk empty.
    """
    desk = _desk()
    desk.desk_splitter.setSizes([500, 918])
    for _ in range(4):
        _app.processEvents()
    dragged = desk.desk_splitter.sizes()

    desk.set_mode("workspace")  # what a settings save triggers
    for _ in range(4):
        _app.processEvents()

    assert desk.desk_splitter is not None, "the desk must still be built"
    assert desk.desk_splitter.sizes() == dragged
    desk.close()


def test_tape_host_survives_a_workspace_tabs_round_trip():
    """Phase 3's mount point must not be destroyed by a mode switch.

    _clear_layout deletes whatever the layout still owns, and only the panels
    named in _detach_mode_panels are rescued.
    """
    from PySide6.QtWidgets import QWidget

    desk = _desk()
    host = desk.tape_host
    probe = QWidget()
    host.layout().addWidget(probe)

    desk.set_mode("tabs")
    desk.set_mode("workspace")
    for _ in range(4):
        _app.processEvents()

    assert desk.tape_host is host
    assert host.layout().indexOf(probe) >= 0
    desk.close()


def test_d1_focus_is_a_badged_tab_counting_only_scan_events():
    """Pins are the trader's own doing and must not inflate the badge."""
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel()
    panel.tabs.setCurrentIndex(0)  # not looking at D1 Focus

    panel._add_d1_alert(
        BounceAlert(
            time_text="09:31:00",
            symbol="NVDA",
            side="LONG",
            raw_text="MASTER_AVWAP_D1_BUCKET_UPGRADE: NVDA",
        )
    )
    assert "(1)" in panel.tabs.tabText(panel._d1_tab_index)

    panel._add_d1_alert(
        BounceAlert(time_text="09:32:00", symbol="AMD", side="LONG", tag="d1_focus_pin")
    )
    assert "(1)" in panel.tabs.tabText(panel._d1_tab_index), "a pin is not an event"

    panel.tabs.setCurrentIndex(panel._d1_tab_index)
    assert panel.tabs.tabText(panel._d1_tab_index) == "D1 Focus"
    panel.close()
