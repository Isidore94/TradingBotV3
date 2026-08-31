from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.panels import desk_layout
from ui.panels.alert_center_panel import AlertCenterPanel
from ui.panels.bounce_panel import BouncePanel
from ui.panels.focus_picks_panel import FocusPicksPanel
from ui.panels.industry_panel import IndustryPanel
from ui.panels.master_avwap_panel import MasterAvwapPanel
from ui.widgets.m5_alert_bar import M5AlertBar
from ui.widgets.swing_favorites_bar import SwingFavoritesBar
from ui.panels.rs_window_panel import RsWindowPanel
from ui.panels.theta_panel import ThetaPanel
from ui.panels.watchlists_panel import WatchlistsPanel
from ui.services.focus_service import FocusService
from ui.services.price_alert_service import PriceAlertService
from ui.services.group_tape_service import GroupTapeService
from ui.services.swing_favorites_service import SwingFavoritesService
from ui.widgets.group_tape_strip import GroupTapeStrip
from ui.widgets.setups_toggle_button import SetupsToggleButton

# v3 (2026-08-27): the M5 alert bar moved to the LEFT of the chart column, so a
# v2 split saved with the bar in the middle must not be replayed onto it.
DESK_SPLIT_KEY = "qt_desk_split_sizes_v3"

# The bar/strip split inside the M5 alerts column. Its own key, so dragging it
# never disturbs the three-column desk split above.
M5_COLUMN_SPLIT_KEY = "qt_m5_column_split_sizes_v1"
#: Opening weights for that split: the alert list leads, the strip takes the
#: bottom quarter, and the trader's drag replaces both from then on.
M5_COLUMN_WEIGHTS = (3, 1)


class TradingDeskPanel(QWidget):
    statusChanged = Signal(str)
    rowsChanged = Signal(int, int, int)
    connectionChanged = Signal(str)

    def __init__(
        self,
        workspace_mode: str = "workspace",
        parent=None,
        *,
        price_alert_engine_enabled: bool = True,
        price_alert_read_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self.workspace_mode = workspace_mode
        self.focus_service = FocusService()
        self.price_alert_service = PriceAlertService(
            self, engine_enabled=price_alert_engine_enabled
        )
        self.master_panel = MasterAvwapPanel(self.focus_service)
        self.theta_panel = ThetaPanel()
        self.watchlists_panel = WatchlistsPanel()
        self.industry_panel = IndustryPanel()
        self.focus_picks_panel = FocusPicksPanel(
            self.focus_service,
            self.price_alert_service,
            price_alert_read_only=price_alert_read_only,
        )
        self.bounce_panel = BouncePanel(self.focus_service)
        self.rs_window_panel = RsWindowPanel(self.bounce_panel.service)
        self.master_workspace = MasterAvwapWorkspace(
            self.master_panel,
            self.theta_panel,
            self.watchlists_panel,
            self.industry_panel,
            rs_window_panel=self.rs_window_panel,
        )
        self.alert_center = AlertCenterPanel(self.focus_service)
        self.alert_center.attach_service(self.bounce_panel.service)
        # A5: the Alert Center arms phone price alerts off painted D1 levels.
        # It borrows the desk's ONE PriceAlertService rather than building a
        # second one, so price_alerts.json keeps a single writer and a single
        # poller (plan.md sec 5). Injected rather than constructed there on
        # purpose: the panel uses the store, the desk owns it.
        self.alert_center.price_alert_service = self.price_alert_service
        self.watchlists_panel.set_bounce_service(self.bounce_panel.service)
        self.master_panel.set_bounce_service(self.bounce_panel.service)
        self.industry_panel.set_bounce_service(self.bounce_panel.service)
        # Master AVWAP setups chart popups carry the Alert Center's chart-only
        # actions (D1 Focus pin + New HOD/LOD/VWAP-bounce watch arming).
        self.master_panel.set_chart_watch_host(self.alert_center)
        # Every chart in the desk arms, not just the setups table's. The
        # Industry Board and RS Window charts sit directly on the trader's
        # sector/industry RS-RW path and used to open read-only.
        self.industry_panel.set_chart_watch_host(self.alert_center)
        self.rs_window_panel.set_chart_watch_host(self.alert_center)
        self.watchlists_panel.set_chart_watch_host(self.alert_center)
        # In workspace mode the Alert Center's embedded plan pane is off; a
        # clicked alert opens in the setups workspace's detail pane instead,
        # so the setup is described in exactly one place.
        self.alert_center.setupRequested.connect(self._show_setup_in_workspace)
        self._mode_widget: QWidget | None = None

        self.master_panel.statusChanged.connect(self.statusChanged)
        self.master_panel.rowsChanged.connect(self.rowsChanged)
        self.theta_panel.statusChanged.connect(self.statusChanged)
        self.watchlists_panel.statusChanged.connect(self.statusChanged)
        self.industry_panel.statusChanged.connect(self.statusChanged)
        self.rs_window_panel.statusChanged.connect(self.statusChanged)
        self.focus_picks_panel.statusChanged.connect(self.statusChanged)
        # Trader rule 2026-08-19: flag the Focus picks that are beyond their
        # previous-day extreme. The Alert Center's 60-second D1 poll already
        # measures exactly that for every Focus name, so the board asks it
        # rather than measuring again - no new timer, no new market data.
        self.focus_picks_panel.set_mover_source(self.alert_center.mover_state)
        self.alert_center.focusBreakStatesChanged.connect(
            self.focus_picks_panel.refresh_mover_flags
        )
        self.bounce_panel.statusChanged.connect(self.statusChanged)
        self.alert_center.statusChanged.connect(self.statusChanged)
        # Trader, 2026-08-27: intraday alerts are a list beside the chart, not
        # charts in the waiting list. The Alert Center posts them; the bar
        # lists them newest first; a click charts one through the same path
        # as a feed-row click. Day-scoped with the queue.
        self.m5_alert_bar = M5AlertBar()
        self.alert_center.m5AlertPosted.connect(self.m5_alert_bar.post)
        self.alert_center.m5AlertsDayRolled.connect(self.m5_alert_bar.clear_all)
        self.m5_alert_bar.alertActivated.connect(self.alert_center.chart_alert)

        # Trader, 2026-08-31: "at the end of the day I have a list of my top
        # swing targets... put it at the very bottom of the M5 alerts tab, the
        # tab is so long and I never use all of it." The M5 alerts surface is a
        # TAB in tabs mode and the tall left COLUMN in workspace mode - the
        # trader runs workspace - so the bar and the strip share one host that
        # both modes mount, and the strip is always the bottom of it. The bar
        # keeps every pixel it wants: it takes the stretch, the strip takes
        # none. Nothing here touches the bar or any alert routing.
        self.swing_favorites_bar = SwingFavoritesBar()
        self.swing_favorites_service = SwingFavoritesService(self.focus_service, parent=self)
        self.swing_favorites_bar.addRequested.connect(self._add_swing_favorites)
        self.swing_favorites_bar.removeRequested.connect(self._remove_swing_favorite)
        self.swing_favorites_service.favoritesChanged.connect(self._refresh_swing_favorites)
        self.swing_favorites_service.takenChanged.connect(self.swing_favorites_bar.set_taken)
        self.swing_favorites_service.statusChanged.connect(self.swing_favorites_bar.set_status)
        self.swing_favorites_service.statusChanged.connect(self.statusChanged)
        # A SPLITTER, not a fixed stack (trader, 2026-08-31: "the tab needs to
        # be resizable relative to the M5 alerts tab, I should be able to drag
        # it up to see more"). Its own settings key, so this drag and the desk's
        # three-column drag never overwrite each other. Neither pane collapses:
        # a strip dragged to nothing is one the trader cannot find again.
        self.m5_column = QSplitter(Qt.Orientation.Vertical)
        self.m5_column.addWidget(self.m5_alert_bar)
        self.m5_column.addWidget(self.swing_favorites_bar)
        self.m5_column.setChildrenCollapsible(False)
        self.m5_column.setStretchFactor(0, 1)
        self.m5_column.setStretchFactor(1, 0)
        desk_layout.apply_saved_sizes(
            self.m5_column, M5_COLUMN_SPLIT_KEY, M5_COLUMN_WEIGHTS
        )
        # Held at the preset until the trader drags it, then saved. Built once
        # here rather than per mode, so the drag survives a workspace<->tabs
        # switch the same way the column itself does.
        desk_layout.track_preset(
            self, self.m5_column, M5_COLUMN_SPLIT_KEY, lambda _extent: M5_COLUMN_WEIGHTS
        )
        desk_layout.persist_sizes(self, self.m5_column, M5_COLUMN_SPLIT_KEY)
        self._refresh_swing_favorites()
        # A day roll starts a new session, so the strip re-derives from the
        # store and comes back empty. Read-only: the rows themselves stay.
        self.alert_center.m5AlertsDayRolled.connect(self._refresh_swing_favorites)
        self.bounce_panel.service.connectionChanged.connect(self.connectionChanged)
        self.bounce_panel.service.alertReceived.connect(self.focus_picks_panel.record_bounce_alert)
        self.bounce_panel.service.rrsSnapshotChanged.connect(self.focus_picks_panel.record_rrs_snapshot)

        self.center_container = QWidget()
        self.center_layout = QVBoxLayout(self.center_container)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_layout.setSpacing(0)

        # Host for the group RS/RW tape. Empty (and zero-height) until the tape
        # is installed, but created here so it is a stable mount point that
        # survives mode switches.
        self.tape_host = QWidget()
        tape_layout = QHBoxLayout(self.tape_host)
        tape_layout.setContentsMargins(0, 0, 0, 0)
        tape_layout.setSpacing(6)
        # Sector/industry strength, always visible across the desk. Since the
        # 2026-08-27 rebuild (plan.md Phase 0.5 item 11) it is fed by its OWN
        # service - one batched yfinance read of today's completed bars every
        # five minutes, zero IB traffic - and no longer by BounceBot's
        # rrsSnapshotChanged, which only moved when a scan cycle's RRS pass
        # finished: 10-30 minutes apart, once 31 minutes late on a flip, and
        # its one intraday number reached across the overnight gap for the
        # first hour. The trader had it hidden between that finding and this
        # rebuild.
        #
        # The RS Window tab still reads rrsSnapshotChanged, deliberately: it
        # answers a different question (who led over the selected window at
        # scan time), so both wirings coexist and neither is a copy of the
        # other.
        self.group_tape = GroupTapeStrip()
        self.group_tape.symbolActivated.connect(self.alert_center.chart_symbol)
        self.group_tape_service = GroupTapeService(self)
        self.group_tape_service.tapeChanged.connect(self.group_tape.update_groups)
        self.group_tape_service.statusChanged.connect(self.group_tape.set_status)
        self.group_tape.set_status(self.group_tape_service.status_text())
        tape_layout.addWidget(self.group_tape, 1)
        # The setups column opens hidden, so the way back has to live somewhere
        # that is always on screen and independent of it. The tape row is the
        # only full-width strip that survives a workspace<->tabs switch.
        self.setups_toggle = SetupsToggleButton(visible=False)
        self.setups_toggle.setupsVisibleChanged.connect(self.set_setups_visible)
        tape_layout.addWidget(self.setups_toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        self.desk_splitter: QSplitter | None = None
        self._setups_expanded = False
        # The desk opens with the setups column hidden: it earns its width only
        # a few hours into the session, and the charts want it before then.
        self._setups_visible = False
        self._setups_restore_sizes: list[int] | None = None

        self._build_layout()
        self.set_mode(workspace_mode)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.center_container)

    def set_mode(self, workspace_mode: str) -> None:
        workspace_mode = workspace_mode if workspace_mode in {"workspace", "tabs"} else "workspace"
        # Any settings save calls this (app.py _apply_state_changes), so
        # changing the theme used to tear down and rebuild the splitter -
        # discarding whatever the trader had dragged. Rebuild only on a real
        # mode change. The _mode_widget check matters: __init__ assigns
        # self.workspace_mode BEFORE the first set_mode call, so guarding on
        # the mode alone would return early and leave the desk empty.
        if self._mode_widget is not None and workspace_mode == self.workspace_mode:
            return
        self.workspace_mode = workspace_mode
        self._detach_mode_panels()
        _clear_layout(self.center_layout)
        if self.workspace_mode == "tabs":
            self.alert_center.set_embedded_detail_enabled(True)
            # Hiding the setups is a workspace-mode idea: as its own tab the
            # column never competes with the charts for width, and a tab that
            # refused to show itself would just look broken.
            self.setups_toggle.setVisible(False)
            self.master_workspace.setVisible(True)
            tabs = QTabWidget()
            tabs.addTab(self.master_workspace, "Master AVWAP")
            tabs.addTab(self.alert_center, "Alert Center")
            tabs.addTab(self.m5_column, "M5 alerts")
            tabs.addTab(self.bounce_panel, "BounceBot")
            self._mode_widget = tabs
            self.center_layout.addWidget(tabs)
            return

        # Workspace mode: the Alert Center (chart column) leads on the left,
        # the setups workspace takes the right, and the BounceBot strip runs
        # full width along the bottom. Alert clicks show their plan in the
        # workspace's detail pane, not in a second embedded pane.
        self.alert_center.set_embedded_detail_enabled(False)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # The M5 alert bar is the LEFT column (trader, 2026-08-27, second
        # pass: "move it to the left of the visual chart"). It takes no
        # stretch: extra width goes to the chart column first, then the
        # setups.
        splitter.addWidget(self.m5_column)
        splitter.addWidget(self.alert_center)
        splitter.addWidget(self.master_workspace)
        # The chart column leads. The old 1:2 stretch meant every pixel
        # added to the window went 2:1 to the setups table, so the charts got
        # relatively SMALLER on a bigger monitor.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setChildrenCollapsible(False)
        # Both columns aggregate large minimumSizeHints from their children
        # (the setups workspace alone hinted 1372px wide). Their sum exceeded
        # the desk, so QSplitter had no freedom and ignored setSizes entirely -
        # the split was decided by size hints, not by the preset. An explicit
        # minimum takes precedence over minimumSizeHint and hands the split
        # back to us; both columns stay usable well below these floors.
        self._apply_column_floors()
        self.desk_splitter = splitter

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        # Mount points for the group RS/RW tape (above) and the BounceBot strip
        # (below). They are held as attributes and rescued in
        # _detach_mode_panels so a workspace<->tabs switch cannot destroy them.
        body_layout.addWidget(self.tape_host)
        body_layout.addWidget(splitter, 1)
        body_layout.addWidget(self.bounce_panel)

        self._mode_widget = body
        self.center_layout.addWidget(body)
        self._apply_desk_split()
        # Applied after the split so the saved drag is what gets restored when
        # the column is shown again, not a width measured while it was hidden.
        self.setups_toggle.setVisible(True)
        self.master_workspace.setVisible(self._setups_visible)

    # ------------------------------------------------------- swing picks
    def _add_swing_favorites(self, text: str, side: str) -> None:
        """Place the typed or pasted names on today's list and in swing Focus."""
        added = self.swing_favorites_service.add(text, side)
        if added:
            self.swing_favorites_bar.set_status(
                f"Added {', '.join(added)} to today's swing picks."
            )
        else:
            self.swing_favorites_bar.set_status("Nothing to add.")

    def _remove_swing_favorite(self, symbol: str, side: str) -> None:
        if self.swing_favorites_service.remove(symbol, side):
            self.swing_favorites_bar.set_status(f"Dropped {symbol}.")

    def _refresh_swing_favorites(self) -> None:
        """Show the current session's list and re-ask the journal about it.

        Cheap by construction: the list is a replay of one session's rows in a
        small JSONL, and the journal join runs on its own thread.
        """
        self.swing_favorites_bar.set_favorites(self.swing_favorites_service.favorites())
        self.swing_favorites_service.refresh_taken()

    def _show_setup_in_workspace(self, payload: dict) -> None:
        self.master_workspace.show_setups()
        self.master_panel.detail_view.show_setup(**payload)

    def shutdown(self) -> None:
        """Release live resources (IB connection, worker threads) on app close."""
        components = []
        price_alert_service = getattr(self, "price_alert_service", None)
        if price_alert_service is not None:
            components.append(("price alerts", price_alert_service.shutdown))
        swing_favorites_service = getattr(self, "swing_favorites_service", None)
        if swing_favorites_service is not None:
            components.append(("swing favorites", swing_favorites_service.shutdown))
        components.extend((
            ("BounceBot", self.bounce_panel.on_close),
            ("industry board", self.industry_panel.shutdown),
            ("master scan service", self.master_panel.scan_service.shutdown),
        ))
        # Resolved the way `price_alert_service` above is, and for the same
        # reason: a desk whose __init__ died partway must still hand every
        # service it DID build its bounded cleanup. Naming the attribute
        # inline would make a missing one raise while the tuple is being
        # built - before the loop below runs at all - so nothing would be
        # released rather than one thing.
        group_tape_service = getattr(self, "group_tape_service", None)
        if group_tape_service is not None:
            components.append(("group tape", group_tape_service.shutdown))
        for label, close in components:
            try:
                close()
            except Exception:
                # App close is a best-effort fan-out: one service exposing a
                # real shutdown bug must not prevent the remaining owned
                # processes and threads from receiving their bounded cleanup.
                logging.exception("%s shutdown failed; continuing app cleanup.", label)

    def _detach_mode_panels(self) -> None:
        # _clear_layout deletes whatever it still owns, so every long-lived
        # child must be reparented out first - including the tape host, which
        # a later mode switch would otherwise destroy under the tape.
        rescued = (
            self.master_workspace,
            self.alert_center,
            self.m5_column,
            self.bounce_panel,
            self.tape_host,
        )
        if isinstance(self._mode_widget, QTabWidget):
            for panel in rescued:
                index = self._mode_widget.indexOf(panel)
                if index >= 0:
                    self._mode_widget.removeTab(index)

        for panel in rescued:
            try:
                panel.setParent(None)
            except RuntimeError:
                pass
        self.desk_splitter = None

    def _apply_column_floors(self) -> None:
        """Explicit minimum widths for the two desk columns, at the UI scale.

        Both columns aggregate large minimumSizeHints from their children (the
        setups workspace alone hinted 1372px wide). Their sum exceeded the desk,
        so QSplitter had no freedom and ignored setSizes entirely - the split
        was decided by size hints, not by the preset. An explicit minimum takes
        precedence over minimumSizeHint and hands the split back to us; both
        columns stay usable well below these floors. They scale because a
        laptop-sized desk cannot afford desktop-sized floors.
        """
        self.alert_center.setMinimumWidth(theme.px(360))
        # Wide enough for "07:09  ▲ SYMBOL  VWAP reclaim" and the two buttons.
        self.m5_column.setMinimumWidth(theme.px(150))
        self.master_workspace.setMinimumWidth(theme.px(420))

    def apply_scaled_metrics(self) -> None:
        """Re-apply scale-dependent pixel budgets after a UI scale change."""
        self._apply_column_floors()
        self.alert_center.apply_scaled_metrics()

    # ------------------------------------------------------------------
    def _apply_desk_split(self) -> None:
        """Open at the preset column weights, or the trader's saved drag."""
        splitter = self.desk_splitter
        if splitter is None:
            return
        desk_layout.apply_saved_sizes(
            splitter,
            DESK_SPLIT_KEY,
            desk_layout.desk_split_for(self.width() or 1640),
        )
        # The chart column's share widens on a bigger desk instead of holding a
        # fixed ratio - the opposite of the old 1:2 stretch, which shrank it.
        desk_layout.track_preset(
            self, splitter, DESK_SPLIT_KEY, desk_layout.desk_split_for
        )
        desk_layout.persist_sizes(self, splitter, DESK_SPLIT_KEY)

    def set_setups_visible(self, visible: bool) -> None:
        """Show or hide the Master AVWAP half of the desk.

        Hidden rather than sized to zero, for the same reason ``toggle_setups_
        expanded`` hides the alert column: both desk columns carry an explicit
        minimum width and the splitter is non-collapsible, so a zero size is
        clamped straight back to that minimum.
        """
        visible = bool(visible)
        # Kept in step first so an F9-driven reveal relabels the button too.
        self.setups_toggle.set_setups_visible(visible)
        if visible == self._setups_visible:
            return
        self._setups_visible = visible
        splitter = self.desk_splitter
        if self.workspace_mode != "workspace" or splitter is None:
            return
        if visible:
            self.master_workspace.setVisible(True)
            saved = self._setups_restore_sizes
            if saved and sum(saved) > 0 and len(saved) == splitter.count():
                splitter.setSizes(saved)
            else:
                # Deliberately not _apply_desk_split(): that also installs the
                # preset tracker and the persist connection, so calling it on
                # every reveal would stack a duplicate event filter and a
                # duplicate splitterMoved handler per toggle.
                desk_layout.apply_saved_sizes(
                    splitter,
                    DESK_SPLIT_KEY,
                    desk_layout.desk_split_for(self.width() or 1640),
                )
            self._setups_restore_sizes = None
        else:
            if self._setups_expanded:
                # The setups own the whole desk at this moment; hiding them
                # without undoing that leaves BOTH columns invisible.
                self.toggle_setups_expanded()
            self._setups_restore_sizes = splitter.sizes()
            self.master_workspace.setVisible(False)

    def setups_visible(self) -> bool:
        return self._setups_visible

    def toggle_setups_expanded(self) -> bool:
        """F9: give the setups table the whole desk, and back again.

        Expanded also switches the table to its full column profile, since the
        reason to want the width is to read the columns the compact profile
        hides. Returns the new state.
        """
        splitter = self.desk_splitter
        if splitter is None:
            return False
        if not self._setups_expanded and not self._setups_visible:
            # F9 means "give the setups the desk", which is incoherent while
            # they are hidden - revealing them is the honest first step.
            self.set_setups_visible(True)
        self._setups_expanded = not self._setups_expanded
        if self._setups_expanded:
            self._collapsed_sizes = splitter.sizes()
            # Hide rather than size-to-zero: the columns carry an explicit
            # minimum width and the splitter is non-collapsible, so a zero size
            # would just be clamped back to that minimum.
            self.alert_center.setVisible(False)
            self.master_panel.set_column_profile("full")
        else:
            self.alert_center.setVisible(True)
            saved = getattr(self, "_collapsed_sizes", None)
            if saved and sum(saved) > 0:
                splitter.setSizes(saved)
            else:
                self._apply_desk_split()
            self.master_panel.set_column_profile("compact")
        return self._setups_expanded


class MasterAvwapWorkspace(QFrame):
    def __init__(
        self,
        master_panel: MasterAvwapPanel,
        theta_panel: ThetaPanel,
        watchlists_panel: WatchlistsPanel,
        industry_panel: IndustryPanel | None = None,
        rs_window_panel: RsWindowPanel | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.master_panel = master_panel
        self.theta_panel = theta_panel
        self.watchlists_panel = watchlists_panel
        self.industry_panel = industry_panel
        self.rs_window_panel = rs_window_panel
        self.tabs = QTabWidget()
        self.tabs.addTab(self.master_panel, "Setups")
        self.tabs.addTab(self.theta_panel, "Theta Plays")
        self.tabs.addTab(self.watchlists_panel, "Watchlists")
        if self.industry_panel is not None:
            self.tabs.addTab(self.industry_panel, "Industry Board")
        if self.rs_window_panel is not None:
            self.tabs.addTab(self.rs_window_panel, "RS Window")
        self.master_panel.scan_service.finished.connect(lambda *_args: self.theta_panel.refresh())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def show_setups(self) -> None:
        self.tabs.setCurrentWidget(self.master_panel)

    def show_theta(self) -> None:
        self.theta_panel.refresh()
        self.tabs.setCurrentWidget(self.theta_panel)

    def show_watchlists(self) -> None:
        self.tabs.setCurrentWidget(self.watchlists_panel)

    def show_industry_board(self) -> None:
        if self.industry_panel is not None:
            self.industry_panel.reload_from_disk()
            self.tabs.setCurrentWidget(self.industry_panel)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
        if child_layout is not None:
            _clear_layout(child_layout)


