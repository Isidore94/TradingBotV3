"""A tab stack shown as a bottom drawer under a chart pane (compact desk).

Collapsed, only the tab bar shows, so tab badges stay in sight. Clicking a tab
opens the drawer to its remembered height; clicking the current tab again
closes it. The drawer is a vertical splitter, so the trader can drag it; the
drag is saved under its own key (`desk_layout.COMPACT_ALERT_SPLIT_KEY`).
The tab widget itself is only re-hosted, never rebuilt, so every tab index
and signal keeps working.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QSplitter, QTabWidget, QWidget

from ui import theme
from ui.panels import desk_layout

#: The opened drawer's default height at scale 1.0.
DEFAULT_EXPANDED_PX = 300
#: The chart pane never gets less than this when the drawer opens.
MIN_TOP_PX = 200


class TabDrawer(QObject):
    """Hosts `top` over `bottom` (which holds `tabs`) in a collapsible split."""

    def __init__(self, owner: QWidget, tabs: QTabWidget, *, key: str | None = None) -> None:
        super().__init__(owner)
        self._tabs = tabs
        self._key = key or desk_layout.COMPACT_ALERT_SPLIT_KEY
        self._active = False
        self._expanded = False
        self._applying = False
        #: The opened height the trader last dragged to, this session.
        self._expanded_px: int | None = None
        self._top: QWidget | None = None
        self._bottom: QWidget | None = None
        self._saved_minimums: dict[int, int] = {}
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setObjectName("CompactDrawerSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setVisible(False)
        self.splitter.installEventFilter(self)
        self.splitter.splitterMoved.connect(self._on_moved)
        desk_layout.persist_sizes(owner, self.splitter, self._key)
        tabs.tabBarClicked.connect(self._on_tab_bar_clicked)

    # ------------------------------------------------------------------ state
    def is_active(self) -> bool:
        return self._active

    def is_expanded(self) -> bool:
        return self._active and self._expanded

    def collapsed_height(self) -> int:
        """The tab bar plus the tab widget's frame: all a closed drawer shows."""
        return self._tabs.tabBar().sizeHint().height() + theme.px(2)

    def expanded_height(self) -> int:
        if self._expanded_px is not None:
            return self._expanded_px
        saved = desk_layout.load_sizes(self._key, 2)
        if saved and saved[1] > self.collapsed_height() + theme.px(8):
            return int(saved[1])
        return theme.px(DEFAULT_EXPANDED_PX)

    # ------------------------------------------------------------ hosting
    def activate(self, top: QWidget, bottom: QWidget) -> None:
        """Take `top` and `bottom` into the drawer split, collapsed."""
        if self._active:
            return
        self._top, self._bottom = top, bottom
        floor = self.collapsed_height()
        for widget in (bottom, self._tabs):
            self._saved_minimums[id(widget)] = widget.minimumHeight()
            widget.setMinimumHeight(floor)
        self.splitter.addWidget(top)
        self.splitter.addWidget(bottom)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self._active = True
        self._expanded = False
        self.splitter.setVisible(True)
        self._apply()

    def deactivate(self) -> None:
        """Let go of both widgets (the caller re-hosts them) and restore floors."""
        if not self._active:
            return
        self._active = False
        self.splitter.setVisible(False)
        for widget in (self._bottom, self._tabs):
            if widget is not None:
                widget.setMinimumHeight(self._saved_minimums.pop(id(widget), 0))
        self._top = self._bottom = None

    # ------------------------------------------------------------ open/close
    def expand(self) -> None:
        if not self._active:
            return
        self._expanded = True
        self._apply()

    def collapse(self) -> None:
        if not self._active:
            return
        self._expanded = False
        self._apply()

    def _on_tab_bar_clicked(self, index: int) -> None:
        # Emitted on press, before the tab widget switches pages.
        if not self._active or index < 0:
            return
        if self._expanded and index == self._tabs.currentIndex():
            self.collapse()
        else:
            self.expand()

    def _apply(self) -> None:
        total = self.splitter.height()
        if total <= 0:
            return
        if self._expanded:
            bottom = min(self.expanded_height(), max(total - theme.px(MIN_TOP_PX), 0))
            bottom = max(bottom, self.collapsed_height())
        else:
            bottom = self.collapsed_height()
        self._applying = True
        try:
            self.splitter.setSizes([max(total - bottom, 1), bottom])
        finally:
            self._applying = False

    def _on_moved(self, *_args) -> None:
        if self._applying or not self._active:
            return
        sizes = self.splitter.sizes()
        if len(sizes) == 2:
            self._expanded = sizes[1] > self.collapsed_height() + theme.px(8)
            if self._expanded:
                self._expanded_px = int(sizes[1])

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if watched is self.splitter and event.type() == QEvent.Type.Resize and self._active:
            self._apply()
        return False
