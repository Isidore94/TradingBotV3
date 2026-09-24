"""The compact desk's thin top row: page tabs, a More menu, and a right slot.

Built from the window's page list, so every page is reachable exactly once:
the primary titles are checkable buttons, every other page is a More-menu
action. The row only asks for a page (`pageRequested`); the window's
`_select_page` stays the one place a page is changed.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QWidget,
)

from ui import theme

#: The pages that get a tab of their own, in order. Everything else is in More.
PRIMARY_PAGE_TITLES = ("Trading Desk", "Journal", "Day Review", "Research")
#: Shorter tab labels for the primary pages that need one.
TAB_LABELS = {"Trading Desk": "Desk"}
MORE_LABEL = "More ▾"


class PageTabRow(QFrame):
    """Page tabs on the left, a More menu, and a right-hand widget slot."""

    pageRequested = Signal(int)

    def __init__(
        self,
        titles: Sequence[str],
        parent=None,
        *,
        primary_titles: Iterable[str] = PRIMARY_PAGE_TITLES,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PageTabRow")
        self._titles = list(titles)
        primary = [title for title in primary_titles if title in self._titles]
        #: page index -> tab button (primary pages) or menu action (the rest).
        self.tab_buttons: dict[int, QPushButton] = {}
        self.more_actions: dict[int, QAction] = {}
        self._current = -1

        self.brand = QLabel("TradingBotV3")
        self.brand.setObjectName("PageTabBrand")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(theme.px(8), theme.px(1), theme.px(8), theme.px(1))
        layout.setSpacing(theme.px(4))
        layout.addWidget(self.brand)
        layout.addSpacing(theme.px(10))

        for title in primary:
            index = self._titles.index(title)
            button = QPushButton(TAB_LABELS.get(title, title))
            button.setObjectName("PageTabButton")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _checked=False, page=index: self._request(page))
            self.tab_buttons[index] = button
            layout.addWidget(button)

        self.more_menu = QMenu(self)
        for index, title in enumerate(self._titles):
            if index in self.tab_buttons:
                continue
            action = QAction(title, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, page=index: self._request(page))
            self.more_menu.addAction(action)
            self.more_actions[index] = action
        self.more_button = QToolButton()
        self.more_button.setObjectName("PageTabMore")
        self.more_button.setText(MORE_LABEL)
        self.more_button.setCheckable(True)
        self.more_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_button.setMenu(self.more_menu)
        layout.addWidget(self.more_button)
        layout.addStretch(1)

        self.right_host = QWidget()
        self.right_layout = QHBoxLayout(self.right_host)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(theme.px(6))
        layout.addWidget(self.right_host)

    # ------------------------------------------------------------------
    def _request(self, index: int) -> None:
        # The window decides; set_current() then reflects what it did.
        self.set_current(self._current)
        self.pageRequested.emit(int(index))

    def page_indices(self) -> list[int]:
        """Every page index this row can reach, tabs first then More."""
        return [*self.tab_buttons, *self.more_actions]

    def set_current(self, index: int) -> None:
        self._current = int(index)
        for page, button in self.tab_buttons.items():
            button.setChecked(page == index)
        for page, action in self.more_actions.items():
            action.setChecked(page == index)
        self.more_button.setChecked(index in self.more_actions)

    def set_label(self, index: int, text: str) -> None:
        """Mirror a nav label (e.g. a review badge) onto the tab or menu entry."""
        title = self._titles[index] if 0 <= index < len(self._titles) else ""
        if index in self.tab_buttons:
            short = TAB_LABELS.get(title)
            self.tab_buttons[index].setText(
                short + text[len(title):] if short and text.startswith(title) else text
            )
        elif index in self.more_actions:
            self.more_actions[index].setText(text)

    def label(self, index: int) -> str:
        if index in self.tab_buttons:
            return self.tab_buttons[index].text()
        if index in self.more_actions:
            return self.more_actions[index].text()
        return ""

    def set_page_visible(self, index: int, visible: bool) -> None:
        if index in self.tab_buttons:
            self.tab_buttons[index].setVisible(bool(visible))
        elif index in self.more_actions:
            self.more_actions[index].setVisible(bool(visible))

    def add_right_widget(self, widget: QWidget) -> None:
        self.right_layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)

    def take_right_widget(self, widget: QWidget) -> None:
        self.right_layout.removeWidget(widget)
