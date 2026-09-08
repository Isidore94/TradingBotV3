"""The Desk's Strength window as ONE flat, scrolling page (trader, 2026-09-07).

Trader: *"the strength tab is unusable there's like 2 tabs and they get no
space each. Create a solution that removes the tabs and just collates all the
data to be more easily readable."*

**What it replaced.** Since V1 (decision 0016 answer 7) the strength column
stacked the Focus strength board over two `CollapsibleSection`s - RS/RW Board
(open) and M5 Strength Board (closed) - and split the column's height between
whichever were open by stretch factor. That column is 40% of the lower third
of the alert column: two open sections shared a few hundred pixels, each got
a window a fraction of the height of the document inside it, each with its
own scrollbar, and a closed section hid a whole read behind one header row.
Nothing was readable without a scroll inside a scroll.

**What this is.** One `QScrollArea` with ONE scrollbar, and the four reads
laid one under another in the order the sections had: the Focus strength
board, the automatic RS/RW entry board, the RRS sweep snapshot (its three
scopes STACKED rather than three-abreast - three four-column tables side by
side in this width read at ~50 px a cell), then the M5 Strength Board under a
heading of its own. Every text board is sized to its DOCUMENT
(`fit_height_to_document`) and every table to its ROWS (capped at
`strength_board_panel.FIT_ROWS_CAP`), so nothing inside the page scrolls on
its own: the page reads top to bottom like a report and the wheel always moves
the page.

**What it costs the charts: nothing.** The boards' own minimum widths stop at
the scroll area exactly as they stopped at the two section scroll areas
before, and the page's floor is the 170 px the column had - 170 plus the tab
stack's 170 stays inside the alert column's 360 px budget. The page's vertical
scrollbar is ALWAYS ON rather than as-needed: a bar that appears takes ~16 px
from the document width, reflows every text board, changes their heights, and
can make itself unnecessary again - a resize loop on the Qt thread. A bar that
is always there costs 16 px once.

This page HOSTS. It owns no data, timer, thread or fetch: `MainWindow` still
owns the one `StrengthBoardService`, and the Alert Center still owns every
board and its signals. Nothing about what a click does changed - every ticker
still charts into the centre Visual Alert Review pane.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui import theme

#: The page's floor: the one the strength column had, so the alert column's
#: 360 px budget (170 for the tab stack, 170 here) is untouched by the move.
MIN_PAGE_WIDTH = 170


class _DocumentFit(QObject):
    """Keep one `QTextBrowser` exactly as tall as its document.

    Parented to the browser, so it lives and dies with it. It re-binds to the
    document layout on every `textChanged` (a `setHtml` may hand the document
    a fresh layout object) and fits synchronously there too: `QTextDocument
    .size()` lays out on demand, so the height is right the moment the HTML
    lands rather than on a later idle tick. A width change reflows the
    document, which fires `documentSizeChanged`, which re-fits - so the board
    follows the page's width without any resize handling of its own.
    """

    def __init__(self, browser: QTextBrowser) -> None:
        super().__init__(browser)
        self._browser = browser
        self._layout = None
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        browser.textChanged.connect(self._rebind)
        self._rebind()

    def _rebind(self) -> None:
        layout = self._browser.document().documentLayout()
        if layout is not self._layout:
            if self._layout is not None:
                try:
                    self._layout.documentSizeChanged.disconnect(self._on_document_size)
                except (RuntimeError, TypeError):
                    pass
            self._layout = layout
            layout.documentSizeChanged.connect(self._on_document_size)
        self._fit()

    def _on_document_size(self, _size) -> None:
        self._fit()

    def _fit(self) -> None:
        browser = self._browser
        document = browser.document()
        if document.pageSize().width() <= 0:
            # Never shown yet. QTextEdit gives the document its page width on
            # its first relayout (show or resize) and a layout at width 0 is
            # empty - measured: `size()` reads (0, 0) until then. Lay out at
            # the viewport's width now so the height is real from the first
            # `setHtml`; the first real relayout replaces the width and
            # re-fits through `documentSizeChanged`.
            document.setTextWidth(max(1, browser.viewport().width()))
        height = max(
            1, math.ceil(document.size().height()) + 2 * browser.frameWidth()
        )
        if browser.minimumHeight() != height or browser.maximumHeight() != height:
            browser.setFixedHeight(height)


def fit_height_to_document(browser: QTextBrowser) -> None:
    """Size `browser` to its document, now and on every change. Idempotent."""
    if getattr(browser, "_document_fit", None) is None:
        browser._document_fit = _DocumentFit(browser)


class StrengthPage(QScrollArea):
    """The one flat page. Built by `AlertCenterPanel`, which keeps the boards."""

    def __init__(
        self,
        *,
        focus_strength,
        entry_board,
        rrs_snapshot,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("StrengthPage")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Always on, never as-needed - see the module docstring.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._blocks: list[QWidget] = []
        self.body = QWidget()
        self._layout = QVBoxLayout(self.body)
        self._layout.setContentsMargins(0, 0, theme.px(4), theme.px(8))
        self._layout.setSpacing(theme.px(12))

        # The Focus board sized itself Expanding/Expanding for a column it
        # shared by stretch factor; on a page it takes the height it needs and
        # the slack goes to the trailing stretch, not to an empty board.
        focus_strength.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        fit_height_to_document(focus_strength.board)
        self._add_block(focus_strength)

        fit_height_to_document(entry_board.view)
        self._add_block(entry_board)

        rrs_snapshot.set_stacked_scopes(True)
        fit_height_to_document(rrs_snapshot.board)
        self._add_block(rrs_snapshot)

        # The TC2000 board carries no title of its own - the section that used
        # to host it named it - so the page names it. Hidden until the board is
        # attached, so a desk without one shows no orphan heading.
        self.strength_board_title = QLabel("M5 Strength Board (TC2000)")
        self.strength_board_title.setObjectName("SectionTitle")
        # Wrapped, so it never asks the page for more width than a word.
        self.strength_board_title.setWordWrap(True)
        self.strength_board_title.setVisible(False)
        self._layout.addWidget(self.strength_board_title)
        self._layout.addStretch(1)

        self.setWidget(self.body)
        self.apply_scaled_metrics()

    def _add_block(self, widget: QWidget) -> None:
        self._layout.addWidget(widget, 0)
        self._blocks.append(widget)

    def attach_strength_board(self, board) -> None:
        """Put the M5 Strength Board at the foot of the page, sized to its rows."""
        board.set_fit_rows(True)
        self.strength_board_title.setVisible(True)
        # Before the trailing stretch, so the page keeps its slack at the bottom.
        self._layout.insertWidget(self._layout.count() - 1, board, 0)
        self._blocks.append(board)

    def blocks(self) -> list[QWidget]:
        """The reads on the page, top to bottom. The seam the tests read."""
        return list(self._blocks)

    def apply_scaled_metrics(self) -> None:
        """The page's floor at the current UI scale (see ui.theme.px)."""
        self.setMinimumWidth(theme.px(MIN_PAGE_WIDTH))
