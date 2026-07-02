from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
)

from setup_docs import STOP_CLOSE_FAILURES, TIME_STOP_SESSIONS, all_setup_docs_by_group
from ui import theme
from ui.widgets.section_header import SectionHeader


class SetupDocsPanel(QFrame):
    """Research tab: the setup encyclopedia — how every setup works, exactly."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self.family_list = QListWidget()
        self.family_list.setMaximumWidth(280)
        self.doc_view = QTextBrowser()
        self.doc_view.setOpenExternalLinks(False)

        self._docs_by_key: dict[str, dict] = {}
        overview_item = QListWidgetItem("All setups (overview)")
        overview_item.setData(Qt.ItemDataRole.UserRole, "__overview__")
        self.family_list.addItem(overview_item)
        for group_name, entries in all_setup_docs_by_group():
            header = QListWidgetItem(f"— {group_name} —")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self.family_list.addItem(header)
            for key, doc in entries:
                self._docs_by_key[key] = doc
                item = QListWidgetItem(doc["label"])
                item.setData(Qt.ItemDataRole.UserRole, key)
                self.family_list.addItem(item)
        self.family_list.currentItemChanged.connect(self._on_family_selected)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.family_list)
        splitter.addWidget(self.doc_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(
            SectionHeader(
                "Setup Playbook",
                "Exact mechanics for every setup family: detection rules, entry trigger, stop placement, and the profit-take plan.",
            )
        )
        layout.addWidget(splitter, 1)
        self.family_list.setCurrentRow(0)

    def _on_family_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        if key == "__overview__":
            self.doc_view.setHtml(render_all_docs_html())
        elif key in self._docs_by_key:
            self.doc_view.setHtml(render_doc_html(key, self._docs_by_key[key]))


def _esc(value) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_doc_html(key: str, doc: dict, *, heading_level: int = 2) -> str:
    body = theme.color("text_primary")
    muted = theme.color("text_secondary")
    favorite = theme.color("favorite")
    long_c = theme.color("long")
    short_c = theme.color("short")

    h = f"h{max(2, min(4, heading_level))}"
    parts = [
        f"<{h} style='margin:12px 0 2px 0; color:{favorite}'>{_esc(doc['label'])}</{h}>",
        f"<div style='color:{muted}; margin-bottom:6px'>{_esc(doc['group'])} &middot; family key: <code>{_esc(key)}</code></div>",
        f"<p style='margin:4px 0'>{_esc(doc['what'])}</p>",
        f"<div style='color:{long_c}; font-weight:bold; margin-top:6px'>Detection — exactly</div>",
        "<ul style='margin:2px 0 6px 18px'>",
    ]
    for rule in doc["detection"]:
        parts.append(f"<li>{_esc(rule)}</li>")
    parts.append("</ul>")
    parts.append(f"<div><b style='color:{long_c}'>Entry:</b> {_esc(doc['entry'])}</div>")
    parts.append(f"<div><b style='color:{short_c}'>Stop:</b> {_esc(doc['stop'])}</div>")
    parts.append(f"<div><b style='color:{favorite}'>Targets:</b> {_esc(doc['targets'])}</div>")
    if doc.get("evidence"):
        parts.append(
            f"<div style='color:{muted}; margin-top:4px'><b>Measured:</b> {_esc(doc['evidence'])}</div>"
        )
    return f"<body style='color:{body}; font-size:9pt'>" + "".join(parts) + "</body>"


def render_all_docs_html() -> str:
    body = theme.color("text_primary")
    muted = theme.color("text_secondary")
    favorite = theme.color("favorite")
    parts = [
        f"<body style='color:{body}; font-size:9pt'>",
        f"<h2 style='margin:0; color:{favorite}'>Setup Playbook — every setup, exactly</h2>",
        f"<p style='color:{muted}'>Shared exit discipline: stops are LEVELS — a stop fires after "
        f"{STOP_CLOSE_FAILURES} daily closes beyond the level (1 close for post-earnings setups), never on an "
        f"intraday wick. Default profit plan: 50% at the 2nd deviation band, rest toward the 3rd band with the "
        f"stop trailed to the 1st band after the partial. Everything times out at {TIME_STOP_SESSIONS} sessions.</p>",
    ]
    for group_name, entries in all_setup_docs_by_group():
        parts.append(f"<h3 style='margin:14px 0 2px 0; color:{muted}'>{_esc(group_name)}</h3><hr/>")
        for key, doc in entries:
            inner = render_doc_html(key, doc, heading_level=4)
            inner = inner.replace(f"<body style='color:{body}; font-size:9pt'>", "").replace("</body>", "")
            parts.append(inner)
    parts.append("</body>")
    return "".join(parts)
