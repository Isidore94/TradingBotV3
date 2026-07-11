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


def render_best_now_html() -> str:
    """Live 'best performing right now' banner from the tracker exports: the
    top short-term (1-2 session) family and the top swing family (30d realized).
    Re-read on every render so the playbook always shows current evidence."""
    # Imported lazily: setup_tracker_panel imports setup_detail_view, which
    # imports this module, so a module-level import here would be circular.
    from ui.panels.setup_tracker_panel import (
        RECENT_SETUP_TYPE_STATS_FILE,
        SHORT_HORIZON_FILE,
        SHORT_TERM_MIN_SAMPLES,
        _float,
        _int,
        _load_csv_rows,
    )

    muted = theme.color("text_secondary")
    favorite = theme.color("favorite")
    long_c = theme.color("long")
    short_c = theme.color("short")

    short_rows = [
        row
        for row in _load_csv_rows(SHORT_HORIZON_FILE)
        if _int(row.get("samples_2d")) >= SHORT_TERM_MIN_SAMPLES and _float(row.get("avg_r_2d")) is not None
    ]
    short_rows.sort(key=lambda row: -_float(row.get("short_term_score"), -1e9))
    swing_rows = [
        row
        for row in _load_csv_rows(RECENT_SETUP_TYPE_STATS_FILE)
        if _int(row.get("closed_setups")) >= 3 and _float(row.get("avg_closed_r")) is not None
    ]
    swing_rows.sort(key=lambda row: -_float(row.get("avg_closed_r"), -1e9))
    if not short_rows and not swing_rows:
        return ""

    def _line(label: str, row: dict, r_key: str, r_suffix: str, count_key: str) -> str:
        side = str(row.get("side") or "").upper()
        side_color = long_c if side == "LONG" else short_c
        r_value = _float(row.get(r_key))
        r_text = f"{r_value:+.2f}R{r_suffix}" if r_value is not None else ""
        return (
            f"<div><b>{label}:</b> <span style='color:{side_color}'><b>{_esc(side)}</b></span> "
            f"<b>{_esc(row.get('setup_family'))}</b> {r_text} "
            f"<span style='color:{muted}'>(n={_int(row.get(count_key))})</span></div>"
        )

    parts = [
        f"<div style='border:1px solid {favorite}; padding:6px; margin:0 0 8px 0'>",
        f"<b style='color:{favorite}; font-size:10pt'>BEST PERFORMING RIGHT NOW</b>",
    ]
    if short_rows:
        parts.append(_line("Short-term (1-2d)", short_rows[0], "avg_r_2d", "@2d", "samples_2d"))
    if swing_rows:
        parts.append(_line("Swing (30d realized)", swing_rows[0], "avg_closed_r", " closed", "closed_setups"))
    parts.append(
        f"<div style='color:{muted}'>Full evidence tables live in the Setup Tracker panel "
        f"(Short-Term 1-2d / Last 30 Days / Playbooks tabs).</div></div>"
    )
    return "".join(parts)


def render_all_docs_html() -> str:
    body = theme.color("text_primary")
    muted = theme.color("text_secondary")
    favorite = theme.color("favorite")
    parts = [
        f"<body style='color:{body}; font-size:9pt'>",
        f"<h2 style='margin:0; color:{favorite}'>Setup Playbook — every setup, exactly</h2>",
        render_best_now_html(),
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
