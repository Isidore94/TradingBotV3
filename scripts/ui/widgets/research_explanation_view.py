from __future__ import annotations

from html import escape
from typing import Any, Mapping

from PySide6.QtWidgets import QTextBrowser

from research_explanations import build_research_explanation
from ui import theme


def render_research_explanation_html(
    kind: str,
    row: Mapping[str, Any] | None,
    *,
    include_body: bool = True,
) -> str:
    explanation = build_research_explanation(kind, row)
    body = theme.color("text_primary")
    muted = theme.color("text_secondary")
    favorite = theme.color("favorite")
    caution = theme.color("caution")
    parts = [f"<body style='color:{body}; font-size:9pt'>"] if include_body else []
    parts.extend(
        [
            f"<div style='color:{favorite}; font-weight:700'>{escape(str(explanation['eyebrow']))}</div>",
            f"<h2 style='margin:2px 0 6px 0'>{escape(str(explanation['title']))}</h2>",
            f"<p>{escape(str(explanation['summary']))}</p>",
            "<h3 style='margin-bottom:2px'>Step by step</h3><ol>",
        ]
    )
    parts.extend(f"<li>{escape(str(step))}</li>" for step in explanation.get("steps") or [])
    parts.append("</ol><h3 style='margin-bottom:2px'>What the evidence says</h3>")
    parts.extend(f"<p style='margin:2px 0'>{escape(str(item))}</p>" for item in explanation.get("evidence") or [])
    parts.append(
        f"<div style='border-left:3px solid {caution}; padding-left:7px; margin-top:8px; color:{muted}'>"
        f"<b>Do not overread it:</b> {escape(str(explanation.get('caution') or ''))}</div>"
    )
    if include_body:
        parts.append("</body>")
    return "".join(parts)


class ResearchExplanationView(QTextBrowser):
    """Shared click-to-explain pane for aggregate research rows."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setMinimumWidth(340)
        self.setVisible(False)

    def show_row(self, kind: str, row: Mapping[str, Any] | None) -> None:
        self.setHtml(render_research_explanation_html(kind, row))
        self.setVisible(True)
