from __future__ import annotations

from PySide6.QtWidgets import QLabel

from ui import theme


# Map badge tones to semantic theme tokens so a light/dark switch restyles chips.
BADGE_TONE_TOKENS = {
    "long": "long",
    "short": "short",
    "favorite": "favorite",
    "near": "near",
    "study": "study",
    "caution": "caution",
    "neutral": "neutral",
    "info": "info",
}


class Badge(QLabel):
    def __init__(self, text: str = "", tone: str = "neutral", parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Badge")
        self.set_tone(tone)
        self.setContentsMargins(8, 2, 8, 2)

    def set_tone(self, tone: str) -> None:
        token = BADGE_TONE_TOKENS.get(tone, "neutral")
        fg = theme.color(token)
        bg = theme.with_alpha(fg, 0.16)
        border = theme.with_alpha(fg, 0.55)
        self.setStyleSheet(
            f"""
            QLabel#Badge {{
                background: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 2px 8px;
                font-weight: 650;
            }}
            """
        )
