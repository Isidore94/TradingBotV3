"""Trade Mentor app: a separate PySide6 process that talks to the RTX 5080's Ollama.

Decision support only. It never places an order, never changes a detector, score
or alert, and owns exactly one store of its own (``MENTOR_CHAT_DB_FILE``).
Importing this package is cheap: Qt and the window load only inside :func:`main`.
"""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the app's Qt loop. The caller already holds the mentor slot."""
    import sys

    from PySide6.QtWidgets import QApplication

    from ui import theme

    theme.configure_platform()
    app = QApplication.instance() or QApplication(list(sys.argv[:1]))
    app.setApplicationName("TradingBotV3 Trade Mentor")
    app.setOrganizationName("TradingBotV3")
    theme.apply_theme(app, "dark")

    from mentor_app.window import MentorWindow

    window = MentorWindow()
    window.show()
    window.start_background()
    try:
        return int(app.exec() or 0)
    finally:
        window.shutdown()
