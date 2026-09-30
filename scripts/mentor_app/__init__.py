"""Trade Mentor app: a separate PySide6 process that talks to the RTX 5080's Ollama.

Decision support only. It never places an order, never changes a detector, score
or alert, and owns exactly one store of its own (``MENTOR_CHAT_DB_FILE``).
Importing this package is cheap: Qt and the window load only inside :func:`main`.
"""

from __future__ import annotations

from typing import Sequence


def _log_to_file() -> None:
    """pythonw has no console: the app logs to LOCAL_LOG_DIR/mentor_app-YYYYMMDD.log."""
    import logging
    from datetime import date
    from pathlib import Path

    try:
        from project_paths import LOCAL_LOG_DIR

        folder = Path(LOCAL_LOG_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(folder / f"mentor_app-{date.today():%Y%m%d}.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        )
    except Exception:  # noqa: BLE001 - logging must never keep the app from starting
        logging.basicConfig(level=logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the app's Qt loop. The caller already holds the mentor slot."""
    import sys

    _log_to_file()

    from PySide6.QtWidgets import QApplication

    from ui import theme

    theme.configure_platform()
    app = QApplication.instance() or QApplication(list(sys.argv[:1]))
    app.setApplicationName("TradingBotV3 Trade Mentor")
    app.setOrganizationName("TradingBotV3")
    theme.apply_theme(app, "dark")

    from mentor_app.window import MentorWindow
    from ui.services.mentor_launcher import FOLLOW_DESK_FLAG

    # Launched by the desk (the flag): close when the desk closes. By hand: never auto-exit.
    window = MentorWindow(follow_desk=FOLLOW_DESK_FLAG in list(argv or ()))
    window.show()
    window.start_background()
    try:
        return int(app.exec() or 0)
    finally:
        window.shutdown()
