from __future__ import annotations

"""Small helpers for keeping periodic GUI work out of one timer herd.

Qt timers created during desk startup otherwise keep nearly identical phases:
every 30-second job lands together, and every second 30-second wave collides
with every 60-second job.  ``start_staggered`` changes only the first phase;
after that the timer keeps its exact contract interval.
"""

from PySide6.QtCore import QTimer


def start_staggered(timer: QTimer, delay_ms: int) -> QTimer:
    """Start ``timer`` after a one-off delay and return the owned starter."""

    starter = QTimer(timer.parent())
    starter.setSingleShot(True)

    def activate() -> None:
        try:
            timer.start()
        except RuntimeError:
            return

    starter.timeout.connect(activate)
    starter.start(max(0, int(delay_ms)))
    timer._stagger_starter = starter  # type: ignore[attr-defined]
    return starter


def stop_staggered(timer: QTimer) -> None:
    """Stop both the periodic timer and a not-yet-fired phase starter."""

    starter = getattr(timer, "_stagger_starter", None)
    if starter is not None:
        try:
            starter.stop()
        except RuntimeError:
            pass
    timer.stop()
