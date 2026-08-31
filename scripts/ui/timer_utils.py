from __future__ import annotations

"""Small helpers for keeping periodic GUI work out of one timer herd.

Qt timers created during desk startup otherwise keep nearly identical phases:
every 30-second job lands together, and every second 30-second wave collides
with every 60-second job.  ``start_staggered`` changes only the first phase;
after that the timer keeps its exact contract interval.

``SignalCoalescer`` answers the other half of the same problem: not timers that
collide, but a *signal* that arrives in bursts and whose every listener treats
each arrival as "rebuild everything".
"""

import logging

from PySide6.QtCore import QObject, QTimer


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


#: The ceiling the trader set on 2026-08-31: a reaction to a signal may lag by
#: this much, and no more. Short enough that a hand-typed ticker still appears
#: instantly; long enough that a whole drain loop lands inside one window.
COALESCE_MS = 200


class SignalCoalescer(QObject):
    """Collapse a burst of "refresh yourself" requests into ONE call.

    Why this exists (2026-08-31): `FocusPickStore` notifies on every add, which
    is the right contract - several surfaces depend on knowing about each
    mutation. What was wrong is that five listeners each treated one add as a
    full rebuild, so the DESK drain adopting 45 staged picks rebuilt four chip
    boards, 350 alert-feed widget trees, an HTML strength board, a combo box and
    the whole setups viewport **45 times in 13 seconds**. The desk was Not
    Responding and the trader killed it twice.

    Leading-edge window, trailing fire
    ----------------------------------
    The FIRST request opens a window and the call happens when the window
    closes; later requests inside an open window are folded into it and
    deliberately **do not restart** it. That matters:

    * a burst arriving inside one event-loop slot - which is exactly what a
      synchronous drain loop is - can never be interleaved with the timer, so
      it produces exactly one call;
    * a plain debounce (restart on every request) would be starved by a stream
      arriving faster than its window and could stop firing altogether. A fixed
      window cannot starve: the reaction is never more than `interval_ms` late.

    The coalescing lives at the LISTENER, never in the store: the signal keeps
    firing per mutation for everyone else.
    """

    def __init__(self, target, interval_ms: int = COALESCE_MS, parent=None) -> None:
        super().__init__(parent)
        self._target = target
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(0, int(interval_ms)))
        self._timer.timeout.connect(self._fire)
        self._pending = False

    def request(self) -> None:
        """Ask for the reaction. Cheap, and safe to call in a tight loop."""
        self._pending = True
        if not self._timer.isActive():
            self._timer.start()

    def flush(self) -> None:
        """Run an owed reaction now (or do nothing if none is owed)."""
        self._timer.stop()
        self._fire()

    def cancel(self) -> None:
        """Drop an owed reaction without running it."""
        self._timer.stop()
        self._pending = False

    def is_pending(self) -> bool:
        return self._pending

    def remaining_ms(self) -> int:
        """Milliseconds until the open window closes; -1 when none is open."""
        return int(self._timer.remainingTime())

    def _fire(self) -> None:
        if not self._pending:
            return
        # Cleared BEFORE the call: a reaction that raises must not leave the
        # coalescer armed forever, and one that requests again (a refresh that
        # notices more work) must be able to open a fresh window.
        self._pending = False
        try:
            self._target()
        except Exception:
            logging.debug("Coalesced GUI refresh failed.", exc_info=True)
