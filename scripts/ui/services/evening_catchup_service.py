"""Builds the EVENING catch-up card off the Qt thread (one owner, single-flight).

The Qt thread hands in a cheap snapshot (the diverted alerts as dicts, the
Movers board dict, when EVENING started). The worker reads the evening
strength checks, the latest swing rows and the price-alert trigger log, builds
the card with `evening_catchup.build_catchup` and emits `ready`.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

import evening_catchup


def _default_swing_rows() -> list:
    try:
        from ui.services.data_feed import load_latest_setup_rows_with_meta

        return list(load_latest_setup_rows_with_meta().get("rows") or [])
    except Exception:
        logging.debug("Catch-up card: swing rows unavailable.", exc_info=True)
        return []


def _default_persistence(now: datetime) -> dict:
    try:
        import evening_mode

        return evening_mode.assess_pick_persistence(evening_mode.load_evening_state(now))
    except Exception:
        logging.debug("Catch-up card: evening checks unavailable.", exc_info=True)
        return {}


def _default_triggers(now: datetime) -> list:
    try:
        import price_alerts

        return price_alerts.todays_triggers(now)
    except Exception:
        logging.debug("Catch-up card: price-alert log unavailable.", exc_info=True)
        return []


class EveningCatchupService(QObject):
    ready = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        swing_rows: Callable[[], list] = _default_swing_rows,
        persistence: Callable[[datetime], dict] = _default_persistence,
        triggers: Callable[[datetime], list] = _default_triggers,
    ) -> None:
        super().__init__(parent)
        self._swing_rows = swing_rows
        self._persistence = persistence
        self._triggers = triggers
        self._running = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def request(self, snapshot: dict[str, Any]) -> bool:
        """Start one build. A build already running is not doubled."""
        with self._lock:
            if self._running:
                return False
            self._running = True
        threading.Thread(
            target=self._build,
            args=(dict(snapshot or {}),),
            name="evening-catchup",
            daemon=True,
        ).start()
        return True

    def build_now(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """The worker's body, callable directly (tests)."""
        now = datetime.now()
        return evening_catchup.build_catchup(
            alerts=snapshot.get("alerts") or [],
            movers_board=snapshot.get("movers_board") or {},
            persistence=self._persistence(now),
            swing_rows=self._swing_rows(),
            price_triggers=self._triggers(now),
            since=snapshot.get("since"),
            now=now,
        )

    def _build(self, snapshot: dict[str, Any]) -> None:
        try:
            payload = self.build_now(snapshot)
        except Exception:
            logging.exception("The evening catch-up card could not be built.")
            payload = None
        finally:
            with self._lock:
                self._running = False
        if payload is not None:
            self.ready.emit(payload)
