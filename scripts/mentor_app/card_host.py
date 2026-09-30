"""The Trade Mentor card inside the app: the same scheduler, card and writers the desk used.

Only built when ``mentor_app_enabled`` is on; then this process is the one owner of
``trade_mentor_slots.json`` (``TradeMentorService`` moves whole) and of the Mentor's
journal events. The host logic is ``MentorHostMixin`` - the desk's own code, not a
fork - pointed at :class:`MentorCardDock` instead of the Alert Center popup. The card
never pops a window or takes focus: it sits under the transcript and the window posts
a quiet line plus an Inbox item.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from ui.services.mentor_host import MentorHostMixin

#: The econ view is re-read this often (same cadence as the desk's reminder service).
ECON_REFRESH_MS = 15 * 60_000


class MentorCardDock(QWidget):
    """The card's home in the app: the morning econ block above the Trade Mentor card.

    Same surface API the desk's ``AlertChartReview`` gives the host, minus the popup.
    """

    def __init__(self, parent: QWidget | None = None, *, context_service=None, card=None) -> None:
        super().__init__(parent)
        self.setObjectName("MentorCardDock")
        from ui.widgets.econ_brief_block import EconBriefBlock
        from ui.widgets.trade_mentor_card import TradeMentorCard

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("TradeMentorScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(self.scroll)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        self.econ_block = EconBriefBlock(body)
        self.econ_block.setVisible(False)
        self.econ_block.hideRequested.connect(self.hide_econ_brief)
        self.mentor_card = card or TradeMentorCard(body, context_service=context_service)
        self.mentor_card.setVisible(False)
        body_layout.addWidget(self.econ_block)
        body_layout.addWidget(self.mentor_card)
        body_layout.addStretch(1)
        self.scroll.setWidget(body)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll)
        self.mentor_card.answered.connect(lambda _slot_id: self._sync())
        self.mentor_card.skipped.connect(lambda _record: self._sync())
        self.setVisible(False)

    def _sync(self) -> None:
        """The dock shows only while the card or the econ block has something up."""
        self.setVisible(self.mentor_card.isVisibleTo(self) or self.econ_block.isVisibleTo(self))

    # -- the surface the host drives (same names as AlertChartReview) --------
    def show_mentor_slot(self, slot, previous=None) -> None:
        try:
            self.mentor_card.show_slot(slot, previous=previous)
        except Exception:  # noqa: BLE001 - a prompt never costs the app
            logging.debug("Trade Mentor card could not be shown.", exc_info=True)
        self._sync()

    def show_econ_brief(self, view) -> None:
        try:
            self.econ_block.set_view(view)
            self.econ_block.setVisible(True)
        except Exception:  # noqa: BLE001
            logging.debug("Econ block could not be shown.", exc_info=True)
        self._sync()

    def update_econ_brief(self, view) -> None:
        try:
            if str(view.get("session") or "") == self.econ_block.session() or self.econ_block.isVisibleTo(self):
                self.econ_block.set_view(view)
        except Exception:  # noqa: BLE001
            logging.debug("Econ block could not be redrawn.", exc_info=True)

    def hide_econ_brief(self) -> None:
        self.econ_block.setVisible(False)
        self._sync()

    def hide_mentor_card(self) -> None:
        try:
            self.mentor_card.hide_card()
        except Exception:  # noqa: BLE001
            logging.debug("Trade Mentor card could not be hidden.", exc_info=True)
        self._sync()

    def give_a_read(self) -> None:
        try:
            self.mentor_card.give_a_read()
        except Exception:  # noqa: BLE001
            logging.debug("Manual read could not be opened.", exc_info=True)
        self._sync()


def _cached_daily_bars(timeframe, symbols, *, now, timeout_seconds):
    """The app has no BounceBot: D1 from the scanner's local cache, M5 left to the batch."""
    if timeframe != "d1":
        return {}
    try:
        from d1_environment_store import _cached_daily_bars as read

        return {str(symbol).strip().upper(): read(str(symbol).strip().upper()) for symbol in symbols}
    except Exception:  # noqa: BLE001
        logging.debug("Trade Mentor D1 cache unreadable.", exc_info=True)
        return {}


class AppMentorHost(MentorHostMixin, QObject):
    """Owns ``TradeMentorService`` + ``TradeMentorContextService`` in the app process."""

    #: (slot) - a card was put up; the window posts a quiet line and an Inbox item.
    cardShown = Signal(object)
    _econViewLoaded = Signal(dict)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        service=None,
        context_service=None,
        dock: MentorCardDock | None = None,
        econ_loader: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(parent)
        from ui.services.trade_mentor_context_service import TradeMentorContextService
        from ui.services.trade_mentor_service import TradeMentorService

        self._clock = clock
        self._econ_loader = econ_loader
        self._journal_importer = None
        self._journal_retry_date = ""
        self.trade_mentor_context_service = context_service or TradeMentorContextService(
            self, cache_loader=_cached_daily_bars
        )
        self.dock = dock or MentorCardDock(context_service=self.trade_mentor_context_service)
        self.dock.mentor_card.set_context_service(self.trade_mentor_context_service)
        self.trade_mentor_service = service or TradeMentorService(self)
        # Same wiring the desk's MainWindow had, pointed at the dock.
        self.trade_mentor_service.promptDue.connect(self._on_prompt_due)
        self.trade_mentor_service.promptExpired.connect(lambda _slot_id: self.dock.hide_mentor_card())
        card = self.dock.mentor_card
        card.regimeLaneChanged.connect(self._on_regime_lane_changed)
        card.answered.connect(self.trade_mentor_service.mark_answered)
        card.skipped.connect(
            lambda record: self.trade_mentor_service.mark_skipped(
                str(record.get("slot_id") or ""), str(record.get("skipped_reason") or "")
            )
        )
        self._econViewLoaded.connect(self._on_econ_view)
        from PySide6.QtCore import QTimer

        self._econ_timer = QTimer(self)
        self._econ_timer.setInterval(ECON_REFRESH_MS)
        self._econ_timer.timeout.connect(self.refresh_econ)

    # -- MentorHostMixin hooks ----------------------------------------------
    def _mentor_review(self):
        return self.dock

    def _now(self) -> datetime:
        if self._clock is not None:
            return self._clock()
        from ui.services.econ_reminder_service import EASTERN

        return datetime.now(EASTERN)

    def _econ_morning_has_started(self) -> bool:
        from ui.services.econ_reminder_service import morning_has_started

        return morning_has_started(self._now())

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self.trade_mentor_service.start()
        self._econ_timer.start()
        self.refresh_econ()

    def shutdown(self) -> None:
        self._econ_timer.stop()
        for name, call in (
            ("trade mentor service", lambda: self.trade_mentor_service.shutdown()),
            ("trade mentor context service", lambda: self.trade_mentor_context_service.shutdown(timeout_ms=250)),
            ("journal importer", lambda: self._journal_importer is not None and self._journal_importer.shutdown()),
        ):
            try:
                call()
            except Exception:  # noqa: BLE001 - shutdown never raises
                logging.debug("Trade Mentor app: %s shutdown failed.", name, exc_info=True)

    # -- prompts --------------------------------------------------------------
    def _on_prompt_due(self, slot) -> None:
        self._show_trade_mentor_prompt(slot)
        self.cardShown.emit(slot)

    def give_a_read(self) -> None:
        self.dock.give_a_read()

    def pause_today(self) -> str:
        paused = self.trade_mentor_service.pause_today()
        self.dock.hide_mentor_card()
        return paused

    # -- the morning econ block (view only: the desk keeps the T-30/T-0 warnings) --
    def refresh_econ(self) -> None:
        from ui.services.econ_reminder_service import _default_loader, session_for

        session = session_for(self._now())
        if not session:
            return
        loader = self._econ_loader or _default_loader

        def work() -> None:
            try:
                view = dict(loader(session) or {})
            except Exception:  # noqa: BLE001 - a failed read shows nothing new
                logging.debug("Econ view could not be built.", exc_info=True)
                return
            if view.get("session"):
                self._econViewLoaded.emit(view)

        threading.Thread(target=work, name="mentor-econ-view", daemon=True).start()
