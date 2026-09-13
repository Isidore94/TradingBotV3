"""The one scheduler behind every Trade Mentor prompt - WISHLIST 10J, WS-TM item 2.

`trade_mentor_schedule` says WHICH instants a session has. This says whether the
trader is asked at one, and it is the only thing that decides: one `QTimer`, one
state file, one place where "was this hour already asked?" is answered. A second
timer somewhere else would be a second opinion about the same hour, and the
symptom of two opinions is two cards.

**A prompt is a slot, and a slot is a RECORD.** `trade_mentor_slots.json` keys
one record per `slot_id` with `delivered_at`, `answered_at` and
`skipped_reason`. Everything the brief asks for falls out of that:

* a desk restart mid-hour re-shows the card (the hour is still open and nobody
  answered) and does NOT write a second record or move `delivered_at`;
* a timer that fired late, or a clock the OS corrected, cannot duplicate an
  hour, because the identity is the wall-clock slot and not the tick;
* a missed hour is `skipped` with the reason it was missed and is never asked
  again - no queue, no catch-up burst on return;
* an unanswered prompt expires one hour after it was scheduled, which on a
  normal session is exactly when the next one arrives.

**Four absences, four reasons, never one boolean.** `away` / `paused` /
`locked` / `idle` are recorded distinctly because the coverage gap they leave in
the Market Journal has to be readable months later: "the trader was away that
morning" and "the trader paused the mentor that day" are different facts.

**Independent of the scanner's Auto setting.** Auto OFF still prompts; the
Mentor checkbox OFF prompts nothing whatever Auto says. The only thing Auto
contributes is AWAY, which means the trader is not at the desk to be asked.

Nothing here pushes to the phone, reaches a detector, a score, a gate, an alert,
a watchlist, Focus, the review queue or `review_policy.json` (plan.md sec 5).
Everything it does on the Qt thread is a small JSON read and a small JSON write.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal

from trade_mentor_schedule import PACIFIC, MentorSlot, slots_for_session

#: How long the trader may be away from the keyboard and still count as present.
#: A trader reading a chart types nothing for minutes at a time, so this is
#: deliberately generous: the cost of asking someone who is there is a card they
#: dismiss, and the cost of not asking is a hole in the record.
IDLE_GRACE_MINUTES = 20

#: One tick a minute. The schedule's grain is an hour; a faster timer would only
#: buy seconds of punctuality and spend them on the Qt thread.
POLL_INTERVAL_MS = 60_000

SKIP_AWAY = "away"
SKIP_PAUSED = "paused"
SKIP_LOCKED = "locked"
SKIP_IDLE = "idle"
SKIP_EXPIRED = "expired"
#: Presence could not be established at all (an injected probe raised). Recorded
#: as its own reason rather than folded into `idle`, so a bug in a probe never
#: reads later as "the trader was away".
SKIP_NOT_PRESENT = "not_present"

_EMPTY_RECORD = {"delivered_at": "", "answered_at": "", "skipped_reason": ""}


class TradeMentorService(QObject):
    """Owns the Trade Mentor's timer and its slot state. One per process."""

    #: (MentorSlot) - ask this now. Fired once per service instance per slot: a
    #: restart re-shows an open hour, a tick inside one never re-asks it.
    promptDue = Signal(object)
    #: (slot_id) - this hour ran out unanswered. Fired once, not once per tick.
    promptExpired = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        clock: Callable[[], datetime] | None = None,
        idle_seconds: Callable[[], float | None] | None = None,
        session_locked: Callable[[], bool] | None = None,
        state_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock or self._default_clock
        if idle_seconds is None or session_locked is None:
            import user_presence

            idle_seconds = idle_seconds or user_presence.idle_seconds
            session_locked = session_locked or user_presence.session_locked
        self._idle_seconds = idle_seconds
        self._session_locked = session_locked
        if state_path is None:
            from project_paths import TRADE_MENTOR_SLOTS_FILE

            state_path = TRADE_MENTOR_SLOTS_FILE
        self._state_path = Path(state_path)
        self._slots: dict[str, dict[str, str]] = {}
        self._paused_date = ""
        self._load()
        # Delivered ONCE per instance. Persisted `delivered_at` says the card
        # was put up at some point; this says this process has already put it
        # up, which is the question a 60-second tick asks.
        self._emitted: set[str] = set()
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # -- lifecycle --------------------------------------------------------
    @staticmethod
    def _default_clock() -> datetime:
        return datetime.now(PACIFIC)

    def start(self) -> None:
        """Begin polling. Called after the window is up, never in a constructor."""
        if not self._timer.isActive():
            self._timer.start()

    def shutdown(self) -> None:
        try:
            self._timer.stop()
        except RuntimeError:  # pragma: no cover - already torn down
            pass

    def _on_tick(self) -> None:
        try:
            self.poll()
        except Exception:  # noqa: BLE001 - a timer slot never raises into Qt
            logging.debug("Trade Mentor poll failed.", exc_info=True)

    # -- state ------------------------------------------------------------
    def _load(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        slots = payload.get("slots")
        if isinstance(slots, dict):
            for slot_id, record in slots.items():
                if isinstance(record, dict):
                    self._slots[str(slot_id)] = {
                        key: str(record.get(key) or "") for key in _EMPTY_RECORD
                    }
        self._paused_date = str(payload.get("paused_date") or "")

    def _save(self) -> None:
        """Persist, and never let a failed persist cost the prompt.

        The record is evidence about what the desk asked, and evidence stores
        are never allowed to cost the thing they record (plan.md sec 5). A
        disk that refuses the write leaves the trader with a card they can
        still answer; the journal entry is the durable half.
        """
        payload = {"slots": self._slots, "paused_date": self._paused_date}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError:
            logging.debug("Trade Mentor slot state not saved.", exc_info=True)

    def _record(self, slot_id: str) -> dict[str, str]:
        return self._slots.setdefault(str(slot_id), dict(_EMPTY_RECORD))

    def slot_state(self, slot_id: str) -> dict[str, str]:
        """What happened to one slot. An unknown slot reads as an empty record."""
        return dict(self._slots.get(str(slot_id), _EMPTY_RECORD))

    def mark_answered(self, slot_id: str, *, now: datetime | None = None) -> None:
        """The trader answered this prompt; it is never shown again."""
        record = self._record(slot_id)
        record["answered_at"] = self._local(now or self._clock()).isoformat()
        record["skipped_reason"] = ""
        self._save()

    def mark_skipped(self, slot_id: str, reason: str, *, now: datetime | None = None) -> None:
        """The trader (or an absence) ended this prompt without an answer."""
        record = self._record(slot_id)
        if record["answered_at"]:
            return
        record["skipped_reason"] = str(reason or "").strip() or SKIP_NOT_PRESENT
        self._save()

    def pause_today(self, *, now: datetime | None = None) -> str:
        """Silence the rest of TODAY. Tomorrow is unaffected.

        A day, not a switch: the trader who is in a meeting all afternoon should
        not have to remember to turn the feature back on tomorrow.
        """
        self._paused_date = self._local(now or self._clock()).date().isoformat()
        self._save()
        return self._paused_date

    def resume_today(self) -> None:
        self._paused_date = ""
        self._save()

    def is_paused(self, now: datetime | None = None) -> bool:
        moment = self._local(now or self._clock())
        return bool(self._paused_date) and self._paused_date == moment.date().isoformat()

    # -- the poll ---------------------------------------------------------
    @staticmethod
    def _local(moment: datetime) -> datetime:
        stamp = moment if moment.tzinfo else moment.astimezone()
        return stamp.astimezone(PACIFIC)

    def enabled(self) -> bool:
        """The Settings checkbox, read at POLL time.

        Read now rather than cached at construction so that turning it off stops
        the next prompt rather than the next desk restart.
        """
        try:
            from ui.state import UiState

            return bool(UiState.load().trade_mentor_enabled)
        except Exception:  # noqa: BLE001 - an unreadable setting asks nothing
            logging.debug("Trade Mentor setting unreadable.", exc_info=True)
            return False

    def _absence_reason(self, now: datetime) -> str:
        """Why the trader cannot be asked right now, or "" if they can."""
        try:
            from autopilot_core import read_auto_pilot_mode

            if str(read_auto_pilot_mode() or "").upper() == "AWAY":
                return SKIP_AWAY
        except Exception:  # noqa: BLE001 - an unreadable mode is not an absence
            logging.debug("Auto mode unreadable for the Trade Mentor.", exc_info=True)
        if self.is_paused(now):
            return SKIP_PAUSED
        try:
            if bool(self._session_locked()):
                return SKIP_LOCKED
        except Exception:  # noqa: BLE001
            logging.debug("Lock probe failed.", exc_info=True)
            return SKIP_NOT_PRESENT
        try:
            idle = self._idle_seconds()
        except Exception:  # noqa: BLE001
            logging.debug("Idle probe failed.", exc_info=True)
            return SKIP_NOT_PRESENT
        # `None` is "unmeasurable", which is uncertainty and never absence
        # (plan.md sec 5): off Windows the feature keeps working.
        if idle is not None and float(idle) > IDLE_GRACE_MINUTES * 60:
            return SKIP_IDLE
        return ""

    def slots_now(self, now: datetime | None = None) -> tuple[MentorSlot, ...]:
        return slots_for_session(self._local(now or self._clock()).date())

    def current_slot(self, now: datetime | None = None) -> MentorSlot | None:
        """The slot whose hour contains `now`, if any. At most one, by design."""
        moment = self._local(now or self._clock())
        for slot in self.slots_now(moment):
            if slot.scheduled_at <= moment < slot.expires_at:
                return slot
        return None

    def next_prompt_at(self, now: datetime | None = None) -> datetime | None:
        """The next prompt instant TODAY, or None.

        Deliberately today-only. A holiday evening answering "Monday 07:00"
        would put a time on a Settings line for a prompt three days away, which
        reads as "it is about to ask" - and the honest answer on a day with
        nothing left is that there is nothing left.
        """
        moment = self._local(now or self._clock())
        for slot in self.slots_now(moment):
            if slot.scheduled_at > moment:
                return slot.scheduled_at
        return None

    def poll(self) -> MentorSlot | None:
        """One decision for this moment. Returns the slot delivered, if any."""
        if not self.enabled():
            return None
        moment = self._local(self._clock())
        slots = self.slots_now(moment)
        if not slots:
            return None

        dirty = False
        # Expire first, so the hour that just ended is closed before the hour
        # that just started is opened. Only a DELIVERED prompt can expire: one
        # the trader never saw was already recorded as skipped, with a reason
        # worth more than "expired".
        for slot in slots:
            record = self._slots.get(slot.slot_id)
            if not record or not record["delivered_at"]:
                continue
            if record["answered_at"] or record["skipped_reason"]:
                continue
            if moment >= slot.expires_at:
                record["skipped_reason"] = SKIP_EXPIRED
                dirty = True
                self.promptExpired.emit(slot.slot_id)

        current = None
        for slot in slots:
            if slot.scheduled_at <= moment < slot.expires_at:
                current = slot
                break
        if current is None:
            if dirty:
                self._save()
            return None

        record = self._record(current.slot_id)
        if record["answered_at"] or record["skipped_reason"]:
            if dirty:
                self._save()
            return None

        reason = self._absence_reason(moment)
        if reason:
            # Recorded, never queued. The hour the trader missed stays missed:
            # a catch-up burst at 11:00 asks four questions about four tapes
            # that are gone.
            record["skipped_reason"] = reason
            self._save()
            return None

        if not record["delivered_at"]:
            record["delivered_at"] = moment.isoformat()
            dirty = True
        if dirty:
            self._save()
        if current.slot_id in self._emitted:
            return None
        self._emitted.add(current.slot_id)
        self.promptDue.emit(current)
        return current
