"""Econ warnings from the pasted brief, and the Mentor's morning econ view.

One owner: one timer, one fired-keys file, one worker for the view. For each
event today with a known ET time at or after 07:00 Pacific it warns twice: 30
minutes before ("In 30 min: ...") and at the time ("Now: ..."). Events at the
same time share one warning.

* The desk always gets the warning (`reminderFired` -> toast + sound).
* AWAY / EVENING also send it to the phone (engine machine only). DESK and
  OFF never push.
* Fired keys are saved per date, so a restart never re-fires one. A warning
  whose moment passed more than `LATE_GRACE` ago is dropped, not sent late.
* The view (`econ_brief.today_view`) is read on a worker, after a brief is
  pasted, and every `REFRESH_MS`. The timer tick only compares times.

Times come from `econ_events` (a fixed parser), never a model.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from PySide6.QtCore import QObject, QTimer, Signal
from swallowed import note_swallowed

EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

TICK_MS = 20_000
REFRESH_MS = 15 * 60_000
LEAD = timedelta(minutes=30)
LATE_GRACE = timedelta(minutes=2)

STAGE_SOON = "t30"
STAGE_NOW = "t0"

#: The morning econ block is not popped before this hour on the trader's own
#: (Pacific) clock.
MORNING_START_PT = 5

#: Modes that send the warning to the phone. DESK and OFF are at the desk.
PHONE_MODES = ("AWAY", "EVENING")

STATUS_SENT = "sent"
STATUS_DROPPED = "dropped"


def _face(time_et: str) -> str:
    """"13:00" -> "1:00 p.m." for a message a person reads."""
    hour, minute = int(time_et[:2]), int(time_et[3:])
    suffix = "a.m." if hour < 12 else "p.m."
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"


def _pacific_face(day: str, time_et: str) -> str:
    moment = _et_moment(day, time_et)
    if moment is None:
        return ""
    local = moment.astimezone(PACIFIC)
    return _face(f"{local.hour:02d}:{local.minute:02d}")


def _et_moment(day: str, time_et: str) -> datetime | None:
    try:
        hour, minute = int(time_et[:2]), int(time_et[3:])
        base = datetime.fromisoformat(day)
    except (TypeError, ValueError):
        return None
    return datetime(base.year, base.month, base.day, hour, minute, tzinfo=EASTERN)


def plan_reminders(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Two warnings per ET time today at or after 07:00 Pacific, in time order."""
    import econ_events

    session = str(view.get("session") or "")
    grouped: dict[str, list[str]] = {}
    for row in view.get("today") or ():
        time_et = str(row.get("time_et") or "")
        if str(row.get("date") or "") != session or not time_et:
            continue
        if not econ_events.alarm_allowed(session, time_et):
            continue
        label = str(row.get("label") or "").strip()
        if label and label not in grouped.setdefault(time_et, []):
            grouped[time_et].append(label)
    out: list[dict[str, Any]] = []
    for time_et in sorted(grouped):
        at = _et_moment(session, time_et)
        if at is None:
            continue
        labels = "; ".join(grouped[time_et])
        clock = f"{_face(time_et)} ET / {_pacific_face(session, time_et)} PT"
        out.append({
            "key": f"{session}|{time_et}|{STAGE_SOON}",
            "at": at - LEAD,
            "stage": STAGE_SOON,
            "title": "Econ in 30 min",
            "message": f"In 30 min: {labels} ({clock})",
        })
        out.append({
            "key": f"{session}|{time_et}|{STAGE_NOW}",
            "at": at,
            "stage": STAGE_NOW,
            "title": "Econ now",
            "message": f"Now: {labels} ({clock})",
        })
    out.sort(key=lambda item: item["at"])
    return out


def session_for(now: datetime) -> str:
    """Today's ET date when it is a trading session, else ""."""
    day = now.astimezone(EASTERN).date()
    try:
        from market_calendar import is_session

        return day.isoformat() if is_session(day) else ""
    except Exception:  # noqa: BLE001 - outside the calendar: no warnings
        return ""


class EconReminderService(QObject):
    """Owns the econ warnings and the Mentor's econ view. One per process."""

    #: (dict) a fresh `econ_brief.today_view`, delivered on the Qt thread.
    viewChanged = Signal(dict)
    #: (dict) {title, message, stage, key, phone, push_ok} - show it on the desk.
    reminderFired = Signal(dict)
    #: Internal: a worker hands the view to the Qt thread.
    _viewLoaded = Signal(dict)

    def __init__(
        self,
        parent=None,
        *,
        engine_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
        loader: Callable[[str], Mapping[str, Any]] | None = None,
        push: Callable[..., Mapping[str, Any]] | None = None,
        state_path: Path | None = None,
        mode: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine_enabled = bool(engine_enabled)
        self._clock = clock or (lambda: datetime.now(EASTERN))
        self._loader = loader
        self._push = push
        if state_path is None:
            from project_paths import ECON_REMINDERS_FILE

            state_path = ECON_REMINDERS_FILE
        self._state_path = Path(state_path)
        self._mode = str(mode).upper() if mode is not None else ""
        self._view: dict[str, Any] = {}
        self._plan: list[dict[str, Any]] = []
        self._fired_date = ""
        self._fired: dict[str, str] = {}
        self._load_fired()
        self._loading = False
        self._reload_wanted = False
        self._push_queue: queue.Queue = queue.Queue()
        self._push_thread: threading.Thread | None = None
        self._push_lock = threading.Lock()
        #: True from a sender's start until it leaves on an empty queue (under the lock).
        self._push_running = False
        self._viewLoaded.connect(self.apply_view)
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(TICK_MS)
        self._tick_timer.timeout.connect(self._on_tick)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(REFRESH_MS)
        self._refresh_timer.timeout.connect(self.refresh)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Begin. Called after the window is up, never in a constructor. Idempotent."""
        if self._tick_timer.isActive():
            return
        if not self._mode:
            self._mode = self._read_mode()
        self._tick_timer.start()
        self._refresh_timer.start()
        self.refresh()

    def shutdown(self) -> None:
        for timer in (self._tick_timer, self._refresh_timer):
            try:
                timer.stop()
            except RuntimeError as exc:  # pragma: no cover - already torn down
                note_swallowed("econ reminder timer already torn down", exc, quiet=True)

    # -- mode -------------------------------------------------------------
    @staticmethod
    def _read_mode() -> str:
        try:
            import autopilot_core as core

            return str(core.read_auto_pilot_mode() or "OFF").upper()
        except Exception:  # noqa: BLE001 - unknown mode reads OFF: desk only
            return "OFF"

    def set_auto_mode(self, mode: str) -> None:
        self._mode = str(mode or "").strip().upper() or "OFF"

    def on_auto_mode_changed(self, previous: str, current: str) -> None:
        """Slot for `AutopilotService.autoModeChanged`. Back at the desk: re-read the view."""
        self.set_auto_mode(current)
        if str(previous or "").upper() in PHONE_MODES and not self.phone_mode():
            self.refresh()

    def morning_has_started(self, now: datetime | None = None) -> bool:
        """At or after `MORNING_START_PT` on the trader's clock, on the ET session's date.

        The date check keeps a late Pacific evening (already tomorrow in ET)
        from counting as tomorrow's morning.
        """
        moment = now or self._clock()
        local = moment.astimezone(PACIFIC)
        return local.date() == moment.astimezone(EASTERN).date() and local.hour >= MORNING_START_PT

    def phone_mode(self) -> bool:
        return self._mode in PHONE_MODES

    # -- the view ---------------------------------------------------------
    def view(self) -> dict[str, Any]:
        return dict(self._view)

    def refresh(self) -> None:
        """Rebuild the view on a worker. Single-flight; a request mid-load reruns once."""
        if self._loading:
            self._reload_wanted = True
            return
        session = session_for(self._clock())
        if not session:
            return
        self._loading = True
        loader = self._loader or _default_loader

        def _work() -> None:
            try:
                view = dict(loader(session) or {})
            except Exception:  # noqa: BLE001 - a failed read keeps the last view
                logging.debug("Econ view could not be built.", exc_info=True)
                view = {}
            view["_loaded"] = True
            self._viewLoaded.emit(view)

        threading.Thread(target=_work, name="econ-view", daemon=True).start()

    def on_journal_entry(self, entry: Mapping[str, Any]) -> None:
        """Slot for `MarketJournalService.entryWritten`: a pasted brief reschedules."""
        try:
            import market_journal

            if str((entry or {}).get("origin") or "") == market_journal.ORIGIN_EXTERNAL_FORECAST:
                self.refresh()
        except Exception:  # noqa: BLE001 - a slot never raises into Qt
            logging.debug("Econ refresh after paste failed.", exc_info=True)

    def apply_view(self, view: Mapping[str, Any]) -> None:
        """Take a new view and reschedule. Qt thread; cheap."""
        from_worker = bool(view.get("_loaded"))
        payload = {key: value for key, value in dict(view).items() if key != "_loaded"}
        if from_worker:
            self._loading = False
        if payload.get("session"):
            self._view = payload
            self._plan = plan_reminders(payload)
            self.viewChanged.emit(dict(payload))
            self.tick()
        if from_worker and self._reload_wanted:
            self._reload_wanted = False
            self.refresh()

    def planned(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._plan]

    # -- firing -----------------------------------------------------------
    def _on_tick(self) -> None:
        try:
            self.tick()
        except Exception:  # noqa: BLE001 - a timer slot never raises into Qt
            logging.debug("Econ reminder tick failed.", exc_info=True)

    def tick(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Fire what is due now; drop what is too late. Returns what fired."""
        moment = now or self._clock()
        today = moment.astimezone(EASTERN).date().isoformat()
        if self._fired_date != today:
            self._fired_date, self._fired = today, {}
        fired: list[dict[str, Any]] = []
        changed = False
        for item in self._plan:
            key = item["key"]
            if key in self._fired or not key.startswith(today) or moment < item["at"]:
                continue
            if moment - item["at"] > LATE_GRACE:
                self._fired[key] = STATUS_DROPPED
                changed = True
                continue
            self._fired[key] = STATUS_SENT
            changed = True
            fired.append(self._fire(item))
        if changed:
            self._save_fired()
        return fired

    def _fire(self, item: Mapping[str, Any]) -> dict[str, Any]:
        phone = self.phone_mode()
        payload = {
            "key": item["key"],
            "stage": item["stage"],
            "title": item["title"],
            "message": item["message"],
            "phone": phone,
            "mode": self._mode or "OFF",
        }
        if phone and self.engine_enabled:
            self._queue_push(str(item["title"]), str(item["message"]))
        logging.info("ECON REMINDER %s (%s)", item["message"], "phone" if phone else "desk")
        try:
            self.reminderFired.emit(dict(payload))
        except Exception:  # noqa: BLE001 - a broken desk path never stops the phone
            logging.debug("Econ reminder display failed.", exc_info=True)
        return payload

    def _queue_push(self, title: str, message: str) -> None:
        """One sender thread drains the queue; a slow ntfy never stacks threads.

        The put and the "is a sender running?" check share `_push_lock` with
        the sender's own empty-queue exit, so a push can never land between
        the sender finding the queue empty and the sender stopping.
        """
        with self._push_lock:
            self._push_queue.put((title, message))
            if self._push_running:
                return
            self._push_running = True
            self._push_thread = threading.Thread(
                target=self._drain_pushes, name="econ-push", daemon=True
            )
            self._push_thread.start()

    def _drain_pushes(self) -> None:
        send = self._push
        if send is None:
            import push_notify

            send = push_notify.send_push
        while True:
            with self._push_lock:
                try:
                    title, message = self._push_queue.get_nowait()
                except queue.Empty:
                    self._push_running = False
                    return
            try:
                result = send(title, message, priority="high", tags="calendar")
                if not (result or {}).get("ok"):
                    logging.info("Econ push not sent: %s", (result or {}).get("error") or "not configured")
            except Exception:  # noqa: BLE001 - send_push does not raise; a transport might
                logging.warning("Econ push failed.", exc_info=True)

    def wait_for_pushes(self, timeout: float = 2.0) -> None:
        thread = self._push_thread
        if thread is not None:
            thread.join(timeout)

    # -- fired keys -------------------------------------------------------
    def _load_fired(self) -> None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(payload, dict) and isinstance(payload.get("fired"), dict):
            self._fired_date = str(payload.get("date") or "")
            self._fired = {str(k): str(v) for k, v in payload["fired"].items()}

    def _save_fired(self) -> None:
        payload = {"date": self._fired_date, "fired": dict(self._fired)}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError:
            logging.debug("Econ reminder state not saved.", exc_info=True)

    def fired(self) -> dict[str, str]:
        return dict(self._fired)


def _default_loader(session: str) -> Mapping[str, Any]:
    import econ_brief

    return econ_brief.today_view(session)
