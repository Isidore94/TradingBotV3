from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from market_environment_annotations import record_market_environment_annotation
from project_paths import MARKET_ENVIRONMENT_ANNOTATIONS_FILE
from technical_integrity import load_technical_integrity_snapshot
from ui.models.bounce import BounceAlert
from ui.timer_utils import start_staggered, stop_staggered

try:  # pragma: no cover - shiboken6 ships with PySide6; guard for odd builds
    from shiboken6 import isValid as _shiboken_is_valid
except Exception:  # pragma: no cover
    _shiboken_is_valid = None


def _qobject_is_alive(obj: QObject) -> bool:
    """True when ``obj``'s C++ half still exists.

    The Python wrapper outlives the C++ QObject, so a worker thread can hold a
    perfectly valid Python reference to a service whose Qt half was already
    destroyed.  Emitting through that wrapper raises
    ``RuntimeError: Signal source has been deleted``.
    """

    if _shiboken_is_valid is None:  # pragma: no cover - fallback path
        return True
    try:
        return bool(_shiboken_is_valid(obj))
    except Exception:  # pragma: no cover - defensive
        return False


class _Liveness:
    """Retirement flag that outlives the QObject it belongs to.

    Kept off the QObject on purpose: worker threads capture *this* object, so
    they can ask "may I still signal?" without touching a wrapper whose C++ half
    may already be gone.  It is latched by ``shutdown()`` and by any code that
    observes the deletion first-hand; it is never wired to Qt's ``destroyed``
    signal, because that slot can fire while this object is itself being
    finalized.
    """

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        self._closed = threading.Event()

    def close(self, *_args: Any) -> None:
        try:
            self._closed.set()
        except AttributeError:  # pragma: no cover - interpreter teardown
            pass

    @property
    def closed(self) -> bool:
        try:
            return self._closed.is_set()
        except AttributeError:  # pragma: no cover - interpreter teardown
            return True


class _StartupSession:
    """One start()->stop() generation of BounceBot ownership.

    It is created by ``start()``, owns the ``qt-bouncebot-start`` thread, and -
    if that thread installs a bot - stays on as that bot's owner token until
    ``stop()``.  ``cancelled`` is the shutdown token: once set, the generation
    may not install a bot, may not mutate service state, and may not emit
    anything, on either the success or the failure path.  ``finished`` is set
    by the worker's ``finally`` so ``stop()`` can prove the generation retired.

    ``construction_done`` is narrower and guards a different hazard: it is set
    the moment ``run_bot_with_gui`` returns or raises.  While it is *unset* the
    generation may be sitting inside ``bot.connect(..., clientId=...)``, and a
    second generation doing the same thing would collide on this repo's
    hard-coded IB client id (Error 326) and open two live IB sessions.  ``start()``
    refuses to open a new generation until every previous one has cleared it.
    """

    __slots__ = ("generation", "cancelled", "finished", "construction_done", "thread")

    def __init__(self, generation: int) -> None:
        self.generation = int(generation)
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.construction_done = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()


#: ``what`` tags for owned threads that outlived their shutdown budget.
_WORK_STARTUP = "startup worker"
_WORK_RETIRE = "bot retirement"
_SERVICE_REFS: "weakref.WeakSet[BounceService]" = weakref.WeakSet()


def owned_bounce_thread_snapshot() -> dict[str, Any]:
    """Read-only lifecycle accounting for the in-process Health audit."""

    services = list(_SERVICE_REFS)
    overdue: list[str] = []
    running = 0
    connected = 0
    for service in services:
        try:
            overdue.extend(service.unretired_workers())
            running += int(service.running)
            connected += int(service.current_bot() is not None)
        except Exception:
            logging.exception("BounceService lifecycle snapshot failed.")
    return {
        "bounce_service_count": len(services),
        "bounce_service_running_count": running,
        "bounce_service_connected_count": connected,
        "bounce_unretired_worker_count": len(overdue),
        "bounce_unretired_workers": overdue,
    }


def load_bounce_config() -> dict[str, Any]:
    from bounce_bot import BOUNCE_TYPE_DEFAULTS, BOUNCE_TYPE_LABELS, MARKET_ENVIRONMENTS, RRS_TIMEFRAMES

    return {
        "bounce_type_defaults": dict(BOUNCE_TYPE_DEFAULTS),
        "bounce_type_labels": dict(BOUNCE_TYPE_LABELS),
        "market_environments": dict(MARKET_ENVIRONMENTS),
        "rrs_timeframes": dict(RRS_TIMEFRAMES),
    }


class BounceService(QObject):
    alertReceived = Signal(object)
    rrsStatusChanged = Signal(str)
    rrsSnapshotChanged = Signal(object)
    statusChanged = Signal(str)
    connectionChanged = Signal(str)
    activeBouncesChanged = Signal(int)
    _activeBouncesReady = Signal(object)
    _entryBoardReady = Signal(object)
    scanningChanged = Signal(bool)
    autoRegimeChanged = Signal(object)  # reading dict from get_auto_regime_reading(), or {}
    technicalIntegrityChanged = Signal(object)  # advisory completed-M5 hierarchy, or {}
    entryAssistChanged = Signal(object)  # state dict from entry_assist_state(), or {}
    entryBoardChanged = Signal(object)  # board dict from entry_assist_board_snapshot(), or {}
    started = Signal()
    stopped = Signal()
    failed = Signal(str)

    #: Bounded wall-clock budget ``stop()`` spends joining an in-flight startup
    #: worker.  Exceeding it is not a leak: the generation is already cancelled,
    #: so the worker retires itself (closing any late bot) without emitting.
    STARTUP_RETIRE_TIMEOUT = 3.0

    #: TOTAL wall-clock budget ``stop()`` may spend on its calling (GUI) thread,
    #: covering bot retirement *and* the startup join.  ``BounceBot.stop`` joins
    #: the strategy and API threads at ``BOT_STOP_TIMEOUT`` each, so retiring a
    #: bot inline used to be able to freeze the desk for ~10s on Stop or window
    #: close; retirement now runs on its own thread inside this budget.
    STOP_BUDGET = 3.0

    #: Extra budget ``shutdown()`` spends making a final attempt on anything
    #: ``stop()`` could not retire, before it latches the service closed.
    SHUTDOWN_BUDGET = 2.0

    #: What we ask ``BounceBot.stop()`` to spend on its own internal joins.
    BOT_STOP_TIMEOUT = 5.0

    def __init__(
        self,
        parent=None,
        *,
        environment_annotations_path: Path = MARKET_ENVIRONMENT_ANNOTATIONS_FILE,
    ) -> None:
        super().__init__(parent)
        config = load_bounce_config()
        self.bounce_type_settings: dict[str, bool] = dict(config["bounce_type_defaults"])
        self.rrs_threshold = 2.0
        self.rrs_timeframe_key = "5m"
        # ``None`` is the user's N/A mode: the bot owns the regime through its
        # automatic SPY read.  A concrete value is a session-only annotation
        # and manual override; it is never silently inferred from Auto.
        self.market_environment: str | None = None
        self.environment_session_id = uuid.uuid4().hex
        self.environment_annotations_path = Path(environment_annotations_path)
        self.scanning_enabled = False
        self.include_approaching = False

        self._bot = None
        self._lock = threading.Lock()
        # Startup lifecycle ownership: exactly one component (this service)
        # owns the ``qt-bouncebot-start`` thread, and every generation of it is
        # represented by a _StartupSession that shutdown can cancel.
        self._generation = 0
        self._session: _StartupSession | None = None
        self._liveness = _Liveness()
        # Separate from QObject liveness: terminal shutdown must refuse a
        # re-entrant start *before* stop signals fire, while those signals
        # should remain observable until stop() has completed.
        self._terminal = threading.Event()
        # Owned threads that outlived their shutdown budget.  They are never
        # forgotten: they are reported (``unretired_workers``), they gate a new
        # startup, and ``shutdown()`` makes one final bounded attempt on them.
        # daemon=True is a backstop, never the only mechanism.
        self._unretired: list[tuple[threading.Thread, str, _StartupSession | None]] = []
        self._start_deferred_reported = False
        # GUI health displays the last completed worker result. The historical
        # AVWAP event ledger is tens of MB, so no health tick may parse it on
        # the Qt thread. A single-flight worker prefers the scanner's compact
        # validated projection and falls back to the ledger off-thread.
        self._active_bounces_count = 0
        self._active_bounces_signature: tuple[int, int] | None = None
        self._active_bounces_refreshing = False
        self._activeBouncesReady.connect(self._on_active_bounces_ready)
        # The entry-assist board is the same shape of problem as the active
        # bounces above: a minute timer calling a snapshot builder that walks
        # the bot's cached bars, on the Qt thread. Same answer - single-flight
        # worker, ready-signal, emitted from the GUI thread.
        self._entry_board_refreshing = False
        self._entryBoardReady.connect(self._on_entry_board_ready)
        _SERVICE_REFS.add(self)

        self._health_timer = QTimer(self)
        self._health_timer.setInterval(3000)
        self._health_timer.timeout.connect(self.refresh_health)
        self.started.connect(self._start_health_timer)

        # Always-on auto-regime readout: what auto tracking thinks right now
        # (even under a manual override), refreshed from cached SPY bars.
        self._regime_timer = QTimer(self)
        self._regime_timer.setInterval(30_000)
        self._regime_timer.timeout.connect(self.refresh_auto_regime)
        self.started.connect(self._start_regime_timer)

        # Advisory Technical Integrity is disk-backed so the UI never reaches
        # into the scan thread. The same cached snapshot feeds every page.
        self._integrity_timer = QTimer(self)
        self._integrity_timer.setInterval(30_000)
        self._integrity_timer.timeout.connect(self.refresh_technical_integrity)
        self.started.connect(self._start_integrity_timer)

        # Always-on RS/RW board: regime + pause detection + live window /
        # pause-preview rankings + both-side trailing movers, recomputed from
        # cached bars every minute with no clicks.
        self._board_timer = QTimer(self)
        self._board_timer.setInterval(60_000)
        self._board_timer.timeout.connect(self.refresh_entry_board)
        self.started.connect(self._start_board_timer)

        # Research-warehouse M5 tee (BD-20). Shadow-only and strictly additive:
        # it reads the bar cache the champion already populated, issues no
        # provider request, and writes only to the GUI-owned spool. It exists
        # only where a bot does - the main desk - and only when the trader has
        # configured a research store; otherwise it is never constructed.
        # 60s rather than the 3s health cadence: M5 bars complete every five
        # minutes, so anything faster is pure re-scanning of the same dict.
        self._warehouse_capture = None
        self._warehouse_timer = QTimer(self)
        self._warehouse_timer.setInterval(60_000)
        self._warehouse_timer.timeout.connect(self.capture_warehouse_tee)
        self.started.connect(self._start_warehouse_timer)

    # ------------------------------------------------------------------
    # Emission guards
    # ------------------------------------------------------------------
    def _emit(self, signal, *args: Any) -> bool:
        """Emit only while the Qt half of this service still exists.

        Returns True when the signal actually went out.  A deleted service is
        silent instead of raising ``RuntimeError: Signal source has been
        deleted`` on a worker thread (which pytest reports as an unhandled
        thread exception and which killed the startup worker mid-flight,
        leaking a live IB-connected bot).
        """

        if self._liveness.closed or not _qobject_is_alive(self):
            return False
        try:
            signal.emit(*args)
        except RuntimeError:
            if _qobject_is_alive(self):
                raise  # a slot raised; not our deletion race - do not swallow
            self._liveness.close()
            return False
        return True

    def _emit_for(self, session: _StartupSession, signal, *args: Any) -> bool:
        """Emit on behalf of a startup generation, unless it was cancelled."""

        if session.is_cancelled:
            return False
        return self._emit(signal, *args)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._bot is not None or self._session is not None

    # ------------------------------------------------------------------
    # Owned-thread bookkeeping (nothing this service starts is forgotten)
    # ------------------------------------------------------------------
    def _prune_unretired_locked(self) -> None:
        self._unretired = [entry for entry in self._unretired if entry[0].is_alive()]

    def _track_unretired(
        self, thread: threading.Thread | None, what: str, session: _StartupSession | None = None
    ) -> None:
        """Remember a thread that outlived its shutdown budget, and say so.

        Without this, a worker that misses the join is untracked forever:
        ``shutdown()`` cannot reach it, ``running`` reports False, and nothing
        logs it - leaving ``daemon=True`` as the only mechanism.
        """

        if thread is None or not thread.is_alive():
            return
        with self._lock:
            self._prune_unretired_locked()
            if not any(entry[0] is thread for entry in self._unretired):
                self._unretired.append((thread, what, session))
            outstanding = len(self._unretired)
        logging.warning(
            "BounceService: %s (%s) did not retire inside its budget; "
            "%d owned thread(s) still outstanding.",
            what,
            thread.name,
            outstanding,
        )

    def unretired_workers(self) -> list[str]:
        """Owned threads still running past their shutdown budget.

        This is the honest answer to the Health page's owned-thread question;
        an empty list means every thread this service ever started is joined.
        """

        with self._lock:
            self._prune_unretired_locked()
            return [f"{what} ({thread.name})" for thread, what, _ in self._unretired]

    def _startup_blocker_locked(self) -> str | None:
        """Name of an overdue worker that makes another IB connect unsafe.

        A startup worker remains a blocker until it exits, not merely until
        ``run_bot_with_gui`` returns: after construction it owns a connected
        bot while applying/synchronizing state.  A retirement worker also
        remains a blocker until ``BounceBot.stop`` has disconnected and joined
        its threads.  Starting across either window can reuse the same IB
        client id while the prior session is still live.
        """

        self._prune_unretired_locked()
        for thread, what, session in self._unretired:
            if thread.is_alive() and what in {_WORK_STARTUP, _WORK_RETIRE}:
                return f"{what} {thread.name}"
        return None

    def start(self) -> bool:
        """Open a startup generation.  True when one was actually opened.

        Returns False when the service is terminal, already running, or a
        previous startup/retirement worker still may own the hard-coded IB
        client id.  That includes post-connect state sync and disconnect/join,
        not only ``run_bot_with_gui`` itself: opening across either window can
        produce Error 326 or two live sessions.  Every blocked attempt emits an
        actionable status; it self-heals when the owner exits, while a worker
        that never exits honestly requires an app restart.
        """

        if self._terminal.is_set() or self._liveness.closed:
            return False  # retired service: never resurrect a startup worker
        with self._lock:
            if self._bot is not None or self._session is not None:
                return False
            blocked_by = self._startup_blocker_locked()
            if blocked_by is None:
                self._generation += 1
                session = _StartupSession(self._generation)
                session.thread = threading.Thread(
                    target=self._start_worker,
                    args=(session,),
                    name="qt-bouncebot-start",
                    daemon=True,
                )
                self._session = session
                self._start_deferred_reported = False
            else:
                first_report = not self._start_deferred_reported
                self._start_deferred_reported = True
        if blocked_by is not None:
            if first_report:
                logging.warning(
                    "BounceService.start deferred: %s has not retired; a second IB connect "
                    "would collide on the same client id.",
                    blocked_by,
                )
            # Every manual click gets truthful feedback.  Suppressing this
            # after the first refusal made a permanently blocked IB connect
            # look like a dead Start button.  Auto Pilot already deduplicates
            # its own activity-log line.
            self._emit(
                self.statusChanged,
                f"start deferred: {blocked_by} has not retired; retry shortly, "
                "or restart the app if it does not clear",
            )
            return False
        self._emit(self.connectionChanged, "IB: connecting")
        with self._lock:
            may_continue = (
                self._session is session
                and not session.is_cancelled
                and not self._terminal.is_set()
                and not self._liveness.closed
            )
        if not may_continue:
            session.construction_done.set()
            session.finished.set()
            return False
        self._emit(self.statusChanged, "connecting")
        # Either signal can run a direct-connected slot synchronously.  A slot
        # is allowed to stop or terminally shut down the service; do not start
        # the parked worker afterwards in that case.
        with self._lock:
            may_start = (
                self._session is session
                and not session.is_cancelled
                and not self._terminal.is_set()
                and not self._liveness.closed
            )
        if not may_start:
            session.construction_done.set()
            session.finished.set()
            return False
        session.thread.start()
        return True

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        deadline = time.monotonic() + max(0.0, float(self.STOP_BUDGET))
        with self._lock:
            bot = self._bot
            self._bot = None
            session = self._session
            self._session = None
            # Bump the generation so anything still in flight is stale by
            # identity as well as by its cancellation flag.
            self._generation += 1
        if session is not None:
            session.cancelled.set()
        if bot is not None:
            # Cooperative shutdown: disconnect alone left the strategy loop
            # alive and auto-reconnecting (plan.md Packet A).
            self._retire_bot_bounded(bot, deadline)
        # Establish the overdue-worker blocker before any signal below can
        # synchronously cause a restart.  Emitting ``stopped`` first briefly
        # reopened the exact two-startup race this lifecycle guard prevents.
        if session is not None:
            remaining = max(0.0, deadline - time.monotonic())
            budget = min(float(self.STARTUP_RETIRE_TIMEOUT), remaining)
            if not self._await_startup_retired(session, timeout=budget):
                self._track_unretired(session.thread, _WORK_STARTUP, session)
        # A RuntimeError here is re-raised (see _stop_timers), but only after the
        # rest of stop() has run: the desk must still be told it stopped and the
        # startup worker must still be joined/tracked.
        timer_error: RuntimeError | None = None
        try:
            self._stop_timers()
        except RuntimeError as exc:
            timer_error = exc
        self._emit(self.autoRegimeChanged, {})
        self._emit(self.entryAssistChanged, {})
        self._emit(self.entryBoardChanged, {})
        self._emit(self.connectionChanged, "IB: disconnected")
        self._emit(self.activeBouncesChanged, 0)
        self._emit(self.statusChanged, "stopped")
        self._emit(self.stopped)
        if timer_error is not None:
            raise timer_error

    def shutdown(self) -> None:
        """Terminal retirement: stop, then forbid any further emit or start.

        ``stop()`` is reversible (the desk can start BounceBot again).  This is
        not: it is what a closing window / a dying service calls so that a
        worker which outlives the QObject can never signal through it.  It also
        makes one final bounded attempt on anything ``stop()`` could not retire,
        and reports whatever is still outstanding.
        """

        # Refuse re-entrant starts before the direct-connected stop signals
        # fire, but keep the QObject observably alive until stop() and the final
        # bounded joins finish.
        self._terminal.set()
        try:
            self.stop()
        finally:
            deadline = time.monotonic() + max(0.0, float(self.SHUTDOWN_BUDGET))
            with self._lock:
                self._prune_unretired_locked()
                pending = list(self._unretired)
            for thread, _what, _session in pending:
                if thread is threading.current_thread():
                    continue
                thread.join(max(0.0, deadline - time.monotonic()))
            outstanding = self.unretired_workers()
            if outstanding:
                logging.warning(
                    "BounceService.shutdown: %d owned thread(s) still running: %s",
                    len(outstanding),
                    ", ".join(outstanding),
                )
            self._liveness.close()

    def _stop_timers(self) -> None:
        """Stop every timer this service owns.

        A RuntimeError here means one of two very different things.  If the Qt
        half is gone the service is genuinely retired and latching liveness is
        right.  If the object is still ALIVE the error came from somewhere else,
        and latching would permanently kill a healthy service - ``start()``
        would return at its liveness guard and every ``_emit`` would
        short-circuit, silently, with no scanner.  So this mirrors ``_emit``:
        latch only when the C++ half is gone, otherwise re-raise (after
        best-effort stopping the remaining timers).
        """

        error: RuntimeError | None = None
        # The tee's worker is this service's to retire, like every other thread
        # it owns; it holds no Qt object, so it is stopped before the timers.
        self._close_warehouse_capture()
        for timer in (
            self._health_timer,
            self._regime_timer,
            self._integrity_timer,
            self._board_timer,
            self._warehouse_timer,
        ):
            try:
                stop_staggered(timer)
            except RuntimeError as exc:
                if _qobject_is_alive(self):
                    if error is None:
                        error = exc
                    continue
                self._liveness.close()  # the C++ side is gone; nothing left to stop
                return
        if error is not None:
            raise error

    def _await_startup_retired(self, session: _StartupSession, timeout: float | None = None) -> bool:
        """Join a cancelled startup generation inside a bounded budget.

        Bot construction is not interruptible (it can sit in an IB connect), so
        the guarantee is: either the worker is joined here, or it is already
        cancelled and will retire itself - closing the late bot and emitting
        nothing.  Never "it's a daemon thread, the process will deal with it".
        """

        thread = session.thread
        budget = self.STARTUP_RETIRE_TIMEOUT if timeout is None else float(timeout)
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(max(0.0, budget))
        return session.finished.is_set()

    def _retire_bot_bounded(self, bot, deadline: float) -> bool:
        """Retire a bot without letting it freeze the GUI thread.

        ``BounceBot.stop`` joins the strategy thread AND the API thread at
        ``BOT_STOP_TIMEOUT`` each, so calling it inline from ``stop()`` could
        block the desk for ~10s on Stop or window close.  It runs on its own
        thread here and the caller waits only for what is left of the stop
        budget; an overrun is tracked and reported, never forgotten.
        """

        worker = threading.Thread(
            target=self._retire_bot,
            args=(bot,),
            name="qt-bouncebot-retire",
            daemon=True,
        )
        worker.start()
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            self._track_unretired(worker, _WORK_RETIRE)
            return False
        return True

    def _retire_bot(self, bot) -> None:
        """Close a bot we are not going to keep (never leak an IB session).

        ``stop()`` failing is not permission to walk away: a bot that was
        neither stopped nor disconnected leaves the IB session live, which is
        the exact leak this lifecycle work exists to close.  So a failed
        ``stop()`` still falls through to ``disconnect()``, and both failures
        are logged rather than swallowed.
        """

        stopped = False
        stopper = getattr(bot, "stop", None)
        if callable(stopper):
            try:
                stopper(timeout=self.BOT_STOP_TIMEOUT)
                stopped = True
            except Exception:
                logging.exception(
                    "BounceBot.stop() failed while retiring the bot; falling back to disconnect()."
                )
        if stopped:
            return
        try:
            bot.disconnect()
        except Exception:
            logging.exception(
                "BounceBot could be neither stopped nor disconnected; the IB session may leak."
            )

    def start_scanning(self) -> None:
        self.set_scanning_enabled(True)

    def stop_scanning(self) -> None:
        self.set_scanning_enabled(False)

    def set_scanning_enabled(self, enabled: bool) -> None:
        self.scanning_enabled = bool(enabled)
        self._with_bot(lambda bot: bot.set_scanning_enabled(self.scanning_enabled))
        self._emit(self.scanningChanged, self.scanning_enabled)
        self._emit(self.statusChanged, "scanning enabled" if self.scanning_enabled else "scanning paused")

    def set_rrs_threshold(self, value: float) -> None:
        self.rrs_threshold = float(value)
        self._with_bot(lambda bot: bot.set_rrs_threshold(self.rrs_threshold))

    def set_rrs_timeframe(self, key: str) -> None:
        config = load_bounce_config()
        if key not in config["rrs_timeframes"]:
            return
        self.rrs_timeframe_key = key
        self._with_bot(lambda bot: bot.set_rrs_timeframe(key))

    def set_market_environment(self, env_key: str) -> None:
        config = load_bounce_config()
        if env_key not in config["market_environments"]:
            return
        self.market_environment = env_key
        self._with_bot(lambda bot: bot.set_market_environment(env_key))
        logged = self._record_environment_annotation(env_key, event="manual_selected")
        label = str(config["market_environments"][env_key].get("label", env_key))
        suffix = "" if logged else " (annotation log unavailable)"
        self._emit(self.statusChanged, f"User market mode: {label}; Auto still records its own read.{suffix}")

    def clear_market_environment_override(self) -> None:
        """Return regime control to the bot's SPY-based auto tracking."""
        self.market_environment = None
        self._with_bot(lambda bot: bot.clear_market_environment_override())
        logged = self._record_environment_annotation(None, event="returned_to_na")
        suffix = "" if logged else " Annotation log unavailable."
        self._emit(self.statusChanged, f"User market mode: N/A; Auto controls the active regime.{suffix}")

    def set_bounce_type_enabled(self, bounce_type: str, enabled: bool) -> None:
        if bounce_type not in self.bounce_type_settings:
            return
        self.bounce_type_settings[bounce_type] = bool(enabled)
        self._with_bot(lambda bot: bot.set_bounce_type_enabled(bounce_type, bool(enabled)))

    # ------------------------------------------------------------------
    # Delivery-time guards for queued (cross-thread) signals
    # ------------------------------------------------------------------
    def _is_live(self) -> bool:
        """False once the service is retired or its Qt half is destroyed."""

        return not self._liveness.closed and _qobject_is_alive(self)

    def _may_arm_timers(self) -> bool:
        """Re-check at DELIVERY time whether ``started`` still means anything.

        ``started`` is emitted on the startup worker, so Qt delivers it to the
        arming slots later, on the GUI thread.  A ``stop()`` landing in that
        window has already run ``_stop_timers()``; arming afterwards leaves four
        timers running forever on a service with no bot - the health timer
        emitting every 3s and the integrity timer doing disk I/O every 30s, with
        nothing left to stop them.  Gating the *emit* cannot close this: the gap
        is between emit and delivery, so the check has to live in the slot.
        """

        if not self._is_live():
            return False
        with self._lock:
            return self._bot is not None

    @Slot()
    def _start_health_timer(self) -> None:
        if not self._may_arm_timers():
            return
        self.refresh_health()
        start_staggered(self._health_timer, 4_000)

    @Slot()
    def _start_regime_timer(self) -> None:
        if not self._may_arm_timers():
            return
        self.refresh_auto_regime()
        start_staggered(self._regime_timer, 37_000)

    @Slot()
    def _start_integrity_timer(self) -> None:
        if not self._may_arm_timers():
            return
        self.refresh_technical_integrity()
        start_staggered(self._integrity_timer, 43_000)

    @Slot()
    def refresh_technical_integrity(self) -> None:
        if not self._is_live():
            return  # never do disk I/O on behalf of a retired service
        self._emit(self.technicalIntegrityChanged, load_technical_integrity_snapshot())

    @Slot()
    def _start_board_timer(self) -> None:
        if not self._may_arm_timers():
            return
        self.refresh_entry_board()
        start_staggered(self._board_timer, 79_000)

    @Slot()
    def _start_warehouse_timer(self) -> None:
        if not self._may_arm_timers():
            return
        self.capture_warehouse_tee()
        start_staggered(self._warehouse_timer, 89_000)

    @Slot()
    def capture_warehouse_tee(self) -> None:
        """Hand the champion's completed M5 bars to the research tee.

        Everything this slot does is memory: constructing the capture object
        (which touches no disk) and copying ``bot.latest_bars``. The copy must
        happen on this thread - it owns that dict, and iterating one the
        champion is resizing would raise - but the spool writer's construction,
        its stale-segment adoption, its cap enforcement and its fsync all run on
        the capture object's own worker thread (review defect D21). No provider
        request, no lake I/O, on any thread.
        """
        if not self._is_live():
            return
        bot = self._current_bot()
        if bot is None:
            return
        capture = self._warehouse_capture
        if capture is None:
            try:
                from ui.services.warehouse_service import WarehouseTeeCapture

                capture = WarehouseTeeCapture()
            except Exception:
                # The package is unavailable: the desk simply has no warehouse.
                # Never retried noisily, never fatal.
                logging.debug("Research warehouse tee unavailable; capture stays off.", exc_info=True)
                self._warehouse_timer.stop()
                return
            self._warehouse_capture = capture
        capture.submit(bot)
        if capture.disabled:
            # The worker looked and found no research store configured. Stop
            # waking a thread that will never have anything to do.
            self._warehouse_timer.stop()

    def _close_warehouse_capture(self) -> None:
        capture = self._warehouse_capture
        if capture is None:
            return
        try:
            capture.close()
        except Exception:
            logging.debug("Research warehouse tee failed to close cleanly.", exc_info=True)

    @Slot()
    def refresh_entry_board(self) -> None:
        """Recompute + emit the always-on entry-assist RS/RW board.

        The snapshot walks the bot's cached bars for every board name, and it
        used to do that on the Qt thread once a minute. It runs on a worker now,
        single-flight, and the result is emitted from the GUI thread through
        `_entryBoardReady`. `entry_assist_board_snapshot` itself is untouched -
        only who calls it moved.
        """
        if not self._is_live():
            return
        bot = self._current_bot()
        if bot is None:
            self._emit(self.entryBoardChanged, {})
            return
        if self._entry_board_refreshing:
            return
        self._entry_board_refreshing = True
        threading.Thread(
            target=self._load_entry_board_worker,
            name="qt-entry-board",
            daemon=True,
        ).start()

    def _load_entry_board_worker(self) -> None:
        board = None
        bot = self._current_bot()
        if bot is not None:
            try:
                board = bot.entry_assist_board_snapshot()
            except Exception:
                logging.debug("Entry-assist board refresh failed.", exc_info=True)
                board = None
        self._emit(self._entryBoardReady, board or {})

    @Slot(object)
    def _on_entry_board_ready(self, payload: object) -> None:
        self._entry_board_refreshing = False
        if not self._is_live():
            return
        self._emit(self.entryBoardChanged, payload if isinstance(payload, dict) else {})

    @Slot()
    def refresh_auto_regime(self) -> None:
        """Emit the bot's read-only auto-regime reading + entry-assist state."""
        if not self._is_live():
            return
        bot = self._current_bot()
        reading = None
        assist = None
        if bot is not None:
            try:
                reading = bot.get_auto_regime_reading()
            except Exception:
                reading = None
            try:
                assist = bot.entry_assist_state()
            except Exception:
                assist = None
        self._emit(self.autoRegimeChanged, reading or {})
        self._emit(self.entryAssistChanged, assist or {})

    def entry_assist(self) -> dict | None:
        """Regime-tailored window toggle / movers output (legacy single button)."""
        result = self._with_bot(lambda bot: bot.entry_assist_action())
        if isinstance(result, dict) and result.get("note"):
            self._emit(self.statusChanged, f"Entry assist: {result['note']}")
        self.refresh_auto_regime()
        return result

    def entry_assist_command(self, command: str) -> dict | None:
        """Explicit button-array action. Every click produces visible output:
        successful actions emit their lists through the bot's gui_callback, and
        failures (bot not connected, no SPY bars, window too short) surface as a
        WATCH note in the Alert Center instead of dying in the status bar."""
        bot = self._current_bot()
        if bot is None:
            self._emit_assist_note("Bot not connected yet - start BounceBot first.")
            return None
        try:
            result = bot.entry_assist_command(command)
        except Exception as exc:
            self._emit_assist_note(f"Command failed: {exc}")
            self.refresh_auto_regime()
            return None
        if isinstance(result, dict) and result.get("note"):
            self._emit(self.statusChanged, f"Entry assist: {result['note']}")
            if not result.get("ok"):
                self._emit_assist_note(str(result["note"]))
        self.refresh_auto_regime()
        self.refresh_entry_board()  # window opens/closes show on the board immediately
        return result

    def _emit_assist_note(self, note: str) -> None:
        self._emit(self.statusChanged, f"Entry assist: {note}")
        self._emit(
            self.alertReceived,
            BounceAlert(
                time_text=datetime.now().strftime("%H:%M:%S"),
                symbol="",
                side="WATCH",
                trigger=str(note),
                tag="entry_assist",
                raw_text=f"ENTRY ASSIST: {note}",
            )
        )

    @Slot()
    def refresh_health(self) -> None:
        if not self._is_live():
            return
        bot = self._current_bot()
        if bot is None:
            self._emit(self.connectionChanged, "IB: disconnected")
            self._emit(self.activeBouncesChanged, 0)
            return

        connected = bool(getattr(bot, "connection_status", False))
        self._emit(self.connectionChanged, "IB: connected" if connected else "IB: retrying")
        self._emit(self.activeBouncesChanged, self._active_bounces_count)
        if self._active_bounces_refreshing:
            return
        # The stat is microseconds; the PARSE is what needed a thread. Doing the
        # stat inline means the 3-second tick creates a thread only when the
        # file actually moved - about 1,200 fewer thread creations an hour.
        if not self._active_bounces_signature_moved():
            return
        self._active_bounces_refreshing = True
        threading.Thread(
            target=self._load_active_bounces_worker,
            name="qt-bounce-health",
            daemon=True,
        ).start()

    def _active_bounces_signature_moved(self) -> bool:
        """Whether the signals file has changed since the last successful read.

        True on any doubt: an unreadable file gets a worker, because the worker
        is the thing that knows how to turn that into an honest zero.
        """
        try:
            from project_paths import AVWAP_SIGNALS_FILE

            stat = Path(AVWAP_SIGNALS_FILE).stat()
        except (OSError, ImportError):
            return True
        return (int(stat.st_size), int(stat.st_mtime_ns)) != self._active_bounces_signature

    def _load_active_bounces_worker(self) -> None:
        count = self._active_bounces_count
        signature = self._active_bounces_signature
        try:
            from datetime import date

            from master_avwap_shared import (
                load_active_bounce_summary,
                load_master_avwap_events_for_date,
            )
            from project_paths import AVWAP_SIGNALS_FILE, MASTER_AVWAP_ACTIVE_EVENTS_FILE

            stat = Path(AVWAP_SIGNALS_FILE).stat()
            current_signature = (int(stat.st_size), int(stat.st_mtime_ns))
            if current_signature != signature:
                compact = load_active_bounce_summary(
                    Path(MASTER_AVWAP_ACTIVE_EVENTS_FILE),
                    signals_path=Path(AVWAP_SIGNALS_FILE),
                    trade_date=date.today(),
                )
                if compact is not None:
                    count = len(compact)
                else:
                    events = load_master_avwap_events_for_date(
                        trade_date=date.today(),
                        signals_path=Path(AVWAP_SIGNALS_FILE),
                    )
                    count = len(
                        {
                            symbol
                            for symbol, rows in events.items()
                            if any(
                                str(row.get("signal_type") or "").upper().startswith("BOUNCE")
                                for row in rows
                            )
                        }
                    )
                signature = current_signature
        except OSError:
            count = 0
            signature = None
        except Exception:
            logging.debug("Active AVWAP health refresh failed.", exc_info=True)
        self._emit(
            self._activeBouncesReady,
            {"count": int(count), "signature": signature},
        )

    @Slot(object)
    def _on_active_bounces_ready(self, payload: object) -> None:
        self._active_bounces_refreshing = False
        if not self._is_live() or not isinstance(payload, dict):
            return
        if self._current_bot() is None:
            self._active_bounces_count = 0
            self._emit(self.activeBouncesChanged, 0)
            return
        self._active_bounces_count = max(0, int(payload.get("count") or 0))
        raw_signature = payload.get("signature")
        self._active_bounces_signature = (
            tuple(raw_signature) if isinstance(raw_signature, (tuple, list)) else None
        )
        self._emit(self.activeBouncesChanged, self._active_bounces_count)

    def _start_worker(self, session: _StartupSession) -> None:
        """Own one startup generation end to end.

        Every stage re-checks ``session.cancelled``: shutdown can arrive before
        we begin, while construction blocks, during saved-state application, or
        one instruction before the success/failure signals.  In all of those
        cases the generation retires quietly and any bot it managed to build is
        closed rather than installed.
        """

        bot = None
        failure: str | None = None
        try:
            try:
                if not session.is_cancelled:
                    from bounce_bot import run_bot_with_gui

                try:
                    if not session.is_cancelled:
                        bot = run_bot_with_gui(
                            self._make_callback(session),
                            start_scanning_enabled=self.scanning_enabled,
                        )
                finally:
                    # The IB connect is behind us (returned or raised): a new
                    # generation may now safely open its own.  Set before any
                    # further work so a parked _apply_saved_state never blocks
                    # the desk from restarting.
                    session.construction_done.set()
                if bot is not None and not session.is_cancelled:
                    self._apply_saved_state(bot)
                if bot is not None and not session.is_cancelled:
                    self._sync_state_from_bot(bot, session)
            except Exception as exc:
                failure = str(exc) or exc.__class__.__name__

            installed = False
            with self._lock:
                if failure is None and bot is not None and self._session is session and not session.is_cancelled:
                    self._bot = bot
                    installed = True  # the session stays: it is now the bot's owner token
                elif self._session is session:
                    # Failed or superseded generation: release ownership so the
                    # desk (and Auto Pilot's ``running`` check) can start again.
                    self._session = None

            if installed:
                connected = bool(getattr(bot, "connection_status", False))
                self._emit_for(session, self.connectionChanged, "IB: connected" if connected else "IB: retrying")
                self._emit_for(session, self.statusChanged, "connected")
                self._emit_for(session, self.started)
                return

            if bot is not None:
                # A stop/restart that raced this startup wins - and so does a
                # half-built startup: never leave an IB-connected bot running
                # with nobody owning it.
                self._retire_bot(bot)
            if failure is not None:
                self._emit_for(session, self.connectionChanged, "IB: disconnected")
                self._emit_for(session, self.statusChanged, f"start failed: {failure}")
                self._emit_for(session, self.failed, failure)
        finally:
            session.construction_done.set()
            session.finished.set()

    def _make_callback(self, session: _StartupSession | None = None) -> Callable[[Any, str], None]:
        """Build the bot's ``gui_callback``, bound to the generation that owns it.

        ``session`` is omitted only by callers that want the pure tag-filtering
        behavior with no bot behind it; the startup worker always passes its own
        generation so the callback falls silent the moment that generation is
        cancelled.
        """

        if session is None:
            session = self._session or _StartupSession(self._generation)

        def gui_callback(message: Any, tag: str) -> None:
            if session.is_cancelled or self._liveness.closed:
                # The bot that owns this callback belongs to a retired
                # generation (e.g. its after-close learning thread finishing
                # late).  Stay silent instead of signalling a dead service.
                return
            tag_text = str(tag or "")
            if tag_text == "rrs_status":
                self._emit_for(session, self.rrsStatusChanged, str(message))
                return
            if tag_text == "rrs_snapshot":
                self._emit_for(session, self.rrsSnapshotChanged, message)
                return
            message_text = str(message)
            if tag_text == "blue" and "removed from" in message_text:
                return
            # Auto-populate watchlist housekeeping is silent by design
            # (user rule 2026-07-24): the adds/rotations into longs/shorts.txt
            # just happen - no Alert Center entry.
            if message_text.startswith("AUTO WATCHLIST"):
                return
            if not self.include_approaching and (
                tag_text == "approaching" or tag_text.startswith("approaching_")
            ):
                return
            for alert in BounceAlert.from_callback_many(message, tag_text):
                self._emit_for(session, self.alertReceived, alert)

        return gui_callback

    def _apply_saved_state(self, bot) -> None:
        bot.set_rrs_threshold(self.rrs_threshold)
        bot.set_rrs_timeframe(self.rrs_timeframe_key)
        if self.market_environment:
            bot.set_market_environment(self.market_environment)
        else:
            bot.clear_market_environment_override()
        bot.set_scanning_enabled(self.scanning_enabled)
        for bounce_key, enabled in self.bounce_type_settings.items():
            bot.set_bounce_type_enabled(bounce_key, enabled)

    def _sync_state_from_bot(self, bot, session: _StartupSession | None = None) -> None:
        """Mirror the freshly built bot's state back onto the service.

        Every read below is a blocking call into the bot, and a ``stop()`` or
        ``restart()`` can land between any two of them.  One cancellation check
        at entry is therefore worthless: a cancelled generation-N worker would
        still overwrite generation-N+1's LIVE state (scanning_enabled, every
        bounce-type toggle, the regime, the RRS threshold/timeframe) - and since
        the trailing emit *is* gated, the desk would never be told, leaving the
        checkboxes and the running bot silently disagreeing.

        So: read into locals first, then re-check cancellation AND session
        identity under the lock before committing anything.  A cancelled
        generation mutates nothing.
        """

        if session is not None and session.is_cancelled:
            return  # shutdown landed mid saved-state application: mutate nothing

        # --- read-only phase: locals only, no service state touched ---------
        rrs_threshold = float(getattr(bot, "rrs_threshold", self.rrs_threshold))
        rrs_timeframe_key = str(getattr(bot, "rrs_timeframe_key", self.rrs_timeframe_key))
        # Keep the user selector independent from the auto read.  The bot has
        # an effective environment even in Auto; only mirror it when a genuine
        # manual override is active.
        market_environment: str | None = None
        if bool(getattr(bot, "market_environment_user_override", False)):
            try:
                market_environment = str(bot.get_market_environment())
            except Exception:
                market_environment = None
        scanning_enabled: bool | None
        try:
            scanning_enabled = bool(bot.is_scanning_enabled())
        except Exception:
            scanning_enabled = None
        bounce_types: dict[str, bool] = {}
        for bounce_key in list(self.bounce_type_settings):
            try:
                bounce_types[bounce_key] = bool(bot.is_bounce_type_enabled(bounce_key))
            except Exception:
                pass

        # --- commit phase: only if this generation still owns the service ---
        with self._lock:
            if session is not None and (session.is_cancelled or self._session is not session):
                return
            self.rrs_threshold = rrs_threshold
            self.rrs_timeframe_key = rrs_timeframe_key
            self.market_environment = market_environment
            if scanning_enabled is not None:
                self.scanning_enabled = scanning_enabled
            for bounce_key, enabled in bounce_types.items():
                if bounce_key in self.bounce_type_settings:
                    self.bounce_type_settings[bounce_key] = enabled
            committed_scanning = self.scanning_enabled

        if session is None:
            self._emit(self.scanningChanged, committed_scanning)
        else:
            self._emit_for(session, self.scanningChanged, committed_scanning)

    def current_bot(self):
        """The live BounceBot instance, or None (used by Auto Pilot)."""
        return self._current_bot()

    def _current_bot(self):
        with self._lock:
            return self._bot

    def _with_bot(self, callback: Callable[[Any], Any]) -> Any:
        bot = self._current_bot()
        if bot is None:
            return None
        try:
            return callback(bot)
        except Exception as exc:
            self._emit(self.statusChanged, f"command failed: {exc}")
            return None

    def _record_environment_annotation(self, selected: str | None, *, event: str) -> bool:
        bot = self._current_bot()
        reading: dict[str, Any] = {}
        if bot is not None:
            try:
                candidate = bot.get_auto_regime_reading()
                if isinstance(candidate, dict):
                    reading = candidate
            except Exception:
                pass
        return record_market_environment_annotation(
            selected_environment=selected,
            auto_reading=reading,
            session_id=self.environment_session_id,
            event=event,
            path=self.environment_annotations_path,
        ) is not None
