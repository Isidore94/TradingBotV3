"""Packet A (plan.md sec 6.2): BounceService startup-worker lifecycle ownership.

The defect these tests lock down: ``BounceService._start_worker`` ran on a bare
daemon thread with no cancellation token, so a startup that was still blocked in
``run_bot_with_gui`` (a real IB connect can sit there for a long time) would come
back *after* the service QObject had been destroyed and emit through it -
``RuntimeError: Signal source has been deleted``.  That killed the worker
mid-flight, which in turn leaked a fully constructed, IB-connected BounceBot that
nobody owned and nobody would ever stop.

Every test here deliberately delays startup, shuts the service down, and only
then releases the worker.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

JOIN_TIMEOUT = 10.0


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class FakeBot:
    """Minimal stand-in for BounceBot's startup + shutdown surface."""

    def __init__(self, name: str = "bot", *, gui_callback=None) -> None:
        self.name = name
        self.gui_callback = gui_callback
        self.connection_status = True
        self.stopped_with: float | None = None
        self.disconnected = False
        self.applied: list[str] = []
        self.rrs_threshold = 2.0
        self.rrs_timeframe_key = "5m"
        self.market_environment_user_override = False
        self.apply_gate: threading.Event | None = None

    # -- shutdown surface ------------------------------------------------
    def stop(self, timeout=None):
        self.stopped_with = timeout

    def disconnect(self):
        self.disconnected = True

    @property
    def retired(self) -> bool:
        return self.stopped_with is not None or self.disconnected

    # -- saved-state surface ---------------------------------------------
    def set_rrs_threshold(self, value):
        self.applied.append("rrs_threshold")

    def set_rrs_timeframe(self, key):
        self.applied.append("rrs_timeframe")

    def set_market_environment(self, key):
        self.applied.append("market_environment")

    def clear_market_environment_override(self):
        self.applied.append("clear_override")

    def set_scanning_enabled(self, enabled):
        self.applied.append("scanning")

    def set_bounce_type_enabled(self, key, enabled):
        self.applied.append(f"bounce:{key}")
        if self.apply_gate is not None:
            self.apply_gate.wait(timeout=JOIN_TIMEOUT)

    # -- read-back surface ------------------------------------------------
    def get_market_environment(self):
        return "bullish_strong"

    def is_scanning_enabled(self):
        return False

    def is_bounce_type_enabled(self, key):
        return True


class ThreadExceptionRecorder:
    """Captures anything that would become a PytestUnhandledThreadExceptionWarning."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def __call__(self, args) -> None:
        self.events.append(f"{args.thread and args.thread.name}: {args.exc_type.__name__}: {args.exc_value}")


@pytest.fixture
def thread_exceptions():
    recorder = ThreadExceptionRecorder()
    previous = threading.excepthook

    def record_and_chain(args):
        recorder(args)
        previous(args)

    # Keep pytest's threadexception hook active: the explicit recorder gives
    # precise assertions, while -W error must still exercise pytest's own
    # PytestUnhandledThreadExceptionWarning path.
    threading.excepthook = record_and_chain
    try:
        yield recorder
    finally:
        threading.excepthook = previous


@pytest.fixture
def service_factory(monkeypatch):
    """Builds services whose bot construction we control, and always retires them."""

    _qapp()
    import bounce_bot  # imported here so the worker's own import is a cache hit

    from ui.services.bounce_service import BounceService

    made: list[object] = []
    threads: list[threading.Thread] = []

    def factory(construct, *, retire_timeout: float = 0.25, cls=None):
        monkeypatch.setattr(bounce_bot, "run_bot_with_gui", construct)
        service = (cls or BounceService)()
        service.STARTUP_RETIRE_TIMEOUT = retire_timeout
        made.append(service)
        return service

    def track(service) -> threading.Thread | None:
        session = getattr(service, "_session", None)
        thread = session.thread if session is not None else None
        if thread is None:  # generation already retired itself
            thread = next(
                (t for t in threading.enumerate() if t.name == "qt-bouncebot-start" and t.is_alive()),
                None,
            )
        if thread is not None:
            threads.append(thread)
        return thread

    factory.track = track  # type: ignore[attr-defined]
    try:
        yield factory
    finally:
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT)
        for service in made:
            try:
                service.shutdown()
            except RuntimeError:
                pass  # deliberately deleted by the test
        # No startup worker may outlive its test.
        leftovers = [t for t in threading.enumerate() if t.name == "qt-bouncebot-start" and t.is_alive()]
        for thread in leftovers:
            thread.join(timeout=JOIN_TIMEOUT)
        assert not [t for t in leftovers if t.is_alive()], "a startup worker outlived its test"


def _join(thread: threading.Thread | None) -> None:
    """Join a startup worker; ``None`` means it had already retired itself."""

    if thread is not None:
        thread.join(timeout=JOIN_TIMEOUT)
        assert not thread.is_alive(), "the startup worker must retire"


def _await(predicate, timeout: float = JOIN_TIMEOUT) -> bool:
    """Wait for a condition, pumping Qt so cross-thread (queued) signals land."""

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return predicate()


# ----------------------------------------------------------------------
# Race 1: shutdown before startup begins
# ----------------------------------------------------------------------
def test_shutdown_before_startup_begins_never_constructs_a_bot(service_factory, thread_exceptions):
    """The worker is parked before its first instruction, then cancelled."""

    from ui.services.bounce_service import BounceService

    gate = threading.Event()
    constructed: list[FakeBot] = []

    class ParkedService(BounceService):
        def _start_worker(self, session):  # noqa: D102 - park before ANY startup work
            gate.wait(timeout=JOIN_TIMEOUT)
            super()._start_worker(session)

    def construct(callback, start_scanning_enabled=False):
        bot = FakeBot("never", gui_callback=callback)
        constructed.append(bot)
        return bot

    parked = service_factory(construct, cls=ParkedService)

    parked.start()
    session = parked._session
    assert session is not None and session.generation == 1
    thread = session.thread
    assert thread is not None

    parked.stop()  # shutdown lands while the worker is still parked
    assert session.is_cancelled

    gate.set()  # release the worker only now
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert session.finished.is_set()

    assert constructed == [], "a cancelled generation must not even attempt construction"
    assert parked.current_bot() is None
    assert not parked.running
    assert thread_exceptions.events == []
    parked.shutdown()


# ----------------------------------------------------------------------
# Race 2: shutdown while bot construction blocks
# ----------------------------------------------------------------------
def test_shutdown_while_construction_blocks_retires_the_late_bot(service_factory, thread_exceptions):
    release = threading.Event()
    built: list[FakeBot] = []

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        bot = FakeBot("late", gui_callback=callback)
        built.append(bot)
        return bot

    service = service_factory(construct)
    service.start()
    thread = service_factory.track(service)
    session = service._session
    assert session is not None
    assert _await(lambda: service.running)

    service.stop()  # user stops while startup is blocked in "IB connect"

    release.set()
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()

    assert len(built) == 1
    assert built[0].retired, "a bot built after shutdown must be closed, not leaked"
    assert built[0].stopped_with == 5.0
    assert service.current_bot() is None
    assert not service.running
    assert thread_exceptions.events == []


# ----------------------------------------------------------------------
# Race 3: shutdown during saved-state application
# ----------------------------------------------------------------------
def test_shutdown_during_saved_state_application_installs_nothing(service_factory, thread_exceptions):
    apply_gate = threading.Event()
    entered = threading.Event()
    built: list[FakeBot] = []

    def construct(callback, start_scanning_enabled=False):
        bot = FakeBot("mid-apply", gui_callback=callback)

        def gated(key, enabled, _bot=bot):
            entered.set()
            apply_gate.wait(timeout=JOIN_TIMEOUT)

        bot.set_bounce_type_enabled = gated
        built.append(bot)
        return bot

    service = service_factory(construct)
    scanning_seen: list[bool] = []
    service.scanningChanged.connect(scanning_seen.append)

    service.start()
    thread = service_factory.track(service)
    assert _await(entered.is_set)

    service.stop()  # shutdown lands inside _apply_saved_state
    apply_gate.set()
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()

    assert built[0].retired, "a half-configured bot must still be closed"
    assert service.current_bot() is None
    assert not service.running
    # _sync_state_from_bot must not run at all: no state mutation, no signal.
    assert not _await(lambda: bool(scanning_seen), timeout=0.3), (
        "a cancelled generation must not sync state back into the service"
    )
    assert thread_exceptions.events == []


# ----------------------------------------------------------------------
# Race 4: shutdown (by deletion) immediately before the success signals
# ----------------------------------------------------------------------
def test_deleted_service_emits_nothing_on_the_success_path(service_factory, thread_exceptions):
    from shiboken6 import delete, isValid

    release = threading.Event()
    built: list[FakeBot] = []

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        bot = FakeBot("deleted-owner", gui_callback=callback)
        built.append(bot)
        return bot

    service = service_factory(construct)
    service.start()
    thread = service_factory.track(service)
    assert _await(lambda: service.running)

    service.shutdown()
    delete(service)  # the Qt half is gone; the Python wrapper is not
    assert not isValid(service)

    release.set()
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()

    assert thread_exceptions.events == [], "no signal may be emitted through a deleted QObject"
    assert built[0].retired, "the late bot must be closed even though its owner is gone"

    # The bot's own late callback (legacy's after-close learning thread does
    # exactly this) must also stay silent instead of raising.
    built[0].gui_callback("Learning refreshed after the close (0 measured segments).", "blue")
    assert thread_exceptions.events == []


# ----------------------------------------------------------------------
# Race 5: shutdown immediately before the failure signals
# ----------------------------------------------------------------------
def test_failed_startup_after_shutdown_emits_no_failure_signal(service_factory, thread_exceptions):
    release = threading.Event()
    seen: list[str] = []

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        raise RuntimeError("IB refused the client id")

    service = service_factory(construct)
    service.failed.connect(seen.append)

    service.start()
    thread = service_factory.track(service)
    assert _await(lambda: service.running)

    service.stop()
    release.set()
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive()

    assert seen == [], "a cancelled generation must not report its failure"
    assert service.current_bot() is None
    assert not service.running
    assert thread_exceptions.events == []


def test_failed_startup_still_reports_when_not_cancelled(service_factory, thread_exceptions):
    """The guard must not silence a genuine live startup failure."""

    seen: list[str] = []

    def construct(callback, start_scanning_enabled=False):
        raise RuntimeError("IB refused the client id")

    service = service_factory(construct)
    service.failed.connect(seen.append)
    service.start()
    thread = service_factory.track(service)
    _join(thread)

    assert _await(lambda: seen == ["IB refused the client id"])
    assert not service.running, "a failed startup must release ownership so a retry can start"


# ----------------------------------------------------------------------
# Race 6: shutdown followed by restart
# ----------------------------------------------------------------------
def test_restart_waits_for_the_old_generation_then_installs_only_the_new_one(
    service_factory, thread_exceptions
):
    """A connected generation remains the sole IB owner until its worker exits."""

    first_release = threading.Event()
    entered_apply = threading.Event()
    built: list[FakeBot] = []
    calls = {"n": 0}

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        if calls["n"] == 1:
            bot = FakeBot("gen1", gui_callback=callback)

            def parked(key, enabled, _bot=bot):
                _bot.applied.append(f"bounce:{key}")
                entered_apply.set()
                first_release.wait(timeout=JOIN_TIMEOUT)

            bot.set_bounce_type_enabled = parked
        else:
            bot = FakeBot("gen2", gui_callback=callback)
        built.append(bot)
        return bot

    service = service_factory(construct)
    service.STOP_BUDGET = 0.2
    service.start()
    first_thread = service_factory.track(service)
    first_session = service._session
    assert _await(entered_apply.is_set)
    assert first_session.construction_done.is_set(), "gen1 is past its IB connect"

    service.restart()  # stop() cancels gen1; start() must defer while it still owns a connected bot
    assert service.current_bot() is None
    assert service._session is None
    assert calls["n"] == 1, "restart must not reuse the IB client id before gen1 has retired"

    first_release.set()
    first_thread.join(timeout=JOIN_TIMEOUT)
    assert not first_thread.is_alive()

    gen1 = [bot for bot in built if bot.name == "gen1"][0]
    assert gen1.retired, "the superseded generation's bot must be closed"

    assert service.start() is True
    second_thread = service_factory.track(service)
    _join(second_thread)
    assert _await(lambda: service.current_bot() is not None)
    assert service.current_bot().name == "gen2"
    assert thread_exceptions.events == []


def test_old_generation_callback_cannot_emit_after_restart(service_factory, thread_exceptions):
    """A retired bot's late gui_callback must not reach the Alert Center."""

    built: list[FakeBot] = []

    def construct(callback, start_scanning_enabled=False):
        bot = FakeBot(f"gen{len(built) + 1}", gui_callback=callback)
        built.append(bot)
        return bot

    service = service_factory(construct)
    service.start()
    thread = service_factory.track(service)
    _join(thread)
    assert _await(lambda: service.current_bot() is not None)
    old = service.current_bot()

    alerts: list[object] = []
    service.alertReceived.connect(alerts.append)
    old.gui_callback("AAPL long bounce trigger", "green")
    assert _await(lambda: len(alerts) == 1), "a live generation must still deliver alerts"

    service.restart()
    second = service_factory.track(service)
    if second is not None:
        second.join(timeout=JOIN_TIMEOUT)

    before = len(alerts)
    old.gui_callback("stale alert from the retired generation", "green")
    time.sleep(0.05)
    assert len(alerts) == before, "a cancelled generation's callback must be silent"
    assert thread_exceptions.events == []


# ----------------------------------------------------------------------
# Bounded shutdown
# ----------------------------------------------------------------------
def test_stop_is_bounded_even_when_construction_never_returns(service_factory, thread_exceptions):
    release = threading.Event()
    built: list[FakeBot] = []

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        bot = FakeBot("stuck", gui_callback=callback)
        built.append(bot)
        return bot

    service = service_factory(construct, retire_timeout=0.3)
    service.start()
    thread = service_factory.track(service)
    assert _await(lambda: service.running)

    started = time.monotonic()
    service.stop()
    elapsed = time.monotonic() - started

    # Measured upper bound: the join budget plus generous scheduling slack.
    assert elapsed < 1.5, f"stop() must be bounded; took {elapsed:.3f}s"
    assert elapsed >= 0.0
    assert not service.running, "stop() must retire ownership even if the worker is still stuck"

    release.set()
    thread.join(timeout=JOIN_TIMEOUT)
    assert built[0].retired, "the stuck worker must retire its own late bot"
    assert thread_exceptions.events == []


def test_stop_joins_a_quick_startup_worker(service_factory, thread_exceptions):
    """When the worker can finish inside the budget, stop() actually joins it."""

    def construct(callback, start_scanning_enabled=False):
        return FakeBot("quick", gui_callback=callback)

    service = service_factory(construct, retire_timeout=5.0)
    service.start()
    thread = service_factory.track(service)
    session = service._session

    service.stop()
    assert session.finished.is_set(), "stop() must join a worker that fits the budget"
    assert not thread.is_alive()
    assert service.current_bot() is None
    assert thread_exceptions.events == []


# ----------------------------------------------------------------------
# Live behavior must be preserved
# ----------------------------------------------------------------------
def test_normal_startup_still_starts_the_bot_and_its_timers(service_factory, thread_exceptions):
    def construct(callback, start_scanning_enabled=False):
        return FakeBot("live", gui_callback=callback)

    service = service_factory(construct)
    statuses: list[str] = []
    service.statusChanged.connect(statuses.append)
    started: list[int] = []
    service.started.connect(lambda: started.append(1))

    service.start()
    thread = service_factory.track(service)
    _join(thread)

    assert _await(lambda: service.current_bot() is not None)
    bot = service.current_bot()
    assert bot.name == "live"
    assert "clear_override" in bot.applied, "N/A must hand the regime to auto tracking"
    assert "scanning" in bot.applied
    assert any(key.startswith("bounce:") for key in bot.applied)
    assert service.running

    assert _await(lambda: started == [1]), "started must fire on a healthy startup"
    assert _await(lambda: "connected" in statuses)
    # started drives the health/regime/integrity/board timers on the GUI thread.
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    assert service._health_timer.isActive()
    assert service._regime_timer.isActive()
    assert service._integrity_timer.isActive()
    assert service._board_timer.isActive()

    service.stop()
    assert bot.stopped_with == 5.0
    assert not service._health_timer.isActive()
    assert not service._board_timer.isActive()
    assert thread_exceptions.events == []


def test_alerts_flow_through_a_live_generation(service_factory, thread_exceptions):
    def construct(callback, start_scanning_enabled=False):
        return FakeBot("live", gui_callback=callback)

    service = service_factory(construct)
    service.start()
    thread = service_factory.track(service)
    _join(thread)
    assert _await(lambda: service.current_bot() is not None)

    alerts: list[object] = []
    rrs: list[str] = []
    service.alertReceived.connect(alerts.append)
    service.rrsStatusChanged.connect(rrs.append)

    callback = service.current_bot().gui_callback
    callback("AAPL long bounce trigger", "green")
    callback("RRS ok", "rrs_status")
    assert _await(lambda: len(alerts) == 1 and rrs == ["RRS ok"])
    assert thread_exceptions.events == []


# ----------------------------------------------------------------------
# Reviewer holes (each of these fails without its fix in bounce_service.py)
# ----------------------------------------------------------------------
def test_queued_started_delivered_after_stop_never_rearms_timers(service_factory, thread_exceptions):
    """HOLE 1: the gap is between EMIT and DELIVERY, not around the emit.

    ``started`` is emitted on the worker thread, so Qt queues it.  If stop()
    runs before the event loop delivers it, the four arming slots used to fire
    *after* _stop_timers() and re-arm every timer on a service with no bot -
    health emitting every 3s and integrity doing disk I/O every 30s, forever,
    with nothing left to stop them.
    """

    from PySide6.QtWidgets import QApplication

    def construct(callback, start_scanning_enabled=False):
        return FakeBot("fast", gui_callback=callback)

    service = service_factory(construct, retire_timeout=5.0)
    service.start()
    thread = service_factory.track(service)
    _join(thread)  # started has been EMITTED; deliberately not delivered yet
    assert service.current_bot() is not None

    service.stop()  # lands in the emit -> delivery window
    QApplication.processEvents()  # only now does Qt deliver the queued started

    assert not service.running
    assert service.current_bot() is None
    assert not service._health_timer.isActive(), "a queued started must not re-arm the health timer"
    assert not service._regime_timer.isActive(), "a queued started must not re-arm the regime timer"
    assert not service._integrity_timer.isActive(), (
        "a queued started must not re-arm the integrity timer (it does disk I/O every 30s)"
    )
    assert not service._board_timer.isActive(), "a queued started must not re-arm the board timer"
    assert thread_exceptions.events == []


def test_cancelled_generation_cannot_overwrite_live_service_state(service_factory, thread_exceptions):
    """HOLE 2: one cancellation check cannot cover five blocking bot reads.

    A cancelled generation-N worker parked inside _sync_state_from_bot used to
    come back and overwrite generation-N+1's LIVE state - and because the
    trailing emit *is* gated, the desk was never told, so the checkboxes and the
    running bot silently disagreed.
    """

    sync_gate = threading.Event()
    entered_sync = threading.Event()
    calls = {"n": 0}

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        bot = FakeBot(f"gen{calls['n']}", gui_callback=callback)
        if calls["n"] == 1:
            def parked(key):
                entered_sync.set()
                sync_gate.wait(timeout=JOIN_TIMEOUT)
                return True  # gen1 believes every bounce type is ON

            bot.is_bounce_type_enabled = parked
            bot.is_scanning_enabled = lambda: False  # ... and that scanning is OFF
            bot.rrs_threshold = 2.0
        return bot

    service = service_factory(construct, retire_timeout=0.2)
    service.STOP_BUDGET = 0.2
    service.start()
    first_thread = service_factory.track(service)
    first_session = service._session
    assert _await(entered_sync.is_set)
    assert first_session.construction_done.is_set(), "gen1 is past its IB connect"

    service.stop()  # cancels gen1 while it is parked mid-sync

    # The desk changes the reversible service settings while no bot is live.
    scanning_seen: list[bool] = []
    service.scanningChanged.connect(scanning_seen.append)
    service.set_scanning_enabled(True)
    key = next(iter(service.bounce_type_settings))
    service.set_bounce_type_enabled(key, False)
    service.set_rrs_threshold(7.5)
    emitted_before = list(scanning_seen)
    assert service.start() is False, "gen1 still owns a connected IB session until it exits"

    sync_gate.set()  # release the cancelled generation
    first_thread.join(timeout=JOIN_TIMEOUT)
    assert not first_thread.is_alive()

    assert service.scanning_enabled is True, "a cancelled generation reverted live scanning state"
    assert service.bounce_type_settings[key] is False, (
        "a cancelled generation reverted a live bounce-type toggle"
    )
    assert service.rrs_threshold == 7.5, "a cancelled generation reverted the live RRS threshold"
    assert scanning_seen == emitted_before, "no silent divergence: nothing was mutated, nothing emitted"
    assert calls["n"] == 1
    assert thread_exceptions.events == []


def test_retirement_that_exceeds_the_budget_blocks_a_new_start(
    service_factory, thread_exceptions
):
    release = threading.Event()
    calls = {"n": 0}

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        bot = FakeBot(f"gen{calls['n']}", gui_callback=callback)

        def slow_stop(timeout=None, _bot=bot):
            release.wait(timeout=JOIN_TIMEOUT)
            _bot.stopped_with = timeout

        bot.stop = slow_stop
        return bot

    service = service_factory(construct)
    service.STOP_BUDGET = 0.2
    assert service.start() is True
    _join(service_factory.track(service))
    assert _await(lambda: service.current_bot() is not None)

    service.stop()
    assert any("bot retirement" in item for item in service.unretired_workers())
    assert service.start() is False
    assert calls["n"] == 1, "the prior bot may still hold the shared IB client id"

    release.set()
    assert _await(lambda: service.unretired_workers() == [])
    assert service.start() is True
    _join(service_factory.track(service))
    assert calls["n"] == 2
    assert thread_exceptions.events == []


def test_stopped_signal_cannot_restart_before_the_timed_out_worker_is_tracked(
    service_factory, thread_exceptions
):
    release = threading.Event()
    entered = threading.Event()
    calls = {"n": 0}

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        entered.set()
        release.wait(timeout=JOIN_TIMEOUT)
        return FakeBot(f"gen{calls['n']}", gui_callback=callback)

    service = service_factory(construct)
    service.STOP_BUDGET = 0.2
    restart_results: list[bool] = []
    service.stopped.connect(lambda: restart_results.append(service.start()))

    assert service.start() is True
    first_thread = service_factory.track(service)
    assert _await(entered.is_set)
    service.stop()

    assert restart_results == [False]
    assert calls["n"] == 1
    assert any("startup worker" in item for item in service.unretired_workers())

    release.set()
    _join(first_thread)
    assert thread_exceptions.events == []


def test_owned_thread_snapshot_exposes_overdue_startup(service_factory, thread_exceptions):
    from ui.services.bounce_service import owned_bounce_thread_snapshot

    release = threading.Event()

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        return FakeBot("late", gui_callback=callback)

    service = service_factory(construct)
    service.STOP_BUDGET = 0.2
    assert service.start() is True
    thread = service_factory.track(service)
    assert _await(lambda: service.running)
    service.stop()

    snapshot = owned_bounce_thread_snapshot()
    assert snapshot["bounce_service_count"] >= 1
    assert snapshot["bounce_unretired_worker_count"] >= 1
    assert any("startup worker" in item for item in snapshot["bounce_unretired_workers"])

    release.set()
    _join(thread)
    assert thread_exceptions.events == []


def test_timer_stop_runtimeerror_does_not_retire_a_live_service(
    service_factory, thread_exceptions, monkeypatch
):
    """HOLE 3: _stop_timers must not latch liveness on a service that is ALIVE.

    Latching there permanently kills a healthy service: start() returns at the
    liveness guard and every _emit short-circuits - a silently dead scanner.
    """

    from PySide6.QtCore import QTimer
    from shiboken6 import isValid

    def construct(callback, start_scanning_enabled=False):
        return FakeBot("live", gui_callback=callback)

    service = service_factory(construct, retire_timeout=5.0)

    def boom(self):
        raise RuntimeError("timer backend hiccup")

    monkeypatch.setattr(QTimer, "stop", boom)
    with pytest.raises(RuntimeError):
        service.stop()  # mirrors _emit: a live object re-raises instead of swallowing
    monkeypatch.undo()

    assert isValid(service), "the Qt half was never destroyed"
    assert not service._liveness.closed, (
        "a RuntimeError from a live service's timer must not permanently retire it"
    )

    # ... and the scanner can still be started.
    assert service.start() is True
    thread = service_factory.track(service)
    _join(thread)
    assert _await(lambda: service.current_bot() is not None)
    assert thread_exceptions.events == []


def test_stop_is_bounded_when_the_bot_stop_is_slow(service_factory, thread_exceptions):
    """HOLE 4: the real bound must include bot retirement, not just the join.

    Real BounceBot.stop joins strategy_thread AND api_thread at ``timeout``
    each, so retiring inline on the GUI thread could freeze the desk for ~10s
    on Stop or window close.
    """

    slow = threading.Event()

    def construct(callback, start_scanning_enabled=False):
        bot = FakeBot("slow-stop", gui_callback=callback)

        def slow_stop(timeout=None, _bot=bot):
            slow.wait(timeout=JOIN_TIMEOUT)
            _bot.stopped_with = timeout

        bot.stop = slow_stop
        return bot

    service = service_factory(construct, retire_timeout=5.0)
    service.STOP_BUDGET = 0.3
    service.start()
    thread = service_factory.track(service)
    _join(thread)
    assert _await(lambda: service.current_bot() is not None)
    bot = service.current_bot()

    started = time.monotonic()
    service.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 1.5, f"stop() must be bounded by STOP_BUDGET; took {elapsed:.3f}s"
    assert service.unretired_workers(), "a retirement that overran the budget must be reported"
    assert not service.running

    slow.set()
    assert _await(lambda: bot.stopped_with == 5.0), "the bot must still be stopped, just not inline"
    assert _await(lambda: service.unretired_workers() == [])
    assert thread_exceptions.events == []


def test_the_real_worst_case_stop_bound_is_the_stop_budget(service_factory, thread_exceptions):
    """The measured bound at PRODUCTION constants, with nothing overridden.

    Bot retirement and the startup join share ONE budget, so neither can
    extend the freeze: stop() <= STOP_BUDGET, and window close
    (shutdown()) <= STOP_BUDGET + SHUTDOWN_BUDGET.  A bot whose stop() never
    returns is the worst case, because real BounceBot.stop joins both the
    strategy and the API thread.
    """

    from ui.services.bounce_service import BounceService

    hang = threading.Event()

    def construct(callback, start_scanning_enabled=False):
        bot = FakeBot("never-stops", gui_callback=callback)

        def never_returns(timeout=None, _bot=bot):
            hang.wait(timeout=JOIN_TIMEOUT)
            _bot.stopped_with = timeout

        bot.stop = never_returns
        return bot

    service = service_factory(construct, retire_timeout=BounceService.STARTUP_RETIRE_TIMEOUT)
    assert service.STOP_BUDGET == 3.0 and service.SHUTDOWN_BUDGET == 2.0
    service.start()
    thread = service_factory.track(service)
    _join(thread)
    assert _await(lambda: service.current_bot() is not None)
    bot = service.current_bot()

    started = time.monotonic()
    service.stop()
    stop_elapsed = time.monotonic() - started
    assert stop_elapsed < service.STOP_BUDGET + 1.0, (
        f"stop() froze the GUI for {stop_elapsed:.3f}s; the bound is {service.STOP_BUDGET}s"
    )

    started = time.monotonic()
    service.shutdown()
    shutdown_elapsed = time.monotonic() - started
    assert shutdown_elapsed < service.SHUTDOWN_BUDGET + 1.0, (
        f"shutdown() froze the GUI for {shutdown_elapsed:.3f}s; "
        f"the extra bound is {service.SHUTDOWN_BUDGET}s"
    )
    assert service.unretired_workers(), "the un-retired retirement thread is reported, not forgotten"

    hang.set()
    assert _await(lambda: bot.stopped_with == 5.0), "the bot is still stopped, just not inline"
    assert _await(lambda: service.unretired_workers() == [])
    assert thread_exceptions.events == []


def test_a_worker_that_outlives_the_join_is_tracked_and_reported(service_factory, thread_exceptions):
    """HOLE 5: a worker that misses the join must not become untracked forever."""

    release = threading.Event()
    built: list[FakeBot] = []

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        bot = FakeBot("stuck", gui_callback=callback)
        built.append(bot)
        return bot

    service = service_factory(construct, retire_timeout=0.2)
    service.start()
    thread = service_factory.track(service)
    assert _await(lambda: service.running)

    service.stop()  # the join times out
    outstanding = service.unretired_workers()
    assert outstanding, "a worker that outlived the join must stay tracked, not vanish"
    assert any("startup worker" in entry for entry in outstanding)

    release.set()
    service.shutdown()  # final bounded attempt actually reaches it
    assert not thread.is_alive()
    assert service.unretired_workers() == []
    assert built[0].retired, "the late bot is still closed"
    assert thread_exceptions.events == []


def test_a_second_startup_cannot_begin_while_the_first_is_connecting(service_factory, thread_exceptions):
    """HOLE 6: two concurrent run_bot_with_gui calls = same hard-coded IB client
    id = Error 326 and two live IB sessions.  The service, not the caller, owns
    the refusal - and the refusal is self-healing."""

    release = threading.Event()
    constructs = {"n": 0}

    def construct(callback, start_scanning_enabled=False):
        constructs["n"] += 1
        release.wait(timeout=JOIN_TIMEOUT)
        return FakeBot(f"gen{constructs['n']}", gui_callback=callback)

    service = service_factory(construct, retire_timeout=0.2)
    service.start()
    thread = service_factory.track(service)
    assert _await(lambda: service.running)

    service.stop()  # join times out; the worker is still inside run_bot_with_gui
    assert not service.running

    assert service.start() is False, "a startup must not begin while a previous one is connecting"
    alive = [t for t in threading.enumerate() if t.name == "qt-bouncebot-start" and t.is_alive()]
    assert len(alive) == 1, f"only one startup worker may exist; saw {len(alive)}"
    assert constructs["n"] == 1, "run_bot_with_gui must not be entered twice concurrently"

    release.set()
    thread.join(timeout=JOIN_TIMEOUT)

    # Self-healing: once the stuck generation clears, the scanner starts again.
    assert _await(lambda: service.start() is True), "the refusal must not permanently block startup"
    second = service_factory.track(service)
    _join(second)
    assert _await(lambda: service.current_bot() is not None)
    assert thread_exceptions.events == []


def test_retire_falls_back_to_disconnect_when_bot_stop_raises(service_factory, thread_exceptions):
    """HOLE 7: a bot that could not be stopped must at least be disconnected."""

    def construct(callback, start_scanning_enabled=False):
        bot = FakeBot("angry", gui_callback=callback)

        def angry_stop(timeout=None):
            raise RuntimeError("stop() blew up")

        bot.stop = angry_stop
        return bot

    service = service_factory(construct, retire_timeout=5.0)
    service.start()
    thread = service_factory.track(service)
    _join(thread)
    assert _await(lambda: service.current_bot() is not None)
    bot = service.current_bot()

    service.stop()
    assert bot.disconnected, (
        "a failed stop() must still fall through to disconnect(); otherwise the IB session leaks"
    )
    assert thread_exceptions.events == []


def test_shutdown_is_terminal_and_blocks_a_later_start(service_factory, thread_exceptions):
    calls = {"n": 0}

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        return FakeBot("post-shutdown", gui_callback=callback)

    service = service_factory(construct)
    service.shutdown()
    service.start()
    time.sleep(0.05)

    assert calls["n"] == 0, "a retired service must never start a new generation"
    assert service.current_bot() is None
    assert thread_exceptions.events == []


def test_shutdown_latches_before_close_time_signals_can_restart(
    service_factory, thread_exceptions
):
    calls = {"n": 0}
    restart_results: list[bool] = []

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        return FakeBot("must-not-start", gui_callback=callback)

    service = service_factory(construct)
    service.stopped.connect(lambda: restart_results.append(service.start()))
    service.statusChanged.connect(
        lambda status: restart_results.append(service.start()) if status == "stopped" else None
    )

    service.shutdown()
    time.sleep(0.05)

    # Stop remains observable, but the separate terminal latch refuses both
    # direct-connected restart attempts before any worker can launch.
    assert restart_results == [False, False]
    assert calls["n"] == 0
    assert service._liveness.closed
    assert service._terminal.is_set()
    assert thread_exceptions.events == []


def test_connecting_signal_can_cancel_start_before_worker_launch(
    service_factory, thread_exceptions
):
    calls = {"n": 0}
    statuses: list[str] = []

    def construct(callback, start_scanning_enabled=False):
        calls["n"] += 1
        return FakeBot("must-not-start", gui_callback=callback)

    service = service_factory(construct)
    service.statusChanged.connect(statuses.append)
    service.connectionChanged.connect(
        lambda status: service.stop() if status == "IB: connecting" else None
    )

    assert service.start() is False
    assert calls["n"] == 0
    assert service.current_bot() is None
    assert not service.running
    assert statuses == ["stopped"], "the cancelled start must not publish a trailing connecting state"
    assert thread_exceptions.events == []


def test_every_blocked_start_is_visible_and_names_the_safe_recovery(
    service_factory, thread_exceptions
):
    release = threading.Event()

    def construct(callback, start_scanning_enabled=False):
        release.wait(timeout=JOIN_TIMEOUT)
        return FakeBot("stuck", gui_callback=callback)

    service = service_factory(construct)
    service.STOP_BUDGET = 0.2
    statuses: list[str] = []
    service.statusChanged.connect(statuses.append)
    assert service.start() is True
    worker = service_factory.track(service)
    assert _await(lambda: service.running)
    service.stop()

    before = len(statuses)
    assert service.start() is False
    assert service.start() is False
    blocked = statuses[before:]
    assert len(blocked) == 2
    assert all("has not retired" in message for message in blocked)
    assert all("restart the app" in message for message in blocked)

    release.set()
    _join(worker)
    assert thread_exceptions.events == []


def test_trading_desk_shutdown_continues_after_one_component_raises(caplog):
    from ui.panels.trading_desk import TradingDeskPanel

    calls: list[str] = []

    def broken_bounce():
        calls.append("bounce")
        raise RuntimeError("timer stop failed")

    fake_desk = SimpleNamespace(
        bounce_panel=SimpleNamespace(on_close=broken_bounce),
        industry_panel=SimpleNamespace(shutdown=lambda: calls.append("industry")),
        master_panel=SimpleNamespace(
            scan_service=SimpleNamespace(shutdown=lambda: calls.append("scan"))
        ),
    )

    TradingDeskPanel.shutdown(fake_desk)

    assert calls == ["bounce", "industry", "scan"]
    assert "continuing app cleanup" in caplog.text
