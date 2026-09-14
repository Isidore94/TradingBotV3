"""RV-H1-PHONE-WORKER - the armed-watch phone push leaves the Qt thread.

Review blocker B3 (WS-10C).  ``PriceAlertService.notify_armed_watch`` calls
``push_notify.send_push`` inline and the caller is the GUI poll
(``_poll_h1_bounce_watches`` -> ``_push_armed_watch``).  ``push_notify``'s HTTP
timeout is 10 s (``scripts/push_notify.py:39``), so a slow ntfy endpoint holds
the desk for up to ten seconds per fire.  CLAUDE.md: nothing expensive belongs
on the Qt thread.  The service's own precedent is ``check_now``: a one-shot
daemon thread, the QObject only orchestrates.

The contract these tests pin:

* dispatch is synchronous, delivery is not.  On the Qt thread the method does
  the cheap decisions - engine check, watch-id de-duplication - and returns at
  once with ``{"ok": True, "queued": True, "watch_id": ...}``.  ``ok`` means
  "accepted for delivery by the one armed sender";
* the worker is OWNED by the service, so the service knows what is in flight;
* the outcome comes back the way ``_notify`` already reports it -
  ``_last_push_error``, the ``ARMED WATCH ...`` log line, ``statusChanged``;
* ``shutdown()`` joins in-flight deliveries with a total budget of
  ``ARMED_PUSH_SHUTDOWN_WAIT_SECONDS = 2.0`` and then returns; the threads are
  daemons, so they never hold the process;
* the feed row for a fired watch is drawn while the phone is still answering;
* no ``auto_mode`` gate is added - DESK, AWAY, EVENING and OFF all deliver.

NO REAL PUSH.  ``push_notify.send_push`` is monkeypatched in every test in this
file and nothing here builds an ntfy URL; the fakes are the only transport.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_ws_10c_h1_retester import (  # noqa: E402
    GOLDEN_CONFIRM_DT,
    WATCH_KIND,
    golden_long_h1_bars,
    golden_m5_series,
)

#: The review's own reproduction on c4df3ac8 was
#: ``send_push_on_qt_thread [True] call_seconds 0.202`` - the sender ran on the
#: GUI thread and the GUI waited for it.  Every budget below is generous
#: against a 0.12-0.2 s fake, so only a synchronous send can blow one.
DISPATCH_BUDGET_SECONDS = 0.10
SLOW_SEND_SECONDS = 0.20


def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


def _service(monkeypatch, sender, *, mode: str = "DESK"):
    """A REAL ``PriceAlertService`` whose only transport is ``sender``."""
    _qt_app()
    import autopilot_core
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    monkeypatch.setattr(push_notify, "send_push", sender)
    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **kw: mode)
    return PriceAlertService()


def _panel(monkeypatch, tmp_path):
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    panel = AlertCenterPanel()
    panel._chart_watches_path = tmp_path / "chart_watches.json"
    return panel


# ---------------------------------------------------------------------------
# 1. Thread identity - the review's reproduction
# ---------------------------------------------------------------------------
def test_the_armed_push_is_delivered_off_the_qt_thread(monkeypatch):
    """The fake sender records whether it ran on the GUI thread and sleeps
    0.2 s.  Today both answers are wrong: it runs ON the Qt thread and the
    caller waits for it."""
    from PySide6.QtCore import QThread

    app = _qt_app()
    on_qt_thread: list[bool] = []

    def fake_send(*args, **kwargs):
        on_qt_thread.append(QThread.currentThread() == app.thread())
        time.sleep(SLOW_SEND_SECONDS)
        return {"ok": True}

    service = _service(monkeypatch, fake_send)
    try:
        started = time.perf_counter()
        result = service.notify_armed_watch(
            watch_id="rv-h1-thread",
            title="H1 retester: AAPL",
            message="AAPL LONG retest confirmed",
        )
        call_seconds = time.perf_counter() - started

        assert call_seconds < DISPATCH_BUDGET_SECONDS, (
            f"notify_armed_watch held the Qt thread for {call_seconds:.3f}s"
        )
        assert result.get("ok") is True
        assert result.get("queued") is True
        assert result.get("watch_id") == "rv-h1-thread"
    finally:
        service.shutdown()

    assert on_qt_thread == [False], (
        f"send_push_on_qt_thread {on_qt_thread} call_seconds {call_seconds:.3f}"
    )


# ---------------------------------------------------------------------------
# 2. The desk draws the row while the phone is still answering
# ---------------------------------------------------------------------------
def test_the_fired_watch_reaches_the_feed_while_the_send_is_still_in_flight(
    monkeypatch, tmp_path
):
    """The REAL poll (``_poll_h1_bounce_watches`` -> ``_push_armed_watch``) with
    a REAL ``PriceAlertService`` whose ``send_push`` blocks on an event.  The
    poll must return and the alert row must exist while the send is stuck."""
    panel = _panel(monkeypatch, tmp_path)
    h1_bars, _ = golden_long_h1_bars()
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda symbol, **kw: golden_m5_series(h1_bars)
    )
    monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol, **kw: [])

    entered = threading.Event()
    release = threading.Event()
    sent: list[tuple] = []

    def fake_send(*args, **kwargs):
        entered.set()
        # Bounded so a synchronous send cannot hang the suite; 2 s is still an
        # eternity on the Qt thread, and well short of push_notify's own 10 s.
        release.wait(2.0)
        sent.append((args, kwargs))
        return {"ok": True}

    service = _service(monkeypatch, fake_send)
    panel.price_alert_service = service
    try:
        panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
        watch = next(w for w in panel._chart_watches if w.kind == WATCH_KIND)
        # The arm predates the golden confirm bar, so the arm-time fence in the
        # previous link of this chain has no reason to refuse it.
        watch = dataclasses.replace(
            watch, armed_at=GOLDEN_CONFIRM_DT - timedelta(days=1)
        )
        panel._chart_watches = [watch]
        before = len(panel._alerts)
        moment = GOLDEN_CONFIRM_DT + timedelta(hours=1)

        started = time.perf_counter()
        panel._poll_d1_event_watches(now=moment)
        poll_seconds = time.perf_counter() - started

        assert poll_seconds < 0.5, (
            f"the GUI poll waited {poll_seconds:.3f}s for the phone"
        )
        assert len(panel._alerts) == before + 1
        row = panel._alerts[0]
        assert row.symbol == "AAPL"
        assert str(dict(row.payload or {}).get("watch_id") or "") == watch.watch_id
        # The delivery really is still in flight: the row is on the feed and
        # the transport has not returned.
        assert entered.wait(2.0) is True
        assert release.is_set() is False
        assert sent == []
    finally:
        release.set()
        service.shutdown()

    assert len(sent) == 1


# ---------------------------------------------------------------------------
# 3. De-duplication is decided synchronously, before the dispatch
# ---------------------------------------------------------------------------
def test_a_repeat_watch_id_is_refused_without_waiting_for_the_first_send(monkeypatch):
    """The id joins ``_announced_watch_ids`` BEFORE the dispatch, so three
    back-to-back calls cost the Qt thread nothing and exactly two pushes go
    out."""
    sent: list[str] = []

    def fake_send(title, message, **kwargs):
        time.sleep(0.15)
        sent.append(str(title))
        return {"ok": True}

    service = _service(monkeypatch, fake_send)
    try:
        started = time.perf_counter()
        first = service.notify_armed_watch(
            watch_id="rv-h1-dup", title="H1 retester: AAPL", message="AAPL LONG"
        )
        again = service.notify_armed_watch(
            watch_id="rv-h1-dup", title="H1 retester: AAPL", message="AAPL LONG"
        )
        other = service.notify_armed_watch(
            watch_id="rv-h1-other", title="H1 retester: MSFT", message="MSFT SHORT"
        )
        call_seconds = time.perf_counter() - started

        assert call_seconds < DISPATCH_BUDGET_SECONDS, (
            f"three dispatches held the Qt thread for {call_seconds:.3f}s"
        )
        assert first.get("ok") is True and first.get("queued") is True
        assert again.get("ok") is False
        assert again.get("deduplicated") is True
        assert again.get("watch_id") == "rv-h1-dup"
        assert other.get("ok") is True and other.get("queued") is True
    finally:
        service.shutdown()

    assert sorted(sent) == ["H1 retester: AAPL", "H1 retester: MSFT"]


# ---------------------------------------------------------------------------
# 4. The outcome comes back the way `_notify` already reports one
# ---------------------------------------------------------------------------
def test_a_failed_delivery_is_reported_after_the_worker_finishes(monkeypatch):
    """The dispatch says "queued"; the FAILURE arrives later, on
    ``_last_push_error`` / ``status_snapshot`` / ``statusChanged``.  The id is
    still marked announced - an arm is one episode and one attempt."""
    def fake_send(*args, **kwargs):
        time.sleep(0.05)
        return {"ok": False, "error": "boom"}

    service = _service(monkeypatch, fake_send)
    snapshots: list[dict] = []
    service.statusChanged.connect(snapshots.append)
    try:
        result = service.notify_armed_watch(
            watch_id="rv-h1-fail", title="H1 retester: AAPL", message="AAPL LONG"
        )
        assert result.get("ok") is True
        assert result.get("queued") is True
        assert result.get("watch_id") == "rv-h1-fail"
    finally:
        service.shutdown()

    assert service._last_push_error == "boom"
    assert service.status_snapshot().get("push_error") == "boom"
    assert any(
        str(snapshot.get("push_error") or "") == "boom" for snapshot in snapshots
    ), snapshots
    # One episode, one attempt: the repeat is still refused.
    assert service.notify_armed_watch(
        watch_id="rv-h1-fail", title="H1 retester: AAPL", message="AAPL LONG"
    ).get("deduplicated") is True


def test_a_sender_that_raises_is_logged_and_never_reaches_the_caller(
    monkeypatch, caplog
):
    """``send_push`` never raises, but the worker wraps it anyway: an exception
    on the delivery thread is logged, not lost, and not thrown at the poll."""
    def fake_send(*args, **kwargs):
        raise RuntimeError("rv-h1 fake transport exploded")

    service = _service(monkeypatch, fake_send)
    try:
        with caplog.at_level(logging.DEBUG):
            result = service.notify_armed_watch(
                watch_id="rv-h1-boom", title="H1 retester: AAPL", message="AAPL LONG"
            )
            assert result.get("ok") is True
            assert result.get("queued") is True
            service.shutdown()
    finally:
        service.shutdown()

    assert any(
        record.exc_info is not None
        or "rv-h1 fake transport exploded" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------------------------
# 5. shutdown() waits, but only so long
# ---------------------------------------------------------------------------
def test_shutdown_waits_for_an_in_flight_delivery_within_a_bounded_budget(monkeypatch):
    """A send that outlasts the budget does not hold the desk's shutdown, and
    the thread carrying it is a daemon so it cannot hold the process."""
    entered = threading.Event()
    release = threading.Event()
    daemon_flags: list[bool] = []

    def fake_send(*args, **kwargs):
        daemon_flags.append(bool(threading.current_thread().daemon))
        entered.set()
        release.wait(3.5)
        return {"ok": True}

    service = _service(monkeypatch, fake_send)
    try:
        started = time.perf_counter()
        service.notify_armed_watch(
            watch_id="rv-h1-shutdown", title="H1 retester: AAPL", message="AAPL LONG"
        )
        service.shutdown()
        elapsed = time.perf_counter() - started

        assert elapsed < 2.6, (
            f"dispatch + shutdown took {elapsed:.3f}s with a hung send"
        )

        from ui.services import price_alert_service as svc

        assert svc.ARMED_PUSH_SHUTDOWN_WAIT_SECONDS == 2.0
        assert entered.wait(1.0) is True
        assert daemon_flags == [True]
    finally:
        release.set()


# ---------------------------------------------------------------------------
# 6. Every Auto mode still delivers, and none of them waits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["AWAY", "EVENING", "OFF"])
def test_every_auto_mode_delivers_the_armed_push_without_holding_the_desk(
    monkeypatch, mode
):
    """The armed price-alert sender is the recorded every-mode exception; this
    packet moves the transport, not the policy."""
    sent: list[str] = []

    def fake_send(title, message, **kwargs):
        time.sleep(SLOW_SEND_SECONDS)
        sent.append(str(message))
        return {"ok": True}

    service = _service(monkeypatch, fake_send, mode=mode)
    try:
        started = time.perf_counter()
        result = service.notify_armed_watch(
            watch_id=f"rv-h1-{mode}", title="H1 retester: AAPL", message=f"AAPL {mode}"
        )
        call_seconds = time.perf_counter() - started

        assert call_seconds < DISPATCH_BUDGET_SECONDS, (
            f"{mode}: notify_armed_watch held the Qt thread for {call_seconds:.3f}s"
        )
        assert result.get("ok") is True
        assert result.get("queued") is True
    finally:
        service.shutdown()

    assert sent == [f"AAPL {mode}"]


# ---------------------------------------------------------------------------
# 7. Guard: the refusals are unchanged (expected GREEN before the fix)
# ---------------------------------------------------------------------------
def test_a_non_engine_desk_still_refuses_the_armed_push_and_sends_nothing(monkeypatch):
    """Not a red test - a regression guard the worker must not disturb: the
    engine-disabled refusal stays synchronous, keeps its wording, and never
    reaches the transport."""
    _qt_app()
    import autopilot_core
    import push_notify
    from ui.services.price_alert_service import PriceAlertService

    sent: list[tuple] = []
    monkeypatch.setattr(
        push_notify, "send_push", lambda *a, **kw: sent.append((a, kw)) or {"ok": True}
    )
    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **kw: "DESK")

    service = PriceAlertService(engine_enabled=False)
    try:
        result = service.notify_armed_watch(
            watch_id="rv-h1-off-desk", title="H1 retester: AAPL", message="AAPL LONG"
        )
    finally:
        service.shutdown()

    assert result.get("ok") is False
    assert result.get("queued") is not True
    assert "main desk only" in str(result.get("error") or "")
    assert sent == []
