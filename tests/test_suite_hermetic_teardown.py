"""P1.1: no BounceBot outlives the test that started it.

**Measured before it was fixed** (2026-08-24, a full-suite run with a
thread-recording plugin): 22 tests left at least one thread running past their
own teardown, and **19 `run_strategy` threads were still alive when the session
ended**. That is the "standing crowd" the garbage-collection block at the top of
`conftest.py` names in its own honesty paragraph - it removed the
threshold-GC hazard and said plainly that it did not join these.

A BounceBot strategy loop is not an idle thread. It reads the watchlist files,
refreshes learning state, re-reads Focus picks and - on a desk with TWS open -
would dial IB. Leaving one running through the rest of the suite means later
tests share mutable state with a scanner nobody is watching, which is the exact
shape of the intermittent failure that is most expensive to chase.

The fix is a bounded, cooperative retirement in `conftest.py`: BounceBot already
exposes `stop(timeout=...)`, which sets its stop event, disconnects, and joins
its own threads. The harness calls it for any strategy thread a test leaves
behind, and FAILS the leaking test by name if one survives - a teardown that
swallows the timeout cannot prove the quiescence it exists to provide.
"""

from __future__ import annotations

import threading
import time


class _FakeBot:
    """The cooperative-shutdown contract BounceBot implements, and nothing else."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self.strategy_thread: threading.Thread | None = None
        self.stopped = False

    def run_strategy(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(0.05)

    def stop(self, timeout: float = 10.0) -> None:
        self.stopped = True
        self._stop_event.set()
        if self.strategy_thread is not None and self.strategy_thread.is_alive():
            self.strategy_thread.join(timeout=timeout)

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_strategy, daemon=True)
        self.strategy_thread = thread
        thread.start()
        return thread


def test_a_leaked_strategy_thread_is_retired_rather_than_left_running():
    """The helper the autouse teardown uses, exercised directly."""
    import conftest

    bot = _FakeBot()
    thread = bot.start()
    assert thread.is_alive()

    still_running = conftest.retire_leaked_bounce_bots([thread], timeout=5.0)

    assert bot.stopped, "the bot's own cooperative stop() is what retires it"
    assert not still_running, f"still running after retirement: {still_running}"
    assert not thread.is_alive()


def test_retirement_is_bounded_and_reports_what_it_could_not_stop():
    """A bot that ignores its stop event is NAMED, never silently tolerated.

    Bounded is the whole point: a teardown that waits forever on a wedged
    worker turns one leak into a hung suite.
    """
    import conftest

    class _Deaf(_FakeBot):
        def run_strategy(self) -> None:  # never observes the stop event
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                time.sleep(0.05)

        def stop(self, timeout: float = 10.0) -> None:
            self.stopped = True  # sets nothing; the loop cannot hear it

    bot = _Deaf()
    thread = bot.start()
    try:
        started = time.monotonic()
        still_running = conftest.retire_leaked_bounce_bots([thread], timeout=0.2)
        elapsed = time.monotonic() - started

        assert still_running == [thread.name]
        assert elapsed < 3.0, f"retirement must be bounded, took {elapsed:.1f}s"
    finally:
        # This test's whole point is a bot that ignores its stop event, so the
        # autouse teardown would (correctly) fail the test over it. Wait it out
        # here instead - the deaf loop ends on its own five-second deadline.
        thread.join(timeout=10.0)
        assert not thread.is_alive()


def test_a_thread_that_is_not_a_bounce_bot_is_left_alone():
    """The teardown retires scanners, not every thread a test happens to start."""
    import conftest

    done = threading.Event()
    thread = threading.Thread(target=done.wait, daemon=True)
    thread.start()
    try:
        assert conftest.owning_bounce_bot(thread) is None
        assert conftest.retire_leaked_bounce_bots([thread], timeout=0.1) == []
    finally:
        done.set()
        thread.join(timeout=2.0)


def test_the_teardown_is_registered_as_an_autouse_fixture():
    """It must apply to every test, including ones written later."""
    import conftest

    fixture = conftest._retire_bounce_bots
    marker = getattr(fixture, "_fixture_function_marker", None) or getattr(
        fixture, "_pytestfixturefunction", None
    )
    assert marker is not None and marker.autouse, "must be autouse"
