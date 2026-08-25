"""Suite-wide isolation for machine-local runtime diagnostics, plus the
Milestone 3 fixture-contract loader.

Tests exercise live composition paths, including the D1 Greatness hook.  They
must never append synthetic events to the running application's evidence.

The second half of this module enforces the fixture contract documented in
plan.md Milestone 3.  Golden fixtures used to declare contract metadata that no
test ever read (a declared ``numeric_tolerance`` while every comparison was an
exact ``==``, acquisition times nobody checked).  Everything a fixture declares
now has to be present, well-formed, and actually used:

* raw input hashes + acquisition times -> ``raw_input_keys`` / ``raw_input_sha256``
  (recomputed from the fixture's own input section on every load) and
  ``acquired_at``;
* universe version -> ``universe_version``;
* configuration and feature versions -> ``feature_version`` (+ the optional
  ``configuration`` block, pinned by the consuming test);
* provider/calendar assumptions -> ``provider_assumptions``;
* exact ``as_of`` time -> ``as_of``;
* expected outputs in stable sorted form -> ``expected_keys``;
* allowed numeric tolerances -> ``numeric_tolerance``, applied by
  :meth:`FixtureContract.assert_matches`;
* intentional-difference approval notes -> ``intentional_difference``.

A missing or malformed field raises :class:`FixtureContractError`, which is an
``AssertionError`` subclass: the suite fails loudly instead of quietly skipping
the contract.
"""

from __future__ import annotations

import atexit
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import socket
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


_TEST_SHARED_DIR = tempfile.mkdtemp(prefix="tradingbotv3-pytest-shared-")
_TEST_DIAGNOSTICS_DIR = tempfile.mkdtemp(prefix="tradingbotv3-pytest-diagnostics-")
_TEST_LOCAL_APPDATA = tempfile.mkdtemp(prefix="tradingbotv3-pytest-localappdata-")
os.environ["LOCALAPPDATA"] = _TEST_LOCAL_APPDATA
os.environ["TRADINGBOTV3_DATA_DIR"] = _TEST_SHARED_DIR
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = _TEST_DIAGNOSTICS_DIR
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
# Shared-writer role (scripts/writer_role.py). Publishing shared mutable output
# fails closed on an unconfigured machine, and conftest.py has already pointed
# LOCALAPPDATA at an empty temp dir, so the suite has no machine-local settings
# file to read. Name this machine the designated writer so tests about
# publication *mechanics* exercise the happy path. Tests about the role gate
# itself (tests/test_writer_lease_adversarial.py) clear these variables through
# monkeypatch and set their own, so this default cannot mask them.
os.environ.setdefault("TRADINGBOT_DESIGNATED_WRITER", socket.gethostname())
os.environ.setdefault("TRADINGBOT_WRITER_ROLE", "designated_writer")
atexit.register(shutil.rmtree, _TEST_SHARED_DIR, ignore_errors=True)
atexit.register(shutil.rmtree, _TEST_DIAGNOSTICS_DIR, ignore_errors=True)
atexit.register(shutil.rmtree, _TEST_LOCAL_APPDATA, ignore_errors=True)


def _make_multitasking_inert() -> None:
    """Stop yfinance's worker pool from outliving the test session.

    Symptom this fixes: the suite prints a green summary and the interpreter
    then sits for ~20 minutes. A Qt test constructs the desk, whose
    universe-self-heal and industry-board timers fire during the run and reach
    yfinance; yfinance farms its per-ticker downloads out through
    ``multitasking``, which creates its workers with ``daemon=False``, and a few
    hundred of them then park in ``threading._shutdown``. The desk's own threads
    are daemon threads and were never the blocker.

    A pool with ``threads=0`` makes ``@multitasking.task`` call its function
    inline and return None, so no worker is ever created. yfinance calls
    ``set_max_threads()`` at download time, but that only writes MAX_THREADS for
    *future* pools and cannot resurrect this one - verified against
    multitasking's own source, not assumed.

    This does not make the suite hermetic: those calls still go out, just
    serially on the calling thread. Making desk construction inert under pytest
    is the real fix and is deliberately left for after the 2026-08-10 testing
    week (SOL_PROGRESS.md). Import failure is ignored on purpose - multitasking
    arrives as a yfinance dependency, and a headless install without it needs no
    fixing.
    """
    try:
        import multitasking
    except Exception:
        return
    multitasking.createPool(name="pytest-inert", threads=0)


_make_multitasking_inert()


# ---------------------------------------------------------------------------
# Main-thread-only garbage collection (test-harness invariant)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
#
# Automatic (threshold) garbage collection runs on whichever thread happens to
# make the allocation that crosses gen-0's threshold, at an arbitrary point in
# whatever that thread is doing. When such a collection frees a cycle holding
# PySide6/shiboken wrappers, the wrapped QObject destructors run right there -
# off the GUI thread, or re-entrantly in the middle of building a widget. Both
# are undefined behavior in Qt, and both corrupt the heap rather than raising.
#
# This suite is a perfect breeding ground for it: many tests construct real
# desk panels while scanner/bounce/chart worker threads are still alive, so
# there is always another thread allocating and always a supply of Qt wrappers
# in cycles.
#
# Production already fixed exactly this: every GUI session on 2026-07-29 died
# with an access violation inside python314.dll while "Garbage-collecting" on a
# worker thread, and ``ui.app.install_gui_thread_gc`` now disables automatic
# collection and sweeps from a main-thread QTimer instead. The test harness had
# no equivalent, so it kept the hazard the app itself had already retired. This
# block is the harness's counterpart to that function - production keeps its
# collections on the GUI thread with an event-loop timer, the suite keeps them
# on the main thread at test teardown. Neither changes what the other does.
#
# WHAT IS DONE - deliberately the same shape as ``install_gui_thread_gc``:
#
# * session start: one full collect, then ``gc.disable()`` - after this no
#   collection can be triggered by an allocation on any thread;
# * after collection: ``gc.freeze()``, once every test module has been
#   imported. Everything alive at that point is the permanent heap (modules,
#   classes, the pandas/pyarrow import graph), and freezing moves it out of the
#   generations the per-test sweeps have to walk;
# * per test: sweep on the main thread in a teardown hook wrapper, so it runs
#   after the test's fixtures have finalized (notably after
#   ``_drain_chart_workers`` has joined the chart pools) and the heap is quiet.
#   Young generation every test, the whole heap every 25th;
# * session end: unfreeze and re-enable, so pytest plugins and atexit handlers
#   run under the interpreter's normal rules.
#
# WHY GEN-0 PER TEST AND NOT A FULL SWEEP PER TEST
#
# Measured here, not assumed. A full ``gc.collect()`` on this suite's heap costs
# ~85ms, and 2612 of them added ~222s to an ~85s suite (one timed full-suite run
# came in at 340s). A gen-0 sweep costs ~1.7ms. Young-generation collection is
# also the sweep that matters for this hazard: the Qt wrappers a test churns
# through are gen-0 garbage, and that is exactly what production's 2-second tick
# sweeps. Full collections every 25th test keep the older generations from
# growing without bound while costing ~87 x 75ms over a run. Total overhead is
# ~11s on ~85s. The threshold-GC hazard is removed by ``gc.disable()`` alone;
# the sweep cadence only decides how much garbage waits, never which thread
# frees it.
#
# WHAT THIS DOES AND DOES NOT BUY - read this before trusting a green run
#
# It removes one specific mechanism: a collection firing on an arbitrary thread
# at an arbitrary allocation. It is NOT a general fix for this suite's Qt
# lifetime problems, and the measurements say so:
#
#   before (10 runs, gc as the interpreter left it): 8 clean, one exit 139, one
#     exit 134. The segfault's faulthandler traceback landed on the main thread
#     inside ``AlertCenterPanel.__init__`` at the bare
#     ``self.min_tier_input = QComboBox()`` allocation - a widget allocation is
#     not a crash site, but a threshold collection triggered BY that allocation
#     is, which is the evidence this block acts on.
#   after (12 consecutive runs with this block): 12 clean, exit 0, junit
#     failures=0 errors=0, 2605 passed / 5 skipped / 7 subtests every time.
#     Wall 90-108s against an 82-92s baseline.
#
# But one crash was seen during a DISCARDED first attempt at this block (the
# variant that ran a full collect after every test), and it is the reason for
# the paragraph above. It hit ``test_qt_industry_panel`` with the main thread
# parked in ``time.sleep(0.01)`` - a sleeping thread cannot fault on its own,
# and automatic gc was already disabled, so that crash was NOT threshold GC. It
# was a leaked service worker thread touching Qt concurrently. Several tests
# leave threads running past their own teardown (the faulthandler dumps show a
# standing crowd of ``bounce_bot_lib.legacy.run_strategy`` threads parked on an
# Event), and nothing in this block joins them.
#
# So: 12/12 is a real improvement over 8/10 but it is not a proof of thread
# safety, and a future crash here is not automatically a regression in this
# block. The remaining work is ownership of those leaked worker threads, which
# lives in the tests and services that start them, not in conftest.
#
# This is a TEST-HARNESS invariant, not a statement about production. The app
# drives its own collections from the GUI event loop; nothing here is imported
# by, or changes the behavior of, any production module.

#: Whether the interpreter had automatic collection on when the session began.
#: Restored verbatim at session end rather than assuming it was enabled.
_GC_ENABLED_AT_SESSION_START = gc.isenabled()

#: Sweep the whole heap every this many tests; young generation on every other.
#: Production's timer uses 30 ticks for the same trade-off.
_FULL_COLLECT_EVERY = 25

_gc_sweeps = 0


def pytest_sessionstart(session: pytest.Session) -> None:
    """Take automatic collection off the table for the whole session."""
    gc.collect()
    gc.disable()


def pytest_collection_finish(session: pytest.Session) -> None:
    """Retire the import-time heap from every later sweep.

    Runs after every test module has been imported, so the objects promoted
    here are the ones that will live for the whole session anyway. Freezing
    them is what keeps the periodic full collections affordable.
    """
    gc.collect()
    gc.freeze()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item):
    """Sweep on the main thread once the test and its fixtures are done.

    A wrapper, not a plain hook: the post-yield half runs after every other
    ``pytest_runtest_teardown`` implementation, which is what puts this sweep
    *after* fixture finalization instead of before it. The ``finally`` keeps
    the sweep honest when a teardown fails (``_drain_chart_workers`` raises by
    design on a leaked worker) - a run that is already failing is exactly when
    the next test least needs a heap full of dead Qt wrappers.

    ``gc.disable()`` is re-asserted every time on purpose: collection is global
    interpreter state, and tests legitimately turn it back on -
    ``tests/test_gui_thread_gc.py`` calls ``gc.enable()`` in its own finally
    while proving that ``install_gui_thread_gc`` disables it. Re-asserting here
    means no test can hand the hazard back to the rest of the session.
    """
    global _gc_sweeps
    try:
        return (yield)
    finally:
        gc.disable()
        _gc_sweeps += 1
        if _gc_sweeps % _FULL_COLLECT_EVERY == 0:
            gc.collect()
        else:
            gc.collect(0)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Give the interpreter its normal collection rules back."""
    gc.unfreeze()
    if _GC_ENABLED_AT_SESSION_START:
        gc.enable()


# ---------------------------------------------------------------------------
# Fixture contract (plan.md Milestone 3)
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Every element of the documented contract, in declaration order.
REQUIRED_CONTRACT_FIELDS = (
    "schema",
    "feature_version",
    "raw_input_keys",
    "raw_input_sha256",
    "acquired_at",
    "universe_version",
    "provider_assumptions",
    "as_of",
    "expected_keys",
    "numeric_tolerance",
    "intentional_difference",
)

#: A tolerance at or above this is not a tolerance, it is a blank cheque.
MAX_DECLARED_TOLERANCE = 1.0

_HEX_DIGITS = frozenset("0123456789abcdef")


class FixtureContractError(AssertionError):
    """A fixture violates the plan.md Milestone 3 fixture contract.

    Deliberately an ``AssertionError``: a fixture that cannot state its own
    provenance must fail the suite, never skip it.
    """


def _canonical_json(payload: Any) -> bytes:
    """The one canonical byte form used for every raw-input hash."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class FixtureContract:
    """A loaded, contract-validated fixture.

    Behaves like the underlying mapping for data access (``contract["events"]``)
    and additionally exposes the contract metadata plus a comparison helper that
    honours the fixture's declared ``numeric_tolerance``.
    """

    name: str
    path: Path
    data: Mapping[str, Any]

    # -- contract metadata ---------------------------------------------------
    @property
    def schema(self) -> str:
        return self.data["schema"]

    @property
    def feature_version(self) -> str:
        return self.data["feature_version"]

    @property
    def universe_version(self) -> str:
        return self.data["universe_version"]

    @property
    def provider_assumptions(self) -> str:
        return self.data["provider_assumptions"]

    @property
    def intentional_difference(self) -> str:
        return self.data["intentional_difference"]

    @property
    def acquired_at(self) -> str:
        return self.data["acquired_at"]

    @property
    def as_of(self) -> str:
        return self.data["as_of"]

    @property
    def tolerance(self) -> float:
        return float(self.data["numeric_tolerance"])

    @property
    def raw_input_keys(self) -> tuple[str, ...]:
        return tuple(self.data["raw_input_keys"])

    @property
    def expected_keys(self) -> tuple[str, ...]:
        return tuple(self.data["expected_keys"])

    @property
    def configuration(self) -> Mapping[str, Any] | None:
        return self.data.get("configuration")

    # -- data access ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def raw_input(self) -> Any:
        """The exact payload covered by ``raw_input_sha256``."""
        keys = self.raw_input_keys
        if len(keys) == 1:
            return self.data[keys[0]]
        return {key: self.data[key] for key in keys}

    def raw_input_digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.raw_input())).hexdigest()

    # -- comparison ----------------------------------------------------------
    def matches(self, actual: Any, expected: Any) -> bool:
        """Structural comparison that applies the declared numeric tolerance."""
        return _values_match(actual, expected, self.tolerance)

    def assert_matches(self, actual: Any, expected: Any, context: str = "") -> None:
        """Assert ``actual`` equals ``expected`` within the declared tolerance."""
        if self.matches(actual, expected):
            return
        where = f" [{context}]" if context else ""
        raise AssertionError(
            f"{self.name}{where}: expected {expected!r}, got {actual!r} "
            f"(numeric_tolerance={self.tolerance!r})"
        )


def _values_match(actual: Any, expected: Any, tolerance: float) -> bool:
    # bool is an int subclass; never let True compare equal to 1.0.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and actual == expected
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
        if math.isinf(expected) or math.isinf(actual):
            return float(expected) == float(actual)
        return abs(float(actual) - float(expected)) <= tolerance
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return False
        return all(_values_match(actual[key], expected[key], tolerance) for key in expected)
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if isinstance(actual, (str, bytes)) or not isinstance(actual, Sequence):
            return False
        if len(actual) != len(expected):
            return False
        return all(
            _values_match(left, right, tolerance) for left, right in zip(actual, expected)
        )
    return actual == expected


def _fail(name: str, message: str) -> None:
    raise FixtureContractError(f"{name}: {message} (plan.md Milestone 3 fixture contract)")


def _require_text(name: str, data: Mapping[str, Any], field: str) -> None:
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        _fail(name, f"{field!r} must be a non-empty string, got {value!r}")


def _require_timestamp(name: str, data: Mapping[str, Any], field: str) -> None:
    value = data[field]
    if not isinstance(value, str):
        _fail(name, f"{field!r} must be an ISO-8601 timestamp string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail(name, f"{field!r} is not a valid ISO-8601 timestamp: {value!r}")
        return
    if parsed.tzinfo is None:
        _fail(name, f"{field!r} must carry a UTC offset (exact time), got {value!r}")


def _require_key_list(name: str, data: Mapping[str, Any], field: str) -> None:
    value = data[field]
    if not isinstance(value, list) or not value:
        _fail(name, f"{field!r} must be a non-empty list of fixture section names, got {value!r}")
    for key in value:
        if not isinstance(key, str) or not key:
            _fail(name, f"{field!r} entries must be non-empty strings, got {key!r}")
        if key not in data:
            _fail(name, f"{field!r} names {key!r}, which the fixture does not contain")


def _require_tolerance(name: str, data: Mapping[str, Any]) -> None:
    value = data["numeric_tolerance"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(name, f"'numeric_tolerance' must be a real number, got {value!r}")
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        _fail(name, f"'numeric_tolerance' must be finite, got {value!r}")
    if numeric < 0.0:
        _fail(name, f"'numeric_tolerance' must not be negative, got {value!r}")
    if numeric >= MAX_DECLARED_TOLERANCE:
        _fail(
            name,
            f"'numeric_tolerance' must be below {MAX_DECLARED_TOLERANCE} to mean anything, "
            f"got {value!r}",
        )


def _require_sha256(name: str, data: Mapping[str, Any]) -> None:
    value = data["raw_input_sha256"]
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX_DIGITS:
        _fail(name, f"'raw_input_sha256' must be 64 lowercase hex digits, got {value!r}")


def validate_fixture_contract(payload: Mapping[str, Any], name: str) -> FixtureContract:
    """Validate ``payload`` against the Milestone 3 contract.

    Raises :class:`FixtureContractError` on the first violation.
    """
    if not isinstance(payload, Mapping):
        _fail(name, f"fixture must be a JSON object, got {type(payload).__name__}")
    missing = [field for field in REQUIRED_CONTRACT_FIELDS if field not in payload]
    if missing:
        _fail(name, f"missing required contract field(s): {', '.join(missing)}")

    for field in ("schema", "feature_version", "universe_version", "provider_assumptions"):
        _require_text(name, payload, field)
    # A characterization fixture legitimately has no intentional difference, so
    # an empty string is allowed - but the approval note must still be declared.
    if not isinstance(payload["intentional_difference"], str):
        _fail(
            name,
            "'intentional_difference' must be a string ('' for a pure "
            f"characterization fixture), got {payload['intentional_difference']!r}",
        )
    for field in ("acquired_at", "as_of"):
        _require_timestamp(name, payload, field)
    for field in ("raw_input_keys", "expected_keys"):
        _require_key_list(name, payload, field)
    _require_tolerance(name, payload)
    _require_sha256(name, payload)
    if "configuration" in payload and not isinstance(payload["configuration"], Mapping):
        _fail(name, f"'configuration' must be an object, got {payload['configuration']!r}")

    contract = FixtureContract(name=name, path=FIXTURES_DIR / f"{name}.json", data=payload)
    digest = contract.raw_input_digest()
    if digest != payload["raw_input_sha256"]:
        _fail(
            name,
            "raw input hash mismatch for section(s) "
            f"{', '.join(contract.raw_input_keys)}: fixture declares "
            f"{payload['raw_input_sha256']}, recomputed {digest}",
        )
    return contract


def load_fixture_contract(fixture: str | Path) -> FixtureContract:
    """Load a fixture by name (under ``tests/fixtures``) or by explicit path.

    The raw-input hash is recomputed and verified on every load, so a fixture
    whose inputs were edited without re-freezing its expectations fails here
    rather than silently changing what a golden test proves.
    """
    path = Path(fixture)
    if path.suffix != ".json":
        path = FIXTURES_DIR / f"{fixture}.json"
    name = path.stem
    if not path.is_file():
        _fail(name, f"fixture file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(name, f"fixture is not valid JSON: {exc}")
        raise  # pragma: no cover - _fail always raises
    contract = validate_fixture_contract(payload, name)
    return FixtureContract(name=name, path=path, data=contract.data)


# ---------------------------------------------------------------------------
# Bounded retirement of leaked scanners (P1.1, 2026-08-24).
#
# MEASURED FIRST. A full-suite run with a thread-recording plugin found 22 tests
# leaving at least one thread running past their own teardown, and 19
# ``run_strategy`` threads still alive when the session ENDED. That is the
# "standing crowd of bounce_bot_lib.legacy.run_strategy threads" the
# garbage-collection block above names in its own honesty paragraph: it removed
# the threshold-GC hazard and said plainly that it joined none of these.
#
# A strategy loop is not an idle thread. It re-reads the watchlist files,
# refreshes learning state, reloads Focus picks, and on a desk with TWS open
# would dial IB. One left running shares mutable state with every later test.
#
# BounceBot already implements cooperative shutdown - ``stop(timeout=...)`` sets
# its stop event, disconnects, and joins its own threads - so the harness does
# not need any new machinery, only to CALL it. A thread that survives anyway is
# named and fails the leaking test, on the same rule as the chart drain above:
# a teardown that swallows the timeout cannot prove the quiescence it exists to
# provide.
#
# Deliberately narrow. The other leaks the same measurement found
# (``scan-*-drain``, ``qt-health-audit``, ``industry-board-refresh``) finish on
# their own and were gone before the session ended; a blanket "no thread may
# outlive its test" rule would fail 22 tests to fix 19 threads that one rule
# already covers.
# ---------------------------------------------------------------------------

#: Per-bot budget for the cooperative stop. Bounded, because a wedged worker
#: must cost one slow teardown rather than a hung suite.
BOUNCE_RETIREMENT_TIMEOUT_SECONDS = 5.0


def owning_bounce_bot(thread):
    """The BounceBot whose strategy loop this thread runs, or None.

    Identified through the thread's own target rather than its name: a name is
    a label anyone can set, while ``_target.__self__`` IS the object that owns
    the loop and the only handle that can stop it.
    """
    target = getattr(thread, "_target", None)
    if getattr(target, "__name__", "") != "run_strategy":
        return None
    owner = getattr(target, "__self__", None)
    if owner is None or not callable(getattr(owner, "stop", None)):
        return None
    return owner


def retire_leaked_bounce_bots(threads, *, timeout=BOUNCE_RETIREMENT_TIMEOUT_SECONDS):
    """Cooperatively stop every bot behind ``threads``; return what still runs.

    One ``stop()`` per BOT, not per thread: a bot's own stop joins its strategy,
    API and evidence threads together, so calling it twice would only spend the
    budget again.
    """
    import logging

    owned = []
    bots = {}
    for thread in threads:
        bot = owning_bounce_bot(thread)
        if bot is not None:
            owned.append(thread)
            bots.setdefault(id(bot), bot)
    failures = []
    for bot in bots.values():
        try:
            bot.stop(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - returned to the fixture below
            logging.exception("test teardown: BounceBot.stop() raised")
            failures.append(
                f"BounceBot.stop() raised {type(exc).__name__}: {exc}"
            )
    # Only threads this function was RESPONSIBLE for are reported. A thread it
    # never tried to stop is not something it can honestly call a leak.
    failures.extend(thread.name for thread in owned if thread.is_alive())
    return failures


@pytest.fixture(autouse=True)
def _retire_bounce_bots():
    """Let no scanner outlive the test that started it."""
    import threading as _threading

    before = {id(thread) for thread in _threading.enumerate()}
    yield
    strayed = [
        thread
        for thread in _threading.enumerate()
        if id(thread) not in before and owning_bounce_bot(thread) is not None
    ]
    if not strayed:
        return
    still_running = retire_leaked_bounce_bots(strayed)
    if still_running:
        raise AssertionError(
            "BounceBot teardown failed for thread(s) started by this test: "
            + ", ".join(still_running)
        )


@pytest.fixture(autouse=True)
def _drain_chart_workers():
    """Let no chart worker outlive the test that queued it.

    Chart snapshots and prefetch batches run on pooled threads, and a test
    that queues work without waiting for it leaves those threads reading
    parquet during whatever test runs next. That is invisible until it lands
    on a timing-sensitive one - a socket handshake in the retired Desk Link
    suite failed once this way - and an intermittent failure in an unrelated
    file is far more
    expensive to chase than this wait, which is a no-op when the pools are
    idle.

    Two pools matter, not one: the module-level shared pool (_POOL) AND the
    shared service's own pool - a default-constructed ChartDataService owns a
    private QThreadPool, so draining _POOL alone missed every worker the
    Alert Center's prefetch queued through shared_service(). And a drain that
    times out FAILS the leaking test by name instead of passing silently -
    a fixture that swallows the timeout cannot prove the quiescence it
    exists to provide.
    """
    yield
    module = sys.modules.get("ui.services.chart_data_service")
    if module is None:
        return
    stalled: list[str] = []
    pool = module._POOL
    if pool is not None and not pool.waitForDone(5000):
        stalled.append("shared pool (_POOL)")
    service = module._SERVICE
    if service is not None and not service.wait_for_idle(5000):
        stalled.append("shared service pool (_SERVICE._pool)")
    if stalled:
        raise AssertionError(
            "chart workers queued by this test were still running at teardown: "
            + ", ".join(stalled)
        )


# ---------------------------------------------------------------------------
# The offline tripwire (hermetic-suite packet, 2026-08-18).
#
# The deterministic suite was only ever ACCIDENTALLY offline. Evening runs
# looked clean because R1's quiet-hours gate (`auto_scanning_due`, weekdays
# 06:00-14:00 PT) was closed; a market-hours run on 2026-08-18 showed the same
# suite connecting to IB, rebuilding a 1,536-symbol universe through
# `fetch_market_caps`, and pulling real SPY quotes. Wall time moved ~152 s ->
# ~217 s and one run passed ten minutes.
#
# So: no unmarked test may open a socket. `network` and `broker` are the
# opt-outs that already existed for the handful of tests whose point IS the
# wire.
#
# LOOPBACK IS NOT EXEMPT, deliberately. TWS listens on 127.0.0.1:7496, so an
# exemption for "local" traffic would wave through the single worst leak this
# guard exists to catch.
#
# Attempts are RECORDED as well as refused, and recorded attempts fail the test
# at teardown. A raise alone is not enough: the universe rebuild runs on a
# background thread, where the exception is logged and swallowed, and the test
# would pass while still having tried to reach the wire.
# ---------------------------------------------------------------------------

_SOCKET_OPT_OUT_MARKERS = ("network", "broker")


@pytest.fixture(autouse=True)
def _no_ib_session(request, monkeypatch):
    """Stop the IB client dialling TWS from an unmarked test.

    The tripwire refuses the socket, but refusing is not the same as not
    trying: `EClient.connect` spins its reader thread and retries around the
    failure, which is live-broker machinery running inside a deterministic
    suite. 132 of the 2026-08-18 inventory's attempts were this one seam.

    This stubs the ADAPTER BOUNDARY, not behaviour - the client simply never
    dials, which is the same state as a desk with TWS closed.
    """
    # Opt-out is `broker` ONLY, not `network`. The two markers mean different
    # things: `network` is "this test uses the internet", `broker` is "this
    # test needs a live IBKR session". Letting `network` disable the IB stub is
    # how the (since-retired) Desk Link wire tests quietly reconnected to TWS
    # after they were marked - a real connection, logged as "Sec-def data farm
    # connection is OK", inside a suite meant to be offline.
    if request.node.get_closest_marker("broker"):
        yield
        return
    try:
        from ibapi.client import EClient
    except ModuleNotFoundError:
        yield
        return

    def refuse(self, host, port, clientId):  # noqa: N803 - ibapi's own spelling
        raise ConnectionRefusedError(
            f"offline suite: {request.node.nodeid} tried to reach TWS at {host}:{port}"
        )

    monkeypatch.setattr(EClient, "connect", refuse, raising=True)
    yield


@pytest.fixture(autouse=True)
def _no_market_prep_feeds(request, monkeypatch):
    """Stop the two market-prep calendar adapters dialling out.

    Both already handle a failed fetch gracefully - the orchestrator logs
    "ForexFactory scrape failed" and carries on - which is exactly why they
    were invisible until the tripwire recorded the attempt. Graceful is not
    hermetic.

    Stubbed at the fetch boundary, so every parse, merge and date-window rule
    above it still runs against the same shapes it always did.
    """
    if any(request.node.get_closest_marker(name) for name in _SOCKET_OPT_OUT_MARKERS):
        yield
        return
    for module_name, attribute, empty in (
        ("market_prep.services.forexfactory_calendar_service", "_refresh_events", ([], [])),
        ("market_prep.services.treasury_calendar_service", "_fetch_upcoming_auctions", []),
        # The Fed calendar/RSS adapter, found 2026-08-18 during the R1-R8
        # integration merge: it reaches www.federalreserve.gov from the daily
        # prep orchestrator, and it only showed up in a FULL-suite run because
        # in isolation the on-disk cache answered first. One boundary covers
        # both callers (the monthly HTML calendar and the three RSS feeds).
        # An empty RSS document is the "no events, no error" shape: the HTML
        # section walk finds no sections in it and the XML parse finds no
        # items, so every parse and date-window rule above still runs and
        # nothing logs a spurious failure.
        (
            "market_prep.services.fed_calendar_service",
            "_fetch_text",
            '<rss version="2.0"><channel></channel></rss>',
        ),
        # The NASDAQ earnings calendar, reached from a BACKGROUND thread inside
        # master_avwap_lib.legacy - which is why its violation kept landing on
        # innocent tests. Its contract is (rows, error), and "no rows, no
        # error" is the shape it already returns for a day with no earnings,
        # so nothing downstream sees a state it has not handled before. The
        # file is ask-first and is NOT edited: this is a test-side stub.
        ("master_avwap_lib.legacy", "_fetch_nasdaq_earnings_rows_with_retries", ([], None)),
        # The universe rebuild - the leak that started this packet. It runs on
        # a background thread from autopilot_core.rebuild_universe_if_stale,
        # and on 2026-08-18 it walked 1,536 symbols against the live wire in
        # the middle of a deterministic suite. Empty list / empty dict are the
        # shapes both already return when their caches are cold and the fetch
        # yields nothing.
        # ALL five of universe_builder's fetch entry points, not the two that
        # happened to fire first. Chasing them one full-suite run at a time is
        # how a guard like this gets declared done while a fourth path is still
        # dialling out on a code path no test exercised that day.
        ("universe_builder", "fetch_all_listed_symbols", []),
        ("universe_builder", "fetch_weekly_option_symbols", []),
        ("universe_builder", "fetch_optionable_symbols", []),
        ("universe_builder", "fetch_market_caps", {}),
        ("universe_builder", "fetch_price_history", None),
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:  # pragma: no cover - optional tree
            continue
        if hasattr(module, attribute):
            # Stash the original so a test that stubs the layer BELOW this one
            # (test_earnings_collect_speed fakes requests.get to exercise the
            # real retry/batching logic) can put it back in one line and stay
            # hermetic while doing it.
            setattr(module, f"_offline_original_{attribute}", getattr(module, attribute))
            if empty is None and attribute == "fetch_price_history":
                import pandas as _pd

                empty = _pd.DataFrame()
            monkeypatch.setattr(module, attribute, lambda *a, _e=empty, **k: _e, raising=True)
    yield


@pytest.fixture(autouse=True)
def _no_yfinance_fetch(request, monkeypatch):
    """The third adapter boundary: the fallback bar provider.

    `yfinance` is the fallback quote/bar source, and the inventory caught it
    being dialled from a BACKGROUND thread - which is why the failure landed on
    whichever test happened to be running rather than the one that started the
    work. Refusing at the boundary makes the reach-out impossible instead of
    merely unlucky.

    Tests that stub yfinance themselves override this, as they already did.
    """
    if any(request.node.get_closest_marker(name) for name in _SOCKET_OPT_OUT_MARKERS):
        yield
        return
    try:
        import yfinance
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        yield
        return

    def refuse(*args, **kwargs):
        raise ConnectionRefusedError(
            f"offline suite: {request.node.nodeid} tried to fetch bars from yfinance"
        )

    for attribute in ("download", "Ticker", "Tickers"):
        if hasattr(yfinance, attribute):
            monkeypatch.setattr(yfinance, attribute, refuse, raising=True)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Remember whether the test body already failed.

    Without this the teardown check reports a SECOND failure for the same
    violation - one FAILED and one ERROR per leak - which turns the inventory
    this guard exists to produce into twice the noise.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        item._offline_tripwire_already_failed = True


class OfflineSuiteViolation(AssertionError, OSError):
    """A test that never declared it needs the wire tried to use it.

    Deliberately an OSError as well as an AssertionError. Product code that
    already handles "TWS is not running" catches OSError, so inheriting from it
    makes a refused connection behave exactly as it does on a machine with no
    broker - graceful, not a novel crash path. The attempt is still recorded,
    so the teardown check fails the test anyway: the goal is that no test
    REACHES for the wire, not that it survives being denied.
    """


@pytest.fixture(autouse=True)
def _offline_tripwire(request, monkeypatch):
    if any(request.node.get_closest_marker(name) for name in _SOCKET_OPT_OUT_MARKERS):
        yield
        return

    import socket as _socket

    test_id = request.node.nodeid
    attempts: list[str] = []

    def _refuse(where: str, address: object):
        import traceback as _traceback

        target = repr(address)
        # Name the CALLER, not just the address. A background-thread attempt is
        # reported against whichever test happens to be running, so "something
        # dialled 23.6.255.188" without a stack sends the reader hunting
        # through an innocent file. The two frames below the guard are the ones
        # that identify the real seam.
        origin = " <- ".join(
            f"{Path(frame.filename).name}:{frame.lineno} {frame.name}"
            for frame in reversed(_traceback.extract_stack()[-14:-1])
            if "site-packages" not in frame.filename
        )
        attempts.append(f"{where} -> {target} from {origin}")
        raise OfflineSuiteViolation(
            f"{test_id} tried to open a socket ({where} -> {target}). "
            "The suite is hermetic: stub the seam, or mark the test "
            "@pytest.mark.network / @pytest.mark.broker if live I/O is its point."
        )

    real_connect = _socket.socket.connect
    real_connect_ex = _socket.socket.connect_ex
    real_create_connection = _socket.create_connection

    def guarded_connect(self, address, *args, **kwargs):
        _refuse("socket.connect", address)

    def guarded_connect_ex(self, address, *args, **kwargs):
        _refuse("socket.connect_ex", address)

    def guarded_create_connection(address, *args, **kwargs):
        _refuse("socket.create_connection", address)

    monkeypatch.setattr(_socket.socket, "connect", guarded_connect, raising=True)
    monkeypatch.setattr(_socket.socket, "connect_ex", guarded_connect_ex, raising=True)
    monkeypatch.setattr(_socket, "create_connection", guarded_create_connection, raising=True)
    try:
        yield
    finally:
        monkeypatch.setattr(_socket.socket, "connect", real_connect, raising=True)
        monkeypatch.setattr(_socket.socket, "connect_ex", real_connect_ex, raising=True)
        monkeypatch.setattr(_socket, "create_connection", real_create_connection, raising=True)
    if attempts and not getattr(request.node, "_offline_tripwire_already_failed", False):
        # Reached when a background thread swallowed the raise. The test looked
        # green; it was not.
        raise OfflineSuiteViolation(
            f"{test_id} attempted live I/O on a background thread: " + "; ".join(sorted(set(attempts)))
        )
