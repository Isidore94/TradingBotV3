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
import hashlib
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


@pytest.fixture(autouse=True)
def _drain_chart_workers():
    """Let no chart worker outlive the test that queued it.

    Chart snapshots and prefetch batches run on pooled threads, and a test
    that queues work without waiting for it leaves those threads reading
    parquet during whatever test runs next. That is invisible until it lands
    on a timing-sensitive one - a Desk Link socket handshake failed once this
    way - and an intermittent failure in an unrelated file is far more
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
