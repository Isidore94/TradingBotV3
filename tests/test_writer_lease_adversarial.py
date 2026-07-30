"""Independent safety spec for the cross-machine writer protection (Packet L1).

Written FIRST, as executable tests. Several of these fail against today's
``scripts/writer_lease.py`` + ``scripts/autopilot_core.publish_away_report``;
a failure here is a finding about the implementation, not a defect in the test.

WHAT THIS FILE MAY AND MAY NOT CLAIM (plan.md sec 4)
----------------------------------------------------
A Google Drive-synchronized file is NOT a compare-and-swap lock. Two machines
can race before sync converges. Nothing below proves distributed exclusion or
that clobbering is impossible. Every test here is about *degrading honestly*
when the configuration is missing, when this machine is not the designated
writer, and when the lease file is absent, half-synced, corrupt, locked,
stale, or old-format -- plus the master invariant that a blocked publish leaves
the last verified report and its metadata byte-identical.

MECHANISM NEUTRALITY
--------------------
Every assertion is on OBSERVABLE SAFETY BEHAVIOR:

* "an unconfigured machine does not modify the report, metadata, or lease" --
  never "function ``_foo`` raised";
* "concurrent publishers produce at most one winner and an intact report" --
  never "a named mutex was acquired".

The implementation is free to choose its exclusion primitive, its config keys,
its telemetry field names and its lease schema. Where a test has to name
something to be executable at all, it accepts a *family* of spellings (see
``_CONFIG_*``, ``_GENERATION_KEY_HINTS``, ``_TELEMETRY_CONCEPTS``,
``_HEALTH_READER_HINTS``) and says so in the failure message.

Two seams are used as-is because they already exist in ``writer_lease``:
``acquire(..., holder=..., now=...)``. They are the only way to represent
"the other machine" and "later" inside one test process. An implementation
that reshapes them should keep an equivalent injection point.

METHODOLOGY
-----------
Nothing here monkeypatches ``acquire`` to fake a corrupt lease. Every
corruption is real bytes on a real disk, a real directory-in-place-of-a-file,
or a real Win32 exclusive-share (dwShareMode=0) handle. Multi-process cases
use real ``subprocess`` children with distinct process instances.

SCENARIO MAP (the letters are the packet's A-U)
    A  unconfigured designated writer -> touches nothing, fails visibly
    B  configured as secondary -> read-only, refuses, says why
    C  configured as designated writer -> happy path works end to end
    D  missing lease file (the only state that may permit acquisition)
    E  truncated JSON lease (real half-written bytes)
    F  malformed JSON lease
    G  real OSError on read (directory-at-path / Win32 deny-sharing handle)
    H  wrong schema / unknown version / missing required keys
    I  missing or invalid holder; missing or invalid expiry
    J  expired but otherwise valid lease -> defined acquisition path only
    K  OLD-FORMAT lease with no instance id -> never "ours"
    L  simultaneous acquisition -> exactly one winner
    M  two processes on ONE hostname -> never conflated (real subprocesses)
    N  PID reuse / restart -> no inherited ownership (real subprocesses)
    O  ownership lost between acquisition and publication -> abort first
    P  fencing generation changes mid-render -> abort before replacement
    Q  sleep/wake or missed renewal -> ownership lost, re-acquire first
    R  expiry boundaries + bounded clock skew, and no permanent lockout
    S  emergency takeover: explicit, time-bounded, auditable, visible
    T  Layer 5 telemetry present and readable; corrupt never reads healthy
    U  partial failure -> report and metadata never disagree
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import writer_lease as wl  # noqa: E402

NOW = datetime(2026, 7, 30, 9, 0)
LATER = NOW + timedelta(minutes=3)

#: Repetition counts. A single round can serialize by luck; the guarantee has
#: to hold every time, not on average.
RACE_ROUNDS = 25
PUBLISH_RACE_ROUNDS = 40
SUBPROCESS_RACE_ROUNDS = 2

FULL_PAYLOAD = {
    "generated_at": "2026-07-30T09:00:00",
    "enabled": True,
    "auto_mode": "AWAY",
    "ib_status": "connected",
    "regime": "risk-on",
    "longs": ["AAPL"],
    "shorts": [],
    "swing_picks": [],
    "alerts": [],
    "slots_done": [],
    "next_slot": "10:00",
    "log_lines": [],
    "auto_longs": [],
    "auto_shorts": [],
}


# ==========================================================================
# helpers: identity, configuration, on-disk lease states
# ==========================================================================
#: Machine-local settings keys an implementation may use to name the writer.
#: The repo's existing machine-local convention is
#: ``project_paths.get_local_setting`` / ``LOCAL_SETTINGS_FILE``.
_CONFIG_WRITER_KEYS = (
    "designated_writer",
    "designated_writer_machine",
    "shared_writer_machine",
    "autopilot_designated_writer",
)
_CONFIG_ROLE_KEYS = (
    "writer_role",
    "shared_writer_role",
    "autopilot_writer_role",
)
_CONFIG_OVERRIDE_KEYS = (
    "writer_emergency_takeover",
    "emergency_takeover",
    "writer_override",
)
_CONFIG_OVERRIDE_EXPIRY_KEYS = (
    "writer_emergency_takeover_expires_at",
    "emergency_takeover_expires_at",
    "writer_override_expires_at",
)
_ENV_WRITER_KEYS = (
    "TRADINGBOT_DESIGNATED_WRITER",
    "TRADINGBOTV3_DESIGNATED_WRITER",
)
_ENV_ROLE_KEYS = (
    "TRADINGBOT_WRITER_ROLE",
    "TRADINGBOTV3_WRITER_ROLE",
)
_ENV_OVERRIDE_KEYS = (
    "TRADINGBOT_WRITER_OVERRIDE",
    "TRADINGBOT_LEASE_TAKEOVER",
    "WRITER_LEASE_FORCE",
)

_CONFIG_FAILURE_WORDS = (
    "designated",
    "configur",
    "role",
    "read-only",
    "read only",
    "readonly",
    "secondary",
    "not the writer",
)


def _pretend_machine(monkeypatch, name: str) -> None:
    """Make every plausible hostname source agree that we are ``name``."""
    monkeypatch.setattr(socket, "gethostname", lambda: name)
    monkeypatch.setattr(platform, "node", lambda: name)
    monkeypatch.setenv("COMPUTERNAME", name)


def _valid_lease(holder: str = "home-desk", *, expires: datetime | None = None) -> dict:
    """A hand-built lease payload in the CURRENT on-disk shape.

    Only used where the test needs precise control of the bytes (corruption,
    missing fields). Wherever a *genuinely valid* lease is needed the tests use
    ``wl.acquire`` itself, so the implementation's own format is exercised.
    """
    expires = expires or (NOW + timedelta(minutes=10))
    return {
        "schema": wl.LEASE_SCHEMA,
        "holder": holder,
        "acquired_at": NOW.isoformat(timespec="seconds"),
        "expires_at": expires.isoformat(timespec="seconds"),
        "takeover": False,
        "clock_skew_seconds": wl.DEFAULT_CLOCK_SKEW_SECONDS,
    }


def _put_raw(path: Path, raw: str | bytes) -> Path:
    """Write literal bytes to the lease path -- no helper, no atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    path.write_bytes(data)
    return path


def _put_json(path: Path, payload) -> Path:
    return _put_raw(path, json.dumps(payload))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_lease_by(lease: Path, holder: str, *, now: datetime = NOW, ttl: int = 10) -> dict:
    """A real, unexpired lease written by the implementation itself."""
    return wl.acquire(lease, holder=holder, ttl_minutes=ttl, now=now)


def _expired_lease_by(lease: Path, holder: str, *, expired_for=timedelta(hours=2)) -> dict:
    """A real, VALID lease written by the implementation that has since expired."""
    return wl.acquire(lease, holder=holder, ttl_minutes=10, now=NOW - expired_for)


def _expect_blocked(lease: Path, **kwargs) -> Exception:
    """acquire() must refuse, with a LeaseUnavailable (or a subclass).

    Blocking is asserted as ``LeaseUnavailable`` because every publish path in
    the repo already fails closed on that type; an implementation is free to
    raise a narrower subclass (``LeaseUnreadable``, ...).
    """
    kwargs.setdefault("holder", "mini-pc")
    kwargs.setdefault("now", LATER)
    try:
        payload = wl.acquire(lease, **kwargs)
    except wl.LeaseUnavailable as exc:
        assert str(exc).strip(), "the loser must be told a reason, not an empty message"
        return exc
    except Exception as exc:  # noqa: BLE001 - reported as a failure, not swallowed
        pytest.fail(
            "acquire() must fail closed with LeaseUnavailable (or a subclass of it, "
            f"so existing publish paths still catch it); it raised {type(exc).__name__}: {exc!r}"
        )
    pytest.fail(f"acquire() silently took a questionable lease and wrote {payload!r}")
    raise AssertionError  # pragma: no cover


def _summarized_as_free(lease: Path, **kwargs) -> bool:
    """Did the lease summary claim 'nobody holds this'? Refusing to answer is honest."""
    try:
        answer = wl.holder_of(lease, **kwargs)
    except Exception:  # noqa: BLE001 - an honest refusal is not "free"
        return False
    return answer in (None, "")


@contextmanager
def _exclusively_locked(path: Path):
    """A REAL Win32 exclusive-share handle (dwShareMode=0) on the lease file.

    The closest available analogue of Google Drive holding the file open
    mid-sync: any other reader gets a genuine PermissionError. Deliberately not
    ``chmod`` (a no-op for this on Windows) and not a monkeypatch.
    """
    if sys.platform != "win32":  # pragma: no cover - Windows-only trick
        pytest.skip("Win32 exclusive-share lock is not portable")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,  # GENERIC_READ
        0,  # dwShareMode = 0 -> exclusive
        None,
        3,  # OPEN_EXISTING
        0x80,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle in (0, None, -1, 2**64 - 1, 2**32 - 1):  # pragma: no cover
        pytest.skip("could not take an exclusive handle on the lease file")
    try:
        with pytest.raises(OSError):
            path.read_text(encoding="utf-8")  # prove the lock is real
        yield
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


# ==========================================================================
# fixtures: machine-local configuration + one genuinely verified publication
# ==========================================================================
class WriterConfig:
    """Writes the designated-writer configuration the way a machine would.

    The implementation may read the role from the repo's machine-local settings
    file (``project_paths.LOCAL_SETTINGS_FILE``, redirected here to ``tmp_path``)
    or from an environment variable; this helper sets every plausible spelling of
    both at once, so the assertions below are about the machine's *behavior*, not
    about which key it happened to read. A synced-from-Drive role file is
    deliberately never used: it has the convergence problem it would be solving.
    """

    def __init__(self, monkeypatch, settings_file: Path):
        self._monkeypatch = monkeypatch
        self.settings_file = settings_file
        self.unconfigured()

    # -- surfaces ----------------------------------------------------------
    def _write_settings(self, payload: dict) -> None:
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _clear_env(self) -> None:
        for key in _ENV_WRITER_KEYS + _ENV_ROLE_KEYS + _ENV_OVERRIDE_KEYS:
            self._monkeypatch.delenv(key, raising=False)

    # -- states ------------------------------------------------------------
    def unconfigured(self) -> None:
        """No designated writer anywhere. Must fail closed, not 'first wins'."""
        self._clear_env()
        self.settings_file.unlink(missing_ok=True)
        self._write_settings({})

    def designate(self, machine: str) -> None:
        """This configuration names ``machine`` as THE writer."""
        payload = {key: machine for key in _CONFIG_WRITER_KEYS}
        payload.update({key: "designated_writer" for key in _CONFIG_ROLE_KEYS})
        self._clear_env()
        self._write_settings(payload)
        for key in _ENV_WRITER_KEYS:
            self._monkeypatch.setenv(key, machine)
        for key in _ENV_ROLE_KEYS:
            self._monkeypatch.setenv(key, "designated_writer")

    def secondary(self, writer_machine: str) -> None:
        """Someone ELSE is the writer; this machine is a read-only secondary."""
        payload = {key: writer_machine for key in _CONFIG_WRITER_KEYS}
        payload.update({key: "secondary" for key in _CONFIG_ROLE_KEYS})
        self._clear_env()
        self._write_settings(payload)
        for key in _ENV_WRITER_KEYS:
            self._monkeypatch.setenv(key, writer_machine)
        for key in _ENV_ROLE_KEYS:
            self._monkeypatch.setenv(key, "secondary")

    def emergency_override(self, value, *, expires_at=None, machine: str | None = None) -> None:
        """Set the emergency-takeover configuration to a literal value."""
        payload = json.loads(self.settings_file.read_text(encoding="utf-8"))
        if machine is not None:
            payload.update({key: machine for key in _CONFIG_WRITER_KEYS})
        payload.update({key: value for key in _CONFIG_OVERRIDE_KEYS})
        if expires_at is not None:
            payload.update({key: expires_at for key in _CONFIG_OVERRIDE_EXPIRY_KEYS})
        self._write_settings(payload)
        for key in _ENV_OVERRIDE_KEYS:
            self._monkeypatch.setenv(key, str(value))


class Publication:
    """A verified report + its metadata + the lease, with byte snapshots."""

    def __init__(self, core, target: Path):
        self.core = core
        self.target = target
        self.metadata = target.with_suffix(target.suffix + ".meta.json")
        self.lease = target.with_suffix(target.suffix + ".lease")
        self.snapshot()

    def snapshot(self) -> None:
        self.report_sha = _sha(self.target)
        self.metadata_sha = _sha(self.metadata)
        self.lease_sha = _sha(self.lease) if self.lease.exists() else None

    def plant_lease(self, payload) -> None:
        """Put a distinctive lease on disk and re-snapshot, so ANY rewrite shows."""
        _put_json(self.lease, payload)
        self.lease_sha = _sha(self.lease)

    def publish(self, payload=None, **kwargs):
        kwargs.setdefault("archive", False)
        return self.core.publish_away_report(payload or dict(FULL_PAYLOAD), self.target, **kwargs)

    def assert_intact(self, note: str = "", *, lease: bool = False) -> None:
        assert self.target.exists(), f"the verified report was deleted ({note})"
        assert self.metadata.exists(), f"the publication metadata was deleted ({note})"
        assert _sha(self.target) == self.report_sha, (
            f"a failed publish rewrote the last verified report ({note})"
        )
        assert _sha(self.metadata) == self.metadata_sha, (
            f"a failed publish rewrote the publication metadata ({note})"
        )
        if lease:
            assert self.lease_sha is not None
            assert self.lease.exists(), f"the lease file was deleted ({note})"
            assert _sha(self.lease) == self.lease_sha, (
                f"a machine that must not publish still rewrote the lease ({note})"
            )


@pytest.fixture
def diagnostics_dir(tmp_path, monkeypatch) -> Path:
    """Per-test machine-local diagnostics root (Layer 5 telemetry lands here)."""
    target = tmp_path / "diagnostics"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(target))
    return target


@pytest.fixture
def config(tmp_path, monkeypatch) -> WriterConfig:
    """Machine-local settings, redirected away from the real machine's file."""
    import project_paths

    settings_dir = tmp_path / "local_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_DIR", settings_dir, raising=False)
    monkeypatch.setattr(
        project_paths, "LOCAL_SETTINGS_FILE", settings_dir / "local_settings.json", raising=False
    )
    return WriterConfig(monkeypatch, settings_dir / "local_settings.json")


@pytest.fixture
def verified(tmp_path, monkeypatch, config, diagnostics_dir) -> Publication:
    """One real, verified publication by the configured designated writer.

    The machine is ``mini-pc`` throughout; ``home-desk`` is always "the other
    machine". Tests that need a different configuration change it afterwards.
    """
    import autopilot_core as core

    _pretend_machine(monkeypatch, "mini-pc")
    config.designate("mini-pc")
    target = tmp_path / "drive" / "autopilot_today.txt"
    first = core.publish_away_report(
        dict(FULL_PAYLOAD, generated_at="VERIFIED-BASELINE"), target, archive=False
    )
    assert first["ok"] and first["verified"], (
        f"the configured designated writer could not make a baseline publication: {first}"
    )
    return Publication(core, target)


# ==========================================================================
# A. UNCONFIGURED designated writer -> must touch NOTHING
# ==========================================================================
def test_a_unconfigured_machine_publishes_nothing(verified, config):
    verified.plant_lease(_valid_lease("mini-pc", expires=NOW + timedelta(days=3650)))
    config.unconfigured()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, (
        "with no designated writer configured this machine fell back to "
        "'first machine wins' and published anyway"
    )
    verified.assert_intact("unconfigured designated writer", lease=True)


def test_a_unconfigured_failure_is_visible_in_the_publish_result(verified, config):
    config.unconfigured()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    reason = str(result.get("error") or "").lower()
    assert any(word in reason for word in _CONFIG_FAILURE_WORDS), (
        "the publish result must name the CONFIGURATION failure (no designated "
        f"writer), not a generic error: {result.get('error')!r}"
    )


def test_a_unconfigured_machine_does_not_create_a_lease_from_scratch(
    tmp_path, monkeypatch, config, diagnostics_dir
):
    """No previous report, no lease: an unconfigured machine still creates nothing."""
    import autopilot_core as core

    _pretend_machine(monkeypatch, "mini-pc")
    config.unconfigured()
    target = tmp_path / "drive" / "autopilot_today.txt"

    result = core.publish_away_report(dict(FULL_PAYLOAD), target, archive=False)

    assert result["ok"] is False
    assert not target.exists(), "an unconfigured machine wrote the shared report"
    assert not target.with_suffix(target.suffix + ".meta.json").exists()
    assert not target.with_suffix(target.suffix + ".lease").exists(), (
        "an unconfigured machine claimed the lease"
    )


def test_a_unconfigured_failure_is_visible_in_telemetry(verified, config, diagnostics_dir):
    config.unconfigured()

    verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    blob = _telemetry_text(diagnostics_dir)
    assert blob, (
        "a configuration failure must be visible in Health telemetry; nothing "
        f"was written under {diagnostics_dir}"
    )
    assert any(word in blob for word in _CONFIG_FAILURE_WORDS), (
        "telemetry does not record that this machine has no designated-writer "
        f"configuration: {blob[:2000]}"
    )


# ==========================================================================
# B. configured as SECONDARY -> read-only, refuses, says why
# ==========================================================================
def test_b_secondary_machine_refuses_to_publish(verified, config):
    verified.plant_lease(_valid_lease("mini-pc", expires=NOW + timedelta(days=3650)))
    config.secondary("home-desk")

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, (
        "this machine is not the designated writer and published the shared "
        "report anyway"
    )
    verified.assert_intact("configured secondary", lease=True)


def test_b_secondary_machine_says_why_it_is_read_only(verified, config):
    config.secondary("home-desk")

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    reason = str(result.get("error") or "").lower()
    assert any(word in reason for word in _CONFIG_FAILURE_WORDS), (
        f"a read-only secondary must explain the role, got: {result.get('error')!r}"
    )


def test_b_secondary_machine_does_not_touch_a_free_lease(verified, config):
    """Being read-only is decided before any lease is taken (gate order 1 then 3)."""
    verified.lease.unlink(missing_ok=True)
    config.secondary("home-desk")

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    assert not verified.lease.exists(), (
        "a read-only secondary acquired the shared writer lease; the role gate "
        "must come before lease acquisition"
    )
    verified.assert_intact("secondary with a free lease")


def test_b_secondary_state_is_visible_in_telemetry(verified, config, diagnostics_dir):
    config.secondary("home-desk")

    verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    blob = _telemetry_text(diagnostics_dir)
    assert blob, f"no Health telemetry was written under {diagnostics_dir}"
    assert "home-desk" in blob, (
        "telemetry must name the configured designated writer so OFF/DESK/AWAY "
        f"can describe reality: {blob[:2000]}"
    )


# ==========================================================================
# C. configured as the DESIGNATED WRITER -> the happy path really works
# ==========================================================================
def test_c_designated_writer_publishes_end_to_end(verified):
    """The baseline publication in the fixture already proves the happy path."""
    assert verified.target.exists() and verified.metadata.exists()
    metadata = json.loads(verified.metadata.read_text(encoding="utf-8"))
    assert metadata["sha256"] == _sha(verified.target)
    assert "VERIFIED-BASELINE" in verified.target.read_text(encoding="utf-8")


def test_c_designated_writer_can_republish_over_its_own_report(verified):
    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SECOND-PUBLISH"))

    assert result["ok"] and result["verified"], result
    assert "SECOND-PUBLISH" in verified.target.read_text(encoding="utf-8")
    metadata = json.loads(verified.metadata.read_text(encoding="utf-8"))
    assert metadata["sha256"] == _sha(verified.target)


def test_c_designated_writer_publish_is_verified_by_readback(verified):
    result = verified.publish(dict(FULL_PAYLOAD, generated_at="READBACK"))

    assert result["verified"] is True
    assert result["sha256"] == _sha(verified.target), (
        "a publish that reports a hash must report the hash actually on disk"
    )


# ==========================================================================
# D. missing lease file -- the ONLY state that may permit acquisition
# ==========================================================================
def test_d_missing_lease_file_permits_acquisition(tmp_path):
    lease = tmp_path / "nested" / "report.txt.lease"
    assert not lease.exists()

    granted = wl.acquire(lease, holder="mini-pc", now=NOW)

    assert granted["holder"] == "mini-pc"
    assert lease.exists(), "an acquired lease must be visible to the other machine"
    on_disk = json.loads(lease.read_text(encoding="utf-8"))
    assert on_disk["holder"] == "mini-pc"
    assert wl.holder_of(lease, now=NOW + timedelta(minutes=1)) == "mini-pc"


def test_d_missing_lease_permits_a_publish_that_replaces_the_report(verified):
    verified.lease.unlink()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SECOND"))

    assert result["ok"] and result["verified"], result
    assert _sha(verified.target) != verified.report_sha, (
        "with no lease on disk this publish is legitimate and must go through"
    )


# ==========================================================================
# E. truncated JSON -- a real half-written file (Drive sync mid-flight)
# ==========================================================================
TRUNCATIONS = [
    pytest.param(8, id="head-only"),
    pytest.param(40, id="mid-holder"),
    pytest.param(-1, id="missing-final-brace"),
]


@pytest.mark.parametrize("cut", TRUNCATIONS)
def test_e_truncated_lease_blocks_acquisition(tmp_path, cut):
    lease = tmp_path / "report.txt.lease"
    whole = json.dumps(_valid_lease("home-desk"))
    _put_raw(lease, whole[:cut])

    exc = _expect_blocked(lease)
    assert "lease" in str(exc).lower() or lease.name in str(exc)


def test_e_truncated_lease_leaves_the_verified_report_byte_identical(verified):
    _put_raw(verified.lease, json.dumps(_valid_lease("home-desk"))[:37])

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, "a half-synced lease must not authorize a publish"
    assert result["error"], "the loser must be given a reason"
    verified.assert_intact("truncated lease")


def test_e_truncated_lease_is_never_reported_as_free(tmp_path):
    lease = tmp_path / "report.txt.lease"
    _put_raw(lease, json.dumps(_valid_lease("home-desk"))[:37])

    assert not _summarized_as_free(lease, now=LATER), (
        "an unreadable lease must never be summarized as 'nobody holds it'"
    )


# ==========================================================================
# F. malformed JSON -- syntactically invalid bytes
# ==========================================================================
MALFORMED = [
    pytest.param("", id="empty-file"),
    pytest.param("   \n\t ", id="whitespace-only"),
    pytest.param("\x00\x00\x00\x00", id="nul-bytes"),
    pytest.param("{holder: home-desk,,,}", id="not-json"),
    pytest.param("{'holder': 'home-desk'}", id="python-repr-not-json"),
    pytest.param('{"holder": "home-desk", }', id="trailing-comma"),
    pytest.param("<<<<<<< HEAD\n{}\n=======", id="conflict-markers"),
]


@pytest.mark.parametrize("raw", MALFORMED)
def test_f_malformed_lease_blocks_acquisition(tmp_path, raw):
    lease = tmp_path / "report.txt.lease"
    _put_raw(lease, raw)

    _expect_blocked(lease)


def test_f_malformed_lease_leaves_the_verified_report_byte_identical(verified):
    _put_raw(verified.lease, "{holder: home-desk,,,}")

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    verified.assert_intact("malformed lease")


def test_f_malformed_lease_yields_only_to_the_explicit_override(tmp_path):
    """The deliberate, audited override is the ONE escape hatch, and it must work."""
    lease = tmp_path / "report.txt.lease"
    _put_raw(lease, "{not json at all")

    forced = wl.acquire(lease, holder="mini-pc", now=LATER, takeover=True)

    assert forced["holder"] == "mini-pc"
    assert forced["takeover"] is True


# ==========================================================================
# G. real OSError on read -- directory-at-path and a Win32 deny-sharing handle
# ==========================================================================
def test_g_directory_at_the_lease_path_blocks_acquisition(tmp_path):
    """A real filesystem failure: a DIRECTORY where the lease file belongs.

    ``Path.read_text`` raises PermissionError (Windows) / IsADirectoryError
    (POSIX) -- both OSError, exactly like an unreadable Drive file. No chmod
    (a no-op for this on Windows) and no monkeypatch.
    """
    lease = tmp_path / "report.txt.lease"
    lease.mkdir(parents=True)
    with pytest.raises(OSError):
        lease.read_text(encoding="utf-8")

    _expect_blocked(lease)


def test_g_exclusively_locked_lease_blocks_acquisition(tmp_path):
    """Real Win32 CreateFileW(dwShareMode=0) handle: reads raise PermissionError."""
    lease = tmp_path / "report.txt.lease"
    _put_json(lease, _valid_lease("home-desk"))

    with _exclusively_locked(lease):
        _expect_blocked(lease)


def test_g_directory_at_the_lease_path_leaves_the_report_byte_identical(verified):
    verified.lease.unlink()
    verified.lease.mkdir()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    assert result["error"]
    verified.assert_intact("unreadable lease path")


def test_g_exclusively_locked_lease_leaves_the_report_byte_identical(verified):
    with _exclusively_locked(verified.lease):
        result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    verified.assert_intact("exclusively locked lease")


# ==========================================================================
# H. wrong schema -- valid JSON, unusable shape / unknown version
# ==========================================================================
WRONG_SHAPE = [
    pytest.param("[]", id="empty-list"),
    pytest.param('["home-desk"]', id="list-of-holders"),
    pytest.param('"home-desk"', id="bare-string"),
    pytest.param("123", id="bare-number"),
    pytest.param("null", id="json-null"),
    pytest.param("true", id="json-bool"),
    pytest.param("{}", id="empty-object"),
    pytest.param('{"nested": {"holder": "home-desk"}}', id="nested-holder"),
]


@pytest.mark.parametrize("raw", WRONG_SHAPE)
def test_h_wrong_shape_blocks_acquisition(tmp_path, raw):
    lease = tmp_path / "report.txt.lease"
    _put_raw(lease, raw)

    _expect_blocked(lease)


def test_h_missing_schema_key_blocks_acquisition(tmp_path):
    lease = tmp_path / "report.txt.lease"
    payload = _valid_lease("home-desk")
    payload.pop("schema")
    _put_json(lease, payload)

    _expect_blocked(lease)


@pytest.mark.parametrize(
    "schema", ["writer_lease_v2", "writer_lease_v99", "", None, 1, "job_ledger_v1"]
)
def test_h_unknown_schema_version_blocks_acquisition(tmp_path, schema):
    lease = tmp_path / "report.txt.lease"
    # Expired on its face -- but an unknown schema means the expiry cannot be
    # trusted either, so this must NOT be treated as a free slot.
    payload = _valid_lease("home-desk", expires=NOW - timedelta(hours=2))
    payload["schema"] = schema
    _put_json(lease, payload)

    _expect_blocked(lease)


def test_h_unknown_schema_leaves_the_verified_report_byte_identical(verified):
    payload = _valid_lease("home-desk", expires=NOW - timedelta(hours=2))
    payload["schema"] = "writer_lease_v99"
    _put_json(verified.lease, payload)

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, "a lease this machine cannot parse is not a free slot"
    verified.assert_intact("unknown schema version")


# ==========================================================================
# I. missing / invalid holder, and missing / invalid expiry (each separately)
# ==========================================================================
def test_i_missing_holder_blocks_acquisition(tmp_path):
    lease = tmp_path / "report.txt.lease"
    payload = _valid_lease("home-desk")
    payload.pop("holder")
    _put_json(lease, payload)

    _expect_blocked(lease)


@pytest.mark.parametrize(
    "holder", [None, "", "   ", 123, 4.5, True, [], {}, ["home-desk"], {"name": "home"}]
)
def test_i_invalid_holder_blocks_acquisition(tmp_path, holder):
    lease = tmp_path / "report.txt.lease"
    payload = _valid_lease("home-desk")
    payload["holder"] = holder
    _put_json(lease, payload)

    _expect_blocked(lease)


def test_i_missing_expiry_blocks_acquisition(tmp_path):
    lease = tmp_path / "report.txt.lease"
    payload = _valid_lease("home-desk")
    payload.pop("expires_at")
    _put_json(lease, payload)

    _expect_blocked(lease)


@pytest.mark.parametrize(
    "expiry",
    [
        pytest.param(None, id="null"),
        pytest.param("", id="empty"),
        pytest.param("not-a-date", id="garbage"),
        pytest.param("2026-13-45T99:99:99", id="impossible-date"),
        pytest.param(1785000000, id="epoch-int"),
        pytest.param("09:00", id="time-only"),
        pytest.param(["2026-07-30T09:10:00"], id="list"),
    ],
)
def test_i_invalid_expiry_blocks_acquisition(tmp_path, expiry):
    lease = tmp_path / "report.txt.lease"
    payload = _valid_lease("home-desk")
    payload["expires_at"] = expiry
    _put_json(lease, payload)

    _expect_blocked(lease)


def test_i_lease_without_a_usable_expiry_is_never_reported_as_free(tmp_path):
    lease = tmp_path / "report.txt.lease"
    payload = _valid_lease("home-desk")
    payload["expires_at"] = "not-a-date"
    _put_json(lease, payload)

    assert not _summarized_as_free(lease, now=LATER), (
        "an unparseable expiry must not be summarized as 'nobody holds it'"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: p.pop("holder"), id="missing-holder"),
        pytest.param(lambda p: p.update(holder=""), id="blank-holder"),
        pytest.param(lambda p: p.pop("expires_at"), id="missing-expiry"),
        pytest.param(lambda p: p.update(expires_at="not-a-date"), id="invalid-expiry"),
    ],
)
def test_i_invalid_lease_fields_leave_the_verified_report_byte_identical(verified, mutate):
    payload = _valid_lease("home-desk")
    mutate(payload)
    _put_json(verified.lease, payload)

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    verified.assert_intact("invalid lease fields")


# ==========================================================================
# J. expired but otherwise VALID lease -> recovery via the acquisition path
# ==========================================================================
def test_j_expired_valid_lease_is_takeable_without_an_override(tmp_path):
    """The expired lease is written by the implementation itself, so this is
    valid in whatever format the implementation actually uses."""
    lease = tmp_path / "report.txt.lease"
    _expired_lease_by(lease, "home-desk")

    granted = wl.acquire(lease, holder="mini-pc", now=NOW)

    assert granted["holder"] == "mini-pc"
    assert granted.get("takeover") is False, "an expiry is not an emergency override"
    assert wl.holder_of(lease, now=NOW + timedelta(minutes=1)) == "mini-pc"


def test_j_unexpired_valid_lease_blocks_and_names_the_holder(tmp_path):
    lease = tmp_path / "report.txt.lease"
    _live_lease_by(lease, "home-desk", now=NOW)

    exc = _expect_blocked(lease)

    assert "home-desk" in str(exc), "the loser must be told WHO holds the lease"


def test_j_expired_valid_lease_permits_a_legitimate_republish(verified):
    _expired_lease_by(verified.lease, "home-desk")

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="AFTER-EXPIRY"))

    assert result["ok"] and result["verified"], result
    metadata = json.loads(verified.metadata.read_text(encoding="utf-8"))
    assert metadata["sha256"] == _sha(verified.target)
    assert metadata["holder"] == result["holder"]


# ==========================================================================
# K. OLD-FORMAT lease (no instance id) -- the classic fail-open hole
# ==========================================================================
def _old_format_lease(holder: str, *, expires: datetime) -> dict:
    """Exactly today's ``writer_lease_v1`` payload: hostname-only holder, no
    process instance identity anywhere."""
    return {
        "schema": "writer_lease_v1",
        "holder": holder,
        "acquired_at": NOW.isoformat(timespec="seconds"),
        "expires_at": expires.isoformat(timespec="seconds"),
        "takeover": False,
        "clock_skew_seconds": 120,
    }


def test_k_unexpired_old_format_lease_is_never_ours(tmp_path, monkeypatch):
    """The old lease names OUR hostname and has no instance id. It is NOT ours:
    it may belong to another process (or another boot) on this machine."""
    _pretend_machine(monkeypatch, "home-desk")
    lease = tmp_path / "report.txt.lease"
    _put_json(lease, _old_format_lease("home-desk", expires=NOW + timedelta(minutes=10)))
    before = _sha(lease)

    _expect_blocked(lease, holder=None, now=LATER)

    assert _sha(lease) == before, (
        "a questionable old-format lease was silently migrated/overwritten "
        "instead of being left alone until it expires"
    )


def test_k_unexpired_old_format_lease_blocks_a_publish(verified, monkeypatch):
    """Same machine name, old format, unexpired -> block, and change nothing."""
    verified.plant_lease(_old_format_lease("mini-pc", expires=NOW + timedelta(days=3650)))

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, (
        "an old-format lease with no instance id was treated as 'ours' and "
        "renewed straight through -- this is the fail-open hole"
    )
    verified.assert_intact("unexpired old-format lease", lease=True)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"schema": "writer_lease_v1", "holder": "home-desk"}', id="no-expiry"),
        pytest.param('{"schema": "writer_lease_v1", "expires_at": "2026-07-30T09:10:00"}', id="no-holder"),
        pytest.param('{"schema": "writer_lease_v1", "holder": "home-desk", "expires_at": "soon"}', id="bad-expiry"),
        pytest.param('{"schema": "writer_lease_v1"', id="truncated"),
    ],
)
def test_k_malformed_old_format_lease_fails_closed(tmp_path, monkeypatch, raw):
    _pretend_machine(monkeypatch, "home-desk")
    lease = tmp_path / "report.txt.lease"
    _put_raw(lease, raw)
    before = lease.read_bytes()

    _expect_blocked(lease, holder=None, now=LATER)

    assert lease.read_bytes() == before, "ambiguous old-format lease state was overwritten"


def test_k_expired_old_format_lease_recovers_only_through_acquisition(tmp_path, monkeypatch):
    """Expired old-format lease is recoverable -- but only by acquiring afresh,
    which must produce a lease this process can prove is its own."""
    _pretend_machine(monkeypatch, "home-desk")
    lease = tmp_path / "report.txt.lease"
    _put_json(lease, _old_format_lease("home-desk", expires=NOW - timedelta(hours=2)))

    granted = wl.acquire(lease, now=NOW)

    assert granted.get("takeover") is False
    on_disk = json.loads(lease.read_text(encoding="utf-8"))
    assert on_disk != _old_format_lease("home-desk", expires=NOW - timedelta(hours=2)), (
        "acquisition must write a fresh lease, not leave the stale one in place"
    )
    assert wl.holder_of(lease, now=NOW + timedelta(minutes=1)), (
        "after acquisition somebody must be the visible holder"
    )


def test_k_old_format_lease_never_grants_a_silent_takeover_to_a_same_named_machine(
    verified,
):
    """Another machine could legitimately be publishing under the old format;
    inheriting its lease by hostname is exactly the lost-update we must avoid."""
    verified.plant_lease(_old_format_lease("home-desk", expires=NOW + timedelta(days=3650)))

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    verified.assert_intact("old-format lease held by the other machine", lease=True)


# ==========================================================================
# L. simultaneous acquisition attempts -> exactly one winner
# ==========================================================================
def _race(lease: Path, holders: list[str], *, now: datetime) -> tuple[list[str], list[str]]:
    barrier = threading.Barrier(len(holders))
    winners: list[str] = []
    blocked: list[str] = []
    exploded: list[str] = []
    guard = threading.Lock()

    def attempt(name: str) -> None:
        barrier.wait()
        try:
            granted = wl.acquire(lease, holder=name, ttl_minutes=10, now=now)
        except wl.LeaseUnavailable:
            with guard:
                blocked.append(name)
        except Exception as exc:  # noqa: BLE001
            with guard:
                exploded.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            with guard:
                winners.append(str(granted.get("holder")))

    threads = [threading.Thread(target=attempt, args=(name,)) for name in holders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not exploded, f"a racing acquirer must never crash: {exploded}"
    return winners, blocked


def test_l_simultaneous_acquirers_produce_exactly_one_winner(tmp_path):
    for round_index in range(RACE_ROUNDS):
        lease = tmp_path / f"round{round_index}" / "report.txt.lease"
        lease.parent.mkdir(parents=True, exist_ok=True)
        holders = [f"machine-{i}" for i in range(4)]

        winners, blocked = _race(lease, holders, now=NOW)

        assert len(winners) == 1, (
            f"round {round_index}: {len(winners)} machines all believed they won "
            f"the lease ({winners}); a read-then-replace on a shared file is not "
            "a compare-and-swap"
        )
        assert len(blocked) == len(holders) - 1
        on_disk = json.loads(lease.read_text(encoding="utf-8"))
        assert on_disk["holder"] == winners[0], (
            "the lease on disk must name the machine that was told it won"
        )


def test_l_simultaneous_acquirers_cannot_all_take_a_live_lease(tmp_path):
    lease = tmp_path / "report.txt.lease"
    _live_lease_by(lease, "home-desk", now=NOW)

    winners, blocked = _race(lease, [f"machine-{i}" for i in range(6)], now=LATER)

    assert winners == [], f"a live lease was stolen by {winners}"
    assert len(blocked) == 6
    assert json.loads(lease.read_text(encoding="utf-8"))["holder"] == "home-desk"


def test_l_racing_publishers_cannot_both_replace_the_report(verified):
    """Two publishers race into a free slot; at most one may win, every round."""
    local = threading.local()
    guard = threading.Lock()

    def holder_id() -> str:
        return getattr(local, "name", "unknown-machine")

    original = wl.machine_holder_id
    wl.machine_holder_id = holder_id
    try:
        for round_index in range(PUBLISH_RACE_ROUNDS):
            verified.lease.unlink(missing_ok=True)
            results: list[dict] = []
            barrier = threading.Barrier(2)

            def attempt(machine: str) -> None:
                local.name = machine
                barrier.wait()
                outcome = verified.core.publish_away_report(
                    dict(FULL_PAYLOAD, generated_at=f"FROM-{machine}"),
                    verified.target,
                    archive=False,
                )
                with guard:
                    results.append(outcome)

            threads = [
                threading.Thread(target=attempt, args=(name,))
                for name in ("mini-pc", "other-pc")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

            winners = [r for r in results if r["ok"]]
            assert len(winners) <= 1, (
                f"round {round_index}: both publishers wrote over each other "
                f"({[r['holder'] for r in winners]}); the shared report is now "
                "whichever write landed last"
            )
            published = verified.target.read_text(encoding="utf-8")
            assert published.count("TRADINGBOT AUTO PILOT - TODAY") == 1, (
                "interleaved report bytes"
            )
    finally:
        wl.machine_holder_id = original


# ==========================================================================
# M. two REAL processes on ONE hostname -> never conflated into one writer
# ==========================================================================
_CHILD_SOURCE = r'''
import json, os, socket, sys, time

scripts_dir, lease_path, machine, fake_pid, start_at, mode = sys.argv[1:7]
sys.path.insert(0, scripts_dir)

# A genuinely separate OS process that reports the given hostname and PID.
socket.gethostname = lambda: machine
_real_getpid = os.getpid
os.getpid = lambda: int(fake_pid)

import writer_lease as wl

if mode == "wait":
    target = float(start_at)
    while time.time() < target:
        time.sleep(0.001)

out = {"machine": machine, "pid": int(fake_pid), "real_pid": _real_getpid()}
try:
    granted = wl.acquire(lease_path, ttl_minutes=10)
except wl.LeaseUnavailable as exc:
    out.update(ok=False, blocked=True, error=str(exc))
except Exception as exc:
    out.update(ok=False, blocked=False, error="%s: %s" % (type(exc).__name__, exc))
else:
    out.update(ok=True, blocked=False, lease=granted)
print("RESULT " + json.dumps(out, default=str))
'''


def _child_env(tmp_path: Path) -> dict:
    env = dict(os.environ)
    hermetic = tmp_path / "child_home"
    (hermetic / "diagnostics").mkdir(parents=True, exist_ok=True)
    env["TRADINGBOTV3_DATA_DIR"] = str(hermetic)
    env["TRADINGBOT_DIAGNOSTICS_DIR"] = str(hermetic / "diagnostics")
    env["LOCALAPPDATA"] = str(hermetic)
    env["TRADINGBOTV3_DRIVE_WAIT_SECONDS"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _spawn_child(script: Path, lease: Path, machine: str, pid: int, start_at, mode, env):
    return subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(SCRIPTS_DIR),
            str(lease),
            machine,
            str(pid),
            str(start_at),
            mode,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )


def _child_result(process) -> dict:
    stdout, stderr = process.communicate(timeout=180)
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"child produced no result.\nstdout={stdout}\nstderr={stderr}")


@pytest.fixture
def child_script(tmp_path) -> Path:
    script = tmp_path / "lease_child.py"
    script.write_text(_CHILD_SOURCE, encoding="utf-8")
    return script


def test_m_two_real_processes_on_one_hostname_do_not_both_win(tmp_path, child_script):
    """Two genuinely separate OS processes, same hostname, different PIDs."""
    env = _child_env(tmp_path)
    for round_index in range(SUBPROCESS_RACE_ROUNDS):
        lease = tmp_path / f"m{round_index}" / "report.txt.lease"
        lease.parent.mkdir(parents=True, exist_ok=True)
        start_at = time.time() + 2.0
        children = [
            _spawn_child(child_script, lease, "home-desk", 4100 + index, start_at, "wait", env)
            for index in range(3)
        ]
        results = [_child_result(child) for child in children]

        assert all(r["ok"] or r["blocked"] for r in results), (
            f"a racing process crashed instead of failing closed: {results}"
        )
        winners = [r for r in results if r["ok"]]
        assert len(winners) == 1, (
            f"round {round_index}: {len(winners)} processes on ONE machine each "
            f"believed they held the writer lease: {results}"
        )


def test_m_second_process_on_the_same_machine_is_blocked(tmp_path, child_script):
    env = _child_env(tmp_path)
    lease = tmp_path / "report.txt.lease"

    first = _child_result(_spawn_child(child_script, lease, "home-desk", 1111, 0, "go", env))
    assert first["ok"], f"the first process could not acquire a free lease: {first}"

    second = _child_result(_spawn_child(child_script, lease, "home-desk", 2222, 0, "go", env))

    assert second["ok"] is False, (
        "a second live process on the same hostname inherited the first "
        "process's lease; hostname alone is not ownership"
    )
    assert second["blocked"] is True, f"it must fail closed, not crash: {second}"


def test_m_second_process_publish_leaves_the_first_process_report_intact(
    tmp_path, monkeypatch, config, diagnostics_dir
):
    import autopilot_core as core

    _pretend_machine(monkeypatch, "home-desk")
    config.designate("home-desk")
    monkeypatch.setattr(os, "getpid", lambda: 1111)
    target = tmp_path / "drive" / "autopilot_today.txt"
    first = core.publish_away_report(
        dict(FULL_PAYLOAD, generated_at="PROCESS-A"), target, archive=False
    )
    assert first["ok"], first
    baseline = Publication(core, target)

    monkeypatch.setattr(os, "getpid", lambda: 2222)  # a second GUI on the same PC
    second = core.publish_away_report(
        dict(FULL_PAYLOAD, generated_at="PROCESS-B"), target, archive=False
    )

    assert second["ok"] is False, (
        "a second process on the same machine inherited the first one's lease "
        "and republished the shared report"
    )
    baseline.assert_intact("second process, same hostname")


def test_m_release_by_a_different_process_on_the_same_machine_is_refused(
    tmp_path, monkeypatch
):
    lease = tmp_path / "report.txt.lease"
    _pretend_machine(monkeypatch, "home-desk")
    monkeypatch.setattr(os, "getpid", lambda: 1111)
    wl.acquire(lease, ttl_minutes=10, now=NOW)

    monkeypatch.setattr(os, "getpid", lambda: 2222)
    assert wl.release(lease) is False, "process B released process A's lease"
    assert lease.exists()


def test_m_identity_distinguishes_two_processes_on_one_host(tmp_path, monkeypatch):
    """Two PIDs, ONE hostname: ownership must never be conflated.

    Asserted through observable lease behavior rather than through the return
    value of any particular identity helper. An implementation that proves
    per-instance ownership with a separate lease field instead of by embedding
    pid+instance in the holder string is equally safe and must pass this too --
    what may not happen is process B being able to renew, release, or claim
    ownership of process A's lease.
    """
    lease = tmp_path / "report.txt.lease"
    _pretend_machine(monkeypatch, "home-desk")

    monkeypatch.setattr(os, "getpid", lambda: 1111)
    first = wl.acquire(lease, ttl_minutes=10, now=NOW)

    monkeypatch.setattr(os, "getpid", lambda: 2222)
    # 1. B cannot take A's live lease, even sharing A's hostname.
    _expect_blocked(lease, holder=None, now=NOW + timedelta(minutes=1))
    # 2. B cannot release it out from under A.
    assert wl.release(lease) is False, "process B released process A's lease"
    assert lease.exists()
    # 3. B cannot prove A's lease is B's own.
    with pytest.raises(wl.LeaseUnavailable):
        wl.assert_still_owned(
            lease,
            holder=wl.machine_holder_id(),
            generation=first["generation"],
            now=NOW + timedelta(minutes=1),
        )

    # 4. Once it has expired, B acquires afresh -- and A is then fenced off,
    #    so the two were never the same writer at any point.
    later = NOW + timedelta(hours=2)
    second = wl.acquire(lease, now=later)
    with pytest.raises(wl.LeaseUnavailable):
        wl.assert_still_owned(
            lease, holder=first["holder"], generation=first["generation"], now=later
        )
    assert second["generation"] > first["generation"], (
        "a change of owning process instance must advance the fencing generation"
    )


# ==========================================================================
# N. PID reuse / restart -> a restarted process inherits nothing
# ==========================================================================
def test_n_restarted_process_with_a_reused_pid_does_not_inherit_ownership(
    tmp_path, child_script
):
    """Two real processes, same hostname AND same PID (Windows reuses PIDs
    freely). The second is a different process instance and must not inherit."""
    env = _child_env(tmp_path)
    lease = tmp_path / "report.txt.lease"

    first = _child_result(_spawn_child(child_script, lease, "home-desk", 5150, 0, "go", env))
    assert first["ok"], f"the first process could not acquire a free lease: {first}"

    reborn = _child_result(_spawn_child(child_script, lease, "home-desk", 5150, 0, "go", env))

    assert reborn["ok"] is False, (
        "a restarted process with a reused PID inherited the previous process "
        "instance's unexpired lease; identity needs a per-process-start "
        "instance id, not just hostname+pid"
    )
    assert reborn["blocked"] is True, f"it must fail closed, not crash: {reborn}"


def test_n_restarted_process_may_recover_only_after_expiry(tmp_path, monkeypatch):
    """The stale lease of a dead process is recovered through the normal
    acquisition path once it expires -- never by claiming it was already ours."""
    lease = tmp_path / "report.txt.lease"
    _pretend_machine(monkeypatch, "home-desk")
    monkeypatch.setattr(os, "getpid", lambda: 3131)
    wl.acquire(lease, ttl_minutes=10, now=NOW - timedelta(hours=1))

    granted = wl.acquire(lease, ttl_minutes=10, now=NOW)

    assert granted.get("takeover") is False
    assert wl.holder_of(lease, now=NOW + timedelta(minutes=1))


def test_n_restart_does_not_let_a_publish_ride_the_old_instance_lease(verified):
    """The lease on disk is unexpired and names this hostname, but was written
    by a process instance that no longer exists."""
    verified.plant_lease(_old_format_lease("mini-pc", expires=NOW + timedelta(days=3650)))

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, (
        "a fresh process published under a lease it never actually acquired"
    )
    verified.assert_intact("restarted process, stale instance lease", lease=True)


# ==========================================================================
# O. ownership lost BETWEEN acquisition and publication -> abort first
# ==========================================================================
class LeaseStealingPayload(dict):
    """A payload that hands the lease to another machine mid-publish.

    The steal happens on a real file, at a real point in the publish sequence
    (while the report is being rendered, before any bytes are staged) -- exactly
    the window where Drive sync can deliver a competing writer's lease.
    """

    def __init__(self, base: dict, lease_path: Path, new_holder: str, at_call: int = 1):
        super().__init__(base)
        self._lease_path = lease_path
        self._new_holder = new_holder
        self._at_call = at_call
        self.calls = 0
        self.stolen = False

    def _steal(self) -> None:
        _put_json(
            self._lease_path,
            _valid_lease(
                self._new_holder, expires=datetime.now() + timedelta(minutes=30)
            ),
        )

    def get(self, key, default=None):
        self.calls += 1
        if self.calls == self._at_call and not self.stolen:
            self.stolen = True
            self._steal()
        return dict.get(self, key, default)


def test_o_losing_the_lease_mid_publish_blocks_the_write(verified):
    verified.lease.unlink()  # mini-pc legitimately acquires...
    payload = LeaseStealingPayload(
        dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"),
        verified.lease,
        "home-desk",  # ...and home-desk takes it back mid-flight
    )

    result = verified.publish(payload)

    assert payload.stolen, "the test did not actually steal the lease"
    assert result["ok"] is False, (
        "ownership was lost between acquisition and the atomic replace, and the "
        "publish went through anyway (no re-verify before replacement)"
    )
    assert json.loads(verified.lease.read_text(encoding="utf-8"))["holder"] == "home-desk"


def test_o_losing_the_lease_mid_publish_leaves_the_report_byte_identical(verified):
    verified.lease.unlink()
    payload = LeaseStealingPayload(
        dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"), verified.lease, "home-desk"
    )

    verified.publish(payload)

    verified.assert_intact("lease stolen mid-publish")


def test_o_lease_deleted_mid_publish_is_still_a_failed_publish(verified):
    """A vanished lease mid-flight is lost ownership, not a free slot."""

    class LeaseDeletingPayload(LeaseStealingPayload):
        def _steal(self) -> None:
            self._lease_path.unlink(missing_ok=True)

    verified.lease.unlink()
    payload = LeaseDeletingPayload(
        dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"), verified.lease, "nobody"
    )

    result = verified.publish(payload)

    assert payload.stolen
    assert result["ok"] is False, "the lease disappeared mid-publish and nothing noticed"
    verified.assert_intact("lease deleted mid-publish")


def test_o_lease_corrupted_mid_publish_is_still_a_failed_publish(verified):
    """Half-synced bytes arriving mid-render mean ownership is unverifiable."""

    class LeaseCorruptingPayload(LeaseStealingPayload):
        def _steal(self) -> None:
            _put_raw(self._lease_path, "{half-synced")

    verified.lease.unlink()
    payload = LeaseCorruptingPayload(
        dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"), verified.lease, "nobody"
    )

    result = verified.publish(payload)

    assert payload.stolen
    assert result["ok"] is False, (
        "ownership became unverifiable mid-publish and the report was replaced anyway"
    )
    verified.assert_intact("lease corrupted mid-publish")


# ==========================================================================
# P. fencing generation -- monotonic, enforced on the write path
# ==========================================================================
_GENERATION_KEY_HINTS = ("generation", "fencing", "fence", "epoch", "sequence", "seq")


def _generation_of(payload) -> int | None:
    """The lease's monotonic fencing value, under any reasonable key name."""
    if not isinstance(payload, dict):
        return None
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if any(hint in str(key).lower() for hint in _GENERATION_KEY_HINTS):
            return value
    return None


def _require_generation(payload, where: str) -> int:
    value = _generation_of(payload)
    assert value is not None, (
        f"{where} carries no monotonic fencing/generation number (looked for an "
        f"integer field named like {_GENERATION_KEY_HINTS}): {payload!r}"
    )
    return value


def test_p_every_acquisition_advances_the_fencing_generation(tmp_path):
    lease = tmp_path / "report.txt.lease"
    first = wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW - timedelta(hours=2))
    second = wl.acquire(lease, holder="mini-pc", ttl_minutes=10, now=NOW)
    third = wl.acquire(lease, holder="other-pc", ttl_minutes=10, now=NOW, takeover=True)

    g1 = _require_generation(first, "the first lease")
    g2 = _require_generation(second, "the lease after a legitimate expiry takeover")
    g3 = _require_generation(third, "the lease after an emergency takeover")
    assert g1 < g2 < g3, (
        f"the fencing generation must increase on every change of ownership: {g1}, {g2}, {g3}"
    )


def test_p_publication_metadata_records_the_fencing_generation(verified):
    result = verified.publish(dict(FULL_PAYLOAD, generated_at="FENCED"))

    assert result["ok"], result
    metadata = json.loads(verified.metadata.read_text(encoding="utf-8"))
    lease = json.loads(verified.lease.read_text(encoding="utf-8"))
    published = _require_generation(metadata, "the publication metadata")
    held = _require_generation(lease, "the lease on disk")
    assert published == held, (
        f"the report says it was published under generation {published} while the "
        f"lease says {held}; the fencing value must be enforced on the write path"
    )


def test_p_generation_change_mid_render_aborts_before_replacement(verified):
    """Same hostname, higher generation: another instance fenced us off while we
    were rendering. The replacement must not happen."""

    class GenerationBumpingPayload(LeaseStealingPayload):
        def _steal(self) -> None:
            current = json.loads(self._lease_path.read_text(encoding="utf-8"))
            bumped = dict(current)
            found = False
            for key, value in current.items():
                if isinstance(value, int) and not isinstance(value, bool) and any(
                    hint in str(key).lower() for hint in _GENERATION_KEY_HINTS
                ):
                    bumped[key] = value + 1
                    found = True
            if not found:
                bumped["generation"] = 99
            bumped["instance_id"] = "a-newer-process-instance"
            _put_json(self._lease_path, bumped)

    verified.lease.unlink()
    payload = GenerationBumpingPayload(
        dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"), verified.lease, "mini-pc"
    )

    result = verified.publish(payload)

    assert payload.stolen
    assert result["ok"] is False, (
        "the fencing generation advanced while the report was being rendered and "
        "the stale writer replaced the report anyway"
    )
    verified.assert_intact("generation bumped mid-render")


# ==========================================================================
# Q. sleep / wake / missed renewal -> ownership is LOST
# ==========================================================================
def test_q_expired_own_lease_is_not_ownership(tmp_path, monkeypatch):
    """The machine slept through its own renewal window."""
    lease = tmp_path / "report.txt.lease"
    _pretend_machine(monkeypatch, "mini-pc")
    wl.acquire(lease, ttl_minutes=10, now=NOW - timedelta(hours=3))

    assert wl.holder_of(lease, now=NOW) is None, (
        "an expired lease is not ownership, even for the machine that wrote it"
    )


def test_q_publish_after_a_missed_renewal_re_acquires_first(verified):
    """A publish that goes through after sleep must be backed by a FRESH lease,
    not by the stale one the sleeping process left behind."""
    _expired_lease_by(verified.lease, "mini-pc", expired_for=timedelta(hours=3))

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="AFTER-SLEEP"))

    if not result["ok"]:
        verified.assert_intact("refused after sleep")
        return
    holder = wl.holder_of(verified.lease)
    assert holder, (
        "the report was published while no unexpired lease existed: the woken "
        "process rode its own dead lease instead of re-acquiring"
    )
    assert str(result.get("lease_expires_at") or ""), "the publish did not record a lease expiry"
    expires = datetime.fromisoformat(str(result["lease_expires_at"]))
    assert expires > datetime.now(), (
        f"the publish reported an already-expired lease ({expires}); ownership "
        "must be re-established before replacing the shared report"
    )


def test_q_other_machine_took_over_during_sleep(verified):
    """We slept, the other machine legitimately took the lease. On wake we lose."""
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=30)

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    assert "home-desk" in str(result.get("error") or "")
    verified.assert_intact("other machine took over during sleep")


def test_q_ttl_is_bounded_and_renewable(tmp_path):
    """Bounded renewal: the holder can extend its own lease while it is alive,
    so a publish cadence longer than one TTL does not mean 'nobody holds it'."""
    lease = tmp_path / "report.txt.lease"
    wl.acquire(lease, holder="mini-pc", ttl_minutes=10, now=NOW)

    for minutes in (5, 10, 15, 20, 25):
        moment = NOW + timedelta(minutes=minutes)
        renewed = wl.acquire(lease, holder="mini-pc", ttl_minutes=10, now=moment)
        assert renewed["holder"] == "mini-pc"
        assert wl.holder_of(lease, now=moment + timedelta(minutes=1)) == "mini-pc", (
            f"a renewal at +{minutes}m did not keep the lease alive"
        )
        with pytest.raises(wl.LeaseUnavailable):
            wl.acquire(lease, holder="home-desk", now=moment + timedelta(minutes=1))


# ==========================================================================
# R. expiry boundaries, bounded clock skew, and no permanent lockout
# ==========================================================================
@pytest.mark.parametrize(
    "offset_seconds",
    [
        pytest.param(0, id="at-acquisition"),
        pytest.param(599, id="one-second-before-expiry"),
        pytest.param(600, id="exactly-at-expiry"),
        pytest.param(601, id="one-second-after-expiry"),
        pytest.param(600 + wl.DEFAULT_CLOCK_SKEW_SECONDS - 1, id="inside-skew-window"),
        pytest.param(600 + wl.DEFAULT_CLOCK_SKEW_SECONDS, id="at-skew-boundary"),
    ],
)
def test_r_no_premature_takeover_inside_the_skew_window(tmp_path, offset_seconds):
    lease = tmp_path / "report.txt.lease"
    wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW)

    moment = NOW + timedelta(seconds=offset_seconds)
    _expect_blocked(lease, now=moment)
    assert wl.holder_of(lease, now=moment) == "home-desk"


def test_r_takeover_becomes_possible_once_the_skew_window_passes(tmp_path):
    lease = tmp_path / "report.txt.lease"
    wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW)

    moment = NOW + timedelta(seconds=600 + wl.DEFAULT_CLOCK_SKEW_SECONDS + 1)
    granted = wl.acquire(lease, holder="mini-pc", now=moment)

    assert granted["holder"] == "mini-pc", "an expired lease must not lock the desk out forever"


def test_r_a_long_dead_lease_never_becomes_a_permanent_lockout(tmp_path):
    lease = tmp_path / "report.txt.lease"
    wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW - timedelta(days=30))

    granted = wl.acquire(lease, holder="mini-pc", now=NOW)

    assert granted["holder"] == "mini-pc"


def test_r_an_absurd_future_expiry_still_has_a_defined_way_out(tmp_path):
    """A lease claiming to run until the year 3000 must not be honored quietly
    forever. Fail closed is fine -- but the audited override must recover it."""
    lease = tmp_path / "report.txt.lease"
    _put_json(lease, _valid_lease("home-desk", expires=datetime(3000, 1, 1)))

    _expect_blocked(lease)
    recovered = wl.acquire(lease, holder="mini-pc", now=LATER, takeover=True)

    assert recovered["holder"] == "mini-pc"
    assert recovered["takeover"] is True


def test_r_a_lease_from_the_future_does_not_silently_become_ours(tmp_path):
    """Clock skew in the other direction: acquired_at is ahead of our clock."""
    lease = tmp_path / "report.txt.lease"
    future = NOW + timedelta(hours=6)
    _put_json(lease, _valid_lease("home-desk", expires=future + timedelta(minutes=10)))

    _expect_blocked(lease, now=NOW)


# ==========================================================================
# S. emergency takeover: explicit, TIME-BOUNDED, auditable, visible
# ==========================================================================
def test_s_takeover_is_never_implicit(tmp_path):
    lease = tmp_path / "report.txt.lease"
    _live_lease_by(lease, "home-desk", now=NOW)

    _expect_blocked(lease)  # no override -> refused

    forced = wl.acquire(lease, holder="mini-pc", now=LATER, takeover=True)
    assert forced["takeover"] is True


def test_s_takeover_records_who_was_displaced(tmp_path):
    lease = tmp_path / "report.txt.lease"
    _live_lease_by(lease, "home-desk", now=NOW)

    forced = wl.acquire(lease, holder="mini-pc", now=LATER, takeover=True)

    assert forced["takeover"] is True
    displaced = str(forced.get("previous_holder") or forced.get("displaced_holder") or "")
    assert displaced == "home-desk", (
        "an emergency takeover that does not record whose lease it broke is not "
        f"auditable: {forced!r}"
    )


def test_s_takeover_stays_visible_after_the_next_normal_renewal(tmp_path, diagnostics_dir):
    """A renewal must not erase the audit trail of a broken lease.

    The record has to be durable, externally visible, and outside the lease file
    itself (which the very next renewal rewrites). WHERE it lives is the
    implementation's choice: beside the lease, so it travels with the shared
    export, or in the machine-local diagnostics root -- both are searched. What
    is asserted is that after repeated renewals a persisted artifact still names
    who was displaced AND who displaced them.
    """
    lease = tmp_path / "report.txt.lease"
    _live_lease_by(lease, "home-desk", now=NOW)

    wl.acquire(lease, holder="mini-pc", now=LATER, takeover=True)
    # Several normal renewals by the new holder, none of them a takeover.
    for minutes in (5, 9, 13):
        renewed = wl.acquire(lease, holder="mini-pc", now=LATER + timedelta(minutes=minutes))
        assert renewed["takeover"] is False, (
            "a plain renewal reported itself as an emergency takeover"
        )

    trail = []
    for root in (lease.parent, diagnostics_dir):
        for path in root.rglob("*"):
            if not path.is_file() or path == lease:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "home-desk" in text and "mini-pc" in text:
                trail.append(path)
    assert trail, (
        "after three renewals there is no durable, externally visible record "
        "that mini-pc broke home-desk's lease; the lease file alone is not an "
        "audit trail, because the next renewal overwrites it"
    )


def test_s_forced_takeover_publish_is_flagged_in_the_metadata(verified):
    """A report published under an emergency takeover must say so on its face."""
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=24 * 60)
    wl.acquire(verified.lease, takeover=True)

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="FORCED"))

    assert result["ok"], result
    metadata = json.loads(verified.metadata.read_text(encoding="utf-8"))
    assert metadata.get("takeover") is True or metadata.get("previous_holder") == "home-desk", (
        "a forcibly taken-over publication is indistinguishable from a normal "
        f"one in its metadata: {metadata!r}"
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("0", id="zero-string"),
        pytest.param("false", id="false-string"),
        pytest.param("no", id="no"),
        pytest.param("maybe", id="garbage-word"),
        pytest.param("null", id="null-string"),
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-object"),
        pytest.param("2020-01-01T00:00:00", id="already-expired-timestamp"),
    ],
)
def test_s_malformed_override_config_never_activates_a_takeover(verified, config, value):
    """A malformed override value must never evaluate truthy into an override."""
    config.designate("mini-pc")
    config.emergency_override(value)
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=60)
    verified.snapshot()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, (
        f"the malformed override value {value!r} evaluated truthy and broke a live lease"
    )
    verified.assert_intact(f"malformed override {value!r}")


def test_s_expired_override_config_does_not_authorize_a_takeover(verified, config):
    config.designate("mini-pc")
    config.emergency_override(
        True, expires_at=(datetime.now() - timedelta(days=1)).isoformat(timespec="seconds")
    )
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=60)
    verified.snapshot()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False, (
        "an emergency override with a past expiry is not an active override"
    )
    verified.assert_intact("expired emergency override")


def test_s_ambient_environment_alone_never_breaks_a_live_lease(tmp_path, monkeypatch):
    """An ambient env var must never be enough to break a live lease."""
    lease = tmp_path / "report.txt.lease"
    _live_lease_by(lease, "home-desk", now=NOW)
    for key in _ENV_OVERRIDE_KEYS:
        monkeypatch.setenv(key, "1")

    _expect_blocked(lease)


# ==========================================================================
# T. Layer 5 health telemetry
# ==========================================================================
#: Every Layer 5 field, with the key spellings that would satisfy it. The
#: implementation picks its own names; the concept has to be there.
_TELEMETRY_CONCEPTS = {
    "configured designated writer": ("designated", "configured_writer", "writer_machine"),
    "local machine identity": ("machine", "host", "node"),
    "configured role": ("role", "read_only", "readonly"),
    "process id": ("pid", "process_id"),
    "process instance uuid": ("instance",),
    "local cross-process exclusion state": ("mutex", "local_lock", "process_lock", "exclusion"),
    "lease holder": ("holder",),
    "acquired timestamp": ("acquired",),
    "renewal timestamp": ("renew",),
    "expiry timestamp": ("expire",),
    "fencing generation": ("generation", "fence", "epoch"),
    "last ownership/configuration failure": ("failure", "error", "blocked"),
    "read-only reason": ("reason", "read_only", "readonly"),
    "emergency override state": ("override", "takeover"),
    "last verified publication": ("publication", "published", "last_verified", "sha256"),
}

_HEALTH_READER_HINTS = ("health", "telemetry", "writer_status", "publication_status")

#: Where a health reader may legitimately live. The implementation picks; every
#: one of these is searched, and every reader found is held to the same rule.
_HEALTH_SURFACE_MODULES = (
    "writer_lease",
    "writer_role",
    "writer_health",
    "local_writer_lock",
    "autopilot_core",
    "operations_audit",
)
#: Name prefixes that mean "this mutates state", so it is not a reader to poll.
_NON_READER_VERBS = frozenset({"write", "set", "clear", "record", "append", "reset"})


def _telemetry_files(diagnostics_dir: Path) -> list[Path]:
    return [p for p in diagnostics_dir.rglob("*.json") if p.is_file()]


def _telemetry_text(diagnostics_dir: Path) -> str:
    parts = []
    for path in _telemetry_files(diagnostics_dir):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts).lower()


def _flatten(payload, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat[path] = value
            flat.update(_flatten(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            flat.update(_flatten(value, f"{prefix}[{index}]"))
    return flat


def _telemetry_keys(diagnostics_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in _telemetry_files(diagnostics_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        keys.update(key.lower() for key in _flatten(payload))
    return keys


def _discover_health_readers():
    """Every public callable on the writer surface that reports health.

    Deliberately not "the first name that matches on one of two named modules":
    which module hosts the reader, and whether it takes the artifact path or
    defaults to it, are implementation choices. Every candidate found anywhere
    on the surface is collected, and the safety assertion below is applied to
    ALL of them -- so this is stricter than picking one, not looser.
    """
    import importlib

    readers = []
    for module_name in _HEALTH_SURFACE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - an absent module is simply not a surface
            continue
        for name in dir(module):
            if name.startswith("_") or name.split("_")[0] in _NON_READER_VERBS:
                continue
            if not any(hint in name.lower() for hint in _HEALTH_READER_HINTS):
                continue
            candidate = getattr(module, name)
            if callable(candidate) and not isinstance(candidate, type):
                readers.append((f"{module_name}.{name}", candidate))
    return readers


def _read_state(reader, artifact: Path):
    """Ask a reader for its state, tolerating either calling convention.

    ``(sentinel, ...)`` is returned when the reader declined to answer at all;
    refusing to answer is honest, and honest is what is being tested.
    """
    for args in ((), (artifact,)):
        try:
            return True, reader(*args)
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - raising instead of lying is honest
            return False, None
    return False, None


def _assert_nothing_reads_as_healthy(artifact: Path, note: str) -> None:
    readers = _discover_health_readers()
    assert readers, (
        "Layer 5 requires a readable health surface that fails honestly; no "
        f"public callable named like {_HEALTH_READER_HINTS} exists anywhere on "
        f"{list(_HEALTH_SURFACE_MODULES)}"
    )
    answered = 0
    for label, reader in readers:
        ok, state = _read_state(reader, artifact)
        if not ok:
            continue
        answered += 1
        flat = {key.lower(): value for key, value in _flatten(state).items()}
        healthy_claims = [
            (key, value)
            for key, value in flat.items()
            if any(word in key for word in ("healthy", "ok", "status"))
            and value in (True, "ok", "healthy", "OK")
        ]
        assert not healthy_claims, (
            f"{note} health telemetry was reported as healthy by {label}: {healthy_claims}"
        )
    assert answered, (
        "no health surface would answer at all (with no arguments or with the "
        f"artifact path); a Health panel has nothing to render for {note} telemetry"
    )


def test_t_a_publish_writes_health_telemetry(verified, diagnostics_dir):
    verified.publish(dict(FULL_PAYLOAD, generated_at="TELEMETRY"))

    files = _telemetry_files(diagnostics_dir)
    assert files, (
        "Layer 5 requires one atomically written health-state artifact; nothing "
        f"was written under {diagnostics_dir}"
    )


def test_t_telemetry_exposes_every_layer_5_field(verified, diagnostics_dir):
    verified.publish(dict(FULL_PAYLOAD, generated_at="TELEMETRY"))

    keys = _telemetry_keys(diagnostics_dir)
    missing = {
        concept: spellings
        for concept, spellings in _TELEMETRY_CONCEPTS.items()
        if not any(any(hint in key for hint in spellings) for key in keys)
    }
    assert not missing, (
        "the health artifact is missing Layer 5 fields (any of the listed key "
        f"spellings would satisfy each): {missing}\nfound keys: {sorted(keys)}"
    )


def test_t_telemetry_records_a_blocked_publish(verified, diagnostics_dir):
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=60)

    verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    blob = _telemetry_text(diagnostics_dir)
    assert "home-desk" in blob, (
        "telemetry does not record who blocked this machine's publish; a health "
        f"panel would render this as healthy: {blob[:2000]}"
    )


def test_t_corrupt_telemetry_never_reads_as_healthy(verified, diagnostics_dir):
    verified.publish(dict(FULL_PAYLOAD, generated_at="TELEMETRY"))
    files = _telemetry_files(diagnostics_dir)
    assert files, "no health artifact to corrupt"
    for path in files:
        path.write_bytes(b'{"schema": "writer_health_v1", "hol')  # real truncation

    _assert_nothing_reads_as_healthy(files[0], "corrupt")


def test_t_absent_telemetry_never_reads_as_healthy(verified, diagnostics_dir):
    verified.publish(dict(FULL_PAYLOAD, generated_at="TELEMETRY"))
    files = _telemetry_files(diagnostics_dir)
    assert files, "no health artifact to remove"
    for path in files:
        path.unlink()

    _assert_nothing_reads_as_healthy(files[0], "absent")


def test_t_stale_telemetry_never_reads_as_healthy(verified, diagnostics_dir):
    """A machine that stopped reporting must not keep serving its last green
    artifact. The write side swallows every failure by design, so age is the
    only thing that distinguishes "publishing fine" from "died at lunchtime"."""
    verified.publish(dict(FULL_PAYLOAD, generated_at="TELEMETRY"))
    files = _telemetry_files(diagnostics_dir)
    assert files, "no health artifact to age"
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["written_at"] = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
        path.write_text(json.dumps(payload), encoding="utf-8")

    _assert_nothing_reads_as_healthy(files[0], "week-old")




# ==========================================================================
# U. partial failure -> report and metadata never disagree
# ==========================================================================
def test_u_metadata_write_failure_rolls_the_report_back(verified):
    """A REAL on-disk failure: a directory sits where the metadata file goes."""
    previous_report = verified.target.read_bytes()
    verified.metadata.unlink()
    verified.metadata.mkdir()

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    assert verified.target.read_bytes() == previous_report, (
        "the report was replaced while its metadata could not be written, so the "
        "shared report and its metadata now describe different publications"
    )


def test_u_report_and_metadata_always_agree_after_a_successful_publish(verified):
    result = verified.publish(dict(FULL_PAYLOAD, generated_at="AGREEMENT"))

    assert result["ok"], result
    metadata = json.loads(verified.metadata.read_text(encoding="utf-8"))
    lease = json.loads(verified.lease.read_text(encoding="utf-8"))
    assert metadata["sha256"] == _sha(verified.target)
    assert metadata["holder"] == result["holder"] == lease["holder"], (
        f"report metadata says {metadata['holder']!r}, the publish result says "
        f"{result['holder']!r}, the lease says {lease['holder']!r}"
    )


def test_u_report_and_metadata_never_disagree_after_a_blocked_publish(verified):
    """Whatever the failure, the pair on disk stays the last verified pair."""
    metadata_before = json.loads(verified.metadata.read_text(encoding="utf-8"))
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=60)

    verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    metadata_after = json.loads(verified.metadata.read_text(encoding="utf-8"))
    assert metadata_after == metadata_before
    assert metadata_after["sha256"] == _sha(verified.target), (
        "the metadata on disk no longer describes the report next to it"
    )


def test_u_no_temp_files_survive_a_blocked_publish(verified):
    _put_raw(verified.lease, "{truncated")

    verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    leftovers = sorted(
        p.name for p in verified.target.parent.glob("*.tmp")
    ) + sorted(p.name for p in verified.target.parent.glob("*.tmp*"))
    assert not leftovers, f"blocked publish left staging files behind: {leftovers}"
    verified.assert_intact("no temp leftovers")


def test_u_blocked_publish_names_the_active_holder_and_the_reason(verified):
    _live_lease_by(verified.lease, "home-desk", now=datetime.now(), ttl=60)

    result = verified.publish(dict(FULL_PAYLOAD, generated_at="SHOULD-NOT-LAND"))

    assert result["ok"] is False
    assert "home-desk" in str(result.get("error") or ""), (
        f"loser was not told the holder: {result.get('error')!r}"
    )
    verified.assert_intact("live lease held by the other machine")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
