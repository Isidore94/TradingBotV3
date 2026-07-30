"""Phase 2.8/2.9 (plan.md): writer lease + heartbeat.

Packet L2 note. None of the original tests below encoded the old fail-OPEN
behavior, so none were weakened or replaced - they all still assert exactly what
they asserted before. What *was* fail-open lived in the implementation
(``writer_lease._read`` swallowed every ``OSError``/``JSONDecodeError`` and
returned ``None``, which the acquire path then read as "nobody holds it") and
was never covered here at all. The second half of this file is the new
coverage: the role gate, unreadable/old-format leases failing closed, per-process
identity, enforced fencing, the machine-local cross-process lock, the shutdown
release wiring, and health telemetry that never reads healthy when it is absent.
"""

import json
import os
import socket
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import writer_lease as wl  # noqa: E402

NOW = datetime(2026, 7, 13, 9, 0)


def test_two_writers_cannot_hold_the_same_lease(tmp_path):
    lease = tmp_path / "report.lease"
    wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW)
    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="mini-pc", ttl_minutes=10, now=NOW + timedelta(minutes=5))
    # same holder renews freely
    renewed = wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW + timedelta(minutes=5))
    assert renewed["holder"] == "home"


def test_expired_lease_is_takeable_and_takeover_is_explicit(tmp_path):
    lease = tmp_path / "report.lease"
    wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW)
    grabbed = wl.acquire(lease, holder="mini-pc", now=NOW + timedelta(minutes=13))
    assert grabbed["holder"] == "mini-pc"
    # forced takeover before expiry must be explicit
    forced = wl.acquire(lease, holder="home", now=NOW + timedelta(minutes=14), takeover=True)
    assert forced["holder"] == "home" and forced["takeover"] is True


def test_bounded_clock_skew_cannot_cause_premature_takeover(tmp_path):
    lease = tmp_path / "report.lease"
    wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW)

    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="mini-pc", now=NOW + timedelta(minutes=11))
    assert wl.holder_of(lease, now=NOW + timedelta(minutes=11)) == "home"


def test_sleeping_holder_reacquires_before_its_next_publish(tmp_path):
    lease = tmp_path / "report.lease"
    wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW)

    assert wl.holder_of(lease, now=NOW + timedelta(minutes=13)) is None
    resumed = wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW + timedelta(minutes=13))
    assert resumed["holder"] == "home"
    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="mini-pc", now=NOW + timedelta(minutes=14))


def test_release_never_drops_someone_elses_lease(tmp_path):
    lease = tmp_path / "report.lease"
    wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW)
    assert wl.release(lease, holder="mini-pc") is False
    assert wl.holder_of(lease, now=NOW + timedelta(minutes=1)) == "home"
    assert wl.release(lease, holder="home") is True
    assert wl.holder_of(lease, now=NOW) is None


def test_publisher_skips_honestly_when_other_machine_holds_lease(tmp_path, monkeypatch):
    import autopilot_core as core

    target = tmp_path / "autopilot_today.txt"
    wl.acquire(target.with_suffix(".txt.lease"), holder="other-machine", ttl_minutes=10)
    monkeypatch.setattr(wl, "machine_holder_id", lambda: "this-machine")

    payload = {"generated_at": "x", "enabled": True, "auto_mode": "DESK", "ib_status": "", "regime": "",
               "longs": [], "shorts": [], "swing_picks": [], "alerts": [], "slots_done": [],
               "next_slot": "", "log_lines": [], "auto_longs": [], "auto_shorts": []}
    result = core.publish_away_report(payload, target)
    assert result["ok"] is False
    assert "active writer" in result["error"]
    assert not target.exists(), "the other machine's report was not clobbered"


def test_publisher_fails_closed_when_lease_check_errors(tmp_path, monkeypatch):
    import autopilot_core as core

    target = tmp_path / "autopilot_today.txt"
    target.write_text("previous verified report", encoding="utf-8")

    def broken_acquire(*_args, **_kwargs):
        raise OSError("shared drive lease is unreadable")

    monkeypatch.setattr(wl, "acquire", broken_acquire)
    result = core.publish_away_report(
        {"generated_at": "x", "enabled": True, "auto_mode": "DESK"},
        target,
    )

    assert result["ok"] is False
    assert "lease check failed" in result["error"]
    assert target.read_text(encoding="utf-8") == "previous verified report"


def test_heartbeat_writes_atomically(tmp_path):
    import autopilot_core as core

    path = core.write_heartbeat(current_job="swing 10:00", next_job="11:00", path=tmp_path / "hb.json")
    assert path is not None and path.exists()

    beat = json.loads(path.read_text(encoding="utf-8"))
    assert beat["current_job"] == "swing 10:00"
    assert beat["machine"] and beat["ts"]


# ===========================================================================
# Packet L2: the designated-writer role gate (Layer 1)
# ===========================================================================
_PAYLOAD = {
    "generated_at": "x",
    "enabled": True,
    "auto_mode": "AWAY",
    "ib_status": "",
    "regime": "",
    "longs": [],
    "shorts": [],
    "swing_picks": [],
    "alerts": [],
    "slots_done": [],
    "next_slot": "",
    "log_lines": [],
    "auto_longs": [],
    "auto_shorts": [],
}


@pytest.fixture
def unconfigured(monkeypatch, tmp_path):
    """No designated writer anywhere this machine can see."""
    import project_paths

    for key in (
        "TRADINGBOT_DESIGNATED_WRITER",
        "TRADINGBOTV3_DESIGNATED_WRITER",
        "TRADINGBOT_WRITER_ROLE",
        "TRADINGBOTV3_WRITER_ROLE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = tmp_path / "local_settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", settings, raising=False)
    return settings


def test_unconfigured_machine_publishes_nothing_at_all(tmp_path, unconfigured):
    """Fail CLOSED, not 'first machine wins': nothing on disk is touched."""
    import autopilot_core as core

    target = tmp_path / "drive" / "autopilot_today.txt"

    result = core.publish_away_report(dict(_PAYLOAD), target, archive=False)

    assert result["ok"] is False
    assert "designated" in result["error"].lower()
    assert not target.exists()
    assert not target.with_suffix(".txt.meta.json").exists()
    assert not target.with_suffix(".txt.lease").exists(), "an unconfigured machine took the lease"


def test_secondary_machine_is_read_only_and_says_so(tmp_path, monkeypatch):
    import autopilot_core as core

    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", "some-other-machine")
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "secondary")
    target = tmp_path / "autopilot_today.txt"
    target.write_text("previous verified report", encoding="utf-8")

    result = core.publish_away_report(dict(_PAYLOAD), target, archive=False)

    assert result["ok"] is False
    assert "read-only" in result["error"].lower() or "secondary" in result["error"].lower()
    assert target.read_text(encoding="utf-8") == "previous verified report"
    assert not target.with_suffix(".txt.lease").exists(), "the role gate must precede the lease"


def test_malformed_role_value_never_authorizes_publishing(tmp_path, monkeypatch):
    import autopilot_core as core

    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", socket.gethostname())
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "sort-of-a-writer-i-guess")
    target = tmp_path / "autopilot_today.txt"

    result = core.publish_away_report(dict(_PAYLOAD), target, archive=False)

    assert result["ok"] is False
    assert not target.exists()


# ===========================================================================
# Layer 3: states that used to fail OPEN now fail CLOSED
# ===========================================================================
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param("{half-written", id="truncated"),
        pytest.param("[]", id="not-an-object"),
        pytest.param('{"schema": "writer_lease_v9", "holder": "a", "expires_at": "2020-01-01T00:00:00"}',
                     id="unknown-schema-expired"),
        pytest.param('{"schema": "writer_lease_v2", "holder": "a", "expires_at": "2020-01-01T00:00:00"}',
                     id="current-schema-without-instance-id"),
    ],
)
def test_unverifiable_lease_state_blocks_instead_of_reading_as_free(tmp_path, raw):
    lease = tmp_path / "report.txt.lease"
    lease.write_text(raw, encoding="utf-8")
    before = lease.read_bytes()

    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="mini-pc", now=NOW)
    assert lease.read_bytes() == before, "questionable lease state was overwritten"


def test_unverifiable_lease_is_never_summarized_as_free(tmp_path):
    lease = tmp_path / "report.txt.lease"
    lease.write_text("{half-written", encoding="utf-8")

    with pytest.raises(wl.LeaseUnavailable):
        wl.holder_of(lease, now=NOW)


def test_unexpired_old_format_lease_is_never_treated_as_ours(tmp_path, monkeypatch):
    """writer_lease_v1 carries no instance id, so a machine of the same name
    cannot prove the lease is its own - it waits or takes over explicitly."""
    monkeypatch.setattr(socket, "gethostname", lambda: "home-desk")
    lease = tmp_path / "report.txt.lease"
    lease.write_text(
        json.dumps(
            {
                "schema": "writer_lease_v1",
                "holder": "home-desk",
                "acquired_at": NOW.isoformat(timespec="seconds"),
                "expires_at": (NOW + timedelta(minutes=10)).isoformat(timespec="seconds"),
                "takeover": False,
                "clock_skew_seconds": 120,
            }
        ),
        encoding="utf-8",
    )
    before = lease.read_bytes()

    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, now=NOW + timedelta(minutes=1))
    assert lease.read_bytes() == before

    # ...and once it expires the normal acquisition path recovers it.
    granted = wl.acquire(lease, now=NOW + timedelta(hours=3))
    assert granted["takeover"] is False
    assert granted["instance_id"] == wl.process_instance_id()


def test_identity_separates_two_processes_and_two_process_starts(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "home-desk")
    monkeypatch.setattr(os, "getpid", lambda: 1111)
    first = wl.machine_holder_id()
    monkeypatch.setattr(os, "getpid", lambda: 2222)
    second = wl.machine_holder_id()

    assert first != second, "hostname-only identity conflates two processes into one writer"
    instance = wl.process_instance_id()[:12]
    assert instance in first and instance in second, "identity carries no process instance"


def test_fencing_generation_is_monotonic_and_enforced_on_the_write_path(tmp_path):
    lease = tmp_path / "report.txt.lease"
    first = wl.acquire(lease, holder="home", ttl_minutes=10, now=NOW - timedelta(hours=2))
    second = wl.acquire(lease, holder="mini-pc", ttl_minutes=10, now=NOW)
    assert first["generation"] < second["generation"]

    # A newer instance fences us off mid-flight: the pre-replacement check fails.
    stolen = dict(second, generation=second["generation"] + 1, instance_id="another-instance")
    lease.write_text(json.dumps(stolen), encoding="utf-8")
    with pytest.raises(wl.LeaseUnavailable):
        wl.assert_still_owned(
            lease, holder="mini-pc", generation=second["generation"], now=NOW
        )


# ===========================================================================
# Layer 2: machine-local cross-process exclusion
# ===========================================================================
def test_local_lock_is_reentrant_for_one_thread_and_exclusive_across_threads(tmp_path):
    from local_writer_lock import local_writer_lock, lock_key_for_path

    key = lock_key_for_path(tmp_path / "report.txt.lease")
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with local_writer_lock(key):
            with local_writer_lock(key):  # re-entrant: must not self-deadlock
                order.append("held")
                inside.set()
                release.wait(timeout=10)
        order.append("released")

    def contender() -> None:
        inside.wait(timeout=10)
        with local_writer_lock(key, timeout_seconds=10):
            order.append("second")

    threads = [threading.Thread(target=holder), threading.Thread(target=contender)]
    for thread in threads:
        thread.start()
    inside.wait(timeout=10)
    release.set()
    for thread in threads:
        thread.join(timeout=20)

    assert order == ["held", "released", "second"], order


def test_local_lock_timeout_fails_closed(tmp_path):
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

    key = lock_key_for_path(tmp_path / "report.txt.lease")
    inside = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with local_writer_lock(key):
            inside.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        inside.wait(timeout=10)
        with pytest.raises(LocalLockUnavailable):
            with local_writer_lock(key, timeout_seconds=0.05):
                pass
    finally:
        release.set()
        thread.join(timeout=10)


# ===========================================================================
# Layer 4: release wiring, and the hard-kill path
# ===========================================================================
def test_shutdown_release_hands_the_lease_back(tmp_path):
    import autopilot_core as core

    target = tmp_path / "autopilot_today.txt"
    assert core.publish_away_report(dict(_PAYLOAD), target, archive=False)["ok"]
    lease = target.with_suffix(".txt.lease")
    assert lease.exists()

    assert core.release_away_report_lease(target) is True
    assert not lease.exists(), "a clean shutdown must not hold the writer slot until TTL"
    # Releasing twice is a harmless no-op, which is what makes the belt-and-
    # braces call in MainWindow.closeEvent safe.
    assert core.release_away_report_lease(target) is True


def test_shutdown_release_never_drops_another_writers_lease(tmp_path):
    import autopilot_core as core

    target = tmp_path / "autopilot_today.txt"
    lease = target.with_suffix(".txt.lease")
    wl.acquire(lease, holder="the-other-machine", ttl_minutes=10)

    assert core.release_away_report_lease(target) is False
    assert lease.exists()


def test_a_hard_killed_writer_recovers_by_expiry_not_by_inheritance(tmp_path, monkeypatch):
    """release() never ran (the process was killed). The lease must not be
    inheritable by a later process, but must not wedge the slot either."""
    lease = tmp_path / "report.txt.lease"
    killed = wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW)

    # A different process instance, same name: no inheritance while it stands.
    monkeypatch.setattr(wl, "_PROCESS_INSTANCE_ID", "a-later-process-instance")
    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="home-desk", now=NOW + timedelta(minutes=1))

    # Once the TTL plus the skew window passes, the slot is recoverable.
    recovered = wl.acquire(
        lease, holder="home-desk", now=NOW + timedelta(minutes=30)
    )
    assert recovered["generation"] > killed["generation"]
    assert recovered["takeover"] is False


# ===========================================================================
# Layer 5: telemetry
# ===========================================================================
def test_health_telemetry_absent_or_corrupt_never_reads_as_healthy(tmp_path):
    from writer_health import read_writer_health, write_writer_health

    path = tmp_path / "writer_health.json"
    assert read_writer_health(path=path)["healthy"] is False

    write_writer_health({"status": "published", "healthy": True}, path=path)
    assert read_writer_health(path=path)["healthy"] is True

    path.write_bytes(b'{"schema": "writer_health_v1", "hol')
    corrupt = read_writer_health(path=path)
    assert corrupt["healthy"] is False and corrupt["status"] == "corrupt"


def test_publish_telemetry_names_the_blocker(tmp_path, monkeypatch):
    import autopilot_core as core

    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(diagnostics))
    target = tmp_path / "autopilot_today.txt"
    wl.acquire(target.with_suffix(".txt.lease"), holder="the-other-machine", ttl_minutes=60)

    result = core.publish_away_report(dict(_PAYLOAD), target, archive=False)

    assert result["ok"] is False
    health = json.loads((diagnostics / "writer_health.json").read_text(encoding="utf-8"))
    assert health["healthy"] is False
    assert "the-other-machine" in health["last_failure"]["message"]


# ===========================================================================
# Packet L4: regressions for the holes the independent reviews found
# ===========================================================================
def test_a_timezone_aware_expiry_blocks_instead_of_raising(tmp_path):
    """A trailing ``Z`` or ``+02:00`` anywhere in a lease used to raise a bare
    TypeError out of the publish path -- no refusal, no telemetry, no report."""
    lease = tmp_path / "report.txt.lease"
    mine = wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW)

    payload = json.loads(lease.read_text(encoding="utf-8"))
    payload.update(
        holder="the-other-machine",
        instance_id="z" * 32,
        expires_at="2099-01-01T00:00:00+02:00",
        expires_at_utc="2099-01-01T00:00:00+02:00",
    )
    lease.write_text(json.dumps(payload), encoding="utf-8")

    assert wl.holder_of(lease) == "the-other-machine"
    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="mini-pc")
    with pytest.raises(wl.LeaseUnavailable):
        wl.assert_still_owned(lease, holder=mine["holder"], generation=mine["generation"])


def test_an_aware_override_expiry_is_honored_not_crashed(monkeypatch):
    from datetime import timezone

    import writer_role

    monkeypatch.setenv("TRADINGBOT_WRITER_OVERRIDE", "true")
    monkeypatch.setenv(
        "TRADINGBOT_WRITER_OVERRIDE_EXPIRES_AT",
        (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(timespec="seconds"),
    )
    assert writer_role.resolve_emergency_override().active is True

    monkeypatch.setenv(
        "TRADINGBOT_WRITER_OVERRIDE_EXPIRES_AT",
        (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(timespec="seconds"),
    )
    assert writer_role.resolve_emergency_override().active is False


def test_an_emergency_override_must_be_bounded_in_duration(monkeypatch):
    """'Time-bounded' has to mean a usable window, not the year 9999."""
    import writer_role

    monkeypatch.setenv("TRADINGBOT_WRITER_OVERRIDE", "true")
    for expiry in ("9999-12-31T23:59:59", "2999-01-01T00:00:00"):
        monkeypatch.setenv("TRADINGBOT_WRITER_OVERRIDE_EXPIRES_AT", expiry)
        override = writer_role.resolve_emergency_override()
        assert override.active is False, f"{expiry} was accepted as time-bounded"
        assert str(writer_role.MAX_OVERRIDE_WINDOW_HOURS) in override.rejected_because

    monkeypatch.setenv(
        "TRADINGBOT_WRITER_OVERRIDE_EXPIRES_AT",
        (datetime.now() + timedelta(hours=2)).isoformat(timespec="seconds"),
    )
    assert writer_role.resolve_emergency_override().active is True


def test_the_fencing_generation_survives_a_clean_restart(tmp_path):
    """``release`` deletes the lease, which used to be the only durable copy of
    the counter -- so the next run handed out generation 1 all over again and
    two publications by different holders could carry the same number."""
    lease = tmp_path / "report.txt.lease"
    generations = [wl.acquire(lease, holder="home-desk")["generation"] for _ in range(3)]
    assert wl.release(lease, holder="home-desk") is True
    assert not lease.exists()

    wl._GENERATION_FLOOR.clear()  # exactly what a process restart does
    restarted = wl.acquire(lease, holder="home-desk")["generation"]

    assert restarted > max(generations), (
        f"generation went backwards across a restart: {generations} then {restarted}"
    )


def test_a_renewal_does_not_fence_off_the_renewing_writer(tmp_path):
    lease = tmp_path / "report.txt.lease"
    acquired = wl.acquire(lease, holder="home-desk", ttl_minutes=10, now=NOW)

    renewed = wl.renew(lease, holder="home-desk", now=NOW + timedelta(minutes=2))

    assert renewed is not None
    assert renewed["generation"] == acquired["generation"], (
        "a renewal advanced the fencing generation, which assert_still_owned "
        "compares by equality -- it would abort the renewing writer's own publish"
    )
    assert renewed["renewed_at"], "a renewal must be distinguishable from an acquisition"
    wl.assert_still_owned(
        lease,
        holder=acquired["holder"],
        generation=acquired["generation"],
        now=NOW + timedelta(minutes=3),
    )


def test_an_unauditable_takeover_is_refused(tmp_path):
    """The design requires an *auditable* emergency takeover, so a takeover
    whose audit record cannot be written must not happen at all."""
    lease = tmp_path / "report.txt.lease"
    wl.acquire(lease, holder="the-other-machine", ttl_minutes=60, now=NOW)
    before = lease.read_bytes()
    (tmp_path / f"{lease.name}.takeover_audit.jsonl").mkdir()  # a real, unwritable path

    with pytest.raises(wl.LeaseUnavailable):
        wl.acquire(lease, holder="mini-pc", now=NOW + timedelta(minutes=1), takeover=True)

    assert lease.read_bytes() == before, "the live lease was broken without an audit record"


def test_an_absurd_clock_skew_claim_cannot_wedge_the_writer_slot(tmp_path):
    lease = tmp_path / "report.txt.lease"
    wl.acquire(lease, holder="the-other-machine", ttl_minutes=10, now=NOW)
    payload = json.loads(lease.read_text(encoding="utf-8"))
    payload["clock_skew_seconds"] = 10**9
    lease.write_text(json.dumps(payload), encoding="utf-8")

    recovered = wl.acquire(lease, holder="mini-pc", now=NOW + timedelta(hours=6))

    assert recovered["holder"] == "mini-pc"
    assert recovered["clock_skew_seconds"] <= wl.MAX_CLOCK_SKEW_SECONDS


def test_the_local_lock_fails_closed_when_no_os_primitive_exists(monkeypatch, tmp_path):
    """held=True with only an in-process RLock behind it is worse than a
    refusal: telemetry says the machine is protected and it is not."""
    import local_writer_lock as lwl

    monkeypatch.setattr(lwl, "_kernel32", lambda: None)
    monkeypatch.setattr(lwl._FileLockLayer, "_try_lock", staticmethod(lambda handle: None))

    with pytest.raises(lwl.LocalLockUnavailable):
        with lwl.local_writer_lock(lwl.lock_key_for_path(tmp_path / "x.txt")):
            pass


def test_health_telemetry_keeps_both_the_last_publication_and_the_last_failure(tmp_path):
    from writer_health import read_writer_health, write_writer_health

    path = tmp_path / "writer_health.json"
    write_writer_health(
        {
            "status": "published",
            "healthy": True,
            "last_verified_publication": {"holder": "home-desk", "generation": 4, "sha256": "abc"},
        },
        path=path,
    )
    write_writer_health(
        {
            "status": "blocked: configuration",
            "healthy": False,
            "last_failure": {"at": "now", "kind": "configuration", "message": "read-only secondary"},
        },
        path=path,
    )

    state = read_writer_health(path=path)
    assert state["last_verified_publication"]["holder"] == "home-desk", (
        "the first refusal after a good publish erased the record of that publish"
    )
    assert state["last_failure"]["kind"] == "configuration"
    assert state["healthy"] is False


def test_stale_health_telemetry_never_reads_as_healthy(tmp_path):
    from writer_health import read_writer_health

    path = tmp_path / "writer_health.json"
    path.write_text(
        json.dumps(
            {
                "schema": "writer_health_v1",
                "written_at": "2019-01-01T00:00:00",
                "status": "published",
                "healthy": True,
                "machine": "LONG-DEAD-BOX",
            }
        ),
        encoding="utf-8",
    )

    state = read_writer_health(path=path)

    assert state["healthy"] is False, "a seven-year-old artifact read as healthy"
    assert "stale" in state["status"]


def test_a_read_only_secondary_does_not_rewrite_shared_watchlists(tmp_path, monkeypatch):
    """The publication gate protected one file out of the shared mutable set."""
    import autopilot_core as core
    import project_paths

    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(project_paths, "SHARED_HOME_DIR", shared, raising=False)
    for name, attr in (
        ("longs.txt", "LONGS_FILE"),
        ("shorts.txt", "SHORTS_FILE"),
        ("autolongs.txt", "AUTO_LONGS_FILE"),
        ("autoshorts.txt", "AUTO_SHORTS_FILE"),
    ):
        (shared / name).write_text("TRADERTYPED\n", encoding="utf-8")
        monkeypatch.setattr(core, attr, shared / name)
    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", "SOME-OTHER-MACHINE")
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "secondary")

    assert core.write_auto_watchlists(["PROBEA"], ["PROBEB"]) is False
    assert core.write_bouncebot_watchlists(["PROBEC"], ["PROBED"]) is False
    assert core.append_watchlist_symbols(shared / "longs.txt", ["PROBEE"]) == []

    for name in ("longs.txt", "shorts.txt", "autolongs.txt", "autoshorts.txt"):
        assert (shared / name).read_text(encoding="utf-8") == "TRADERTYPED\n", (
            f"a read-only secondary rewrote the shared {name}"
        )

    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", socket.gethostname())
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "designated_writer")
    assert core.write_bouncebot_watchlists(["OKNOW"], []) is True
    assert (shared / "longs.txt").read_text(encoding="utf-8").split() == ["OKNOW"]


def test_a_machine_local_path_is_never_gated_by_the_writer_role(tmp_path, monkeypatch):
    import autopilot_core as core

    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", "SOME-OTHER-MACHINE")
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "secondary")

    target = tmp_path / "local_longs.txt"
    assert core.write_watchlist_file(target, ["AAPL"]) is True
    assert target.read_text(encoding="utf-8").split() == ["AAPL"]


def test_an_interrupt_between_the_two_replaces_restores_the_verified_pair(
    tmp_path, monkeypatch
):
    """A BaseException (hard interrupt, Qt/thread abort) between the report
    replace and the metadata replace used to skip every handler and therefore
    the rollback, leaving the pair describing two different publications."""
    import autopilot_core as core

    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", socket.gethostname())
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "designated_writer")
    target = tmp_path / "drive" / "autopilot_today.txt"
    metadata = target.with_suffix(target.suffix + ".meta.json")

    first = core.publish_away_report(dict(_PAYLOAD, generated_at="FIRST"), target, archive=False)
    assert first["ok"], first
    report_before, metadata_before = target.read_bytes(), metadata.read_bytes()

    real_dumps = core.json.dumps

    def interrupt_on_metadata(obj, *args, **kwargs):
        if isinstance(obj, dict) and obj.get("schema") == "away_report_publish_v1":
            raise KeyboardInterrupt("simulated hard interrupt")
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(core.json, "dumps", interrupt_on_metadata)
    with pytest.raises(KeyboardInterrupt):
        core.publish_away_report(dict(_PAYLOAD, generated_at="SECOND"), target, archive=False)

    assert target.read_bytes() == report_before, (
        "the shared report was left replaced while its metadata still described "
        "the previous publication"
    )
    assert metadata.read_bytes() == metadata_before


def test_write_away_report_refuses_to_report_a_refusal_as_a_write(tmp_path, monkeypatch):
    import autopilot_core as core

    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", "SOME-OTHER-MACHINE")
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "secondary")
    target = tmp_path / "drive" / "autopilot_today.txt"

    with pytest.raises(RuntimeError):
        core.write_away_report(dict(_PAYLOAD), target)
    assert not target.exists()


def test_a_torn_report_metadata_pair_is_detected_on_the_next_publish(tmp_path, monkeypatch):
    import autopilot_core as core

    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("TRADINGBOT_DESIGNATED_WRITER", socket.gethostname())
    monkeypatch.setenv("TRADINGBOT_WRITER_ROLE", "designated_writer")
    target = tmp_path / "drive" / "autopilot_today.txt"

    assert core.publish_away_report(dict(_PAYLOAD), target, archive=False)["ok"]
    target.write_text("a report nobody metadata describes\n", encoding="utf-8")

    result = core.publish_away_report(dict(_PAYLOAD, generated_at="NEXT"), target, archive=False)

    assert result["ok"], result
    assert result.get("previous_pair_disagreement"), (
        "a report whose sha256 does not match its .meta.json passed unnoticed"
    )
