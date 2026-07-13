"""Phase 2.8/2.9 (plan.md): writer lease + heartbeat."""

import sys
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
    grabbed = wl.acquire(lease, holder="mini-pc", now=NOW + timedelta(minutes=11))
    assert grabbed["holder"] == "mini-pc"
    # forced takeover before expiry must be explicit
    forced = wl.acquire(lease, holder="home", now=NOW + timedelta(minutes=12), takeover=True)
    assert forced["holder"] == "home" and forced["takeover"] is True


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
    import json

    beat = json.loads(path.read_text(encoding="utf-8"))
    assert beat["current_job"] == "swing 10:00"
    assert beat["machine"] and beat["ts"]
