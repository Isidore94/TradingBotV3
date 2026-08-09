"""AI store location and off-hours window (plan sec 3.3, sec 2 / 6.1).

Two invariants carry the weight here. The store must refuse to live inside the
Drive-synced home folder or the research lake, because separate storage classes
with separate writers is what stops an AI-job bug from corrupting operational
or lake data. And no job may launch during market hours no matter what the
window setting says -- that is a plan sec 2 hard rule, and a fat-fingered
window must not be able to put a 14GB model load in front of the open.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")


def _settings(**values):
    """Patch the settings reader the modules actually resolve through.

    ``_paths()`` reaches ``scripts.project_paths`` (a namespace-package
    import), which is a *different module object* from a plain
    ``import project_paths``. Patching the wrong one silently does nothing, so
    resolve it the same way the code does -- the precedent is
    tests/test_warehouse_config.py.
    """
    from ai_jobs import store

    return mock.patch.object(
        store._paths(),
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def test_store_is_disabled_until_configured():
    from ai_jobs import store

    with _settings():
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(store.AI_STORE_DIR_ENV, None)
            assert store.get_ai_store_dir() is None
            assert store.ai_store_enabled() is False


def test_store_refuses_a_path_inside_the_shared_home_folder(tmp_path):
    from ai_jobs import store

    home = tmp_path / "home"
    (home / "ai_store").mkdir(parents=True)
    with mock.patch.object(store._paths(), "SHARED_HOME_DIR", home):
        with _settings(ai_store_dir=str(home / "ai_store")):
            with pytest.raises(ValueError, match="shared home folder"):
                store.get_ai_store_dir()
            # A misconfiguration must read as disabled, not crash callers.
            assert store.ai_store_enabled() is False


def test_store_refuses_a_path_inside_the_research_lake(tmp_path):
    from ai_jobs import store

    lake = tmp_path / "lake"
    (lake / "ai_store").mkdir(parents=True)
    with mock.patch(
        "research_warehouse.config.get_research_store_dir", return_value=lake
    ):
        with _settings(ai_store_dir=str(lake / "ai_store")):
            with pytest.raises(ValueError, match="research lake"):
                store.get_ai_store_dir()


def test_a_network_path_outside_both_is_accepted(tmp_path):
    from ai_jobs import store

    target = tmp_path / "nas" / "ai_store"
    with _settings(ai_store_dir=str(target)):
        assert store.get_ai_store_dir() == target
        assert store.ai_store_enabled() is True


def test_layout_bootstrap_is_idempotent_and_additive(tmp_path):
    from ai_jobs import store

    target = tmp_path / "ai_store"
    (target).mkdir()
    keep = target / "digests" / "2026"
    keep.mkdir(parents=True)
    (keep / "existing.json").write_text("{}", encoding="utf-8")

    store.ensure_ai_store_layout(target)
    store.ensure_ai_store_layout(target)

    for name in store.AI_STORE_SUBDIRS:
        assert (target / name).is_dir()
    assert (keep / "existing.json").exists(), "bootstrap must never clear content"


def test_unreachable_store_is_reported_not_raised(tmp_path):
    from ai_jobs import store

    target = tmp_path / "ai_store"
    with mock.patch.object(
        store, "ensure_ai_store_layout", side_effect=OSError("network name no longer available")
    ):
        ok, reason = store.store_available(target)

    assert ok is False
    assert "unreachable" in reason
    # A sleeping NAS is "no digest tonight", never a half-written artifact.


def test_available_store_round_trips_a_write_probe(tmp_path):
    from ai_jobs import store

    target = tmp_path / "ai_store"
    ok, reason = store.store_available(target)

    assert ok is True
    assert "ready" in reason
    assert not (target / ".write_probe").exists(), "probe must clean up after itself"


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------
def _no_session(monkeypatch):
    """Neutralize the market block so window logic can be tested alone.

    Patches the block itself, not the calendar lookup: an unanswerable
    calendar now *blocks* rather than waving the job through, so stubbing
    ``_session_bounds`` to a failure no longer means "no session today".
    """
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")


def test_configured_window_converts_the_desk_clock_correctly(monkeypatch):
    """The trader asked for 22:00-06:00 Pacific; that is 01:00-09:00 ET."""
    from ai_jobs import window

    _no_session(monkeypatch)
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        # 22:00 Pacific on a Tuesday == 01:00 ET Wednesday: open.
        assert window.in_offhours_window(datetime(2026, 8, 11, 22, 0, tzinfo=PACIFIC))
        # 05:59 Pacific == 08:59 ET: still open.
        assert window.in_offhours_window(datetime(2026, 8, 12, 5, 59, tzinfo=PACIFIC))
        # 06:00 Pacific == 09:00 ET: closed.
        assert not window.in_offhours_window(datetime(2026, 8, 12, 6, 0, tzinfo=PACIFIC))
        # 21:59 Pacific == 00:59 ET: not yet open.
        assert not window.in_offhours_window(datetime(2026, 8, 11, 21, 59, tzinfo=PACIFIC))


def test_default_window_wraps_past_midnight(monkeypatch):
    from ai_jobs import window

    _no_session(monkeypatch)
    with _settings():  # 18:30-08:00 ET default
        assert window.in_offhours_window(datetime(2026, 8, 11, 19, 0, tzinfo=ET))
        assert window.in_offhours_window(datetime(2026, 8, 12, 2, 0, tzinfo=ET))
        assert window.in_offhours_window(datetime(2026, 8, 12, 7, 59, tzinfo=ET))
        assert not window.in_offhours_window(datetime(2026, 8, 12, 12, 0, tzinfo=ET))


def test_weekends_are_open_all_day(monkeypatch):
    from ai_jobs import window

    _no_session(monkeypatch)
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        saturday_noon = datetime(2026, 8, 8, 12, 0, tzinfo=ET)
        assert saturday_noon.weekday() == 5
        assert window.in_offhours_window(saturday_noon)


def test_market_session_blocks_inference_whatever_the_window_says(monkeypatch):
    """The sec 2 hard rule outranks the trader-adjustable preference."""
    from ai_jobs import window

    open_local = datetime(2026, 8, 11, 9, 30, tzinfo=ET)
    close_local = datetime(2026, 8, 11, 16, 0, tzinfo=ET)
    monkeypatch.setattr(window, "_session_bounds", lambda day: (open_local, close_local))

    # A window that (wrongly) declares the whole day open.
    with _settings(ai_offhours_start="00:00", ai_offhours_end="23:59"):
        midday = datetime(2026, 8, 11, 12, 0, tzinfo=ET)
        assert window.in_offhours_window(midday) is True
        allowed, reason = window.launch_allowed(midday)
        assert allowed is False
        assert "market session" in reason


def test_launch_is_allowed_inside_the_configured_window(monkeypatch):
    from ai_jobs import window

    open_local = datetime(2026, 8, 12, 9, 30, tzinfo=ET)
    close_local = datetime(2026, 8, 12, 16, 0, tzinfo=ET)
    monkeypatch.setattr(window, "_session_bounds", lambda day: (open_local, close_local))

    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        allowed, reason = window.launch_allowed(datetime(2026, 8, 12, 2, 0, tzinfo=ET))
        assert allowed is True
        assert "window open" in reason


def test_a_job_that_cannot_finish_before_the_close_does_not_launch(monkeypatch):
    """The trader's window ends 30 min before the open, so a long job must
    skip rather than start and run into market hours."""
    from ai_jobs import window

    _no_session(monkeypatch)
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        near_close = datetime(2026, 8, 12, 8, 45, tzinfo=ET)  # 15 min left
        allowed, reason = window.launch_allowed(near_close, reserve_minutes=40)
        assert allowed is False
        assert "running into the open" in reason

        # The same moment is fine for a short job.
        allowed, _reason = window.launch_allowed(near_close, reserve_minutes=5)
        assert allowed is True


def test_minutes_until_close_is_none_when_the_window_is_shut(monkeypatch):
    from ai_jobs import window

    _no_session(monkeypatch)
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        assert window.minutes_until_window_close(datetime(2026, 8, 12, 12, 0, tzinfo=ET)) is None


def test_preopen_guard_extends_the_block_when_configured(monkeypatch):
    from ai_jobs import window

    open_local = datetime(2026, 8, 12, 9, 30, tzinfo=ET)
    close_local = datetime(2026, 8, 12, 16, 0, tzinfo=ET)
    monkeypatch.setattr(window, "_session_bounds", lambda day: (open_local, close_local))

    at_0845 = datetime(2026, 8, 12, 8, 45, tzinfo=ET)
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        assert window.launch_allowed(at_0845)[0] is True
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00", ai_preopen_guard_minutes=60):
        allowed, reason = window.launch_allowed(at_0845)
        assert allowed is False
        assert "pre-open guard" in reason


def test_describe_window_reports_both_clocks(monkeypatch):
    from ai_jobs import window

    _no_session(monkeypatch)
    with _settings(ai_offhours_start="01:00", ai_offhours_end="09:00"):
        described = window.describe_window()

    assert described["window_et"] == "01:00-09:00"
    assert ":" in described["window_desk_local"]
    assert described["launch_allowed"] in {"yes", "no"}


# ---------------------------------------------------------------------------
# hard-rule gaps (checkpoint review 2026-08-08 second review)
# ---------------------------------------------------------------------------
def test_an_unanswerable_calendar_blocks_instead_of_unlocking_the_day():
    """"No local inference during market hours" is a plan sec 2 hard rule.

    The lookup used to return None on a failed import or a raising calendar,
    and the caller read None as "not a session day" -- so a broken calendar
    silently unlocked inference for the whole trading day. Missing data is
    uncertainty, never confirmation.
    """
    from ai_jobs import window

    def _broken(day):
        raise window.SessionLookupFailed("calendar unavailable")

    with mock.patch.object(window, "_session_bounds", _broken):
        weekday_midnight = datetime(2026, 8, 12, 0, 30, tzinfo=ET)
        blocked = window.market_session_block(weekday_midnight)
        assert blocked, "an unanswerable calendar must block, not wave the job through"
        assert "cannot determine" in blocked
        allowed, reason = window.launch_allowed(weekday_midnight)
        assert allowed is False
        assert "cannot determine" in reason


def test_a_raising_calendar_becomes_a_lookup_failure_not_a_free_day():
    from ai_jobs import window

    with mock.patch(
        "market_session.get_market_session_window", side_effect=RuntimeError("boom")
    ):
        try:
            window._session_bounds(date(2026, 8, 12))
        except window.SessionLookupFailed as exc:
            assert "boom" in str(exc)
        else:  # pragma: no cover - the failure path is the whole point
            raise AssertionError("_session_bounds must fail closed, not return None")


def test_weekends_still_short_circuit_without_consulting_the_calendar():
    # The fail-closed rule must not turn every weekend into a refusal.
    from ai_jobs import window

    def _explode(day):  # pragma: no cover - must never be reached
        raise AssertionError("weekend must not need the calendar")

    with mock.patch.object(window, "_session_bounds", _explode):
        saturday = datetime(2026, 8, 8, 12, 0, tzinfo=ET)
        assert saturday.weekday() == 5
        assert window.market_session_block(saturday) == ""


def test_read_only_availability_never_creates_or_probes_the_store(tmp_path):
    """--status says "print state, run nothing"; that includes writing nothing."""
    from ai_jobs import store

    root = tmp_path / "ai_store"
    root.mkdir()
    with _settings(ai_store_dir=str(root)):
        ok, reason = store.store_available(read_only=True)

    assert ok is True
    assert "not write-probed" in reason
    # No skeleton, no probe file: the directory is exactly as it was found.
    assert sorted(p.name for p in root.iterdir()) == []


def test_read_only_availability_still_reports_an_absent_store(tmp_path):
    from ai_jobs import store

    missing = tmp_path / "nope"
    with _settings(ai_store_dir=str(missing)):
        ok, reason = store.store_available(read_only=True)

    assert ok is False
    assert "does not exist" in reason
    assert not missing.exists()


def test_the_writable_check_still_creates_and_probes(tmp_path):
    # The job path must keep proving writability before anything is written.
    from ai_jobs import store

    root = tmp_path / "ai_store"
    with _settings(ai_store_dir=str(root)):
        ok, reason = store.store_available()

    assert ok is True and "ready" in reason
    assert sorted(p.name for p in root.iterdir()) == ["briefs", "digests", "logs", "models", "retros"]
    assert not (root / ".write_probe").exists()


def test_store_subdirs_can_resolve_without_creating(tmp_path):
    from ai_jobs import ledger, store

    root = tmp_path / "ai_store"
    root.mkdir()
    with _settings(ai_store_dir=str(root)):
        logs = store.store_logs_dir(create=False)
        path = ledger.ledger_path(create=False)

    assert logs == root / "logs"
    assert path == root / "logs" / ledger.LEDGER_NAME
    assert not logs.exists(), "resolving a path must not create it"


def test_the_preopen_reserve_is_part_of_the_session_block(monkeypatch):
    """A 14 GB model load must not start 60 seconds before the bell.

    The guard defaulted to 0, on the reasoning that the session block already
    protects the session. It does not protect the run-up: the desk's own
    launch task fires at 06:00 Pacific and pre-market prep is competing for
    the box well before the open (Sol 5.6 verification review, item 9).
    """
    from ai_jobs import window

    assert window.DEFAULT_PREOPEN_GUARD_MINUTES == 15

    open_local = datetime(2026, 8, 12, 9, 30, tzinfo=ET)
    close_local = datetime(2026, 8, 12, 16, 0, tzinfo=ET)
    monkeypatch.setattr(window, "_session_bounds", lambda day: (open_local, close_local))

    with _settings():  # no explicit setting: the default reserve applies
        # 09:20 is outside the session but inside the reserve.
        blocked = window.market_session_block(datetime(2026, 8, 12, 9, 20, tzinfo=ET))
        assert blocked
        assert "pre-open guard" in blocked
        # 09:10 is clear of both.
        assert window.market_session_block(datetime(2026, 8, 12, 9, 10, tzinfo=ET)) == ""


def test_force_cannot_spend_the_preopen_reserve(tmp_path):
    # --force skips window *timing*. The reserve lives inside the session
    # block, which force never reaches, so this holds by construction -- and
    # that construction is what this test pins.
    from ai_jobs import runner, store

    led = tmp_path / "ledger.jsonl"
    ran = []

    open_local = datetime(2026, 8, 12, 9, 30, tzinfo=ET)
    close_local = datetime(2026, 8, 12, 16, 0, tzinfo=ET)

    from ai_jobs import window as window_module

    with (
        mock.patch.object(store, "store_available", return_value=(True, "ready")),
        mock.patch.object(window_module, "_session_bounds", lambda day: (open_local, close_local)),
        _settings(),
    ):
        report = runner.run_slots(
            [runner.JobSlot(name="ai_summary", run=lambda **k: ran.append(True))],
            now=datetime(2026, 8, 12, 9, 20, tzinfo=ET),
            force=True,
            ledger_path=led,
        )

    assert ran == []
    assert report.skipped == 1
    assert "pre-open guard" in report.results[0]["reason"]
