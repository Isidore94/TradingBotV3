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
from datetime import datetime
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
    """Neutralize the market block so window logic can be tested alone."""
    from ai_jobs import window

    monkeypatch.setattr(window, "_session_bounds", lambda day: None)


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
