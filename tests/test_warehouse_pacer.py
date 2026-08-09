"""The shared IB pacer and capture isolation (plan sec 5.3, risks R1/R2).

Pinned here, in the plan's own words:

* champion traffic is "counted, never delayed, never queued" - the pacer has
  no path that can refuse or slow a champion;
* capture yields instantly to champion activity and on IB error 162/366;
* a capture-caused pacing error NEVER reaches the champion fetch boundary's
  Yahoo-only circuit breaker (`_IBKR_HISTORICAL_FAILURE_COUNT`), which is the
  BF.B/LC blackout precedent;
* client-ID allocation is asserted at connect: 1003 retired, 1010 streamer,
  1011 nightly backfill, mini-PC excluded.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import pacer as pacer_mod
from scripts.research_warehouse.pacer import IbPacer

UTC = timezone.utc
T0 = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)


@pytest.fixture()
def pacer():
    return IbPacer(clock=lambda: T0)


# --- champions are never metered -------------------------------------------
def test_champion_requests_are_counted_never_gated(pacer):
    for index in range(200):  # far past any capture budget
        pacer.note_champion_request("daily_bars", f"SYM{index}", now=T0)
    snapshot = pacer.snapshot(now=T0)
    assert snapshot.champion_requests_in_window == 200
    # Counting is the whole champion interaction: there is no return value to
    # obey and no method that can make a champion wait.
    assert pacer.note_champion_request("daily_bars", "AAPL", now=T0) is None
    signature = inspect.signature(IbPacer.note_champion_request)
    assert "timeout" not in signature.parameters and "block" not in signature.parameters


def test_capture_budget_shrinks_as_champions_consume_the_floor(pacer):
    assert pacer.snapshot(now=T0).capture_budget == pacer_mod.CAPTURE_WINDOW_ALLOWANCE
    for index in range(50):
        pacer.note_champion_request("daily_bars", f"SYM{index}", now=T0)
    # 60 published - 50 champion = 10 left, below capture's own allowance.
    assert pacer.snapshot(now=T0).capture_budget == 10
    for index in range(15):
        pacer.note_champion_request("daily_bars", f"MORE{index}", now=T0)
    assert pacer.snapshot(now=T0).capture_budget == 0
    quiet = T0 + timedelta(seconds=2)
    assert pacer.try_acquire(now=quiet).reason == pacer_mod.DENY_BUDGET_EXHAUSTED


def test_capture_yields_while_a_champion_request_is_in_flight(pacer):
    with pacer.champion_window("intraday_bars", "AAPL"):
        decision = pacer.try_acquire(now=T0)
        assert decision.granted is False
        assert decision.reason == pacer_mod.DENY_CHAMPION_ACTIVE
    # Still inside the quiet period right after the champion finished.
    assert pacer.try_acquire(now=T0).reason == pacer_mod.DENY_CHAMPION_ACTIVE
    assert pacer.try_acquire(now=T0 + timedelta(seconds=2)).granted is True


def test_capture_budget_recovers_after_the_window_rolls(pacer):
    quiet = T0 + timedelta(seconds=2)
    for index in range(pacer_mod.CAPTURE_WINDOW_ALLOWANCE):
        assert pacer.try_acquire(key=f"k{index}", now=quiet).granted is True
    denied = pacer.try_acquire(key="one-too-many", now=quiet)
    assert denied.reason == pacer_mod.DENY_BUDGET_EXHAUSTED and denied.wait_seconds > 0

    later = quiet + timedelta(seconds=pacer_mod.PACING_WINDOW_SECONDS + 1)
    assert pacer.try_acquire(key="after-window", now=later).granted is True


# --- pacing errors ---------------------------------------------------------
@pytest.mark.parametrize("code", sorted(pacer_mod.PACING_ERROR_CODES))
def test_capture_backs_off_on_162_and_366(pacer, code):
    quiet = T0 + timedelta(seconds=2)
    assert pacer.try_acquire(key="a", now=quiet).granted is True
    assert pacer.note_error(code, "pacing violation", capture=True, now=quiet) is True

    denied = pacer.try_acquire(key="b", now=quiet + timedelta(seconds=1))
    assert denied.granted is False and denied.reason == pacer_mod.DENY_PACING_BACKOFF
    assert pacer.try_acquire(key="b", now=quiet + timedelta(seconds=61)).granted is True


def test_repeated_pacing_errors_escalate_and_success_clears_the_cool_off(pacer):
    quiet = T0 + timedelta(seconds=2)
    pacer.note_error(162, capture=True, now=quiet)
    pacer.note_error(162, capture=True, now=quiet)
    # 60s then 120s: still refused a minute later.
    assert pacer.try_acquire(now=quiet + timedelta(seconds=61)).reason == pacer_mod.DENY_PACING_BACKOFF

    pacer.note_capture_success(now=quiet)
    assert pacer.try_acquire(now=quiet + timedelta(seconds=61)).granted is True


def test_a_champion_pacing_error_also_backs_capture_off_but_not_the_champion(pacer):
    quiet = T0 + timedelta(seconds=2)
    assert pacer.note_error(162, "pacing violation", capture=False, now=quiet) is True
    assert pacer.try_acquire(now=quiet).reason == pacer_mod.DENY_PACING_BACKOFF
    # The champion keeps going regardless: nothing here can stop it.
    pacer.note_champion_request("daily_bars", "AAPL", now=quiet)
    assert pacer.snapshot(now=quiet).champion_requests_in_window == 1


def test_non_pacing_errors_do_not_stop_capture(pacer):
    quiet = T0 + timedelta(seconds=2)
    assert pacer.note_error(200, "No security definition found", capture=True, now=quiet) is False
    assert pacer.try_acquire(now=quiet).granted is True
    assert pacer.snapshot(now=quiet).capture_errors == 1


def test_capture_errors_never_touch_the_champion_circuit_breaker():
    """Risk R1: capture must not push live scans onto the Yahoo-only path."""
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from master_avwap_lib import legacy

    before_count = legacy._IBKR_HISTORICAL_FAILURE_COUNT
    before_yahoo_only = legacy._IBKR_HISTORICAL_YAHOO_ONLY

    arbiter = IbPacer(clock=lambda: T0)
    for _ in range(legacy.IBKR_HISTORICAL_FAILURE_THRESHOLD * 3):
        arbiter.note_error(162, "pacing violation", capture=True, now=T0)

    # The champion's counter is a different object in a different module and
    # capture has no path to it - by construction, not by convention.
    assert legacy._IBKR_HISTORICAL_FAILURE_COUNT == before_count
    assert legacy._IBKR_HISTORICAL_YAHOO_ONLY == before_yahoo_only
    assert arbiter.snapshot(now=T0).capture_errors == legacy.IBKR_HISTORICAL_FAILURE_THRESHOLD * 3

    pacer_source = open(pacer_mod.__file__, encoding="utf-8").read()
    assert "_IBKR_HISTORICAL_FAILURE_COUNT" not in pacer_source.split('"""', 2)[-1]


# --- identical-request cooldown --------------------------------------------
def test_identical_requests_are_refused_inside_the_cooldown(pacer):
    quiet = T0 + timedelta(seconds=2)
    key = "AAPL|M5|2026-08-03|0"
    assert pacer.try_acquire(key=key, now=quiet).granted is True
    repeat = pacer.try_acquire(key=key, now=quiet + timedelta(seconds=5))
    assert repeat.granted is False and repeat.reason == pacer_mod.DENY_IDENTICAL_COOLDOWN
    assert pacer.try_acquire(key=key, now=quiet + timedelta(seconds=16)).granted is True
    # A different window is a different request.
    assert pacer.try_acquire(key="AAPL|M5|2026-08-04|0", now=quiet + timedelta(seconds=5)).granted is True


# --- client-ID allocation (risk R2) ----------------------------------------
def test_client_id_allocation_is_asserted_at_connect():
    assert pacer_mod.assert_client_id(1010, pacer_mod.ROLE_CAPTURE_STREAM) == 1010
    assert pacer_mod.assert_client_id(1011, pacer_mod.ROLE_NIGHTLY_BACKFILL) == 1011

    with pytest.raises(pacer_mod.ClientIdError, match="retired"):
        pacer_mod.assert_client_id(1003, pacer_mod.ROLE_NIGHTLY_BACKFILL)
    with pytest.raises(pacer_mod.ClientIdError, match="must connect with client id 1011"):
        pacer_mod.assert_client_id(1010, pacer_mod.ROLE_NIGHTLY_BACKFILL)
    with pytest.raises(pacer_mod.ClientIdError, match="unknown capture role"):
        pacer_mod.assert_client_id(1010, "something_else")


def test_the_mini_pc_is_excluded_from_phases_0_to_8():
    with pytest.raises(pacer_mod.ClientIdError, match="excluded from warehouse Phases 0-8"):
        pacer_mod.assert_client_id(1020, pacer_mod.ROLE_MINI_PC_BUNDLE)
    with pytest.raises(pacer_mod.ClientIdError, match="mini-PC bundle ids"):
        pacer_mod.assert_client_id(1099, pacer_mod.ROLE_MINI_PC_BUNDLE)


# --- process-wide singleton ------------------------------------------------
def test_there_is_exactly_one_arbiter_per_process():
    first = pacer_mod.get_pacer()
    assert pacer_mod.get_pacer() is first
    fresh = pacer_mod.reset_pacer()
    assert fresh is not first and pacer_mod.get_pacer() is fresh
    pacer_mod.reset_pacer()


def test_blocking_acquire_waits_without_a_real_sleep(pacer):
    slept = []
    quiet = T0 + timedelta(seconds=2)
    for index in range(pacer_mod.CAPTURE_WINDOW_ALLOWANCE):
        pacer.try_acquire(key=f"k{index}", now=quiet)
    pacer._clock = lambda: quiet
    decision = pacer.acquire(key="late", timeout=0.05, sleep=slept.append)
    assert decision.granted is False and slept  # it waited, then gave up cleanly


def test_snapshot_reports_denial_reasons_for_health(pacer):
    with pacer.champion_window("daily_bars", "AAPL"):
        pacer.try_acquire(now=T0)
    snapshot = pacer.snapshot(now=T0)
    assert snapshot.denials.get(pacer_mod.DENY_CHAMPION_ACTIVE) == 1
    assert snapshot.champion_requests_in_window == 1
    assert snapshot.capture_requests_in_window == 0
