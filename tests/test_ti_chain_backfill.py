"""Technical Integrity follow-up chain sweeper (durability packet sec 2.3).

Incomplete +30/60/90 chains (42 on 2026-08-04, 1106 on 08-06, 691 on 08-07)
are a pure function of completed M5 bars, which is exactly what Tier B permits
recomputing. The sweeper finishes them at the close, or on the next startup
when the close was missed, and marks every recomputed row
``capture_mode: "backfill"`` so research can separate reconstructed evidence
from what the live process actually observed -- forever.

The boundary matters as much as the recovery: bars that genuinely cannot be
fetched still produce explicit ``data_gap`` rows, and Tier C artifacts (frozen
snapshots, opening-range baselines) are not touched here at all.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

NY = ZoneInfo("America/New_York")
SESSION = "2026-07-15"
RESOLUTION_AT = datetime(2026, 7, 15, 10, 0, tzinfo=NY)
AFTER_CLOSE = datetime(2026, 7, 15, 16, 5, tzinfo=NY)
NEXT_MORNING = datetime(2026, 7, 16, 7, 30, tzinfo=NY)


def _resolution(event_id="resolved-1", *, resolved_at=RESOLUTION_AT):
    return {
        "event_type": "level_resolved",
        "event_id": event_id,
        "session_date": SESSION,
        "resolved_at": resolved_at.isoformat(timespec="seconds"),
        "symbol": "MU",
        "level_family": "vwap",
        "level_timeframe": "intraday",
        "level_value": 100.0,
        "atr": 2.0,
        "outcome": "broke",
        "break_direction": "down",
        "approach_side": "above",
        "event_weight": 1.2,
    }


def _followup_bars(resolution_at, count, *, step=-0.5):
    rows = []
    prior = 100.0
    for index in range(count):
        close = prior + step
        rows.append(
            {
                "datetime": resolution_at + timedelta(minutes=5 * index),
                "open": prior,
                "high": max(prior, close) + 0.2,
                "low": min(prior, close) - 0.2,
                "close": close,
                "volume": 1000,
            }
        )
        prior = close
    return rows


def _monitor_with_pending_chain(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    paths = {
        "events_path": tmp_path / "events.jsonl",
        "state_path": tmp_path / "state.json",
        "snapshot_path": tmp_path / "snapshot.json",
    }
    monitor = TechnicalIntegrityMonitor(**paths)
    monitor._ensure_session(SESSION)
    assert monitor._start_followup(_resolution())
    monitor._save_state()
    return monitor, paths


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_live_followup_rows_are_marked_live_and_absence_still_means_live():
    from technical_integrity import (
        CAPTURE_MODE_LIVE,
        _followup_tracking_event,
        _post_resolution_events,
        completed_m5_bars,
        row_capture_mode,
    )

    tracking = _followup_tracking_event(_resolution())
    bars = completed_m5_bars(
        _followup_bars(RESOLUTION_AT, 8),
        now=RESOLUTION_AT + timedelta(minutes=31),
    )
    events = _post_resolution_events(tracking, bars, now=RESOLUTION_AT + timedelta(minutes=31))
    assert events and all(row["capture_mode"] == CAPTURE_MODE_LIVE for row in events)
    # capture_mode is additive: every row written before this change has none,
    # and reading its absence as anything but "live" would rewrite history.
    assert row_capture_mode({"event_type": "post_resolution_followup"}) == CAPTURE_MODE_LIVE


def test_completed_m5_bars_can_select_an_earlier_session():
    from technical_integrity import completed_m5_bars

    yesterday = _followup_bars(RESOLUTION_AT, 4)
    today = _followup_bars(RESOLUTION_AT + timedelta(days=1), 4)
    rows = yesterday + today

    # Default: the newest session only, which is what live capture wants.
    latest = completed_m5_bars(rows, now=NEXT_MORNING + timedelta(hours=6))
    assert {bar["_start_local"].date() for bar in latest} == {RESOLUTION_AT.date() + timedelta(days=1)}

    # The sweeper asks for the session it is recovering.
    earlier = completed_m5_bars(
        rows,
        now=NEXT_MORNING + timedelta(hours=6),
        session_date=RESOLUTION_AT.date(),
    )
    assert {bar["_start_local"].date() for bar in earlier} == {RESOLUTION_AT.date()}


def test_sweeper_completes_the_chain_and_marks_every_row_backfilled(tmp_path):
    from technical_integrity import CAPTURE_MODE_BACKFILL

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    fetched: list[tuple[str, str]] = []

    def _fetch(symbol, session_date):
        fetched.append((symbol, session_date))
        return _followup_bars(RESOLUTION_AT, 20)

    summary = monitor.sweep_incomplete_followups(_fetch, now=AFTER_CLOSE, trigger="close_of_day")

    assert summary["ran"] is True
    assert summary["events"] == 3  # +30, +60, +90
    assert fetched == [("MU", SESSION)]
    assert monitor.pending_followups == {}

    followups = [row for row in _rows(paths["events_path"]) if row["event_type"] == "post_resolution_followup"]
    assert sorted(row["horizon_minutes"] for row in followups) == [30, 60, 90]
    assert all(row["capture_mode"] == CAPTURE_MODE_BACKFILL for row in followups)
    assert all(row["data_gap"] is False for row in followups)


def test_sweeper_never_rewrites_a_live_row(tmp_path):
    from technical_integrity import CAPTURE_MODE_BACKFILL, CAPTURE_MODE_LIVE

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    # +30 resolves live during the session; the outage starts after that.
    assert monitor.observe_followups(
        "MU",
        _followup_bars(RESOLUTION_AT, 8),
        now=RESOLUTION_AT + timedelta(minutes=31),
    ) == 1

    monitor.sweep_incomplete_followups(
        lambda symbol, session_date: _followup_bars(RESOLUTION_AT, 20),
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    followups = [row for row in _rows(paths["events_path"]) if row["event_type"] == "post_resolution_followup"]
    by_horizon = {row["horizon_minutes"]: row for row in followups}
    assert len(followups) == 3, "backfill must append, never duplicate or rewrite"
    assert by_horizon[30]["capture_mode"] == CAPTURE_MODE_LIVE
    assert by_horizon[60]["capture_mode"] == CAPTURE_MODE_BACKFILL
    assert by_horizon[90]["capture_mode"] == CAPTURE_MODE_BACKFILL


def test_unfetchable_bars_stay_an_explicit_data_gap(tmp_path):
    from technical_integrity import CAPTURE_MODE_BACKFILL

    monitor, paths = _monitor_with_pending_chain(tmp_path)

    # The gap is written only once the symbol's whole entitlement is spent.
    # Sweeping repeatedly is what an unreachable provider actually looks like.
    summaries = [
        monitor.sweep_incomplete_followups(
            lambda symbol, session_date: [],
            now=AFTER_CLOSE,
            trigger="close_of_day",
            sleep=lambda seconds: None,
        )
        for _ in range(6)
    ]

    gapped = next(row for row in summaries if row["data_gap_symbols"])
    assert gapped["data_gap_symbols"] == ["MU"]
    followups = [row for row in _rows(paths["events_path"]) if row["event_type"] == "post_resolution_followup"]
    assert len(followups) == 3
    assert all(row["data_gap"] is True for row in followups)
    assert all(row["capture_mode"] == CAPTURE_MODE_BACKFILL for row in followups)
    # Missing data is uncertainty, never confirmation: no metric is invented.
    assert all(row["displacement_atr_30"] is None for row in followups if row["horizon_minutes"] == 30)


def test_a_broker_error_is_a_data_gap_not_a_crash(tmp_path):
    monitor, paths = _monitor_with_pending_chain(tmp_path)

    def _boom(symbol, session_date):
        raise RuntimeError("IB pacing violation")

    summaries = [
        monitor.sweep_incomplete_followups(
            _boom, now=AFTER_CLOSE, trigger="close_of_day", sleep=lambda seconds: None
        )
        for _ in range(6)
    ]

    gapped = next(row for row in summaries if row["data_gap_symbols"])
    assert gapped["ran"] is True
    assert gapped["data_gap_symbols"] == ["MU"]


def test_sweeper_is_disabled_by_setting(tmp_path, monkeypatch):
    import technical_integrity

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    monkeypatch.setattr(technical_integrity, "get_local_setting", lambda key, default=None: False)

    summary = monitor.sweep_incomplete_followups(
        lambda symbol, session_date: _followup_bars(RESOLUTION_AT, 20),
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    assert summary["ran"] is False
    assert "disabled" in summary["reason"]
    assert monitor.pending_followups  # untouched


def test_sweep_is_due_at_the_close_and_only_once_per_session(tmp_path):
    monitor, _paths = _monitor_with_pending_chain(tmp_path)

    assert monitor.followup_sweep_trigger(now=RESOLUTION_AT + timedelta(minutes=31)) == ""
    assert monitor.followup_sweep_trigger(now=AFTER_CLOSE) == "close_of_day"

    monitor.sweep_incomplete_followups(
        lambda symbol, session_date: _followup_bars(RESOLUTION_AT, 20),
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )
    # A finished sweep must not re-spend IB requests every loop.
    assert monitor.followup_sweep_trigger(now=AFTER_CLOSE + timedelta(minutes=30)) == ""


def test_sweep_marker_survives_a_restart(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    monitor.sweep_incomplete_followups(
        lambda symbol, session_date: _followup_bars(RESOLUTION_AT, 20),
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    restarted = TechnicalIntegrityMonitor(**paths)
    assert restarted.followup_sweep_trigger(now=AFTER_CLOSE + timedelta(minutes=5)) == ""


def test_sweep_is_due_on_the_next_startup_when_the_close_was_missed(tmp_path):
    monitor, _paths = _monitor_with_pending_chain(tmp_path)

    assert monitor.followup_sweep_trigger(now=NEXT_MORNING) == "startup_after_missed_close"


def test_no_pending_chains_means_no_sweep(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    monitor = TechnicalIntegrityMonitor(
        events_path=tmp_path / "events.jsonl",
        state_path=tmp_path / "state.json",
        snapshot_path=tmp_path / "snapshot.json",
    )
    monitor._ensure_session(SESSION)

    assert monitor.followup_sweep_trigger(now=AFTER_CLOSE) == ""
    summary = monitor.sweep_incomplete_followups(lambda s, d: [], now=AFTER_CLOSE)
    assert summary["ran"] is False


def test_evidence_clock_adapter_widens_the_window_for_an_earlier_session():
    from bounce_bot_lib import legacy

    captured: dict[str, object] = {}

    def _sweep(fetch, **kwargs):
        captured["fetch"] = fetch
        captured.update(kwargs)
        return {"ran": True, "events": 0, "symbols": ["MU"], "data_gap_symbols": [], "session_date": SESSION}

    monitor = Mock()
    monitor.followup_sweep_trigger.return_value = "startup_after_missed_close"
    monitor.session_date = SESSION
    monitor.sweep_incomplete_followups = _sweep

    bot = object.__new__(legacy.BounceBot)
    bot.connection_status = True
    bot.request_historical_bars = Mock(return_value=_followup_bars(RESOLUTION_AT, 20))

    legacy.BounceBot._sweep_technical_followup_chains(bot, monitor, NEXT_MORNING)

    # Yesterday's session is not inside a "1 D" request that ends right now.
    captured["fetch"]("MU", SESSION)
    bot.request_historical_bars.assert_called_once_with("MU", "2 D", "5 mins", timeout=12.0)

    bot.request_historical_bars.reset_mock()
    captured["fetch"]("MU", "2026-07-16")
    bot.request_historical_bars.assert_called_once_with("MU", "1 D", "5 mins", timeout=12.0)


def test_sweep_waits_for_ib_instead_of_gapping_every_chain():
    from bounce_bot_lib import legacy

    monitor = Mock()
    monitor.followup_sweep_trigger.return_value = "startup_after_missed_close"

    bot = object.__new__(legacy.BounceBot)
    bot.connection_status = False
    bot.request_historical_bars = Mock()

    legacy.BounceBot._sweep_technical_followup_chains(bot, monitor, NEXT_MORNING)

    # Finalizing recoverable chains as data gaps because IB has not connected
    # yet would destroy evidence the sweeper exists to save.
    monitor.sweep_incomplete_followups.assert_not_called()
    bot.request_historical_bars.assert_not_called()


def test_pending_level_test_resolves_identically_across_a_mid_session_restart(tmp_path):
    """Characterization (sec 2.3): a restart between touch and resolution must
    produce the same resolution the uninterrupted run produces."""
    from technical_integrity import TechnicalIntegrityMonitor

    start = datetime(2026, 7, 15, 9, 30, tzinfo=NY)
    touch = [
        {"datetime": start, "open": 101.2, "high": 101.4, "low": 100.8, "close": 101.0, "volume": 1000},
        {"datetime": start + timedelta(minutes=5), "open": 101.0, "high": 101.1, "low": 99.98, "close": 100.2, "volume": 1400},
    ]
    resolved = touch + [
        {
            "datetime": start + timedelta(minutes=10 + 5 * index),
            "open": 99.7,
            "high": 99.8,
            "low": 99.3,
            "close": 99.7,
            "volume": 1500,
        }
        for index in range(3)
    ]
    observe = {
        "metrics": {"std_vwap": 100.0},
        "atr": 1.0,
        "classification": {
            "sectorKey": "technology",
            "sector": "Technology",
            "industryKey": "memory",
            "industry": "Memory",
        },
        "market_environment": "bearish_strong",
    }

    def _paths(name):
        root = tmp_path / name
        root.mkdir()
        return {
            "events_path": root / "events.jsonl",
            "state_path": root / "state.json",
            "snapshot_path": root / "snapshot.json",
        }

    uninterrupted_paths = _paths("uninterrupted")
    uninterrupted = TechnicalIntegrityMonitor(**uninterrupted_paths)
    uninterrupted.observe_symbol("MU", touch, now=start + timedelta(minutes=11), **observe)
    uninterrupted.observe_symbol("MU", resolved, now=start + timedelta(minutes=26), **observe)

    restarted_paths = _paths("restarted")
    first = TechnicalIntegrityMonitor(**restarted_paths)
    first.observe_symbol("MU", touch, now=start + timedelta(minutes=11), **observe)
    del first
    second = TechnicalIntegrityMonitor(**restarted_paths)
    second.observe_symbol("MU", resolved, now=start + timedelta(minutes=26), **observe)

    def _comparable(path):
        rows = []
        for row in _rows(path):
            row.pop("written_at", None)  # wall clock of the append, not evidence
            rows.append(row)
        return rows

    assert _comparable(restarted_paths["events_path"]) == _comparable(
        uninterrupted_paths["events_path"]
    )


def test_audit_reports_backfilled_and_live_chains_separately(tmp_path):
    from regime_collection_audit import audit_regime_collection

    day = "2026-07-30"

    def _base(event_type, event_id):
        return {
            "code_version": "regime_infrastructure_phase1_v1",
            "event_type": event_type,
            "event_id": event_id,
            "session_date": day,
            "as_of": f"{day}T10:30:00-04:00",
            "written_at": f"{day}T10:30:01-04:00",
        }

    rows = [
        {**_base("level_resolved", "r1"), "followup_tracking_version": "regime_infrastructure_phase1_v1"},
        {**_base("post_resolution_tracking_started", "r1|followup"), "source_resolution_id": "r1"},
    ]
    for horizon, capture_mode in ((30, "live"), (60, "backfill"), (90, "backfill")):
        row = {
            **_base("post_resolution_followup", f"r1|followup|{horizon}"),
            "source_resolution_id": "r1",
            "horizon_minutes": horizon,
            "truncated": False,
            "data_gap": False,
        }
        if capture_mode != "live":
            row["capture_mode"] = capture_mode
        rows.append(row)

    technical = tmp_path / "technical.jsonl"
    technical.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "vold.jsonl").write_text("", encoding="utf-8")

    report = audit_regime_collection(
        session_date=day,
        technical_events_path=technical,
        breadth_events_path=tmp_path / "vold.jsonl",
        now=datetime(2026, 7, 30, 16, 30, tzinfo=NY),
    )

    followups = report["technical_followups"]
    assert followups["backfilled_count"] == 2
    # The +30 row predates the schema field, so its absence reads as live.
    assert followups["live_count"] == 1
    assert "backfilled=2" in __import__("regime_collection_audit").format_audit(report)


# --- bounded retry before the gap becomes permanent ------------------------
#
# The sweep writes its data_gap rows *and* its per-session marker in the same
# pass, and the marker is what stops the session from ever being swept again.
# Finalising both off one failed request turned a transient broker hiccup into
# permanently missing evidence (checkpoint review 2026-08-08 second review).


class _Sleeps:
    def __init__(self):
        self.pauses: list[float] = []

    def __call__(self, seconds):
        self.pauses.append(seconds)


def test_a_transient_fetch_failure_is_retried_and_the_chain_completes(tmp_path):
    from technical_integrity import CAPTURE_MODE_BACKFILL

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    attempts = {"n": 0}
    sleeps = _Sleeps()

    def _flaky(symbol, session_date):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("IBKR pacing violation")
        return _followup_bars(RESOLUTION_AT, 20)

    summary = monitor.sweep_incomplete_followups(
        _flaky, now=AFTER_CLOSE, trigger="close_of_day", sleep=sleeps
    )

    assert attempts["n"] == 2
    assert summary["retry_attempts"] == {"MU": 2}
    assert summary["data_gap_symbols"] == []
    followups = [
        row for row in _rows(paths["events_path"])
        if row["event_type"] == "post_resolution_followup"
    ]
    assert len(followups) == 3
    assert all(row["data_gap"] is False for row in followups)
    assert all(row["capture_mode"] == CAPTURE_MODE_BACKFILL for row in followups)


def test_retries_are_bounded_and_exhaustion_still_writes_the_honest_gap(tmp_path):
    """The entitlement is per symbol and spans sweeps.

    A shared wall-clock sleep budget rationed the wrong thing: on a bad night
    the first symbols consumed it and everything after them got one attempt
    and a permanent gap (Sol 5.6 verification review, item 6).
    """
    from technical_integrity import FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    attempts = {"n": 0}
    sleeps = _Sleeps()

    def _broken(symbol, session_date):
        attempts["n"] += 1
        raise RuntimeError("IBKR disconnected")

    summaries = []
    for _ in range(FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT):
        summaries.append(
            monitor.sweep_incomplete_followups(
                _broken, now=AFTER_CLOSE, trigger="close_of_day", sleep=sleeps
            )
        )

    # Bounded: the entitlement is spent exactly once, never in an open loop.
    assert attempts["n"] == FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT
    # Deferred while attempts remained; gapped only at the end.
    assert summaries[0]["deferred_symbols"] == ["MU"]
    assert summaries[0]["data_gap_symbols"] == []
    assert summaries[0]["marker_written"] is False
    gapped = next(row for row in summaries if row["data_gap_symbols"])
    assert gapped["data_gap_symbols"] == ["MU"]
    assert gapped["deferred_symbols"] == []

    followups = [
        row for row in _rows(paths["events_path"])
        if row["event_type"] == "post_resolution_followup"
    ]
    assert len(followups) == 3, "the gap is written once, not once per sweep"
    assert all(row["data_gap"] is True for row in followups)
    # The gap is still explicit, and now says how hard the sweeps tried.
    assert all(
        f"{FOLLOWUP_SYMBOL_ATTEMPT_ENTITLEMENT} attempt(s) across sweeps"
        in row["data_gap_reason"]
        for row in followups
    )
    # Only now is the session marked, because only now is there nothing to
    # come back for.
    assert f"{SESSION}|close_of_day" in monitor.followup_sweep_markers


def test_a_persistently_empty_response_is_retried_too(tmp_path):
    # "Nothing" from a provider still catching up is indistinguishable from
    # "this data does not exist" on a single attempt.
    monitor, _paths = _monitor_with_pending_chain(tmp_path)
    attempts = {"n": 0}

    def _empty(symbol, session_date):
        attempts["n"] += 1
        return []

    summary = monitor.sweep_incomplete_followups(
        _empty, now=AFTER_CLOSE, trigger="close_of_day", sleep=_Sleeps()
    )

    assert attempts["n"] == 2, "one retry inside a sweep; the rest is entitlement"
    assert summary["deferred_symbols"] == ["MU"]
    assert summary["data_gap_symbols"] == []


def test_one_sweep_holds_the_lock_for_at_most_one_backoff_per_symbol(tmp_path):
    """What the shared sleep budget was really protecting.

    That budget rationed retries across symbols, which is the wrong thing to
    ration. Capping the *per-sweep* attempts achieves the same bounded lock
    hold without taking a later symbol's entitlement away.
    """
    monitor, _paths = _monitor_with_pending_chain(tmp_path)
    sleeps = _Sleeps()

    summary = monitor.sweep_incomplete_followups(
        lambda symbol, session_date: [],
        now=AFTER_CLOSE,
        trigger="close_of_day",
        sleep=sleeps,
    )

    assert len(sleeps.pauses) == 1, "one backoff per symbol per sweep, no more"
    assert sum(sleeps.pauses) <= 1.0
    assert summary["deferred_symbols"] == ["MU"]


# --- point-in-time as_of on an empty follow-up window ----------------------


def test_an_absent_window_is_stamped_at_the_horizon_not_the_resolution():
    from technical_integrity import _followup_tracking_event, _post_resolution_events

    tracking = _followup_tracking_event(_resolution())
    events = _post_resolution_events(
        tracking, [], now=RESOLUTION_AT + timedelta(minutes=95)
    )

    by_horizon = {row["horizon_minutes"]: row for row in events}
    assert set(by_horizon) == {30, 60, 90}
    for horizon, row in by_horizon.items():
        assert row["data_gap"] is True
        # The absence was only knowable once the window ran out. Stamping it at
        # resolution time backdated the finding by up to 90 minutes.
        assert row["as_of"] == row["window_target_at"], horizon
        assert row["as_of"] > row["resolution_bar_close"]


def test_a_truncated_horizon_is_stamped_at_the_close_it_actually_ended_at():
    from technical_integrity import _followup_tracking_event, _post_resolution_events

    # Resolve 45 minutes before the close: +60 and +90 run past it.
    late = datetime(2026, 7, 15, 15, 15, tzinfo=NY)
    tracking = _followup_tracking_event(_resolution(resolved_at=late))
    events = _post_resolution_events(tracking, [], now=late + timedelta(minutes=95))

    by_horizon = {row["horizon_minutes"]: row for row in events}
    assert by_horizon[30]["truncated"] is False
    assert by_horizon[30]["as_of"] == by_horizon[30]["window_target_at"]
    for horizon in (60, 90):
        row = by_horizon[horizon]
        assert row["truncated"] is True
        # The window ended at the close, so that is when the absence was known;
        # the target time is after the close and nobody could observe it there.
        # (Stamps are market-local, i.e. this desk's clock, so compare instants.)
        assert datetime.fromisoformat(row["as_of"]) == datetime(2026, 7, 15, 16, 0, tzinfo=NY)
        assert datetime.fromisoformat(row["as_of"]) < datetime.fromisoformat(
            row["window_target_at"]
        )


# --- completeness, entitlement, deferral (Sol 5.6 review, item 6) ----------


def test_a_partial_response_is_retried_like_a_failure(tmp_path):
    """A provider under load returns half a window.

    Truthy, so it used to sail through as success -- and the short window was
    recorded as though the sweep had asked and been told that was all there
    was. Indistinguishable, afterwards, from a real one.
    """
    monitor, paths = _monitor_with_pending_chain(tmp_path)
    calls = []

    def _partial_then_full(symbol, session_date):
        calls.append(1)
        # Two bars is nowhere near the +90 window this chain still owes.
        return _followup_bars(RESOLUTION_AT, 2 if len(calls) == 1 else 20)

    summary = monitor.sweep_incomplete_followups(
        _partial_then_full,
        now=AFTER_CLOSE,
        trigger="close_of_day",
        sleep=lambda seconds: None,
    )

    assert len(calls) == 2, "the partial response must be retried, not accepted"
    assert summary["data_gap_symbols"] == []
    assert summary["deferred_symbols"] == []
    followups = [
        row for row in _rows(paths["events_path"])
        if row["event_type"] == "post_resolution_followup"
    ]
    assert len(followups) == 3
    assert all(row["data_gap"] is False for row in followups)


def test_a_complete_response_is_accepted_first_time(tmp_path):
    # The completeness check must not make every fetch look incomplete and
    # burn the entitlement on nothing.
    monitor, _paths = _monitor_with_pending_chain(tmp_path)
    calls = []

    def _full(symbol, session_date):
        calls.append(1)
        return _followup_bars(RESOLUTION_AT, 20)

    summary = monitor.sweep_incomplete_followups(
        _full, now=AFTER_CLOSE, trigger="close_of_day"
    )

    assert len(calls) == 1
    assert summary["marker_written"] is True


def test_expected_bar_count_is_a_lower_bound_on_the_matured_window(tmp_path):
    monitor, _paths = _monitor_with_pending_chain(tmp_path)

    # Nothing has matured 10 minutes after resolution, so nothing is demanded.
    assert monitor._expected_followup_bars("MU", now=RESOLUTION_AT + timedelta(minutes=10)) == 0
    # By +31 the 30-minute window is owed: six 5-minute bars.
    assert monitor._expected_followup_bars("MU", now=RESOLUTION_AT + timedelta(minutes=31)) == 6
    # After the close every horizon is owed; +90 is eighteen bars.
    assert monitor._expected_followup_bars("MU", now=AFTER_CLOSE) == 18
    # A symbol with no pending chains demands nothing.
    assert monitor._expected_followup_bars("ZZZ", now=AFTER_CLOSE) == 0


def test_a_deferred_symbol_blocks_the_sweep_marker(tmp_path):
    """The marker is permanent; writing it while work remains is the defect."""
    monitor, _paths = _monitor_with_pending_chain(tmp_path)

    summary = monitor.sweep_incomplete_followups(
        lambda symbol, session_date: [],
        now=AFTER_CLOSE,
        trigger="close_of_day",
        sleep=lambda seconds: None,
    )

    assert summary["deferred_symbols"] == ["MU"]
    assert summary["marker_written"] is False
    assert f"{SESSION}|close_of_day" not in monitor.followup_sweep_markers
    # ...so the sweep is still due, and the chain is still pending.
    assert monitor.followup_sweep_trigger(now=AFTER_CLOSE + timedelta(minutes=30)) == "close_of_day"
    assert monitor.pending_followups


def test_a_deferred_symbol_recovers_on_a_later_sweep(tmp_path):
    # The whole point of deferring: the outage ends and the evidence arrives,
    # instead of a permanent gap written during the outage.
    monitor, paths = _monitor_with_pending_chain(tmp_path)

    first = monitor.sweep_incomplete_followups(
        lambda symbol, session_date: [],
        now=AFTER_CLOSE,
        trigger="close_of_day",
        sleep=lambda seconds: None,
    )
    assert first["deferred_symbols"] == ["MU"]

    second = monitor.sweep_incomplete_followups(
        lambda symbol, session_date: _followup_bars(RESOLUTION_AT, 20),
        now=AFTER_CLOSE + timedelta(minutes=30),
        trigger="close_of_day",
    )

    assert second["data_gap_symbols"] == []
    assert second["deferred_symbols"] == []
    assert second["marker_written"] is True
    followups = [
        row for row in _rows(paths["events_path"])
        if row["event_type"] == "post_resolution_followup"
    ]
    assert all(row["data_gap"] is False for row in followups), "no gap was ever written"


def test_the_entitlement_survives_a_restart(tmp_path):
    """A symbol that ran out of luck at the close keeps its remaining attempts."""
    from technical_integrity import TechnicalIntegrityMonitor

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    monitor.sweep_incomplete_followups(
        lambda symbol, session_date: [],
        now=AFTER_CLOSE,
        trigger="close_of_day",
        sleep=lambda seconds: None,
    )
    spent = dict(monitor.followup_symbol_attempts)
    assert spent == {f"{SESSION}|MU": 2}

    restarted = TechnicalIntegrityMonitor(**paths)
    assert restarted.followup_symbol_attempts == spent
    assert f"{SESSION}|close_of_day" not in restarted.followup_sweep_markers


def test_one_symbol_cannot_spend_another_symbols_entitlement(tmp_path):
    """The shared budget's real defect, stated as a property.

    With a wall-clock budget shared across symbols, the first few failures
    consumed it and every symbol after them got one attempt and a permanent
    gap. Entitlement is per symbol now, so a hopeless symbol costs its
    neighbours nothing.
    """

    monitor, _paths = _monitor_with_pending_chain(tmp_path)
    second = _resolution(event_id="resolved-2")
    second["symbol"] = "NVDA"
    assert monitor._start_followup(second)

    calls = []

    def _only_mu_is_broken(symbol, session_date):
        calls.append(symbol)
        return [] if symbol == "MU" else _followup_bars(RESOLUTION_AT, 20)

    summary = monitor.sweep_incomplete_followups(
        _only_mu_is_broken,
        now=AFTER_CLOSE,
        trigger="close_of_day",
        sleep=lambda seconds: None,
    )

    assert summary["deferred_symbols"] == ["MU"]
    assert calls.count("NVDA") == 1, "the healthy symbol was not made to retry"
    assert monitor.followup_symbol_attempts[f"{SESSION}|NVDA"] == 1
    assert monitor.followup_symbol_attempts[f"{SESSION}|MU"] == 2
