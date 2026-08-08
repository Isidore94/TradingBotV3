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

    summary = monitor.sweep_incomplete_followups(
        lambda symbol, session_date: [],
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )

    assert summary["data_gap_symbols"] == ["MU"]
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

    summary = monitor.sweep_incomplete_followups(_boom, now=AFTER_CLOSE, trigger="close_of_day")

    assert summary["ran"] is True
    assert summary["data_gap_symbols"] == ["MU"]


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
        lambda symbol, session_date: [],
        now=AFTER_CLOSE,
        trigger="close_of_day",
    )
    # Chains that could not complete must not re-spend IB requests every loop.
    assert monitor.followup_sweep_trigger(now=AFTER_CLOSE + timedelta(minutes=30)) == ""


def test_sweep_marker_survives_a_restart(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    monitor, paths = _monitor_with_pending_chain(tmp_path)
    monitor.sweep_incomplete_followups(
        lambda symbol, session_date: [],
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
