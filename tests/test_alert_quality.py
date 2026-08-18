"""Phase 0 alert-quality audit: honest Unknowns, sample floors, episode identity.

The property under test throughout is the one the module exists to protect: a
metric that cannot be computed must read ``Unknown``, never ``0``. A zero is a
measurement ("the desk was quiet"); an Unknown is the absence of one ("nothing
was recorded"). Confusing the two would make the scoreboard actively
misleading, so every blocked path is asserted rather than assumed.
"""

import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import alert_quality
from alert_quality import (
    DELIVERY_ACTIONS,
    METRIC_REGISTRY,
    MIN_SAMPLES,
    STATUS_BLOCKED,
    STATUS_DEFERRED,
    audit_capture,
    build_report,
    LATENCY_BOUND_MS,
    compute_alert_to_action,
    compute_armed_hit_delivery,
    compute_duplicate_loud_rate,
    compute_loud_per_session,
    compute_metrics,
    compute_watch_conversion,
    filter_recent,
    run_audit,
)


def _row(action, symbol="AAPL", *, side="LONG", trade_date="2026-08-10", **extra):
    row = {
        "schema": "review_events_v2",
        "action": action,
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        "installation_id": "a" * 32,
        "machine": "DESK",
    }
    row.update(extra)
    return row


def _delivery(
    symbol="AAPL",
    *,
    side="LONG",
    trade_date="2026-08-10",
    loud=True,
    sounded=True,
    tier="",
    is_armed_fire=False,
    event_id=None,
    ts="2026-08-10T09:30:00",
    action="delivered",
    **extra,
):
    row = {
        "schema": "alert_delivery_events_v1",
        "action": action,
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        "ts": ts,
        "loud": loud,
        "sounded": sounded,
        "tier": tier,
        "is_armed_fire": is_armed_fire,
        "alert_event_id": event_id or f"{trade_date}|{symbol}|{side}|m5_bounce|reclaim",
    }
    row.update(extra)
    return row


def _impressions(count, *, trade_date="2026-08-10"):
    return [
        _row("shown", symbol=f"SYM{index}", trade_date=trade_date)
        for index in range(count)
    ]


# --- capture coverage -------------------------------------------------------


def test_empty_store_reports_nothing_rather_than_zero():
    coverage = audit_capture([])
    assert coverage.rows == 0
    assert coverage.sessions == 0
    assert coverage.has_delivery_capture is False

    results = compute_metrics([])
    assert all(result.value is None for result in results)
    assert all(result.display_value() == "Unknown" for result in results)


def test_empty_report_distinguishes_empty_store_from_quiet_desk():
    _, _, report = run_audit(rows=[], delivery_rows=[])
    assert "empty store" in report
    assert "NOT a quiet desk" in report


def test_audit_summarises_span_writers_and_actions():
    rows = [
        _row("shown", trade_date="2026-08-10"),
        _row("skip", trade_date="2026-08-11"),
        _row("shown", trade_date="2026-08-12", installation_id="b" * 32, machine="MINI"),
    ]
    coverage = audit_capture(rows)
    assert coverage.rows == 3
    assert coverage.sessions == 3
    assert coverage.first_trade_date == "2026-08-10"
    assert coverage.last_trade_date == "2026-08-12"
    assert coverage.impressions == 2
    assert coverage.action_counts == {"shown": 2, "skip": 1}
    assert coverage.installations == 2
    assert coverage.machines == ("DESK", "MINI")


def test_multiple_installations_raise_a_visible_warning():
    rows = [
        _row("shown"),
        _row("shown", symbol="MSFT", installation_id="b" * 32, machine="MINI"),
    ]
    coverage = audit_capture(rows)
    _, _, report = run_audit(rows=rows, delivery_rows=[])
    assert coverage.installations == 2
    assert "more than one installation" in report


# --- the delivery gap -------------------------------------------------------


def test_delivery_capture_absent_is_detected_and_named():
    rows = [_row("shown"), _row("skip"), _row("arm_watch")]
    coverage = audit_capture(rows)
    assert coverage.has_delivery_capture is False

    _, _, report = run_audit(rows=rows, delivery_rows=[])
    assert "ABSENT" in report
    assert "unmeasured" in report


def test_delivery_capture_present_once_phase_1_rows_exist():
    """Deliveries live in their own machine-local store, not the review log."""

    deliveries = [_delivery(), _delivery()]
    coverage = audit_capture([_row("shown")], deliveries)
    assert coverage.has_delivery_capture is True
    assert coverage.delivery_rows == 2

    _, _, report = run_audit(rows=[_row("shown")], delivery_rows=deliveries)
    assert "Phase 1 capture is running" in report


def test_delivery_rows_in_the_review_log_do_not_count_as_capture():
    """Guards the storage-class split: the review store is the wrong place."""

    coverage = audit_capture([_row("delivered"), _row("watch_delivered")], [])
    assert coverage.has_delivery_capture is False


def test_delivery_actions_are_not_confused_with_take_or_pass():
    assert not (DELIVERY_ACTIONS & alert_quality.TAKE_ACTIONS)
    assert not (DELIVERY_ACTIONS & alert_quality.PASS_ACTIONS)


# --- blocked and deferred metrics never invent a number ---------------------


def test_delivery_metrics_are_unknown_when_the_delivery_store_is_empty():
    """The four that Phase 1 unblocks must not read zero without capture."""

    rows = _impressions(50) + [_row("add_focus", symbol=f"SYM{i}") for i in range(50)]
    results = {result.spec.key: result for result in compute_metrics(rows, [])}
    for key in ("loud_per_session", "duplicate_loud_rate", "armed_hit_delivery"):
        assert results[key].value is None, key
        assert results[key].display_value() == "Unknown", key


def test_blocked_and_deferred_metrics_stay_unknown_even_with_rich_data():
    rows = _impressions(50) + [_row("add_focus", symbol=f"SYM{i}") for i in range(50)]
    results = {result.spec.key: result for result in compute_metrics(rows)}
    for spec in METRIC_REGISTRY:
        if spec.status in {STATUS_BLOCKED, STATUS_DEFERRED}:
            assert results[spec.key].value is None, spec.key
            assert results[spec.key].display_value() == "Unknown", spec.key


def test_report_prints_unknown_and_a_reason_for_every_blocked_metric():
    _, _, report = run_audit(rows=_impressions(20), delivery_rows=[])
    for spec in METRIC_REGISTRY:
        if spec.status in {STATUS_BLOCKED, STATUS_DEFERRED}:
            assert spec.title in report
            assert spec.blocker in report


def test_every_metric_declares_a_frozen_outcome_definition_id():
    ids = [spec.outcome_definition_id for spec in METRIC_REGISTRY]
    assert all(ids)
    assert len(set(ids)) == len(ids)
    for spec in METRIC_REGISTRY:
        if spec.status in {STATUS_BLOCKED, STATUS_DEFERRED}:
            assert spec.blocker, spec.key


# --- alert-to-action --------------------------------------------------------


def test_take_rate_withheld_below_the_sample_floor():
    rows = _impressions(MIN_SAMPLES - 1)
    result = compute_alert_to_action(rows)
    assert result.denominator == MIN_SAMPLES - 1
    assert result.value is None
    assert str(MIN_SAMPLES) in result.note


def test_take_rate_reported_once_the_floor_clears():
    rows = _impressions(MIN_SAMPLES)
    rows += [_row("add_focus", symbol="SYM0"), _row("add_focus", symbol="SYM1")]
    result = compute_alert_to_action(rows)
    assert result.denominator == MIN_SAMPLES
    assert result.numerator == 2
    assert result.value == 2 / MIN_SAMPLES


def test_unresolved_impressions_are_not_counted_as_passes():
    rows = _impressions(MIN_SAMPLES)
    rows.append(_row("skip", symbol="SYM0"))
    rows.append(_row("add_focus", symbol="SYM1"))
    result = compute_alert_to_action(rows)
    assert result.breakdown["unresolved"] == MIN_SAMPLES - 2
    assert result.breakdown["skip"] == 1
    assert result.breakdown["add_focus"] == 1
    assert "unresolved" in result.note


def test_arming_a_watch_counts_as_acting_on_the_alert():
    rows = _impressions(MIN_SAMPLES)
    rows.append(_row("arm_watch", symbol="SYM0", detail={"kind": "reclaim"}))
    result = compute_alert_to_action(rows)
    assert result.numerator == 1


def test_long_and_short_on_one_symbol_stay_separate_episodes():
    rows = [
        _row("shown", symbol="AAPL", side="LONG"),
        _row("shown", symbol="AAPL", side="SHORT"),
        _row("add_focus", symbol="AAPL", side="LONG"),
    ]
    result = compute_alert_to_action(rows)
    assert result.denominator == 2
    assert result.numerator == 1


def test_same_symbol_on_two_sessions_is_two_episodes():
    rows = [
        _row("shown", trade_date="2026-08-10"),
        _row("shown", trade_date="2026-08-11"),
    ]
    result = compute_alert_to_action(rows)
    assert result.denominator == 2
    assert result.sessions == 2


# --- watch conversion -------------------------------------------------------


def test_watch_conversion_excludes_watches_that_are_still_armed():
    rows = []
    for index in range(MIN_SAMPLES):
        symbol = f"SYM{index}"
        rows.append(_row("arm_watch", symbol=symbol, detail={"kind": "reclaim"}))
        action = "watch_fired" if index % 2 == 0 else "watch_expired"
        rows.append(_row(action, symbol=symbol, detail={"kind": "reclaim"}))
    rows.append(_row("arm_watch", symbol="PENDING", detail={"kind": "reclaim"}))

    result = compute_watch_conversion(rows)
    assert result.denominator == MIN_SAMPLES
    assert result.numerator == MIN_SAMPLES // 2
    assert result.breakdown["still_armed"] == 1
    assert result.value == (MIN_SAMPLES // 2) / MIN_SAMPLES


def test_watch_conversion_withheld_below_the_floor():
    rows = [
        _row("arm_watch", detail={"kind": "reclaim"}),
        _row("watch_fired", detail={"kind": "reclaim"}),
    ]
    result = compute_watch_conversion(rows)
    assert result.value is None


def test_watches_of_different_kinds_do_not_resolve_each_other():
    rows = [
        _row("arm_watch", detail={"kind": "reclaim"}),
        _row("watch_fired", detail={"kind": "breakout"}),
    ]
    result = compute_watch_conversion(rows)
    assert result.breakdown["armed"] == 1
    assert result.breakdown["fired"] == 0
    assert result.breakdown["still_armed"] == 1


# --- windowing --------------------------------------------------------------


def test_window_keeps_recent_rows_and_drops_older_ones():
    rows = [
        _row("shown", trade_date="2026-08-01"),
        _row("shown", trade_date="2026-08-15"),
    ]
    kept = filter_recent(rows, 7, today=date(2026, 8, 18))
    assert len(kept) == 1
    assert kept[0]["trade_date"] == "2026-08-15"


def test_window_drops_rows_that_cannot_prove_they_belong():
    rows = [
        _row("shown", trade_date="2026-08-15"),
        _row("shown", trade_date="not-a-date"),
        _row("shown", trade_date=""),
    ]
    kept = filter_recent(rows, 7, today=date(2026, 8, 18))
    assert len(kept) == 1


def test_no_window_keeps_undated_rows():
    rows = [_row("shown", trade_date=""), _row("shown", trade_date="2026-08-15")]
    assert len(filter_recent(rows, None)) == 2
    assert len(filter_recent(rows, 0)) == 2


# --- report shape -----------------------------------------------------------


def test_report_states_the_episode_identity_divergence():
    _, _, report = run_audit(rows=_impressions(10), delivery_rows=[])
    assert "EPISODE IDENTITY" in report
    assert "review_learning.py folds" in report
    assert "FIFO" in report


def test_report_is_deterministic_for_a_fixed_clock():
    rows = _impressions(12)
    stamp = datetime(2026, 8, 18, 9, 30, 0)
    first = build_report(audit_capture(rows), compute_metrics(rows), now=stamp)
    second = build_report(audit_capture(rows), compute_metrics(rows), now=stamp)
    assert first == second


def test_run_audit_reads_the_store_when_no_rows_are_supplied(tmp_path):
    store = tmp_path / "alert_review_events.jsonl"
    store.write_text("", encoding="utf-8")
    coverage, results, report = run_audit(path=store, shards_dir=tmp_path / "shards", delivery_dir=tmp_path / "deliveries")
    assert coverage.rows == 0
    assert "CAPTURE AUDIT" in report


# --- duplicate loud rate ----------------------------------------------------


def test_first_loud_delivery_of_a_thesis_is_never_a_duplicate():
    result = compute_duplicate_loud_rate([_delivery()])
    assert result.breakdown["duplicates"] == 0
    assert result.breakdown["loud_deliveries"] == 1


def test_repeat_loud_delivery_without_escalation_is_a_duplicate():
    rows = [_delivery(ts=f"2026-08-10T09:3{i}:00") for i in range(MIN_SAMPLES)]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["duplicates"] == MIN_SAMPLES - 1
    assert result.value == (MIN_SAMPLES - 1) / MIN_SAMPLES


def test_a_rising_tier_is_an_escalation_not_a_duplicate():
    rows = [
        _delivery(tier="C", ts="2026-08-10T09:30:00"),
        _delivery(tier="A", ts="2026-08-10T09:31:00"),
    ]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["escalations"] == 1
    assert result.breakdown["duplicates"] == 0


def test_a_falling_tier_is_still_a_duplicate():
    rows = [
        _delivery(tier="A", ts="2026-08-10T09:30:00"),
        _delivery(tier="C", ts="2026-08-10T09:31:00"),
    ]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["duplicates"] == 1


def test_quiet_alert_becoming_loud_is_an_escalation():
    rows = [
        _delivery(loud=False, ts="2026-08-10T09:30:00"),
        _delivery(loud=True, ts="2026-08-10T09:31:00"),
    ]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["escalations"] == 1
    assert result.breakdown["duplicates"] == 0
    assert result.breakdown["loud_deliveries"] == 1


def test_an_armed_condition_firing_is_always_an_escalation():
    rows = [
        _delivery(ts="2026-08-10T09:30:00"),
        _delivery(is_armed_fire=True, ts="2026-08-10T09:31:00"),
    ]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["escalations"] == 1


def test_different_theses_never_duplicate_each_other():
    rows = [
        _delivery(symbol="AAA", ts="2026-08-10T09:30:00"),
        _delivery(symbol="BBB", ts="2026-08-10T09:31:00"),
    ]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["duplicates"] == 0
    assert result.breakdown["distinct_alerts"] == 2


def test_quiet_repeats_are_not_counted_in_the_loud_denominator():
    rows = [
        _delivery(loud=True, ts="2026-08-10T09:30:00"),
        _delivery(loud=False, ts="2026-08-10T09:31:00"),
    ]
    result = compute_duplicate_loud_rate(rows)
    assert result.breakdown["loud_deliveries"] == 1
    assert result.breakdown["duplicates"] == 0


def test_duplicate_rate_withheld_below_the_floor():
    result = compute_duplicate_loud_rate([_delivery(), _delivery()])
    assert result.value is None


# --- loud per session -------------------------------------------------------


def test_loud_per_session_counts_sessions_not_alerts():
    rows = []
    for day in range(MIN_SAMPLES):
        trade_date = f"2026-08-{10 + day:02d}"
        for index in range(3):
            rows.append(
                _delivery(trade_date=trade_date, ts=f"{trade_date}T09:3{index}:00")
            )
    result = compute_loud_per_session(rows)
    assert result.sessions == MIN_SAMPLES
    assert result.value == 3.0
    assert result.display_value() == "3.0 per session"


def test_loud_per_session_withheld_until_enough_sessions():
    rows = [_delivery(ts=f"2026-08-10T09:3{i}:00") for i in range(100)]
    result = compute_loud_per_session(rows)
    assert result.value is None
    assert "sessions" in result.note


def test_muted_loud_alerts_are_counted_and_named():
    rows = []
    for day in range(MIN_SAMPLES):
        trade_date = f"2026-08-{10 + day:02d}"
        rows.append(_delivery(trade_date=trade_date, sounded=False, ts=f"{trade_date}T09:30:00"))
    result = compute_loud_per_session(rows)
    assert result.breakdown["sounded"] == 0
    assert "made no sound" in result.note


# --- armed hit delivery -----------------------------------------------------


def _fired(watch_id, trade_date="2026-08-10"):
    return _row("watch_fired", trade_date=trade_date, detail={"watch_id": watch_id})


def _watch_delivery(watch_id, latency, trade_date="2026-08-10"):
    return _delivery(
        action="watch_delivered",
        trade_date=trade_date,
        ts=f"{trade_date}T09:30:00",
        watch_id=watch_id,
        fired_to_delivered_ms=latency,
    )


def test_armed_hits_delivered_within_the_bound_count():
    fired = [_fired(f"w{i}") for i in range(MIN_SAMPLES)]
    delivered = [_watch_delivery(f"w{i}", 100) for i in range(MIN_SAMPLES)]
    result = compute_armed_hit_delivery(delivered, fired)
    assert result.denominator == MIN_SAMPLES
    assert result.value == 1.0


def test_a_fired_watch_never_delivered_counts_against_the_rate():
    fired = [_fired(f"w{i}") for i in range(MIN_SAMPLES)]
    delivered = [_watch_delivery(f"w{i}", 100) for i in range(MIN_SAMPLES - 2)]
    result = compute_armed_hit_delivery(delivered, fired)
    assert result.denominator == MIN_SAMPLES
    assert result.numerator == MIN_SAMPLES - 2
    assert result.breakdown["fired"] - result.breakdown["delivered"] == 2


def test_a_slow_delivery_is_not_a_hit():
    fired = [_fired(f"w{i}") for i in range(MIN_SAMPLES)]
    delivered = [
        _watch_delivery(f"w{i}", LATENCY_BOUND_MS + 1) for i in range(MIN_SAMPLES)
    ]
    result = compute_armed_hit_delivery(delivered, fired)
    assert result.value == 0.0


def test_a_delivery_without_a_latency_cannot_prove_it_met_the_bound():
    fired = [_fired(f"w{i}") for i in range(MIN_SAMPLES)]
    delivered = [_watch_delivery(f"w{i}", None) for i in range(MIN_SAMPLES)]
    result = compute_armed_hit_delivery(delivered, fired)
    assert result.value == 0.0
    assert result.breakdown["no_latency_recorded"] == MIN_SAMPLES


def test_a_delivered_watch_missing_from_the_review_log_still_counts():
    delivered = [_watch_delivery(f"w{i}", 100) for i in range(MIN_SAMPLES)]
    result = compute_armed_hit_delivery(delivered, [])
    assert result.denominator == MIN_SAMPLES
    assert result.value == 1.0


def test_armed_hit_rate_quotes_the_bound_it_was_measured_against():
    fired = [_fired(f"w{i}") for i in range(MIN_SAMPLES)]
    delivered = [_watch_delivery(f"w{i}", 100) for i in range(MIN_SAMPLES)]
    result = compute_armed_hit_delivery(delivered, fired)
    assert str(LATENCY_BOUND_MS) in result.note
