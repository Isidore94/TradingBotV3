import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


FIXTURE = Path(__file__).parent / "fixtures" / "technical_integrity_scoring_v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _entity(snapshot, entity_type, key):
    return next(
        row
        for row in snapshot["entities"]
        if row["entity_type"] == entity_type and row["entity_key"] == key
    )


def test_technical_integrity_golden_hierarchy():
    from technical_integrity import TechnicalIntegrityConfig, aggregate_technical_integrity

    fixture = _fixture()
    event_payload = json.dumps(fixture["events"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(event_payload.encode()).hexdigest() == fixture["raw_input_sha256"]
    config = TechnicalIntegrityConfig()
    assert config.to_dict() == fixture["configuration"]
    snapshot = aggregate_technical_integrity(
        fixture["events"],
        as_of=fixture["as_of"],
        session_date=fixture["session_date"],
        config=config,
    )
    expected = fixture["expected"]
    market = snapshot["market"]
    for key, value in expected["market"].items():
        assert market[key] == value
    for entity_type, key, expected_key in (
        ("sector", "technology", "technology"),
        ("industry", "memory", "memory"),
        ("industry", "software", "software"),
        ("stock", "MU", "mu"),
    ):
        row = _entity(snapshot, entity_type, key)
        for field, value in expected[expected_key].items():
            assert row[field] == value
    # Entities without enough decisive evidence report no score, so they are
    # not rankable - the boards stay empty rather than ranking noise.
    assert len(snapshot["weakest_industries"]) == expected["scored_industry_count"]
    assert len(snapshot["strongest_industries"]) == expected["scored_industry_count"]

    # With the evidence-sufficiency rule disabled the same events must still
    # produce the exact scores above, which pins the scoring math itself
    # independently of the suppression policy.
    math_only = aggregate_technical_integrity(
        fixture["events"],
        as_of=fixture["as_of"],
        session_date=fixture["session_date"],
        config=TechnicalIntegrityConfig(min_decisive_weight_for_score=0.0),
    )
    expected_math = fixture["expected_math_only"]
    for entity_type, key, expected_key in (
        ("industry", "memory", "memory"),
        ("industry", "software", "software"),
        ("stock", "MU", "mu"),
    ):
        row = _entity(math_only, entity_type, key)
        for field, value in expected_math[expected_key].items():
            assert row[field] == value
    assert math_only["weakest_industries"][0]["entity_key"] == expected_math["weakest_industry"]
    assert math_only["strongest_industries"][0]["entity_key"] == expected_math["strongest_industry"]


def test_chop_is_counted_but_never_scored():
    """Chop was 69% of resolutions and its value equalled the prior, so it
    added weight and zero information - the mechanical cause of every symbol
    converging on ~6.15."""
    from technical_integrity import TechnicalIntegrityConfig, aggregate_technical_integrity

    def event(event_id, outcome):
        return {
            "event_type": "level_resolved", "event_id": event_id,
            "session_date": "2026-07-22", "resolved_at": f"2026-07-22T10:00:0{event_id[-1]}-04:00",
            "symbol": "MU", "sector_key": "technology", "sector": "Technology",
            "industry_key": "memory", "industry": "Memory", "level_family": "vwap",
            "level_timeframe": "intraday", "outcome": outcome, "break_direction": "",
            "event_weight": 1.0, "approach_side": "above",
        }

    decisive = [event(f"h{i}", "held") for i in range(9)]
    padded = decisive + [event(f"c{i}", "chop") for i in range(9)]
    config = TechnicalIntegrityConfig()
    kwargs = {"as_of": "2026-07-22T11:00:00-04:00", "session_date": "2026-07-22"}

    clean = aggregate_technical_integrity(decisive, config=config, **kwargs)["market"]
    with_chop = aggregate_technical_integrity(padded, config=config, **kwargs)["market"]

    # Nine inconclusive tests must not drag a strong reading toward neutral.
    assert with_chop["score"] == clean["score"]
    assert with_chop["test_count"] == clean["test_count"] == 9
    # ...but they are still reported, so the inconclusiveness stays visible.
    assert with_chop["chop_count"] == 9
    assert clean["chop_count"] == 0


def test_score_is_centred_on_the_measured_base_respect_rate():
    """5.5 must mean 'levels respected as often as they typically are'.

    Under the old 1+9p map the neutral prior scored 5.5 only because the prior
    was 0.5, while levels actually hold ~74% of the time - so a typical symbol
    read 6.55 and 'above the midpoint' meant nothing.
    """
    from technical_integrity import (
        TECHNICAL_INTEGRITY_BASE_RESPECT,
        _score_from_probability,
    )

    base = TECHNICAL_INTEGRITY_BASE_RESPECT
    assert _score_from_probability(base, base) == 5.5
    assert _score_from_probability(0.0, base) == 1.0
    assert _score_from_probability(1.0, base) == 10.0
    # Respect below the norm reads below the midpoint, and vice versa.
    assert _score_from_probability(base - 0.2, base) < 5.5
    assert _score_from_probability(base + 0.1, base) > 5.5


def test_thin_evidence_reports_building_instead_of_a_number():
    from technical_integrity import TechnicalIntegrityConfig, aggregate_technical_integrity

    one_test = [{
        "event_type": "level_resolved", "event_id": "e1", "session_date": "2026-07-22",
        "resolved_at": "2026-07-22T10:00:00-04:00", "symbol": "MU",
        "sector_key": "technology", "sector": "Technology", "industry_key": "memory",
        "industry": "Memory", "level_family": "vwap", "level_timeframe": "intraday",
        "outcome": "held", "break_direction": "", "event_weight": 1.0,
        "approach_side": "above",
    }]
    snapshot = aggregate_technical_integrity(
        one_test, as_of="2026-07-22T11:00:00-04:00", session_date="2026-07-22",
        config=TechnicalIntegrityConfig(),
    )
    row = _entity(snapshot, "stock", "MU")
    assert row["score"] is None
    assert row["state"] == "BUILDING"


def test_completed_m5_filter_excludes_forming_bar():
    from technical_integrity import completed_m5_bars

    tz = ZoneInfo("America/New_York")
    start = datetime(2026, 7, 15, 9, 30, tzinfo=tz)
    rows = [
        {
            "datetime": start + timedelta(minutes=5 * index),
            "open": 100 + index,
            "high": 100.5 + index,
            "low": 99.5 + index,
            "close": 100.2 + index,
            "volume": 1000,
        }
        for index in range(4)
    ]
    complete = completed_m5_bars(rows, now=start + timedelta(minutes=19, seconds=59))
    assert len(complete) == 3
    assert complete[-1]["bar_end"] == (start + timedelta(minutes=15)).isoformat(timespec="seconds")


def test_bounce_observer_recomputes_levels_without_forming_bar(monkeypatch):
    import pandas as pd

    import bounce_bot_lib.legacy as legacy

    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 7, 15, 9, 42, tzinfo=tz)
    monkeypatch.setattr(legacy, "get_market_local_now", lambda: now)
    rows = [
        {"datetime": datetime(2026, 7, 14, 15, 55), "open": 90, "high": 91, "low": 89, "close": 90, "volume": 1000},
        {"datetime": datetime(2026, 7, 15, 9, 30), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        {"datetime": datetime(2026, 7, 15, 9, 35), "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
        # 09:40-09:45 is still forming at 09:42 and must not enter any level.
        {"datetime": datetime(2026, 7, 15, 9, 40), "open": 101, "high": 201, "low": 100, "close": 200, "volume": 1000},
    ]
    captured = {}

    class FakeMonitor:
        def observe_symbol(self, symbol, frame, metrics, **kwargs):
            captured.update(symbol=symbol, frame=frame.copy(), metrics=dict(metrics), kwargs=kwargs)

    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    bot._technical_integrity_monitor = FakeMonitor()
    bot._d1_extra_levels_provider = lambda symbol, **kwargs: [
        {"family": "d1_sma_50", "value": 99.5, "weight": 1.6}
    ]
    bot.atr_cache = {"MU": 2.0}
    bot.symbol_classification_cache = {"MU": {"sectorKey": "technology"}}
    bot.get_market_environment = lambda: "bearish_strong"
    bot.calculate_vwap_with_stdev_bands = lambda frame: (
        float(frame["close"].iloc[-1]),
        float(frame["close"].iloc[-1]) + 1,
        float(frame["close"].iloc[-1]) - 1,
    )
    bot.calculate_dynamic_vwap_with_stdev_bands = bot.calculate_vwap_with_stdev_bands
    bot.calculate_eod_vwap_with_stdev_bands = bot.calculate_vwap_with_stdev_bands

    legacy.BounceBot._observe_technical_integrity(bot, "MU", pd.DataFrame(rows))

    assert list(captured["frame"]["close"]) == [100, 101]
    assert captured["metrics"]["std_vwap"] == 100.0
    assert captured["metrics"]["dynamic_vwap"] == 100.0
    assert captured["kwargs"]["now"] == now
    assert captured["kwargs"]["extra_levels"] == [
        {"family": "d1_sma_50", "value": 99.5, "weight": 1.6}
    ]


def test_monitor_resolves_support_break_and_dedupes(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    tz = ZoneInfo("America/New_York")
    start = datetime(2026, 7, 15, 9, 30, tzinfo=tz)
    bars = [
        {"datetime": start, "open": 101.2, "high": 101.4, "low": 100.8, "close": 101.0, "volume": 1000},
        {"datetime": start + timedelta(minutes=5), "open": 101.0, "high": 101.1, "low": 99.98, "close": 100.2, "volume": 1400},
    ]
    monitor = TechnicalIntegrityMonitor(
        events_path=tmp_path / "events.jsonl",
        state_path=tmp_path / "state.json",
        snapshot_path=tmp_path / "snapshot.json",
    )
    classification = {
        "sectorKey": "technology",
        "sector": "Technology",
        "industryKey": "memory",
        "industry": "Memory",
    }
    monitor.observe_symbol(
        "MU",
        bars,
        {"std_vwap": 100.0},
        atr=1.0,
        classification=classification,
        market_environment="bearish_strong",
        now=start + timedelta(minutes=11),
    )
    assert monitor.pending_count == 1

    resolved_bars = bars + [
        {"datetime": start + timedelta(minutes=10), "open": 100.1, "high": 100.2, "low": 99.6, "close": 99.8, "volume": 1500},
        {"datetime": start + timedelta(minutes=15), "open": 99.8, "high": 99.9, "low": 99.4, "close": 99.7, "volume": 1600},
        {"datetime": start + timedelta(minutes=20), "open": 99.7, "high": 99.8, "low": 99.3, "close": 99.6, "volume": 1700},
    ]
    snapshot = monitor.observe_symbol(
        "MU",
        resolved_bars,
        {"std_vwap": 100.0},
        atr=1.0,
        classification=classification,
        market_environment="bearish_strong",
        now=start + timedelta(minutes=26),
    )
    assert monitor.pending_count == 0
    assert snapshot["market"]["test_count"] == 1
    # One resolved test is not a measurement: below min_decisive_weight_for_score
    # the reading is the prior, so the score is withheld rather than printed.
    assert snapshot["market"]["score"] is None
    assert snapshot["market"]["state"] == "BUILDING"
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event_type"] for row in rows] == ["level_test_started", "level_resolved"]
    assert rows[-1]["outcome"] == "broke"
    assert rows[-1]["break_direction"] == "down"

    monitor.observe_symbol(
        "MU",
        resolved_bars,
        {"std_vwap": 100.0},
        atr=1.0,
        classification=classification,
        market_environment="bearish_strong",
        now=start + timedelta(minutes=26),
    )
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_monitor_recovers_pending_test_from_append_only_ledger(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    tz = ZoneInfo("America/New_York")
    start = datetime(2026, 7, 15, 9, 30, tzinfo=tz)
    paths = {
        "events_path": tmp_path / "events.jsonl",
        "state_path": tmp_path / "state.json",
        "snapshot_path": tmp_path / "snapshot.json",
    }
    monitor = TechnicalIntegrityMonitor(**paths)
    touch = [
        {"datetime": start, "open": 101.2, "high": 101.4, "low": 100.8, "close": 101.0, "volume": 1000},
        {"datetime": start + timedelta(minutes=5), "open": 101.0, "high": 101.1, "low": 99.98, "close": 100.2, "volume": 1400},
    ]
    monitor.observe_symbol("MU", touch, {"std_vwap": 100.0}, atr=1.0, now=start + timedelta(minutes=11))
    paths["state_path"].unlink()  # simulate loss/crash before the state replacement survives

    recovered = TechnicalIntegrityMonitor(**paths)
    assert recovered.pending_count == 0  # session is established on first observation
    resolved_bars = touch + [
        {"datetime": start + timedelta(minutes=10 + 5 * index), "open": 99.7, "high": 99.8, "low": 99.3, "close": 99.7, "volume": 1500}
        for index in range(3)
    ]
    snapshot = recovered.observe_symbol(
        "MU",
        resolved_bars,
        {"std_vwap": 100.0},
        atr=1.0,
        now=start + timedelta(minutes=26),
    )
    assert snapshot["market"]["test_count"] == 1
    rows = [json.loads(line) for line in paths["events_path"].read_text().splitlines()]
    assert [row["event_type"] for row in rows] == ["level_test_started", "level_resolved"]


def test_calibration_report_ranks_candidate_configs():
    from technical_integrity import TechnicalIntegrityConfig, compare_scoring_configs

    events = _fixture()["events"]
    report = compare_scoring_configs(
        events,
        {
            "baseline": TechnicalIntegrityConfig(),
            "lighter_prior": TechnicalIntegrityConfig(prior_weight=0.5),
        },
    )
    assert report["schema"] == "technical_integrity_calibration_v1"
    assert report["event_count"] == len(events)
    assert {row["name"] for row in report["configs"]} == {"baseline", "lighter_prior"}
    assert all(0.0 <= row["brier_score"] <= 1.0 for row in report["configs"])
    assert report["configs"] == sorted(report["configs"], key=lambda row: row["brier_score"])
    assert not report["review_gate"]["eligible"]
    assert report["best_replay_config"] is None


def test_calibration_report_scores_only_predictions_recorded_before_resolution(tmp_path):
    from technical_integrity import write_technical_integrity_calibration_report

    events_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "report.json"
    rows = []
    for index, (prediction, actual) in enumerate(((0.8, "held"), (0.3, "broke"))):
        rows.append(
            {
                "event_type": "level_resolved",
                "event_id": f"e{index}",
                "session_date": "2026-07-15",
                "resolved_at": f"2026-07-15T10:{index:02d}:00-04:00",
                "symbol": "MU",
                "predicted_hold_probability": prediction,
                "outcome": actual,
                "event_weight": 1.0,
            }
        )
    events_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = write_technical_integrity_calibration_report(
        events_path=events_path,
        output_path=output_path,
    )
    assert report["recorded_live_predictions"]["event_count"] == 2
    assert report["recorded_live_predictions"]["brier_score"] == 0.065
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema"] == report["schema"]


def test_formatter_does_not_present_prior_session_as_today():
    from technical_integrity import aggregate_technical_integrity, format_technical_integrity_snapshot

    fixture = _fixture()
    snapshot = aggregate_technical_integrity(
        fixture["events"],
        as_of=fixture["as_of"],
        session_date=fixture["session_date"],
    )
    chip, tooltip, color = format_technical_integrity_snapshot(
        snapshot,
        now=datetime(2026, 7, 16, 9, 31, tzinfo=ZoneInfo("America/New_York")),
    )
    assert chip == "Technicals: building today"
    assert "2026-07-15" in tooltip
    assert color == "#8b8fa3"


def test_integrity_dialog_exposes_searchable_full_hierarchy():
    from PySide6.QtWidgets import QApplication

    from technical_integrity import aggregate_technical_integrity
    from ui.widgets.technical_integrity_dialog import TechnicalIntegrityDialog

    QApplication.instance() or QApplication([])
    fixture = _fixture()
    snapshot = aggregate_technical_integrity(
        fixture["events"],
        as_of=fixture["as_of"],
        session_date=fixture["session_date"],
    )
    dialog = TechnicalIntegrityDialog(snapshot)
    assert dialog.table.rowCount() == len(snapshot["entities"])
    dialog.search.setText("MU")
    visible = [
        dialog.table.item(row, 1).text()
        for row in range(dialog.table.rowCount())
        if not dialog.table.isRowHidden(row)
    ]
    assert visible == ["MU"]


def test_d1_extra_levels_use_d1_resolution_and_allow_concurrent_family_tests(tmp_path):
    from technical_integrity import TechnicalIntegrityMonitor

    tz = ZoneInfo("America/New_York")
    start = datetime(2026, 7, 15, 9, 30, tzinfo=tz)
    touch = [
        {"datetime": start, "open": 101.2, "high": 101.4, "low": 100.9, "close": 101.0, "volume": 1000},
        {"datetime": start + timedelta(minutes=5), "open": 101.0, "high": 101.1, "low": 99.98, "close": 100.2, "volume": 1400},
    ]
    monitor = TechnicalIntegrityMonitor(
        events_path=tmp_path / "events.jsonl",
        state_path=tmp_path / "state.json",
        snapshot_path=tmp_path / "snapshot.json",
    )
    extra_levels = [
        {"family": "d1_horizontal", "value": 100.0, "weight": 1.5, "detail": {"strength": 1.4}},
        {"family": "d1_horizontal", "value": 100.6, "weight": 1.5},
    ]
    monitor.observe_symbol(
        "MU",
        touch,
        {"std_vwap": 100.0},  # confluent with the 100.0 horizontal; D1 weight wins dedupe
        atr=1.0,
        now=start + timedelta(minutes=11),
        extra_levels=extra_levels,
    )
    # Two different D1 horizontals may be under test at once (per-price dedupe),
    # and the confluent VWAP candidate collapsed into the D1 test.
    assert monitor.pending_count == 2
    pending = list(monitor.pending.values())
    assert {row["level_family"] for row in pending} == {"d1_horizontal"}
    assert all(row["level_timeframe"] == "d1" for row in pending)
    assert all(row["resolution_bars"] == 6 for row in pending)
    assert all(row["break_buffer_atr"] == 0.15 for row in pending)

    resolved_bars = touch + [
        {
            "datetime": start + timedelta(minutes=10 + 5 * index),
            "open": 99.7,
            "high": 99.8,
            "low": 99.5,
            "close": 99.7,
            "volume": 1500,
        }
        for index in range(6)
    ]
    snapshot = monitor.observe_symbol(
        "MU",
        resolved_bars,
        {},
        atr=1.0,
        now=start + timedelta(minutes=41),
        extra_levels=extra_levels,
    )
    assert monitor.pending_count == 0
    market = snapshot["market"]
    assert market["d1_test_count"] == 2
    assert market["intraday_test_count"] == 0
    assert market["d1_pressure"] == "BEARISH"
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    resolved = [row for row in rows if row["event_type"] == "level_resolved"]
    assert {row["outcome"] for row in resolved} == {"broke"}
    assert all(row["level_timeframe"] == "d1" for row in resolved)
    assert all(row["level_family"] == "d1_horizontal" for row in resolved)
    assert next(
        row.get("level_detail") for row in resolved if row["level_value"] == 100.0
    ) == {"strength": 1.4}


def test_aggregation_splits_d1_and_intraday_and_chip_leads_with_d1():
    from technical_integrity import aggregate_technical_integrity, format_technical_integrity_snapshot

    def event(event_id, family, timeframe, outcome, weight, break_direction=""):
        return {
            "event_type": "level_resolved",
            "event_id": event_id,
            "session_date": "2026-07-15",
            "resolved_at": f"2026-07-15T10:0{event_id[-1]}:00-04:00",
            "symbol": "MU",
            "sector_key": "technology",
            "sector": "Technology",
            "industry_key": "memory",
            "industry": "Memory",
            "level_family": family,
            "level_timeframe": timeframe,
            "outcome": outcome,
            "break_direction": break_direction,
            "event_weight": weight,
            "approach_side": "above",
        }

    # Each timeframe needs enough decisive weight to earn a score of its own
    # (min_decisive_weight_for_score), so the split is exercised with a real
    # sample rather than two tests per side.
    events = [event(f"d{i}", "d1_sma_50", "d1", "held", 1.6) for i in range(5)]
    events += [event(f"m{i}", "vwap", "intraday", "broke", 1.2, "down") for i in range(7)]
    snapshot = aggregate_technical_integrity(
        events,
        as_of="2026-07-15T11:00:00-04:00",
        session_date="2026-07-15",
    )
    market = snapshot["market"]
    assert market["d1_test_count"] == 5
    assert market["d1_score"] == 9.1
    assert market["d1_state"] == "STRONG"
    assert market["d1_pressure"] == "BALANCED"
    assert market["intraday_test_count"] == 7
    assert market["intraday_score"] == 1.9
    assert market["pressure"] == "BEARISH"  # combined still sees the M5 breaks

    chip, tooltip, color = format_technical_integrity_snapshot(
        snapshot,
        now=datetime(2026, 7, 15, 11, 31, tzinfo=ZoneInfo("America/New_York")),
    )
    # Confidence stays LOW: one symbol, however many tests it contributes.
    assert chip == "Technicals D1: STRONG 9.1/10 | BALANCED | LOW · M5 1.9/10"
    assert "D1 major levels" in tooltip
    assert "Intraday M5 levels" in tooltip
    assert color == "#58a6ff"  # colored by the D1 verdict, not the M5 breaks


def test_legacy_events_without_timeframe_count_as_intraday():
    from technical_integrity import aggregate_technical_integrity

    fixture = _fixture()
    snapshot = aggregate_technical_integrity(
        fixture["events"],
        as_of=fixture["as_of"],
        session_date=fixture["session_date"],
    )
    market = snapshot["market"]
    assert market["d1_test_count"] == 0
    assert market["d1_state"] == "BUILDING"
    assert market["intraday_test_count"] == market["test_count"]
    assert market["intraday_score"] == market["score"]


def test_market_state_formatter_is_plain_and_actionable():
    from technical_integrity import aggregate_technical_integrity, format_technical_integrity_snapshot

    fixture = _fixture()
    snapshot = aggregate_technical_integrity(
        fixture["events"],
        as_of=fixture["as_of"],
        session_date=fixture["session_date"],
    )
    chip, tooltip, color = format_technical_integrity_snapshot(
        snapshot,
        now=datetime(2026, 7, 15, 11, 31, tzinfo=ZoneInfo("America/New_York")),
    )
    # No D1 evidence yet: the chip says so and falls back to the M5 read.
    # Recentred on the measured base respect rate: a tape with four clean
    # breaks reads below the 5.5 midpoint instead of a reassuring "MIXED 5.7".
    assert chip == "Technicals D1: building · M5 WEAK 4.5/10 | BEARISH | MED"
    assert "D1 major levels: building" in tooltip
    # The industry leader/laggard lines are gone from this fixture on purpose:
    # 10 tests across 6 symbols cannot support a per-industry reading, so no
    # industry is scored and none is ranked. Previously the tooltip named a
    # "weakest" and "strongest" industry off 4 and 2 tests respectively.
    assert "Memory" not in tooltip
    assert "Software" not in tooltip
    assert color == "#f85149"
