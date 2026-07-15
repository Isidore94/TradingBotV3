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
    assert snapshot["weakest_industries"][0]["entity_key"] == expected["weakest_industry"]
    assert snapshot["strongest_industries"][0]["entity_key"] == expected["strongest_industry"]


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
    # VWAP carries 1.2x evidence weight, so one clean break moves the neutral
    # prior slightly farther than a generic 1.0-weight test.
    assert snapshot["market"]["score"] == 3.8
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
    assert chip == "Technicals: MIXED 5.7/10 | BEARISH | MED"
    assert "Memory 4.0/10 WEAK" in tooltip
    assert "Software 8.0/10 STRONG" in tooltip
    assert color == "#f85149"
