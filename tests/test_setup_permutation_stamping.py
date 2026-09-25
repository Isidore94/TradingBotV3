"""P1-4 / 4a stamping: MA distances, the honest input view, ctx from other stores,
the tracker episode and warehouse occurrence stamps, and old-header history files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_context as spc  # noqa: E402
import setup_permutations as sp  # noqa: E402
import setup_tracker_ledger as ledger  # noqa: E402

NAN = float("nan")


def _row(**overrides):
    row = {
        "symbol": "ABC",
        "side": "LONG",
        "setup_family": "avwap_band_bounce",
        "last_close": 100.0,
        "atr20": 2.0,
        "trend_ma_alignment": False,
        "previous_day_high": 99.0,
        "previous_day_low": 97.0,
        "previous_day_range_break": False,
        "top_pattern_weekly_ema15_hold": False,
        "top_pattern_weekly_above_sma100": False,
        "top_pattern_weekly_sma50_retest_recent": False,
        "top_pattern_weekly_ema15_hold_ratio": None,
        **sp.ma_distance_columns(100.0, 2.0, {"sma20": 99.0, "sma50": 95.0, "sma100": 90.0, "sma200": 80.0,
                                               "ema8": 101.0, "ema15": 99.5, "ema21": 104.0}),
    }
    row.update(overrides)
    return row


# --- MA distances


def test_ma_distance_columns_use_the_warehouse_sign():
    got = sp.ma_distance_columns(100.0, 2.0, {"sma20": 99.0, "ema8": 101.0})
    assert got["perm_dist_sma20_atr"] == 0.5  # MA below price: positive
    assert got["perm_dist_ema8_atr"] == -0.5  # MA above price: negative
    assert got["perm_dist_sma200_atr"] is None  # missing MA: blank, never 0
    assert list(got) == list(sp.MA_DISTANCE_COLUMNS)


@pytest.mark.parametrize(("close", "atr"), [(None, 2.0), (100.0, 0.0), (100.0, NAN), ("x", 2.0)])
def test_ma_distance_columns_blank_without_close_or_atr(close, atr):
    assert set(sp.ma_distance_columns(close, atr, {"sma20": 99.0}).values()) == {None}


def test_the_written_distances_feed_ma_support_and_ma_order():
    key = sp.facets_for_row(_row())
    assert key.get("ma_support") == "multiple_ma_support"  # sma20 +0.5 and ema15 +0.25
    assert key.get("ma_order") == "ema21>price>sma50>sma200"


def test_scan_row_columns_are_the_appended_set_in_order():
    assert sp.SCAN_ROW_COLUMNS == (*sp.MA_DISTANCE_COLUMNS, "perm_weekly_ema8_hold_weeks", *sp.STAMP_COLUMNS)


# --- the honest input view


def test_trend_alignment_false_is_unknown_when_its_mas_were_missing():
    row = _row(perm_dist_ema15_atr=None)
    assert sp.facets_for_row(row).get("trend_ma_alignment") == "ema15_sma20_not_aligned"
    view = sp.scan_row_view(row, has_ma_columns=True)
    assert sp.facets_for_row(view).get("trend_ma_alignment") == sp.UNKNOWN
    # A row older than the MA columns keeps its written value.
    assert sp.scan_row_view(row, has_ma_columns=False)["trend_ma_alignment"] is False
    assert sp.scan_row_view(_row(), has_ma_columns=True)["trend_ma_alignment"] is False


def test_weekly_flags_are_unknown_when_the_weekly_structure_was_not_computed():
    view = sp.scan_row_view(_row(), has_ma_columns=True)
    key = sp.facets_for_row(view)
    assert key.get("weekly_ema15_hold") == sp.UNKNOWN
    assert key.get("weekly_above_sma100") == sp.UNKNOWN
    computed = sp.scan_row_view(_row(top_pattern_weekly_ema15_hold_ratio=0.9), has_ma_columns=True)
    assert sp.facets_for_row(computed).get("weekly_above_sma100") == "weekly_below_sma100"


def test_previous_day_break_false_is_unknown_without_the_side_level():
    assert sp.facets_for_row(sp.scan_row_view(_row(), has_ma_columns=True)).get(
        "prev_day_range_break") == "no_pdr_break"
    long_blank = sp.scan_row_view(_row(previous_day_high=None), has_ma_columns=True)
    assert sp.facets_for_row(long_blank).get("prev_day_range_break") == sp.UNKNOWN
    short_ok = sp.scan_row_view(_row(side="SHORT", previous_day_high=None), has_ma_columns=True)
    assert sp.facets_for_row(short_ok).get("prev_day_range_break") == "no_pdr_break"
    # A row without the column at all is left as written.
    bare = {key: value for key, value in _row().items() if key != "previous_day_high"}
    assert sp.scan_row_view(bare, has_ma_columns=True)["previous_day_range_break"] is False


def test_the_view_never_mutates_the_row():
    row = _row(perm_dist_ema15_atr=None)
    before = dict(row)
    sp.scan_row_view(row, has_ma_columns=True)
    assert row == before


def test_stamp_fields_are_the_key_label_and_version():
    fields = sp.stamp_fields(_row(run_date="2026-09-24"), {"discovery_slot": "1000"})
    assert set(fields) == set(sp.STAMP_COLUMNS)
    assert fields["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION
    assert fields["permutation_key"].startswith(f"{sp.PERMUTATION_RULE_VERSION}|avwap_band_bounce|LONG|")
    assert "slot_1000" in fields["permutation_label"].split("|")
    # Compact form: an unknown facet is simply absent from the stored key.
    assert "weekly_ema15_hold=" not in fields["permutation_key"]
    assert "ma_support=multiple_ma_support" in fields["permutation_key"]


# --- ctx from other stores


def _snapshot(checkpoint, stamp, rows):
    return {"_checkpoint": checkpoint, "_exchange": stamp, "rows": rows}


def test_discovery_slot_is_the_first_checkpoint_that_put_the_name_in_a_bucket():
    snaps = [
        _snapshot("midday", "2026-09-24T13:00", [{"symbol": "ABC", "side": "LONG", "bucket": "favorite_setup"}]),
        _snapshot("open", "2026-09-24T10:30", [{"symbol": "ABC", "side": "LONG", "bucket": ""},
                                                {"symbol": "XYZ", "side": "SHORT", "bucket": "near_favorite_zone"}]),
        _snapshot("close", "2026-09-24T16:05", [{"symbol": "ABC", "side": "LONG", "bucket": "favorite_setup"},
                                                 {"symbol": "ABC", "side": "SHORT", "bucket": "x"}]),
    ]
    slots = spc.discovery_slots(snaps)
    assert slots[("ABC", "LONG")] == "1000"
    assert slots[("XYZ", "SHORT")] == "0730"
    assert slots[("ABC", "SHORT")] == "close"
    assert sp.facets_for_row(_row(), {"discovery_slot": slots[("ABC", "LONG")]}).get("discovery_slot") == "slot_1000"


def test_entry_trigger_is_the_first_watch_fired_of_the_session():
    events = [
        {"action": "watch_fired", "trade_date": "2026-09-24", "ts": "2026-09-24T11:00", "symbol": "abc",
         "side": "LONG", "detail": {"kind": "pullback", "trigger": "sma_retest"}},
        {"action": "watch_fired", "trade_date": "2026-09-24", "ts": "2026-09-24T07:00", "symbol": "ABC",
         "side": "LONG", "detail": {"kind": "band_bounce"}},
        {"action": "watch_fired", "trade_date": "2026-09-23", "ts": "2026-09-23T07:00", "symbol": "XYZ",
         "side": "SHORT", "detail": {"kind": "hod_avwap"}},
        {"action": "arm_watch", "trade_date": "2026-09-24", "ts": "2026-09-24T06:00", "symbol": "XYZ",
         "side": "SHORT", "detail": {"kind": "hod_avwap"}},
    ]
    got = spc.entry_triggers(events, "2026-09-24")
    assert got == {("ABC", "LONG"): "band_bounce"}
    later = spc.entry_triggers(events[:1], "2026-09-24")
    assert later == {("ABC", "LONG"): "pullback_sma_retest"}


def _m5_rows():
    return [
        {"event_id": "ABC_long_20260924_07_10_00_ema_15", "trade_date": "2026-09-24", "symbol": "ABC",
         "direction": "long", "entry_time": "2026-09-24T07:15:00"},
        {"event_id": "ABC_long_20260924_06_40_00_prev_day_high", "trade_date": "2026-09-24", "symbol": "ABC",
         "direction": "long", "entry_time": "2026-09-24T06:45:00"},
        {"event_id": "XYZ_short_20260923_06_40_00_vwap", "trade_date": "2026-09-23", "symbol": "XYZ",
         "direction": "short", "entry_time": "2026-09-23T06:45:00"},
    ]


def test_m5_confirmation_is_the_first_alert_of_the_session():
    assert spc.m5_bounce_types(_m5_rows(), "2026-09-24") == {("ABC", "LONG"): "prev_day_high"}


def test_the_m5_log_is_read_backwards_to_the_session(tmp_path, monkeypatch):
    path = tmp_path / "intraday_bounce_outcomes.csv"
    fields = ["schema_version", "event_id", "event_type", "logged_at", "trade_date", "symbol", "direction",
              "entry_time", "context_json"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for day in range(1, 25):
            for n in range(20):
                writer.writerow({"schema_version": 1, "event_id": f"S{n}_long_202609{day:02d}_07_00_00_ema_8",
                                 "event_type": "registered", "trade_date": f"2026-09-{day:02d}",
                                 "symbol": f"S{n}", "direction": "long",
                                 "entry_time": f"2026-09-{day:02d}T07:05:00",
                                 "context_json": json.dumps({"a, b": "c\"d"})})
    monkeypatch.setattr(spc, "_TAIL_CHUNK", 512)
    got = spc.load_m5_bounce_types("2026-09-20", path=path)
    assert got == {(f"S{n}", "LONG"): "ema_8" for n in range(20)}
    assert spc.load_m5_bounce_types("2026-09-24", path=path)[("S3", "LONG")] == "ema_8"
    assert spc.load_m5_bounce_types("2026-09-24", path=tmp_path / "missing.csv") is None


def test_ctx_is_unknown_when_a_store_was_not_read_and_none_when_it_was():
    unread = spc.SessionContext().ctx_for("ABC", "LONG")
    got = sp.facets_for_row(_row(), unread).as_dict()
    for name in ("discovery_slot", "entry_trigger", "m5_confirmation", "d1_environment"):
        assert got[name] == sp.UNKNOWN
    read = spc.SessionContext(slots={}, triggers={}, m5={}, environment="trend_up").ctx_for("ABC", "LONG")
    got = sp.facets_for_row(_row(), read).as_dict()
    assert got["discovery_slot"] == sp.UNKNOWN  # never shown in a bucket: no slot to claim
    assert got["entry_trigger"] == "no_trigger"
    assert got["m5_confirmation"] == "no_m5_confirmation"
    assert got["d1_environment"] == "env_trend_up"


def test_stamp_scan_rows_adds_only_the_stamp_columns():
    rows = [_row(), _row(symbol="XYZ", side="SHORT"), "not a row"]
    before = [dict(row) for row in rows[:2]]
    context = spc.SessionContext(slots={("ABC", "LONG"): "0730"}, triggers={}, m5={("XYZ", "SHORT"): "vwap"})
    assert spc.stamp_scan_rows(rows, session="2026-09-24", context=context) == 2
    for original, stamped in zip(before, rows[:2]):
        assert set(stamped) - set(original) == set(sp.STAMP_COLUMNS)
        assert {key: stamped[key] for key in original} == original
    assert "slot_0730" in rows[0]["permutation_label"]
    assert "m5_vwap" in rows[1]["permutation_label"]
    assert "|thu" not in rows[0]["permutation_label"] and "weekday=thu" in rows[0]["permutation_key"]


def test_stamp_scan_rows_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(spc.SessionContext, "load", classmethod(lambda cls, session, **k: boom()))
    rows = [_row()]
    assert spc.stamp_scan_rows(rows, session="2026-09-24") == 1
    assert "discovery_slot=" not in rows[0]["permutation_key"]
    monkeypatch.setattr(sp, "stamp_fields", boom)
    fresh = [_row()]
    assert spc.stamp_scan_rows(fresh, session="2026-09-24", context=spc.SessionContext()) == 0
    assert "permutation_key" not in fresh[0]


# --- tracker episode and warehouse occurrence


def _setup(stamped=True):
    feature_row = {"symbol": "ABC", "side": "LONG"}
    if stamped:
        feature_row.update(sp.stamp_fields(_row(run_date="2026-09-24")))
    return {"symbol": "ABC", "side": "LONG", "scan_date": "2026-09-24", "setup_status": "OPEN",
            "setup_family": "avwap_band_bounce", "feature_row": feature_row}


def test_the_tracker_episode_carries_the_key_it_opened_with():
    events = ledger.diff_setups({"s1": _setup(), "s2": _setup(stamped=False)}, {}, data_session="2026-09-24")
    by_id = {event["setup_id"]: event for event in events}
    assert by_id["s1"]["event_type"] == ledger.EVENT_INITIAL
    assert by_id["s1"]["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION
    assert by_id["s1"]["permutation_key"].startswith(sp.PERMUTATION_RULE_VERSION)
    assert "permutation_key" not in by_id["s2"]
    # The key is not a state field: stamping it never makes a transition on its own.
    assert "permutation_key" not in ledger.STATE_FIELDS


def test_the_warehouse_occurrence_tags_carry_the_first_scan_key():
    from research_warehouse import tracker_adapter

    event = ledger.diff_setups({"s1": _setup()}, {}, data_session="2026-09-24")[0]
    event["event_at"] = "2026-09-24T20:00:00+00:00"
    scenario = {"setup_id": "s1", "scan_date": "2026-09-24", "symbol": "ABC", "side": "LONG",
                "setup_family": "avwap_band_bounce", "entry_price": "100", "tradeable": "True",
                "stop_source_type": "ema", "stop_reference_label": "EMA_21", "initial_risk_per_share": "2",
                "close_failure_limit": "1"}
    detections = tracker_adapter.detections_from_tracker(scenario_rows=[scenario], event_rows=[event])
    assert len(detections) == 1
    tags = json.loads(detections[0]["tags"])
    assert tags["permutation_key"] == event["permutation_key"]
    assert tags["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION
    unstamped = ledger.diff_setups({"s1": _setup(stamped=False)}, {}, data_session="2026-09-24")[0]
    unstamped["event_at"] = event["event_at"]
    plain = tracker_adapter.detections_from_tracker(scenario_rows=[scenario], event_rows=[unstamped])
    assert "permutation_key" not in json.loads(plain[0]["tags"])


# --- old-header history files


def test_an_old_header_history_file_widens_and_old_readers_still_parse(tmp_path, monkeypatch):
    from master_avwap_lib import legacy

    target = tmp_path / "d1_features_history.csv"
    old = pd.DataFrame([{"feature_history_schema_version": 1, "run_id": "r0", "run_timestamp": "t0",
                         "run_date": "2026-09-23", "watchlist_label": "w", "scoring_config_hash": "h",
                         "scoring_config_updated_at": "", "symbol": "OLD", "side": "LONG",
                         "trend_ma_alignment": True, "last_close": 10.0}])
    old.to_csv(target, index=False)
    monkeypatch.setattr(legacy, "D1_FEATURE_HISTORY_FILE", target)
    new_row = {"symbol": "NEW", "side": "LONG", "trend_ma_alignment": False, "last_close": 20.0,
               **{column: None for column in sp.SCAN_ROW_COLUMNS}}
    new_row.update(sp.stamp_fields({**new_row, "run_date": "2026-09-24"}))
    legacy.append_d1_feature_history(pd.DataFrame([new_row]), {"run_id": "r1", "run_date": "2026-09-24"})

    with target.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    header = list(rows[0])
    assert header[-len(sp.SCAN_ROW_COLUMNS):] == list(sp.SCAN_ROW_COLUMNS)
    assert rows[0]["symbol"] == "OLD" and rows[0]["trend_ma_alignment"] == "True"
    assert all(rows[0][column] == "" for column in sp.SCAN_ROW_COLUMNS)
    assert rows[1]["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION

    frame = pd.read_csv(target, low_memory=False)
    assert list(frame["symbol"]) == ["OLD", "NEW"]
    # The old row keys through the pre-4a path: no MA columns, unknown MA facets, no crash.
    old_key = sp.facets_for_row(sp.scan_row_view(frame.iloc[0].to_dict(), has_ma_columns=False))
    assert old_key.get("ma_support") == sp.UNKNOWN
    assert old_key.get("trend_ma_alignment") == "ema15_sma20_aligned"


# --- review blocker 2: a source's silence before its first logged event is unknown


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_live_ctx_is_unknown_before_each_source_s_first_logged_session(tmp_path):
    events = _write_jsonl(tmp_path / "alert_review_events.jsonl", [
        {"action": "watch_fired", "trade_date": "2026-07-31", "ts": "2026-07-31T10:00:00-04:00",
         "symbol": "XYZ", "side": "LONG", "detail": {"kind": "band_bounce"}},
    ])
    m5 = tmp_path / "intraday_bounce_outcomes.csv"
    with m5.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "event_type", "trade_date", "symbol", "direction",
                                                    "entry_time"])
        writer.writeheader()
        writer.writerow({"event_id": "XYZ_long_20260801_07_00_00_ema_8", "event_type": "registered",
                         "trade_date": "2026-08-01", "symbol": "XYZ", "direction": "long",
                         "entry_time": "2026-08-01T07:00:00"})
    paths = {"review_events_path": events, "m5_outcomes_path": m5, "reports_dir": tmp_path / "none",
             "environment_path": tmp_path / "none.jsonl"}
    before = spc.SessionContext.load("2026-04-24", **paths).ctx_for("ABC", "LONG")
    key = sp.facets_for_row(_row(), before).as_dict()
    assert key["entry_trigger"] == sp.UNKNOWN  # the log did not exist yet: not "no_trigger"
    assert key["m5_confirmation"] == sp.UNKNOWN
    after = spc.SessionContext.load("2026-08-03", **paths).ctx_for("ABC", "LONG")
    key = sp.facets_for_row(_row(), after).as_dict()
    assert key["entry_trigger"] == "no_trigger"
    assert key["m5_confirmation"] == "no_m5_confirmation"
    assert spc.covered("2026-07-31", "2026-07-31") and not spc.covered("2026-07-30", "2026-07-31")
    assert not spc.covered("2026-08-03", None)


def test_the_shadow_columns_are_never_contrasted():
    from ai_jobs import miss_contrast

    row = {"symbol": "ABC", "run_id": "r", "relvol": 1.2, "perm_dist_sma50_atr": 0.4,
           "perm_weekly_ema8_hold_weeks": 3, "permutation_key": "k", "permutation_label": "l",
           "permutation_rule_version": "v"}
    assert miss_contrast._feature_mapping(row) == {"relvol": 1.2}
