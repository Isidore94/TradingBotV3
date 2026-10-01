"""P17 alerts_pack: what the bot alerted (M5 bounces, D1 fired events, D1 bucket upgrades)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach  # noqa: E402
from mentor_packs import alerts_pack, registry  # noqa: E402

NOW = alerts_pack.FIXTURE_NOW
UNK = "; follow-through unknown (no cached M5 bars around the alert)"


def test_golden_pack_both_kinds():
    pack = alerts_pack.fixture()
    by_id = {row["id"]: row["text"] for row in pack.rows}
    assert list(by_id) == ["alert:2026-09-30:summary", "alert:2026-09-30:m5:followthrough",
                           "alert:2026-09-30:m5:1", "alert:2026-09-30:m5:2",
                           "alert:2026-09-30:d1:1", "alert:2026-09-30:d1:2"]
    assert by_id["alert:2026-09-30:summary"] == ("Alerts on 2026-09-30: M5 2 (1 long, 1 short); D1 2 (0 long, 2 short); "
                                                 "in your book: ALL; liked: CE")
    assert by_id["alert:2026-09-30:m5:1"] == "detected 12:02 ET (bar time earlier) NVDA LONG M5 bounce eod_vwap" + UNK
    assert by_id["alert:2026-09-30:m5:2"] == ("detected 10:15 ET (bar time earlier) ALL SHORT M5 bounce ema_21, tier B "
                                              "(alert score 0.129: the setup's past average R, not this move) "
                                              "[book SHORT]" + UNK)
    assert by_id["alert:2026-09-30:m5:followthrough"] == (
        "M5 alerts ranked by the move since the alert in its favour (cached M5 bars; not the alert score): "
        "none measurable; 2 of 2 unknown (no cached bars)")
    assert by_id["alert:2026-09-30:d1:1"] == ("D1 scan CE SHORT D1 bucket upgrade to near_favorite_zone "
                                              "(Trendline break) @ 45.12 [liked]")
    assert by_id["alert:2026-09-30:d1:2"] == "detected 09:31 ET (bar time earlier) CL SHORT d1 event fired: D1 15EMA rejection (short)"
    assert len(pack.ids) == len(set(pack.ids))


def test_symbol_kind_and_day_filters():
    src = alerts_pack.fixture_sources()
    only_all = alerts_pack.build(symbol="all", now=NOW, sources=src)
    assert [row["id"] for row in only_all.rows] == ["alert:2026-09-30:summary", "alert:2026-09-30:m5:followthrough",
                                                    "alert:2026-09-30:m5:1"]
    assert only_all.rows[0]["text"].startswith("Alerts for ALL on 2026-09-30: M5 1 (0 long, 1 short); D1 0")
    d1 = alerts_pack.build(kind="d1", now=NOW, sources=src)
    assert not any(":m5:" in row_id for row_id in d1.ids) and "M5" not in d1.rows[0]["text"]
    yesterday = alerts_pack.build(day="yesterday", now=NOW, sources=src)
    assert yesterday.ids == ("alert:2026-09-29:none",)


def _bars(symbol, closes, start="2026-10-01T09:30:00-04:00", vwap=None):
    from mentor_packs import bars_pack

    first = datetime.fromisoformat(start)
    return [{"symbol": symbol, "start": (first + i * bars_pack.BAR).isoformat(), "open": close,
             "interval_start": (first + i * bars_pack.BAR).isoformat(),
             "high": close + 0.05, "low": close - 0.05, "close": close, "volume": 1000, "vwap": vwap}
            for i, close in enumerate(closes)]


def test_follow_through_ranks_by_the_real_move_not_the_alert_score():
    """2026-10-01: DUK (tier A, score 0.151) was called the top short while it was back above VWAP."""
    now = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
    m5 = [
        {"time_local": "07:00:30", "trade_date": "2026-10-01", "symbol": "DUK", "direction": "short",
         "bounce_types": "lrsi_cross_50", "tier": "A", "composite_r": "0.151"},
        {"time_local": "07:00:30", "trade_date": "2026-10-01", "symbol": "CIFR", "direction": "short",
         "bounce_types": "lrsi_cross_20", "tier": "B", "composite_r": "0.090"},
        {"time_local": "07:00:30", "trade_date": "2026-10-01", "symbol": "NOBARS", "direction": "short",
         "bounce_types": "ema_15", "tier": "A", "composite_r": "0.300"},
    ]
    # 18 bars 09:30-10:55 ET; the alert is detected 10:00:30 ET, so its entry is the 09:55 bar's close.
    duk = _bars("DUK", [114.0, 113.8, 113.6, 113.4, 113.2, 113.0] + [113.2 + 0.2 * i for i in range(12)], vwap=113.5)
    cifr = _bars("CIFR", [10.0] * 6 + [9.9 - 0.05 * i for i in range(12)], vwap=10.0)
    cached = {"DUK": duk, "CIFR": cifr}
    src = replace(alerts_pack.fixture_sources(), m5_rows=lambda day: m5, d1_events=lambda day: [],
                  upgrades=lambda: {}, book=dict,
                  bars=lambda symbols, day: {s: cached[s] for s in symbols if s in cached})
    pack = alerts_pack.build(day="today", kind="m5", now=now, sources=src)
    rows = {row["id"]: row for row in pack.rows}
    ranked = rows["alert:2026-10-01:m5:followthrough"]
    assert ranked["ranked"] == ["CIFR", "DUK"], "the move since the alert ranks, never the tier or score"
    assert "CIFR SHORT at 10:00 ET: now +6.50%" in ranked["text"]
    assert "DUK SHORT at 10:00 ET: now -2.12%" in ranked["text"] and "back through VWAP" in ranked["text"]
    assert "1 of 3 unknown (no cached bars)" in ranked["text"]
    duk_row = next(row["text"] for row in pack.rows if row.get("symbol") == "DUK")
    assert "alert score 0.151" in duk_row and "(0.151R)" not in duk_row
    assert "entry 113.00" in duk_row and "now above session VWAP 113.50 (back through VWAP, against the alert)" in duk_row
    nobars = next(row["text"] for row in pack.rows if row.get("symbol") == "NOBARS")
    assert nobars.endswith(UNK), "missing bars are unknown, never confirmed"


def test_read_day_bars_reads_one_file_once_for_many_symbols(tmp_path):
    path = tmp_path / "2026-10-01.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _bars("DUK", [1.0, 2.0]) + _bars("X", [3.0]))
                    + "\n{torn", encoding="utf-8")
    got = alerts_pack.read_day_bars({"DUK"}, path)
    assert list(got) == ["DUK"] and len(got["DUK"]) == 2
    assert alerts_pack.read_day_bars({"DUK"}, tmp_path / "missing.jsonl") == {}


def _write_desk(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    bounces = base / "intraday_bounces.csv"
    bounces.write_text(
        "time_local,trade_date,symbol,direction,bounce_types,tier,composite_r,shadow_s9_tier,shadow_s9_composite_r\n"
        "07:15:05,2026-09-30,ALL,short,ema_21,B,0.129,S,0.24\n"
        "09:02:00,2026-09-30,NVDA,long,eod_vwap,,,,\n"
        "08:00:00,2026-09-29,TSLA,short,ema_21,A,0.3,,\n", encoding="utf-8")
    events_dir = base / "alert_review_events"
    events_dir.mkdir()
    lines = [
        {"action": "d1_event_fired", "detail": {"kind": "ema15_reject", "message": "D1 15EMA rejection (short)"},
         "schema": "review_events_v2", "side": "SHORT", "symbol": "CL", "trade_date": "2026-09-30",
         "ts": "2026-09-30T06:31:14.296826"},
        {"action": "shown", "symbol": "CL", "trade_date": "2026-09-30", "ts": "2026-09-30T06:31:15"},
        {"action": "level_fired", "detail": {"message": "fired"}, "side": "LONG", "symbol": "OLD",
         "trade_date": "2026-09-29", "ts": "2026-09-29T07:00:00"},
    ]
    (events_dir / "review-events-abc.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n{\"action\": \"d1_event_fi", encoding="utf-8")
    legacy = base / "alert_review_events.jsonl"
    legacy.write_text("", encoding="utf-8")
    upgrades = base / "master_avwap_d1_upgrade_alerts.json"
    upgrades.write_text(json.dumps({"schema_version": 1, "run_date": "2026-09-30", "symbols": {"CE": {
        "symbol": "CE", "side": "SHORT", "priority_bucket": "near_favorite_zone",
        "bucket_upgrade_events": [{"alert_label": "Trendline break", "level": 45.1169}]}}}), encoding="utf-8")
    outcomes = base / "intraday_bounce_outcomes.csv"
    outcomes.write_text("event_id\n", encoding="utf-8")
    return {"INTRADAY_BOUNCES_FILE": bounces, "ALERT_REVIEW_EVENTS_DIR": events_dir,
            "ALERT_REVIEW_EVENTS_FILE": legacy, "MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE": upgrades,
            "INTRADAY_BOUNCE_OUTCOMES_FILE": outcomes}


def test_live_readers_on_real_format_files_never_open_the_outcome_store(monkeypatch, tmp_path):
    import builtins
    import io

    import project_paths

    for key, path in _write_desk(tmp_path / "desk").items():
        monkeypatch.setattr(project_paths, key, path)
    opened: list[str] = []

    def audit(real):
        def wrapper(file, *args, **kwargs):
            if "intraday_bounce_outcomes" in str(file):
                opened.append(str(file))
                raise AssertionError(f"alerts_pack opened {file}")
            return real(file, *args, **kwargs)
        return wrapper

    real_path_open = Path.open
    monkeypatch.setattr(builtins, "open", audit(builtins.open))
    monkeypatch.setattr(io, "open", audit(io.open))
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: audit(lambda f, *x, **y: real_path_open(self, *x, **y))(
        self, *a, **k))
    src = replace(alerts_pack.live_sources(), local_tz=lambda: ZoneInfo("America/Los_Angeles"),
                  book={"ALL": "SHORT"}.copy)
    pack = alerts_pack.build(liked=["CE"], now=NOW, sources=src)
    assert pack.rows == alerts_pack.fixture().rows
    assert opened == []


def test_bad_day_is_a_bad_argument_not_a_raise():
    pack = registry.build("alerts_pack", day="last tuesday")
    assert not pack.rows and "bad arguments" in pack.empty_text or "could not be built" in pack.empty_text


def test_registered_and_attached():
    assert "alerts_pack" in registry.names()
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    known = {"ALL": "SHORT", "NVDA": "LONG"}
    found = [r for r in attach.plan_attachments("what alerts fired today", known, now) if r.name == "alerts_pack"]
    assert found and found[0].args == {"day": "today"}
    found = [r for r in attach.plan_attachments("did ALL alert this morning", known, now) if r.name == "alerts_pack"]
    assert found and found[0].args == {"day": "today", "symbol": "ALL"}
    found = [r for r in attach.plan_attachments("what alerted yesterday", known, now) if r.name == "alerts_pack"]
    assert found and found[0].args == {"day": "yesterday"}


@pytest.mark.parametrize("question", ["what's the bot flagging", "any bounce alerts", "wick alerts on NVDA"])
def test_alert_words(question):
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    assert "alerts_pack" in [r.name for r in attach.plan_attachments(question, {"NVDA": "LONG"}, now)]


def test_this_week_is_per_day_summaries_and_totals_never_today():
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # a Wednesday
    found = [r for r in attach.plan_attachments("any alerts this week", {}, now) if r.name == "alerts_pack"]
    assert found[0].args == {"day": "week"}
    pack = alerts_pack.build(day="week", now=NOW, sources=alerts_pack.fixture_sources())
    assert pack.ids == ("alert:week:summary", "alert:week:2026-09-28", "alert:week:2026-09-29",
                        "alert:week:2026-09-30")
    texts = {row["id"]: row["text"] for row in pack.rows}
    assert texts["alert:week:summary"] == ("Alerts this week (2026-09-28 to 2026-09-30): M5 2 (1 long, 1 short); "
                                           "D1 2 (0 long, 2 short); in your book: ALL; liked: none")
    assert texts["alert:week:2026-09-28"] == ("Alerts on Mon 2026-09-28: M5 0 (0 long, 0 short); "
                                              "D1 0 (0 long, 0 short)")


def test_a_named_weekday_is_its_date():
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # a Wednesday
    found = [r for r in attach.plan_attachments("what alerted monday", {}, now) if r.name == "alerts_pack"]
    assert found[0].args == {"day": "2026-09-28"}
