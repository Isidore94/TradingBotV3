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


def test_golden_pack_both_kinds():
    pack = alerts_pack.fixture()
    by_id = {row["id"]: row["text"] for row in pack.rows}
    assert list(by_id) == ["alert:2026-09-30:summary", "alert:2026-09-30:m5:1", "alert:2026-09-30:m5:2",
                           "alert:2026-09-30:d1:1", "alert:2026-09-30:d1:2"]
    assert by_id["alert:2026-09-30:summary"] == ("Alerts on 2026-09-30: M5 2 (1 long, 1 short); D1 2 (0 long, 2 short); "
                                                 "in your book: ALL; liked: CE")
    assert by_id["alert:2026-09-30:m5:1"] == "12:02 ET NVDA LONG M5 bounce eod_vwap"
    assert by_id["alert:2026-09-30:m5:2"] == "10:15 ET ALL SHORT M5 bounce ema_21, tier B (0.129R) [book SHORT]"
    assert by_id["alert:2026-09-30:d1:1"] == ("scan ET CE SHORT D1 bucket upgrade to near_favorite_zone "
                                              "(Trendline break) @ 45.12 [liked]")
    assert by_id["alert:2026-09-30:d1:2"] == "09:31 ET CL SHORT d1 event fired: D1 15EMA rejection (short)"
    assert len(pack.ids) == len(set(pack.ids))


def test_symbol_kind_and_day_filters():
    src = alerts_pack.fixture_sources()
    only_all = alerts_pack.build(symbol="all", now=NOW, sources=src)
    assert [row["id"] for row in only_all.rows] == ["alert:2026-09-30:summary", "alert:2026-09-30:m5:1"]
    assert only_all.rows[0]["text"].startswith("Alerts for ALL on 2026-09-30: M5 1 (0 long, 1 short); D1 0")
    d1 = alerts_pack.build(kind="d1", now=NOW, sources=src)
    assert not any(":m5:" in row_id for row_id in d1.ids) and "M5" not in d1.rows[0]["text"]
    yesterday = alerts_pack.build(day="yesterday", now=NOW, sources=src)
    assert yesterday.ids == ("alert:2026-09-29:none",)


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


def test_a_named_weekday_is_its_date():
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)  # a Wednesday
    found = [r for r in attach.plan_attachments("what alerted monday", {}, now) if r.name == "alerts_pack"]
    assert found[0].args == {"day": "2026-09-28"}
