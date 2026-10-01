"""P17 bars_pack: the bot's cached M5 bars (the research spool's M5 tee), completed bars only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach  # noqa: E402
from mentor_packs import bars_pack, registry  # noqa: E402


def test_golden_pack_never_shows_the_forming_bar():
    pack = bars_pack.fixture()
    by_id = {row["id"]: row["text"] for row in pack.rows}
    assert list(by_id) == ["bars:ALL:asof", "bars:ALL:bar:1", "bars:ALL:bar:2", "bars:ALL:bar:3",
                           "bars:ALL:day", "bars:ALL:last"]
    assert by_id["bars:ALL:asof"] == "ALL cached M5 bars: last completed bar closed 2026-09-30 10:00 ET, 3 min ago"
    assert by_id["bars:ALL:bar:3"] == "09:55 ET O 105.00 H 106.00 L 104.50 C 105.50 V 6,000"
    assert by_id["bars:ALL:day"] == ("ALL 2026-09-30 regular session so far: open 100.00, high 106.00, low 99.50, "
                                     "6 bars; approx session VWAP 103.67 (typical price x volume of the cached "
                                     "bars; VWAP not in cache)")
    assert by_id["bars:ALL:last"] == "ALL last 105.50 at 10:00 ET (close of the last completed bar)"
    assert not any("10:00 ET O" in text for text in by_id.values())  # the 10:00-10:05 bar is still forming
    assert len(pack.ids) == len(set(pack.ids))


def test_stale_cache_is_flagged():
    later = bars_pack.FIXTURE_NOW + timedelta(minutes=40)
    pack = bars_pack.build("ALL", now=later, sources=bars_pack.fixture_sources())
    asof = pack.rows[0]
    assert asof["stale"] is True and "STALE > 15 min" in asof["text"]
    assert "stale" in pack.rows[-1]["text"]


def test_long_ages_read_in_hours_then_days():
    assert bars_pack.age_text(119) == "119 min" and bars_pack.age_text(458) == "7.6 h"
    later = bars_pack.FIXTURE_NOW + timedelta(days=5, hours=14)
    pack = bars_pack.build("ALL", now=later, sources=bars_pack.fixture_sources())
    assert "5.6 days ago (STALE" in pack.rows[0]["text"] and " min ago" not in pack.rows[0]["text"]
    assert pack.rows[-1]["text"].endswith("stale, 5.6 days old")


def test_naive_bar_times_are_market_local_and_shown_in_et():
    # The desk machine runs Pacific: a naive 06:30 bar is 09:30 ET.
    rows = [{"symbol": "NVDA", "dt": "2026-09-30T06:30:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5,
             "volume": 10}]
    pack = bars_pack.build("NVDA", now=datetime(2026, 9, 30, 13, 40, tzinfo=timezone.utc),
                           sources=bars_pack.fixture_sources(rows))
    texts = [row["text"] for row in pack.rows]
    assert any(text.startswith("09:30 ET O 1.00") for text in texts), texts
    assert datetime.fromisoformat(pack.rows[0]["at_utc"]) == datetime(2026, 9, 30, 13, 35, tzinfo=timezone.utc)


def test_uncached_symbol_says_the_bot_only_caches_watched_names():
    pack = bars_pack.build("TSLA", now=bars_pack.FIXTURE_NOW, sources=bars_pack.fixture_sources())
    assert pack.ids == ("bars:TSLA:none",) and "only caches names it is watching" in pack.rows[0]["text"]


def test_reads_the_real_spool_format(tmp_path):
    segment = tmp_path / "segment-20260930T133000-abcd1234.open.jsonl"
    lines = []
    for row in bars_pack.fixture_rows():
        lines.append(json.dumps({"dataset": "bar_m5", "shed_class": "PROTECTED", "row": row}))
    lines.append(json.dumps({"dataset": "scan_coverage", "shed_class": "SHEDDABLE",
                             "row": {"symbol": "ALL", "x": 1}}))
    segment.write_text("\n".join(lines) + "\n{\"dataset\": \"bar_m5\", \"row\": {\"symbol\": \"ALL\",", encoding="utf-8")
    rows = bars_pack.read_spool_bars("ALL", [segment])
    assert len(rows) == 7 and all("interval_start" in row for row in rows)
    src = bars_pack.Sources(bars=lambda sym: bars_pack.read_spool_bars(sym, [segment]),
                            market_tz=lambda: ZoneInfo("America/Los_Angeles"))
    assert bars_pack.build("ALL", n=3, now=bars_pack.FIXTURE_NOW, sources=src).rows == bars_pack.fixture().rows


def test_vwap_unknown_when_the_session_is_partial():
    rows = bars_pack.fixture_rows()[2:]
    pack = bars_pack.build("ALL", now=bars_pack.FIXTURE_NOW, sources=bars_pack.fixture_sources(rows))
    day = next(row["text"] for row in pack.rows if row["id"] == "bars:ALL:day")
    assert "VWAP not in cache" in day and "approx" not in day
    assert "cached from 09:40 ET only (today's open not cached)" in day


def test_registered_and_attached():
    assert "bars_pack" in registry.names()
    now = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    known = {"NVDA": "LONG", "ALL": "SHORT", "TSLA": "SHORT"}
    for question, sym in (("where is NVDA trading right now", "NVDA"), ("is ALL above vwap", "ALL"),
                          ("how's TSLA acting intraday", "TSLA")):
        found = [r for r in attach.plan_attachments(question, known, now) if r.name == "bars_pack"]
        assert found and found[0].args["symbol"] == sym, question
    # A pre-trade question attaches the gate, and the gate itself carries bars_pack(symbol, n=6).
    names = [r.name for r in attach.plan_attachments("thinking of shorting ALL here", known, now)]
    assert "gate_pack" in names and "bars_pack" not in names
    from mentor_packs import gate_pack

    assert isinstance(gate_pack.live_sources().bars_sources, bars_pack.Sources)
