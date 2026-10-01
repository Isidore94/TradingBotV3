"""P17 wiring: ctx:tape_now, /rs and /alerts, and the gate's cached-bars rows."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import commands  # noqa: E402
from mentor_packs import alerts_pack, bars_pack, context_pack, gate_pack, registry, rs_pack  # noqa: E402


def test_tape_now_line_from_the_rs_and_alert_packs():
    rs = rs_pack.build(top=2, now=rs_pack.FIXTURE_NOW, sources=rs_pack.fixture_sources())
    text = context_pack.tape_now_text(rs, alerts_pack.fixture())
    assert text == ("Tape now: leading Cybersecurity, Semiconductors; lagging Autos, Insurance; "
                    "alerts today M5 2 (1 long, 1 short); D1 2 (0 long, 2 short)")


def test_tape_now_says_unknown_when_the_board_is_missing():
    rs = rs_pack.build(now=rs_pack.FIXTURE_NOW, sources=replace(rs_pack.fixture_sources(), snapshot=lambda: None))
    none = alerts_pack.build(day="yesterday", now=alerts_pack.FIXTURE_NOW, sources=alerts_pack.fixture_sources())
    text = context_pack.tape_now_text(rs, none)
    assert text.startswith("Tape now: Industry board: no snapshot on disk") and text.endswith("no alerts today yet")


def test_context_pack_carries_tape_now_and_a_failure_is_unknown():
    rows = {row["id"]: row for row in context_pack.fixture().rows}
    assert rows["ctx:tape_now"]["text"].startswith("Tape now: leading Cybersecurity")

    def boom(moment):
        raise OSError("gone")

    broken = context_pack.build(now=rs_pack.FIXTURE_NOW, sources=replace(context_pack.fixture_sources(), tape_now=boom))
    row = next(row for row in broken.rows if row["id"] == "ctx:tape_now")
    assert row["kind"] == "unknown"


def test_live_tape_now_is_cached_for_five_minutes(monkeypatch):
    calls = []
    monkeypatch.setattr(context_pack, "_tape_now_cache", {})
    monkeypatch.setattr(rs_pack, "build", lambda **kw: calls.append("rs") or registry.make_pack("rs_pack", ()))
    monkeypatch.setattr(alerts_pack, "build", lambda **kw: calls.append("al") or registry.make_pack("alerts_pack", ()))
    first = context_pack._live_tape_now(rs_pack.FIXTURE_NOW)
    second = context_pack._live_tape_now(rs_pack.FIXTURE_NOW)
    assert first == second and calls == ["rs", "al"]


def test_rs_and_alerts_commands():
    assert commands.handle("/rs").arg == ("rs_pack", {"level": "industry"})
    assert commands.handle("/rs sectors").arg == ("rs_pack", {"level": "sector"})
    assert commands.handle("/rs foo").action == "error"
    assert commands.handle("/alerts").arg == ("alerts_pack", {})
    assert commands.handle("/alerts ALL yesterday").arg == ("alerts_pack", {"symbol": "ALL", "day": "yesterday"})
    assert commands.handle("/alerts m5").arg == ("alerts_pack", {"kind": "m5"})
    assert commands.handle("/alerts ALL NVDA").action == "error"
    assert "/rs" in commands.HELP_TEXT and "/alerts" in commands.HELP_TEXT


def test_the_gate_sees_where_price_is_now(tmp_path):
    world_sources = gate_pack.fixture_sources(tmp_path)
    src = replace(world_sources, bars_sources=bars_pack.fixture_sources())
    pack = gate_pack.build("short", "ALL", now=bars_pack.FIXTURE_NOW, sources=src)
    ids = pack.ids
    assert "gate:ALL:bars:ALL:last" in ids and "gate:ALL:bars:ALL:bar:6" in ids and "gate:ALL:bars:ALL:bar:7" not in ids
    without = gate_pack.build("short", "ALL", now=bars_pack.FIXTURE_NOW, sources=world_sources)
    assert not any(":bars:" in row_id for row_id in without.ids)
