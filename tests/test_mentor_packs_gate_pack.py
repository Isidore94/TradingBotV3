"""gate_pack: the /check pre-trade pack. Fixture paths only; never a live store."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import gate_pack, pick_pack, plan_lines, regime_pack, registry  # noqa: E402

NOW = gate_pack.FIXTURE_NOW


def _build(tmp_path, *args, **kw):
    src_kw = {k: kw.pop(k) for k in ("risk", "trades", "plan_text") if k in kw}
    return gate_pack.build(*args, now=NOW, sources=gate_pack.fixture_sources(tmp_path, **src_kw), **kw)


def _row(pack, suffix):
    return next(row for row in pack.rows if row["id"] == f"gate:{suffix}")


def _trade(trade_id, symbol, side, qty, entry, stop):
    return {"trade_id": trade_id, "symbol": symbol, "direction": side, "quantity_opened": qty, "quantity_closed": 0,
            "average_entry_price": entry, "planned_stop": stop, "opened_at": f"2026-09-28T07:0{trade_id[-1]}:00-07:00"}


def test_short_with_stop_size_entry_and_set_risk(tmp_path):
    pack = _build(tmp_path, "SHORT", "nvda", 400, 3.20, 3.05, risk=100)
    req = _row(pack, "NVDA:req")
    assert (req["side"], req["symbol"], req["size"], req["stop"], req["entry"]) == ("SHORT", "NVDA", 400, 3.2, 3.05)
    risk = _row(pack, "NVDA:risk")
    assert risk["at_risk"] == pytest.approx(60.0)
    assert risk["ratio"] == pytest.approx(0.6)
    assert "$100.00" in risk["text"] and "$60.00" in risk["text"]
    assert not risk.get("wrong_side")


def test_long_with_setting_unset_says_not_set_and_no_ratio(tmp_path):
    pack = _build(tmp_path, "LONG", "NVDA", 100, 9.0, 10.0, risk=None)
    risk = _row(pack, "NVDA:risk")
    assert gate_pack.NOT_SET_TEXT in risk["text"]
    assert "ratio" not in risk and risk["setting"] is None
    assert risk["at_risk"] == pytest.approx(100.0)


def test_request_parts_not_given_and_no_guess(tmp_path):
    pack = _build(tmp_path, "LONG", "NVDA", risk=100)
    assert _row(pack, "NVDA:req")["text"].count("not given") == 3
    risk = _row(pack, "NVDA:risk")
    assert risk["at_risk"] is None and "ratio" not in risk and "unknown" in risk["text"]


def test_wrong_side_stop_is_flagged(tmp_path):
    risk = _row(_build(tmp_path, "SHORT", "NVDA", 100, 3.0, 3.05), "NVDA:risk")
    assert risk["wrong_side"] is True


def test_same_symbol_open_is_flagged(tmp_path):
    pack = _build(tmp_path, "LONG", "NVDA", trades=[_trade("T9", "NVDA", "LONG", 50, 120.0, None)])
    row = _row(pack, "NVDA:book:T9")
    assert row["same_symbol"] and "SAME SYMBOL" in row["text"] and "no stop on file" in row["text"]
    assert _row(pack, "NVDA:book:industry")["same_symbol_open"] is True


def test_two_open_names_in_same_industry(tmp_path):
    trades = [_trade("T1", "AMD", "SHORT", 100, 150.0, 152.5), _trade("T2", "AVGO", "SHORT", 10, 300.0, 310.0),
              _trade("T3", "TSLA", "SHORT", 5, 250.0, None)]
    src = gate_pack.fixture_sources(tmp_path, trades=trades)
    imap = {**src.industry_map(), "AVGO": {"industry": "Semiconductors"}}
    src = gate_pack.Sources(**{**src.__dict__, "industry_map": lambda: imap})
    pack = gate_pack.build("SHORT", "NVDA", now=NOW, sources=src)
    book = _row(pack, "NVDA:book:industry")
    assert book["same_industry_count"] == 2 and book["open_count"] == 3
    assert "(AMD, AVGO)" in book["text"]
    assert _row(pack, "NVDA:book:T1")["at_risk"] == pytest.approx(250.0)
    assert _row(pack, "NVDA:book:T2")["at_risk"] == pytest.approx(100.0)
    assert _row(pack, "NVDA:book:T3")["at_risk"] is None


def test_empty_plan_says_no_plan_lines(tmp_path):
    pack = _build(tmp_path, "SHORT", "NVDA", plan_text="")
    assert _row(pack, "NVDA:plan:none")["text"] == f"Plan: {plan_lines.EMPTY_TEXT}"
    assert gate_pack.plan_ids(pack) == set()


def test_plan_pick_and_tape_rows_embedded_under_the_gate_prefix(tmp_path):
    pack = _build(tmp_path, "SHORT", "NVDA", 400, 3.2, 3.05)
    plan = plan_lines.build(path=tmp_path / "trading_plan.md")
    assert plan.ids and gate_pack.plan_ids(pack) == set(plan.ids)
    for plan_id in plan.ids:
        assert f"gate:NVDA:{plan_id}" in pack.ids
    pick = pick_pack.build("NVDA", "SHORT", now=NOW, paths=gate_pack.fixture_sources(tmp_path / "p").pick_paths)
    for row in pick.rows:
        if row["kind"] not in ("plan_line", "plan_empty"):
            assert f"gate:NVDA:{row['id']}" in pack.ids
    for tape_id in sorted(gate_pack.TAPE_IDS):
        assert f"gate:NVDA:{tape_id}" in pack.ids
    assert not any(":tape:econ" in i or ":tape:night" in i for i in pack.ids)
    assert regime_pack.fixture().rows  # the tape fixture is the one embedded


def test_ids_unique_prefixed_and_tz_aware(tmp_path):
    pack = _build(tmp_path, "SHORT", "NVDA", 400, 3.2, 3.05)
    assert len(pack.ids) == len(set(pack.ids)) == len(pack.rows)
    assert all(i.startswith("gate:NVDA:") for i in pack.ids)
    stamped = [row["at_utc"] for row in pack.rows if "at_utc" in row]
    assert stamped and all(datetime.fromisoformat(s).tzinfo is not None for s in stamped)


def test_a_different_request_changes_the_hash(tmp_path):
    a = _build(tmp_path, "SHORT", "NVDA", 400, 3.2, 3.05)
    b = _build(tmp_path / "b", "SHORT", "NVDA", 300, 3.2, 3.05)
    assert gate_pack.pack_hash(a) != gate_pack.pack_hash(b)
    assert gate_pack.pack_hash(a) == gate_pack.pack_hash(_build(tmp_path / "c", "SHORT", "NVDA", 400, 3.2, 3.05))


def test_bad_request_is_an_empty_pack():
    assert not gate_pack.build("SHORT", "").rows
    assert not gate_pack.build("SIDEWAYS", "NVDA").rows


def test_read_open_trades_is_read_only(tmp_path):
    db = tmp_path / "j.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, symbol TEXT, direction TEXT, status TEXT, opened_at TEXT,"
        " quantity_opened REAL, quantity_closed REAL, average_entry_price REAL);"
        "CREATE TABLE trade_annotations (trade_id TEXT, planned_stop REAL);"
        "INSERT INTO trades VALUES ('T1','AMD','SHORT','OPEN','2026-09-28',100,0,150);"
        "INSERT INTO trades VALUES ('T2','MU','LONG','CLOSED','2026-09-27',10,10,90);"
        "INSERT INTO trade_annotations VALUES ('T1',152.5);"
    )
    conn.commit()
    conn.close()
    before = db.stat().st_mtime_ns
    rows = gate_pack.read_open_trades(db)
    assert [(r["trade_id"], r["planned_stop"]) for r in rows] == [("T1", 152.5)]
    assert db.stat().st_mtime_ns == before
    assert gate_pack.read_open_trades(tmp_path / "missing.sqlite3") == []


def test_registered_and_fixture_reads_no_live_source(monkeypatch):
    assert "gate_pack" in registry.names()

    def boom(*_a, **_k):
        raise AssertionError("the fixture read a live source")

    monkeypatch.setattr(gate_pack, "live_sources", boom)
    monkeypatch.setattr(pick_pack, "live_paths", boom)
    monkeypatch.setattr(regime_pack, "live_sources", boom)
    pack = gate_pack.fixture()
    assert not any(row["kind"] == "unknown" for row in pack.rows)
