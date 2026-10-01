"""Mentor P8: the book pack. One source per pack (fresh Questrade, else the labelled journal), accounts
with their tax class, positions with $ at risk only when a stop is on file, exposure, and
deterministic hints (no shorts in registered accounts, industry clusters, room only when set)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import questrade_positions as qp  # noqa: E402
from mentor_packs import book_pack  # noqa: E402

NOW = book_pack.FIXTURE_NOW
FRESH = NOW.astimezone(timezone.utc).isoformat(timespec="seconds")


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


def _build(**kwargs):
    now = kwargs.pop("now", NOW)
    return book_pack.build(now=now, sources=book_pack.fixture_sources(**kwargs))


GOLDEN_QUESTRADE = [
    ("book:asof", "Book as of Tue 09-29 07:05 PT"),
    ("book:source", "Source: Questrade positions at Tue 09-29 07:05 PT"),
    ("book:acct:111", "Account TFSA 111 (TFSA): registered, tax-free, no shorts; 1 open position(s); "
                      "cash CAD 2,500.00, USD 1,200.00"),
    ("book:acct:222", "Account Margin 222 (Margin): margin; 1 open position(s); cash unknown"),
    ("book:pos:111:NVDA", "LONG NVDA 50 @ 120.00 in TFSA 111; market value $6,250.00; stop unknown"),
    ("book:pos:222:AMD", "SHORT AMD 100 @ 150.00 in Margin 222; market value -$14,800.00; "
                         "$ at risk $250.00 (stop 152.5)"),
    ("book:side:LONG", "LONG: 1 position(s), gross market value $6,250.00"),
    ("book:side:SHORT", "SHORT: 1 position(s), gross market value $14,800.00"),
    ("book:industry:Semiconductors", "Industry Semiconductors: 2 open name(s) (AMD, NVDA)"),
    ("book:hint:no_shorts_registered", "Short candidates cannot go in TFSA 111 (registered accounts hold no shorts)"),
    ("book:hint:cluster:Semiconductors", "2 open names in Semiconductors (AMD, NVDA)"),
]


def test_golden_questrade_snapshot_two_accounts():
    pack = _build(snapshot=qp.fixture_snapshot(FRESH).as_dict())
    assert [(row["id"], row["text"]) for row in pack.rows] == GOLDEN_QUESTRADE
    assert _rows(pack)["book:source"]["source"] == "questrade"


def test_golden_journal_fallback_says_why_and_never_mixes():
    pack = _build(status={"reason": "no token", "at_utc": FRESH})
    rows = _rows(pack)
    assert rows["book:source"]["text"] == "Source: journal open trades (Questrade unavailable: no token)"
    assert rows["book:source"]["source"] == "journal"
    assert "cash unknown (the journal has no cash)" in rows["book:acct:111"]["text"]
    assert rows["book:pos:222:AMD"]["text"] == ("SHORT AMD 100 @ 150.00 in Margin 222; market value unknown; "
                                                 "$ at risk $250.00 (stop 152.5); "
                                                 "source: journal (not checked against broker)")
    assert rows["book:side:LONG"]["text"] == "LONG: 1 position(s), gross market value unknown (1 without a market value)"
    assert "Questrade positions" not in pack.as_text()


def test_a_stale_snapshot_is_not_used_and_the_journal_says_its_age():
    pack = _build(snapshot=qp.fixture_snapshot(FRESH).as_dict(), now=NOW + timedelta(minutes=16))
    rows = _rows(pack)
    assert rows["book:source"]["source"] == "journal"
    assert "last snapshot 16 min old" in rows["book:source"]["text"]
    assert all(row.get("market_value") is None for row in pack.rows if row.get("kind") == "position"), \
        "journal rows never carry a Questrade market value"


def test_a_failure_after_the_snapshot_names_the_failure():
    stale = (NOW - timedelta(hours=2)).astimezone(timezone.utc).isoformat()
    pack = _build(snapshot=qp.fixture_snapshot(stale).as_dict(),
                  status={"reason": "RuntimeError: 400 Client Error", "at_utc": FRESH})
    assert "Questrade unavailable: RuntimeError: 400 Client Error" in _rows(pack)["book:source"]["text"]


def test_no_token_and_no_trades_is_a_labelled_empty_book():
    pack = _build(status={"reason": "no token", "at_utc": FRESH}, trades=[], accounts=[])
    rows = _rows(pack)
    assert rows["book:pos:none"]["text"] == "No open positions"
    assert rows["book:side:SHORT"]["text"] == "SHORT: 0 position(s), gross market value $0.00"
    assert not any(row_id.startswith("book:hint:no_shorts") for row_id in rows)


def test_never_fetched_says_not_fetched_yet():
    assert "Questrade unavailable: not fetched yet" in _rows(_build())["book:source"]["text"]


def test_a_same_industry_cluster_and_top_three_industries():
    positions = [
        {"account_number": "222", "symbol": sym, "side": "LONG", "open_qty": 10, "avg_price": 10.0,
         "market_value": 100.0} for sym in ("NVDA", "AMD", "AVGO", "TSLA", "ZZZ")
    ]
    pack = _build(snapshot=qp.fixture_snapshot(FRESH, positions).as_dict())
    rows = _rows(pack)
    assert rows["book:industry:Semiconductors"]["names"] == ["AMD", "AVGO", "NVDA"]
    assert rows["book:hint:cluster:Semiconductors"]["text"] == "3 open names in Semiconductors (AMD, AVGO, NVDA)"
    assert "book:industry:Auto-Manufacturers" in rows and "book:hint:cluster:Auto-Manufacturers" not in rows
    assert rows["book:industry:unknown"]["names"] == ["ZZZ"]


def test_room_hints_only_when_the_setting_exists():
    assert not any(r["kind"] == "hint_room" for r in _build(snapshot=qp.fixture_snapshot(FRESH).as_dict()).rows)
    rows = _rows(_build(snapshot=qp.fixture_snapshot(FRESH).as_dict(), max_positions="1"))
    assert rows["book:hint:room:111"]["text"] == "TFSA 111 is full: 1 of 1 positions"
    rows = _rows(_build(snapshot=qp.fixture_snapshot(FRESH).as_dict(), max_positions=3))
    assert rows["book:hint:room:111"]["text"] == "TFSA 111 has room: 2 position(s) (max 3 per account)"


@pytest.mark.parametrize("words,expected", [
    ({"account_type": "TFSA"}, "tax_free"), ({"account_type": "RRSP"}, "tax_deferred"),
    ({"account_type": "Margin"}, "margin"), ({"account_type": "Cash"}, "cash"),
    ({"account_type": "Margin", "tax_status": "TAX_FREE"}, "tax_free"), ({"account_type": ""}, "unknown"),
])
def test_tax_class(words, expected):
    assert book_pack.tax_class(words) == expected


def test_ids_unique_citable_and_times_tz_aware():
    from mentor_packs.citations import CITATION_RE

    pack = book_pack.fixture()
    assert len(pack.ids) == len(set(pack.ids))
    for row_id in pack.ids:
        assert CITATION_RE.fullmatch(f"[{row_id}]"), row_id
    load = book_pack.load_book(book_pack.fixture_sources(snapshot=qp.fixture_snapshot(FRESH).as_dict()), NOW)
    assert datetime.fromisoformat(load.asof).tzinfo is not None


def test_no_sizing_advice_and_no_live_path_in_the_fixture(monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", Path("Z:/must/not/read.sqlite3"))
    text = book_pack.fixture().as_text().lower()
    for word in ("buy ", "sell ", "size", "shares to", "order"):
        assert word not in text, word


def test_journal_readers_carry_the_account_and_its_tax_status(tmp_path):
    import sqlite3

    from mentor_packs import gate_pack

    path = tmp_path / "trade_journal.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, account_number TEXT, account_label TEXT, symbol TEXT, direction TEXT, "
        "status TEXT, quantity_opened REAL, quantity_closed REAL, average_entry_price REAL, opened_at TEXT);"
        "INSERT INTO trades VALUES ('T1', '111', 'TFSA 111', 'NVDA', 'LONG', 'OPEN', 50, 0, 120, '2026-09-28');"
        "CREATE TABLE accounts (broker TEXT, account_number TEXT, account_label TEXT, account_type TEXT, "
        "raw_json TEXT, tax_status TEXT);"
        "INSERT INTO accounts VALUES ('QUESTRADE', '111', 'TFSA 111', 'TFSA', '{}', 'TAX_FREE');"
    )
    conn.commit()
    conn.close()
    trades = gate_pack.read_open_trades(path)
    assert trades[0]["account_number"] == "111" and trades[0]["planned_stop"] is None
    accounts = book_pack.read_accounts(path)
    assert accounts == [{"broker": "QUESTRADE", "account_number": "111", "account_label": "TFSA 111",
                         "account_type": "TFSA", "tax_status": "TAX_FREE"}]
    src = book_pack.Sources(snapshot=lambda: None, status=lambda: None, open_trades=lambda: trades,
                            accounts=lambda: accounts, industry_map=lambda: {})
    rows = _rows(book_pack.build(now=NOW, sources=src))
    assert "registered, tax-free" in rows["book:acct:111"]["text"] and "book:pos:111:NVDA" in rows


def test_live_readers_are_read_only(tmp_path):
    from mentor_app.store import MentorChatStore

    store = MentorChatStore(tmp_path / "mentor_chat.sqlite3")
    store.set_state(book_pack.SNAPSHOT_KEY, qp.fixture_snapshot(FRESH).as_json())
    assert book_pack.db_snapshot_reader(store.path)()["fetched_utc"] == FRESH
    assert book_pack.db_status_reader(store.path)() is None
    assert book_pack.read_accounts(tmp_path / "missing.sqlite3") == []
    source = (SCRIPTS_DIR / "mentor_packs" / "book_pack.py").read_text(encoding="utf-8")
    assert "JournalStore(" not in source and "FocusPickStore(" not in source and "mode=ro" in source


# ---------------------------------------------------------------- P8 review follow-ups (P9 step A)
@pytest.mark.parametrize("account,expected", [
    ({"account_type": "TFSA", "tax_status": "TAXABLE"}, "taxable"),
    ({"account_type": "RRSP", "account_label": "RRSP x", "tax_status": "TAXABLE"}, "taxable"),
    ({"account_type": "Margin", "tax_status": "TAXABLE"}, "margin"),
    ({"account_type": "TFSA", "tax_status": "TAX_DEFERRED"}, "tax_deferred"),
    ({"account_type": "Margin", "account_label": "California Margin"}, "margin"),
    ({"account_type": "Individual", "account_label": "Clifton"}, "unknown"),
    ({"account_type": "Individual", "account_label": "My TFSA-2"}, "tax_free"),
])
def test_tax_status_wins_outright_and_words_are_whole_tokens(account, expected):
    assert book_pack.tax_class(account) == expected


def _stop_trade(trade_id, acct, sym, side, stop, qty=100, entry=150.0):
    return {"trade_id": trade_id, "account_number": acct, "account_label": f"A{acct}", "symbol": sym,
            "direction": side, "quantity_opened": qty, "quantity_closed": 0, "average_entry_price": entry,
            "planned_stop": stop, "opened_at": "2026-09-28T07:00:00-07:00"}


def _qt_pos(acct, sym="AMD", **extra):
    return {"account_number": acct, "account_label": f"Acct {acct}", "symbol": sym, "open_qty": 10.0,
            "side": "LONG", "avg_price": 150.0, "market_value": 1500.0, "security_type": "STK", **extra}


def test_stop_matches_the_position_in_the_same_account():
    trades = [_stop_trade("T1", "111", "AMD", "LONG", 140.0), _stop_trade("T2", "222", "AMD", "LONG", 145.0)]
    rows = _rows(_build(snapshot=qp.fixture_snapshot(FRESH, [_qt_pos("111"), _qt_pos("222")]).as_dict(),
                        trades=trades))
    assert rows["book:pos:111:AMD"]["at_risk"] == 100.0
    assert rows["book:pos:222:AMD"]["at_risk"] == 50.0


def test_stop_without_an_account_matches_by_symbol_and_side():
    trades = [_stop_trade("T1", "", "AMD", "LONG", 145.0)]
    rows = _rows(_build(snapshot=qp.fixture_snapshot(FRESH, [_qt_pos("222")]).as_dict(), trades=trades))
    assert rows["book:pos:222:AMD"]["at_risk"] == 50.0


def test_a_stop_from_another_account_never_prices_this_one():
    trades = [_stop_trade("T1", "111", "AMD", "LONG", 140.0)]
    rows = _rows(_build(snapshot=qp.fixture_snapshot(FRESH, [_qt_pos("222")]).as_dict(), trades=trades))
    assert rows["book:pos:222:AMD"]["at_risk"] is None


@pytest.mark.parametrize("extra", [
    {"symbol": "AMD18Oct26C150.00", "security_type": "Option"},
    {"symbol": "AMD18Oct26C150.00", "security_type": ""},
    {"symbol": "AMD", "security_type": "STK", "multiplier": 100},
])
def test_option_positions_say_not_computed_never_a_number(extra):
    pos = _qt_pos("222", **{"open_qty": 2.0, "avg_price": 3.0, "market_value": 600.0, **extra})
    trades = [_stop_trade("T1", "222", extra["symbol"], "LONG", 1.5, qty=2, entry=3.0)]
    pack = _build(snapshot=qp.fixture_snapshot(FRESH, [pos]).as_dict(), trades=trades)
    row = next(r for r in pack.rows if r["kind"] == "position")
    assert row["at_risk"] is None
    assert "$ at risk: not computed (option)" in row["text"]


# ---------------------------------------------------------------- 2026-10-01: stale journal opens (book ghosts)
GHOST_REPORT = {
    "checked_at": "2026-09-28T22:04:00",
    "brokers": ["QUESTRADE"],
    "agreed": [{"broker": "QUESTRADE", "account_number": "111", "symbol": "NVDA", "trade_ids": ["T2"]}],
    "mismatched": [
        {"kind": "JOURNAL_OPEN_BROKER_FLAT", "broker": "QUESTRADE", "account_number": "222", "symbol": "AMD",
         "journal_quantity": -100, "broker_quantity": 0, "trade_ids": ["T1"]},
        {"kind": "BROKER_OPEN_JOURNAL_FLAT", "broker": "QUESTRADE", "account_number": "111", "symbol": "QTUM",
         "journal_quantity": 0, "broker_quantity": 20, "trade_ids": []},
    ],
}


def _ghost_build(report, **kwargs):
    import dataclasses

    src = dataclasses.replace(book_pack.fixture_sources(**kwargs), reconciliation=lambda: report)
    return book_pack.build(now=NOW, sources=src)


def test_journal_book_drops_trades_the_broker_reports_flat():
    rows = _rows(_ghost_build(GHOST_REPORT, status={"reason": "no token", "at_utc": FRESH}))
    positions = {k: r for k, r in rows.items() if r["kind"] == "position"}
    assert set(positions) == {"book:pos:111:NVDA", "book:pos:111:QTUM"}
    assert rows["book:stale"]["text"].startswith("Journal shows 1 stale open trade(s) the broker reports flat as of ")
    assert rows["book:stale"]["text"].endswith(": AMD - not positions")
    nvda = positions["book:pos:111:NVDA"]["text"]
    assert "; source: journal, broker agreed as of " in nvda and nvda.endswith(" PT")
    assert "(broker only, not in journal" in positions["book:pos:111:QTUM"]["text"]
    assert positions["book:pos:111:QTUM"]["side"] == "LONG" and positions["book:pos:111:QTUM"]["qty"] == 20
    assert rows["book:side:SHORT"]["count"] == 0, "the stale AMD short is not exposure"
    assert "book:hint:cluster:Semiconductors" not in rows
    assert rows["book:acct:222"]["count"] == 0 and rows["book:acct:111"]["count"] == 2


def test_journal_book_without_a_check_says_it_is_unchecked():
    for report in (None, {}):
        rows = _rows(_ghost_build(report, status={"reason": "no token", "at_utc": FRESH}))
        positions = [r for r in rows.values() if r["kind"] == "position"]
        assert len(positions) == 2
        assert all(r["text"].endswith("; source: journal (not checked against broker)") for r in positions)
        assert "book:stale" not in rows
        assert rows["book:asof"]["text"].endswith("from the journal, not a broker snapshot")


def test_an_unreadable_check_never_breaks_the_book():
    broken = _ghost_build(None, status={"reason": "no token", "at_utc": FRESH})
    import dataclasses

    src = dataclasses.replace(book_pack.fixture_sources(status={"reason": "no token", "at_utc": FRESH}),
                              reconciliation=lambda: (_ for _ in ()).throw(OSError("locked")))
    assert book_pack.build(now=NOW, sources=src).as_text() == broken.as_text()


def test_a_fresh_broker_snapshot_ignores_the_check():
    pack = _ghost_build(GHOST_REPORT, snapshot=qp.fixture_snapshot(FRESH).as_dict())
    assert [(row["id"], row["text"]) for row in pack.rows] == GOLDEN_QUESTRADE


def test_live_book_reads_the_reconciliation_read_only(tmp_path, monkeypatch):
    import json
    import sqlite3

    import project_paths

    db = tmp_path / "trade_journal.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES ('last_reconciliation', ?)", (json.dumps(GHOST_REPORT),))
    conn.commit()
    conn.close()
    before = db.stat().st_mtime_ns
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", db)
    assert book_pack.live_sources().reconciliation() == GHOST_REPORT
    assert db.stat().st_mtime_ns == before
