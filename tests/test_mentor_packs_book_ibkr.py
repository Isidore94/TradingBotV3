"""Mentor P12: /book reads IBKR beside Questrade. Each broker is its own source (fresh snapshot,
else its own journal trades), IBKR ids carry the broker, cash stays per currency with no FX
conversion, exposure and clusters add up across brokers, the registered / no-shorts hint uses
the journal's tax status for IBKR too (never assumed margin), and each broker keeps its own
15-min cache and 1-hour backoff."""

from __future__ import annotations

import dataclasses
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ibkr_positions as ip  # noqa: E402
import questrade_positions as qp  # noqa: E402
from mentor_app import book_jobs  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import book_pack, gate_pack  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = book_pack.FIXTURE_NOW
FRESH = NOW.astimezone(timezone.utc).isoformat(timespec="seconds")


class _Contract:
    def __init__(self, symbol, sec_type="STK", currency="USD", multiplier="", local=""):
        self.symbol, self.secType, self.currency = symbol, sec_type, currency
        self.multiplier, self.localSymbol = multiplier, local


JOURNAL_ACCOUNTS = [*book_pack.FIXTURE_JOURNAL_ACCOUNTS,
                    {"broker": "IBKR", "account_number": "U900", "account_label": "IBKR RRSP",
                     "account_type": "RRSP", "tax_status": "TAX_DEFERRED"}]


def ibkr_snapshot(fetched_utc: str = FRESH) -> qp.BookSnapshot:
    """IBKR: an RRSP (from the journal) holding AVGO and an unknown-type account holding a short and a call."""
    moment = datetime.fromisoformat(fetched_utc)
    return ip.build_snapshot(
        moment, ["U900", "U901"],
        [{"account": "U900", "contract": _Contract("AVGO"), "position": 10, "avg_cost": 1500.0},
         {"account": "U901", "contract": _Contract("TSLA"), "position": -20, "avg_cost": 250.0},
         {"account": "U901", "contract": _Contract("AAPL", "OPT", multiplier="100", local="AAPL  261016C00250000"),
          "position": 1, "avg_cost": 420.0}],
        [("U900", "CashBalance", "1000", "CAD"), ("U900", "CashBalance", "250.5", "USD"),
         ("U900", "CashBalance", "1340", "BASE"), ("U900", "NetLiquidation", "16500", "CAD")],
        JOURNAL_ACCOUNTS)


IBKR_TRADES = [
    {"trade_id": "I1", "broker": "IBKR", "account_number": "U901", "account_label": "U901", "symbol": "TSLA",
     "direction": "SHORT", "quantity_opened": 20, "quantity_closed": 0, "average_entry_price": 250.0,
     "planned_stop": 260.0, "opened_at": "2026-09-28T09:00:00-07:00"},
]


def _build(*, q=None, q_status=None, i=None, i_status=None, trades=None, now=NOW):
    src = dataclasses.replace(
        book_pack.fixture_sources(snapshot=q, status=q_status, accounts=JOURNAL_ACCOUNTS,
                                  trades=[*book_pack.FIXTURE_TRADES, *IBKR_TRADES] if trades is None else trades),
        ibkr_snapshot=lambda: i, ibkr_status=lambda: i_status)
    return book_pack.build(now=now, sources=src)


def _rows(pack):
    return {row["id"]: row for row in pack.rows}


# ---------------------------------------------------------------- the pack
def test_both_brokers_fresh_side_by_side_with_broker_ids():
    pack = _build(q=qp.fixture_snapshot(FRESH).as_dict(), i=ibkr_snapshot().as_dict())
    rows = _rows(pack)
    assert rows["book:source"]["source"] == "brokers"
    assert rows["book:source"]["text"] == ("Source: Questrade positions at Tue 09-29 07:05 PT; "
                                           "IBKR positions at Tue 09-29 07:05 PT")
    assert {"book:acct:111", "book:acct:222", "book:acct:IBKR:U900", "book:acct:IBKR:U901"} <= set(rows)
    assert {"book:pos:111:NVDA", "book:pos:222:AMD", "book:pos:IBKR:U900:AVGO", "book:pos:IBKR:U901:TSLA",
            "book:pos:IBKR:U901:AAPL-261016C00250000"} <= set(rows)
    assert rows["book:acct:IBKR:U900"]["text"] == (
        "Account IBKR RRSP (RRSP): registered, tax-deferred, no shorts; 1 open position(s); "
        "cash CAD 1,000.00, USD 250.50 (no FX conversion); net liquidation CAD 16,500.00")
    assert rows["book:acct:IBKR:U901"]["text"] == (
        "Account IBKR U901 (type unknown): account type unknown; 2 open position(s); cash unknown"), \
        "no journal row: never assumed margin"
    assert rows["book:pos:IBKR:U901:AAPL-261016C00250000"]["text"].endswith("$ at risk: not computed (option)")
    assert rows["book:pos:IBKR:U900:AVGO"]["text"] == (
        "LONG AVGO 10 @ 1,500.00 in IBKR RRSP; market value unknown; stop unknown")


def test_exposure_and_clusters_add_up_across_brokers():
    rows = _rows(_build(q=qp.fixture_snapshot(FRESH).as_dict(), i=ibkr_snapshot().as_dict()))
    assert rows["book:side:LONG"]["count"] == 3 and rows["book:side:SHORT"]["count"] == 2
    assert rows["book:industry:Semiconductors"]["names"] == ["AMD", "AVGO", "NVDA"]
    assert rows["book:hint:cluster:Semiconductors"]["text"] == "3 open names in Semiconductors (AMD, AVGO, NVDA)"
    assert rows["book:hint:no_shorts_registered"]["text"] == (
        "Short candidates cannot go in TFSA 111, IBKR RRSP (registered accounts hold no shorts)")


def test_ibkr_down_keeps_questrade_and_labels_ibkr_journal_with_the_backoff():
    failed = (NOW - timedelta(minutes=8)).astimezone(timezone.utc).isoformat(timespec="seconds")
    rows = _rows(_build(q=qp.fixture_snapshot(FRESH).as_dict(), i_status={"reason": "TWS not running",
                                                                            "at_utc": failed}))
    assert rows["book:source"]["source"] == "mixed"
    assert rows["book:source"]["text"] == (
        "Source: Questrade positions at Tue 09-29 07:05 PT; journal open trades for IBKR "
        "(IBKR unavailable: TWS not running (backing off 52 min))")
    assert rows["book:pos:IBKR:U901:TSLA"]["text"] == (
        "SHORT TSLA 20 @ 250.00 in IBKR U901; market value unknown; $ at risk $200.00 (stop 260); "
        "source: journal (not checked against broker)")
    assert "cash unknown (the journal has no cash)" in rows["book:acct:IBKR:U900"]["text"]
    assert rows["book:pos:222:AMD"]["market_value"] == -14800.0, "Questrade still from its snapshot"
    assert not any(k.startswith("book:pos:IBKR") and rows[k]["market_value"] is not None for k in rows
                   if rows[k].get("kind") == "position")


def test_questrade_down_keeps_ibkr_and_the_questrade_journal_is_per_broker():
    rows = _rows(_build(q_status={"reason": "no token", "at_utc": FRESH}, i=ibkr_snapshot().as_dict()))
    assert rows["book:source"]["text"] == (
        "Source: IBKR positions at Tue 09-29 07:05 PT; journal open trades for Questrade "
        "(Questrade unavailable: no token)")
    assert rows["book:pos:222:AMD"]["market_value"] is None, "Questrade from its journal"
    assert "book:pos:IBKR:U901:TSLA" in rows and rows["book:pos:IBKR:U901:TSLA"]["avg_price"] == 250.0
    tsla = [k for k in rows if k.endswith(":TSLA")]
    assert tsla == ["book:pos:IBKR:U901:TSLA"], "the IBKR journal trade is not shown beside the IBKR snapshot"


def test_neither_fresh_is_the_journal_with_both_reasons():
    stale = (NOW - timedelta(minutes=20)).astimezone(timezone.utc).isoformat(timespec="seconds")
    rows = _rows(_build(q_status={"reason": "no token", "at_utc": FRESH}, i=ibkr_snapshot(stale).as_dict()))
    assert rows["book:source"]["source"] == "journal"
    assert rows["book:source"]["text"] == ("Source: journal open trades (Questrade unavailable: no token; "
                                           "IBKR unavailable: last snapshot 20 min old)")
    assert {"book:pos:222:AMD", "book:pos:IBKR:U901:TSLA"} <= set(rows)


def test_ibkr_never_read_is_the_p8_book_unchanged():
    golden = [(row["id"], row["text"]) for row in book_pack.build(
        now=NOW, sources=book_pack.fixture_sources(snapshot=qp.fixture_snapshot(FRESH).as_dict())).rows]
    pack = _build(q=qp.fixture_snapshot(FRESH).as_dict(), trades=book_pack.FIXTURE_TRADES)
    assert [(row["id"], row["text"]) for row in pack.rows] == golden
    assert not any("IBKR" in row["id"] for row in pack.rows)


def test_read_open_trades_carries_the_broker(tmp_path):
    path = tmp_path / "journal.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (trade_id TEXT, broker TEXT, account_number TEXT, account_label TEXT, "
                 "symbol TEXT, direction TEXT, status TEXT, quantity_opened REAL, quantity_closed REAL, "
                 "average_entry_price REAL, opened_at TEXT)")
    conn.execute("INSERT INTO trades VALUES ('I1','IBKR','U901','','TSLA','SHORT','OPEN',20,0,250,'2026-09-28')")
    conn.commit()
    conn.close()
    assert gate_pack.read_open_trades(path)[0]["broker"] == "IBKR"


# ---------------------------------------------------------------- the gate
def _gate(tmp_path, **book):
    world = gate_pack.fixture_sources(tmp_path / "world", trades=[*book_pack.FIXTURE_TRADES, *IBKR_TRADES])
    src = dataclasses.replace(world, accounts=lambda: JOURNAL_ACCOUNTS,
                              book_snapshot=lambda: book.get("q"), book_status=lambda: book.get("q_status"),
                              ibkr_book_snapshot=lambda: book.get("i"), ibkr_book_status=lambda: book.get("i_status"))
    return {row["id"]: row for row in gate_pack.build("SHORT", "NVDA", 400, 3.20, 3.05, now=NOW, sources=src).rows}


def test_the_gate_embeds_both_brokers(tmp_path):
    rows = _gate(tmp_path, q=qp.fixture_snapshot(FRESH).as_dict(), i=ibkr_snapshot().as_dict())
    assert {"gate:NVDA:book:pos:222:AMD", "gate:NVDA:book:pos:IBKR:U901:TSLA",
            "gate:NVDA:book:acct:IBKR:U900"} <= set(rows)
    assert rows["gate:NVDA:book:industry"]["text"].startswith("Open book (Questrade + IBKR): 5 open position(s); ")
    assert rows["gate:NVDA:book:industry"]["open_count"] == 5
    assert "IBKR RRSP" in rows["gate:NVDA:book:hint:short_account"]["text"]


def test_the_gate_journal_mode_ids_are_unchanged(tmp_path):
    rows = _gate(tmp_path, q_status={"reason": "no token", "at_utc": FRESH},
                 i_status={"reason": "TWS not running", "at_utc": FRESH})
    assert {"gate:NVDA:book:T1", "gate:NVDA:book:T2", "gate:NVDA:book:I1"} <= set(rows)
    assert not any(":book:pos:" in row_id for row_id in rows)


# ---------------------------------------------------------------- the fetch policy, per broker
class _Fetch:
    def __init__(self, snap_of, fail=""):
        self.calls, self.fail, self.snap_of = [], fail, snap_of

    def __call__(self, now):
        self.calls.append(now)
        if self.fail:
            return None, self.fail
        return self.snap_of(now.astimezone(timezone.utc).isoformat(timespec="seconds")), ""


@pytest.fixture
def store(tmp_path):
    return MentorChatStore(tmp_path / "mentor_chat.sqlite3")


def test_each_broker_has_its_own_cache_and_backoff(store):
    t0 = datetime(2026, 9, 29, 7, 0, tzinfo=PT)
    q, i = _Fetch(qp.fixture_snapshot), _Fetch(ibkr_snapshot, fail="TWS not running")
    out = book_jobs.ensure_book(store, t0, fetch=q, ibkr_fetch=i)
    assert out["fetched"] and out["ibkr"] == {"fetched": False, "reason": "TWS not running", "failed": True}
    assert book_jobs.stored_status(store) is None and book_jobs.stored_status(store, "IBKR")["reason"] == \
        "TWS not running"
    out = book_jobs.ensure_book(store, t0 + timedelta(minutes=20), fetch=q, ibkr_fetch=i)
    assert out["fetched"] and "backing off" in out["ibkr"]["reason"] and len(i.calls) == 1
    i.fail = ""
    out = book_jobs.ensure_book(store, t0 + timedelta(minutes=61), fetch=q, ibkr_fetch=i)
    assert out["ibkr"]["fetched"] and book_jobs.stored_snapshot(store, "IBKR").broker == "IBKR"
    assert book_jobs.stored_snapshot(store).broker == "QUESTRADE"
    out = book_jobs.ensure_book(store, t0 + timedelta(minutes=70), fetch=q, ibkr_fetch=i)
    assert out["ibkr"]["reason"] == "fresh" and len(i.calls) == 2 and len(q.calls) == 3
    assert book_jobs.fetch_note(out) == ""


def test_a_questrade_failure_never_stops_the_ibkr_read(store):
    t0 = datetime(2026, 9, 29, 7, 0, tzinfo=PT)
    q, i = _Fetch(qp.fixture_snapshot, fail="RuntimeError: 400"), _Fetch(ibkr_snapshot)
    out = book_jobs.ensure_book(store, t0, fetch=q, ibkr_fetch=i)
    assert out["failed"] and out["ibkr"]["fetched"]
    assert book_jobs.fetch_note(out) == "not read from Questrade now: RuntimeError: 400"


def test_the_desk_closed_reads_neither(store):
    q, i = _Fetch(qp.fixture_snapshot), _Fetch(ibkr_snapshot)
    out = book_jobs.ensure_book(store, NOW, fetch=q, ibkr_fetch=i, desk_closed=lambda: True)
    assert q.calls == [] and i.calls == [] and out["ibkr"]["reason"] == "the desk is closed"


def test_the_store_sources_carry_ibkr_into_the_card(store):
    t0 = datetime(2026, 9, 29, 7, 0, tzinfo=PT)
    book_jobs.ensure_book(store, t0, fetch=_Fetch(qp.fixture_snapshot), ibkr_fetch=_Fetch(ibkr_snapshot))
    src = book_jobs.store_sources(store, dataclasses.replace(book_pack.fixture_sources(),
                                                             accounts=lambda: JOURNAL_ACCOUNTS))
    text = book_jobs.card_markdown(book_pack.build(now=t0, sources=src))
    assert "IBKR positions at Tue 09-29 07:00 PT" in text and "`[book:acct:IBKR:U900]`" in text
    assert "no FX conversion" in text and text.rstrip().endswith(book_jobs.FOOTER)


def test_the_window_reads_both_on_the_news_thread(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mentor_app import settings
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    t0 = datetime(2026, 9, 29, 7, 0, tzinfo=PT)
    q, i = _Fetch(qp.fixture_snapshot), _Fetch(ibkr_snapshot)
    base = dataclasses.replace(book_pack.fixture_sources(), accounts=lambda: JOURNAL_ACCOUNTS)
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), stream_post=lambda *a, **k: [],
                       post=lambda *a, **k: {}, now=lambda: t0, mentor_enabled=False, desk_probe=lambda: False,
                       book_fetch=q, ibkr_fetch=i, book_sources=lambda: base)
    try:
        win.send("/book")
        assert win.queue.pending() == [] and win.news_queue.pending() == ["book"]
        while win.news_queue.run_one() or win.queue.run_one():
            pass
        text = win.transcript.toPlainText()
        assert len(q.calls) == 1 and len(i.calls) == 1
        assert "IBKR positions at" in text and "SHORT TSLA 20" in text and "Questrade positions at" in text
    finally:
        win.shutdown()
        win.deleteLater()


def test_the_window_default_ibkr_reader_is_the_real_one():
    import inspect

    from mentor_app import window

    assert "ibkr_positions.fetch_book(now)" in inspect.getsource(window._ibkr_fetch_book)


# ---------------------------------------------------------------- contract kinds (review advisory)
@pytest.mark.parametrize("extra,expected", [
    ({"security_type": "OPT", "multiplier": 100}, "not computed (option)"),
    ({"security_type": "WAR", "multiplier": 1}, "not computed (option)"),
    ({"security_type": "STK", "symbol": "AAPL 261016C00250000"}, "not computed (option)"),
    ({"security_type": "FUT", "multiplier": 50}, "not computed (future)"),
    ({"security_type": "STK", "multiplier": 1}, "$ at risk $300.00 (stop 257)"),
])
def test_only_options_and_futures_skip_the_risk_number(extra, expected):
    pos = {"account_number": "222", "symbol": "TSLA", "side": "LONG", "open_qty": 100.0, "avg_price": 260.0,
           "market_value": 26000.0, **extra}
    trade = {"trade_id": "T9", "account_number": "222", "symbol": pos["symbol"], "direction": "LONG",
             "quantity_opened": 100, "quantity_closed": 0, "average_entry_price": 260.0, "planned_stop": 257.0,
             "opened_at": "2026-09-28T09:00:00-07:00"}
    rows = _build(q=qp.fixture_snapshot(FRESH, [pos]).as_dict(), trades=[trade]).rows
    row = next(r for r in rows if r["kind"] == "position")
    assert row["text"].endswith(expected), row["text"]
