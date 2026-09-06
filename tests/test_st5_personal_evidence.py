"""Packet ST5 - personal evidence usable without inventing it.

Written BEFORE the fix (tester-first, `docs/AGENT_TEAM.md`). Every test below
fails on `main` at 84ee24d6; the builder's job is to make them pass without
weakening one of them.

WHAT THE JOURNAL REALLY LOOKS LIKE (read-only `immutable=1` query of
`C:\\TradingBotData\\data\\runtime\\trade_journal.sqlite3`, 2026-09-06)

* status: CLOSED 165, CLOSED_PARTIAL 7, OPEN 32.
* `trades.security_type`: OPT 89, UNKNOWN 86, STK 27, **BAG 1**, CASH 1.
* `trade_annotations.tag_status`: confirmed **1**, provisional 26,
  needs_review 145. `planned_risk` non-null: **0**.
* closed by (instrument, direction): OPT/LONG 32 (17 winners), OPT/SHORT 53 (39),
  STK/LONG 18 (10), STK/SHORT 7 (5), UNKNOWN/LONG 44 (22), UNKNOWN/SHORT 11 (5).
* An OPT trade carries 1..9 rows in ``trade_legs`` (39 trades have exactly 2).
  **A leg row is a FILL, not a contract leg** - every closed option trade has at
  least two of them. So "more than one option leg" can only mean *more than one
  distinct option CONTRACT among the legs*, and the fixtures below are built that
  way. `trades.symbol` is the OCC identity for an OPT row
  (`AA260522P00062000`); the right/strike/expiry also ride in
  `raw_executions.raw_json["option"]`, which is what
  `journal_statement_import._execution_from_row` writes.

THE API THESE TESTS PIN (the packet names each one; the exact spelling is fixed
here so the builder is not guessing)

1. ``preference_trade_outcomes.TRADE_WINDOW_SESSIONS`` - int, 10, and
   ``match_trade`` walks the exchange calendar forward that many SESSIONS.
2. ``preference_trade_outcomes.trade_level_summary(rows) -> dict`` with
   ``n_statements_matched``, ``n_trades_matched``, ``net_pnl``,
   ``duplicate_statement_rows``, ``statements_per_trade``,
   ``planned_risk_recorded``, ``planned_risk_note``; and
   ``run_preference_trade_outcomes`` carrying ``n_trades_matched`` /
   ``n_statements_matched`` in its result (that pair is live gate #79).
3. ``scripts/journal_exposure.py``: ``classify_exposure(trade) -> Exposure`` with
   ``.instrument``, ``.ownership_direction``, ``.market_bias``, ``.structure``,
   ``.certainty``. The trade it is handed is a real ``list_trades()`` row with a
   ``legs`` key: ``list_trade_legs()`` rows, each carrying the ``raw_json`` of the
   execution behind it (``_with_legs`` below composes exactly that, from the
   store, through one SELECT). ``list_trade_legs`` does not select ``raw_json``
   today - widening it is one way to close the seam, and the fixture stays valid
   either way.
4. ``journal_analytics.personal_evidence_summary(trades) -> dict`` with the four
   never-pooled populations ``complete`` / ``partly_closed`` / ``open_exposure``
   / ``uncertain`` (each ``{"n", "winners", "trade_ids", "by_market_bias"}``),
   plus ``headline``, ``best_setup`` and ``coverage``. The annotation columns
   (``tag_status``, ``setup_tags``, ``planned_risk``) already ride on every
   ``list_trades()`` row, which is why one argument is enough.

NOTHING HERE TOUCHES A LIVE STORE. Every journal is a temp SQLite file built
through ``JournalStore.upsert_executions`` + ``rebuild_trades`` -- the real
assembly path -- and every report is written to ``tmp_path``.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# fixture plumbing - real executions through the real assembler
# ---------------------------------------------------------------------------
def _execution(
    uid,
    symbol,
    security_type,
    side,
    quantity,
    price,
    timestamp,
    *,
    account="111",
    currency="CAD",
    option=None,
):
    """One broker fill in the shape ``upsert_executions`` actually stores.

    ``option`` mirrors ``journal_statement_import``'s ``raw_json["option"]``
    payload - right / underlying / expiry / strike - because that is where the
    contract lives for a Questrade option row.
    """
    raw: dict = {}
    if option is not None:
        raw["option"] = option
        raw["multiplier"] = 100.0
    return {
        "execution_uid": uid,
        "broker": "QUESTRADE",
        "account_number": account,
        "account_label": "TFSA",
        "account_type": "TFSA",
        "symbol": symbol,
        "security_type": security_type,
        "currency": currency,
        "side": side,
        "quantity": quantity,
        "price": price,
        "timestamp": timestamp,
        "trade_date": timestamp[:10],
        "commission": 0.0,
        "fees": 0.0,
        "gross_amount": None,
        "net_amount": None,
        "order_id": "",
        "exchange_exec_id": "",
        "raw_json": json.dumps(raw),
    }


def _build_store(tmp_path, executions, *, name="trade_journal.sqlite3"):
    from journal_store import JournalStore

    store = JournalStore(tmp_path / name)
    store.upsert_executions(executions)
    store.rebuild_trades(refresh_tags=False)
    # CAD-native rows book at 1.0, so `net_pnl_cad` is populated without any
    # fetched rate. `_canonical_r` and every CAD total read that column.
    store.book_currency_values()
    return store


def _raw_json_by_uid(store) -> dict[str, str]:
    with store.connection() as conn:
        return {
            str(row["execution_uid"]): str(row["raw_json"] or "{}")
            for row in conn.execute("SELECT execution_uid, raw_json FROM raw_executions")
        }


def _with_legs(store, trade) -> dict:
    """A ``list_trades()`` row plus its legs, each carrying its raw payload.

    This is what a caller composes; nothing here is hand-written.
    """
    payloads = _raw_json_by_uid(store)
    legs = []
    for leg in store.list_trade_legs(trade["trade_id"]):
        leg = dict(leg)
        leg.setdefault("raw_json", payloads.get(str(leg.get("execution_uid")), "{}"))
        legs.append(leg)
    return dict(trade, legs=legs)


def _trades_with_legs(store) -> list[dict]:
    return [_with_legs(store, trade) for trade in store.list_trades()]


def _by_symbol(trades, symbol) -> dict:
    matches = [trade for trade in trades if trade["symbol"] == symbol]
    assert len(matches) == 1, f"expected exactly one {symbol} trade, got {len(matches)}"
    return matches[0]


def _feedback_file(tmp_path, rows, name="pick_feedback.jsonl") -> Path:
    target = tmp_path / name
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return target


def _like(trade_date, symbol="AAPL", side="long"):
    """One `pick_feedback` like row, the shape `_feedback_statements` reads."""
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "side": side,
        "verdict": "like",
        "origin": "strength_board",
    }


def _statements(tmp_path, feedback_rows, *, since, until):
    """Statements through `collect_statements` - the module's own reader."""
    import preference_trade_outcomes as pto

    empty_annotations = tmp_path / "trader_annotations.jsonl"
    empty_annotations.write_text("", encoding="utf-8")
    empty_favorites = tmp_path / "swing_favorites.jsonl"
    empty_favorites.write_text("", encoding="utf-8")
    return pto.collect_statements(
        since=since,
        until=until,
        annotations_path=empty_annotations,
        feedback_path=_feedback_file(tmp_path, feedback_rows),
        favorites_path=empty_favorites,
    )


# ---------------------------------------------------------------------------
# The six-trade population fixture (tests 5, 6 and 7)
# ---------------------------------------------------------------------------
def _population_executions():
    """Six positions covering every population the summary must separate.

    * AAPL  STK  long, closed, +250      -> complete, bullish
    * SPY   OPT  long PUT, closed, +200  -> complete, BEARISH (a bought put)
    * DRAM  OPT  short PUT, closed, +100 -> complete, bullish_or_neutral
    * MSFT  STK  long, half exited       -> partly_closed
    * NVDA  STK  long, never exited      -> open_exposure
    * TSLA  UNKNOWN instrument, closed   -> uncertain
    * CVNA  BAG holding TWO option contracts (a 60 put and a 70 call), closed
            -> uncertain / multi_leg
    """
    return [
        _execution("E-AAPL-1", "AAPL", "STK", "BUY", 100, 10.0, "2026-09-11T09:31:00-04:00"),
        _execution("E-AAPL-2", "AAPL", "STK", "SELL", 100, 12.5, "2026-09-11T11:00:00-04:00"),
        _execution(
            "E-SPY-1", "SPY260918P00600000", "OPT", "BUY", 1, 3.0, "2026-09-09T09:40:00-04:00",
            option={"right": "PUT", "underlying": "SPY", "expiry": "260918", "strike": 600.0},
        ),
        _execution(
            "E-SPY-2", "SPY260918P00600000", "OPT", "SELL", 1, 5.0, "2026-09-10T09:40:00-04:00",
            option={"right": "PUT", "underlying": "SPY", "expiry": "260918", "strike": 600.0},
        ),
        _execution(
            "E-DRAM-1", "DRAM260918P00055000", "OPT", "SELL", 1, 2.0, "2026-09-09T09:45:00-04:00",
            option={"right": "PUT", "underlying": "DRAM", "expiry": "260918", "strike": 55.0},
        ),
        _execution(
            "E-DRAM-2", "DRAM260918P00055000", "OPT", "BUY", 1, 1.0, "2026-09-10T09:45:00-04:00",
            option={"right": "PUT", "underlying": "DRAM", "expiry": "260918", "strike": 55.0},
        ),
        _execution("E-MSFT-1", "MSFT", "STK", "BUY", 100, 20.0, "2026-09-09T10:00:00-04:00"),
        _execution("E-MSFT-2", "MSFT", "STK", "SELL", 40, 22.0, "2026-09-10T10:00:00-04:00"),
        _execution("E-NVDA-1", "NVDA", "STK", "BUY", 50, 100.0, "2026-09-09T10:05:00-04:00"),
        _execution("E-TSLA-1", "TSLA", "UNKNOWN", "BUY", 10, 100.0, "2026-09-09T10:10:00-04:00"),
        _execution("E-TSLA-2", "TSLA", "UNKNOWN", "SELL", 10, 90.0, "2026-09-10T10:10:00-04:00"),
        _execution(
            "E-CVNA-1", "CVNA", "BAG", "BUY", 1, 2.0, "2026-09-09T10:15:00-04:00",
            option={"right": "PUT", "underlying": "CVNA", "expiry": "260918", "strike": 60.0},
        ),
        _execution(
            "E-CVNA-2", "CVNA", "BAG", "BUY", 1, 2.5, "2026-09-09T10:15:01-04:00",
            option={"right": "CALL", "underlying": "CVNA", "expiry": "260918", "strike": 70.0},
        ),
        _execution(
            "E-CVNA-3", "CVNA", "BAG", "SELL", 1, 3.0, "2026-09-10T10:15:00-04:00",
            option={"right": "PUT", "underlying": "CVNA", "expiry": "260918", "strike": 60.0},
        ),
        _execution(
            "E-CVNA-4", "CVNA", "BAG", "SELL", 1, 1.0, "2026-09-10T10:15:01-04:00",
            option={"right": "CALL", "underlying": "CVNA", "expiry": "260918", "strike": 70.0},
        ),
    ]


@pytest.fixture
def population_store(tmp_path):
    return _build_store(tmp_path, _population_executions())


# ---------------------------------------------------------------------------
# ST5.1 - the window is SESSIONS, and Labor Day is not one
# ---------------------------------------------------------------------------
def test_the_statement_window_walks_ten_sessions_and_skips_labor_day(tmp_path):
    """A like on Friday 2026-09-04 reaches Monday 2026-09-21, not 2026-09-14.

    Ten sessions after 2026-09-04 is 2026-09-21 because 2026-09-07 is Labor Day
    (`market_calendar.trading_days_between(2026-09-04, 2026-09-21) == 10`).
    Ten CALENDAR days is 2026-09-14, which drops five real sessions on the floor
    and with them the trade the trader actually took on 2026-09-18.
    """
    import market_calendar
    import preference_trade_outcomes as pto

    # The premise, stated as arithmetic rather than trusted.
    assert market_calendar.trading_days_between(date(2026, 9, 4), date(2026, 9, 21)) == 10
    assert market_calendar.trading_days_between(date(2026, 9, 4), date(2026, 9, 22)) == 11
    assert market_calendar.is_session(date(2026, 9, 7)) is False

    assert pto.TRADE_WINDOW_SESSIONS == 10

    store = _build_store(
        tmp_path,
        [
            # Inside 10 sessions, outside 10 calendar days.
            _execution("E-IN-1", "AAPL", "STK", "BUY", 100, 10.0, "2026-09-18T09:31:00-04:00", account="111"),
            _execution("E-IN-2", "AAPL", "STK", "SELL", 100, 11.0, "2026-09-18T15:31:00-04:00", account="111"),
            # The eleventh session. A different decision, and it must stay out.
            _execution("E-OUT-1", "AAPL", "STK", "BUY", 100, 10.0, "2026-09-22T09:31:00-04:00", account="222"),
            _execution("E-OUT-2", "AAPL", "STK", "SELL", 100, 11.0, "2026-09-22T15:31:00-04:00", account="222"),
        ],
    )
    trades = store.list_trades()
    inside = [t for t in trades if str(t["opened_at"]).startswith("2026-09-18")]
    outside = [t for t in trades if str(t["opened_at"]).startswith("2026-09-22")]
    assert len(inside) == 1 and len(outside) == 1

    statements = _statements(
        tmp_path,
        [_like("2026-09-04")],
        since=date(2026, 8, 1),
        until=date(2026, 9, 30),
    )
    assert len(statements) == 1
    assert statements[0]["session_date"] == date(2026, 9, 4)

    matched = pto.match_trade(statements[0], trades)
    assert matched["trade"] is not None
    assert matched["trade"]["trade_id"] == inside[0]["trade_id"]
    # Confidence labels are PRESERVED by this packet, so the basis is the one
    # that was already spelled for an in-window same-side match.
    assert matched["basis"] == "symbol+side+window"
    assert matched["confidence"] == pytest.approx(0.7)

    # The eleventh session on its own is no match at all.
    beyond = pto.match_trade(statements[0], outside)
    assert beyond["trade"] is None
    assert beyond["basis"] == "no match"
    assert beyond["confidence"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ST5.2 - three statements, one trade, one P&L
# ---------------------------------------------------------------------------
def test_three_statements_about_one_trade_count_as_one_trade(tmp_path, monkeypatch):
    """Live: 13 `traded=yes` rows over 10 distinct trade ids. Same defect, small.

    Every statement row is KEPT (the skip is the interesting row and the trade
    row is its evidence), but the P&L is +250.00 once, never 3 x 250.00.
    """
    import preference_trade_outcomes as pto

    store = _build_store(
        tmp_path,
        [
            _execution("E-1", "AAPL", "STK", "BUY", 100, 10.0, "2026-09-11T09:31:00-04:00"),
            _execution("E-2", "AAPL", "STK", "SELL", 100, 12.5, "2026-09-11T15:31:00-04:00"),
        ],
    )
    trades = store.list_trades()
    assert len(trades) == 1
    assert trades[0]["net_pnl_cad"] == pytest.approx(250.0)

    statements = _statements(
        tmp_path,
        [_like("2026-09-08"), _like("2026-09-09"), _like("2026-09-10")],
        since=date(2026, 8, 1),
        until=date(2026, 9, 30),
    )
    assert len(statements) == 3

    rows = pto.build_rows(statements, trades, grades={}, now=datetime(2026, 9, 25, 20, 0, 0))

    # One row per statement, unchanged.
    assert len(rows) == 3
    assert [row["traded"] for row in rows] == ["yes", "yes", "yes"]
    assert len({row["trade_id"] for row in rows}) == 1

    summary = pto.trade_level_summary(rows)
    assert summary["n_statements_matched"] == 3
    assert summary["n_trades_matched"] == 1
    # THE NUMBER. A per-statement sum would say 750.00.
    assert summary["net_pnl"] == pytest.approx(250.0)
    assert summary["duplicate_statement_rows"] == 2
    assert summary["statements_per_trade"][trades[0]["trade_id"]] == 3

    # Live gate #79 reads these two off the nightly slot's own result.
    monkeypatch.setattr(
        "swing_favorites.load_rows", lambda *args, **kwargs: [], raising=False
    )
    import project_paths

    monkeypatch.setattr(
        project_paths, "PICK_FEEDBACK_FILE", _feedback_file(
            tmp_path, [_like("2026-09-08"), _like("2026-09-09"), _like("2026-09-10")], name="nightly.jsonl"
        )
    )
    monkeypatch.setattr(
        project_paths, "TRADER_ANNOTATIONS_FILE", tmp_path / "absent_annotations.jsonl"
    )
    result = pto.run_preference_trade_outcomes(
        now=datetime(2026, 9, 25, 20, 0, 0),
        window_days=45,
        report_path=tmp_path / "report.csv",
        trades=trades,
    )
    assert result["status"] == "ok"
    assert result["rows"] == 3
    assert result["n_statements_matched"] == 3
    assert result["n_trades_matched"] == 1


# ---------------------------------------------------------------------------
# ST5.4 - a provisional tag is not a setup the trader stands behind
# ---------------------------------------------------------------------------
def test_a_provisional_tag_never_names_a_best_personal_setup(tmp_path):
    """Live: 26 provisional, 1 confirmed. Neither is 30, so nothing is "best"."""
    import evidence_stats
    import journal_analytics

    store = _build_store(
        tmp_path,
        [
            _execution("E-A1", "AAPL", "STK", "BUY", 100, 10.0, "2026-09-09T09:31:00-04:00"),
            _execution("E-A2", "AAPL", "STK", "SELL", 100, 12.0, "2026-09-09T15:31:00-04:00"),
            _execution("E-M1", "MSFT", "STK", "BUY", 100, 20.0, "2026-09-10T09:31:00-04:00"),
            _execution("E-M2", "MSFT", "STK", "SELL", 100, 19.0, "2026-09-10T15:31:00-04:00"),
            _execution("E-N1", "NVDA", "STK", "BUY", 100, 30.0, "2026-09-11T09:31:00-04:00"),
            _execution("E-N2", "NVDA", "STK", "SELL", 100, 31.0, "2026-09-11T15:31:00-04:00"),
        ],
    )
    trade_ids = [trade["trade_id"] for trade in store.list_trades()]
    assert len(trade_ids) == 3
    # The machine lane, through the store's own authorized writer.
    assert store.apply_provisional_tags(trade_ids[0], "avwap-reclaim") is True
    assert store.apply_provisional_tags(trade_ids[1], "avwap-reclaim") is True

    trades = store.list_trades()
    assert sum(1 for t in trades if t["tag_status"] == "provisional") == 2

    # Already true today and it must STAY true: the machine's guess is not a
    # bucket of "my setups".
    groups = journal_analytics.build_analytics_summary(trades)["groups"]
    assert "avwap-reclaim" not in {row["label"] for row in groups["my setups"]}
    assert "avwap-reclaim" in {row["label"] for row in groups["provisional setups"]}

    summary = journal_analytics.personal_evidence_summary(trades)
    assert summary["best_setup"] is None
    assert summary["headline"] == (
        "No confirmed setup tags (2 provisional awaiting review) - "
        "no personal setup can be called best."
    )

    # One confirmed tag is still far below the reportable floor, so still no
    # "best" - and the refusal must not silently disappear at n == 1.
    assert store.confirm_tags(trade_ids[0]) is True
    below_floor = journal_analytics.personal_evidence_summary(store.list_trades())
    assert evidence_stats.MIN_REPORTABLE_N == 30
    assert below_floor["coverage"]["confirmed"] == 1
    assert below_floor["best_setup"] is None
    assert "no personal setup can be called best" in below_floor["headline"]


# ---------------------------------------------------------------------------
# ST5.2 / ST5.5 - a missing plan is a blank and a count, never a reconstruction
# ---------------------------------------------------------------------------
def test_missing_planned_risk_is_blank_counted_and_never_written(tmp_path, monkeypatch):
    """Live: `planned_risk` non-null on 0 of 204 trades and `journal_r` blank on
    all 538 report rows. Nothing in this packet's read path may fix that by
    computing a risk from an outcome - `save_risk_fields` is the ONLY writer and
    it is a trader action in the Trades tab.
    """
    import journal_analytics
    import preference_trade_outcomes as pto
    from journal_store import JournalStore

    store = _build_store(
        tmp_path,
        [
            _execution("E-A1", "AAPL", "STK", "BUY", 100, 10.0, "2026-09-09T09:31:00-04:00"),
            _execution("E-A2", "AAPL", "STK", "SELL", 100, 12.0, "2026-09-09T15:31:00-04:00"),
            _execution("E-M1", "MSFT", "STK", "BUY", 100, 20.0, "2026-09-10T09:31:00-04:00"),
            _execution("E-M2", "MSFT", "STK", "SELL", 100, 19.0, "2026-09-10T15:31:00-04:00"),
            _execution("E-N1", "NVDA", "STK", "BUY", 100, 30.0, "2026-09-11T09:31:00-04:00"),
            _execution("E-N2", "NVDA", "STK", "SELL", 100, 31.0, "2026-09-11T15:31:00-04:00"),
        ],
    )
    ids = [trade["trade_id"] for trade in store.list_trades()]
    assert store.apply_provisional_tags(ids[0], "avwap-reclaim") is True
    assert store.apply_provisional_tags(ids[1], "earnings-gap") is True

    calls: list[tuple] = []

    def _spy(self, trade_id, **kwargs):
        calls.append((trade_id, kwargs))
        raise AssertionError(
            "planned_risk was written by a read path; it is the trader's own "
            "number and may never be reconstructed from an outcome"
        )

    monkeypatch.setattr(JournalStore, "save_risk_fields", _spy)

    trades = store.list_trades()
    assert all(trade["planned_risk"] is None for trade in trades)

    statements = _statements(
        tmp_path,
        [_like("2026-09-08")],
        since=date(2026, 8, 1),
        until=date(2026, 9, 30),
    )
    rows = pto.build_rows(statements, trades, grades={}, now=datetime(2026, 9, 25, 20, 0, 0))
    report = tmp_path / "preference_trade_outcomes.csv"
    assert pto.write_rows(rows, report) is True

    with report.open("r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["traded"] == "yes"
    # Blank, not zero. A zero R is a statement about a trade; this is silence.
    assert written[0]["journal_r"] == ""

    summary = pto.trade_level_summary(rows)
    assert summary["n_trades_matched"] == 1
    assert summary["planned_risk_recorded"] == 0
    assert summary["planned_risk_note"] == "planned risk recorded on 0 of 1 matched trades"

    coverage = journal_analytics.personal_evidence_summary(trades)["coverage"]
    assert coverage["confirmed"] == 0
    assert coverage["closed"] == 3
    assert coverage["provisional"] == 2
    assert coverage["planned_risk"] == 0
    assert coverage["line"] == (
        "Confirmed tags: 0 of 3 closed trades. Provisional awaiting review: 2. "
        "Planned risk recorded: 0 of 3."
    )

    assert calls == []
    with store.connection() as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM trade_annotations WHERE planned_risk IS NOT NULL"
        ).fetchone()[0]
    assert remaining == 0


# ---------------------------------------------------------------------------
# ST5.3 - an unknown instrument stays unknown
# ---------------------------------------------------------------------------
def test_an_unknown_instrument_is_uncertain_and_never_pooled_into_complete(population_store):
    """86 of the 204 live trades carry `security_type = 'UNKNOWN'`; 55 of those
    are CLOSED. Counting them under "complete" would put a quarter of the
    journal's P&L behind a noun the data does not support.
    """
    import journal_analytics
    from journal_exposure import classify_exposure

    trades = _trades_with_legs(population_store)
    unknown = _by_symbol(trades, "TSLA")
    assert unknown["security_type"] == "UNKNOWN"
    assert unknown["status"] == "CLOSED"
    assert unknown["direction"] == "LONG"

    exposure = classify_exposure(unknown)
    assert exposure.instrument == "UNKNOWN"
    # LONG is what was OWNED. It is not a claim about the market.
    assert exposure.market_bias == "unknown"

    summary = journal_analytics.personal_evidence_summary(population_store.list_trades())
    assert unknown["trade_id"] in summary["uncertain"]["trade_ids"]
    assert unknown["trade_id"] not in summary["complete"]["trade_ids"]


# ---------------------------------------------------------------------------
# ST5.3 - ownership is not market bias
# ---------------------------------------------------------------------------
def test_option_ownership_is_not_market_direction(population_store):
    """53 of the 89 live option trades are SHORT and 39 of them were winners.
    Read as "short = bearish" that is a bearish trader with a bullish record;
    they are sold puts, and a sold put is bullish-to-neutral.
    """
    import journal_analytics
    from journal_exposure import classify_exposure

    trades = _trades_with_legs(population_store)

    long_put = _by_symbol(trades, "SPY260918P00600000")
    assert long_put["security_type"] == "OPT"
    assert long_put["direction"] == "LONG"
    bought = classify_exposure(long_put)
    assert bought.instrument == "OPT"
    assert bought.ownership_direction == "LONG"
    # A LONG option is never a bullish setup.
    assert bought.market_bias == "bearish"
    # Two FILL legs on ONE contract is not a multi-leg structure.
    assert len(long_put["legs"]) == 2
    assert bought.structure not in {"multi_leg", "partial_of_spread"}

    short_put = _by_symbol(trades, "DRAM260918P00055000")
    assert short_put["direction"] == "SHORT"
    sold = classify_exposure(short_put)
    assert sold.instrument == "OPT"
    assert sold.ownership_direction == "SHORT"
    assert sold.market_bias == "bullish_or_neutral"
    assert sold.structure not in {"multi_leg", "partial_of_spread"}

    # Two CONTRACTS under one trade: a 60 put and a 70 call. Neither leg's
    # direction is the position's opinion.
    combo = _by_symbol(trades, "CVNA")
    assert combo["status"] == "CLOSED"
    assert len(combo["legs"]) == 4
    straddle = classify_exposure(combo)
    assert straddle.structure == "multi_leg"
    assert straddle.market_bias == "unknown"

    summary = journal_analytics.personal_evidence_summary(population_store.list_trades())
    directional = set()
    for population in ("complete", "partly_closed", "open_exposure", "uncertain"):
        for bias, cell in summary[population]["by_market_bias"].items():
            if bias in {"bullish", "bearish", "bullish_or_neutral"}:
                directional.update(cell["trade_ids"])
    assert combo["trade_id"] not in directional
    assert long_put["trade_id"] in directional
    assert short_put["trade_id"] in directional


# ---------------------------------------------------------------------------
# ST5.4 - four populations, and every trade in exactly one of them
# ---------------------------------------------------------------------------
def test_the_four_populations_partition_every_trade(population_store):
    """Live: 165 CLOSED, 7 CLOSED_PARTIAL, 32 OPEN. Pooling them is how an open
    position's unrealized number gets read as a result.
    """
    import journal_analytics

    trades = population_store.list_trades()
    assert len(trades) == 7
    assert sorted(trade["status"] for trade in trades) == [
        "CLOSED", "CLOSED", "CLOSED", "CLOSED", "CLOSED", "CLOSED_PARTIAL", "OPEN"
    ]

    summary = journal_analytics.personal_evidence_summary(trades)
    names = ("complete", "partly_closed", "open_exposure", "uncertain")
    buckets = {name: set(summary[name]["trade_ids"]) for name in names}

    everything = {trade["trade_id"] for trade in trades}
    union = set().union(*buckets.values())
    assert union == everything
    assert sum(len(ids) for ids in buckets.values()) == len(everything) == 7

    # AAPL (+250), SPY put (+200) and the DRAM sold put (+100) are the three
    # complete, unambiguous results. MSFT is half out, NVDA never left, TSLA is
    # an unknown instrument and CVNA is a two-contract structure.
    assert summary["complete"]["n"] == 3
    assert summary["complete"]["winners"] == 3
    assert summary["partly_closed"]["n"] == 1
    assert summary["open_exposure"]["n"] == 1
    assert summary["uncertain"]["n"] == 2

    with_legs = _trades_with_legs(population_store)
    assert _by_symbol(with_legs, "MSFT")["trade_id"] in buckets["partly_closed"]
    assert _by_symbol(with_legs, "NVDA")["trade_id"] in buckets["open_exposure"]
    assert buckets["uncertain"] == {
        _by_symbol(with_legs, "TSLA")["trade_id"],
        _by_symbol(with_legs, "CVNA")["trade_id"],
    }

    # An open position has no result. Not zero - none.
    assert summary["open_exposure"]["net_pnl"] is None
