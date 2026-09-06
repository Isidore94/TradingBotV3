"""Packet ST5.5 (and the ST5.3 corners the tester's file does not reach).

BUILDER-WRITTEN, and every test here was run against the pre-change file first:
the four `_read_week_tag_rows` / missing-risk / referral tests fail on
`main` with `TypeError: _read_week_tag_rows() got an unexpected keyword
argument 'store'` or `AttributeError`, and the exposure tests fail with
`ModuleNotFoundError: No module named 'journal_exposure'`.

The fixture pattern is the tester's: a temp SQLite journal built through
`JournalStore.upsert_executions` + `rebuild_trades` - the real assembly path -
and NOTHING here touches a live store. The journal is READ-ONLY to this packet;
the one write these tests make is through the store's own authorized writers
(`apply_provisional_tags`, `save_risk_fields`) on a temp file, which is what
proves the read paths never write.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

WEEK = ("2026-09-07", "2026-09-11")


def _execution(uid, symbol, security_type, side, quantity, price, timestamp, *, option=None):
    raw: dict = {}
    if option is not None:
        raw["option"] = option
        raw["multiplier"] = 100.0
    return {
        "execution_uid": uid,
        "broker": "QUESTRADE",
        "account_number": "111",
        "account_label": "TFSA",
        "account_type": "TFSA",
        "symbol": symbol,
        "security_type": security_type,
        "currency": "CAD",
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


def _round_trip(tag, symbol, day, *, price_in=10.0, price_out=12.0, security_type="STK", option=None):
    return [
        _execution(f"{tag}-1", symbol, security_type, "BUY", 100, price_in, f"{day}T09:31:00-04:00", option=option),
        _execution(f"{tag}-2", symbol, security_type, "SELL", 100, price_out, f"{day}T15:31:00-04:00", option=option),
    ]


def _store(tmp_path, executions, name="trade_journal.sqlite3"):
    from journal_store import JournalStore

    store = JournalStore(tmp_path / name)
    store.upsert_executions(executions)
    store.rebuild_trades(refresh_tags=False)
    store.book_currency_values()
    return store


def _id_for(store, symbol) -> str:
    matches = [t for t in store.list_trades() if t["symbol"] == symbol]
    assert len(matches) == 1, f"{symbol}: {len(matches)} trades"
    return str(matches[0]["trade_id"])


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# ST5.5 - the backlog reaches the review list; the week keeps the top
# ---------------------------------------------------------------------------
def test_every_provisional_trade_is_listed_not_only_this_weeks(tmp_path):
    """Live: 26 provisional tags waiting, and the page was scoped to one week.

    Gate #36 has been owed since 2026-09-01 - "the trader confirms or edits at
    least ten of the 24 provisional tags" - on a screen that could not show
    most of them.
    """
    from ui.panels import weekend_prep_panel

    store = _store(
        tmp_path,
        _round_trip("W", "INWEEK", "2026-09-09")
        + _round_trip("O", "OLD", "2026-08-12")
        + _round_trip("R", "REVIEWME", "2026-09-10")
        + _round_trip("X", "OLDREVIEW", "2026-08-13"),
    )
    assert store.apply_provisional_tags(_id_for(store, "INWEEK"), "avwap-reclaim") is True
    assert store.apply_provisional_tags(_id_for(store, "OLD"), "earnings-gap") is True
    store.mark_tags_needing_review(_id_for(store, "REVIEWME"))
    store.mark_tags_needing_review(_id_for(store, "OLDREVIEW"))

    rows = weekend_prep_panel._read_week_tag_rows(WEEK, store=store)
    symbols = [row["symbol"] for row in rows]

    # The provisional backlog is IN, whatever week it is from...
    assert "OLD" in symbols
    # ...and needs_review stays this week's, or 145 blank rows bury the 26 that
    # carry a proposal.
    assert "OLDREVIEW" not in symbols
    assert set(symbols) == {"INWEEK", "REVIEWME", "OLD"}

    # This week sorts first, so widening never pushes the week off the top.
    assert symbols[:2] == ["INWEEK", "REVIEWME"]
    assert [row["in_review_week"] for row in rows] == [True, True, False]


def test_a_confirmed_tag_is_never_offered_for_a_second_confirmation(tmp_path):
    from ui.panels import weekend_prep_panel

    store = _store(tmp_path, _round_trip("C", "MINE", "2026-09-09"))
    trade_id = _id_for(store, "MINE")
    assert store.apply_provisional_tags(trade_id, "avwap-reclaim") is True
    assert weekend_prep_panel._read_week_tag_rows(WEEK, store=store)

    assert store.confirm_tags(trade_id) is True
    assert weekend_prep_panel._read_week_tag_rows(WEEK, store=store) == []


# ---------------------------------------------------------------------------
# ST5.5 - the missing-risk worklist reads, refers, and never writes
# ---------------------------------------------------------------------------
def test_missing_planned_risk_lists_closed_trades_newest_first(tmp_path):
    from ui.panels import weekend_prep_panel

    store = _store(
        tmp_path,
        _round_trip("A", "OLDEST", "2026-09-08")
        + _round_trip("B", "NEWEST", "2026-09-10")
        + [_execution("OPEN-1", "STILLON", "STK", "BUY", 10, 5.0, "2026-09-09T09:31:00-04:00")],
    )
    rows = weekend_prep_panel._read_missing_planned_risk_rows(store=store)
    assert [row["symbol"] for row in rows] == ["NEWEST", "OLDEST"]
    # An OPEN trade has no result to plan against yet and is not a worklist item.
    assert "STILLON" not in {row["symbol"] for row in rows}

    # The trader types a plan through the store's own writer; the row leaves.
    store.save_risk_fields(
        _id_for(store, "NEWEST"), planned_entry=10.0, planned_stop=9.0, planned_risk=100.0
    )
    assert [row["symbol"] for row in weekend_prep_panel._read_missing_planned_risk_rows(store=store)] == [
        "OLDEST"
    ]


def test_no_reader_on_this_page_ever_writes_a_planned_risk(tmp_path, monkeypatch):
    """The packet forbids reconstructing a risk from an outcome. So does I7.

    `save_risk_fields` is spied to RAISE, so any read path that reached for it
    fails the test rather than quietly filling a column.
    """
    from journal_store import JournalStore
    from ui.panels import weekend_prep_panel

    store = _store(tmp_path, _round_trip("A", "AAPL", "2026-09-09"))

    def _spy(self, trade_id, **kwargs):
        raise AssertionError("a read path wrote planned_risk")

    monkeypatch.setattr(JournalStore, "save_risk_fields", _spy)

    weekend_prep_panel._read_week_tag_rows(WEEK, store=store)
    weekend_prep_panel._read_missing_planned_risk_rows(store=store)
    line = weekend_prep_panel._read_personal_evidence_coverage(store=store)
    assert line == (
        "Confirmed tags: 0 of 1 closed or partly closed trades. "
        "Provisional awaiting review: 0. Planned risk recorded: 0 of 1."
    )
    with store.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM trade_annotations WHERE planned_risk IS NOT NULL"
        ).fetchone()[0] == 0


def test_the_worklist_refers_the_trade_and_writes_nothing(qapp, tmp_path, monkeypatch):
    """A row opens the trade where `save_risk_fields` lives. That is all it does."""
    import project_paths
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    monkeypatch.setattr(
        project_paths, "WEEKEND_PREP_STATE_FILE", tmp_path / "state.json", raising=False
    )
    store = _store(tmp_path, _round_trip("A", "AAPL", "2026-09-09"))
    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    page._on_rows_ready(
        {
            "tags": [],
            "missing_risk": weekend_prep_panel._read_missing_planned_risk_rows(store=store),
            "coverage": weekend_prep_panel._read_personal_evidence_coverage(store=store),
        }
    )
    assert page.risk_table.rowCount() == 1
    assert "Missing planned risk: 1 closed trade(s)" in page.risk_note.text()
    assert "Planned risk recorded: 0 of 1" in page.coverage_note.text()

    seen: list[str] = []
    page.openTradeRequested.connect(seen.append)
    page.risk_table.selectRow(0)
    page._open_selected_risk_row()
    assert seen == [_id_for(store, "AAPL")]


def test_the_tag_page_still_renders_a_plain_list_of_rows(qapp, tmp_path, monkeypatch):
    """`_on_rows_ready` gained a dict payload; the old list shape still renders."""
    import project_paths
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    monkeypatch.setattr(
        project_paths, "WEEKEND_PREP_STATE_FILE", tmp_path / "state.json", raising=False
    )
    store = _store(tmp_path, _round_trip("A", "AAPL", "2026-09-09"))
    assert store.apply_provisional_tags(_id_for(store, "AAPL"), "avwap-reclaim") is True
    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    page._on_rows_ready(weekend_prep_panel._read_week_tag_rows(WEEK, store=store))
    assert page.table.rowCount() == 1
    assert "1 trade(s) waiting" in page.note.text()


def test_the_coverage_line_sits_UNDER_the_card_and_never_joins_it(qapp, monkeypatch):
    """The card is five to eight lines by the trader's own request (V2 2b).

    ST5.4's coverage sentence would be a ninth, so it has its own label under
    the card, fed by the tag page's worker through `coverageChanged`. This is
    the deviation from the packet's wording and it is deliberate: the code's
    five-to-eight contract is the fact and two tests already pin it.
    """
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    monkeypatch.setattr(panel_module, "_read_like_cohort", lambda: [])
    monkeypatch.setattr(panel_module, "_read_veto_cohort", lambda: [])
    monkeypatch.setattr(panel_module, "_read_week_trades", lambda bounds: [])
    monkeypatch.setattr(panel_module, "_read_awaiting_review", lambda: 0)
    monkeypatch.setattr("review_learning.build_review_learning_state", lambda **_k: {})

    lines = panel._read_verdict()
    assert 5 <= len(lines) <= 8
    assert not any("Confirmed tags:" in line for line in lines)

    sentence = "Confirmed tags: 1 of 9 closed trades. Provisional awaiting review: 2. Planned risk recorded: 0 of 9."
    panel.tag_week._on_rows_ready({"tags": [], "missing_risk": [], "coverage": sentence})
    assert panel.coverage_note.text() == sentence


# ---------------------------------------------------------------------------
# REVIEW BLOCKER 1 - populations are by STATUS; uncertainty is a LABEL
#
# Reproduced by the reviewer on a copy of the live journal 2026-09-06: with
# uncertainty checked FIRST, `uncertain` came out n=120 holding 84 CLOSED
# trades, ALL 7 CLOSED_PARTIAL and 29 of the 32 OPEN ones - so `partly_closed`
# read n=0 while seven exist, `open_exposure` read n=3 with a notional about 9%
# of the real one, and ONE pooled P&L figure summed realized results together
# with open positions' unrealized marks and counted those marks as WINNERS.
#
# The fixtures below use OPT and BAG for the partly-closed and open rows on
# purpose. The tester's population fixture used STK for both, which is exactly
# why the uncertainty branch was never exercised there.
# ---------------------------------------------------------------------------
def _uncertain_population_store(tmp_path):
    """Two closed, one partly closed and one open - the last three UNCERTAIN."""
    spread = {"right": "CALL", "underlying": "AA", "expiry": "260918", "strike": 60.0}
    return _store(
        tmp_path,
        # CLOSED, clean: +200
        _round_trip("K", "AAPL", "2026-09-09", price_in=10.0, price_out=12.0)
        # CLOSED, UNKNOWN instrument: -100, still a finished result
        + [
            _execution("U-1", "TSLA", "UNKNOWN", "BUY", 10, 100.0, "2026-09-09T10:10:00-04:00"),
            _execution("U-2", "TSLA", "UNKNOWN", "SELL", 10, 90.0, "2026-09-10T10:10:00-04:00"),
        ]
        # CLOSED_PARTIAL on an OPTION the store cannot name a contract for -
        # a plain ticker, no `option` payload. Half out, realized +100, and
        # UNCERTAIN because the right is unknown so the bias is unreadable.
        + [
            _execution("P-1", "NVDA", "OPT", "BUY", 2, 3.0, "2026-09-09T10:00:00-04:00"),
            _execution("P-2", "NVDA", "OPT", "SELL", 1, 4.0, "2026-09-10T10:00:00-04:00"),
        ]
        # OPEN on a BAG: never exited, and its mark is not a result
        + [
            _execution("O-1", "CVNA", "BAG", "BUY", 1, 5.0, "2026-09-09T10:15:00-04:00", option=spread),
        ],
    )


def test_the_three_status_populations_partition_and_uncertainty_only_labels(tmp_path):
    import journal_analytics

    store = _uncertain_population_store(tmp_path)
    trades = store.list_trades()
    assert sorted(t["status"] for t in trades) == [
        "CLOSED", "CLOSED", "CLOSED_PARTIAL", "OPEN"
    ]

    summary = journal_analytics.personal_evidence_summary(trades)

    # THE PARTITION IS THE THREE, and it is exact.
    status_ids = {
        name: set(summary[name]["trade_ids"])
        for name in ("complete", "partly_closed", "open_exposure")
    }
    everything = {t["trade_id"] for t in trades}
    assert set().union(*status_ids.values()) == everything
    assert sum(len(ids) for ids in status_ids.values()) == len(everything) == 4

    assert summary["complete"]["n"] == 2          # was 1: uncertainty ate TSLA
    assert summary["partly_closed"]["n"] == 1     # was 0 - the reviewer's n=0
    assert summary["open_exposure"]["n"] == 1     # was 0

    # Uncertainty is a LABEL across the three, and it says where each member is.
    uncertain = summary["uncertain"]
    assert uncertain["cross_cutting"] is True
    assert uncertain["n"] == 3
    assert uncertain["by_population"] == {
        "complete": 1, "partly_closed": 1, "open_exposure": 1
    }
    assert {member["status"] for member in uncertain["members"]} == {
        "CLOSED", "CLOSED_PARTIAL", "OPEN"
    }
    # NO POOLED MONEY over three statuses - that is what summed a mark into a
    # result. The money is reported once, inside the status population.
    assert uncertain["net_pnl"] is None
    assert uncertain["winners"] is None

    # Each status population carries its own uncertainty count.
    assert summary["complete"]["n_uncertain"] == 1
    assert summary["partly_closed"]["n_uncertain"] == 1
    assert summary["open_exposure"]["n_uncertain"] == 1


def test_an_open_positions_mark_is_never_a_winner_and_never_a_net(tmp_path):
    """The harm behind blocker 1, stated on its own.

    The OPEN BAG row books a `net_pnl_cad` of 0.0 - a mark, not a result - and
    under the first cut it landed in `uncertain` beside realized numbers and
    could be counted.
    """
    import journal_analytics

    store = _uncertain_population_store(tmp_path)
    summary = journal_analytics.personal_evidence_summary(store.list_trades())

    assert summary["open_exposure"]["net_pnl"] is None
    assert summary["open_exposure"]["winners"] is None
    assert summary["open_exposure"]["net_pnl_usd"] is None
    for cell in summary["open_exposure"]["by_market_bias"].values():
        assert cell["net_pnl"] is None
        assert cell["winners"] is None

    # AAPL +200 and TSLA -100 are the only measured results, and they are the
    # complete population's - one winner, not two, and not three.
    assert summary["complete"]["net_pnl"] == pytest.approx(100.0)
    assert summary["complete"]["winners"] == 1
    assert summary["partly_closed"]["winners"] == 1


def test_an_empty_population_prints_blank_not_a_net_of_zero(tmp_path):
    """A net of 0.00 says "measured, and it came to nothing". Blank says none."""
    import journal_analytics

    store = _store(tmp_path, _round_trip("K", "AAPL", "2026-09-09"))
    summary = journal_analytics.personal_evidence_summary(store.list_trades())
    assert summary["partly_closed"]["n"] == 0
    assert summary["partly_closed"]["net_pnl"] is None
    assert summary["open_exposure"]["n"] == 0
    assert summary["open_exposure"]["net_pnl"] is None


# ---------------------------------------------------------------------------
# REVIEW BLOCKER 2 - one confirmed tag is not "no confirmed tags"
# ---------------------------------------------------------------------------
def test_a_confirmed_tag_on_a_partly_closed_trade_is_counted_and_named(tmp_path):
    """The live journal's ONE confirmed tag sits on a CLOSED_PARTIAL trade.

    Counting confirmed over CLOSED and provisional over ALL put the numerator
    and the denominator in different populations, so the headline said "No
    confirmed setup tags" about a journal that holds one (EAT, 2026-08-21).
    """
    import journal_analytics

    store = _store(
        tmp_path,
        # CLOSED_PARTIAL - the one that carries the trader's own answer.
        [
            _execution("E-1", "EAT", "STK", "BUY", 100, 10.0, "2026-09-09T09:31:00-04:00"),
            _execution("E-2", "EAT", "STK", "SELL", 40, 12.0, "2026-09-10T09:31:00-04:00"),
        ]
        + _round_trip("A", "AAPL", "2026-09-09")
        + _round_trip("M", "MSFT", "2026-09-10"),
    )
    partly = [t for t in store.list_trades() if t["status"] == "CLOSED_PARTIAL"]
    assert len(partly) == 1
    assert store.save_trade_annotation(partly[0]["trade_id"], setup_tags="avwap-reclaim", notes="") is None
    assert store.apply_provisional_tags(_id_for(store, "AAPL"), "earnings-gap") is True

    trades = store.list_trades()
    summary = journal_analytics.personal_evidence_summary(trades)
    coverage = summary["coverage"]

    assert coverage["confirmed"] == 1
    assert coverage["closed"] == 2
    assert coverage["partly_closed"] == 1
    # ONE denominator for both lanes.
    assert coverage["reviewable"] == 3
    assert coverage["provisional"] == 1
    assert coverage["line"] == (
        "Confirmed tags: 1 of 3 closed or partly closed trades. "
        "Provisional awaiting review: 1. Planned risk recorded: 0 of 3."
    )
    # It names the count it has. "No confirmed setup tags" is a false statement
    # about the trader's own work here, not a conservative one.
    assert summary["headline"] == (
        "1 confirmed setup tag - under the n=30 floor "
        "(1 provisional awaiting review) - no personal setup can be called best."
    )
    assert summary["best_setup"] is None


def test_the_no_confirmed_wording_survives_for_a_true_zero(tmp_path):
    import journal_analytics

    store = _store(tmp_path, _round_trip("A", "AAPL", "2026-09-09"))
    assert store.apply_provisional_tags(_id_for(store, "AAPL"), "earnings-gap") is True
    summary = journal_analytics.personal_evidence_summary(store.list_trades())
    assert summary["coverage"]["confirmed"] == 0
    assert summary["headline"] == (
        "No confirmed setup tags (1 provisional awaiting review) - "
        "no personal setup can be called best."
    )


# ---------------------------------------------------------------------------
# Advisory - the whole-journal pass is linear, and the worklist is capped
# ---------------------------------------------------------------------------
def test_classify_all_is_linear_not_quadratic():
    """The reviewer measured 130 ms at 1,020 trades, on the Qt thread.

    Timing is a smell, not a proof, so the assertion is on SHAPE: doubling the
    input must not quadruple the work. A quadratic pass takes ~4x; the index
    takes ~2x, and the bar is set loose enough that a slow machine still passes
    while an n-squared regression cannot.
    """
    import time

    from journal_exposure import classify_all

    def _rows(count):
        return [
            {
                "trade_id": f"t{index}",
                "symbol": f"AA2609{18:02d}C{index:08d}",
                "security_type": "OPT",
                "direction": "LONG",
                "status": "CLOSED",
                "trade_date": "2026-09-09",
            }
            for index in range(count)
        ]

    small, large = _rows(400), _rows(800)
    classify_all(small)  # warm the interpreter, not a cache - there is none

    start = time.perf_counter()
    classify_all(small)
    small_seconds = time.perf_counter() - start
    start = time.perf_counter()
    classify_all(large)
    large_seconds = time.perf_counter() - start

    assert large_seconds < max(small_seconds * 3.0, 0.5), (small_seconds, large_seconds)


def test_the_missing_risk_table_is_capped_and_says_what_it_hid(qapp, tmp_path, monkeypatch):
    import project_paths
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    monkeypatch.setattr(
        project_paths, "WEEKEND_PREP_STATE_FILE", tmp_path / "state.json", raising=False
    )
    monkeypatch.setattr(weekend_prep_panel, "MISSING_RISK_ROWS_SHOWN", 3, raising=False)
    executions = []
    for index in range(5):
        executions += _round_trip(f"R{index}", f"SYM{index}", f"2026-09-0{index + 1}")
    store = _store(tmp_path, executions)

    rows = weekend_prep_panel._read_missing_planned_risk_rows(store=store, limit=3)
    assert len(rows) == 3
    assert rows[0]["missing_risk_total"] == 5
    # Newest first, so the cap keeps the newest three.
    assert [row["symbol"] for row in rows] == ["SYM4", "SYM3", "SYM2"]

    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    page._on_rows_ready({"tags": [], "missing_risk": rows, "coverage": ""})
    assert "Missing planned risk: 5 closed trade(s) - showing 3 of 5" in page.risk_note.text()


# ---------------------------------------------------------------------------
# ST5.3 - the corners the tester's fixture does not reach
# ---------------------------------------------------------------------------
def test_a_cash_row_is_uncertain_and_a_short_call_is_bearish_or_neutral(tmp_path):
    from journal_exposure import classify_exposure

    store = _store(
        tmp_path,
        [
            _execution("CASH-1", "CAD", "CASH", "BUY", 1, 1.0, "2026-09-09T09:31:00-04:00"),
            _execution("CASH-2", "CAD", "CASH", "SELL", 1, 1.0, "2026-09-10T09:31:00-04:00"),
        ]
        + [
            # SOLD to open, bought back. Ownership SHORT; the bet is that NVDA
            # does not reach 200 - bearish with a neutral half.
            _execution(
                "SC-1", "NVDA260918C00200000", "OPT", "SELL", 1, 2.0,
                "2026-09-09T09:31:00-04:00",
                option={"right": "CALL", "underlying": "NVDA", "expiry": "260918", "strike": 200.0},
            ),
            _execution(
                "SC-2", "NVDA260918C00200000", "OPT", "BUY", 1, 1.0,
                "2026-09-10T09:31:00-04:00",
                option={"right": "CALL", "underlying": "NVDA", "expiry": "260918", "strike": 200.0},
            ),
        ],
    )
    trades = {t["symbol"]: t for t in store.list_trades()}

    cash = classify_exposure(trades["CAD"])
    assert cash.instrument == "CASH"
    assert cash.market_bias == "unknown"
    assert cash.certainty == "uncertain"

    sold_call = trades["NVDA260918C00200000"]
    assert sold_call["direction"] == "SHORT"
    exposure = classify_exposure(sold_call)
    assert exposure.market_bias == "bearish_or_neutral"
    assert exposure.certainty == "known"


def test_two_option_trades_on_one_underlying_and_expiry_are_a_spread_CANDIDATE(tmp_path):
    """The store has NO sibling seam, so the label says candidate and stops.

    Two spread legs arrive as two `trades` rows keyed by their own OCC symbols
    and nothing links them. What can be observed is a second option trade on the
    same underlying and expiry in the same session on a different contract - and
    that is also what two independent ideas look like. So the row is labelled
    `partial_of_spread_candidate`, lands in the UNCERTAIN population, and no
    claim is made that it is half a spread.
    """
    import journal_analytics
    from journal_exposure import classify_all

    store = _store(
        tmp_path,
        _round_trip(
            "L", "AA260918C00060000", "2026-09-09", price_in=3.0, price_out=4.0,
            security_type="OPT",
            option={"right": "CALL", "underlying": "AA", "expiry": "260918", "strike": 60.0},
        )
        + _round_trip(
            "S", "AA260918C00070000", "2026-09-09", price_in=2.0, price_out=1.0,
            security_type="OPT",
            option={"right": "CALL", "underlying": "AA", "expiry": "260918", "strike": 70.0},
        )
        + _round_trip("U", "MSFT", "2026-09-09"),
    )
    trades = store.list_trades()
    exposures = classify_all(trades)
    legs = [
        exposure
        for trade_id, exposure in exposures.items()
        if exposure.instrument == "OPT"
    ]
    assert len(legs) == 2
    assert {exposure.structure for exposure in legs} == {"partial_of_spread_candidate"}
    assert {exposure.market_bias for exposure in legs} == {"unknown"}

    summary = journal_analytics.personal_evidence_summary(trades)
    # All three are CLOSED, so all three are `complete` - the status partition.
    # Two of them are LABELLED uncertain and the stock trade is not.
    assert summary["complete"]["n"] == 3
    assert summary["uncertain"]["n"] == 2
    assert summary["uncertain"]["by_population"] == {
        "complete": 2, "partly_closed": 0, "open_exposure": 0
    }
    assert summary["complete"]["n_uncertain"] == 2
    stock = _id_for(store, "MSFT")
    assert stock not in set(summary["uncertain"]["trade_ids"])
    assert stock in set(summary["complete"]["trade_ids"])


def test_a_lone_option_trade_is_not_a_spread_candidate(tmp_path):
    """One leg with nothing beside it is a single, not a suspicion."""
    from journal_exposure import classify_all

    store = _store(
        tmp_path,
        _round_trip(
            "L", "AA260918C00060000", "2026-09-09", price_in=3.0, price_out=4.0,
            security_type="OPT",
            option={"right": "CALL", "underlying": "AA", "expiry": "260918", "strike": 60.0},
        ),
    )
    exposures = list(classify_all(store.list_trades()).values())
    assert [exposure.structure for exposure in exposures] == ["single"]
    assert [exposure.market_bias for exposure in exposures] == ["bullish"]


def test_list_trade_legs_carries_the_contract_payload(tmp_path):
    """ST5.3's one widened SELECT. Without it a leg's contract is invisible."""
    store = _store(
        tmp_path,
        _round_trip(
            "L", "AA260918C00060000", "2026-09-09", price_in=3.0, price_out=4.0,
            security_type="OPT",
            option={"right": "CALL", "underlying": "AA", "expiry": "260918", "strike": 60.0},
        ),
    )
    trade_id = _id_for(store, "AA260918C00060000")
    legs = store.list_trade_legs(trade_id)
    assert legs and all("raw_json" in leg for leg in legs)
    assert json.loads(legs[0]["raw_json"])["option"]["right"] == "CALL"
    # Additive: the keys the old SELECT produced are all still there.
    for key in ("leg_id", "trade_id", "execution_uid", "side", "role", "quantity",
                "price", "timestamp", "broker", "account_number", "symbol",
                "security_type", "currency"):
        assert key in legs[0]


# ---------------------------------------------------------------------------
# ST5.1 - the window's honest fallback
# ---------------------------------------------------------------------------
def test_a_calendar_that_refuses_falls_back_NARROWER_never_wider():
    """Uncertainty may not manufacture a match.

    Outside the calendar's validated range `market_calendar` raises rather than
    extrapolate. The fallback is the OLD calendar-day arithmetic, which is
    strictly narrower than ten sessions - a window that grew on a refusal would
    invent trades the trader never linked.
    """
    from datetime import date, timedelta

    import preference_trade_outcomes as pto

    inside = pto.statement_window_end(date(2026, 9, 4))
    assert inside == date(2026, 9, 21)

    beyond = date(2040, 1, 3)  # past market_calendar.VALID_THROUGH
    assert pto.statement_window_end(beyond) == beyond + timedelta(
        days=pto.TRADE_WINDOW_SESSIONS
    )
    assert pto.statement_window_end(beyond) < beyond + timedelta(days=14)


def test_the_report_names_its_window_in_sessions():
    import preference_trade_outcomes as pto

    assert pto.TRADE_WINDOW_NOTE == "10 sessions"
    note = pto.summary_note(
        [
            {"trade_id": "t1", "journal_net_pnl": "100.0", "journal_r": ""},
            {"trade_id": "t1", "journal_net_pnl": "100.0", "journal_r": ""},
            {"trade_id": "", "journal_net_pnl": "", "journal_r": ""},
        ]
    )
    assert "10 sessions" in note
    assert "2 matched a trade over 1 distinct trade(s)" in note
    assert "planned risk recorded on 0 of 1 matched trades" in note
