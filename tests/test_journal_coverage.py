"""R7 §9 step 6 - the coverage ledger, partial persistence, and self-heal.

Root causes A3, A5, A6; invariant I2. Before this the journal could not answer
"which days do I actually have?" - a day nobody imported and a day with no
trades were the same absence of rows. That is the shape of the trader's report
that the journal misses trades.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_coverage as jc  # noqa: E402
import journal_runner  # noqa: E402
from journal_importers import parse_ibkr_flex_document  # noqa: E402
from journal_store import JournalStore  # noqa: E402

# 2026-08-03 is a Monday; 08-08 a Saturday; 08-09 a Sunday.
MONDAY = date(2026, 8, 3)
FRIDAY = date(2026, 8, 7)
SATURDAY = date(2026, 8, 8)
SUNDAY = date(2026, 8, 9)


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


@pytest.fixture(autouse=True)
def no_fx_network(monkeypatch):
    monkeypatch.setattr(
        journal_runner.journal_fx, "ensure_rates",
        lambda *args, **kwargs: {
            "booked": 0, "carried_back": 0, "unavailable": [], "errors": []
        },
    )


# ---------------------------------------------------------------------------
# I2 - a day is COVERED only when an import actually spanned it
# ---------------------------------------------------------------------------


def test_a_day_with_no_row_is_a_gap_not_a_quiet_success(store):
    """Absence of evidence is not coverage. This is the whole of I2."""
    gaps = jc.find_gaps(store, broker="QUESTRADE", account_number="5", start=MONDAY, end=FRIDAY)
    assert gaps == [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5), date(2026, 8, 6), date(2026, 8, 7)]


def test_a_covered_day_with_zero_executions_is_not_a_gap(store):
    """A quiet day is a fact, not a hole. The distinction did not exist before."""
    jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=MONDAY,
                     status=jc.COVERED, message="0 execution(s)")
    assert MONDAY not in jc.find_gaps(
        store, broker="QUESTRADE", account_number="5", start=MONDAY, end=FRIDAY
    )


def test_a_failed_day_is_a_gap_because_it_is_the_day_that_needs_another_try(store):
    jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=MONDAY,
                     status=jc.FAILED, message="timeout")
    assert MONDAY in jc.find_gaps(
        store, broker="QUESTRADE", account_number="5", start=MONDAY, end=FRIDAY
    )


def test_a_weekend_is_never_a_gap_and_never_retried(store):
    gaps = jc.find_gaps(store, broker="QUESTRADE", account_number="5", start=SATURDAY, end=SUNDAY)
    assert gaps == []


def test_marking_a_closed_day_covered_records_it_as_no_session_instead(store):
    """A caller's belief does not reopen the market."""
    jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=SATURDAY, status=jc.COVERED)
    rows = jc.coverage_rows(store, broker="QUESTRADE", account_number="5")
    assert [row["status"] for row in rows] == [jc.NO_SESSION]


def test_a_day_outside_the_calendars_validated_range_fails_open_as_a_gap(store):
    """`market_calendar` refuses to extrapolate past 2032. Unknown means work.

    The alternative - treating an unanswerable day as a holiday - would file a
    real trading day as "market closed" and hide it forever. Under I2 the
    failure mode has to be visible work.
    """
    assert jc.is_trading_day(date(2040, 6, 12)) is True


def test_attempts_count_failures_and_never_go_down(store):
    for _ in range(3):
        jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=MONDAY,
                         status=jc.FAILED, message="nope")
    assert jc.attempts_for(store, broker="QUESTRADE", account_number="5", day=MONDAY) == 3
    jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=MONDAY, status=jc.COVERED)
    assert jc.attempts_for(store, broker="QUESTRADE", account_number="5", day=MONDAY) == 3, (
        "the history of a day that fought back is worth keeping"
    )


def test_marking_a_range_covers_the_weekdays_and_skips_the_weekend(store):
    assert jc.mark_range(store, broker="QUESTRADE", account_number="5", start=MONDAY, end=SUNDAY,
                         status=jc.COVERED) == 7
    rows = {row["day"]: row["status"] for row in jc.coverage_rows(store, broker="QUESTRADE")}
    assert rows["2026-08-07"] == jc.COVERED
    assert rows["2026-08-08"] == jc.NO_SESSION and rows["2026-08-09"] == jc.NO_SESSION


def test_an_unknown_status_is_refused(store):
    with pytest.raises(ValueError, match="unsupported coverage status"):
        jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=MONDAY, status="PROBABLY")


# ---------------------------------------------------------------------------
# A5 - one bad chunk costs one chunk
# ---------------------------------------------------------------------------


class _ChunkedImporter:
    """A Questrade importer whose second account's first chunk always fails."""

    refresh_token = "token"
    access_token = ""
    api_server = ""

    def __init__(self):
        self.quarantined = []

    def get_activities(self, account_number, start, end):
        return []

    def iter_execution_chunks(self, start_date, end_date):
        chunk_day = MONDAY
        yield {
            "account": {"number": "51830546", "type": "TFSA"},
            "account_number": "51830546",
            "start": chunk_day,
            "end": chunk_day,
            "executions": [
                {
                    "execution_uid": "QT:51830546:e1", "broker": "QUESTRADE",
                    "account_number": "51830546", "account_label": "TFSA", "account_type": "TFSA",
                    "symbol": "AAPL", "security_type": "STK", "currency": "USD", "side": "BUY",
                    "quantity": 10, "price": 150.0, "timestamp": f"{chunk_day}T09:31:00-07:00",
                    "trade_date": str(chunk_day), "commission": 0.0, "fees": 0.0,
                    "gross_amount": None, "net_amount": None, "order_id": "",
                    "exchange_exec_id": "", "raw_json": "{}",
                }
            ],
        }
        yield {
            "account": {"number": "29347316", "type": "Margin"},
            "account_number": "29347316",
            "start": chunk_day,
            "end": chunk_day,
            "error": "Questrade returned 503",
        }


def test_a_failed_chunk_costs_only_its_own_days(store, monkeypatch):
    """Root cause A5, end to end.

    The old code accumulated every account's executions in one list and wrote
    the store after the whole pull returned, so the 503 below would have thrown
    away the first account's fill as well.
    """
    monkeypatch.setattr(journal_runner, "QuestradeImporter", _ChunkedImporter)
    result = journal_runner.run_journal_backfill(days=1, store=store, include_ibkr_flex=False)

    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_executions").fetchone()[0] == 1, (
            "the account that worked keeps its execution"
        )
    statuses = {
        (row["account_number"], row["status"]) for row in jc.coverage_rows(store, broker="QUESTRADE")
    }
    assert ("51830546", jc.COVERED) in statuses
    assert ("29347316", jc.FAILED) in statuses
    assert result["status"] == "FAILED", "and the run still reports the failure honestly"


def test_each_chunk_records_the_span_it_covered(store, monkeypatch):
    """A6: an import run that records no dates makes gaps undetectable."""
    monkeypatch.setattr(journal_runner, "QuestradeImporter", _ChunkedImporter)
    journal_runner.run_journal_backfill(days=1, store=store, include_ibkr_flex=False)
    runs = store.list_import_runs()
    spans = {(row["account_number"], row["coverage_start"], row["coverage_end"]) for row in runs}
    assert all(start and end for _, start, end in spans), "every run names the days it looked at"
    assert {row["trigger"] for row in runs} == {"backfill"}


def test_gui_backfill_requests_fx_for_the_executions_it_just_imported(store, monkeypatch):
    requested = []
    monkeypatch.setattr(journal_runner, "QuestradeImporter", _ChunkedImporter)
    monkeypatch.setattr(
        journal_runner.journal_fx, "ensure_rates",
        lambda _store, pairs, **kwargs: requested.extend(pairs) or {
            "booked": 0, "carried_back": 0, "unavailable": [], "errors": []
        },
    )

    journal_runner.run_journal_backfill(
        days=1, store=store, include_ibkr_flex=False, rebuild=True
    )

    assert any(currency == "USD" for _day, currency in requested)


def test_a_questrade_chunk_with_a_quarantined_row_is_not_covered(store, monkeypatch):
    class _QuarantinedImporter(_ChunkedImporter):
        def iter_execution_chunks(self, start_date, end_date):
            yield {
                "account": {"number": "51830546", "type": "TFSA"},
                "account_number": "51830546", "start": MONDAY, "end": MONDAY,
                "executions": [], "quarantined": [{"reason": "bad timestamp"}],
            }

    monkeypatch.setattr(journal_runner, "QuestradeImporter", _QuarantinedImporter)
    result = journal_runner.run_journal_backfill(
        days=1, store=store, include_ibkr_flex=False
    )

    assert result["status"] == "FAILED"
    assert {row["status"] for row in jc.coverage_rows(store, broker="QUESTRADE")} == {jc.FAILED}


# ---------------------------------------------------------------------------
# Flex marks coverage from the statement's own span
# ---------------------------------------------------------------------------


FLEX_XML = """
<FlexQueryResponse>
  <FlexStatements count="1">
    <FlexStatement accountId="U4867396" fromDate="20260803" toDate="20260807">
      <Trades>
        <Trade accountId="U4867396" symbol="AAPL" dateTime="20260803;093100" quantity="10"
               tradePrice="150" buySell="BUY" ibExecID="x1" ibCommission="-1.0" assetCategory="STK"/>
      </Trades>
      <OptionEAE>
        <OptionEAE accountId="U4867396" symbol="SPY260116C00500000" date="20260807"
                   transactionType="Expiration" quantity="-1" strike="500" putCall="C"/>
      </OptionEAE>
      <OpenPositions>
        <OpenPosition accountId="U4867396" symbol="MSFT" position="100" assetCategory="STK"/>
      </OpenPositions>
      <CashTransactions>
        <CashTransaction accountId="U4867396" type="Dividends" amount="12.50" currency="USD"
                         dateTime="20260805" symbol="MSFT" description="MSFT dividend"/>
      </CashTransactions>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
"""


def test_the_statement_reports_the_span_it_actually_covered():
    document = parse_ibkr_flex_document(FLEX_XML)
    assert document["from_date"] == date(2026, 8, 3)
    assert document["to_date"] == date(2026, 8, 7)
    assert document["accounts"] == ["U4867396"]
    assert [item.symbol for item in document["executions"]] == ["AAPL"]


def test_the_statements_other_sections_are_read_even_before_step_7_uses_them():
    document = parse_ibkr_flex_document(FLEX_XML)
    assert len(document["option_eae"]) == 1
    assert len(document["open_positions"]) == 1
    assert len(document["cash_transactions"]) == 1


def test_flex_coverage_comes_from_the_statement_not_from_the_range_requested(store, monkeypatch):
    """A 365-day query says nothing about day 366, and the service caps at 365."""
    monkeypatch.setattr(
        journal_runner, "import_ibkr_flex_executions",
        lambda **kwargs: parse_ibkr_flex_document(FLEX_XML),
    )
    journal_runner.run_journal_backfill(
        days=3650, store=store, include_questrade=False, include_ibkr_flex=True
    )
    days = sorted(row["day"] for row in jc.coverage_rows(store, broker="IBKR"))
    assert days[0] == "2026-08-03" and days[-1] == "2026-08-07", (
        "coverage spans the statement, not the decade that was asked for"
    )


def test_flex_without_a_declared_statement_span_marks_no_coverage(store, monkeypatch):
    xml = FLEX_XML.replace(' fromDate="20260803" toDate="20260807"', "")
    monkeypatch.setattr(
        journal_runner, "import_ibkr_flex_executions",
        lambda **kwargs: parse_ibkr_flex_document(xml),
    )

    result = journal_runner.run_journal_backfill(
        days=365, store=store, include_questrade=False, include_ibkr_flex=True
    )

    assert result["status"] == "FAILED"
    assert jc.coverage_rows(store, broker="IBKR") == []


def test_flex_quarantine_marks_its_declared_span_failed_not_covered(store, monkeypatch):
    xml = FLEX_XML.replace(
        "</Trades>",
        '<Trade accountId="U4867396" symbol="MSFT" dateTime="not-a-time" quantity="5" '
        'tradePrice="300" buySell="BUY" ibExecID="bad"/></Trades>',
    )
    monkeypatch.setattr(
        journal_runner, "import_ibkr_flex_executions",
        lambda **kwargs: parse_ibkr_flex_document(xml),
    )

    result = journal_runner.run_journal_backfill(
        days=365, store=store, include_questrade=False, include_ibkr_flex=True
    )

    assert result["status"] == "FAILED"
    assert {row["status"] for row in jc.coverage_rows(store, broker="IBKR")} == {jc.FAILED}


# ---------------------------------------------------------------------------
# A3 - a failed day gets picked up again
# ---------------------------------------------------------------------------


def test_self_heal_repairs_gaps_oldest_first(store):
    jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=date(2026, 8, 6),
                     status=jc.FAILED, message="503")
    seen: list[date] = []

    def fetch(broker, account, day):
        seen.append(day)
        return 2

    summary = jc.self_heal(
        store, fetch, accounts=[("QUESTRADE", "5")], today=date(2026, 8, 10), lookback_days=7
    )
    assert seen == sorted(seen), "oldest first - the oldest gap is the likeliest to age out"
    assert date(2026, 8, 6) in seen
    assert len(summary["repaired"]) == len(seen)
    assert jc.find_gaps(store, broker="QUESTRADE", account_number="5",
                        start=date(2026, 8, 3), end=date(2026, 8, 9)) == []


def test_self_heal_never_claims_today(store):
    """Today is still being traded; marking it covered freezes a partial day."""
    seen: list[date] = []
    jc.self_heal(store, lambda b, a, d: seen.append(d) or 0, accounts=[("QUESTRADE", "5")],
                 today=date(2026, 8, 6), lookback_days=7)
    assert date(2026, 8, 6) not in seen


def test_retry_failed_only_does_not_consume_unattempted_gaps(store):
    jc.mark_coverage(
        store, broker="QUESTRADE", account_number="5", day=date(2026, 8, 6),
        status=jc.FAILED, message="503",
    )
    seen = []

    jc.self_heal(
        store, lambda b, a, d: seen.append(d) or 0, accounts=[("QUESTRADE", "5")],
        today=date(2026, 8, 10), lookback_days=7, failed_only=True,
    )

    assert seen == [date(2026, 8, 6)]


def test_a_days_failure_is_recorded_and_does_not_stop_the_rest(store):
    def fetch(broker, account, day):
        if day == date(2026, 8, 5):
            raise RuntimeError("broker said no")
        return 1

    summary = jc.self_heal(store, fetch, accounts=[("QUESTRADE", "5")],
                           today=date(2026, 8, 10), lookback_days=7)
    assert [item["day"] for item in summary["failed"]] == ["2026-08-05"]
    assert len(summary["repaired"]) >= 3, "one bad day does not end the night"
    rows = {row["day"]: row for row in jc.coverage_rows(store, broker="QUESTRADE")}
    assert rows["2026-08-05"]["status"] == jc.FAILED
    assert "broker said no" in rows["2026-08-05"]["message"]


def test_a_permanently_broken_day_stops_eating_the_budget_but_stays_visible(store):
    """A3's fix must not become its own denial of service."""
    for _ in range(5):
        jc.mark_coverage(store, broker="QUESTRADE", account_number="5", day=date(2026, 8, 5),
                         status=jc.FAILED, message="always fails")
    attempted: list[date] = []
    summary = jc.self_heal(
        store, lambda b, a, d: attempted.append(d) or 1, accounts=[("QUESTRADE", "5")],
        today=date(2026, 8, 10), lookback_days=7, max_attempts_per_day=5,
    )
    assert date(2026, 8, 5) not in attempted
    assert [item["day"] for item in summary["exhausted"]] == ["2026-08-05"]
    # Still a gap, still on the Health tab. Skipped is not resolved.
    assert date(2026, 8, 5) in jc.find_gaps(
        store, broker="QUESTRADE", account_number="5", start=date(2026, 8, 3), end=date(2026, 8, 9)
    )


def test_one_night_is_bounded(store):
    attempted: list[date] = []
    summary = jc.self_heal(
        store, lambda b, a, d: attempted.append(d) or 1, accounts=[("QUESTRADE", "5")],
        today=date(2026, 8, 14), lookback_days=365, max_days_per_night=3,
    )
    assert len(attempted) == 3
    assert summary["budget_exhausted"] is True


def test_the_inception_date_stops_the_ledger_running_off_the_end(store, monkeypatch):
    monkeypatch.setattr(
        jc, "get_local_setting",
        lambda key, default="": "2026-08-05" if key == "journal_inception_date_questrade" else default,
    )
    attempted: list[date] = []
    jc.self_heal(store, lambda b, a, d: attempted.append(d) or 1, accounts=[("QUESTRADE", "5")],
                 today=date(2026, 8, 10), lookback_days=365)
    assert min(attempted) >= date(2026, 8, 5)


def test_self_heal_with_nothing_to_do_is_a_quiet_success(store):
    jc.mark_range(store, broker="QUESTRADE", account_number="5", start=date(2026, 8, 3),
                  end=date(2026, 8, 9), status=jc.COVERED)
    summary = jc.self_heal(store, lambda b, a, d: 1 / 0, accounts=[("QUESTRADE", "5")],
                           today=date(2026, 8, 10), lookback_days=7)
    assert summary["attempted"] == [] and summary["failed"] == []


# ---------------------------------------------------------------------------
# 2026-08-25: a day whose CAUSE was repaired must be reachable again
# ---------------------------------------------------------------------------
def test_an_exhausted_day_is_retried_when_the_trader_asks_for_it(store):
    """The 140 stale OAuth days.

    They failed while the Questrade chain was broken, burned their attempt
    budget, and were then permanently skipped - so a repaired chain could never
    turn them green and the Health tile stayed red forever on a fixed problem.
    `attempts` still only goes up (it is history, and history is not rewritten);
    what changes is that an EXPLICIT trader retry may ignore the cap. The
    nightly still respects it, or a dead chain eats the budget every night.
    """
    day = date(2026, 2, 3)
    for _ in range(jc.DEFAULT_MAX_ATTEMPTS_PER_DAY):
        jc.mark_coverage(store, broker="QUESTRADE", account_number="A1", day=day,
                         status=jc.FAILED, message="oauth chain was dead")
    assert jc.attempts_for(store, broker="QUESTRADE", account_number="A1", day=day) >= (
        jc.DEFAULT_MAX_ATTEMPTS_PER_DAY
    )

    calls: list[date] = []

    def _fetch(broker, account, when):
        calls.append(when)
        return 2

    skipped = jc.self_heal(store, _fetch, accounts=[("QUESTRADE", "A1")],
                           today=date(2026, 2, 10), failed_only=True)
    assert calls == [], "the nightly budget must still respect the attempt cap"
    assert skipped["exhausted"], "and it must say the day was skipped, not silently drop it"

    repaired = jc.self_heal(store, _fetch, accounts=[("QUESTRADE", "A1")],
                            today=date(2026, 2, 10), failed_only=True,
                            include_exhausted=True)

    assert calls == [day], "an explicit retry must reach the day whose cause was fixed"
    assert [row["day"] for row in repaired["repaired"]] == [day.isoformat()]
    assert repaired["exhausted"] == []
    # History is not rewritten: the attempts it burned are still on the record.
    assert jc.attempts_for(store, broker="QUESTRADE", account_number="A1", day=day) >= (
        jc.DEFAULT_MAX_ATTEMPTS_PER_DAY
    )


def test_the_explicit_retry_reports_how_many_exhausted_days_it_reopened(store):
    """A number the trader can check against the tile, rather than a silent
    change in what the button does."""
    for day in (date(2026, 2, 3), date(2026, 2, 4)):
        for _ in range(jc.DEFAULT_MAX_ATTEMPTS_PER_DAY):
            jc.mark_coverage(store, broker="QUESTRADE", account_number="A1", day=day,
                             status=jc.FAILED, message="oauth chain was dead")

    summary = jc.self_heal(store, lambda *a: 1, accounts=[("QUESTRADE", "A1")],
                           today=date(2026, 2, 10), failed_only=True,
                           include_exhausted=True)

    assert summary["reopened_exhausted"] == 2
