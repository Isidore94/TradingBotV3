"""R7 §9 step 1 - the three import-path defects that corrupt data quietly.

A10 misreads unrelated errors as client-id conflicts, B5 stamps unparseable
broker timestamps with the import time, and A4 reports a truncated IBKR read as
a complete one. None of the three ever raised; all three produced data that
looks right. Each test below was checked against the pre-fix code first, and
the docstrings say what it did there.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_importers as ji  # noqa: E402
from journal_importers import (  # noqa: E402
    BrokerTimestampError,
    QuestradeImporter,
    _is_ibkr_client_id_conflict,
    manual_execution_from_fields,
    parse_broker_datetime,
    parse_ibkr_flex_statement,
)


# ---------------------------------------------------------------------------
# A10 - "326" is an error code, not a substring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "326[7001]: Unable to connect as the client id is already in use",
        "326[-1]: client id in use",
        "2104[-1]: Market data farm ok; 326[7001]: duplicate client id",
        "Client id is already in use by another application",
    ],
)
def test_real_client_id_conflicts_are_recognized(message):
    assert _is_ibkr_client_id_conflict(message) is True


@pytest.mark.parametrize(
    "message",
    [
        # Every one of these was read as a client-id conflict before the fix,
        # because "326" appeared somewhere in the text.
        "201[7001]: Order rejected - price 326.50 is outside the limit",
        "1326[7001]: Connectivity between IB and TWS has been lost",
        "2326[7001]: Cross-side warning",
        "162[7001]: Historical Market Data Service error message: no data for 3260 bars",
        "504[7001]: Not connected",
    ],
)
def test_unrelated_errors_are_no_longer_client_id_conflicts(message):
    assert _is_ibkr_client_id_conflict(message) is False


def test_an_unrelated_error_is_not_retried_on_three_client_ids(monkeypatch):
    """Before the fix this connected three times and re-raised the first error.

    The visible cost was three TWS connections and a ~60s delay on an error
    that would never have been fixed by a different client id.
    """
    attempts: list[int] = []

    class _Refusing:
        def fetch(self, *, host, port, client_id, account, timeout_sec):
            attempts.append(client_id)
            raise RuntimeError("201[7001]: Order rejected - price 326.50 is outside the limit")

    monkeypatch.setattr(ji, "IBAPI_AVAILABLE", True)
    monkeypatch.setattr(ji, "IBKRExecutionImporter", _Refusing)
    with pytest.raises(RuntimeError, match="326.50"):
        ji.import_ibkr_executions(host="127.0.0.1", port=7496, client_id=9125)
    assert len(attempts) == 1


def test_a_real_conflict_still_walks_the_client_id(monkeypatch):
    """The retry that A10's fix must not remove."""
    attempts: list[int] = []

    class _Conflicting:
        def fetch(self, *, host, port, client_id, account, timeout_sec):
            attempts.append(client_id)
            if len(attempts) < 3:
                raise RuntimeError("326[7001]: client id is already in use")
            return []

    monkeypatch.setattr(ji, "IBAPI_AVAILABLE", True)
    monkeypatch.setattr(ji, "IBKRExecutionImporter", _Conflicting)
    assert ji.import_ibkr_executions(host="127.0.0.1", port=7496, client_id=9125) == []
    assert attempts == [9125, 9126, 9127]


# ---------------------------------------------------------------------------
# B5 - a timestamp is parsed or refused, never invented
# ---------------------------------------------------------------------------


def test_ibapi_10x_timezone_suffix_is_parsed_in_the_exchange_timezone():
    """The format every IBKR socket fill arrives in, which used to fall through.

    Before the fix this returned ``pacific_now()``: the fill's real time was
    discarded and replaced with the moment the import ran.
    """
    parsed = parse_broker_datetime("20260804 09:31:00 US/Eastern", strict=True)
    assert parsed.year == 2026 and parsed.month == 8 and parsed.day == 4
    assert (parsed.hour, parsed.minute) == (9, 31)
    assert parsed.utcoffset().total_seconds() == -4 * 3600  # EDT in August


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20260804-09:31:00", (2026, 8, 4, 9, 31)),
        ("20260804  09:31:00", (2026, 8, 4, 9, 31)),
        ("20260804 09:31:00", (2026, 8, 4, 9, 31)),
        ("2026-08-04 09:31:00", (2026, 8, 4, 9, 31)),
        ("2026-08-04T09:31:00-07:00", (2026, 8, 4, 9, 31)),
        ("2026-08-04T16:31:00Z", (2026, 8, 4, 16, 31)),
        ("20260804", (2026, 8, 4, 0, 0)),
        ("2026-08-04", (2026, 8, 4, 0, 0)),
    ],
)
def test_the_broker_formats_that_must_keep_parsing(text, expected):
    parsed = parse_broker_datetime(text, strict=True)
    assert (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute) == expected
    assert parsed.tzinfo is not None, "a naive timestamp is an ambiguous tax date"


@pytest.mark.parametrize("text", ["", "   ", "not a date", "20261301 09:31:00", "0000"])
def test_strict_mode_refuses_rather_than_guesses(text):
    with pytest.raises(BrokerTimestampError):
        parse_broker_datetime(text, strict=True)


def test_the_lenient_default_is_unchanged():
    """Nothing outside the import paths changes behaviour in this step."""
    before = ji.pacific_now()
    parsed = parse_broker_datetime("not a date")
    assert (parsed - before).total_seconds() < 5


def test_a_questrade_row_with_an_unreadable_time_is_quarantined_not_imported():
    importer = QuestradeImporter.__new__(QuestradeImporter)
    importer.quarantined = []
    executions: list = []
    importer._append_normalized(
        executions,
        {"id": "e1", "symbol": "AAPL", "quantity": 10, "price": 1.0, "timestamp": "yesterday-ish"},
        {"number": "51234567"},
    )
    assert executions == []
    assert len(importer.quarantined) == 1
    record = importer.quarantined[0]
    assert record["broker"] == "QUESTRADE"
    assert "yesterday-ish" in record["raw_json"], "the raw payload must survive quarantine"


def test_one_bad_flex_row_does_not_cost_the_good_ones():
    xml = """
    <FlexQueryResponse>
      <FlexStatements>
        <FlexStatement>
          <Trades>
            <Trade accountId="U1" symbol="AAPL" dateTime="20260804;093100" quantity="10"
                   tradePrice="150" buySell="BUY" ibExecID="x1" ibCommission="-1.0"/>
            <Trade accountId="U1" symbol="MSFT" dateTime="sometime tuesday" quantity="5"
                   tradePrice="300" buySell="BUY" ibExecID="x2" ibCommission="-1.0"/>
          </Trades>
        </FlexStatement>
      </FlexStatements>
    </FlexQueryResponse>
    """
    quarantine: list = []
    executions = parse_ibkr_flex_statement(xml, quarantine=quarantine)
    assert [e.symbol for e in executions] == ["AAPL"]
    assert len(quarantine) == 1
    assert "MSFT" in quarantine[0]["raw_json"]


def test_a_hand_typed_timestamp_is_refused_not_rewritten():
    with pytest.raises(BrokerTimestampError):
        manual_execution_from_fields(
            {"symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 1.0, "timestamp": "whenever"}
        )


def test_a_blank_manual_timestamp_still_means_now():
    execution = manual_execution_from_fields({"symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 1.0})
    assert datetime.fromisoformat(execution.timestamp).tzinfo is not None


# ---------------------------------------------------------------------------
# A4 - a truncated read is not a successful one
# ---------------------------------------------------------------------------


def _socket_importer(monkeypatch, *, signal_end: bool, executions: list | None = None):
    """A real IBKRExecutionImporter with its socket calls stubbed out."""
    importer = ji.IBKRExecutionImporter()
    monkeypatch.setattr(importer, "connect", lambda *a, **k: None)
    monkeypatch.setattr(importer, "run", lambda *a, **k: None)
    monkeypatch.setattr(importer, "disconnect", lambda *a, **k: None)
    monkeypatch.setattr(ji.time, "sleep", lambda *_: None)

    def _req(req_id, filter_obj):
        importer.executions.extend(executions or [])
        if signal_end:
            importer.exec_end.set()

    monkeypatch.setattr(importer, "reqExecutions", _req)
    return importer


def _execution_item(exec_id: str, time_text: str = "20260804 09:31:00 US/Eastern"):
    return {
        "contract": SimpleNamespace(symbol="AAPL", secType="STK", currency="USD", localSymbol=""),
        "execution": SimpleNamespace(
            execId=exec_id, acctNumber="U1234567", time=time_text, shares=10.0, price=150.0, side="BOT"
        ),
    }


@pytest.mark.skipif(not ji.IBAPI_AVAILABLE, reason="ibapi is not installed")
def test_a_timeout_without_execdetailsend_raises(monkeypatch):
    """The whole defect: partial fills used to be returned as the full day.

    Before the fix this returned the one execution below and the caller wrote a
    successful import run for a day it had only partly seen.
    """
    importer = _socket_importer(monkeypatch, signal_end=False, executions=[_execution_item("e1")])
    with pytest.raises(RuntimeError, match="without execDetailsEnd"):
        importer.fetch(host="127.0.0.1", port=7496, client_id=1, timeout_sec=1.0)


@pytest.mark.skipif(not ji.IBAPI_AVAILABLE, reason="ibapi is not installed")
def test_a_completed_read_still_returns_its_executions(monkeypatch):
    importer = _socket_importer(monkeypatch, signal_end=True, executions=[_execution_item("e1")])
    results = importer.fetch(host="127.0.0.1", port=7496, client_id=1, timeout_sec=1.0)
    assert [item.symbol for item in results] == ["AAPL"]
    assert results[0].timestamp.startswith("2026-08-04T09:31:00")
    assert importer.quarantined == []


@pytest.mark.skipif(not ji.IBAPI_AVAILABLE, reason="ibapi is not installed")
def test_a_socket_row_with_an_unreadable_time_is_quarantined(monkeypatch):
    importer = _socket_importer(
        monkeypatch,
        signal_end=True,
        executions=[_execution_item("e1"), _execution_item("e2", time_text="tuesday")],
    )
    results = importer.fetch(host="127.0.0.1", port=7496, client_id=1, timeout_sec=1.0)
    assert len(results) == 1
    assert len(importer.quarantined) == 1
    assert "tuesday" in importer.quarantined[0]["raw_json"]


@pytest.mark.skipif(not ji.IBAPI_AVAILABLE, reason="ibapi is not installed")
def test_an_error_with_no_executions_still_raises_first(monkeypatch):
    """The pre-existing guard keeps priority - its message is the useful one."""
    importer = _socket_importer(monkeypatch, signal_end=False)
    importer.errors.append("504[7001]: Not connected")
    with pytest.raises(RuntimeError, match="Not connected"):
        importer.fetch(host="127.0.0.1", port=7496, client_id=1, timeout_sec=1.0)
