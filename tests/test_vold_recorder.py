from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


LOCAL = ZoneInfo("America/Vancouver")
DAY = datetime(2026, 7, 30, 6, 30, tzinfo=LOCAL)


def _row(minutes, *, value=100.0):
    start = DAY + timedelta(minutes=minutes)
    return {
        "time": start.replace(tzinfo=None).strftime("%Y%m%d  %H:%M:%S"),
        "open": value,
        "high": value + 10.0,
        "low": value - 10.0,
        "close": value + 2.0,
        "volume": 0,
    }


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _recorder(tmp_path):
    from vold_recorder import VoldSessionRecorder

    recorder = VoldSessionRecorder(
        ledger_path=tmp_path / "vold.jsonl",
        state_path=tmp_path / "state.json",
    )
    recorder.activate_contract(
        {
            "con_id": 26718738,
            "symbol": "TICK-NYSE",
            "exchange": "NYSE",
            "proxy_kind": "nyse_tick_proxy",
            "is_exact_vold": False,
        },
        as_of=DAY,
        now=DAY,
    )
    return recorder


def test_completed_rows_append_once_and_forming_bar_is_excluded(tmp_path):
    recorder = _recorder(tmp_path)
    rows = [_row(0), _row(5), _row(10), _row(15)]

    assert recorder.observe(rows, now=DAY + timedelta(minutes=18)) == 3
    assert recorder.observe(rows, now=DAY + timedelta(minutes=18)) == 0
    events = _events(recorder.ledger_path)
    bars = [row for row in events if row["event_type"] == "breadth_bar"]
    assert len(bars) == 3
    assert bars[-1]["bar_end"].endswith("06:45:00-07:00")
    assert all(row["code_version"] and row["as_of"] and row["written_at"] for row in events)
    assert all(row["contract"]["symbol"] == "TICK-NYSE" for row in bars)


def test_restart_recovers_ledger_and_does_not_double_write(tmp_path):
    recorder = _recorder(tmp_path)
    rows = [_row(0), _row(5), _row(10)]
    assert recorder.observe(rows, now=DAY + timedelta(minutes=20)) == 3

    restarted = _recorder(tmp_path)
    assert restarted.observe(rows, now=DAY + timedelta(minutes=20)) == 0
    events = _events(recorder.ledger_path)
    assert sum(row["event_type"] == "contract_verified" for row in events) == 1
    assert sum(row["event_type"] == "breadth_bar" for row in events) == 3


def test_internal_and_poll_data_gaps_are_explicit_and_deduplicated(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.observe(
        [_row(0), _row(10)],
        now=DAY + timedelta(minutes=20),
    ) == 2
    assert recorder.record_data_gap(
        reason="historical request returned no rows",
        now=DAY + timedelta(minutes=25),
    )
    assert not recorder.record_data_gap(
        reason="same interval retry",
        now=DAY + timedelta(minutes=25),
    )

    gaps = [row for row in _events(recorder.ledger_path) if row["event_type"] == "data_gap"]
    assert len(gaps) == 2
    assert gaps[0]["missing_bar_count"] == 1
    assert all(row["data_gap"] is True for row in gaps)


def test_negative_index_values_are_valid_bars(tmp_path):
    recorder = _recorder(tmp_path)
    row = _row(0, value=-100.0)
    row.update({"open": -100.0, "high": 50.0, "low": -250.0, "close": -75.0})

    assert recorder.observe([row], now=DAY + timedelta(minutes=10)) == 1
    bar = next(row for row in _events(recorder.ledger_path) if row["event_type"] == "breadth_bar")
    assert bar["low"] == -250.0
    assert bar["close"] == -75.0


def test_contract_candidates_are_ordered_exact_then_explicit_proxies():
    from vold_recorder import CONTRACT_CANDIDATES

    assert CONTRACT_CANDIDATES[0].proxy_kind == "exact_vold"
    assert CONTRACT_CANDIDATES[-1].symbol == "TICK-NYSE"
    assert all(candidate.exchange == "NYSE" for candidate in CONTRACT_CANDIDATES)


def _contract(symbol, con_id):
    return SimpleNamespace(
        conId=con_id,
        symbol=symbol,
        localSymbol=symbol,
        secType="IND",
        exchange="NYSE",
        primaryExchange="",
        currency="USD",
        tradingClass="",
    )


def test_live_adapter_qualifies_and_data_verifies_before_activation(monkeypatch, caplog):
    from bounce_bot_lib import legacy
    from vold_recorder import BreadthContractCandidate

    exact = BreadthContractCandidate(
        "VOLD", "IND", "NYSE", "USD", "exact_vold", "exact"
    )
    proxy = BreadthContractCandidate(
        "TICK-NYSE", "IND", "NYSE", "USD", "nyse_tick_proxy", "proxy"
    )
    monkeypatch.setattr(legacy, "VOLD_CONTRACT_CANDIDATES", (exact, proxy))
    bot = object.__new__(legacy.BounceBot)
    bot.is_stopping = Mock(return_value=False)
    exact_details = SimpleNamespace(
        contract=_contract("VOLD", 1),
        longName="VOLD",
        validExchanges="NYSE",
    )
    proxy_details = SimpleNamespace(
        contract=_contract("TICK-NYSE", 26718738),
        longName="NYSE TICK INDEX",
        validExchanges="NYSE",
    )
    bot._request_contract_details = Mock(
        side_effect=[[exact_details], [proxy_details]]
    )
    bot._request_historical_contract_bars = Mock(side_effect=[[], [_row(0)]])
    bot._vold_recorder = Mock()
    bot._vold_contract = None
    bot._vold_contract_metadata = {}

    with caplog.at_level(logging.CRITICAL):
        assert legacy.BounceBot._qualify_vold_contract(bot)

    assert bot._vold_contract.symbol == "TICK-NYSE"
    assert bot._vold_contract_metadata["proxy_kind"] == "nyse_tick_proxy"
    bot._vold_recorder.activate_contract.assert_called_once()
    bot._vold_recorder.observe.assert_called_once()
    assert "proxy=nyse_tick_proxy" in caplog.text


def test_live_adapter_records_and_logs_unavailability(monkeypatch, caplog):
    from bounce_bot_lib import legacy
    from vold_recorder import BreadthContractCandidate

    candidate = BreadthContractCandidate(
        "VOLD", "IND", "NYSE", "USD", "exact_vold", "exact"
    )
    monkeypatch.setattr(legacy, "VOLD_CONTRACT_CANDIDATES", (candidate,))
    bot = object.__new__(legacy.BounceBot)
    bot.is_stopping = Mock(return_value=False)
    bot._request_contract_details = Mock(return_value=[])
    bot._vold_recorder = Mock()

    with caplog.at_level(logging.CRITICAL):
        assert not legacy.BounceBot._qualify_vold_contract(bot)

    bot._vold_recorder.record_unavailable.assert_called_once()
    assert "Collection is unhealthy; no silent skip" in caplog.text
