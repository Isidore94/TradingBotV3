"""Tests for journal completeness: Questrade range backfill + IBKR Flex parsing."""

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_importers as ji  # noqa: E402

FLEX_SAMPLE = """<FlexQueryResponse queryName="trades" type="AF">
  <FlexStatements count="1">
    <FlexStatement accountId="U1234567" fromDate="20260101" toDate="20260630">
      <Trades>
        <Trade accountId="U1234567" symbol="AAPL" assetCategory="STK" currency="USD"
               buySell="BUY" quantity="100" tradePrice="187.25" dateTime="20260415;093211"
               ibCommission="-1.00" ibExecID="0000e0d5.660d1a2b.01.01" ibOrderID="55501" netCash="-18726.00"/>
        <Trade accountId="U1234567" symbol="AAPL" assetCategory="STK" currency="USD"
               buySell="SELL" quantity="-100" tradePrice="192.10" dateTime="20260422;140501"
               ibCommission="-1.00" ibExecID="0000e0d5.660d1a2b.01.02" ibOrderID="55502" netCash="19209.00"/>
      </Trades>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
"""

FLEX_ERROR_SAMPLE = """<FlexStatementResponse timestamp="01 July, 2026 10:00 AM EDT">
  <Status>Fail</Status>
  <ErrorCode>1012</ErrorCode>
  <ErrorMessage>Token has expired.</ErrorMessage>
</FlexStatementResponse>
"""


class FlexParseTests(unittest.TestCase):
    def test_parse_flex_trades(self):
        executions = ji.parse_ibkr_flex_statement(FLEX_SAMPLE)
        self.assertEqual(len(executions), 2)
        buy, sell = executions
        self.assertEqual(buy.broker, "IBKR")
        self.assertEqual(buy.symbol, "AAPL")
        self.assertEqual(buy.side, "BUY")
        self.assertEqual(buy.quantity, 100.0)
        self.assertEqual(buy.price, 187.25)
        self.assertEqual(buy.trade_date, "2026-04-15")
        self.assertEqual(buy.commission, 1.0)
        self.assertEqual(buy.account_number, "U1234567")
        self.assertEqual(sell.side, "SELL")
        self.assertEqual(sell.quantity, 100.0)  # stored unsigned; side carries direction
        # Distinct exec ids -> distinct uids (dedupe key).
        self.assertNotEqual(buy.execution_uid, sell.execution_uid)

    def test_parse_flex_error_response_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            ji.parse_ibkr_flex_statement(FLEX_ERROR_SAMPLE)
        self.assertIn("Token has expired", str(ctx.exception))


class QuestradeRangeTests(unittest.TestCase):
    def test_range_import_chunks_requests(self):
        importer = ji.QuestradeImporter.__new__(ji.QuestradeImporter)  # skip token setup
        calls: list[tuple[str, str, str]] = []

        def fake_get_accounts():
            return [{"number": "111", "type": "Margin", "name": "Main", "currency": "USD"}]

        def fake_get_executions(account_number, start, end):
            calls.append((account_number, start.date().isoformat(), end.date().isoformat()))
            return [
                {
                    "id": f"e{len(calls)}",
                    "symbol": "NVDA",
                    "side": "Buy",
                    "quantity": 10,
                    "price": 100.0,
                    "timestamp": start.isoformat(),
                }
            ]

        with (
            patch.object(importer, "get_accounts", side_effect=fake_get_accounts, create=True),
            patch.object(importer, "get_executions", side_effect=fake_get_executions, create=True),
        ):
            executions, accounts = importer.import_executions_for_range(date(2026, 1, 1), date(2026, 3, 15))

        # 74 days -> three 31-day-max chunks, contiguous and covering the range.
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1], "2026-01-01")
        self.assertEqual(calls[-1][2], "2026-03-15")
        self.assertEqual(len(executions), 3)
        self.assertTrue(all(e.broker == "QUESTRADE" for e in executions))
        self.assertEqual(accounts[0]["number"], "111")


def test_questrade_activities_use_full_offset_datetimes():
    """Questrade rejects the date-only values accepted by executions."""
    importer = ji.QuestradeImporter.__new__(ji.QuestradeImporter)
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return {"activities": []}

    importer._authorized_get = fake_get

    assert importer.get_activities("111", date(2026, 8, 17), date(2026, 8, 24)) == []
    assert calls == [
        (
            "v1/accounts/111/activities",
            {
                "startTime": "2026-08-17T00:00:00-07:00",
                "endTime": "2026-08-24T23:59:59-07:00",
            },
        )
    ]


def test_questrade_local_settings_win_over_stale_environment_tokens(monkeypatch):
    saved = {
        ji.QUESTRADE_REFRESH_TOKEN_SETTING: "local-refresh",
        ji.QUESTRADE_ACCESS_TOKEN_SETTING: "local-access",
        ji.QUESTRADE_API_SERVER_SETTING: "https://local.example/",
    }
    monkeypatch.setenv("QUESTRADE_REFRESH_TOKEN", "stale-env-refresh")
    monkeypatch.setenv("QUESTRADE_ACCESS_TOKEN", "stale-env-access")
    monkeypatch.setenv("QUESTRADE_API_SERVER", "https://stale.example/")
    monkeypatch.setattr(ji, "get_local_setting", lambda key, default="": saved.get(key, default))
    monkeypatch.setattr(ji, "save_local_setting", lambda key, value: saved.__setitem__(key, value))

    importer = ji.QuestradeImporter()

    assert importer.refresh_token == "local-refresh"
    assert importer.access_token == "local-access"
    assert importer.api_server == "https://local.example/"
    assert any("bootstrap seeds" in line for line in importer.status_lines())


def test_questrade_rotation_is_saved_even_when_environment_seeds_exist(monkeypatch):
    saved = {ji.QUESTRADE_REFRESH_TOKEN_SETTING: "current-local"}
    monkeypatch.setenv("QUESTRADE_REFRESH_TOKEN", "old-env-seed")
    monkeypatch.setenv("QUESTRADE_ACCESS_TOKEN", "old-env-access")
    monkeypatch.setattr(ji, "get_local_setting", lambda key, default="": saved.get(key, default))
    monkeypatch.setattr(ji, "save_local_setting", lambda key, value: saved.__setitem__(key, value))
    # The rotation is written in ONE save now: four read-modify-write cycles
    # were four chances for a concurrent writer to drop the new refresh token.
    monkeypatch.setattr(ji, "save_local_settings", lambda values: saved.update(values))

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "access_token": "rotated-access",
                "refresh_token": "rotated-refresh",
                "api_server": "https://api.example/",
                "expires_in": 1800,
            }

    class _Session:
        def get(self, *args, **kwargs):
            return _Response()

    ji.QuestradeImporter(session=_Session()).refresh_access_token()

    assert saved[ji.QUESTRADE_REFRESH_TOKEN_SETTING] == "rotated-refresh"
    assert saved[ji.QUESTRADE_ACCESS_TOKEN_SETTING] == "rotated-access"
    assert saved[ji.QUESTRADE_API_SERVER_SETTING] == "https://api.example/"


if __name__ == "__main__":
    unittest.main()
