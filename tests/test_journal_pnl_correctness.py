"""Journal P&L correctness packet (2026-09-23).

Four input defects, each measured on a read-only copy of the trader's journal:
Flex times labelled with the desk's zone, socket/Flex duplicate fills, trades
with a made-up entry counted in totals, and side words that disagree.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_importers import parse_ibkr_flex_statement  # noqa: E402

CVNA_FLEX_XML = """
<FlexQueryResponse><FlexStatements><FlexStatement accountId="U5102524"><Trades>
  <Trade accountId="U5102524" assetCategory="OPT" symbol="CVNA  260618P00063000"
         underlyingSymbol="CVNA" expiry="20260618" strike="63" putCall="P" multiplier="100"
         buySell="SELL" quantity="-1" tradePrice="1.51" dateTime="20260612;145118"
         ibExecID="00021ab9.6a2c031e.03.01" ibOrderID="393387315" ibCommission="-0.7046506"
         netCash="150.2953494" currency="USD"/>
  <Trade accountId="U5102524" assetCategory="OPT" symbol="CVNA  260618P00064500"
         underlyingSymbol="CVNA" expiry="20260618" strike="64.5" putCall="P" multiplier="100"
         buySell="BUY" quantity="1" tradePrice="2.17" dateTime="20260612;145118"
         ibExecID="00021ab9.6a2c031e.02.01" ibOrderID="393387315" ibCommission="-0.69825"
         netCash="-217.69825" currency="USD"/>
</Trades></FlexStatement></FlexStatements></FlexQueryResponse>
"""


# ---------------------------------------------------------------------------
# 1. Flex dateTime is Eastern
# ---------------------------------------------------------------------------


def test_a_flex_datetime_is_read_as_eastern_not_the_desk_zone():
    """The live CVNA spread filled at 11:51:18 Pacific (the socket says so).
    Flex writes that fill as 14:51:18 with no zone; it is New York time."""
    execution = parse_ibkr_flex_statement(CVNA_FLEX_XML)[0]
    stamp = datetime.fromisoformat(execution.timestamp)

    assert stamp.utcoffset() == timedelta(hours=-4)
    assert stamp == datetime(2026, 6, 12, 11, 51, 18, tzinfo=ZoneInfo("America/Vancouver"))
    assert execution.trade_date == "2026-06-12"


def test_a_flex_datetime_that_names_its_zone_keeps_it():
    xml = CVNA_FLEX_XML.replace('dateTime="20260612;145118"', 'dateTime="20260612;145118 US/Central"', 1)
    execution = parse_ibkr_flex_statement(xml)[0]
    stamp = datetime.fromisoformat(execution.timestamp)

    assert stamp.utcoffset() == timedelta(hours=-5)
