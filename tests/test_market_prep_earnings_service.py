import unittest
from datetime import date
from unittest.mock import patch

from market_prep.report_builder import build_earnings_report
from market_prep.services.earnings_service import NasdaqEarningsProvider, get_upcoming_earnings
from market_prep.services.yfinance_service import enrich_event_with_metadata


class MarketPrepEarningsServiceTests(unittest.TestCase):
    def test_nasdaq_provider_normalizes_rows_and_sorts_by_market_cap(self):
        provider = NasdaqEarningsProvider(config=None, include_manual=False)

        def fake_fetch(date_str: str, *, config=None):
            self.assertEqual(date_str, "2026-04-27")
            return [
                {
                    "symbol": "SMOL",
                    "name": "Small Cap Inc.",
                    "time": "time-pre-market",
                    "marketCap": "$1,000,000,000",
                    "epsForecast": "$0.10",
                    "noOfEsts": "2",
                    "fiscalQuarterEnding": "Mar/2026",
                },
                {
                    "symbol": "BIG",
                    "name": "Big Market Mover",
                    "time": "time-after-hours",
                    "marketCap": "$250,000,000,000",
                    "epsForecast": "$2.50",
                    "noOfEsts": "19",
                    "fiscalQuarterEnding": "Mar/2026",
                },
            ]

        with patch("market_prep.services.earnings_service.fetch_nasdaq_earnings_for_date", fake_fetch):
            payload = get_upcoming_earnings(
                start_date=date(2026, 4, 27),
                days=0,
                provider=provider,
                config=None,
            )

        rows = payload["earnings"]
        self.assertEqual([row["ticker"] for row in rows], ["BIG", "SMOL"])
        self.assertEqual(rows[0]["time"], "AMC")
        self.assertEqual(rows[0]["importance"], "HIGH")
        self.assertEqual(rows[0]["market_cap"], 250_000_000_000)
        self.assertEqual(rows[0]["source"], "nasdaq")
        self.assertIn("EPS est $2.50", rows[0]["notes"])

    def test_yfinance_enrichment_preserves_nasdaq_market_cap_when_metadata_is_empty(self):
        event = {
            "ticker": "BIG",
            "market_cap": 250_000_000_000,
            "market_cap_fmt": "$250.00B",
            "importance": "LOW",
            "source": "nasdaq",
            "metadata_source": "nasdaq",
        }

        enriched = enrich_event_with_metadata(
            event,
            {
                "BIG": {
                    "ticker": "BIG",
                    "company_name": "",
                    "market_cap": None,
                    "source": "yfinance",
                }
            },
        )

        self.assertEqual(enriched["market_cap"], 250_000_000_000)
        self.assertEqual(enriched["market_cap_fmt"], "$250.00B")
        self.assertEqual(enriched["importance"], "HIGH")
        self.assertEqual(enriched["metadata_source"], "nasdaq")

    def test_earnings_report_flags_market_moving_focus(self):
        report = build_earnings_report(
            {
                "generated_at": "2026-04-27T07:00:00",
                "source": "nasdaq",
                "start_date": "2026-04-27",
                "end_date": "2026-04-27",
                "earnings": [
                    {
                        "date": "2026-04-27",
                        "time": "AMC",
                        "ticker": "BIG",
                        "company": "Big Market Mover",
                        "importance": "HIGH",
                        "market_cap": 250_000_000_000,
                    }
                ],
            },
            title="Earnings Next 7 Days",
        )

        self.assertIn("Market-moving focus:", report)
        self.assertIn("BIG | 2026-04-27 AMC | HIGH | $250.00B | Big Market Mover", report)


if __name__ == "__main__":
    unittest.main()
