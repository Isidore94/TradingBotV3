import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from market_prep.models import MarketPrepConfig
from market_prep.report_builder import build_daily_report_object
from market_prep.services import rss_news_service, sec_service
from market_prep.services.treasury_calendar_service import _normalize_auction_row
from market_prep.services.yfinance_service import _yfinance_lookup_symbol
from market_prep.services.watchlist_service import scan_watchlist_risk


class MarketPrepEnricherTests(unittest.TestCase):
    def test_treasury_priority_respects_rates_driver_setting(self):
        row = {
            "auction_date": "2026-04-29",
            "security_type": "Bond",
            "security_term": "30-Year",
            "offering_amt": "25000000000",
            "reopening": "No",
            "issue_date": "2026-05-01",
            "cusip": "912810ZZ9",
        }

        high = _normalize_auction_row(row, {"rates_market_driver": True, "important_auctions": []})
        medium = _normalize_auction_row(row, {"rates_market_driver": False, "important_auctions": []})

        self.assertEqual(high["priority"], "HIGH")
        self.assertEqual(medium["priority"], "MEDIUM")

    def test_yfinance_lookup_symbol_uses_yahoo_class_share_format(self):
        self.assertEqual(_yfinance_lookup_symbol("HVT.A"), "HVT-A")
        self.assertEqual(_yfinance_lookup_symbol("brk.b"), "BRK-B")
        self.assertEqual(_yfinance_lookup_symbol("SHOP.TO"), "SHOP.TO")

    def test_sec_submission_flags_danger_keyword(self):
        submission = {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2026-04-24"],
                    "accessionNumber": ["0000000000-26-000001"],
                    "primaryDocument": ["filing.htm"],
                    "items": [""],
                }
            }
        }
        settings = sec_service.get_sec_filings_settings(None)

        with patch.object(sec_service, "_fetch_text_limited", return_value="At-the-market offering program"):
            rows = sec_service._normalize_submission_filings(
                "ABC",
                "ABC Corp",
                "0000000000",
                submission,
                date(2026, 4, 20),
                date(2026, 4, 26),
                settings,
            )

        self.assertEqual(rows[0]["risk_classification"], "HIGH")
        self.assertIn("offering", [item.lower() for item in rows[0]["matched_keywords"]])

    def test_google_news_rss_generates_bounded_watchlist_queries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MarketPrepConfig.from_mapping(
                {
                    "features": {"news": False},
                    "google_news_rss": {
                        "enabled": True,
                        "max_watchlist_tickers": 1,
                        "queries": ["{ticker} earnings", "Federal Reserve inflation"],
                    },
                    "paths": {"cache_dir": str(Path(temp_dir) / "cache")},
                },
                repo_root=Path(temp_dir),
            )

            def fake_entries(url, *, timeout):
                return [
                    {
                        "title": "NVDA earnings preview",
                        "url": url,
                        "published": "2026-04-26",
                        "summary": "",
                    }
                ]

            with patch.object(rss_news_service, "_parse_feed_entries", side_effect=fake_entries):
                payload = rss_news_service.fetch_rss_headlines(
                    config=config,
                    tickers=["NVDA", "AAPL"],
                    limit=10,
                    force_refresh=True,
                )

        queries = {row["query"] for row in payload["headlines"]}
        self.assertIn("NVDA earnings", queries)
        self.assertIn("Federal Reserve inflation", queries)
        self.assertNotIn("AAPL earnings", queries)

    def test_watchlist_sympathy_risk_flags_related_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "longs.txt").write_text("AMD\n", encoding="utf-8")
            (root / "shorts.txt").write_text("", encoding="utf-8")
            config = MarketPrepConfig.from_mapping(
                {
                    "features": {"yfinance_earnings_fallback": False},
                    "paths": {
                        "longs_file": "longs.txt",
                        "shorts_file": "shorts.txt",
                        "cache_dir": "cache",
                    },
                },
                repo_root=root,
            )
            upcoming = {
                "source": "test",
                "earnings": [
                    {
                        "date": "2026-04-29",
                        "time": "AMC",
                        "ticker": "NVDA",
                        "company": "NVIDIA",
                        "importance": "MEGA",
                    }
                ],
            }

            with patch("market_prep.services.watchlist_service.get_many_ticker_metadata", return_value={"metadata": {}}):
                payload = scan_watchlist_risk(
                    config,
                    todays_events={"events": []},
                    today_tomorrow_earnings={"earnings": []},
                    upcoming_earnings=upcoming,
                    start_date=date(2026, 4, 26),
                )

        self.assertEqual(payload["sympathy_risks"][0]["ticker"], "AMD")
        self.assertIn("Sympathy Risk", payload["sympathy_risks"][0]["classification"])

    def test_daily_report_includes_chronological_catalyst_clock_and_new_sections(self):
        report = build_daily_report_object(
            todays_events={"events": []},
            next_7_events={
                "events": [
                    {
                        "date": "2026-04-29",
                        "time_et": "14:00",
                        "event": "FOMC Statement",
                        "priority": "HIGH",
                        "currency": "USD",
                    }
                ]
            },
            next_7_earnings={
                "earnings": [
                    {
                        "date": "2026-04-28",
                        "time": "BMO",
                        "ticker": "AAPL",
                        "company": "Apple",
                        "importance": "MEGA",
                        "market_cap": 4000000000000,
                    }
                ]
            },
            fed_calendar={
                "events": [
                    {
                        "date": "2026-04-29",
                        "time_et": "14:30",
                        "event": "FOMC Press Conference",
                        "priority": "HIGH",
                    }
                ]
            },
            treasury_calendar={
                "events": [
                    {
                        "date": "2026-04-27",
                        "time_et": "",
                        "event": "5-Year Treasury Note Auction",
                        "priority": "MEDIUM",
                    }
                ]
            },
            sec_filings={
                "filings": [
                    {
                        "ticker": "ABC",
                        "form": "8-K",
                        "filing_date": "2026-04-26",
                        "risk_classification": "HIGH",
                        "matched_keywords": ["offering"],
                        "url": "https://sec.example/filing",
                    }
                ]
            },
            report_date="2026-04-27",
        )
        markdown = report["markdown"]

        self.assertIn("## 7. Catalyst Clock", markdown)
        self.assertIn("## 10. Fed Risk", markdown)
        self.assertIn("## 11. Treasury Auction Risk", markdown)
        self.assertIn("## 14. SEC Filing Risk", markdown)
        self.assertLess(markdown.index("2026-04-27 TBD ET | Treasury"), markdown.index("2026-04-28 08:00 ET | Earnings"))


if __name__ == "__main__":
    unittest.main()
