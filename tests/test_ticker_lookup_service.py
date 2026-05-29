import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from market_prep.models import MarketPrepConfig
from market_prep.services import ticker_lookup_service


class TickerLookupServiceTests(unittest.TestCase):
    def test_peer_lookup_includes_big_semiconductor_players_for_tsm(self):
        peers = ticker_lookup_service.peer_tickers_for_lookup(
            "TSM",
            {"sector": "Technology", "industry": "Semiconductors"},
            settings={"max_peer_tickers": 8},
        )

        self.assertIn("NVDA", peers)
        self.assertIn("AMD", peers)
        self.assertNotIn("TSM", peers)

    def test_default_lookup_settings_use_short_swing_window_and_expanded_queries(self):
        settings = ticker_lookup_service.get_ticker_lookup_settings(None)

        self.assertEqual(settings["days_ahead"], 10)
        self.assertIn("{ticker} catalyst", settings["queries"])
        self.assertIn("{ticker} analyst rating", settings["queries"])
        self.assertIn("{ticker} offering", settings["queries"])
        self.assertIn("{ticker} strategic investment", settings["queries"])
        self.assertIn("{ticker} stake", settings["queries"])
        self.assertIn("{ticker} Anthropic", settings["queries"])

    def test_landmine_headline_ranking_flags_hidden_exposure_terms(self):
        rows = ticker_lookup_service.rank_landmine_headlines(
            [
                {
                    "title": "SK Telecom expands Anthropic investment stake",
                    "query": "SK Telecom Anthropic",
                    "source": "Google News",
                },
                {"title": "Routine product update", "query": "SKM product", "source": "Google News"},
            ]
        )

        self.assertEqual(rows[0]["title"], "SK Telecom expands Anthropic investment stake")
        self.assertIn("AI/private exposure", rows[0]["landmine_tags"])
        self.assertIn("strategic stake", rows[0]["landmine_tags"])

    def test_lookup_ticker_context_composes_earnings_news_sec_and_peers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = MarketPrepConfig.from_mapping(
                {
                    "features": {"sec_filings": True, "yfinance_metadata": True},
                    "paths": {"cache_dir": "cache", "output_dir": "output"},
                    "ticker_lookup": {
                        "days_ahead": 45,
                        "max_peer_tickers": 3,
                        "news_limit": 10,
                        "include_ai_brief": False,
                    },
                },
                repo_root=root,
            )

            with (
                patch.object(
                    ticker_lookup_service,
                    "get_ticker_metadata",
                    return_value={
                        "ticker": "TSM",
                        "company_name": "Taiwan Semiconductor",
                        "sector": "Technology",
                        "industry": "Semiconductors",
                        "market_cap_fmt": "$900.00B",
                    },
                ),
                patch.object(
                    ticker_lookup_service,
                    "get_watchlist_earnings",
                    return_value={
                        "source": "nasdaq",
                        "earnings": [
                            {"ticker": "TSM", "date": "2026-05-15", "time": "BMO", "importance": "HIGH"},
                            {"ticker": "NVDA", "date": "2026-05-20", "time": "AMC", "importance": "MEGA"},
                        ],
                        "yfinance_status": {"status_label": "Loaded metadata from cache"},
                    },
                ),
                patch.object(
                    ticker_lookup_service,
                    "get_sec_filing_risk",
                    return_value={
                        "source": "SEC EDGAR",
                        "status_label": "Refreshed",
                        "filings": [
                            {
                                "ticker": "TSM",
                                "form": "6-K",
                                "filing_date": "2026-05-01",
                                "risk_classification": "MEDIUM",
                                "matched_keywords": ["guidance"],
                            }
                        ],
                    },
                ),
                patch.object(
                    ticker_lookup_service,
                    "fetch_rss_headlines",
                    return_value={
                        "source": "rss+google_news",
                        "status_label": "Refreshed",
                        "headlines": [
                            {"title": "TSM conference update", "query": "TSM conference", "source": "Google News"},
                            {"title": "NVDA earnings preview", "query": "NVDA earnings", "source": "Google News"},
                        ],
                    },
                ),
            ):
                payload = ticker_lookup_service.lookup_ticker_context("tsm", config=config, days_ahead=45)

        self.assertEqual(payload["ticker"], "TSM")
        self.assertEqual(payload["target_earnings"][0]["ticker"], "TSM")
        self.assertEqual(payload["peer_earnings"][0]["ticker"], "NVDA")
        self.assertEqual(payload["target_headlines"][0]["title"], "TSM conference update")
        self.assertEqual(payload["industry_headlines"][0]["title"], "NVDA earnings preview")
        self.assertIn("brief swing-trade read", payload["ai_swing_query"])
        self.assertIn("AI Swing Query", payload["markdown"])
        self.assertIn("Ticker Lookup - TSM", payload["markdown"])


if __name__ == "__main__":
    unittest.main()
