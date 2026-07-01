import tempfile
import unittest
from datetime import date, timedelta
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
        self.assertEqual(settings["headline_lookback_days"], 14)
        self.assertIn("{ticker} reports earnings", settings["queries"])
        self.assertIn("{ticker} raises guidance", settings["queries"])
        self.assertIn("{ticker} offering", settings["queries"])
        self.assertIn("{ticker} analyst day", settings["queries"])
        self.assertIn("{ticker} strategic investment", settings["queries"])
        self.assertIn("{ticker} stake", settings["queries"])
        self.assertIn("{ticker} Anthropic", settings["queries"])
        self.assertNotIn("{ticker} catalyst", settings["queries"])
        self.assertNotIn("{ticker} price target", settings["queries"])

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

    def test_landmine_matching_does_not_treat_sector_as_sec_risk(self):
        rows = ticker_lookup_service.rank_landmine_headlines(
            [
                {
                    "title": "Technology sector outlook improves",
                    "query": "Technology sector news",
                    "source": "Google News",
                }
            ]
        )

        self.assertEqual(rows, [])

    def test_focus_lookup_headlines_keeps_recent_material_announcements_only(self):
        rows = ticker_lookup_service.focus_lookup_headlines(
            [
                {
                    "title": "ABC announces shelf offering",
                    "query": "ABC offering",
                    "source": "Google News",
                    "published": "Tue, 09 Jun 2026 14:00:00 GMT",
                },
                {
                    "title": "ABC earnings preview: what analysts expect",
                    "query": "ABC earnings",
                    "source": "Google News",
                    "published": "Wed, 10 Jun 2026 14:00:00 GMT",
                },
                {
                    "title": "ABC announces customer contract",
                    "query": "ABC contract",
                    "source": "Google News",
                    "published": "Fri, 01 May 2026 14:00:00 GMT",
                },
            ],
            reference_date=ticker_lookup_service._parse_date("2026-06-10"),
            lookback_days=14,
        )

        self.assertEqual([row["title"] for row in rows], ["ABC announces shelf offering"])
        self.assertIn("SEC/financing", rows[0]["material_tags"])

    def test_swing_risk_avoids_near_earnings_and_confirmed_road_bumps(self):
        risk = ticker_lookup_service.build_swing_risk_assessment(
            {
                "ticker": "ABC",
                "report_date": "2026-06-10",
                "window_days": 10,
                "headline_lookback_days": 14,
                "target_earnings": [
                    {
                        "ticker": "ABC",
                        "company": "ABC Corp",
                        "date": "2026-06-12",
                        "time": "AMC",
                        "importance": "HIGH",
                    }
                ],
                "peer_earnings": [],
                "sec_filings": {
                    "filings": [
                        {
                            "ticker": "ABC",
                            "form": "8-K",
                            "filing_date": "2026-06-09",
                            "risk_classification": "HIGH",
                            "matched_keywords": ["offering"],
                        }
                    ]
                },
                "landmine_headlines": [
                    {
                        "title": "ABC announces shelf offering",
                        "source": "Google News",
                        "published": "2026-06-09",
                        "landmine_tags": ["financing/dilution"],
                    }
                ],
                "target_headlines": [
                    {
                        "title": "ABC to present at investor day next week",
                        "query": "ABC investor day",
                        "source": "Google News",
                        "published": "2026-06-09",
                    }
                ],
                "earnings_payload": {"source": "test"},
                "news_headlines": {"source": "test"},
            }
        )

        self.assertEqual(risk["verdict"], "AVOID / WAIT")
        self.assertEqual(risk["risk_score"], 100)
        self.assertIn("SEC filing", {item["category"] for item in risk["risk_items"]})
        self.assertIn("Possible upcoming event", {item["category"] for item in risk["upcoming_catalysts"]})

    def test_swing_risk_ignores_stale_or_speculative_headline_noise(self):
        risk = ticker_lookup_service.build_swing_risk_assessment(
            {
                "ticker": "ABC",
                "report_date": "2026-06-10",
                "window_days": 10,
                "headline_lookback_days": 14,
                "target_earnings": [],
                "peer_earnings": [],
                "sec_filings": {"filings": []},
                "landmine_headlines": [
                    {
                        "title": "ABC announces customer contract",
                        "source": "Google News",
                        "published": "2026-05-01",
                        "landmine_tags": ["customer/supplier"],
                    }
                ],
                "target_headlines": [
                    {
                        "title": "ABC earnings preview: what analysts expect",
                        "query": "ABC earnings",
                        "source": "Google News",
                        "published": "2026-06-09",
                    }
                ],
                "earnings_payload": {"source": "test"},
                "news_headlines": {"source": "test"},
            }
        )

        self.assertEqual(risk["verdict"], "CLEAN")
        self.assertEqual(risk["risk_score"], 0)
        self.assertEqual(risk["risk_items"], [])

    def test_swing_risk_can_be_clean_but_keeps_missing_checks_visible(self):
        risk = ticker_lookup_service.build_swing_risk_assessment(
            {
                "ticker": "XYZ",
                "report_date": "2026-06-10",
                "window_days": 10,
                "target_earnings": [],
                "peer_earnings": [],
                "sec_filings": {"filings": []},
                "landmine_headlines": [],
                "target_headlines": [],
                "earnings_payload": {"source": "test"},
                "news_headlines": {"source": "test"},
            }
        )

        self.assertEqual(risk["verdict"], "CLEAN")
        self.assertEqual(risk["risk_score"], 0)
        self.assertTrue(any("earnings date" in item for item in risk["missing_checks"]))

    def test_lookup_ticker_context_composes_earnings_news_sec_and_peers(self):
        # Keep mocked headlines inside the 14-day headline lookback regardless of
        # when the test runs; hardcoded dates rotted out of the window before.
        recent_published = (date.today() - timedelta(days=2)).isoformat()
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
                            {
                                "title": "TSM announces investor day",
                                "query": "TSM investor day",
                                "source": "Google News",
                                "published": recent_published,
                            },
                            {
                                "title": "NVDA earnings preview",
                                "query": "NVDA earnings",
                                "source": "Google News",
                                "published": recent_published,
                            },
                        ],
                    },
                ),
            ):
                payload = ticker_lookup_service.lookup_ticker_context("tsm", config=config, days_ahead=45)

        self.assertEqual(payload["ticker"], "TSM")
        self.assertEqual(payload["target_earnings"][0]["ticker"], "TSM")
        self.assertEqual(payload["peer_earnings"][0]["ticker"], "NVDA")
        self.assertEqual(payload["target_headlines"][0]["title"], "TSM announces investor day")
        self.assertEqual(payload["industry_headlines"], [])
        self.assertEqual(payload["swing_risk"]["verdict"], "AVOID / WAIT")
        self.assertIn("Ticker earnings", {item["category"] for item in payload["swing_risk"]["risk_items"]})
        self.assertIn("Swing Safety", payload["markdown"])
        self.assertIn("decide whether it looks CLEAN, CAUTION, or AVOID/WAIT", payload["ai_swing_query"])
        self.assertIn("CLEAN, CAUTION, or AVOID/WAIT", payload["ai_swing_query"])
        self.assertIn("AI Swing Query", payload["markdown"])
        self.assertIn("Ticker Lookup - TSM", payload["markdown"])


if __name__ == "__main__":
    unittest.main()
