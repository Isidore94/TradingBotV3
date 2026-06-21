import json
import unittest
from datetime import date, datetime, timedelta
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import earnings_history
from market_prep.models import MarketPrepConfig
from market_prep.report_builder import build_earnings_report
from market_prep.services.earnings_service import NasdaqEarningsProvider, fetch_nasdaq_earnings_for_date, get_upcoming_earnings
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

        with (
            patch("market_prep.services.earnings_service.fetch_nasdaq_earnings_for_date", fake_fetch),
            patch(
                "market_prep.services.earnings_service.merge_shared_earnings_events",
                side_effect=lambda events: list(events or []),
            ),
        ):
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

    def test_upcoming_earnings_filters_small_and_illiquid_names_when_configured(self):
        provider = NasdaqEarningsProvider(config=None, include_manual=False)
        config = MarketPrepConfig.from_mapping(
            {
                "features": {"yfinance_metadata": True},
                "earnings": {
                    "filter_by_market_cap_and_volume": True,
                    "min_market_cap": 1_000_000_000,
                    "min_average_volume": 1_000_000,
                    "exclude_unknown_market_cap": True,
                    "exclude_unknown_average_volume": True,
                },
            }
        )

        def fake_fetch(date_str: str, *, config=None):
            return [
                {"symbol": "BIG", "name": "Big Liquid", "marketCap": "$250,000,000,000"},
                {"symbol": "ILLQ", "name": "Illiquid Large", "marketCap": "$50,000,000,000"},
                {"symbol": "SMOL", "name": "Small Cap", "marketCap": "$500,000,000"},
                {"symbol": "UNKN", "name": "Unknown Volume", "marketCap": "$2,000,000,000"},
            ]

        enriched_input = []

        def fake_enrich(rows, *, config=None):
            enriched_input.extend(row["ticker"] for row in rows)
            overrides = {
                "BIG": {"average_volume": 2_500_000},
                "ILLQ": {"average_volume": 500_000},
                "UNKN": {"average_volume": None},
            }
            return [
                {**row, **overrides.get(row["ticker"], {})}
                for row in rows
            ], {"status": "cache", "status_label": "Loaded metadata from cache"}

        with (
            patch("market_prep.services.earnings_service.fetch_nasdaq_earnings_for_date", fake_fetch),
            patch("market_prep.services.earnings_service.enrich_events_with_metadata", side_effect=fake_enrich),
            patch(
                "market_prep.services.earnings_service.merge_shared_earnings_events",
                side_effect=lambda events: list(events or []),
            ),
        ):
            payload = get_upcoming_earnings(
                start_date=date(2026, 4, 27),
                days=0,
                provider=provider,
                config=config,
            )

        self.assertEqual([row["ticker"] for row in payload["earnings"]], ["BIG"])
        self.assertEqual(enriched_input, ["BIG", "ILLQ", "UNKN"])
        self.assertEqual(payload["filters"]["prefilter_removed_count"], 1)
        self.assertEqual(payload["filters"]["removed_count"], 2)

    def test_nasdaq_provider_writes_shared_earnings_history(self):
        provider = NasdaqEarningsProvider(config=None, include_manual=False)

        def fake_fetch(date_str: str, *, config=None):
            return [
                {
                    "symbol": "BIG",
                    "name": "Big Market Mover",
                    "time": "time-after-hours",
                    "marketCap": "$250,000,000,000",
                    "epsForecast": "$2.50",
                    "noOfEsts": "19",
                    "fiscalQuarterEnding": "Mar/2026",
                }
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "earnings_calendar_history.json"
            with (
                patch("market_prep.services.earnings_service.fetch_nasdaq_earnings_for_date", fake_fetch),
                patch.object(earnings_history, "DEFAULT_HISTORY_FILE", history_path),
            ):
                payload = get_upcoming_earnings(
                    start_date=date(2026, 4, 27),
                    days=0,
                    provider=provider,
                    config=None,
                )
            stored = earnings_history.get_events_for_symbols(["BIG"], path=history_path)["BIG"]

        self.assertEqual(payload["earnings"][0]["ticker"], "BIG")
        self.assertEqual(stored[0]["release_session"], "AMC")

    def test_fetch_nasdaq_earnings_for_date_uses_fresh_future_cache(self):
        target_date = (datetime.now().date() + timedelta(days=10)).isoformat()
        cached_rows = [{"symbol": "FRESH", "time": "time-after-hours"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "nasdaq_earnings_calendar_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "nasdaq",
                        "dates": {
                            target_date: {
                                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                                "rows": cached_rows,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("market_prep.services.earnings_service.get_default_cache_dir", return_value=Path(temp_dir)),
                patch("market_prep.services.earnings_service.requests.get") as get_mock,
            ):
                rows = fetch_nasdaq_earnings_for_date(target_date)

        self.assertEqual(rows, cached_rows)
        get_mock.assert_not_called()

    def test_fetch_nasdaq_earnings_for_date_retries_then_uses_stale_future_cache(self):
        target_date = (datetime.now().date() + timedelta(days=10)).isoformat()
        cached_rows = [{"symbol": "STALE", "time": "time-pre-market"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "nasdaq_earnings_calendar_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "generated_at": (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds"),
                        "source": "nasdaq",
                        "dates": {
                            target_date: {
                                "fetched_at": (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds"),
                                "rows": cached_rows,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("market_prep.services.earnings_service.get_default_cache_dir", return_value=Path(temp_dir)),
                patch("market_prep.services.earnings_service.requests.get", side_effect=RuntimeError("nasdaq down")) as get_mock,
                patch("market_prep.services.earnings_service.time.sleep") as sleep_mock,
            ):
                rows = fetch_nasdaq_earnings_for_date(target_date)

        self.assertEqual(rows, cached_rows)
        self.assertEqual(get_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

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
                    "average_volume": 1_500_000,
                    "source": "yfinance",
                }
            },
        )

        self.assertEqual(enriched["market_cap"], 250_000_000_000)
        self.assertEqual(enriched["market_cap_fmt"], "$250.00B")
        self.assertEqual(enriched["average_volume"], 1_500_000)
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
                "filters": {
                    "enabled": True,
                    "min_market_cap": 1_000_000_000,
                    "min_average_volume": 1_000_000,
                    "prefilter_removed_count": 4,
                    "removed_count": 2,
                    "removed_unknown_count": 1,
                },
            },
            title="Earnings Next 7 Days",
        )

        self.assertIn("Filter: min market cap $1.00B; min avg volume 1.0M shares", report)
        self.assertIn("filtered out 6 row(s)", report)
        self.assertIn("Market-moving focus:", report)
        self.assertIn("BIG | 2026-04-27 AMC | HIGH | $250.00B | Big Market Mover", report)


if __name__ == "__main__":
    unittest.main()
