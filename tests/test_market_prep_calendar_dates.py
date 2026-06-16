import json
import tempfile
import unittest
import urllib.error
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from market_prep.models import MarketPrepConfig
from market_prep.orchestrator import MarketPrepOrchestrator, resolve_daily_prep_date, resolve_weekly_prep_window
from market_prep.services import fed_calendar_service
from market_prep.services.forexfactory_calendar_service import load_forexfactory_events


class MarketPrepCalendarDateTests(unittest.TestCase):
    def test_forexfactory_respects_zero_day_window_from_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_path = cache_dir / "forexfactory_calendar_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "ForexFactory",
                        "start_date": "2026-04-26",
                        "end_date": "2026-05-10",
                        "events": [
                            {
                                "date": "2026-04-26",
                                "time_et": "19:00",
                                "event": "Sunday Event",
                                "currency": "USD",
                                "priority": "MEDIUM",
                            },
                            {
                                "date": "2026-04-29",
                                "time_et": "14:00",
                                "event": "FOMC Statement",
                                "currency": "USD",
                                "priority": "HIGH",
                            },
                        ],
                        "warnings": [],
                    }
                ),
                encoding="utf-8",
            )
            config = MarketPrepConfig(
                features={"forexfactory_calendar": True},
                forexfactory={"enabled": True, "cache_ttl_hours": 999},
            )

            with patch(
                "market_prep.services.forexfactory_calendar_service.get_default_cache_dir",
                return_value=cache_dir,
            ):
                payload = load_forexfactory_events(
                    config,
                    start_date=date(2026, 4, 26),
                    days_ahead=0,
                )

        self.assertEqual(payload["start_date"], "2026-04-26")
        self.assertEqual(payload["end_date"], "2026-04-26")
        self.assertEqual([event["event"] for event in payload["events"]], ["Sunday Event"])

    def test_fed_calendar_fetch_falls_back_to_month_name_url(self):
        calls = []

        def fake_fetch(url, *, timeout):
            calls.append(url)
            if url.endswith("/2026-05.htm"):
                raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
            return ""

        with patch.object(fed_calendar_service, "_fetch_text", side_effect=fake_fetch):
            text = fed_calendar_service._fetch_month_calendar_text(2026, 5, timeout=1)

        self.assertEqual(text, "")
        self.assertEqual(
            calls,
            [
                "https://www.federalreserve.gov/newsevents/2026-05.htm",
                "https://www.federalreserve.gov/newsevents/2026-may.htm",
            ],
        )

    def test_daily_prep_date_rolls_weekend_to_next_weekday(self):
        self.assertEqual(resolve_daily_prep_date(date(2026, 4, 24)), date(2026, 4, 24))
        self.assertEqual(resolve_daily_prep_date(date(2026, 4, 25)), date(2026, 4, 27))
        self.assertEqual(resolve_daily_prep_date(date(2026, 4, 26)), date(2026, 4, 27))

    def test_weekly_prep_window_uses_remaining_week_or_next_weekend_week(self):
        monday = resolve_weekly_prep_window(date(2026, 4, 27))
        self.assertEqual(monday["week_start"], date(2026, 4, 27))
        self.assertEqual(monday["window_start"], date(2026, 4, 27))
        self.assertEqual(monday["week_end"], date(2026, 5, 3))

        friday = resolve_weekly_prep_window(date(2026, 5, 1))
        self.assertEqual(friday["week_start"], date(2026, 4, 27))
        self.assertEqual(friday["window_start"], date(2026, 5, 1))
        self.assertEqual(friday["week_end"], date(2026, 5, 3))

        saturday = resolve_weekly_prep_window(date(2026, 5, 2))
        self.assertEqual(saturday["week_start"], date(2026, 5, 4))
        self.assertEqual(saturday["window_start"], date(2026, 5, 4))
        self.assertEqual(saturday["week_end"], date(2026, 5, 10))

        sunday = resolve_weekly_prep_window(date(2026, 5, 3))
        self.assertEqual(sunday["week_start"], date(2026, 5, 4))
        self.assertEqual(sunday["window_start"], date(2026, 5, 4))
        self.assertEqual(sunday["week_end"], date(2026, 5, 10))

    def test_run_daily_prep_uses_resolved_prep_date(self):
        orchestrator = MarketPrepOrchestrator()
        calls = []

        def fake_today_events(*, target_date=None, refresh_forexfactory=False):
            calls.append(("today_events", target_date))
            return {"events": [], "start_date": target_date.isoformat(), "end_date": target_date.isoformat()}

        def fake_upcoming_events(*, days: int, start_date=None, refresh_forexfactory=False):
            calls.append(("upcoming_events", start_date, days))
            return {"events": [], "start_date": start_date.isoformat(), "end_date": start_date.isoformat()}

        def fake_today_earnings(*, target_date=None):
            calls.append(("today_earnings", target_date))
            return {"earnings": [], "start_date": target_date.isoformat(), "end_date": target_date.isoformat()}

        def fake_upcoming_earnings(*, days: int, start_date=None):
            calls.append(("upcoming_earnings", start_date, days))
            return {"earnings": [], "start_date": start_date.isoformat(), "end_date": start_date.isoformat()}

        def fake_watchlist_risk(*, todays_events, today_tomorrow_earnings, upcoming_earnings, start_date=None):
            calls.append(("watchlist", start_date))
            return {"risks": [], "message": "No watchlist tickers found."}

        with (
            patch("market_prep.orchestrator.resolve_daily_prep_date", return_value=date(2026, 4, 27)),
            patch.object(orchestrator, "_load_todays_events", side_effect=fake_today_events),
            patch.object(orchestrator, "_load_upcoming_events", side_effect=fake_upcoming_events),
            patch.object(orchestrator, "_load_today_tomorrow_earnings", side_effect=fake_today_earnings),
            patch.object(orchestrator, "_load_upcoming_earnings", side_effect=fake_upcoming_earnings),
            patch.object(orchestrator, "_load_watchlist_risk", side_effect=fake_watchlist_risk),
            patch.object(orchestrator, "_load_rss_headlines", return_value={"headlines": [], "message": ""}),
            patch.object(orchestrator, "_load_youtube_links", return_value={"videos": [], "message": ""}),
            patch.object(orchestrator, "_load_market_snapshot", return_value={"classification": {}, "rows": []}),
            patch.object(orchestrator, "_attach_ai_brief", side_effect=lambda report: report),
        ):
            result = orchestrator.run_daily_prep()

        self.assertEqual(result["prep_date"], "2026-04-27")
        self.assertEqual(result["daily_report"]["report_date"], "2026-04-27")
        self.assertTrue(all(call[1] == date(2026, 4, 27) for call in calls if len(call) >= 2))

    def test_run_weekly_prep_uses_resolved_remaining_week_window(self):
        orchestrator = MarketPrepOrchestrator()
        calls = []
        weekly_window = {
            "week_start": date(2026, 4, 27),
            "window_start": date(2026, 5, 1),
            "week_end": date(2026, 5, 3),
        }

        def fake_upcoming_events(*, days: int, start_date=None, refresh_forexfactory=False):
            calls.append(("upcoming_events", start_date, days))
            return {
                "events": [],
                "start_date": start_date.isoformat(),
                "end_date": (start_date).isoformat(),
            }

        def fake_calendar(*, days: int, start_date=None):
            calls.append(("calendar", start_date, days))
            return {"events": [], "start_date": start_date.isoformat(), "end_date": start_date.isoformat()}

        def fake_today_events(*, target_date=None, refresh_forexfactory=False):
            calls.append(("today_events", target_date))
            return {"events": [], "start_date": target_date.isoformat(), "end_date": target_date.isoformat()}

        def fake_today_earnings(*, target_date=None):
            calls.append(("today_earnings", target_date))
            return {"earnings": [], "start_date": target_date.isoformat(), "end_date": target_date.isoformat()}

        def fake_upcoming_earnings(*, days: int, start_date=None):
            calls.append(("upcoming_earnings", start_date, days))
            return {"earnings": [], "start_date": start_date.isoformat(), "end_date": start_date.isoformat()}

        def fake_watchlist_risk(*, todays_events, today_tomorrow_earnings, upcoming_earnings, start_date=None):
            calls.append(("watchlist", start_date))
            return {"risks": [], "tickers": [], "message": "No watchlist tickers found."}

        with (
            patch("market_prep.orchestrator.resolve_weekly_prep_window", return_value=weekly_window),
            patch.object(orchestrator, "_load_upcoming_events", side_effect=fake_upcoming_events),
            patch.object(orchestrator, "_load_fed_calendar", side_effect=fake_calendar),
            patch.object(orchestrator, "_load_treasury_calendar", side_effect=fake_calendar),
            patch.object(orchestrator, "_load_todays_events", side_effect=fake_today_events),
            patch.object(orchestrator, "_load_today_tomorrow_earnings", side_effect=fake_today_earnings),
            patch.object(orchestrator, "_load_upcoming_earnings", side_effect=fake_upcoming_earnings),
            patch.object(orchestrator, "_load_watchlist_risk", side_effect=fake_watchlist_risk),
            patch.object(orchestrator, "_load_sec_filings", return_value={"filings": [], "message": ""}),
            patch.object(orchestrator, "_load_rss_headlines", return_value={"headlines": [], "message": ""}),
            patch.object(orchestrator, "_load_youtube_links", return_value={"videos": [], "message": ""}),
            patch.object(orchestrator, "_load_market_snapshot", return_value={"classification": {}, "rows": []}),
            patch.object(orchestrator, "_attach_ai_brief", side_effect=lambda report: report),
        ):
            result = orchestrator.run_weekly_prep()

        self.assertEqual(result["week_start"], "2026-04-27")
        self.assertEqual(result["window_start"], "2026-05-01")
        self.assertEqual(result["window_end"], "2026-05-03")
        self.assertEqual(result["weekly_report"]["report_date"], "2026-04-27")
        self.assertIn(("upcoming_events", date(2026, 5, 1), 2), calls)
        self.assertIn(("today_events", date(2026, 5, 1)), calls)
        self.assertIn(("today_earnings", date(2026, 5, 1)), calls)
        self.assertIn(("watchlist", date(2026, 5, 1)), calls)


if __name__ == "__main__":
    unittest.main()
