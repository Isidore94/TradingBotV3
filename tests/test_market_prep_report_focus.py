import unittest

from market_prep.report_builder import build_daily_markdown, build_daily_report_object, build_weekly_markdown, build_weekly_report_object


class MarketPrepReportFocusTests(unittest.TestCase):
    def test_daily_report_puts_focus_first_and_hides_low_priority_landmine_earnings(self):
        report = {
            "report_date": "2026-04-27",
            "scheduled_landmines": {
                "high_priority_events": [],
                "earnings_today_tomorrow": [
                    {
                        "date": "2026-04-27",
                        "time": "AMC",
                        "ticker": "SMOL",
                        "company": "Small Co",
                        "importance": "LOW",
                        "market_cap": 1_000_000_000,
                    },
                    {
                        "date": "2026-04-27",
                        "time": "BMO",
                        "ticker": "VZ",
                        "company": "Verizon",
                        "importance": "MEDIUM",
                        "market_cap": 193_000_000_000,
                    },
                ],
                "watchlist_earnings_today_tomorrow": [],
            },
            "todays_events": {"events": []},
            "next_7_events": {
                "events": [
                    {
                        "date": "2026-04-29",
                        "time_et": "14:00",
                        "priority": "HIGH",
                        "currency": "USD",
                        "event": "FOMC Statement",
                    }
                ]
            },
            "today_tomorrow_earnings": {
                "earnings": [
                    {
                        "date": "2026-04-27",
                        "time": "BMO",
                        "ticker": "VZ",
                        "company": "Verizon",
                        "importance": "MEDIUM",
                        "market_cap": 193_000_000_000,
                    }
                ]
            },
            "next_7_earnings": {
                "earnings": [
                    {
                        "date": "2026-04-30",
                        "time": "AMC",
                        "ticker": "AAPL",
                        "company": "Apple",
                        "importance": "MEGA",
                        "market_cap": 4_000_000_000_000,
                    }
                ]
            },
            "watchlist_risk": {"risks": []},
            "rss_headlines": {},
            "youtube_links": {},
            "trading_posture": [],
        }

        markdown = build_daily_markdown(report)

        self.assertLess(markdown.index("## 1. Highest Importance Focus"), markdown.index("## 6. Catalyst Clock"))
        self.assertLess(markdown.index("## 6. Catalyst Clock"), markdown.index("## 7. Scheduled Landmines Today"))
        self.assertIn("UPCOMING MACRO: 2026-04-29 14:00 ET [HIGH] USD FOMC Statement", markdown)
        self.assertIn("UPCOMING EARNINGS: 2026-04-30 | AAPL", markdown)
        self.assertIn("Market-moving earnings today/tomorrow:", markdown)
        self.assertIn("VZ | Verizon", markdown)
        self.assertIn("lower-priority earnings hidden", markdown)

        landmine_block = markdown.split("## 7. Scheduled Landmines Today", 1)[1].split("## 8. Economic Speedbumps", 1)[0]
        self.assertNotIn("SMOL | Small Co", landmine_block)

    def test_daily_report_adds_landmine_checklist_market_snapshot_and_hold_warnings(self):
        report = build_daily_report_object(
            todays_events={
                "events": [
                    {
                        "date": "2026-04-27",
                        "time_et": "10:00",
                        "priority": "HIGH",
                        "currency": "USD",
                        "event": "CPI",
                    }
                ]
            },
            today_tomorrow_earnings={"earnings": []},
            watchlist_risk={
                "risks": [
                    {
                        "ticker": "NVDA",
                        "classification": "Earnings Today/Tomorrow",
                        "reason": "NVDA earnings 2026-04-27 AMC",
                    }
                ]
            },
            market_snapshot={
                "classification": {
                    "label": "Noisy",
                    "reason": "SPY and QQQ are both below 21 SMA.",
                },
                "rows": [],
            },
        )

        markdown = report["markdown"]

        self.assertIn("## 2. Daily Landmine Checklist", markdown)
        self.assertIn("Market regime: Noisy", markdown)
        self.assertIn("No-trade: Macro - 2026-04-27 10:00 ET [HIGH] USD CPI", markdown)
        self.assertIn("Review before overnight hold: NVDA", markdown)
        self.assertIn("Broad market is noisy/risk-off", " ".join(report["trading_posture"]))

    def test_weekly_report_focus_and_major_earnings_sort_by_importance_before_date(self):
        report = {
            "report_date": "2026-04-27",
            "week_risk_level": {"level": "HIGH", "reason": "Major catalysts are scheduled."},
            "economic_calendar": {
                "events": [
                    {
                        "date": "2026-04-28",
                        "time_et": "10:00",
                        "priority": "MEDIUM",
                        "currency": "USD",
                        "event": "Consumer Confidence",
                    },
                    {
                        "date": "2026-04-29",
                        "time_et": "14:00",
                        "priority": "HIGH",
                        "currency": "USD",
                        "event": "FOMC Statement",
                    },
                ]
            },
            "major_earnings": {
                "earnings": [
                    {
                        "date": "2026-04-27",
                        "time": "BMO",
                        "ticker": "VZ",
                        "company": "Verizon",
                        "importance": "MEDIUM",
                        "market_cap": 193_000_000_000,
                    },
                    {
                        "date": "2026-04-30",
                        "time": "AMC",
                        "ticker": "AAPL",
                        "company": "Apple",
                        "importance": "MEGA",
                        "market_cap": 4_000_000_000_000,
                    },
                ]
            },
            "watchlist_earnings_risk": {"risks": []},
            "rss_headlines": {},
            "youtube_links": {},
            "swing_trading_conditions": [],
        }

        markdown = build_weekly_markdown(report)

        self.assertLess(markdown.index("## 1. Highest Importance Focus"), markdown.index("## 2. Week Risk Level"))
        self.assertIn("MACRO: 2026-04-29 14:00 ET [HIGH] USD FOMC Statement", markdown)
        major_block = markdown.split("## 10. Major Earnings", 1)[1].split("## 11. Watchlist Risks", 1)[0]
        self.assertLess(major_block.index("AAPL"), major_block.index("VZ"))

    def test_weekly_report_adds_thesis_checklist_and_market_regime(self):
        report = build_weekly_report_object(
            economic_calendar={
                "events": [
                    {
                        "date": "2026-04-29",
                        "time_et": "14:00",
                        "priority": "HIGH",
                        "currency": "USD",
                        "event": "FOMC Statement",
                    }
                ]
            },
            earnings_calendar={
                "earnings": [
                    {
                        "date": "2026-04-30",
                        "time": "AMC",
                        "ticker": "AAPL",
                        "company": "Apple",
                        "importance": "MEGA",
                        "market_cap": 4_000_000_000_000,
                    }
                ]
            },
            watchlist_risk={
                "risks": [
                    {
                        "ticker": "TSLA",
                        "classification": "Earnings Within 14 Days",
                        "reason": "TSLA earnings soon",
                    }
                ]
            },
            market_snapshot={
                "classification": {
                    "label": "Clean",
                    "reason": "SPY and QQQ are above 21 SMA.",
                },
                "rows": [],
            },
        )

        markdown = report["markdown"]

        self.assertIn("## 3. Weekly Thesis Checklist", markdown)
        self.assertIn("Market regime: Clean", markdown)
        self.assertIn("Main catalyst days:", markdown)
        self.assertIn("Watchlist names needing review: TSLA", markdown)
        self.assertIn("Clean tape:", " ".join(report["swing_trading_conditions"]))


if __name__ == "__main__":
    unittest.main()
