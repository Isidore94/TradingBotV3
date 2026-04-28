import unittest

from market_prep.report_builder import build_daily_markdown, build_weekly_markdown


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

        self.assertLess(markdown.index("## 1. Highest Importance Focus"), markdown.index("## 2. Catalyst Clock"))
        self.assertLess(markdown.index("## 2. Catalyst Clock"), markdown.index("## 3. Scheduled Landmines Today"))
        self.assertIn("UPCOMING MACRO: 2026-04-29 14:00 ET [HIGH] USD FOMC Statement", markdown)
        self.assertIn("UPCOMING EARNINGS: 2026-04-30 | AAPL", markdown)
        self.assertIn("Market-moving earnings today/tomorrow:", markdown)
        self.assertIn("VZ | Verizon", markdown)
        self.assertIn("lower-priority earnings hidden", markdown)

        landmine_block = markdown.split("## 3. Scheduled Landmines Today", 1)[1].split("## 4. Economic Speedbumps", 1)[0]
        self.assertNotIn("SMOL | Small Co", landmine_block)

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
        major_block = markdown.split("## 6. Major Earnings", 1)[1].split("## 7. Watchlist Risks", 1)[0]
        self.assertLess(major_block.index("AAPL"), major_block.index("VZ"))


if __name__ == "__main__":
    unittest.main()
