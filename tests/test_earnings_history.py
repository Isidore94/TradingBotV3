import tempfile
import unittest
from pathlib import Path

from scripts import earnings_history


class EarningsHistoryTests(unittest.TestCase):
    def test_nasdaq_release_time_normalization(self):
        event_date = "2026-05-07"

        bmo = earnings_history.normalize_nasdaq_event(
            {"symbol": "BMOX", "time": "time-pre-market"},
            event_date,
        )
        amc = earnings_history.normalize_nasdaq_event(
            {"symbol": "AMCX", "time": "time-after-hours"},
            event_date,
        )
        tbd = earnings_history.normalize_nasdaq_event(
            {"symbol": "TBDX", "time": "time-not-supplied"},
            event_date,
        )

        self.assertEqual(bmo["release_session"], "BMO")
        self.assertEqual(amc["release_session"], "AMC")
        self.assertEqual(tbd["release_session"], "TBD")

    def test_confirmed_session_upgrades_tbd_without_duplicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "earnings_calendar_history.json"
            earnings_history.merge_events(
                [
                    {
                        "ticker": "XYZ",
                        "earnings_date": "2026-05-07",
                        "release_session": "TBD",
                        "source": "nasdaq",
                    }
                ],
                path=path,
            )
            earnings_history.merge_events(
                [
                    {
                        "ticker": "XYZ",
                        "earnings_date": "2026-05-07",
                        "release_session": "AMC",
                        "source": "nasdaq",
                        "source_confidence": "confirmed",
                    }
                ],
                path=path,
            )

            events = earnings_history.get_events_for_symbols(["XYZ"], path=path)["XYZ"]

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["release_session"], "AMC")

    def test_yfinance_date_does_not_overwrite_confirmed_nasdaq_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "earnings_calendar_history.json"
            earnings_history.merge_events(
                [
                    {
                        "ticker": "NVDA",
                        "earnings_date": "2026-05-20",
                        "release_session": "AMC",
                        "source": "nasdaq",
                        "source_confidence": "confirmed",
                    }
                ],
                path=path,
            )
            earnings_history.record_yfinance_rows(
                [{"ticker": "NVDA", "date": "2026-05-20", "time": ""}],
                path=path,
            )

            events = earnings_history.get_events_for_symbols(["NVDA"], path=path)["NVDA"]

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["release_session"], "AMC")
        self.assertIn("yfinance", events[0]["sources"])
        self.assertIn("nasdaq", events[0]["sources"])

    def test_gap_inference_records_inferred_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "earnings_calendar_history.json"
            earnings_history.merge_events(
                [
                    {
                        "ticker": "SOBO",
                        "earnings_date": "2026-05-07",
                        "release_session": "TBD",
                        "source": "nasdaq",
                    }
                ],
                path=path,
            )
            earnings_history.record_inferred_release_session("SOBO", "2026-05-07", "AMC", path=path)

            events = earnings_history.get_events_for_symbols(["SOBO"], path=path)["SOBO"]

        self.assertEqual(events[0]["release_session"], "AMC")
        self.assertEqual(events[0]["source_confidence"], "inferred")
        self.assertEqual(events[0]["inferred_release_session"], "inferred_amc")


if __name__ == "__main__":
    unittest.main()
