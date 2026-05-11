import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from market_prep_tab import MarketPrepTab


class FakeVar:
    def __init__(self, value: str):
        self.value = value

    def get(self) -> str:
        return self.value


class MarketPrepTabHelperTests(unittest.TestCase):
    def _tab_with_filter(self, value: str) -> MarketPrepTab:
        tab = MarketPrepTab.__new__(MarketPrepTab)
        tab.catalyst_filter_var = FakeVar(value)
        return tab

    def test_catalyst_filter_hides_low_by_default_threshold(self):
        tab = self._tab_with_filter("medium_plus")
        rows = [
            {"priority": "HIGH", "text": "CPI"},
            {"priority": "MEDIUM", "text": "Auction"},
            {"priority": "LOW", "text": "Minor event"},
            {"priority": "", "text": "Unknown event"},
        ]

        filtered = tab._filter_catalyst_items(rows)

        self.assertEqual([row["text"] for row in filtered], ["CPI", "Auction"])

    def test_catalyst_filter_can_show_only_high_and_mega(self):
        tab = self._tab_with_filter("high")
        rows = [
            {"priority": "MEGA", "text": "NVDA earnings"},
            {"priority": "HIGH", "text": "FOMC"},
            {"priority": "MEDIUM", "text": "Auction"},
        ]

        filtered = tab._filter_catalyst_items(rows)

        self.assertEqual([row["text"] for row in filtered], ["NVDA earnings", "FOMC"])


if __name__ == "__main__":
    unittest.main()
