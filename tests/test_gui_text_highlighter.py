import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gui_text_highlighter import line_tag, tree_tags_for_values


class GuiTextHighlighterTests(unittest.TestCase):
    def test_line_tags_common_trading_output(self):
        self.assertEqual(line_tag("MASTER AVWAP PRIORITY SETUPS"), "mt_header")
        self.assertEqual(line_tag("LONG: AAPL, MSFT, NVDA"), "mt_long")
        self.assertEqual(line_tag("SHORT: TSLA, RIVN"), "mt_short")
        self.assertEqual(line_tag("OpenAI request failed with HTTP 400"), "mt_error")
        self.assertEqual(line_tag("edge=+0.42R win=61%"), "mt_positive")
        self.assertEqual(line_tag("edge=-0.25R win=41%"), "mt_negative")

    def test_tree_tags_common_values(self):
        self.assertIn("mt_long", tree_tags_for_values(("NVDA", "LONG", "OPEN", "+0.50")))
        self.assertIn("mt_short", tree_tags_for_values(("TSLA", "SHORT", "WATCH")))
        self.assertIn("mt_error", tree_tags_for_values(("AMD", "CLOSED", "STOP")))


if __name__ == "__main__":
    unittest.main()
