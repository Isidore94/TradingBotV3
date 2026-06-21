import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_mini_pc import (  # noqa: E402
    _parse_priority_ranked_rows,
    _scan_result_followup_note,
    build_best_trade_snapshot_lines,
)


class MasterAvwapMiniPcStatusTests(unittest.TestCase):
    def test_best_trade_snapshot_places_paste_lists_and_rankings_first(self):
        priority_report = "\n".join(
            [
                "Master AVWAP priority setups",
                "",
                "Ranked by total score",
                "---------------------",
                "NVDA LONG score=245 family=AVWAPE break bucket=favorite_setup",
                "TSLA SHORT score=211 family=post earnings gap bucket=near_favorite_zone",
                "",
                "Detailed setup notes",
                "====================",
                "Best current favorite setups",
                "----------------------------",
                "  NVDA   LONG  score=245   family=AVWAPE break trend=UP zone=upper_band",
                "    signals: fav=Cross up UPPER_2 | context=Prev cross up VWAP",
                "    setup notes: clean zone; tracker edge",
            ]
        )

        lines = build_best_trade_snapshot_lines(
            priority_report=priority_report,
            favorite_symbols=["NVDA"],
            near_favorite_symbols=["TSLA"],
            theta_symbols=["AMD"],
        )

        self.assertEqual(lines[0], "Best Trades First")
        self.assertIn("TV Paste: NVDA, TSLA", lines)
        self.assertIn("Theta Paste: AMD", lines)
        self.assertIn("1. NVDA LONG | score=245 | favorite | AVWAPE break | zone=upper_band | clean zone; tracker edge", lines)
        self.assertIn("2. TSLA SHORT | score=211 | near | post earnings gap", lines)

    def test_priority_rank_parser_stops_after_ranked_block(self):
        rows = _parse_priority_ranked_rows(
            "\n".join(
                [
                    "Ranked by total score",
                    "---------------------",
                    "NVDA LONG score=245 family=AVWAPE break bucket=favorite_setup",
                    "",
                    "TSLA SHORT score=211 family=ignored bucket=near_favorite_zone",
                ]
            )
        )

        self.assertEqual([row["symbol"] for row in rows], ["NVDA"])

    def test_theta_pending_note_only_when_deferred(self):
        self.assertIn("background", _scan_result_followup_note({"theta_enrichment_pending": True}))
        self.assertEqual(_scan_result_followup_note({"theta_enrichment_pending": False}), "")


if __name__ == "__main__":
    unittest.main()
