"""R9.4: `thetalongs.txt`, an optional home-folder list of premium underlyings.

`evaluate_theta_put_candidate` returns ``None`` unless ``side == "LONG"``, and
``side`` is long-watchlist membership (`runner.py`, the scan loop). So a wheeled
underlying that sits on neither trend list is never evaluated at all - which is
why the 2026-07-24..08-21 window's entire positive P&L (+$1,087.72, four DRAM
short puts) was invisible to the engine that exists to find exactly that trade.

The list says "evaluate this one for premium regardless of which trend list it
is on". It is the trader's own file, so plan.md sec 5's never-auto-remove
invariant applies to it like every other watchlist.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402

# Captured before any test patches the module attribute.
REAL_THETA_LONGS_FILE = Path(master_avwap.THETA_LONGS_FILE)


class ThetaLongsFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "thetalongs.txt"
        patcher = patch.object(master_avwap, "THETA_LONGS_FILE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_it_lives_in_the_shared_home_beside_the_other_watchlists(self) -> None:
        from project_paths import LONGS_FILE

        self.assertEqual(REAL_THETA_LONGS_FILE.name, "thetalongs.txt")
        self.assertEqual(REAL_THETA_LONGS_FILE.parent, Path(LONGS_FILE).parent)

    def test_an_absent_file_is_an_empty_list_not_an_error(self) -> None:
        """Optional means optional: no file is the normal state for most desks."""
        self.assertEqual(master_avwap.load_theta_long_symbols(), [])

    def test_names_are_normalised_and_deduplicated(self) -> None:
        self.path.write_text("dram\nDRAM\n  mu  \n\nBRK.B\n", encoding="utf-8")
        self.assertEqual(master_avwap.load_theta_long_symbols(), ["BRK.B", "DRAM", "MU"])

    def test_comments_and_junk_do_not_become_symbols(self) -> None:
        self.path.write_text("# wheel candidates\nDRAM\nnot a symbol\n", encoding="utf-8")
        loaded = master_avwap.load_theta_long_symbols()
        self.assertIn("DRAM", loaded)
        self.assertNotIn("#", "".join(loaded))
        self.assertNotIn("NOT A SYMBOL", loaded)

    def test_an_unreadable_file_costs_the_list_and_never_the_scan(self) -> None:
        """A locked file must not take the whole Master AVWAP run down with it."""
        self.path.write_text("DRAM\n", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=OSError("locked")):
            self.assertEqual(master_avwap.load_theta_long_symbols(), [])


class ThetaSideResolutionTests(unittest.TestCase):
    """The seam itself: which side each surface sees for a given symbol."""

    def test_a_theta_long_is_long_for_theta_and_unchanged_everywhere_else(self) -> None:
        longs, shorts, theta = {"AAPL"}, {"XOM"}, {"XOM", "DRAM"}
        # On the short list AND on thetalongs: still a SHORT for every detector,
        # and a LONG for the premium evaluation. That is the whole point of the
        # list - "regardless of long/short list membership".
        side, theta_side = master_avwap.resolve_scan_sides("XOM", longs, shorts, theta)
        self.assertEqual(side, "SHORT")
        self.assertEqual(theta_side, "LONG")

    def test_an_ordinary_long_is_unaffected(self) -> None:
        side, theta_side = master_avwap.resolve_scan_sides("AAPL", {"AAPL"}, set(), set())
        self.assertEqual((side, theta_side), ("LONG", "LONG"))

    def test_an_ordinary_short_is_unaffected(self) -> None:
        side, theta_side = master_avwap.resolve_scan_sides("XOM", set(), {"XOM"}, set())
        self.assertEqual((side, theta_side), ("SHORT", "SHORT"))

    def test_a_theta_only_name_is_a_long_rather_than_a_phantom_short(self) -> None:
        """It is reachable only via thetalongs.txt, which is a long-side list.

        Falling through to SHORT would hand every other detector a bearish
        thesis on a name the trader is bullish enough on to sell puts against.
        """
        side, theta_side = master_avwap.resolve_scan_sides("DRAM", set(), set(), {"DRAM"})
        self.assertEqual((side, theta_side), ("LONG", "LONG"))

    def test_an_empty_list_changes_no_side_at_all(self) -> None:
        """The characterization guarantee: no file, no behavior change.

        Adding symbols to a scan changes what the detectors see, so the absent
        and empty cases have to be provably inert before the populated case is
        allowed to be interesting.
        """
        for symbol, longs, shorts, expected in (
            ("AAPL", {"AAPL"}, set(), "LONG"),
            ("XOM", set(), {"XOM"}, "SHORT"),
        ):
            side, theta_side = master_avwap.resolve_scan_sides(symbol, longs, shorts, set())
            self.assertEqual(side, expected, symbol)
            self.assertEqual(theta_side, expected, symbol)



class ThetaReportProvenanceTests(unittest.TestCase):
    """A LONG-only section has to say why a non-long name is in it."""

    @staticmethod
    def _row(symbol: str, source: str | None) -> dict:
        row = {
            "symbol": symbol,
            "last_close": 57.68,
            "score": 40,
            "support_count": 4,
            "option_status": "recommended",
            "best_option": {"status": "recommended", "credit": 1.10},
        }
        if source is not None:
            row["theta_list_source"] = source
        return row

    def _render(self, rows: list[dict]) -> str:
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "theta.txt"
            master_avwap.write_theta_put_report(out, rows, [])
            return out.read_text(encoding="utf-8")

    def test_a_thetalongs_row_is_labelled(self) -> None:
        text = self._render([self._row("DRAM", "thetalongs")])
        self.assertIn("DRAM", text)
        self.assertIn("via thetalongs.txt", text)

    def test_an_ordinary_watchlist_row_carries_no_label(self) -> None:
        text = self._render([self._row("AAPL", "watchlist")])
        self.assertIn("AAPL", text)
        self.assertNotIn("via thetalongs.txt", text)

    def test_a_row_from_before_this_change_still_renders(self) -> None:
        """Provenance is additive: an older row without the key is not a crash."""
        text = self._render([self._row("MU", None)])
        self.assertIn("MU", text)
        self.assertNotIn("via thetalongs.txt", text)

    def test_the_rules_header_names_the_list(self) -> None:
        text = self._render([])
        self.assertIn("thetalongs.txt", text)


if __name__ == "__main__":
    unittest.main()
