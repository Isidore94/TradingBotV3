"""The Chart Review workspace: lookup, the capture rail, and what stays put.

The invariant under test here is the mirror of plan.md sec 5's "user-entered
watchlist names are never auto-removed": looking at a symbol must never *add*
one either. A lookup that quietly wrote to longs.txt or the CandidateRegistry
would put names into the scan universe that the trader never chose, so the
tests below check it three ways - the module imports no writer, a full lookup
cycle touches nothing but its own machine-local recents file, and the recents
file is not in the shared home where the watchlists live.

The rest covers the rail's contract: each action writes exactly one annotation
row, a like goes through the EXISTING focus machinery rather than a second
likes store, and a capture that fails to reach disk says so instead of looking
like it worked.
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui.services.symbol_lookup import (  # noqa: E402
    MAX_RECENT_LOOKUPS,
    RECENT_LOOKUPS_FILE,
    RecentLookups,
    is_lookupable,
    normalize_symbol,
)

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


class _SpyFocusService:
    """Stands in for FocusService and records what the rail asked of it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def add(self, symbol, side, category="m5", *, origin="", context="") -> bool:
        self.calls.append((symbol, side, category, origin, context))
        return True


def _panel(tmp: Path, *, focus_service=None):
    from ui.panels.chart_review_panel import ChartReviewPanel

    return ChartReviewPanel(
        focus_service=focus_service,
        recent_lookups=RecentLookups(tmp / "recents.json"),
        annotations_path=tmp / "trader_annotations.jsonl",
        setup_tracker_path=tmp / "setup_tracker.json",
    )


class SymbolNormalizationTests(unittest.TestCase):
    def test_plain_tickers(self) -> None:
        self.assertEqual(normalize_symbol("nvda"), "NVDA")
        self.assertEqual(normalize_symbol("  spy  "), "SPY")
        self.assertEqual(normalize_symbol("$AMD"), "AMD")

    def test_class_shares_and_hyphens(self) -> None:
        self.assertEqual(normalize_symbol("brk.b"), "BRK.B")
        self.assertEqual(normalize_symbol("RDS-A"), "RDS-A")

    def test_rejects_non_symbols(self) -> None:
        for bad in ("", "   ", "not a symbol", "AAPL CALL 200", "../etc/passwd", "A" * 20, "9NVDA"):
            self.assertEqual(normalize_symbol(bad), "", bad)
            self.assertFalse(is_lookupable(bad), bad)


class RecentLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "recents.json"
        self.addCleanup(self._tmp.cleanup)

    def test_most_recent_first_without_duplicates(self) -> None:
        recents = RecentLookups(self.path)
        for symbol in ("AAA", "BBB", "AAA"):
            recents.remember(symbol)
        self.assertEqual(recents.symbols(), ["AAA", "BBB"])

    def test_capped(self) -> None:
        recents = RecentLookups(self.path, limit=3)
        for index in range(6):
            recents.remember(f"SYM{index}")
        self.assertEqual(recents.symbols(), ["SYM5", "SYM4", "SYM3"])

    def test_persisted_and_reloaded(self) -> None:
        RecentLookups(self.path).remember("NVDA")
        self.assertEqual(RecentLookups(self.path).symbols(), ["NVDA"])

    def test_junk_is_not_remembered(self) -> None:
        recents = RecentLookups(self.path)
        self.assertEqual(recents.remember("not a symbol"), "")
        self.assertEqual(recents.symbols(), [])

    def test_corrupt_file_reads_as_empty(self) -> None:
        self.path.write_text("{broken", encoding="utf-8")
        self.assertEqual(RecentLookups(self.path).symbols(), [])

    def test_default_limit_is_positive(self) -> None:
        self.assertGreater(MAX_RECENT_LOOKUPS, 0)


class LookupNeverWritesWatchlistsTests(unittest.TestCase):
    """plan.md sec 5's mirror: a lookup adds nothing to any list."""

    def test_the_module_imports_no_watchlist_writer(self) -> None:
        source = (SCRIPTS_DIR / "ui" / "services" / "symbol_lookup.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        forbidden = (
            "candidate_registry",
            "focus_picks",
            "watchlist",
            "ui.services.focus_service",
        )
        for name in forbidden:
            self.assertFalse(
                any(name in entry for entry in imported),
                f"symbol_lookup must not import {name}: {sorted(imported)}",
            )

    def test_the_module_names_no_watchlist_path(self) -> None:
        source = (SCRIPTS_DIR / "ui" / "services" / "symbol_lookup.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.alias)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for forbidden in ("LONGS_FILE", "SHORTS_FILE", "SWING_LONGS_FILE", "SWING_SHORTS_FILE",
                          "FOCUS_LONGS_FILE", "FOCUS_SHORTS_FILE", "UNIVERSE_LONGS_FILE"):
            self.assertNotIn(forbidden, names)

    def test_recents_live_machine_local_not_in_the_shared_home(self) -> None:
        """The watchlists are in the shared home; the recents cache is not."""
        from project_paths import LOCAL_SETTINGS_DIR, PERSISTENT_DATA_DIR

        self.assertTrue(
            str(RECENT_LOOKUPS_FILE).startswith(str(LOCAL_SETTINGS_DIR)),
            RECENT_LOOKUPS_FILE,
        )
        self.assertFalse(
            str(RECENT_LOOKUPS_FILE).startswith(str(PERSISTENT_DATA_DIR)),
            RECENT_LOOKUPS_FILE,
        )

    def test_a_lookup_writes_only_its_own_recents_file(self) -> None:
        with TemporaryDirectory() as name:
            tmp = Path(name)
            panel = _panel(tmp)
            self.assertEqual(panel.open_symbol("nvda"), "NVDA")
            self.assertEqual(panel.open_symbol("brk.b"), "BRK.B")
            written = {path.name for path in tmp.iterdir()}
            self.assertEqual(written, {"recents.json"})

    def test_a_lookup_does_not_touch_the_focus_service(self) -> None:
        with TemporaryDirectory() as name:
            spy = _SpyFocusService()
            panel = _panel(Path(name), focus_service=spy)
            panel.open_symbol("NVDA")
            panel.open_symbol("AMD")
            self.assertEqual(spy.calls, [])


class WorkspaceLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_setups_drawer_is_hidden_by_default(self) -> None:
        """The trader's ask: AVWAP setups matter less early in the day."""
        panel = _panel(self.tmp)
        self.assertFalse(panel.setups_visible())
        self.assertFalse(panel.setups_button.isChecked())

    def test_setups_button_slides_the_drawer_in_and_out(self) -> None:
        panel = _panel(self.tmp)
        self.assertTrue(panel.toggle_setups())
        self.assertTrue(panel.setups_visible())
        self.assertFalse(panel.toggle_setups())
        self.assertFalse(panel.setups_visible())

    def test_drawer_survives_an_unreadable_tracker(self) -> None:
        panel = _panel(self.tmp)
        panel.set_setups_visible(True)
        self.assertIn("not readable", panel.setups_body.text().lower())

    def test_drawer_renders_tracked_setups(self) -> None:
        (self.tmp / "setup_tracker.json").write_text(
            json.dumps(
                {"updated_at": "2026-08-07T16:00:00", "setups": {"NVDA": {"setup_family": "avwape_to_1stdev"}}}
            ),
            encoding="utf-8",
        )
        panel = _panel(self.tmp)
        panel.set_setups_visible(True)
        self.assertIn("NVDA", panel.setups_body.text())
        self.assertIn("avwape_to_1stdev", panel.setups_body.text())

    def test_recent_chips_reopen_a_symbol(self) -> None:
        panel = _panel(self.tmp)
        panel.open_symbol("NVDA")
        panel.open_symbol("AMD")
        self.assertEqual(panel.recent_symbols(), ["AMD", "NVDA"])
        panel.open_symbol("NVDA")
        self.assertEqual(panel.symbol, "NVDA")

    def test_rail_is_disarmed_until_a_symbol_is_open(self) -> None:
        panel = _panel(self.tmp)
        self.assertFalse(panel.capture_rail.veto_button.isEnabled())
        panel.open_symbol("NVDA")
        self.assertTrue(panel.capture_rail.veto_button.isEnabled())


class CaptureRailTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "trader_annotations.jsonl"
        self.addCleanup(self._tmp.cleanup)
        self.spy = _SpyFocusService()
        self.panel = _panel(self.tmp, focus_service=self.spy)
        self.panel.open_symbol("NVDA")
        self.rail = self.panel.capture_rail

    def _rows(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_veto_in_two_interactions(self) -> None:
        """Alt+V then a digit. select_reason is what the digit shortcut runs."""
        self.rail.focus_veto()
        self.rail.select_reason("volume_dry")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_type"], "veto")
        self.assertEqual(rows[0]["reason_code"], "volume_dry")
        self.assertEqual(rows[0]["symbol"], "NVDA")
        self.assertEqual(rows[0]["source"], "chart_review")

    def test_other_waits_for_the_note_instead_of_writing(self) -> None:
        self.rail.select_reason("other")
        self.assertEqual(self._rows(), [])
        self.assertIn("note", self.rail.status_text().lower())
        self.rail.veto_note_input.setText("gapped on an upgrade")
        self.rail.commit_veto()
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["note"], "gapped on an upgrade")

    def test_veto_without_a_reason_writes_nothing(self) -> None:
        self.rail.reason_list.setCurrentRow(-1)
        self.assertIsNone(self.rail.commit_veto())
        self.assertEqual(self._rows(), [])

    def test_veto_records_the_side_for_forward_tracking(self) -> None:
        self.rail.side_input.setCurrentText("SHORT")
        self.rail.select_reason("too_extended_from_base")
        self.assertEqual(self._rows()[0]["side"], "SHORT")

    def test_like_goes_through_the_existing_focus_machinery(self) -> None:
        """Extend, do not build a parallel likes system."""
        self.rail.setup_input.setCurrentIndex(0)
        claimed = self.rail.setup_input.currentData()
        self.rail.commit_like()
        self.assertEqual(len(self.spy.calls), 1)
        symbol, side, category, origin, context = self.spy.calls[0]
        self.assertEqual(symbol, "NVDA")
        self.assertEqual(side, "LONG")
        self.assertEqual(category, "swing")
        self.assertEqual(origin, "chart_review")
        self.assertIn(claimed, context)
        rows = self._rows()
        self.assertEqual(rows[0]["event_type"], "like_claim")
        self.assertEqual(rows[0]["claimed_setup_id"], claimed)

    def test_chart_review_is_a_documented_pick_origin(self) -> None:
        from pick_feedback import PICK_ORIGINS

        self.assertIn("chart_review", PICK_ORIGINS)

    def test_hypothetical_stop_records_a_price_and_no_order(self) -> None:
        self.rail.stop_input.setValue(101.25)
        self.rail.commit_hypo_stop()
        row = self._rows()[0]
        self.assertEqual(row["event_type"], "hypo_stop")
        self.assertEqual(row["stop_price"], 101.25)
        self.assertEqual(row["side"], "LONG")
        self.assertIn("no order", self.rail.status_text().lower())

    def test_hypothetical_stop_needs_a_price(self) -> None:
        self.rail.stop_input.setValue(0.0)
        self.assertIsNone(self.rail.commit_hypo_stop())
        self.assertEqual(self._rows(), [])

    def test_note_is_recorded_and_cleared(self) -> None:
        self.rail.note_input.setText("watching the 200 into the close")
        self.rail.commit_note()
        row = self._rows()[0]
        self.assertEqual(row["event_type"], "note")
        self.assertEqual(row["note"], "watching the 200 into the close")
        self.assertEqual(self.rail.note_input.text(), "")

    def test_empty_note_writes_nothing(self) -> None:
        self.rail.note_input.setText("   ")
        self.assertIsNone(self.rail.commit_note())
        self.assertEqual(self._rows(), [])

    def test_a_capture_emits_the_captured_signal(self) -> None:
        seen: list[tuple] = []
        self.rail.captured.connect(lambda kind, row: seen.append((kind, row["symbol"])))
        self.rail.select_reason("volume_dry")
        self.assertEqual(seen, [("veto", "NVDA")])

    def test_a_failed_write_is_shown_not_swallowed(self) -> None:
        blocked = self.tmp / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        self.rail._annotations_path = blocked / "x.jsonl"
        self.rail.note_input.setText("this will not land")
        self.assertIsNone(self.rail.commit_note())
        self.assertIn("not saved", self.rail.status_text().lower())

    def test_a_veto_creates_its_forward_tracking_cohort_row(self) -> None:
        picks = self.tmp / "veto_cohort_picks.csv"
        calls: list[dict] = []

        def _merge(**kwargs):
            calls.append(kwargs)
            from ui.annotations.veto_cohort import merge_veto_cohort_picks

            return merge_veto_cohort_picks(picks_path=picks, **kwargs)

        self.rail._merge_veto_cohort = _merge
        self.rail.select_reason("incoming_trendline")
        self.assertEqual(len(calls), 1)
        self.assertTrue(picks.exists())
        self.assertIn("veto_incoming_trendline", picks.read_text(encoding="utf-8"))


class NavigationRegistrationTests(unittest.TestCase):
    """Adding a page shifts every later index; these must stay aligned."""

    def test_nav_items_pages_and_titles_agree(self) -> None:
        source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
        self.assertIn('("Chart Review", "mdi.chart-line")', source)
        self.assertIn("self.pages.addWidget(self.chart_review_panel)", source)
        # Chart Review sits directly after Trading Desk in both lists.
        nav_index = source.index('("Chart Review"')
        focus_nav_index = source.index('("Focus Picks"')
        self.assertLess(nav_index, focus_nav_index)
        title_index = source.index('            "Chart Review",')
        focus_title_index = source.index('            "Focus Picks",')
        self.assertLess(title_index, focus_title_index)

    def test_trading_desk_is_still_page_zero(self) -> None:
        """F9's setups expand selects page 0 explicitly."""
        source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
        desk_add = source.index("self.pages.addWidget(self.trading_panel)")
        review_add = source.index("self.pages.addWidget(self.chart_review_panel)")
        self.assertLess(desk_add, review_add)


if __name__ == "__main__":
    unittest.main()
