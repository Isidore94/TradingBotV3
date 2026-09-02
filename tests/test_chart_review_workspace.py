"""The Chart Review workspace: lookup, the capture rail, and what stays put.

The invariant under test here is the mirror of plan.md sec 5's "user-entered
watchlist names are never auto-removed": looking at a symbol must never *add*
one either - and neither may CAPTURING a judgement about it. A lookup or a
like that quietly wrote to longs.txt, the CandidateRegistry, or a Focus list
would put names into the scan universe (and give them alert privileges) that
the trader never explicitly granted, so the tests below check it several
ways - the modules import no writer, a full lookup cycle touches nothing but
its own machine-local recents file, a like writes exactly one annotation row
and nothing else, and the recents file is not in the shared home where the
watchlists live.

The rest covers the rail's contract: each action writes exactly one annotation
row, and a capture that fails to reach disk says so instead of looking like it
worked. The Setups drawer reads the compact scoring snapshot off the GUI
thread, bounded by a byte ceiling - never the raw 762MB tracker.
"""

from __future__ import annotations

import ast
import json
import sys
import time
import unittest
from datetime import datetime
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


def _panel(tmp: Path):
    from ui.panels.chart_review_panel import ChartReviewPanel

    return ChartReviewPanel(
        recent_lookups=RecentLookups(tmp / "recents.json"),
        annotations_path=tmp / "trader_annotations.jsonl",
        setups_snapshot_path=tmp / "setups_snapshot.json",
    )


def _pump_until(predicate, timeout=10.0):
    """Drive the event loop until ``predicate`` holds (drawer reads are async)."""
    app = _QT.QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _drawer_settled(panel) -> bool:
    return bool(
        panel.setups_body.text()
        and "Reading setups snapshot" not in panel.setups_body.text()
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

    def test_the_workspace_modules_import_no_focus_writer(self) -> None:
        """The panel and the rail can not write a Focus list they never import.

        An earlier draft handed the rail a FocusService and a like wrote a
        swing watchlist entry with Focus alert privileges - a capture surface
        crossing into live behavior. This pins the repair at the import level.
        """
        for relative in (
            Path("ui") / "panels" / "chart_review_panel.py",
            Path("ui") / "widgets" / "capture_rail.py",
        ):
            source = (SCRIPTS_DIR / relative).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
                    imported.update(f"{node.module}.{alias.name}" for alias in node.names)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
            for forbidden in ("focus_picks", "focus_service", "candidate_registry", "watchlist"):
                self.assertFalse(
                    any(forbidden in entry for entry in imported),
                    f"{relative} must not import {forbidden}: {sorted(imported)}",
                )
            self.assertNotIn("focus_service", source, f"{relative} still names a focus service")


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

    def test_drawer_survives_an_unreadable_snapshot(self) -> None:
        panel = _panel(self.tmp)
        panel.set_setups_visible(True)
        self.assertTrue(_pump_until(lambda: _drawer_settled(panel)))
        self.assertIn("not readable", panel.setups_body.text().lower())

    def test_drawer_renders_symbols_from_production_shaped_snapshot(self) -> None:
        """The snapshot's ``setups`` keys are setup ids, NOT symbols.

        Production ids look like ``date:symbol:side:anchor:bucket`` (built in
        master_avwap_lib). A drawer that printed sorted keys would show ids in
        historical order; it must show each row's symbol, newest scan first.
        """
        (self.tmp / "setups_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_at": "2026-08-07T16:05:00",
                    "source_updated_at": "2026-08-07T16:00:00",
                    "setups": {
                        "2026-03-02:AAA:LONG:2026-02-20:tracked": {
                            "symbol": "AAA",
                            "setup_family": "old_family",
                            "scan_date": "2026-03-02",
                        },
                        "2026-08-07:NVDA:LONG:2026-07-30:favorite_setup": {
                            "symbol": "NVDA",
                            "setup_family": "avwape_to_1stdev",
                            "scan_date": "2026-08-07",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        panel = _panel(self.tmp)
        panel.set_setups_visible(True)
        self.assertTrue(_pump_until(lambda: _drawer_settled(panel)))
        text = panel.setups_body.text()
        self.assertIn("NVDA", text)
        self.assertIn("avwape_to_1stdev", text)
        self.assertIn("as of 2026-08-07T16:00:00", text)
        self.assertNotIn("2026-08-07:NVDA", text, "the drawer must not print raw setup ids")
        self.assertLess(text.index("NVDA"), text.index("AAA"), "newest scan date first")

    def test_drawer_never_reads_the_raw_tracker_and_never_reads_on_the_gui_thread(self) -> None:
        """Two pins in one: the panel's source names only the compact scoring
        snapshot (the raw tracker was measured at 762MB), and the read runs
        via the pool-thread task, not inline in set_setups_visible."""
        source = (SCRIPTS_DIR / "ui" / "panels" / "chart_review_panel.py").read_text(encoding="utf-8")
        self.assertNotIn("MASTER_AVWAP_SETUP_TRACKER_FILE", source)
        self.assertIn("MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE", source)

        from ui.panels import chart_review_panel as module

        reads: list[str] = []
        original = module.read_setups_summary

        def _spy(path, **kwargs):
            import threading

            reads.append(threading.current_thread().name)
            return original(path, **kwargs)

        module.read_setups_summary = _spy
        try:
            panel = _panel(self.tmp)
            panel.set_setups_visible(True)
            self.assertTrue(_pump_until(lambda: _drawer_settled(panel)))
        finally:
            module.read_setups_summary = original
        self.assertTrue(reads)
        self.assertNotIn("MainThread", reads)

    def test_snapshot_past_the_byte_ceiling_is_refused_not_parsed(self) -> None:
        from ui.panels.chart_review_panel import read_setups_summary

        big = self.tmp / "setups_snapshot.json"
        big.write_text('{"setups": {"x": {}}}', encoding="utf-8")
        text = read_setups_summary(big, max_bytes=4)
        self.assertIn("refusing to parse", text)

    def test_summary_row_cap_is_reported(self) -> None:
        from ui.panels.chart_review_panel import read_setups_summary

        path = self.tmp / "setups_snapshot.json"
        path.write_text(
            json.dumps(
                {
                    "setups": {
                        f"2026-08-0{1 + index % 7}:SYM{index}:LONG:a:tracked": {
                            "symbol": f"SYM{index}",
                            "scan_date": f"2026-08-0{1 + index % 7}",
                        }
                        for index in range(5)
                    }
                }
            ),
            encoding="utf-8",
        )
        text = read_setups_summary(path, max_rows=2)
        self.assertIn("... and 3 more", text)

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

    def test_chart_area_hosts_shared_snapshot_on_the_one_data_path(self) -> None:
        from ui.panels.chart_review_panel import CHART_REVIEW_D1_SESSIONS
        from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

        panel = _panel(self.tmp)
        self.assertIsInstance(panel.snapshot, SymbolSnapshotWidget)
        self.assertTrue(panel.snapshot._compact)
        self.assertEqual(panel.snapshot._d1_sessions, CHART_REVIEW_D1_SESSIONS)
        self.assertGreaterEqual(CHART_REVIEW_D1_SESSIONS, 520)

        source = (
            SCRIPTS_DIR / "ui" / "panels" / "chart_review_panel.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertIn("ui.widgets.symbol_snapshot_dialog", imports)
        self.assertNotIn("chart_snapshot", imports)
        self.assertNotIn("ui.services.bar_cache", imports)
        self.assertNotIn("ui.services.chart_data_service", imports)

    def test_painted_level_flows_to_the_next_capture_fields(self) -> None:
        panel = _panel(self.tmp)
        panel.open_symbol("NVDA")
        panel._on_d1_level_selected("NVDA", "hv:NVDA:42", "hv_horizontal", 101.5)
        panel.capture_rail.setup_list.setCurrentRow(0)
        panel.capture_rail.like_note_input.setText("level held")  # R9.2: why required
        row = panel.capture_rail.commit_like()
        self.assertIsNotNone(row)
        self.assertEqual(row["ref_level_id"], "hv:NVDA:42")
        self.assertEqual(row["ref_level_family"], "hv_horizontal")
        self.assertEqual(row["timeframe"], "D1")

    def test_provenance_strip_makes_yfinance_fallback_loud(self) -> None:
        from ui.panels.chart_review_panel import provenance_state

        panel = _panel(self.tmp)
        panel._symbol = "NVDA"
        meta = {
            "source": "yfinance-fallback",
            "storage_tier": "local",
            "bar_timestamp": datetime(2026, 8, 9, 9, 30),
            "bar_timeframe": "M5",
        }
        text, degraded = provenance_state(meta, now=datetime(2026, 8, 9, 10, 0))
        self.assertTrue(degraded)
        self.assertIn("YFINANCE FALLBACK", text)
        self.assertIn("M5 age 30m", text)
        panel._on_snapshot_meta("NVDA", meta)
        self.assertTrue(panel.provenance_label.property("degraded"))
        self.assertIn("font-weight: 800", panel.provenance_label.styleSheet())

        healthy, degraded = provenance_state(
            meta | {"source": "ibkr-cache"}, now=datetime(2026, 8, 9, 10, 0)
        )
        self.assertFalse(degraded)
        self.assertIn("IBKR live cache", healthy)

    def test_workspace_snapshot_exposes_no_alert_arming_affordance(self) -> None:
        panel = _panel(self.tmp)
        self.assertFalse(panel.snapshot._allow_alerts)
        emitted: list[tuple] = []
        panel.snapshot.d1LevelAlertRequested.connect(lambda *args: emitted.append(args))
        panel.snapshot._symbol = "NVDA"
        panel.snapshot.d1_chart.set_data(
            [
                {
                    "dt": datetime(2026, 8, 8),
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 10_000,
                }
            ],
            timeframe="d1",
        )
        panel.snapshot.request_d1_level_alert("above", 0)
        self.assertEqual(emitted, [])
        button_text = [
            button.text().lower()
            for button in panel.snapshot.findChildren(_QT.QPushButton)
        ]
        self.assertFalse(any("alert" in text or "arm" in text for text in button_text))


class CaptureRailTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "trader_annotations.jsonl"
        self.addCleanup(self._tmp.cleanup)
        self.panel = _panel(self.tmp)
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

    def test_like_records_the_claim_and_writes_nothing_else(self) -> None:
        """A like is a recorded judgement - one annotation row, no side effects.

        The earlier draft routed likes through FocusService.add, which put the
        symbol into a swing watchlist and gave it Focus alert privileges - a
        capture surface crossing the analysis-only boundary. This pins the
        repair behaviorally: after a like, the working directory holds only
        the annotation log and the lookup's own recents file.
        """
        self.rail.setup_list.setCurrentRow(0)
        claimed = self.rail.selected_setup_id()
        self.rail.like_note_input.setText("level held")  # R9.2: why required
        self.rail.commit_like()
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_type"], "like_claim")
        self.assertEqual(rows[0]["claimed_setup_id"], claimed)
        self.assertEqual(rows[0]["symbol"], "NVDA")
        written = {path.name for path in self.tmp.iterdir()}
        self.assertEqual(written, {"trader_annotations.jsonl", "recents.json"})

    def test_the_hypothetical_stop_control_is_gone_but_its_rows_still_parse(self) -> None:
        """Trader, 2026-08-20: "get rid of hypothetical stop for now its not
        useful."

        The CONTROL is what was removed. `ui.annotations.store` still builds
        and validates hypo_stop rows, because the annotation stream is
        append-only evidence: deleting the schema would make rows already on
        disk unreadable to buy nothing. Re-adding the control later is a
        layout change, not a migration.
        """
        from ui.annotations.store import EVENT_HYPO_STOP, EVENT_TYPES, build_annotation

        self.assertFalse(hasattr(self.rail, "stop_input"))
        self.assertFalse(hasattr(self.rail, "commit_hypo_stop"))
        self.assertNotIn("Alt+S", dict(self.rail.action_shortcuts()))

        self.assertIn(EVENT_HYPO_STOP, EVENT_TYPES)
        row = build_annotation(
            EVENT_HYPO_STOP, symbol="NVDA", stop_price=101.25, side="LONG"
        )
        self.assertEqual(row["stop_price"], 101.25)

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
        # The cohort key carries its vocabulary version (2026-08-20): a code
        # is only guaranteed stable WITHIN a vocabulary, so pooling two
        # versions under one key would average two judgements into a number
        # that reads as evidence. Asserted against the LOADED vocabulary, never
        # a literal version - a bump must not have to edit this test.
        from ui.annotations.vocabulary import load_veto_vocabulary

        version = load_veto_vocabulary().vocab_version
        self.assertIn(
            f"veto_v{version}_incoming_trendline", picks.read_text(encoding="utf-8")
        )


    def test_a_like_creates_its_forward_tracking_cohort_row_on_the_click(self) -> None:
        """The LIKE half of the same decision, merged on the same click.

        It used to be nightly-only: `like_cohort_picks.csv` was last written
        2026-08-27 against likes recorded through 2026-09-01, so a like was
        invisible to its own cohort for up to a day - and on any day the
        overnight job did not run, indefinitely. The veto has merged at click
        time since it shipped; these are read side by side, so a difference
        between them has to come from the data.

        Fail-before-fix: on the un-fixed rail the merge is never called and the
        picks file is never created.
        """
        picks = self.tmp / "like_cohort_picks.csv"
        calls: list[dict] = []

        def _merge(**kwargs):
            calls.append(kwargs)
            from ui.annotations.like_cohort import merge_like_cohort_picks

            return merge_like_cohort_picks(picks_path=picks, **kwargs)

        self.rail._merge_like_cohort = _merge
        self.rail.setup_list.setCurrentRow(0)
        claimed = self.rail.selected_setup_id()
        self.rail.like_note_input.setText("level held")
        self.assertIsNotNone(self.rail.commit_like())

        self.assertEqual(len(calls), 1)
        self.assertTrue(picks.exists())
        self.assertIn(f"like_{claimed}", picks.read_text(encoding="utf-8"))

    def test_a_failed_like_cohort_merge_never_costs_the_like(self) -> None:
        """An evidence store never costs the event it records. The annotation
        row is already on disk when the merge runs, so a merge that raises
        degrades to a status suffix and the next call picks the row up."""

        def _explode(**kwargs):
            raise RuntimeError("disk gone")

        self.rail._merge_like_cohort = _explode
        self.rail.setup_list.setCurrentRow(0)
        self.rail.like_note_input.setText("level held")
        row = self.rail.commit_like()

        self.assertIsNotNone(row)
        self.assertEqual(len(self._rows()), 1)
        self.assertIn("deferred", self.rail.status_text())

    def test_a_like_merge_that_cannot_write_says_so_exactly_as_a_veto_does(self) -> None:
        self.rail._merge_like_cohort = lambda **kwargs: {"written": False}
        self.rail.setup_list.setCurrentRow(0)
        self.rail.like_note_input.setText("level held")
        self.assertIsNotNone(self.rail.commit_like())
        self.assertIn("deferred", self.rail.status_text())


class NavigationRegistrationTests(unittest.TestCase):
    """Adding a page shifts every later index; these must stay aligned.

    Rewritten in R8 §9 step 1 against ``PAGE_SPECS``. The previous versions
    grepped ``app.py`` for the three parallel structures and compared the
    *positions of two string literals* - Chart Review before Focus Picks, at
    indices 1 and 2. Both passed, uninterrupted, while the titles tuple ran a
    full entry short from index 3 onward and ``Settings`` raised IndexError.

    Checking two adjacent early entries is what gave that false confidence, so
    these now check the whole list. ``tests/test_qt_page_specs.py`` carries the
    per-index behavioural half.
    """

    def _specs(self):
        from ui.app import PAGE_SPECS

        return PAGE_SPECS

    def test_chart_review_sits_directly_after_the_trading_desk(self) -> None:
        titles = [spec.title for spec in self._specs()]
        self.assertEqual(titles[:3], ["Trading Desk", "Chart Review", "Focus Picks"])

    def test_every_page_is_declared_exactly_once(self) -> None:
        """The alignment these tests were always about, checked end to end."""
        specs = self._specs()
        titles = [spec.title for spec in specs]
        attributes = [spec.attribute for spec in specs]
        self.assertEqual(len(set(titles)), len(titles))
        self.assertEqual(len(set(attributes)), len(attributes))
        # One structure means a page cannot be half-added; the old failure mode
        # needed three, so guard against a second one reappearing.
        source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("nav_items = (", source)
        self.assertNotIn("titles = (", source)

    def test_trading_desk_is_still_page_zero(self) -> None:
        """F9's setups expand selects page 0 explicitly."""
        self.assertEqual(self._specs()[0].title, "Trading Desk")
        self.assertEqual(self._specs()[0].attribute, "trading_panel")


if __name__ == "__main__":
    unittest.main()


class VetoVocabularyTests(unittest.TestCase):
    """What every shipped vocabulary must still be true of.

    v2 replaced the S/R slot with "Compressed" (trader, 2026-08-20) and v3
    added "SMA incoming" (trader, 2026-08-21). Both were NEW codes, not
    renames. v1's own description states the rule: "a code is never renamed or
    reused for a different meaning, because rows already written carry it".
    "S/R cluttered" meant too many levels in the path; "compressed" means the
    range is too tight to work with. Different judgements, so reusing the code
    would silently re-label history.

    These assert against the newest vocabulary present rather than a literal
    version number, which is the repo rule (CLAUDE.md): the next bump should
    not have to come and edit them.
    """

    def setUp(self) -> None:
        from ui.annotations.vocabulary import clear_vocabulary_cache

        clear_vocabulary_cache()

    def test_the_newest_vocabulary_is_the_default_and_carries_compressed(self) -> None:
        from ui.annotations.vocabulary import (
            available_veto_versions,
            load_veto_vocabulary,
        )

        vocab = load_veto_vocabulary()
        self.assertEqual(vocab.vocab_version, max(available_veto_versions()))
        self.assertIsNotNone(vocab.reason("compressed"))
        self.assertEqual(vocab.reason("compressed").label, "Compressed")

    def test_the_newest_vocabulary_carries_the_sma_reason(self) -> None:
        """v3, trader 2026-08-21. Its hotkey is 0 because 1-9 were already
        spoken for and renumbering them would break eight learned digits."""
        from ui.annotations.vocabulary import load_veto_vocabulary

        reason = load_veto_vocabulary().reason("sma_incoming")
        self.assertIsNotNone(reason)
        self.assertEqual(reason.label, "SMA incoming")
        self.assertEqual(reason.hotkey, "0")
        self.assertFalse(reason.note_required)

    def test_every_version_keeps_its_hotkeys_unique(self) -> None:
        from ui.annotations.vocabulary import (
            available_veto_versions,
            load_veto_vocabulary,
        )

        for version in available_veto_versions():
            vocab = load_veto_vocabulary(version=version)
            keys = [reason.hotkey for reason in vocab.reasons]
            self.assertEqual(len(keys), len(set(keys)), f"v{version} reuses a hotkey")

    def test_the_old_code_is_gone_from_the_newest_vocabulary(self) -> None:
        from ui.annotations.vocabulary import load_veto_vocabulary

        self.assertIsNone(
            load_veto_vocabulary().reason("support_resistance_cluttered")
        )

    def test_v1_is_still_loadable_so_old_rows_stay_readable(self) -> None:
        from ui.annotations.vocabulary import load_veto_vocabulary

        v1 = load_veto_vocabulary(version=1)
        self.assertEqual(v1.vocab_version, 1)
        reason = v1.reason("support_resistance_cluttered")
        self.assertIsNotNone(reason)
        self.assertEqual(reason.label, "S/R cluttered")

    def test_the_new_code_never_appears_in_v1(self) -> None:
        """The other half of "a code is never reused": if 'compressed' existed
        in v1 with another meaning, a v1-stamped row would be misread."""
        from ui.annotations.vocabulary import load_veto_vocabulary

        self.assertIsNone(load_veto_vocabulary(version=1).reason("compressed"))

    def test_every_surviving_code_kept_its_meaning_and_its_digit(self) -> None:
        from ui.annotations.vocabulary import load_veto_vocabulary

        v1 = {r.code: r for r in load_veto_vocabulary(version=1).reasons}
        v2 = {r.code: r for r in load_veto_vocabulary(version=2).reasons}
        for code, reason in v2.items():
            if code == "compressed":
                continue
            self.assertIn(code, v1, f"{code} appeared without a v1 counterpart")
            self.assertEqual(reason.label, v1[code].label)
            self.assertEqual(
                reason.hotkey, v1[code].hotkey, f"{code} moved digit - muscle memory"
            )


class PassCohortMergeTests(unittest.TestCase):
    """R1: the capture-time PASS merge had no test at all.

    Reverting `capture_rail.py` to its base left the suite green, which means
    nothing was holding the merge in place - the cohort would simply have
    stopped accruing at capture time and nobody would have known until the
    numbers were read weeks later.
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "trader_annotations.jsonl"
        self.addCleanup(self._tmp.cleanup)
        self.panel = _panel(self.tmp)
        self.panel.open_symbol("NVDA")
        self.rail = self.panel.capture_rail

    def _first_pass_code(self) -> str:
        from ui.annotations.vocabulary import load_pass_vocabulary

        return load_pass_vocabulary().reasons[0].code

    def test_a_pass_click_runs_the_cohort_merge(self) -> None:
        calls: list[dict] = []

        def _merge(**kwargs):
            calls.append(kwargs)
            from ui.annotations.pass_cohort import merge_pass_cohort_picks

            return merge_pass_cohort_picks(
                picks_path=self.tmp / "pass_cohort_picks.csv", **kwargs
            )

        self.rail._merge_pass_cohort = _merge
        self.rail.toggle_pass_reason(self._first_pass_code())
        row = self.rail.commit_pass()

        self.assertIsNotNone(row)
        self.assertEqual(len(calls), 1, "the pass merge must run on the click")
        self.assertTrue((self.tmp / "pass_cohort_picks.csv").exists())

    def test_a_raising_merge_still_returns_the_row(self) -> None:
        """An evidence store never costs the event it records.

        The annotation is already on disk when the merge runs, so a merge that
        blows up degrades to a status suffix - it must not lose the pass.
        """

        def _explode(**_kwargs):
            raise RuntimeError("the cohort file is unwritable")

        self.rail._merge_pass_cohort = _explode
        self.rail.toggle_pass_reason(self._first_pass_code())
        row = self.rail.commit_pass()

        self.assertIsNotNone(row, "the pass row must survive a failing merge")
        self.assertEqual(row["event_type"], "pass")
        self.assertIn("deferred", self.rail.status_label.text())
        self.assertTrue(
            any(item.get("event_type") == "pass" for item in
                [json.loads(line) for line in
                 self.log.read_text(encoding="utf-8").splitlines() if line.strip()]),
            "and it must still be in the annotation log",
        )


class QuickLikeTests(unittest.TestCase):
    """P9 - one key that says "something about this was good".

    Trader, 2026-09-02: *"anytime I like and claim a setup or like a day trade
    setup I just want to let the bot and the future AI know 'something about
    this was good' and then we can figure out what about it / what's the best
    entry later."*
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.log = self.tmp / "trader_annotations.jsonl"
        self.addCleanup(self._tmp.cleanup)
        self.panel = _panel(self.tmp)
        self.panel.open_symbol("NVDA")
        self.rail = self.panel.capture_rail

    def _rows(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_one_key_writes_a_like_with_no_claim_and_no_why(self) -> None:
        row = self.rail.commit_quick_like()

        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "like_claim")
        self.assertEqual(row["like_mode"], "quick")
        self.assertNotIn("claimed_setup_id", row)
        self.assertEqual(row.get("note", ""), "")
        self.assertEqual([r["event_id"] for r in self._rows()], [row["event_id"]])

    def test_the_status_line_says_the_claim_path_still_exists(self) -> None:
        self.rail.commit_quick_like()
        self.assertIn("Alt+K", self.rail.status_label.text())
        self.assertIn("quick", self.rail.status_label.text().lower())

    def test_the_claimed_path_is_untouched(self) -> None:
        """R9.2(a)'s "why is required" is superseded for the QUICK path only."""
        self.assertIsNone(self.rail.commit_like())  # no digit
        self.rail.setup_list.setCurrentRow(0)
        self.assertIsNone(self.rail.commit_like())  # no why
        self.rail.like_note_input.setText("held the band all day")
        row = self.rail.commit_like()
        self.assertIsNotNone(row)
        self.assertEqual(row["like_mode"], "claimed")
        self.assertTrue(row["claimed_setup_id"])
        self.assertEqual(row["note"], "held the band all day")

    def test_a_quick_like_binds_a_key_nothing_else_uses(self) -> None:
        """Two live bindings for one sequence fire NEITHER, so a clash would
        silently cost the trader both verbs."""
        sequences = [sequence for sequence, _handler in self.rail.action_shortcuts()]
        self.assertIn("Alt+L", sequences)
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_a_quick_like_places_nothing(self) -> None:
        """A like carries zero privileges (plan.md P3.1)."""
        self.rail.commit_quick_like()
        row = self._rows()[0]
        for forbidden in ("focus", "watchlist", "parked", "armed", "alert"):
            self.assertNotIn(forbidden, json.dumps(row).lower())
