import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_analytics import AutoTagger, build_analytics_summary  # noqa: E402
from journal_importers import QuestradeImporter, chunk_date_ranges, manual_execution_from_fields  # noqa: E402
from journal_runner import run_journal_import_for_date  # noqa: E402
from journal_store import JournalStore  # noqa: E402


def _execution(
    execution_id,
    *,
    side,
    qty,
    price,
    timestamp,
    symbol="AAPL",
    security_type="STK",
    broker="MANUAL",
    account="ACCT",
    commission=0,
    fees=0,
):
    return manual_execution_from_fields(
        {
            "broker": broker,
            "account_number": account,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "timestamp": timestamp,
            "security_type": security_type,
            "currency": "USD",
            "commission": commission,
            "fees": fees,
            "execution_id": execution_id,
        }
    )


class JournalSystemTests(unittest.TestCase):
    def _store(self, temp_dir):
        return JournalStore(Path(temp_dir) / "journal.sqlite3")

    def test_grouped_round_trip_handles_scale_in_and_partial_exits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            executions = [
                _execution("1", side="BUY", qty=100, price=10, timestamp="2026-06-01T09:30:00", commission=1),
                _execution("2", side="BUY", qty=100, price=12, timestamp="2026-06-01T10:00:00", commission=1),
                _execution("3", side="SELL", qty=50, price=13, timestamp="2026-06-02T10:00:00", commission=1),
                _execution("4", side="SELL", qty=150, price=14, timestamp="2026-06-03T10:00:00", commission=1),
            ]

            self.assertEqual(store.upsert_executions(executions), 4)
            self.assertEqual(store.upsert_executions(executions), 4)
            self.assertEqual(store.rebuild_trades(refresh_tags=False), 1)
            trades = store.list_trades()

            self.assertEqual(len(trades), 1)
            trade = trades[0]
            self.assertEqual(trade["status"], "CLOSED")
            self.assertEqual(trade["direction"], "LONG")
            self.assertAlmostEqual(trade["quantity_opened"], 200.0)
            self.assertAlmostEqual(trade["quantity_closed"], 200.0)
            self.assertAlmostEqual(trade["average_entry_price"], 11.0)
            self.assertAlmostEqual(trade["gross_pnl"], 550.0)
            self.assertAlmostEqual(trade["net_pnl"], 546.0)
            self.assertEqual(len(store.list_trade_legs(trade["trade_id"])), 4)

    def test_grouped_round_trip_handles_short_trades(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            store.upsert_executions(
                [
                    _execution("1", side="SELL", qty=100, price=20, timestamp="2026-06-01T09:30:00"),
                    _execution("2", side="BUY", qty=100, price=15, timestamp="2026-06-02T09:30:00"),
                ]
            )

            store.rebuild_trades(refresh_tags=False)
            trade = store.list_trades()[0]

            self.assertEqual(trade["direction"], "SHORT")
            self.assertEqual(trade["status"], "CLOSED")
            self.assertAlmostEqual(trade["gross_pnl"], 500.0)
            self.assertAlmostEqual(trade["net_pnl"], 500.0)

    def test_options_use_contract_multiplier_for_pnl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            store.upsert_executions(
                [
                    _execution(
                        "1",
                        side="BUY",
                        qty=1,
                        price=2,
                        timestamp="2026-06-01T09:30:00",
                        symbol="AAPL 260619C00100000",
                        security_type="OPT",
                    ),
                    _execution(
                        "2",
                        side="SELL",
                        qty=1,
                        price=3,
                        timestamp="2026-06-01T11:30:00",
                        symbol="AAPL 260619C00100000",
                        security_type="OPT",
                    ),
                ]
            )

            store.rebuild_trades(refresh_tags=False)
            trade = store.list_trades()[0]

            self.assertEqual(trade["security_type"], "OPT")
            self.assertAlmostEqual(trade["gross_pnl"], 100.0)

    def test_auto_tagger_uses_tracker_context_and_manual_corrections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tracker_path = root / "tracker.json"
            tracker_path.write_text(
                json.dumps(
                    {
                        "setups": {
                            "s1": {
                                "setup_id": "s1",
                                "symbol": "AAPL",
                                "side": "LONG",
                                "scan_date": "2026-06-01",
                                "setup_family": "avwap_breakout",
                                "priority_bucket": "favorite_setup",
                                "favorite_zone": "AVWAPE to UPPER_1",
                                "priority_score": 120,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = self._store(temp_dir)
            store.upsert_executions(
                [
                    _execution("1", side="BUY", qty=100, price=10, timestamp="2026-06-01T09:30:00"),
                    _execution("2", side="SELL", qty=100, price=11, timestamp="2026-06-02T09:30:00"),
                ]
            )
            store.rebuild_trades(refresh_tags=False)
            tagger = AutoTagger(
                setup_tracker_path=tracker_path,
                focus_path=root / "missing_focus.json",
                avwap_signals_path=root / "missing_signals.csv",
                intraday_bounces_path=root / "missing_bounces.csv",
            )

            store.refresh_auto_tags(tagger)
            trade = store.list_trades()[0]
            candidates = store.list_auto_tag_candidates(trade["trade_id"])

            self.assertTrue(candidates)
            self.assertIn("avwap_breakout", candidates[0]["tag"])
            self.assertIn("favorite_setup", trade["auto_tag_summary"])

            store.save_trade_annotation(trade["trade_id"], setup_tags="my corrected swing", notes="")
            store.record_tag_corrections(trade, "my corrected swing")
            store.refresh_auto_tags(tagger)
            boosted = store.list_auto_tag_candidates(trade["trade_id"])
            self.assertTrue(any(row["tag"] == "my corrected swing" for row in boosted))

    def test_analytics_wr_pf_and_calendar_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            store.upsert_executions(
                [
                    _execution("1", side="BUY", qty=100, price=10, timestamp="2026-06-01T09:30:00", symbol="WIN"),
                    _execution("2", side="SELL", qty=100, price=12, timestamp="2026-06-01T10:30:00", symbol="WIN"),
                    _execution("3", side="BUY", qty=100, price=10, timestamp="2026-06-01T09:30:00", symbol="LOSS"),
                    _execution("4", side="SELL", qty=100, price=9, timestamp="2026-06-01T10:30:00", symbol="LOSS"),
                ]
            )
            store.rebuild_trades(refresh_tags=False)
            summary = build_analytics_summary(store.list_trades())

            self.assertEqual(summary["overall"]["closed"], 2)
            self.assertEqual(summary["overall"]["wins"], 1)
            self.assertEqual(summary["overall"]["losses"], 1)
            self.assertAlmostEqual(summary["overall"]["win_rate"], 0.5)
            self.assertAlmostEqual(summary["overall"]["profit_factor"], 2.0)

    def test_regime_mid_and_short_carry_forward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            store.upsert_regime(
                "2026-06-01",
                mid_term_regime="Bull trend",
                short_term_regime="Risk on",
                intraday_regime="Trend up",
            )

            regime = store.get_regime_for_date("2026-06-05")

            self.assertEqual(regime["mid_term_regime"], "Bull trend")
            self.assertEqual(regime["short_term_regime"], "Risk on")
            self.assertEqual(regime["intraday_regime"], "")

    def test_questrade_normalization_and_activity_chunks(self):
        chunks = chunk_date_ranges(
            __import__("datetime").date(2026, 1, 1),
            __import__("datetime").date(2026, 3, 5),
            max_days=31,
        )
        self.assertEqual(chunks[0][0].isoformat(), "2026-01-01")
        self.assertLessEqual((chunks[0][1] - chunks[0][0]).days + 1, 31)
        self.assertEqual(chunks[-1][1].isoformat(), "2026-03-05")

        normalized = QuestradeImporter().normalize_execution(
            {
                "id": "ex1",
                "symbol": "AAPL",
                "side": "Buy",
                "quantity": 10,
                "price": 100.5,
                "timestamp": "2026-06-01T13:00:00-04:00",
                "commission": 4.95,
                "secFee": 0.01,
                "orderId": 123,
            },
            {"number": "123456", "type": "Margin", "currency": "USD"},
        )

        self.assertEqual(normalized.broker, "QUESTRADE")
        self.assertEqual(normalized.account_number, "123456")
        self.assertEqual(normalized.symbol, "AAPL")
        self.assertEqual(normalized.side, "BUY")
        self.assertAlmostEqual(normalized.fees, 0.01)
        self.assertIn("QT:123456:ex1", normalized.execution_uid)

    def test_journal_runner_skips_disabled_brokers_and_rebuilds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)

            summary = run_journal_import_for_date(
                date(2026, 6, 1),
                trigger="test",
                store=store,
                include_questrade=False,
                include_ibkr=False,
            )

            self.assertEqual(summary["status"], "OK")
            self.assertEqual(summary["total_imported"], 0)
            self.assertEqual(summary["trade_count"], 0)
            self.assertIn("Questrade skipped: disabled.", summary["messages"])
            self.assertIn("IBKR skipped: disabled.", summary["messages"])

    def test_gui_wires_journal_only_into_tabbed_layout(self):
        gui_text = (SCRIPTS_DIR / "gui.py").read_text(encoding="utf-8")
        self.assertIn("from journal_tab import JournalTab", gui_text)
        self.assertIn('notebook.add(journal_tab, text="Journal")', gui_text)
        combined_start = gui_text.index("    def _build_combined_layout")
        combined_end = gui_text.index("    def _set_combined_initial_panes", combined_start)
        combined_block = gui_text[combined_start:combined_end]
        self.assertNotIn("JournalTab", combined_block)
        self.assertNotIn('text="Journal"', combined_block)


if __name__ == "__main__":
    unittest.main()
