"""The fact-derived auto-tag lane, and the tools for adjusting tags after it.

The trader's ask was "auto tagging, then I can come back and adjust". Both
halves are pinned here: that a trade with no scanner context still gets tags,
and that a tag can afterwards be filtered on, renamed, retired, and stops
proposing itself once accepted.
"""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from journal_analytics import AutoTagger, split_tags  # noqa: E402
from journal_importers import manual_execution_from_fields  # noqa: E402
from journal_store import (  # noqa: E402
    TRADE_SHAPE_SOURCE,
    JournalStore,
    _merge_auto_tag_summary,
)
from journal_trade_shape import (  # noqa: E402
    describe_vocabulary,
    execution_shape,
    hold_bucket,
    instrument_tag,
    is_shape_tag,
    session_bucket,
    shape_tags,
)


#: Broker timestamps in the fixtures below carry an EXPLICIT Eastern offset.
#: A naive timestamp is not neutral here: ``parse_broker_datetime`` attaches
#: the desk's own zone (Pacific) to one, so "09:45" would be stored as 09:45 PT
#: and bucketed as 12:45 ET midday. Writing the zone out keeps each fixture's
#: intent readable and pins the conversion rather than the desk's location.
ET_OPENING_DRIVE = "2026-06-01T09:45:00-04:00"
ET_MIDDAY = "2026-06-01T12:30:00-04:00"
ET_AFTERNOON = "2026-06-01T15:00:00-04:00"


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
            "execution_id": execution_id,
        }
    )


class SessionBucketTests(unittest.TestCase):
    def test_buckets_span_the_session_and_both_sides_of_it(self):
        cases = {
            "2026-06-01T08:00:00": "premarket",
            "2026-06-01T09:29:00": "premarket",
            "2026-06-01T09:30:00": "opening_drive",
            "2026-06-01T10:29:00": "opening_drive",
            "2026-06-01T10:30:00": "late_morning",
            "2026-06-01T11:59:00": "late_morning",
            "2026-06-01T12:00:00": "midday",
            "2026-06-01T13:59:00": "midday",
            "2026-06-01T14:00:00": "afternoon",
            "2026-06-01T15:29:00": "afternoon",
            "2026-06-01T15:30:00": "closing_window",
            "2026-06-01T16:00:00": "after_hours",
            "2026-06-01T18:00:00": "after_hours",
        }
        for stamp, expected in cases.items():
            with self.subTest(stamp=stamp):
                self.assertEqual(session_bucket(stamp), expected)

    def test_a_naive_timestamp_is_read_as_market_local_not_as_utc(self):
        """The seam rule: ATTACH market-local to a naive value, never strip.

        Stored broker timestamps are naive. Reading one as UTC would shift a
        09:45 ET entry to 05:45 and call an opening drive a premarket fill.
        """
        self.assertEqual(session_bucket("2026-06-01T09:45:00"), "opening_drive")

    def test_an_aware_timestamp_is_converted_rather_than_truncated(self):
        pacific = datetime(2026, 6, 1, 6, 45, tzinfo=ZoneInfo("America/Los_Angeles"))
        self.assertEqual(session_bucket(pacific), "opening_drive")
        utc = datetime(2026, 6, 1, 13, 45, tzinfo=ZoneInfo("UTC"))
        self.assertEqual(session_bucket(utc), "opening_drive")

    def test_an_unreadable_timestamp_yields_no_bucket_rather_than_a_default(self):
        for value in ("", None, "not a date", "2026-13-45T99:99:99"):
            with self.subTest(value=value):
                self.assertIsNone(session_bucket(value))


class HoldBucketTests(unittest.TestCase):
    def test_same_session_splits_on_the_scalp_threshold(self):
        scalp = hold_bucket("2026-06-01T09:30:00", "2026-06-01T09:33:00")
        self.assertIsNotNone(scalp)
        self.assertEqual(scalp[0], "scalp")
        day = hold_bucket("2026-06-01T09:30:00", "2026-06-01T14:00:00")
        self.assertIsNotNone(day)
        self.assertEqual(day[0], "day_trade")

    def test_sessions_not_calendar_days_decide_the_hold(self):
        """A Friday-to-Monday hold is ONE night, not three.

        Counting calendar days would call every weekend hold a swing and make
        the two buckets mean different things in different weeks.
        """
        friday_to_monday = hold_bucket("2026-06-05T15:00:00", "2026-06-08T10:00:00")
        self.assertIsNotNone(friday_to_monday)
        self.assertEqual(friday_to_monday[0], "overnight")

    def test_longer_holds_climb_to_swing_and_position(self):
        swing = hold_bucket("2026-06-01T09:30:00", "2026-06-05T09:30:00")
        self.assertIsNotNone(swing)
        self.assertEqual(swing[0], "swing")
        position = hold_bucket("2026-06-01T09:30:00", "2026-08-01T09:30:00")
        self.assertIsNotNone(position)
        self.assertEqual(position[0], "position")

    def test_an_open_trade_has_no_hold_yet(self):
        self.assertIsNone(hold_bucket("2026-06-01T09:30:00", ""))

    def test_a_close_before_the_open_is_refused_rather_than_negated(self):
        self.assertIsNone(hold_bucket("2026-06-05T09:30:00", "2026-06-01T09:30:00"))


class ExecutionShapeTests(unittest.TestCase):
    def test_roles_decide_the_shape(self):
        self.assertEqual(
            execution_shape([{"role": "OPEN"}, {"role": "CLOSE"}])[0], "one_and_done"
        )
        self.assertEqual(
            execution_shape([{"role": "OPEN"}, {"role": "SCALE"}, {"role": "CLOSE"}])[0],
            "scaled_in",
        )
        self.assertEqual(
            execution_shape([{"role": "OPEN"}, {"role": "CLOSE"}, {"role": "CLOSE"}])[0],
            "scaled_out",
        )
        self.assertEqual(
            execution_shape(
                [{"role": "OPEN"}, {"role": "SCALE"}, {"role": "CLOSE"}, {"role": "CLOSE"}]
            )[0],
            "scaled_both",
        )

    def test_a_reconstructed_entry_has_no_known_shape(self):
        """A SYNTHETIC_OPEN means the opening fill was never imported.

        Calling that a clean single entry would put a tag on a position the
        journal had to invent half of.
        """
        self.assertIsNone(execution_shape([{"role": "SYNTHETIC_OPEN"}, {"role": "CLOSE"}]))

    def test_no_legs_yields_no_shape(self):
        self.assertIsNone(execution_shape(None))
        self.assertIsNone(execution_shape([]))


class InstrumentTagTests(unittest.TestCase):
    def test_a_plain_share_says_nothing_worth_tagging(self):
        for value in ("", "STK", "STOCK", "EQUITY", "COMMON", None):
            with self.subTest(value=value):
                self.assertIsNone(instrument_tag(value))

    def test_anything_else_is_kept(self):
        self.assertEqual(instrument_tag("OPT"), "opt")
        self.assertEqual(instrument_tag("Future"), "future")


class ShapeTagPolicyTests(unittest.TestCase):
    def test_no_tag_is_ever_derived_from_the_outcome(self):
        """The load-bearing rule: a tag may not encode the result.

        If it did, every per-tag statistic would be circular - a ``winners``
        bucket posting a 100% win rate that explains nothing.
        """
        base = {
            "opened_at": "2026-06-01T09:45:00",
            "closed_at": "2026-06-01T15:00:00",
            "security_type": "STK",
        }
        legs = [{"role": "OPEN"}, {"role": "CLOSE"}]
        winner = shape_tags({**base, "net_pnl": 900.0, "gross_pnl": 950.0}, legs=legs)
        loser = shape_tags({**base, "net_pnl": -900.0, "gross_pnl": -880.0}, legs=legs)
        self.assertEqual(winner, loser)
        self.assertTrue(winner)

    def test_tags_are_ordered_by_kind_so_a_rebuild_does_not_reshuffle(self):
        tags = shape_tags(
            {
                "opened_at": "2026-06-01T09:45:00",
                "closed_at": "2026-06-03T15:00:00",
                "security_type": "OPT",
            },
            legs=[{"role": "OPEN"}, {"role": "CLOSE"}],
        )
        self.assertEqual([tag.kind for tag in tags], ["hold", "entry_time", "execution", "instrument"])

    def test_every_emitted_name_is_in_the_declared_vocabulary(self):
        """``is_shape_tag`` gates the rename tool, so it must not miss a name."""
        for names in describe_vocabulary().values():
            for name in names:
                with self.subTest(name=name):
                    self.assertTrue(is_shape_tag(name))
        self.assertFalse(is_shape_tag("my gap and go"))
        self.assertFalse(is_shape_tag(""))

    def test_the_vocabulary_lists_both_same_session_names(self):
        self.assertIn("scalp", describe_vocabulary()["hold"])
        self.assertIn("day_trade", describe_vocabulary()["hold"])


class SummaryMergeTests(unittest.TestCase):
    def test_setup_tags_lead_and_shape_tags_fill_the_rest(self):
        merged = _merge_auto_tag_summary(["setup_a", "setup_b", "setup_c"], ["swing", "midday"])
        self.assertEqual(merged, ["setup_a", "setup_b", "swing", "midday"])

    def test_a_trade_with_no_setup_match_still_says_what_kind_it_was(self):
        merged = _merge_auto_tag_summary([], ["swing", "midday", "one_and_done"])
        self.assertEqual(merged, ["swing", "midday", "one_and_done"])

    def test_setup_tags_spread_into_the_gap_when_there_are_no_shape_tags(self):
        merged = _merge_auto_tag_summary(["a", "b", "c", "d", "e"], [])
        self.assertEqual(merged, ["a", "b", "c", "d"])

    def test_duplicates_collapse(self):
        self.assertEqual(_merge_auto_tag_summary(["swing"], ["swing", "midday"]), ["swing", "midday"])


class StoreAutoTagTests(unittest.TestCase):
    def _store(self, temp_dir):
        return JournalStore(Path(temp_dir) / "journal.sqlite3")

    def _empty_tagger(self, temp_dir):
        """An AutoTagger pointed at files that do not exist.

        This is the imported-history case: the scan outputs do not cover the
        trade, so the setup lane scores nothing at all.
        """
        root = Path(temp_dir)
        return AutoTagger(
            setup_tracker_path=root / "missing_tracker.json",
            focus_path=root / "missing_focus.json",
            avwap_signals_path=root / "missing_signals.csv",
            intraday_bounces_path=root / "missing_bounces.csv",
        )

    def _one_trade(self, store, *, opened=ET_OPENING_DRIVE, closed=ET_AFTERNOON):
        store.upsert_executions(
            [
                _execution("1", side="BUY", qty=100, price=10, timestamp=opened),
                _execution("2", side="SELL", qty=100, price=11, timestamp=closed),
            ]
        )
        store.rebuild_trades(refresh_tags=False)
        return store.list_trades()[0]

    def test_a_trade_with_no_scanner_context_is_still_tagged(self):
        """The whole point. Imported history used to arrive blank."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            self._one_trade(store)
            store.refresh_auto_tags(self._empty_tagger(temp_dir))

            trade = store.list_trades()[0]
            summary = split_tags(trade["auto_tag_summary"])
            self.assertIn("day_trade", summary)
            self.assertIn("opening_drive", summary)
            self.assertIn("one_and_done", summary)

    def test_shape_candidates_are_stored_with_a_traceable_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trade = self._one_trade(store)
            store.refresh_auto_tags(self._empty_tagger(temp_dir))

            candidates = store.list_auto_tag_candidates(trade["trade_id"])
            by_tag = {row["tag"]: row for row in candidates}
            self.assertEqual(by_tag["day_trade"]["source"], f"{TRADE_SHAPE_SOURCE}:hold")
            self.assertEqual(by_tag["opening_drive"]["source"], f"{TRADE_SHAPE_SOURCE}:entry_time")
            self.assertTrue(by_tag["day_trade"]["rationale"])

    def test_a_setup_match_outranks_a_certain_fact_in_the_candidate_list(self):
        """Ordering is by lane, not confidence.

        Shape tags are facts and carry 1.0, so a plain confidence sort would
        bury every scanner match under ``midday`` - and the scanner match is
        what the trader opened the pane to see.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trade = self._one_trade(store)
            tracker = Path(temp_dir) / "tracker.json"
            tracker.write_text(
                '{"setups": {"AAPL": {"symbol": "AAPL", "side": "long", '
                '"scan_date": "2026-06-01", "setup_family": "avwap_breakout", '
                '"priority_bucket": "favorite_setup", "priority_score": 900}}}',
                encoding="utf-8",
            )
            tagger = AutoTagger(
                setup_tracker_path=tracker,
                focus_path=Path(temp_dir) / "missing_focus.json",
                avwap_signals_path=Path(temp_dir) / "missing_signals.csv",
                intraday_bounces_path=Path(temp_dir) / "missing_bounces.csv",
            )
            store.refresh_auto_tags(tagger)

            candidates = store.list_auto_tag_candidates(trade["trade_id"])
            self.assertIn("avwap_breakout", candidates[0]["tag"])
            shape_rows = [
                row
                for row in candidates
                if str(row["source"]).startswith(f"{TRADE_SHAPE_SOURCE}:")
            ]
            self.assertTrue(shape_rows)
            first_shape = candidates.index(shape_rows[0])
            self.assertGreater(first_shape, 0)

    def test_an_open_trade_gets_an_entry_tag_but_no_hold_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            store.upsert_executions(
                [_execution("1", side="BUY", qty=100, price=10, timestamp=ET_MIDDAY)]
            )
            store.rebuild_trades(refresh_tags=False)
            store.refresh_auto_tags(self._empty_tagger(temp_dir))

            summary = split_tags(store.list_trades()[0]["auto_tag_summary"])
            self.assertIn("midday", summary)
            for name in describe_vocabulary()["hold"]:
                self.assertNotIn(name, summary)


class TagAdjustmentTests(unittest.TestCase):
    def _store(self, temp_dir):
        return JournalStore(Path(temp_dir) / "journal.sqlite3")

    def _two_trades(self, store):
        store.upsert_executions(
            [
                _execution("1", side="BUY", qty=10, price=10, timestamp="2026-06-01T09:45:00", symbol="AAA"),
                _execution("2", side="SELL", qty=10, price=11, timestamp="2026-06-01T14:45:00", symbol="AAA"),
                _execution("3", side="BUY", qty=10, price=10, timestamp="2026-06-02T09:45:00", symbol="BBB"),
                _execution("4", side="SELL", qty=10, price=9, timestamp="2026-06-02T14:45:00", symbol="BBB"),
            ]
        )
        store.rebuild_trades(refresh_tags=False)
        return {trade["symbol"]: trade for trade in store.list_trades()}

    def test_distinct_tags_separates_what_the_trader_typed_from_the_auto_lane(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            store.save_trade_annotation(trades["AAA"]["trade_id"], setup_tags="gap and go", notes="")

            entries = {row["tag"]: row for row in store.distinct_tags()}
            self.assertEqual(entries["gap and go"]["own"], 1)
            self.assertEqual(entries["gap and go"]["auto"], 0)
            self.assertFalse(entries["gap and go"]["derived"])

    def test_an_auto_tag_is_counted_only_where_the_trader_has_not_typed_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            store.refresh_auto_tags(
                AutoTagger(
                    setup_tracker_path=Path(temp_dir) / "a.json",
                    focus_path=Path(temp_dir) / "b.json",
                    avwap_signals_path=Path(temp_dir) / "c.csv",
                    intraday_bounces_path=Path(temp_dir) / "d.csv",
                )
            )
            store.save_trade_annotation(trades["AAA"]["trade_id"], setup_tags="mine", notes="")

            entries = {row["tag"]: row for row in store.distinct_tags()}
            self.assertEqual(entries["day_trade"]["auto"], 1)
            self.assertEqual(entries["day_trade"]["own"], 0)
            self.assertTrue(entries["day_trade"]["derived"])

    def test_rename_rewrites_the_tag_everywhere_and_reports_the_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            for trade in trades.values():
                store.save_trade_annotation(trade["trade_id"], setup_tags="gapngo; runner", notes="")

            changed = store.rename_tag("gapngo", "gap and go")

            self.assertEqual(changed, 2)
            for trade in store.list_trades():
                self.assertEqual(split_tags(trade["setup_tags"]), ["gap and go", "runner"])

    def test_rename_keeps_position_and_collapses_onto_an_existing_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            store.save_trade_annotation(
                trades["AAA"]["trade_id"], setup_tags="alpha; gapngo; omega", notes=""
            )
            store.save_trade_annotation(
                trades["BBB"]["trade_id"], setup_tags="gapngo; gap and go", notes=""
            )

            store.rename_tag("gapngo", "gap and go")

            by_symbol = {trade["symbol"]: trade for trade in store.list_trades()}
            self.assertEqual(
                split_tags(by_symbol["AAA"]["setup_tags"]), ["alpha", "gap and go", "omega"]
            )
            self.assertEqual(split_tags(by_symbol["BBB"]["setup_tags"]), ["gap and go"])

    def test_an_empty_new_name_retires_the_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            store.save_trade_annotation(
                trades["AAA"]["trade_id"], setup_tags="typo; keeper", notes=""
            )

            changed = store.rename_tag("typo", "")

            self.assertEqual(changed, 1)
            by_symbol = {trade["symbol"]: trade for trade in store.list_trades()}
            self.assertEqual(split_tags(by_symbol["AAA"]["setup_tags"]), ["keeper"])

    def test_renaming_a_tag_nobody_carries_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            self._two_trades(store)
            self.assertEqual(store.rename_tag("absent", "other"), 0)
            self.assertEqual(store.rename_tag("", "other"), 0)

    def test_list_trades_filters_to_one_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            store.save_trade_annotation(trades["AAA"]["trade_id"], setup_tags="gap and go", notes="")
            store.save_trade_annotation(trades["BBB"]["trade_id"], setup_tags="reversal", notes="")

            filtered = store.list_trades(tag="gap and go")

            self.assertEqual([trade["symbol"] for trade in filtered], ["AAA"])
            self.assertEqual(len(store.list_trades(tag="All")), 2)
            self.assertEqual(len(store.list_trades()), 2)

    def test_the_tag_filter_matches_whole_tags_not_substrings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            trades = self._two_trades(store)
            store.save_trade_annotation(trades["AAA"]["trade_id"], setup_tags="swing", notes="")
            store.save_trade_annotation(trades["BBB"]["trade_id"], setup_tags="swing failed", notes="")

            self.assertEqual([t["symbol"] for t in store.list_trades(tag="swing")], ["AAA"])

    def test_the_tag_filter_reaches_the_auto_lane_for_untagged_trades(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._store(temp_dir)
            self._two_trades(store)
            store.refresh_auto_tags(
                AutoTagger(
                    setup_tracker_path=Path(temp_dir) / "a.json",
                    focus_path=Path(temp_dir) / "b.json",
                    avwap_signals_path=Path(temp_dir) / "c.csv",
                    intraday_bounces_path=Path(temp_dir) / "d.csv",
                )
            )
            self.assertEqual(len(store.list_trades(tag="day_trade")), 2)


if __name__ == "__main__":
    unittest.main()
