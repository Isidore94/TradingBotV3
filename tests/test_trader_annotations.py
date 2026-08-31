"""The trader annotation store and its versioned vocabulary (schema v1).

This stream is the packet's whole point, so the tests here are about the
properties that make it trustworthy months from now rather than about the
happy path:

* the row shape is the documented v1 shape, with an explicitly zoned
  ``created_at`` (plan.md sec 5);
* a row that would be uninterpretable later - unknown event type, a reason
  outside the vocabulary, ``other`` with no note - is refused, not written;
* the file is append-only and rows survive concurrent writers intact, because
  a torn or truncated decision log cannot be repaired after the fact;
* a shipped vocabulary version validates its own contract, so an edit that
  would silently change what an already-written ``vocab_version: 1`` meant
  fails the suite instead of the analysis.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui.annotations import store  # noqa: E402
from ui.annotations import vocabulary  # noqa: E402
from ui.annotations.setup_claims import (  # noqa: E402
    all_setup_claims,
    is_valid_setup_claim,
)
from ui.annotations import pass_bars  # noqa: E402
from ui.annotations.store import (  # noqa: E402
    EVENT_HYPO_STOP,
    EVENT_LIKE_CLAIM,
    EVENT_NOTE,
    EVENT_PASS,
    EVENT_VETO,
    AnnotationError,
    append_annotation_row,
    build_annotation,
    load_annotations,
    record_annotation,
    record_pass_annotation,
)
from ui.annotations.vocabulary import (  # noqa: E402
    VocabularyError,
    available_pass_versions,
    available_veto_versions,
    clear_vocabulary_cache,
    load_pass_vocabulary,
    load_veto_vocabulary,
)


def _write_vocab(directory: Path, version: int, reasons: list[dict]) -> Path:
    path = directory / f"veto_reasons_v{version}.json"
    path.write_text(
        json.dumps({"vocabulary_id": "veto_reasons", "vocab_version": version, "reasons": reasons}),
        encoding="utf-8",
    )
    clear_vocabulary_cache()
    return path


def _reason(code: str, hotkey: str, *, note_required: bool = False) -> dict:
    return {
        "code": code,
        "label": code.replace("_", " ").title(),
        "hotkey": hotkey,
        "note_required": note_required,
    }


class ShippedVocabularyTests(unittest.TestCase):
    """The v1 file that ships is the contract every v1 row was written under."""

    def setUp(self) -> None:
        clear_vocabulary_cache()

    def test_v1_is_present_and_valid(self) -> None:
        self.assertIn(1, available_veto_versions())
        vocab = load_veto_vocabulary(1)
        self.assertEqual(vocab.vocab_version, 1)

    def test_v1_carries_the_agreed_starting_vocabulary(self) -> None:
        # Pinned deliberately. Adding a reason means shipping v2, not editing
        # v1: rows already stamped vocab_version 1 must keep meaning what they
        # meant when they were written.
        self.assertEqual(
            load_veto_vocabulary(1).codes,
            (
                "incoming_trendline",
                "overhead_horizontal",
                "support_resistance_cluttered",
                "sector_mate_earnings_pending",
                "too_extended_from_base",
                "volume_dry",
                "earnings_too_close",
                "spread_liquidity",
                "other",
            ),
        )

    def test_other_is_the_only_reason_requiring_a_note(self) -> None:
        vocab = load_veto_vocabulary(1)
        requiring = [reason.code for reason in vocab.reasons if reason.note_required]
        self.assertEqual(requiring, ["other"])

    def test_every_reason_has_a_unique_hotkey(self) -> None:
        vocab = load_veto_vocabulary(1)
        hotkeys = [reason.hotkey for reason in vocab.reasons]
        self.assertEqual(len(hotkeys), len(set(hotkeys)))
        self.assertIsNotNone(vocab.by_hotkey(hotkeys[0]))


class VocabularyVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_vocabulary_cache()
        self._tmp = TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(clear_vocabulary_cache)

    def test_default_load_picks_the_newest_version(self) -> None:
        _write_vocab(self.directory, 1, [_reason("alpha", "1")])
        _write_vocab(self.directory, 2, [_reason("alpha", "1"), _reason("beta", "2")])
        self.assertEqual(available_veto_versions(self.directory), (1, 2))
        self.assertEqual(load_veto_vocabulary(directory=self.directory).vocab_version, 2)

    def test_an_older_version_stays_loadable(self) -> None:
        """What makes an old row interpretable after the vocabulary moves on."""
        _write_vocab(self.directory, 1, [_reason("alpha", "1")])
        _write_vocab(self.directory, 2, [_reason("alpha", "1"), _reason("beta", "2")])
        self.assertEqual(load_veto_vocabulary(1, directory=self.directory).codes, ("alpha",))

    def test_missing_vocabulary_fails_closed(self) -> None:
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(directory=self.directory)

    def test_unknown_version_fails_closed(self) -> None:
        _write_vocab(self.directory, 1, [_reason("alpha", "1")])
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(7, directory=self.directory)

    def test_declared_version_must_match_the_filename(self) -> None:
        path = self.directory / "veto_reasons_v2.json"
        path.write_text(
            json.dumps({"vocab_version": 1, "reasons": [_reason("alpha", "1")]}),
            encoding="utf-8",
        )
        clear_vocabulary_cache()
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(2, directory=self.directory)

    def test_duplicate_code_is_rejected(self) -> None:
        _write_vocab(self.directory, 1, [_reason("alpha", "1"), _reason("alpha", "2")])
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(directory=self.directory)

    def test_duplicate_hotkey_is_rejected(self) -> None:
        _write_vocab(self.directory, 1, [_reason("alpha", "1"), _reason("beta", "1")])
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(directory=self.directory)

    def test_malformed_code_is_rejected(self) -> None:
        for bad in ("Alpha", "a", "has space", "trailing-hyphen", "9leading"):
            _write_vocab(self.directory, 1, [_reason(bad, "1")])
            with self.assertRaises(VocabularyError, msg=bad):
                load_veto_vocabulary(directory=self.directory)

    def test_reserved_cohort_prefix_is_rejected(self) -> None:
        """``veto_<code>`` is a cohort source; a ``veto_``/``focus_`` code
        would land veto rows in - or next to - a focus cohort."""
        for bad in ("veto_thing", "focus_thing"):
            _write_vocab(self.directory, 1, [_reason(bad, "1")])
            with self.assertRaises(VocabularyError, msg=bad):
                load_veto_vocabulary(directory=self.directory)

    def test_empty_reason_list_is_rejected(self) -> None:
        _write_vocab(self.directory, 1, [])
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(directory=self.directory)

    def test_corrupt_json_is_rejected(self) -> None:
        (self.directory / "veto_reasons_v1.json").write_text("{not json", encoding="utf-8")
        clear_vocabulary_cache()
        with self.assertRaises(VocabularyError):
            load_veto_vocabulary(directory=self.directory)


class AnnotationSchemaTests(unittest.TestCase):
    def test_veto_row_carries_the_v1_shape(self) -> None:
        row = build_annotation(
            EVENT_VETO,
            symbol="nvda",
            session_date="2026-08-07",
            reason_code="volume_dry",
            side="long",
            last_price=100.5,
            ref_level_id="sma_200",
            ref_level_family="sma",
            note="thin all morning",
            timeframe="d1",
        )
        self.assertEqual(row["schema_version"], 1)
        self.assertEqual(row["event_type"], EVENT_VETO)
        self.assertEqual(row["symbol"], "NVDA")
        self.assertEqual(row["source"], "chart_review")
        self.assertEqual(row["session_date"], "2026-08-07")
        self.assertEqual(row["reason_code"], "volume_dry")
        # The row stamps the vocabulary it was BUILT with, not a fixed number.
        # Asserting the literal made this test fail the day v2 shipped, while
        # the property it exists to protect - that a row can always be read
        # back against the list it was written from - was never at risk.
        self.assertEqual(row["vocab_version"], load_veto_vocabulary().vocab_version)
        self.assertEqual(row["side"], "LONG")
        self.assertEqual(row["last_price"], 100.5)
        self.assertEqual(row["ref_level_id"], "sma_200")
        self.assertEqual(row["ref_level_family"], "sma")
        self.assertEqual(row["note"], "thin all morning")
        self.assertEqual(row["timeframe"], "D1")
        self.assertEqual(len(row["event_id"]), 32)

    def test_created_at_is_timezone_aware(self) -> None:
        """plan.md sec 5: timestamps carry explicit timezones."""
        row = build_annotation(EVENT_NOTE, symbol="AAPL", note="x")
        parsed = datetime.fromisoformat(row["created_at"])
        self.assertIsNotNone(parsed.tzinfo)
        self.assertIsNotNone(parsed.utcoffset())

    def test_a_naive_timestamp_is_given_an_offset(self) -> None:
        row = build_annotation(EVENT_NOTE, symbol="AAPL", note="x", created_at=datetime(2026, 8, 7, 9, 30))
        self.assertIsNotNone(datetime.fromisoformat(row["created_at"]).tzinfo)

    def test_an_aware_timestamp_is_preserved(self) -> None:
        moment = datetime(2026, 8, 7, 9, 30, tzinfo=timezone(timedelta(hours=-4)))
        row = build_annotation(EVENT_NOTE, symbol="AAPL", note="x", created_at=moment)
        self.assertEqual(datetime.fromisoformat(row["created_at"]), moment)

    def test_event_ids_are_unique(self) -> None:
        ids = {build_annotation(EVENT_NOTE, symbol="A", note="n")["event_id"] for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_unknown_event_type_is_refused(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation("mute", symbol="NVDA")

    def test_symbol_is_required(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_NOTE, symbol="   ", note="x")

    def test_reason_outside_the_vocabulary_is_refused(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_VETO, symbol="NVDA", reason_code="i_dont_like_it")

    def test_other_requires_a_note(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_VETO, symbol="NVDA", reason_code="other")
        row = build_annotation(EVENT_VETO, symbol="NVDA", reason_code="other", note="held up by news")
        self.assertEqual(row["reason_code"], "other")

    def test_a_normal_reason_does_not_require_a_note(self) -> None:
        row = build_annotation(EVENT_VETO, symbol="NVDA", reason_code="volume_dry")
        self.assertNotIn("note", row)

    def test_like_claim_requires_a_setup(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_LIKE_CLAIM, symbol="NVDA")
        row = build_annotation(EVENT_LIKE_CLAIM, symbol="NVDA", claimed_setup_id="avwape_to_1stdev")
        self.assertEqual(row["claimed_setup_id"], "avwape_to_1stdev")

    def test_hypo_stop_requires_price_and_side(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_HYPO_STOP, symbol="NVDA", side="LONG")
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_HYPO_STOP, symbol="NVDA", stop_price=10.0)
        row = build_annotation(EVENT_HYPO_STOP, symbol="NVDA", stop_price=10.5, side="short", last_price=11.0)
        self.assertEqual(row["stop_price"], 10.5)
        self.assertEqual(row["side"], "SHORT")

    def test_non_positive_or_unparseable_prices_are_refused(self) -> None:
        for bad in (0, -1, "abc", float("nan")):
            with self.assertRaises(AnnotationError, msg=repr(bad)):
                build_annotation(EVENT_HYPO_STOP, symbol="NVDA", stop_price=bad, side="LONG")

    def test_note_event_requires_a_note(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_NOTE, symbol="NVDA")

    def test_oversized_note_is_refused(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_NOTE, symbol="NVDA", note="x" * (store.MAX_NOTE_CHARS + 1))

    def test_a_capped_note_still_fits_the_atomic_row(self) -> None:
        row = build_annotation(EVENT_NOTE, symbol="NVDA", note="x" * store.MAX_NOTE_CHARS)
        encoded = len(json.dumps(row, sort_keys=True, default=str).encode("utf-8")) + 1
        self.assertLessEqual(encoded, store.MAX_ROW_BYTES)


class AnnotationStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "trader_annotations.jsonl"
        self.addCleanup(self._tmp.cleanup)

    def test_records_are_appended_one_per_line(self) -> None:
        for index in range(3):
            self.assertIsNotNone(
                record_annotation(EVENT_NOTE, symbol=f"SYM{index}", note="n", path=self.path)
            )
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual([json.loads(line)["symbol"] for line in lines], ["SYM0", "SYM1", "SYM2"])

    def test_the_store_never_rewrites_existing_rows(self) -> None:
        """Append-only: an earlier decision is never edited or dropped."""
        record_annotation(EVENT_NOTE, symbol="FIRST", note="original", path=self.path)
        original = self.path.read_bytes()
        for index in range(5):
            record_annotation(EVENT_NOTE, symbol=f"LATER{index}", note="n", path=self.path)
        self.assertTrue(self.path.read_bytes().startswith(original))

    def test_a_pre_existing_file_is_not_truncated(self) -> None:
        self.path.write_text('{"schema_version": 1, "symbol": "OLD"}\n', encoding="utf-8")
        record_annotation(EVENT_NOTE, symbol="NEW", note="n", path=self.path)
        rows = load_annotations(self.path)
        self.assertEqual([row.get("symbol") for row in rows], ["OLD", "NEW"])

    def test_concurrent_appends_produce_intact_rows(self) -> None:
        """Atomicity: every row is complete and parseable, none interleaved."""
        writers = 8
        per_writer = 25
        barrier = threading.Barrier(writers)

        def _write(worker: int) -> None:
            barrier.wait()
            for index in range(per_writer):
                append_annotation_row(
                    build_annotation(EVENT_NOTE, symbol=f"W{worker}", note=f"note-{index}"),
                    path=self.path,
                )

        threads = [threading.Thread(target=_write, args=(worker,)) for worker in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), writers * per_writer)
        for line in lines:
            parsed = json.loads(line)  # raises if a row was torn
            self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(len({json.loads(line)["event_id"] for line in lines}), writers * per_writer)

    def test_an_oversized_row_is_refused_rather_than_torn(self) -> None:
        row = build_annotation(EVENT_NOTE, symbol="NVDA", note="ok")
        row["note"] = "x" * (store.MAX_ROW_BYTES + 10)
        with self.assertRaises(AnnotationError):
            append_annotation_row(row, path=self.path)
        self.assertFalse(self.path.exists())

    def test_a_failed_append_reports_false(self) -> None:
        """The rail shows a failure; a capture never vanishes silently."""
        row = build_annotation(EVENT_NOTE, symbol="NVDA", note="ok")
        directory = Path(self._tmp.name) / "blocked"
        directory.write_text("not a directory", encoding="utf-8")
        self.assertFalse(append_annotation_row(row, path=directory / "x.jsonl"))

    def test_a_torn_tail_costs_only_its_own_row_never_the_next_one(self) -> None:
        """Crash confinement: this is the reviewer's exact failure scenario.

        A writer that dies mid-row leaves an unterminated prefix. Without the
        tail heal, the NEXT good append joins that prefix on the same line
        and the reader drops both decisions - the torn one and an innocent
        one. With it, the torn fragment costs exactly itself.
        """
        record_annotation(EVENT_NOTE, symbol="GOOD1", note="n", path=self.path)
        with self.path.open("ab") as handle:
            handle.write(b'{"schema_version": 1, "symbol": "TORN", "no')  # no newline
        record_annotation(EVENT_NOTE, symbol="GOOD2", note="n", path=self.path)
        self.assertEqual(
            [row["symbol"] for row in load_annotations(self.path)], ["GOOD1", "GOOD2"]
        )

    def test_a_successful_append_is_fsynced_not_just_buffered(self) -> None:
        """For a non-reconstructable stream, "saved" must survive a power cut."""
        synced: list[int] = []
        original = store.os.fsync

        def _spy(fd: int) -> None:
            synced.append(fd)
            original(fd)

        store.os.fsync = _spy
        try:
            self.assertIsNotNone(
                record_annotation(EVENT_NOTE, symbol="NVDA", note="n", path=self.path)
            )
        finally:
            store.os.fsync = original
        self.assertEqual(len(synced), 1)

    def test_corrupt_lines_are_skipped_not_fatal(self) -> None:
        record_annotation(EVENT_NOTE, symbol="GOOD1", note="n", path=self.path)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{torn\n")
            handle.write("[]\n")
        record_annotation(EVENT_NOTE, symbol="GOOD2", note="n", path=self.path)
        self.assertEqual(
            [row["symbol"] for row in load_annotations(self.path)], ["GOOD1", "GOOD2"]
        )

    def test_load_filters_by_symbol_date_and_type(self) -> None:
        record_annotation(EVENT_NOTE, symbol="AAA", session_date="2026-08-06", note="n", path=self.path)
        record_annotation(EVENT_VETO, symbol="AAA", session_date="2026-08-07", reason_code="volume_dry", path=self.path)
        record_annotation(EVENT_VETO, symbol="BBB", session_date="2026-08-07", reason_code="volume_dry", path=self.path)
        self.assertEqual(len(load_annotations(self.path, symbol="aaa")), 2)
        self.assertEqual(len(load_annotations(self.path, session_date="2026-08-07")), 2)
        self.assertEqual(len(load_annotations(self.path, event_types=(EVENT_VETO,))), 2)
        self.assertEqual(
            len(load_annotations(self.path, symbol="AAA", event_types=(EVENT_VETO,))), 1
        )

    def test_missing_file_reads_as_empty(self) -> None:
        self.assertEqual(load_annotations(Path(self._tmp.name) / "absent.jsonl"), [])


class SetupClaimTests(unittest.TestCase):
    def test_claims_come_from_the_setup_registry(self) -> None:
        from setup_docs import SETUP_DOCS

        claim_ids = {claim.setup_id for claim in all_setup_claims()}
        self.assertTrue(set(SETUP_DOCS).issubset(claim_ids))

    def test_study_setups_are_claimable(self) -> None:
        """A claim on a measured-only setup is the evidence that promotes it."""
        groups = {claim.group for claim in all_setup_claims()}
        self.assertIn("Study (measured only)", groups)

    def test_none_of_these_is_offered(self) -> None:
        self.assertTrue(is_valid_setup_claim("none_of_these"))

    def test_unknown_claim_is_rejected(self) -> None:
        self.assertFalse(is_valid_setup_claim("made_up_setup"))

    def test_every_claim_has_a_label(self) -> None:
        for claim in all_setup_claims():
            self.assertTrue(claim.label.strip(), claim.setup_id)


# --------------------------------------------------------------------------
# The day-trade pass (trader, 2026-08-31)
#
# "Many times I really like this stock for a daytrade but it has this ONE
# issue" - and they pass. That judgement was going nowhere; these pin the shape
# it now gets recorded in. The properties that matter are the ones that make it
# readable in a year: its own vocabulary family, several reasons per pass in a
# stable order, and a chart attached only when the desk already had one.
# --------------------------------------------------------------------------
def _write_pass_vocab(directory: Path, version: int, reasons: list[dict]) -> Path:
    path = directory / f"pass_reasons_v{version}.json"
    path.write_text(
        json.dumps(
            {
                "vocabulary_id": "pass_reasons",
                "vocab_version": version,
                "reasons": reasons,
            }
        ),
        encoding="utf-8",
    )
    clear_vocabulary_cache()
    return path


class ShippedPassVocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_vocabulary_cache()

    def test_the_pass_family_ships_and_loads(self) -> None:
        self.assertTrue(available_pass_versions())
        vocab = load_pass_vocabulary()
        self.assertEqual(vocab.vocabulary_id, "pass_reasons")
        self.assertTrue(vocab.reasons)

    def test_it_carries_the_five_reasons_the_trader_listed(self) -> None:
        # The trader wrote these five, in this order, with these words. The
        # LABELS are pinned rather than paraphrased: the picklist is the
        # question being asked, and rewording it silently changes the answers.
        self.assertEqual(
            [reason.label for reason in load_pass_vocabulary(1).reasons],
            [
                "Poor market conditions",
                "Low rvol",
                "LRSI/SMI incongruency",
                "Incoming Horizontal",
                "Other incoming S/R",
            ],
        )

    def test_no_pass_reason_demands_a_note(self) -> None:
        """Ticking a box is the whole capture; the note stays optional."""
        vocab = load_pass_vocabulary()
        self.assertEqual([r.code for r in vocab.reasons if r.note_required], [])

    def test_the_pass_family_is_separate_from_the_veto_family(self) -> None:
        """Separate files, separate version series, no shared codes.

        A pass is not a veto, and pooling them would restamp cohort identity
        for every veto reason already accruing forward returns.
        """
        pass_vocab = load_pass_vocabulary()
        veto_vocab = load_veto_vocabulary()
        self.assertNotEqual(pass_vocab.vocabulary_id, veto_vocab.vocabulary_id)
        self.assertEqual(set(pass_vocab.codes) & set(veto_vocab.codes), set())


class PassVocabularyVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_vocabulary_cache()
        self._tmp = TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(clear_vocabulary_cache)

    def test_codes_round_trip_from_the_file(self) -> None:
        _write_pass_vocab(self.directory, 1, [_reason("alpha", "1"), _reason("beta", "2")])
        vocab = load_pass_vocabulary(directory=self.directory)
        self.assertEqual(vocab.codes, ("alpha", "beta"))
        self.assertEqual(vocab.reason("beta").label, "Beta")

    def test_a_reused_code_fails_closed(self) -> None:
        _write_pass_vocab(self.directory, 1, [_reason("alpha", "1"), _reason("alpha", "2")])
        with self.assertRaises(VocabularyError):
            load_pass_vocabulary(directory=self.directory)

    def test_a_family_reads_only_its_own_files(self) -> None:
        """A veto file in the folder is not a pass version, and vice versa."""
        _write_vocab(self.directory, 7, [_reason("alpha", "1")])
        self.assertEqual(available_pass_versions(self.directory), ())
        _write_pass_vocab(self.directory, 3, [_reason("beta", "2")])
        self.assertEqual(available_pass_versions(self.directory), (3,))
        self.assertEqual(available_veto_versions(self.directory), (7,))

    def test_a_file_that_declares_the_wrong_family_is_refused(self) -> None:
        path = self.directory / "pass_reasons_v1.json"
        path.write_text(
            json.dumps(
                {
                    "vocabulary_id": "veto_reasons",
                    "vocab_version": 1,
                    "reasons": [_reason("alpha", "1")],
                }
            ),
            encoding="utf-8",
        )
        clear_vocabulary_cache()
        with self.assertRaises(VocabularyError):
            load_pass_vocabulary(directory=self.directory)


class PassRowTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_vocabulary_cache()
        self.vocab = load_pass_vocabulary()

    def test_a_pass_carries_every_ticked_reason(self) -> None:
        row = build_annotation(
            EVENT_PASS,
            symbol="aapl",
            side="LONG",
            reason_codes=[self.vocab.codes[2], self.vocab.codes[0]],
            note="one issue",
        )
        self.assertEqual(row["event_type"], EVENT_PASS)
        self.assertEqual(row["symbol"], "AAPL")
        self.assertEqual(row["note"], "one issue")
        # Never a literal: the version is whatever the loaded file declares.
        self.assertEqual(row["vocab_version"], self.vocab.vocab_version)
        self.assertEqual(row["vocabulary_id"], self.vocab.vocabulary_id)

    def test_reasons_are_written_in_vocabulary_order_not_click_order(self) -> None:
        first, second = self.vocab.codes[0], self.vocab.codes[3]
        clicked = build_annotation(
            EVENT_PASS, symbol="AAPL", reason_codes=[second, first]
        )
        self.assertEqual(clicked["reason_codes"], [first, second])

    def test_a_repeated_tick_is_recorded_once(self) -> None:
        code = self.vocab.codes[1]
        row = build_annotation(EVENT_PASS, symbol="AAPL", reason_codes=[code, code])
        self.assertEqual(row["reason_codes"], [code])

    def test_a_pass_with_no_reason_is_refused(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(EVENT_PASS, symbol="AAPL", reason_codes=[])

    def test_a_reason_outside_the_vocabulary_is_refused(self) -> None:
        with self.assertRaises(AnnotationError):
            build_annotation(
                EVENT_PASS, symbol="AAPL", reason_codes=["made_up_reason"]
            )

    def test_the_note_stays_optional(self) -> None:
        row = build_annotation(
            EVENT_PASS, symbol="AAPL", reason_codes=[self.vocab.codes[0]]
        )
        self.assertNotIn("note", row)

    def test_the_timestamp_is_zoned_so_a_chart_can_be_found_by_it(self) -> None:
        """The trader's fallback: "just store the exact timestamp"."""
        row = build_annotation(
            EVENT_PASS, symbol="AAPL", reason_codes=[self.vocab.codes[0]]
        )
        self.assertIsNotNone(datetime.fromisoformat(row["created_at"]).tzinfo)


class PassBarSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_vocabulary_cache()
        self._tmp = TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.directory / "trader_annotations.jsonl"
        self.code = load_pass_vocabulary().codes[0]

    def _bars(self, day: int, count: int = 3) -> list[dict]:
        return [
            {
                "dt": datetime(2026, 8, day, 9, 30 + 5 * index),
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "volume": 1000.0 + index,
            }
            for index in range(count)
        ]

    def test_cached_bars_are_attached_through_a_sidecar(self) -> None:
        row = record_pass_annotation(
            symbol="AAPL",
            reason_codes=[self.code],
            m5_bars=self._bars(31),
            path=self.path,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["m5_bar_count"], 3)
        stored = pass_bars.read_pass_bars(row, annotations_path=self.path)
        self.assertEqual(len(stored["bars"]), 3)
        self.assertEqual(stored["event_id"], row["event_id"])
        self.assertEqual(stored["bars"][0]["close"], 10.2)

    def test_only_the_newest_session_is_kept(self) -> None:
        """The desk hands out two sessions; a pass is about today's chart."""
        row = record_pass_annotation(
            symbol="AAPL",
            reason_codes=[self.code],
            m5_bars=self._bars(30, 4) + self._bars(31, 2),
            path=self.path,
        )
        stored = pass_bars.read_pass_bars(row, annotations_path=self.path)
        self.assertEqual(row["m5_bar_count"], 2)
        self.assertTrue(all(bar["dt"].startswith("2026-08-31") for bar in stored["bars"]))

    def test_with_nothing_cached_the_row_still_writes_with_its_timestamp(self) -> None:
        row = record_pass_annotation(
            symbol="AAPL", reason_codes=[self.code], m5_bars=[], path=self.path
        )
        self.assertIsNotNone(row)
        self.assertNotIn("m5_bars_ref", row)
        self.assertTrue(row["created_at"])
        self.assertEqual(len(load_annotations(self.path, event_types=(EVENT_PASS,))), 1)

    def test_a_failed_sidecar_costs_the_bars_and_never_the_row(self) -> None:
        """Evidence stores are never allowed to cost the thing they record."""
        blocker = pass_bars.sidecar_dir(self.path)
        blocker.write_text("not a directory", encoding="utf-8")
        row = record_pass_annotation(
            symbol="AAPL",
            reason_codes=[self.code],
            m5_bars=self._bars(31),
            path=self.path,
        )
        self.assertIsNotNone(row)
        self.assertNotIn("m5_bars_ref", row)
        self.assertEqual(len(load_annotations(self.path, event_types=(EVENT_PASS,))), 1)

    def test_a_referenced_sidecar_always_exists_on_disk(self) -> None:
        """Sidecar first, row second - a reference in the stream never lies."""
        row = record_pass_annotation(
            symbol="AAPL",
            reason_codes=[self.code],
            m5_bars=self._bars(31),
            path=self.path,
        )
        self.assertTrue((self.path.parent / row["m5_bars_ref"]).is_file())

    def test_a_missing_sidecar_reads_as_empty_rather_than_raising(self) -> None:
        row = record_pass_annotation(
            symbol="AAPL",
            reason_codes=[self.code],
            m5_bars=self._bars(31),
            path=self.path,
        )
        (self.path.parent / row["m5_bars_ref"]).unlink()
        self.assertEqual(pass_bars.read_pass_bars(row, annotations_path=self.path), {})

    def test_a_failed_append_reports_none(self) -> None:
        """An unwritable stream is reported, never raised at the capture."""
        blocked = self.directory / "as_a_directory"
        blocked.mkdir()
        self.assertIsNone(
            record_pass_annotation(
                symbol="AAPL", reason_codes=[self.code], m5_bars=[], path=blocked
            )
        )

    def test_a_pass_row_never_carries_a_suppression_field(self) -> None:
        """plan.md sec 5: this stream annotates, and has no way to mute."""
        row = record_pass_annotation(
            symbol="AAPL", reason_codes=[self.code], m5_bars=[], path=self.path
        )
        forbidden = {"suppress", "suppressed", "mute", "muted", "hide", "score"}
        self.assertEqual(set(row) & forbidden, set())


if __name__ == "__main__":
    unittest.main()
