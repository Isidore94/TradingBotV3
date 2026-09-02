"""P10 Part A - one like and one dislike, from every screen, note optional.

Trader, 2026-09-02: *"the veto and like+claim tabs are just quicker ways to make
a note for a stock. when I hit the dislike button in master avwap setups or
not-for-today in visual chart review I SHOULD get a little pop-up that lets me
write a note if I am not using the quick buttons. same if I like a stock.
sometimes I may not want to write a note but the fact I clicked like should be
processed by the bot eventually."*

And, decisively: a star in Master AVWAP setups and a like in chart review are the
SAME thing. **One bucket, graded together, and the screen it came from is a
column** - never two cohorts.

What was true before this packet, measured on the live tree: the Master AVWAP ★
and ✕ wrote a review event and reached no graded cohort at all; "Not today" wrote
a `pick_feedback` verdict with the hardcoded string `"not today"`; only the
capture rail's like wrote a `trader_annotations` row. Three writers, one of them
graded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# A1 - the surface, and one row per click
# ---------------------------------------------------------------------------


def test_a_like_from_each_surface_writes_one_row_naming_its_screen(tmp_path):
    from ui.annotations import verdicts

    path = tmp_path / "trader_annotations.jsonl"
    for surface in verdicts.SURFACES:
        verdicts.record_like(
            symbol="NVDA", side="LONG", surface=surface, path=path
        )

    rows = _rows(path)
    assert len(rows) == len(verdicts.SURFACES), "one click, one row"
    assert [row["surface"] for row in rows] == list(verdicts.SURFACES)
    assert {row["event_type"] for row in rows} == {"like_claim"}
    # One bucket: the mode is what a later split reads, and it is the same for
    # every screen. The SCREEN is a column, never a second cohort.
    assert {row["like_mode"] for row in rows} == {"quick"}


def test_an_unknown_screen_is_refused_rather_than_written(tmp_path):
    """Rows are never rewritten, so a typo would be permanent."""
    from ui.annotations.store import AnnotationError
    from ui.annotations import verdicts

    path = tmp_path / "trader_annotations.jsonl"
    with pytest.raises(AnnotationError) as caught:
        verdicts.record_like(symbol="NVDA", surface="chart-review", path=path)
    assert "unknown surface" in str(caught.value)
    assert _rows(path) == []


def test_an_uncoded_veto_carries_no_vocabulary_version(tmp_path):
    """A version stamp without a code would pool it with somebody else's cohort.

    `_rebuild_pooled_performance` pools on `(vocab_version, reason_code)`. A row
    that cites no code but claims a version is a row in a pool it was never part
    of, and the pooling happens at rebuild time over rows that are never
    rewritten - so the mistake would be permanent and invisible.
    """
    from ui.annotations import verdicts

    path = tmp_path / "trader_annotations.jsonl"
    written = verdicts.record_not_today(symbol="AMD", side="LONG", path=path)

    assert written is not None
    assert written["event_type"] == "veto"
    assert written["surface"] == "chart_review"
    assert "reason_code" not in written
    assert "vocab_version" not in written


def test_a_coded_dislike_still_validates_its_code(tmp_path):
    """The uncoded path is an addition; it must not weaken the coded one."""
    from ui.annotations.store import AnnotationError
    from ui.annotations import verdicts

    path = tmp_path / "trader_annotations.jsonl"
    with pytest.raises(AnnotationError):
        verdicts.record_dislike(
            symbol="AMD",
            surface=verdicts.SURFACE_MASTER_AVWAP,
            reason_code="not_a_real_code",
            path=path,
        )
    assert _rows(path) == []


# ---------------------------------------------------------------------------
# A2 - the note is a SECOND row, and the click stands without it
# ---------------------------------------------------------------------------


def test_a_click_with_no_note_leaves_exactly_one_row(tmp_path):
    """*"sometimes I may not want to write a note but the fact I clicked like
    should be processed by the bot eventually."*"""
    from ui.annotations import verdicts

    path = tmp_path / "trader_annotations.jsonl"
    written = verdicts.record_like(
        symbol="NVDA", side="LONG", surface=verdicts.SURFACE_RAIL, path=path
    )
    assert verdicts.record_note_on(written, "", path=path) is None
    assert verdicts.record_note_on(written, "   ", path=path) is None

    assert len(_rows(path)) == 1


def test_a_click_with_a_note_leaves_two_rows_joined_by_supersedes(tmp_path):
    from ui.annotations import verdicts

    path = tmp_path / "trader_annotations.jsonl"
    written = verdicts.record_like(
        symbol="NVDA", side="LONG", surface=verdicts.SURFACE_MASTER_AVWAP, path=path
    )
    note = verdicts.record_note_on(written, "base holding, want the retest", path=path)

    rows = _rows(path)
    assert len(rows) == 2
    assert note["supersedes"] == written["event_id"]
    assert note["note"] == "base holding, want the retest"
    # NEVER AN EDIT. The click row on disk is byte-identical to what was written.
    assert rows[0] == written
    assert "note" not in rows[0]
    # It stands alone: identity is repeated, not only pointed at.
    assert note["symbol"] == "NVDA" and note["surface"] == "master_avwap_setups"


def test_a_note_on_a_coded_veto_keeps_the_code(tmp_path):
    """Otherwise the pair would grade in two different cohorts."""
    from ui.annotations import verdicts
    from ui.annotations.vocabulary import load_veto_vocabulary

    code = load_veto_vocabulary().reasons[0].code
    path = tmp_path / "trader_annotations.jsonl"
    written = verdicts.record_dislike(
        symbol="AMD",
        side="LONG",
        surface=verdicts.SURFACE_MASTER_AVWAP,
        reason_code=code,
        note="a required detail",
        path=path,
    )
    note = verdicts.record_note_on(written, "and one more thought", path=path)

    assert note["reason_code"] == written["reason_code"] == code


# ---------------------------------------------------------------------------
# B1 - the scanner row under the click
# ---------------------------------------------------------------------------


class _Row:
    """Shaped like a `SetupRow` far enough for the stamping."""

    def __init__(self, **fields):
        self.symbol = fields.pop("symbol", "NVDA")
        self.side = fields.pop("side", "LONG")
        self.score = fields.pop("score", None)
        self.expected_r = fields.pop("expected_r", None)
        self.bucket = fields.pop("bucket", "")
        self.raw = fields.pop("raw", {})


def test_a_like_on_a_scanner_row_records_which_search_found_it():
    """*"anytime I like a D1 it should be treated with respect by the bot in
    regards to finding out what's good about it, how we can replicate those
    searches"* - and a search cannot be replicated from a symbol and a clock."""
    from ui.annotations import verdicts

    context = verdicts.scan_context_from_row(
        _Row(
            score=91.5,
            expected_r=2.4,
            bucket="favorite_setup",
            raw={"setup_family": "AVWAP_BREAKOUT", "scan_date": "2026-09-02"},
        )
    )

    assert context["score"] == 91.5
    assert context["expected_r"] == 2.4
    assert context["priority_bucket"] == "favorite_setup"
    assert context["scan_date"] == "2026-09-02"
    # P7's registry spells the canonical id in the family's own case; asserted
    # against the registry rather than against a guess at its convention.
    import setup_registry

    assert context["canonical_setup_id"] == setup_registry.canonical_setup_id(
        "AVWAP_BREAKOUT"
    )


def test_a_bare_lookup_stamps_nothing_and_never_fetches(tmp_path):
    """Absent is a real answer; "" would be indistinguishable from measured."""
    from ui.annotations import verdicts

    assert verdicts.scan_context_from_row(None) == {}

    path = tmp_path / "trader_annotations.jsonl"
    written = verdicts.record_like(
        symbol="ZZZZ",
        side="LONG",
        surface=verdicts.SURFACE_CHART_REVIEW,
        scan_context=verdicts.scan_context_from_row(None),
        path=path,
    )
    for field in ("score", "expected_r", "priority_bucket", "canonical_setup_id"):
        assert field not in written


def test_an_unmapped_family_costs_the_id_and_never_the_verdict():
    """P7's registry RAISES on an unknown name, deliberately. Not here."""
    from ui.annotations import verdicts

    context = verdicts.scan_context_from_row(
        _Row(score=1.0, raw={"setup_family": "A_FAMILY_NOBODY_REGISTERED"})
    )
    assert context["score"] == 1.0
    assert "canonical_setup_id" not in context


# ---------------------------------------------------------------------------
# The schema stays at 1, proven against the live readers
# ---------------------------------------------------------------------------


def test_the_new_fields_are_additive_and_every_reader_still_answers(tmp_path):
    from ui.annotations import like_cohort, verdicts
    from ui.annotations.store import SCHEMA_VERSION, load_annotations

    assert SCHEMA_VERSION == 1, "these fields are additive; nothing needs a bump"

    path = tmp_path / "trader_annotations.jsonl"
    verdicts.record_like(
        symbol="NVDA",
        side="LONG",
        surface=verdicts.SURFACE_MASTER_AVWAP,
        session_date="2026-09-02",
        scan_context={"score": 91.5, "canonical_setup_id": "avwap_breakout"},
        path=path,
    )

    loaded = load_annotations(path)
    assert len(loaded) == 1 and loaded[0]["surface"] == "master_avwap_setups"

    rows, skipped = like_cohort.like_pick_rows(loaded)
    assert skipped == 0
    assert len(rows) == 1, "a P10 row still grades as a like"
    assert rows[0]["source"] == "like_unclaimed"


# ---------------------------------------------------------------------------
# A3 - one bucket, and the screen as a column
# ---------------------------------------------------------------------------


def test_likes_from_every_screen_land_in_ONE_cohort(tmp_path):
    """The trader's rule, tested directly: a star and a like are one thing."""
    from ui.annotations import like_cohort, verdicts
    from ui.annotations.store import load_annotations

    path = tmp_path / "trader_annotations.jsonl"
    for index, surface in enumerate(verdicts.SURFACES):
        verdicts.record_like(
            symbol=f"SYM{index}",
            side="LONG",
            surface=surface,
            session_date="2026-09-02",
            path=path,
        )

    rows, skipped = like_cohort.like_pick_rows(load_annotations(path))

    assert skipped == 0
    assert len(rows) == len(verdicts.SURFACES)
    assert {row["source"] for row in rows} == {"like_unclaimed"}, "one bucket"
    assert sorted(row["surface"] for row in rows) == sorted(verdicts.SURFACES)


def test_a_quick_like_is_never_pooled_into_a_claimed_setups_cohort(tmp_path):
    from ui.annotations import like_cohort, verdicts
    from ui.annotations.store import LIKE_MODE_CLAIMED, load_annotations

    path = tmp_path / "trader_annotations.jsonl"
    verdicts.record_like(
        symbol="NVDA",
        side="LONG",
        surface=verdicts.SURFACE_MASTER_AVWAP,
        session_date="2026-09-02",
        path=path,
    )
    verdicts.record_like(
        symbol="AMD",
        side="LONG",
        surface=verdicts.SURFACE_RAIL,
        session_date="2026-09-02",
        like_mode=LIKE_MODE_CLAIMED,
        claimed_setup_id="avwap_breakout",
        note="why",
        path=path,
    )

    rows, _ = like_cohort.like_pick_rows(load_annotations(path))
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["NVDA"]["source"] == "like_unclaimed"
    assert by_symbol["AMD"]["source"] != "like_unclaimed"


def test_an_uncoded_veto_grades_under_its_own_name(tmp_path):
    """It used to be dropped: `like_pick_rows`' veto twin skipped a codeless row.

    So the most-used dismissal on the desk - "Not today" - had no forward record
    at all, while a coded veto from the same afternoon did.
    """
    from ui.annotations import veto_cohort, verdicts
    from ui.annotations.store import load_annotations
    from ui.annotations.vocabulary import load_veto_vocabulary

    path = tmp_path / "trader_annotations.jsonl"
    verdicts.record_not_today(
        symbol="AMD", side="LONG", session_date="2026-09-02", path=path
    )
    vocabulary = load_veto_vocabulary()
    code = vocabulary.reasons[0].code
    verdicts.record_dislike(
        symbol="NVDA",
        side="LONG",
        surface=verdicts.SURFACE_MASTER_AVWAP,
        session_date="2026-09-02",
        reason_code=code,
        note="a detail",
        path=path,
    )

    rows, skipped = veto_cohort.veto_pick_rows(load_annotations(path))
    by_symbol = {row["symbol"]: row for row in rows}

    assert skipped == 0
    assert by_symbol["AMD"]["source"] == veto_cohort.VETO_UNCODED_SOURCE
    assert by_symbol["AMD"]["source"] == "veto_uncoded"
    # Never pooled with a coded one, and never carrying a version it cannot cite.
    assert by_symbol["NVDA"]["source"] != by_symbol["AMD"]["source"]
    assert str(vocabulary.vocab_version) not in by_symbol["AMD"]["source"]


def test_the_uncoded_cohort_still_belongs_to_the_veto_family():
    """`_outcome_base_cohort` matches on `startswith(prefix + "_")`."""
    from human_focus_tracking import COHORT_BASE_BY_SOURCE_PREFIX
    from ui.annotations.veto_cohort import VETO_UNCODED_SOURCE

    matched = [
        base
        for base, prefix in COHORT_BASE_BY_SOURCE_PREFIX
        if VETO_UNCODED_SOURCE == prefix
        or VETO_UNCODED_SOURCE.startswith(prefix + "_")
    ]
    assert matched == ["human_focus_veto"], matched
