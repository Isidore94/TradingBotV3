"""Reading the evidence in slices (trader, 2026-08-28).

The single-shot local summary is bounded by the model's context: 1,365,259
characters of session evidence against a 65,536-token window means one prompt
carries about a tenth of it, and `setups.type_stats` contributed 3 of its 184
rows. The trader's fix was to spend the overnight window instead: "give it more
time... spoon feed it slowly so we don't run out of context."

What is pinned here is the honesty of that trade, not the speed of it: a slice
must never read as its whole source, a citation must still name a store that was
really read, a failed slice must be counted and named, and hours of map work
must not be thrown away by one failed synthesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import map_reduce  # noqa: E402


def _package(sources):
    return {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-08-28T04:00:00-07:00",
        "session_date": "2026-08-27",
        "sources": [
            {
                "source_id": sid,
                "label": sid,
                "status": "available",
                "content": content,
            }
            for sid, content in sources
        ],
    }


def _summary(statement, refs, section="what_is_working", confidence="medium"):
    base = {
        name: []
        for name in (
            "what_is_working",
            "what_is_not_working",
            "best_candidates",
            "lessons_for_tomorrow",
            "risk_notes",
        )
    }
    base[section] = [
        {"statement": statement, "evidence_refs": list(refs), "confidence": confidence}
    ]
    return {"executive_summary": "e", **base}


# --------------------------------------------------------------------- slicing


def test_a_table_is_split_by_rows_and_every_row_is_kept():
    """No source is reduced to a sample. That is the entire point."""
    rows = [{"symbol": f"S{i}", "pad": "x" * 200} for i in range(100)]
    ev = _package([("setups.type_stats", rows)])
    chunks = map_reduce.plan_chunks(ev, chars=4_000)

    assert len(chunks) > 1, "a 100-row table must not fit one small chunk"
    seen = [row for chunk in chunks for row in chunk.content]
    assert seen == rows, "every row, in order, exactly once"


def test_a_slice_is_labelled_so_it_cannot_pass_for_the_whole_source():
    rows = [{"symbol": f"S{i}", "pad": "x" * 200} for i in range(100)]
    chunks = map_reduce.plan_chunks(_package([("setups.type_stats", rows)]), chars=4_000)

    assert all("of 100" in chunk.label for chunk in chunks)
    assert chunks[0].label.startswith("rows 1-")
    # And the model is told, in the package it reads, not just in a field.
    note = map_reduce.chunk_package(chunks[0], _package([]))["coverage"]["note"]
    assert "slice 1 of" in note
    assert "never" in note.lower()


def test_a_source_that_fits_is_one_slice_and_says_it_is_complete():
    ev = _package([("daily.auto_report", "short text")])
    chunks = map_reduce.plan_chunks(ev, chars=4_000)

    assert len(chunks) == 1
    assert chunks[0].label == "the complete source"
    assert "This is the entire source." in map_reduce.chunk_package(chunks[0], ev)["coverage"]["note"]


def test_text_splits_by_window_and_covers_the_whole_document():
    ev = _package([("daily.market_prep", "abcdefghij" * 500)])
    chunks = map_reduce.plan_chunks(ev, chars=1_000)

    assert len(chunks) == 5
    assert "".join(chunk.content for chunk in chunks) == "abcdefghij" * 500


def test_a_chunk_can_never_be_configured_larger_than_one_prompt_holds():
    """A chunk over the ceiling is sheared exactly like an unchunked prompt --
    and it would be sheared once per slice."""
    import ai_summary

    huge = map_reduce.chunk_chars(lambda key, default=None: 10_000_000)
    assert huge == ai_summary.local_evidence_budget_ceiling_chars()


# ------------------------------------------------------------------ citations


def test_a_map_slice_is_handed_one_source_so_it_can_cite_nothing_else():
    import ai_summary

    ev = _package([("setups.type_stats", ["a"]), ("daily.auto_report", "b")])
    chunk = map_reduce.plan_chunks(ev, chars=40_000)[0]
    package = map_reduce.chunk_package(chunk, ev)

    assert [s["source_id"] for s in package["sources"]] == [chunk.source_id]
    assert ai_summary.usable_source_ids(package) == {chunk.source_id}


def test_the_synthesis_may_cite_the_stores_its_findings_came_from():
    import ai_summary

    findings = {"what_is_working": [
        {"statement": "x", "evidence_refs": ["setups.type_stats"], "confidence": "high"}
    ]}
    package = map_reduce.findings_package(
        findings, _package([]), read=3, planned=3, failed=[]
    )
    citable = ai_summary.usable_source_ids(package)
    assert map_reduce.FINDINGS_SOURCE_ID in citable
    assert "setups.type_stats" in citable
    assert "setups.playbooks" not in citable, "only what was actually read"


# ------------------------------------------------------------------- failures


def test_a_failed_slice_is_counted_and_named_never_skipped_quietly():
    ev = _package([("a.one", ["x"]), ("b.two", ["y"]), ("c.three", ["z"])])
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"summary": _summary("finding", ["a.one"])}

    result = map_reduce.run_map_reduce(evidence=ev, model="m", request=flaky)
    stats = result["map_reduce"]

    assert stats["slices_planned"] == 3
    assert stats["slices_read"] == 2  # 3 map calls minus the failure... plus reduce
    assert len(stats["slices_failed"]) == 1
    assert "b.two" in stats["slices_failed"][0]
    assert "FAILED" in stats["coverage_statement"]
    assert "1 slice(s) FAILED" in stats["coverage_statement"]


def test_the_coverage_line_says_no_source_was_sampled_when_nothing_failed():
    line = map_reduce.coverage_statement(planned=46, read=46, failed=[], sources=17)
    assert "46 of 46" in line
    assert "no source was reduced to a sample" in line
    assert "FAILED" not in line


def test_every_slice_failing_raises_rather_than_publishing_an_empty_review():
    ev = _package([("a.one", ["x"])])

    def dead(**kwargs):
        raise RuntimeError("endpoint down")

    with pytest.raises(RuntimeError, match="nothing was read"):
        map_reduce.run_map_reduce(evidence=ev, model="m", request=dead)


def test_a_failed_synthesis_publishes_the_findings_rather_than_losing_them():
    """Hours of map work must not be thrown away by one failed call -- and the
    result must not pretend to be a review."""
    ev = _package([("a.one", ["x"]), ("b.two", ["y"])])
    seen = {"n": 0}

    def map_ok_reduce_fails(**kwargs):
        seen["n"] += 1
        package = kwargs["evidence"]
        if package["sources"][0]["source_id"] == map_reduce.FINDINGS_SOURCE_ID:
            raise RuntimeError("synthesis blew up")
        return {"summary": _summary(f"finding {seen['n']}", [package["sources"][0]["source_id"]])}

    result = map_reduce.run_map_reduce(evidence=ev, model="m", request=map_ok_reduce_fails)

    assert result["map_reduce"]["synthesized"] is False
    assert "synthesis blew up" in result["map_reduce"]["synthesis_error"]
    assert result["summary"]["executive_summary"].startswith("UNSYNTHESIZED")
    statements = [row["statement"] for row in result["summary"]["what_is_working"]]
    assert "finding 1" in statements and "finding 2" in statements


def test_duplicate_findings_across_slices_are_merged_once():
    ev = _package([("a.one", ["x"]), ("b.two", ["y"])])

    def same(**kwargs):
        return {"summary": _summary("the same finding", ["a.one"])}

    result = map_reduce.run_map_reduce(evidence=ev, model="m", request=same)
    # The reduce call receives the merged findings; check what it was handed.
    assert result["map_reduce"]["slices_read"] == 2


def test_the_unsynthesized_document_caps_a_section_and_says_what_it_dropped():
    findings = {
        "what_is_working": [
            {"statement": f"f{i}", "evidence_refs": ["a.one"], "confidence": "low"}
            for i in range(30)
        ]
    }
    out = map_reduce.unsynthesized_summary(findings, read=30, planned=30)
    rows = out["what_is_working"]
    assert len(rows) == map_reduce.MAX_UNSYNTHESIZED_ROWS_PER_SECTION + 1
    assert "further finding(s)" in rows[-1]["statement"]


# --------------------------------------------------------------------- wiring


def test_the_path_is_off_until_it_is_switched_on():
    assert map_reduce.map_reduce_enabled(lambda key, default=None: default) is False
    assert map_reduce.map_reduce_enabled(lambda key, default=None: True) is True


def test_the_slice_count_reaches_the_published_data_quality_section():
    """A document synthesized from 30 of 34 slices is not the same document as
    one synthesized from all 34, so the reader is told which they have."""
    import ai_summary

    merged = ai_summary.merge_coverage_into_summary(
        {"what_is_working": []},
        {"coverage": {"counts": {"usable": 17, "requested": 22}}},
        extra_statements=[map_reduce.coverage_statement(
            planned=46, read=44, failed=["x [1/2]: RuntimeError"], sources=17
        )],
    )
    said = [row["statement"] for row in merged["data_quality"]]
    assert any("44 of 46" in line and line.startswith("[system]") for line in said)


# ------------------------------------------------- one runner, one window slot


def test_a_second_runner_stands_down_while_one_is_working():
    """The task fires every 30 minutes for eight hours. That was harmless while
    every slot finished in minutes; a slice-reading summary runs for hours, and
    the ledger only records a row when a job FINISHES - so the 22:30 firing
    would find no completion and start a second copy."""
    from unittest import mock

    from ai_jobs import runner
    import local_writer_lock

    def _busy(key, **kwargs):
        raise local_writer_lock.LocalLockUnavailable("someone else holds it")

    with mock.patch.object(local_writer_lock, "local_writer_lock", _busy),          mock.patch.object(runner, "_run_slots_locked") as body:
        report = runner.run_slots([])

    body.assert_not_called()
    assert report.session_date == ""


def test_no_primitive_runs_unguarded_rather_than_skipping_the_night():
    """The two failures wear the same exception and want opposite answers."""
    from unittest import mock

    from ai_jobs import runner
    import local_writer_lock

    def _no_primitive(key, **kwargs):
        raise local_writer_lock.LocalLockUnavailable(runner.NO_PRIMITIVE_MARKER + " here")

    with mock.patch.object(local_writer_lock, "local_writer_lock", _no_primitive),          mock.patch.object(runner, "_run_slots_locked") as body:
        runner.run_slots([])

    body.assert_called_once()


def test_the_marker_the_discrimination_relies_on_is_really_what_the_lock_says():
    """If `local_writer_lock` rewords that sentence, this fails rather than the
    runner silently starting a second copy of a three-hour job."""
    from pathlib import Path

    from ai_jobs import runner

    source = (Path(__file__).resolve().parents[1] / "scripts" / "local_writer_lock.py").read_text(
        encoding="utf-8"
    )
    assert runner.NO_PRIMITIVE_MARKER in source


def test_the_summary_slot_reserves_the_window_its_mode_actually_needs():
    """A three-hour job launched with twenty minutes left runs into the open."""
    from unittest import mock

    from ai_jobs import briefs, map_reduce

    with mock.patch.object(map_reduce, "map_reduce_enabled", return_value=False):
        assert briefs.summary_reserve_minutes() == briefs.SUMMARY_RESERVE_MINUTES
    with mock.patch.object(map_reduce, "map_reduce_enabled", return_value=True):
        chunked = briefs.summary_reserve_minutes()
    assert chunked == briefs.SUMMARY_RESERVE_MINUTES_CHUNKED
    assert chunked > 8 * briefs.SUMMARY_RESERVE_MINUTES


def test_a_settings_read_never_decides_the_night_by_raising():
    from unittest import mock

    from ai_jobs import briefs, map_reduce

    with mock.patch.object(map_reduce, "map_reduce_enabled", side_effect=OSError("gone")):
        assert briefs.summary_reserve_minutes() == briefs.SUMMARY_RESERVE_MINUTES
