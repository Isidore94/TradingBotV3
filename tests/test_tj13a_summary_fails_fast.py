"""TJ-13A item 3: ``ai_summary`` fails fast or finishes.

Reproduced from the live ledger, verbatim (2026-09-17 row,
``\\\\MINI-PC\\Trading Bot Data\\ai_store\\logs\\ai_job_ledger.jsonl``)::

    summary for 2026-09-17 from 19 usable source(s) read as 53 of 53 slice(s);
    completion=unsynthesized_fallback after RuntimeError: local AI endpoint at
    http://127.0.0.1:11434/v1/chat/completions is unreachable:
    HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out.
    (read timeout=900)

Three nights in a row (09-15, 09-16, 09-17) and again on 09-18, at
12,453-18,540 s a night. Two separate defects sit in that one line:

1. **53 of 53 slices were read and then the synthesis call timed out.** The
   reduce package is built from every finding the night produced and is handed
   to the model unbounded, so the one call that has to hold the whole night is
   the only one nothing budgets. It gets the existing budget setting.
2. **A dead endpoint costs the whole night before anything says so.** Every
   slice is attempted in turn and each one has to reach its own timeout, so an
   endpoint that is not answering at 22:00 is discovered at 03:00. It gives up
   on the first call instead.

NO REAL MODEL IS CALLED HERE. Every request is a fake that raises or answers
from a literal, which is also the trader's rule: inference is night-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import map_reduce  # noqa: E402

#: The exception text the live ledger recorded, verbatim.
UNREACHABLE = (
    "local AI endpoint at http://127.0.0.1:11434/v1/chat/completions is "
    "unreachable: HTTPConnectionPool(host='127.0.0.1', port=11434): "
    "Read timed out. (read timeout=900)"
)


def _package(sources):
    return {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": "2026-09-18T22:00:00-07:00",
        "session_date": "2026-09-17",
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


def test_an_unreachable_endpoint_is_found_on_the_first_call_not_the_fifty_third():
    """Nine slices, one dead endpoint, and the night must cost ONE attempt.

    At the measured 900 s read timeout, 53 slices against a silent server is
    over thirteen hours of window spent discovering a fact the first call knew.
    """
    evidence = _package(
        [(f"store.source_{index}", [f"row {index}"]) for index in range(9)]
    )
    calls = {"n": 0}

    def dead(**kwargs):
        calls["n"] += 1
        raise RuntimeError(UNREACHABLE)

    with pytest.raises(RuntimeError):
        map_reduce.run_map_reduce(evidence=evidence, model="gemma3:12b", request=dead)

    assert calls["n"] == 1, (
        "an endpoint that is unreachable on the first call must not be asked "
        f"{calls['n']} times"
    )


def test_a_single_failing_slice_still_does_not_stop_the_night():
    """The other side of the same line, and it is a guard.

    ``test_a_failed_slice_is_counted_and_named_never_skipped_quietly`` in
    ``tests/test_ai_map_reduce.py`` says a slice failure costs its own slice and
    nothing else. Giving up early must key on the endpoint being UNREACHABLE,
    never on any failure at all.
    """
    evidence = _package([("a.one", ["x"]), ("b.two", ["y"]), ("c.three", ["z"])])
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the model returned no text content")
        return {"summary": _summary("finding", ["a.one"])}

    result = map_reduce.run_map_reduce(evidence=evidence, model="m", request=flaky)

    assert result["map_reduce"]["slices_planned"] == 3
    assert result["map_reduce"]["slices_read"] == 2
    assert len(result["map_reduce"]["slices_failed"]) == 1


def test_the_synthesis_call_is_bounded_by_the_local_evidence_budget():
    """The one call that holds the whole night is the one nothing budgeted.

    53 slices of findings went into a single reduce prompt; the server answers
    at ~8 tok/s on this desk, so an unbounded package is how a 900 s read
    timeout became the normal ending. The bound is the existing setting, not a
    new number.
    """
    import ai_summary

    budget = ai_summary.evidence_budget_for("local", tier="medium")
    # 40 sources, each big enough to yield a long finding: a realistic 53-slice
    # night, built here so the assertion is about the REDUCE package's size.
    evidence = _package(
        [(f"store.source_{index}", [f"row {index}"]) for index in range(40)]
    )
    reduce_packages = []

    def answer(**kwargs):
        package = kwargs["evidence"]
        source_id = package["sources"][0]["source_id"]
        if source_id == map_reduce.FINDINGS_SOURCE_ID:
            reduce_packages.append(package)
            return {"summary": _summary("synthesized", [map_reduce.FINDINGS_SOURCE_ID])}
        return {
            "summary": _summary(
                f"a finding about {source_id} " + ("detail " * 400), [source_id]
            )
        }

    map_reduce.run_map_reduce(evidence=evidence, model="gemma3:12b", request=answer)

    assert len(reduce_packages) == 1
    encoded = json.dumps(reduce_packages[0], sort_keys=True, default=str)
    assert len(encoded) <= budget, (
        f"the synthesis package is {len(encoded)} chars against a local budget "
        f"of {budget}"
    )


def test_a_bounded_synthesis_still_says_what_it_left_out():
    """Bounding is not silent truncation: missing data is uncertainty.

    Whatever the reduce package drops to fit must be counted where the reader of
    the published document can see it, the way ``coverage_statement`` already
    names a failed slice.
    """
    evidence = _package(
        [(f"store.source_{index}", [f"row {index}"]) for index in range(40)]
    )

    def answer(**kwargs):
        package = kwargs["evidence"]
        source_id = package["sources"][0]["source_id"]
        if source_id == map_reduce.FINDINGS_SOURCE_ID:
            return {"summary": _summary("synthesized", [map_reduce.FINDINGS_SOURCE_ID])}
        return {
            "summary": _summary(
                f"a finding about {source_id} " + ("detail " * 400), [source_id]
            )
        }

    result = map_reduce.run_map_reduce(
        evidence=evidence, model="gemma3:12b", request=answer
    )
    stats = result["map_reduce"]

    assert "findings_dropped_to_fit" in stats, (
        "a synthesis that could not carry every finding must say how many it "
        "left behind, never present the remainder as the whole"
    )
    assert isinstance(stats["findings_dropped_to_fit"], int)
    assert stats["findings_dropped_to_fit"] > 0
