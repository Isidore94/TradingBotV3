"""AI-R1: code-owned links for day stories and word tags.

The packet repairs two unreliable model contracts.  These tests drive the real
nightly slots with injected callables and scratch stores.  They never call a
provider or resolve a live data root.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tj4_support as day_fx  # noqa: E402
import tj16_support as tag_fx  # noqa: E402


def _write_day_pack(root: Path, pack) -> None:
    import day_review_pack

    day_review_pack.write_pack(pack, root=root)


def _day_v2_reply(evidence: dict) -> dict:
    offered = evidence["read_explanations"]
    assert offered, "fixture drift: the pack must carry one clicked prediction"
    explanations = {
        read_id: "You called the move before it happened."
        for read_id in offered
    }
    return {
        "model": "local-test-medium",
        "summary": {
            "headline": "One measured call from the trader.",
            "what_happened": "The market moved during the measured session.",
            "what_you_thought": "The trader made a stated call.",
            "read_explanations": explanations,
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "Keep the explanation tied to the measured read.",
            "sources": list(evidence["allowed_source_ids"]),
        },
    }


def test_day_story_v2_offers_canonical_pairs_and_materializes_legacy_were_you_right(
    tmp_path,
):
    """AI-R1 item 1: the model writes only an explanation, never its verdict.

    A real pack supplies the minted ids.  The request checks that each offered
    read already has its exact verdict and its matching prediction source.  The
    returned v2 object has no model-selected verdict or source mapping; the
    persisted v1 narration must regain those canonical fields in code.
    """
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    _write_day_pack(root, day_fx.build())
    calls: list[dict] = []

    def request(**kwargs):
        evidence = kwargs["evidence"]
        calls.append(evidence)
        offered = evidence["read_explanations"]
        assert isinstance(offered, dict) and offered
        for read_id, canonical in offered.items():
            assert isinstance(canonical, dict), canonical
            assert canonical["read_source_id"] == read_id
            assert canonical["prediction_source_id"] in evidence["allowed_source_ids"]
            assert canonical["verdict"]
        return _day_v2_reply(evidence)

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION,
        now=day_fx.OVERNIGHT,
        root=root,
        request=request,
        only_this_session=True,
    )

    assert outcome["status"] == "ok", outcome
    assert len(calls) == 1
    saved = narration.read_narration(day_fx.SESSION, root=root)
    assert saved is not None
    canonical = calls[0]["read_explanations"]
    claims = saved["narration"]["were_you_right"]
    assert [claim["evidence_id"] for claim in claims] == list(canonical)
    for claim in claims:
        pair = canonical[claim["evidence_id"]]
        assert claim["source_id"] == pair["prediction_source_id"]
        assert claim["verdict"] == pair["verdict"]
        assert claim["claim"] == "You called the move before it happened."


def test_day_story_rejects_a_prediction_source_from_a_different_entry_byte_identically(
    tmp_path,
):
    """AI-R1 item 1: kind alone is not enough to ground a prediction source."""
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    first = day_fx.observing_and_predicting_entry(direction="up")
    second = day_fx.observing_and_predicting_entry(
        direction="down", stamp=day_fx.fx.STAMP + timedelta(hours=1)
    )
    entries = [first, second]
    reads, _ = day_fx.graded_reads(entries)
    pack = day_fx.build(
        entries=entries,
        reads=reads,
        congruence=day_fx.congruence(reads),
        story=day_fx.daily_story(entries),
    )
    root = tmp_path / "day_review"
    day_review_pack.write_pack(pack, root=root)

    said = {row["entry_id"]: row for row in pack["trader_said"] if row["kind"] == "prediction"}
    reads_by_entry = {row["entry_id"]: row for row in pack["reads"]}
    assert len(said) == len(reads_by_entry) == 2, (said, reads_by_entry)
    first_entry, second_entry = list(reads_by_entry)
    bad_source = said[second_entry]["source_id"]
    measured = reads_by_entry[first_entry]
    prior = narration.narration_path(day_fx.SESSION, root=root)
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text('{"verified":"keep these bytes"}\n', encoding="utf-8")
    before = prior.read_bytes()
    legacy_reply = {
        "model": "local-test-medium",
        "summary": {
            "headline": "The links must be entry-local.",
            "what_happened": "One measured session.",
            "what_you_thought": "Two stated calls.",
            "were_you_right": [{
                "claim": "A cross-entry claim.",
                "source_id": bad_source,
                "verdict": measured["verdict"],
                "evidence_id": measured["source_id"],
            }],
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "The reader must refuse the wrong pairing.",
            "sources": [bad_source, measured["source_id"]],
        },
    }

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION,
        now=day_fx.OVERNIGHT,
        root=root,
        request=lambda **_kwargs: legacy_reply,
        only_this_session=True,
    )

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert prior.read_bytes() == before


def test_observation_tags_v2_uses_distinct_exact_fragments_instead_of_model_offsets(
    tmp_path,
):
    """AI-R1 item 2: code owns offsets, including repeated words at two places."""
    from ai_jobs import observation_tags

    text = "gap and go so far. holding the gains from overnight. gap and go again."
    entry = tag_fx.click_entry(
        session=tag_fx.LAST_SESSION,
        hour=9,
        direction="up",
        confidence="high",
        observation=text,
        because="",
    )
    vocabulary = observation_tags.load_vocabulary()
    calls: list[dict] = []

    def request(**kwargs):
        evidence = kwargs["evidence"]
        calls.append(evidence)
        fragments = evidence["fragments"]
        exact = [row for row in fragments if "gap and go" in row["text"]]
        assert len(exact) == 2, fragments
        assert exact[0]["fragment_id"] != exact[1]["fragment_id"]
        assert exact[0]["start"] != exact[1]["start"]
        code_one, code_two = vocabulary["codes"][:2]
        return {
            "model": "local-test-medium",
            "summary": {
                "tags": [
                    {"fragment_id": exact[0]["fragment_id"], "code": code_one},
                    {"fragment_id": exact[1]["fragment_id"], "code": code_two},
                ]
            },
        }

    root = tmp_path / "tags"
    outcome = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION,
        now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=root,
        entries=[entry],
        request=request,
    )

    assert outcome["status"] == "ok", outcome
    assert len(calls) == 1
    saved = observation_tags.read_latest(tag_fx.LAST_SESSION, root=root)
    assert saved is not None
    tags = saved["tags"]
    chosen = [row for row in calls[0]["fragments"] if "gap and go" in row["text"]]
    assert [tag["quote"] for tag in tags] == [row["text"] for row in chosen]
    assert [tag["span"] for tag in tags] == [
        [row["start"], row["end"]] for row in chosen
    ]
    assert {tag["code"] for tag in tags} == set(vocabulary["codes"][:2])


def test_observation_tags_v2_refuses_unknown_fragment_without_changing_verified_bytes(
    tmp_path,
):
    """AI-R1 item 2: fragment ids are closed, and a mixed/unknown reply is whole-fail."""
    from ai_jobs import observation_tags

    entry = tag_fx.click_entry(
        session=tag_fx.LAST_SESSION,
        hour=9,
        direction="up",
        confidence="high",
        observation="gap and go so far. holding the gains from overnight.",
        because="",
    )
    vocabulary = observation_tags.load_vocabulary()
    root = tmp_path / "tags"

    def good_request(**kwargs):
        fragment = kwargs["evidence"]["fragments"][0]
        return {
            "model": "local-test-medium",
            "summary": {"tags": [{"fragment_id": fragment["fragment_id"], "code": vocabulary["codes"][0]}]},
        }

    first = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION,
        now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=root,
        entries=[entry],
        request=good_request,
    )
    assert first["status"] == "ok", first
    files = sorted(root.glob("*.json"))
    before = {path.name: path.read_bytes() for path in files}

    bad = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION,
        now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=root,
        entries=[entry],
        request=lambda **_kwargs: {
            "model": "local-test-medium",
            "summary": {"tags": [{"fragment_id": "frag-not-offered", "code": vocabulary["codes"][0]}]},
        },
    )

    assert bad["status"] == "degraded_no_narrative", bad
    assert {path.name: path.read_bytes() for path in sorted(root.glob("*.json"))} == before
