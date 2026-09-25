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


def test_day_story_v2_names_reads_without_a_prediction_and_accepts_zero_pairs(tmp_path):
    """An extracted/observation-only read is counted, never made into a click."""
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    entry = day_fx.observing_and_predicting_entry()
    reads, _ = day_fx.graded_reads([entry])
    pack = day_fx.build(
        entries=[entry], reads=reads, congruence=day_fx.congruence(reads),
        story=day_fx.daily_story([entry]),
    )
    pack["trader_said"] = [
        row for row in pack["trader_said"] if row["kind"] != "prediction"
    ]
    root = tmp_path / "day_review"
    day_review_pack.write_pack(pack, root=root)

    def request(**kwargs):
        evidence = kwargs["evidence"]
        assert evidence["read_explanations"] == {}
        assert evidence["read_explanations_not_offered"] == [{
            "read_source_id": pack["reads"][0]["source_id"], "reason": "no_prediction"
        }]
        return {
            "model": "local-test-medium",
            "summary": {
                "headline": "No clicked call was available.",
                "what_happened": "The session was measured.",
                "what_you_thought": "The trader left an observation.",
                "read_explanations": {},
                "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
                "process": "Do not turn an observation into a prediction.",
                "sources": list(evidence["allowed_source_ids"]),
            },
        }

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=request, only_this_session=True,
    )
    assert outcome["status"] == "ok", outcome
    assert narration.read_narration(day_fx.SESSION, root=root)["narration"]["were_you_right"] == []


def test_day_story_v2_rejects_every_closed_mapping_breach_byte_identically(tmp_path):
    """Missing, unknown and too-long explanations are all whole-answer failures."""
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    for name, broken in (
        ("missing", lambda offered: {}),
        ("unknown", lambda offered: {"not-a-read": "Wrong key"}),
        ("oversize", lambda offered: {next(iter(offered)): "x" * 241}),
    ):
        root = tmp_path / name
        pack = day_fx.build()
        day_review_pack.write_pack(pack, root=root)
        prior = narration.narration_path(day_fx.SESSION, root=root)
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_bytes(b'{"verified":"keep"}\n')
        before = prior.read_bytes()

        def request(_broken=broken, **kwargs):
            evidence = kwargs["evidence"]
            return {
                "model": "local-test-medium",
                "summary": {
                    "headline": "The closed mapping matters.",
                    "what_happened": "One session.",
                    "what_you_thought": "One call.",
                    "read_explanations": _broken(evidence["read_explanations"]),
                    "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
                    "process": "Reject the whole reply.",
                    "sources": list(evidence["allowed_source_ids"]),
                },
            }

        outcome = narration.run_day_review_narration(
            session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
            request=request, only_this_session=True,
        )
        assert outcome["status"] == "degraded_no_narrative", (name, outcome)
        assert prior.read_bytes() == before


def test_observation_v2_keeps_legacy_strict_and_names_fragment_cap(tmp_path):
    """The old offset contract remains strict while v2 reports every omission."""
    from ai_jobs import observation_tags

    vocabulary = observation_tags.load_vocabulary()
    note = {"note_id": "nt-test", "entry_id": "entry", "field": "observation",
            "text": " ".join(f"Sentence {index}." for index in range(61))}
    fragments, omitted = observation_tags.fragments_for([note])
    assert len(fragments) == observation_tags.MAX_FRAGMENTS
    assert omitted["count"] == 1 and omitted["limit"] == observation_tags.MAX_FRAGMENTS
    assert omitted["fragment_ids"] and omitted["more"] == 0

    root = tmp_path / "tags"
    entry = tag_fx.click_entry(
        session=tag_fx.LAST_SESSION, hour=9, direction="up", confidence="high",
        observation="gap and go so far.", because="",
    )

    def legacy_bad_span(**_kwargs):
        return {"model": "local-test-medium", "summary": {"tags": [{
            "note_id": observation_tags.notes_for([entry])[0]["note_id"],
            "code": vocabulary["codes"][0], "span": [0, 3], "quote": "bad",
        }]}}

    outcome = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION, now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=root, entries=[entry], request=legacy_bad_span,
    )
    assert outcome["status"] == "degraded_no_narrative", outcome
    assert list(root.glob("*.json")) == []


def test_observation_v2_allows_two_codes_but_rejects_duplicate_or_mixed_transport(tmp_path):
    """Distinct codes share a fragment; duplicate and offset-mixed rows do not."""
    from ai_jobs import observation_tags

    entry = tag_fx.click_entry(
        session=tag_fx.LAST_SESSION, hour=9, direction="up", confidence="high",
        observation="gap and go so far.", because="",
    )
    vocabulary = observation_tags.load_vocabulary()

    def reply_for(kind):
        def request(**kwargs):
            fragment = kwargs["evidence"]["fragments"][0]
            row = {"fragment_id": fragment["fragment_id"], "code": vocabulary["codes"][0]}
            if kind == "valid":
                rows = [row, {**row, "code": vocabulary["codes"][1]}]
            elif kind == "duplicate":
                rows = [row, dict(row)]
            elif kind == "over_cap":
                rows = [dict(row) for _ in range(observation_tags.MAX_TAGS + 1)]
            else:
                rows = [{**row, "extra": "forbidden"}]
            return {"model": "local-test-medium", "summary": {"tags": rows}}
        return request

    good_root = tmp_path / "good"
    good = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION, now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=good_root, entries=[entry], request=reply_for("valid"),
    )
    assert good["status"] == "ok", good
    assert len(observation_tags.read_latest(tag_fx.LAST_SESSION, root=good_root)["tags"]) == 2

    for kind in ("duplicate", "over_cap", "extra"):
        root = tmp_path / kind
        prior = root / "verified.json"
        root.mkdir()
        prior.write_bytes(b'{"verified":"keep"}\n')
        before = prior.read_bytes()
        bad = observation_tags.run_observation_tags(
            session_date=tag_fx.LAST_SESSION, now=tag_fx.morning_after(tag_fx.LAST_SESSION),
            root=root, entries=[entry], request=reply_for(kind),
        )
        assert bad["status"] == "degraded_no_narrative", (kind, bad)
        assert prior.read_bytes() == before


def test_observation_v2_fragments_cover_all_punctuation_without_overlimit_quotes():
    """v2 never loses a leading mark and every offered quote stays bounded."""
    from ai_jobs import observation_tags

    for text in ("x" * 400 + " " + "y", "!!!hello.", "!" * 401):
        spans = observation_tags._fragment_spans(text)
        assert spans and spans[0][0] == 0 and spans[-1][1] == len(text), spans
        assert "".join(text[start:end] for start, end in spans) == text
        assert all(end - start <= observation_tags.MAX_FRAGMENT_LENGTH for start, end in spans)


def test_observation_v2_persists_honest_partial_coverage_for_one_long_note(tmp_path):
    """A cap is visible after a successful real publish, not just in the request."""
    from ai_jobs import observation_tags

    entry = tag_fx.click_entry(
        session=tag_fx.LAST_SESSION, hour=9, direction="up", confidence="high",
        observation=" ".join(f"Sentence {index}." for index in range(61)), because="",
    )
    vocabulary = observation_tags.load_vocabulary()

    def request(**kwargs):
        evidence = kwargs["evidence"]
        omitted = evidence["fragments_omitted"]
        assert len(evidence["fragments"]) == observation_tags.MAX_FRAGMENTS
        assert omitted["count"] == 1 and omitted["limit"] == observation_tags.MAX_FRAGMENTS
        assert omitted["notes_total"] == omitted["notes_represented"] == 1
        assert omitted["notes_partially_offered"] == 1
        return {
            "model": "local-test-medium",
            "summary": {"tags": [{
                "fragment_id": evidence["fragments"][0]["fragment_id"],
                "code": vocabulary["codes"][0],
            }]},
        }

    outcome = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION, now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=tmp_path / "tags", entries=[entry], request=request,
    )
    assert outcome["status"] == "ok", outcome
    saved = observation_tags.read_latest(tag_fx.LAST_SESSION, root=tmp_path / "tags")
    assert saved is not None
    assert saved["notes_offered"] == saved["notes_total"] == 1
    assert saved["fragments_offered"] == observation_tags.MAX_FRAGMENTS
    assert saved["fragments_total"] == observation_tags.MAX_FRAGMENTS + 1
    assert saved["fragments_omitted"] == saved["notes_partially_offered"] == 1
    assert saved["notes_represented"] == 1
    assert saved["fragment_limit"] == observation_tags.MAX_FRAGMENTS
    assert outcome["extra"]["fragments_omitted"] == 1
    assert "1 omitted at the named limit" in outcome["reason"]

    # The original review case: a whole note beyond the request is not counted
    # as represented, while the long-note run above proves partial coverage too.
    many_entries = []
    for index in range(observation_tags.MAX_FRAGMENTS + 1):
        many_entry = tag_fx.click_entry(
            session=tag_fx.LAST_SESSION, hour=9, direction="up", confidence="high",
            observation=f"Sentence {index}.", because="",
        )
        many_entry["entry_id"] = f"entry-{index}"
        many_entries.append(many_entry)

    def many_request(**kwargs):
        evidence = kwargs["evidence"]
        assert len(evidence["fragments"]) == observation_tags.MAX_FRAGMENTS
        return {
            "model": "local-test-medium",
            "summary": {"tags": [{
                "fragment_id": evidence["fragments"][0]["fragment_id"],
                "code": vocabulary["codes"][0],
            }]},
        }

    many_outcome = observation_tags.run_observation_tags(
        session_date=tag_fx.LAST_SESSION, now=tag_fx.morning_after(tag_fx.LAST_SESSION),
        root=tmp_path / "many-tags", entries=many_entries, request=many_request,
    )
    assert many_outcome["status"] == "ok", many_outcome
    many_saved = observation_tags.read_latest(tag_fx.LAST_SESSION, root=tmp_path / "many-tags")
    assert many_saved["notes_total"] == observation_tags.MAX_FRAGMENTS + 1
    assert many_saved["notes_offered"] == many_saved["notes_represented"] == observation_tags.MAX_FRAGMENTS
    assert many_saved["notes_partially_offered"] == 0


def test_observation_v2_evidence_has_no_result_fields():
    """The fragment transport carries words and codes, never an outcome."""
    from ai_jobs import observation_tags

    entry = tag_fx.click_entry(
        session=tag_fx.LAST_SESSION, hour=9, direction="up", confidence="high",
        observation="gap and go so far.", because="",
    )
    evidence = observation_tags.build_evidence(
        observation_tags.notes_for([entry]), vocabulary=observation_tags.load_vocabulary()
    )

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    forbidden = {"verdict", "grade", "price", "bar", "outcome", "result", "move"}
    assert not (keys(evidence) & forbidden)


def test_day_story_v2_request_schema_excludes_legacy_verdict_fields():
    """A v2 provider is offered explanations only; legacy replies validate later."""
    from ai_jobs import day_review_narration as narration

    schema = narration._narration_schema_for(day_fx.build())
    assert "were_you_right" not in schema["properties"]
    assert "were_you_right" not in schema["required"]
    assert "read_explanations" in schema["properties"]


def test_day_story_v2_rejects_ambiguous_input_and_names_render_cap(monkeypatch, tmp_path):
    """Ambiguous pairs refuse before a call; the cap reports the other read."""
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    first = day_fx.observing_and_predicting_entry(direction="up")
    second = day_fx.observing_and_predicting_entry(
        direction="down", stamp=day_fx.fx.STAMP + timedelta(hours=1)
    )
    entries = [first, second]
    reads, _ = day_fx.graded_reads(entries)
    pack = day_fx.build(
        entries=entries, reads=reads, congruence=day_fx.congruence(reads),
        story=day_fx.daily_story(entries),
    )
    monkeypatch.setattr(narration, "MAX_GRADED_CLAIMS", 1)
    evidence = narration._day_evidence(pack, tmp_path / "cap")
    assert list(evidence["read_explanations"]) == [pack["reads"][0]["source_id"]]
    assert evidence["read_explanations_not_offered"] == [{
        "read_source_id": pack["reads"][1]["source_id"], "reason": "render_cap"
    }]

    prediction = next(row for row in pack["trader_said"] if row["kind"] == "prediction")
    pack["trader_said"].append({**prediction, "source_id": prediction["source_id"] + "-two"})
    root = tmp_path / "ambiguous"
    day_review_pack.write_pack(pack, root=root)
    prior = narration.narration_path(day_fx.SESSION, root=root)
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(b'{"verified":"keep"}\n')
    before = prior.read_bytes()
    calls: list[dict] = []
    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=lambda **kwargs: calls.append(kwargs), only_this_session=True,
    )
    assert outcome["status"] == "degraded_no_narrative", outcome
    assert calls == []
    assert prior.read_bytes() == before
