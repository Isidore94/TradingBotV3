"""Two nightly AI failures seen in the live ledger, 2026-09-17..23.

B. `journal_enrichment`: every rejected reply on disk (7 of 7) was the
   enrichment JSON Schema itself. Ollama 0.32 answers HTTP 400 "failed to parse
   grammar" for the schema's `maxLength: 2000`, the provider fell back to plain
   JSON mode, and the prompt said "return exactly this JSON object" above the
   schema - so gemma3:12b returned the schema.
A. `day_review_narration`: the 09-22 packs made a ~100 KB prompt (~50k tokens)
   and the medium model timed out at 540 s. The 09-21 "graded ... twice" error
   came from the v1 transport that c5e7feb8 replaced with code-owned links.

No model is called here; every transport is faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tj4_support as day_fx  # noqa: E402

ENDPOINT = "http://127.0.0.1:11434/v1"


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _chat(text: str) -> _Response:
    return _Response({"choices": [{"message": {"role": "assistant", "content": text}}]})


def _grammar_400() -> _Response:
    # The exact body Ollama 0.32 returned for the enrichment schema (probe 2026-09-23).
    return _Response(
        {"error": {"message": '{"error":{"code":400,"message":"Failed to initialize '
                              'samplers: failed to parse grammar"}}'}},
        status_code=400,
    )


ANSWER = {
    "summary": "Bought the reclaim and sold into the fade.",
    "tags": [],
    "confidence": "medium",
    "sources": ["journal.trade"],
    "unknowns": [],
}


def _enrich(post, *, schema=None):
    import ai_summary
    from ai_jobs import enrichment

    with mock.patch.object(
        ai_summary, "get_local_setting",
        lambda key, default=None: {"ai_local_endpoint_url": ENDPOINT}.get(key, default),
    ), mock.patch.object(ai_summary, "record_rejected_reply", lambda **kwargs: None):
        return ai_summary.request_ai_summary(
            provider="local", model="gemma3:12b", api_key="",
            evidence={"package_id": "p", "evidence_hash": "h"},
            schema=schema or enrichment.ENRICHMENT_JSON_SCHEMA,
            schema_name="tradingbot_trade_enrichment",
            prompt_version=enrichment.ENRICHMENT_PROMPT_VERSION,
            post=post,
        )


# ---------------------------------------------------------------------------
# B. journal_enrichment
# ---------------------------------------------------------------------------
def test_a_grammar_400_retries_the_same_contract_without_bounds_before_json_mode():
    """Root cause B: keep the grammar on, so the shape is enforced by the decoder."""
    sent: list[dict] = []
    replies = [_grammar_400(), _chat(json.dumps(ANSWER))]

    def post(url, **kwargs):
        sent.append(json.loads(json.dumps(kwargs["json"])))
        return replies[len(sent) - 1]

    result = _enrich(post)

    assert result["summary"] == ANSWER
    assert len(sent) == 2
    retry = sent[1]["response_format"]
    assert retry["type"] == "json_schema"
    schema = retry["json_schema"]["schema"]
    assert "maxLength" not in json.dumps(schema)
    assert schema["required"] == ["summary", "tags", "confidence", "sources", "unknowns"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


def test_the_unbounded_grammar_still_validates_against_the_full_contract():
    """A bound left out of the grammar is still enforced after the answer."""
    sent: list[dict] = []
    long_answer = {**ANSWER, "summary": "x" * 2001}
    replies = [_grammar_400(), _chat(json.dumps(long_answer)), _chat(json.dumps(long_answer))]

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        return replies[len(sent) - 1]

    with pytest.raises(RuntimeError, match="2000"):
        _enrich(post)


def test_grammar_steps_do_not_spend_the_validation_retry():
    """Two grammar 400s, then JSON mode still gets its one corrected retry."""
    sent: list[dict] = []
    replies = [
        _grammar_400(),
        _grammar_400(),
        _chat(json.dumps({"summary": "only"})),
        _chat(json.dumps(ANSWER)),
    ]

    def post(url, **kwargs):
        sent.append(json.loads(json.dumps(kwargs["json"])))
        return replies[len(sent) - 1]

    result = _enrich(post)

    assert result["summary"] == ANSWER
    assert [row["response_format"]["type"] for row in sent] == [
        "json_schema", "json_schema", "json_object", "json_object",
    ]
    assert "missing required field" in sent[3]["messages"][1]["content"]


def test_an_echoed_schema_is_named_as_an_echo_and_the_retry_says_so():
    """The live reply shape: rejected cleanly, and the model is told what it did."""
    from ai_jobs import enrichment

    sent: list[dict] = []
    echo = json.dumps(enrichment.ENRICHMENT_JSON_SCHEMA)
    replies = [_chat(echo), _chat(json.dumps(ANSWER))]

    def post(url, **kwargs):
        sent.append(json.loads(json.dumps(kwargs["json"])))
        return replies[len(sent) - 1]

    result = _enrich(post)

    assert result["summary"] == ANSWER
    assert "was the JSON Schema itself, not an answer" in sent[1]["messages"][1]["content"]


def test_two_echoes_degrade_with_a_plain_reason():
    from ai_jobs import enrichment

    echo = json.dumps(enrichment.ENRICHMENT_JSON_SCHEMA)
    with pytest.raises(RuntimeError, match="JSON Schema itself"):
        _enrich(lambda url, **kwargs: _chat(echo))


def test_the_schema_prompt_never_asks_for_the_schema_verbatim():
    """The sentence that invited the echo is gone; the schema is labelled a description."""
    import ai_summary
    from ai_jobs import enrichment

    prompt = ai_summary._local_schema_prompt(
        {"package_id": "p"}, enrichment.ENRICHMENT_JSON_SCHEMA
    )
    assert "return exactly this JSON object" not in prompt
    assert "never return the schema itself" in prompt


# ---------------------------------------------------------------------------
# A. day_review_narration
# ---------------------------------------------------------------------------
def _live_sized_pack():
    """The fixture pack inflated to the 2026-09-22 section sizes (~100 KB)."""
    pack = day_fx.build()
    table = [["SYM%d" % i, 0.1234567, "up", "above", 1.2345678, 0.55, "above", "below"]
             for i in range(50)]
    pack["internals"] = [
        {
            "at": f"2026-09-22T{9 + i:02d}:00:00-04:00",
            "kind": "mentor",
            "source_id": f"internals:mentor:{i}",
            "context": {
                "captured_at": "x", "columns": ["symbol"] * 8, "rows": table,
                "common": {
                    "availability": "available", "reason": "",
                    "internals": "Breadth +0.04% - Fear VXX down / SPY down",
                    "derived": {f"d{j}": {"inputs": ["XLB"] * 11, "value": 0.1}
                                for j in range(8)},
                },
            },
        }
        for i in range(9)
    ]
    pack["skill"] = {
        "source_id": "skill:session",
        "session": {"n": 3},
        "lately": {"cells": [{"high": 0.21245740530, "low": 0.059934981, "n": 331,
                              "population": "liked_or_claimed", "side": "LONG"}] * 200},
    }
    pack["trades"] = {
        "n": 2, "wins": 1, "losses": 1, "net_pnl": 3,
        "rows": [
            {"source_id": f"trade:t{i}", "symbol": "DRAM", "account_number": "29347316",
             "anchor_execution_uid": "QT:29347316:1", "fx_rate": 1.4, "notes": "held it",
             "note_lane_json": "{}" * 400,
             "trade_review": {"entry_raw": {"text": "saw the reclaim"}, "exit_raw": {"text": ""},
                              "entry_answers": {}, "noise": "n" * 1500}}
            for i in range(2)
        ],
    }
    pack["report_card"] = {"session": "2026-09-22", "lines": [
        {"source_id": f"report_card:k{i}", "key": f"k{i}", "text": "a line",
         "counts": ["c" * 900]} for i in range(6)
    ]}
    forecast = dict(pack.get("forecast") or {})
    forecast["text"] = "forecast words " * 800
    pack["forecast"] = forecast
    return pack


def test_the_day_evidence_fits_the_medium_model_and_keeps_every_citable_id(tmp_path):
    """Root cause A: a live-sized pack must fit well inside the 540 s call."""
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    pack = _live_sized_pack()
    assert len(json.dumps(pack, default=str)) > 90_000, "fixture drift: pack too small"

    evidence = narration._day_evidence(pack, tmp_path)
    text = json.dumps(evidence, sort_keys=True, default=str)

    assert len(text) <= narration.MAX_DAY_EVIDENCE_CHARS
    for source_id in day_review_pack.allowed_source_ids(pack):
        assert json.dumps(source_id) in text, f"{source_id} could not be cited"
    assert "29347316" not in text, "an account number reached the model"
    assert "lately" not in json.dumps(evidence["pack"]["skill"])
    assert evidence["pack"]["internals"][0]["internals"].startswith("Breadth")
    assert "rows" not in evidence["pack"]["internals"][0]


def test_an_oversized_day_evidence_is_refused_before_the_model_loads(tmp_path, monkeypatch):
    """A prompt that cannot finish is not sent; the prior story stays byte-identical."""
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    day_review_pack.write_pack(day_fx.build(), root=root)
    prior = narration.narration_path(day_fx.SESSION, root=root)
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(b'{"verified":"keep"}\n')
    monkeypatch.setattr(narration, "MAX_DAY_EVIDENCE_CHARS", 500)
    calls: list[dict] = []

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=lambda **kwargs: calls.append(kwargs), only_this_session=True,
    )

    assert calls == []
    assert outcome["status"] == "degraded_no_narrative"
    assert "the medium model reads at most 500" in outcome["reason"]
    assert prior.read_bytes() == b'{"verified":"keep"}\n'


def test_a_repeated_read_id_in_a_v2_reply_cannot_grade_a_read_twice(tmp_path):
    """The 09-21 error shape: a model repeating a read id yields one claim per read."""
    import ai_summary
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    day_review_pack.write_pack(day_fx.build(), root=root)

    def post(url, **kwargs):
        prompt = kwargs["json"]["messages"][1]["content"]
        evidence = json.loads(prompt.split("EVIDENCE PACKAGE:\n", 1)[1].split("\n\n", 1)[0])
        read_id = next(iter(evidence["read_explanations"]))
        others = {key: "Explained." for key in evidence["read_explanations"]}
        body = json.dumps({
            "headline": "A day.", "what_happened": "It moved.", "what_you_thought": "Up.",
            "read_explanations": others, "chased_against_news":
                {"verdict": "unknown", "evidence_id": ""},
            "process": "Fine.", "sources": evidence["allowed_source_ids"][:1],
        })
        # The raw text repeats the read id as a key, as a looping model would.
        repeated = body.replace(
            '"read_explanations": {', '"read_explanations": {"%s": "Again.", ' % read_id, 1
        )
        return _chat(repeated)

    with mock.patch.object(
        ai_summary, "get_local_setting",
        lambda key, default=None: {"ai_local_endpoint_url": ENDPOINT}.get(key, default),
    ):
        outcome = narration.run_day_review_narration(
            session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
            request=lambda **kwargs: ai_summary.request_ai_summary(post=post, **kwargs),
            only_this_session=True,
        )

    assert "graded" not in outcome["reason"] or "twice" not in outcome["reason"], outcome
    stored = narration.read_narration(day_fx.SESSION, root=root)
    assert stored is not None, outcome
    ids = [claim["evidence_id"] for claim in stored["narration"]["were_you_right"]]
    assert len(ids) == len(set(ids))
