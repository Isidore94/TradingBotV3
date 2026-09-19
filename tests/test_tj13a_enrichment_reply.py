"""TJ-13A item 5: the ``journal_enrichment`` schema failure.

**Reproduced from the live ledger** (three attempts plus the terminal cap row on
2026-09-17, `\\\\MINI-PC\\Trading Bot Data\\ai_store\\logs\\ai_job_ledger.jsonl`,
all three against the same trade ``d22101759e8f30c0163e35ee``)::

    enriched 1, abstained 0, failed 1 of 2 trade(s) for 2026-09-17;
    1 failure(s): d22101759e8f30c0163e35ee: RuntimeError: local provider
    returned invalid summary JSON after 2 attempt(s):
    tradingbot_trade_enrichment is missing required field(s): confidence,
    sources, summary, tags, unknowns

**THE RAW MODEL REPLY WAS NOT PERSISTED.** ``ai_summary._request_local_summary``
holds it in ``text`` and folds only ``str(last_error)`` into the ``RuntimeError``
it raises; nothing under ``ai_store/`` carries it (searched
``ai_store/logs/`` - the ledger only - and ``ai_store/retros/``; no ``*enrich*``
file exists anywhere in the store). So the two replies below are **HAND-BUILT
from the ledger's error text and from the prompt the code actually sends**, and
they are the two shapes that produce that error verbatim:

* ALL FIVE required fields missing means the parsed value WAS a JSON object and
  carried none of them at the top level. The wrapper shape is the first
  candidate: the payload sets ``response_format.json_schema.name =
  "tradingbot_trade_enrichment"`` and a local backend that echoes the schema
  name as the envelope key produces exactly that. Its CONTENT is a complete,
  honest answer, so it must validate.
* The second candidate is the prompt's own doing:
  ``ai_summary._local_schema_prompt`` writes *"REQUIRED OUTPUT SHAPE - return
  exactly this JSON object:"* followed by ``json.dumps(schema)``. A model that
  obeys that sentence literally returns the SCHEMA. It carries no answer, so it
  must stay rejected - and the prior row must stay where it is.

NO MODEL IS CALLED HERE: every request is a literal.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import enrichment  # noqa: E402
from tests.test_ws_ai1_enrichment_status import (  # noqa: E402
    SESSION as TRADE_SESSION,
    _gate_is_met,
    _journal_with_one_closed_trade,
)

SESSION = "2026-09-17"
NOW = datetime(2026, 9, 18, 10, 38, tzinfo=timezone.utc)
ENDPOINT = "http://127.0.0.1:11434/v1"

#: The exact sentence the ledger carried, three nights running.
LEDGER_ERROR = (
    "tradingbot_trade_enrichment is missing required field(s): "
    "confidence, sources, summary, tags, unknowns"
)

#: The answer the model gave, as far as the ledger can tell: complete, and
#: wrapped in the schema name the request payload named.
ANSWER = {
    "summary": (
        "Bought the earnings-anchored reclaim at 09:41 and closed it into the "
        "12:04 fade; the level held on the retest."
    ),
    "tags": [],  # filled per test from the real vocabulary
    "confidence": "medium",
    "sources": ["journal.trade"],
    "unknowns": [],
}


def _settings(**values):
    import ai_summary

    return mock.patch.object(
        ai_summary,
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _replies(text):
    """A chat-completions body carrying ``text``, with no ``usage`` block.

    Some llama.cpp builds omit ``usage``; omitting it here keeps the prompt
    truncation tripwire out of the way of what this file is measuring.
    """
    calls = {"n": 0}

    def _post(url, **kwargs):
        calls["n"] += 1
        return _Response(
            {
                "id": "chatcmpl-local",
                "choices": [{"message": {"role": "assistant", "content": text}}],
            }
        )

    return _post, calls


def _evidence():
    """The package shape the enrichment pass sends, in miniature."""
    return {
        "schema_version": "ai_evidence_package_v2",
        "session_date": SESSION,
        "package_id": "pkg-tj13a",
        "evidence_hash": "hash-tj13a",
        "sources": [
            {
                "source_id": "journal.trade",
                "label": "the trade",
                "status": "available",
                "content": {"trade_id": "d22101759e8f30c0163e35ee", "symbol": "AAPL"},
            }
        ],
    }


def _request(text):
    import ai_summary

    post, calls = _replies(text)
    with _settings(ai_local_endpoint_url=ENDPOINT, ai_local_model_medium="gemma3:12b"):
        result = ai_summary.request_ai_summary(
            provider="local",
            model="gemma3:12b-tbv3ctx-64k",
            api_key="",
            evidence=_evidence(),
            timeout_seconds=900,
            post=post,
            schema=enrichment.ENRICHMENT_JSON_SCHEMA,
            schema_name="tradingbot_trade_enrichment",
            prompt_version=enrichment.ENRICHMENT_PROMPT_VERSION,
        )
    return result, calls


def test_the_recorded_enrichment_reply_validates():
    """HAND-BUILT fixture (the raw reply was not persisted - see the module
    docstring). A complete answer wrapped in the schema name the request itself
    supplied must not be thrown away three nights running."""
    known = enrichment.setup_vocabulary()[0]
    answer = {**ANSWER, "tags": [known]}
    reply = json.dumps({"tradingbot_trade_enrichment": answer})

    result, calls = _request(reply)

    summary = result["summary"]
    assert enrichment._summary_text(summary).startswith("Bought the earnings-anchored")
    assert enrichment._proposed_tags(summary) == [known]
    assert enrichment._confidence_text(summary) == "medium"
    assert calls["n"] == 1, "a reply this path can read needs no retry"


def test_the_recorded_reply_reaches_the_enrichment_seam_whole(monkeypatch, tmp_path):
    """The slot itself, not only the provider function.

    The enrichment pass is what writes the row, so the repair has to be visible
    from ``run_journal_enrichment``: one trade, one reply, one ``enriched`` row.
    """
    import ai_summary

    _gate_is_met(monkeypatch)
    store = _journal_with_one_closed_trade(tmp_path / "j.sqlite3")
    known = enrichment.setup_vocabulary()[0]
    answer = {**ANSWER, "tags": [known]}
    post, _calls = _replies(json.dumps({"tradingbot_trade_enrichment": answer}))

    # The REAL provider function, with only the transport faked.
    real_request = ai_summary.request_ai_summary
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "gemma3:12b")
    monkeypatch.setattr(
        ai_summary,
        "get_local_setting",
        lambda key, default=None: {
            "ai_local_endpoint_url": ENDPOINT,
            "ai_local_model_medium": "gemma3:12b",
        }.get(key, default),
    )
    monkeypatch.setattr(
        ai_summary,
        "request_ai_summary",
        lambda **kwargs: real_request(post=post, **kwargs),
    )

    result = enrichment.run_journal_enrichment(
        session_date=TRADE_SESSION, now=NOW, store=store, review_rows=[],
    )

    rows = store.list_ai_enrichment("T-AI1")
    assert len(rows) == 1
    assert rows[0]["status"] == "enriched"
    assert "enriched 1, abstained 0, failed 0 of 1" in result["reason"]
    assert result["status"] == "ok"


def test_a_reply_that_echoes_the_schema_is_still_rejected():
    """The other hand-built candidate, and it must NOT be made to validate.

    ``_local_schema_prompt`` tells the model to *"return exactly this JSON
    object"* and then prints the schema. A model that complies returns the
    contract, not an answer - and an answer-shaped hole must be a failure, never
    a row.
    """
    reply = json.dumps(enrichment.ENRICHMENT_JSON_SCHEMA)

    with pytest.raises(RuntimeError, match="invalid summary JSON"):
        _request(reply)


def test_an_invalid_reply_leaves_the_prior_enrichment_row_intact(tmp_path, monkeypatch):
    """Gate #144's contract stays: a bad night never edits a good row.

    The trade already has an enrichment from an earlier session. Tonight's reply
    is unreadable; the store must still hold that row, byte for byte, with the
    failure appended beside it.
    """
    import ai_summary

    _gate_is_met(monkeypatch)
    store = _journal_with_one_closed_trade(tmp_path / "j.sqlite3")
    store.save_ai_enrichment(
        trade_id="T-AI1",
        session_date="2026-09-10",
        summary="An earlier night's good answer.",
        tags=[],
        evidence=[],
        model="gemma3:12b",
        now="2026-09-11T02:00:00+00:00",
        status="enriched",
    )
    before = [dict(row) for row in store.list_ai_enrichment("T-AI1")]
    assert len(before) == 1

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "gemma3:12b")

    def _rejects(**kwargs):
        raise RuntimeError(
            "local provider returned invalid summary JSON after 2 attempt(s): "
            + LEDGER_ERROR
        )

    monkeypatch.setattr(ai_summary, "request_ai_summary", _rejects)

    result = enrichment.run_journal_enrichment(
        session_date=TRADE_SESSION, now=NOW, store=store, review_rows=[],
    )

    rows = store.list_ai_enrichment("T-AI1")
    kept = [row for row in rows if row["enrichment_id"] == before[0]["enrichment_id"]]
    assert kept and dict(kept[0]) == before[0], "the prior row is untouched"
    assert result["status"] != "ok"
    assert LEDGER_ERROR in result["reason"]
