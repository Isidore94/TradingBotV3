"""TJ-13A item 5, second half: a rejected model reply is kept.

Added by the BUILDER on the lead's decision in the packet: *"ALSO persist every
rejected raw reply (one bounded file per rejection under the AI store's logs,
local first, never raising into the slot, add your own test) so the next night
shows the true shape."*

Why it is owed. ``journal_enrichment`` failed on three consecutive nights
(2026-09-15/16/17) with the identical sentence, and the raw reply behind it was
held in a local variable and folded into ``str(last_error)``. Nothing under
``ai_store/`` carried it - the tester searched ``logs/`` and ``retros/`` and
found no ``*enrich*`` file anywhere in the store - so the fixture that proves
the repair had to be HAND-BUILT from the error text. A defect that repeats
nightly should cost one file, not an archaeology pass.

The binding constraint is the other way round, though: **an evidence store is
never allowed to cost the thing it records.** This file is about a failure that
is already being reported properly through the ledger, so every way of failing
to write it returns ``None`` and the slot never sees it.

NO MODEL IS CALLED HERE.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402
from ai_jobs import enrichment  # noqa: E402

NOW = datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)
SCHEMA_NAME = "tradingbot_trade_enrichment"


def test_a_rejected_reply_is_written_whole_with_its_error(tmp_path):
    reply = json.dumps(enrichment.ENRICHMENT_JSON_SCHEMA)

    path = ai_summary.record_rejected_reply(
        schema_name=SCHEMA_NAME,
        text=reply,
        error=(
            "tradingbot_trade_enrichment is missing required field(s): "
            "confidence, sources, summary, tags, unknowns"
        ),
        logs_dir=tmp_path,
        now=NOW,
    )

    assert path is not None and path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_name"] == SCHEMA_NAME
    assert payload["reply"] == reply
    assert payload["reply_truncated"] is False
    assert "missing required field(s)" in payload["error"]
    # No half-written file is left beside it.
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]


def test_two_rejections_in_one_night_are_two_files(tmp_path):
    """A retry rejects a second reply; the first must still be readable."""
    first = ai_summary.record_rejected_reply(
        schema_name=SCHEMA_NAME, text='{"a": 1}', error="one", logs_dir=tmp_path
    )
    second = ai_summary.record_rejected_reply(
        schema_name=SCHEMA_NAME, text='{"b": 2}', error="two", logs_dir=tmp_path
    )

    assert first != second
    assert len(sorted(tmp_path.glob("*.json"))) == 2


def test_a_huge_reply_is_bounded_and_says_that_it_was_cut(tmp_path):
    reply = "x" * (ai_summary.MAX_REJECTED_REPLY_CHARS + 5_000)

    path = ai_summary.record_rejected_reply(
        schema_name=SCHEMA_NAME, text=reply, error="too long", logs_dir=tmp_path
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reply_chars"] == len(reply)
    assert payload["reply_kept_chars"] == ai_summary.MAX_REJECTED_REPLY_CHARS
    assert payload["reply_truncated"] is True


def test_an_unwritable_store_costs_nothing(tmp_path):
    """The rule that outranks the feature: a failed append loses the event.

    No AI store configured is the ordinary case on a fresh machine, and it must
    not turn a reported model failure into an unhandled exception inside the
    slot.
    """
    from ai_jobs import store

    with mock.patch.object(
        store, "store_logs_dir", side_effect=ValueError("No AI store configured")
    ):
        assert (
            ai_summary.record_rejected_reply(
                schema_name=SCHEMA_NAME, text='{"a": 1}', error="boom"
            )
            is None
        )


def test_the_provider_path_persists_what_it_rejected(tmp_path, monkeypatch):
    """End to end through the real provider function, transport faked.

    The echoed-schema reply stays rejected (that is the other TJ-13A test); the
    point here is that the night now leaves the evidence of WHY behind.
    """
    written: list[dict] = []
    real_record = ai_summary.record_rejected_reply

    def _record(**kwargs):
        written.append(kwargs)
        return real_record(logs_dir=tmp_path, **kwargs)

    monkeypatch.setattr(ai_summary, "record_rejected_reply", _record)

    class _Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self.payload

    reply = json.dumps(enrichment.ENRICHMENT_JSON_SCHEMA)

    def _post(url, **kwargs):
        return _Response(
            {"choices": [{"message": {"role": "assistant", "content": reply}}]}
        )

    monkeypatch.setattr(
        ai_summary,
        "get_local_setting",
        lambda key, default=None: {
            "ai_local_endpoint_url": "http://127.0.0.1:11434/v1",
            "ai_local_model_medium": "gemma3:12b",
        }.get(key, default),
    )

    try:
        ai_summary.request_ai_summary(
            provider="local",
            model="gemma3:12b",
            api_key="",
            evidence={
                "schema_version": "ai_evidence_package_v2",
                "session_date": "2026-09-17",
                "sources": [
                    {
                        "source_id": "journal.trade",
                        "label": "the trade",
                        "status": "available",
                        "content": {"trade_id": "T-1"},
                    }
                ],
            },
            timeout_seconds=60,
            post=_post,
            schema=enrichment.ENRICHMENT_JSON_SCHEMA,
            schema_name=SCHEMA_NAME,
            prompt_version=enrichment.ENRICHMENT_PROMPT_VERSION,
        )
    except RuntimeError as exc:
        assert "invalid summary JSON" in str(exc)
    else:  # pragma: no cover - the echoed schema must not validate
        raise AssertionError("an echoed schema must stay rejected")

    assert written, "the raw reply the night threw away is now on disk"
    assert all(row["text"] == reply for row in written)
    saved = sorted(tmp_path.glob("*.json"))
    assert saved
    assert json.loads(saved[0].read_text(encoding="utf-8"))["reply"] == reply
