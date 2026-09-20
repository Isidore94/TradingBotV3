"""TJ-13B item 2: a `local_large` provider that falls back and SAYS so.

`plan.md` §12.4 TJ-13 item 7 / decision 0021 answer 20: *"the week story is
written by the large LOCAL model … `ai_week_review_provider` defaults to
`local_large`, falls back to local medium and says so in the ledger; `openai`
remains a setting, off."*

The week slot itself is TJ-5's. What TJ-13B owes it is this seam: one request
path that asks the 27B, falls back to the 12B when the 27B cannot answer, and
returns an attribution saying WHICH MODEL WAS ASKED, WHICH ANSWERED and WHY -
because on the live ledger the `model` column is a single string, and a row
that simply says `gemma3:12b-tbv3ctx-64k` cannot be told apart from a night
where the large model was never attempted.

**NO MODEL IS LOADED HERE.** Every endpoint is a fake `post`.

Contract pinned by these tests, for the builder::

    ai_jobs.provider.LOCAL_LARGE == "local_large"
    ai_jobs.provider.WEEK_REVIEW_PROVIDER_SETTING == "ai_week_review_provider"
    ai_jobs.provider.LEDGER_FIELD == "model_attribution"
    ai_jobs.provider.week_review_provider() -> str        # default "local_large"
    ai_jobs.provider.request_with_fallback(
        *, provider="local_large", evidence, schema=None,
        schema_name="tradingbot_ai_summary", post=<requests.post>,
        timeout_seconds=900, ledger_path=None) -> dict
            {"status": <ai_jobs.ledger status>,
             "result": <ai_summary result envelope | None>,
             "attribution": {"provider", "model_asked", "model_answered",
                             "fallback_reason"}}
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from zoneinfo import ZoneInfo  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
SATURDAY_NIGHT = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)
WEEKEND_SESSION = "2026-09-18"

ENDPOINT = "http://127.0.0.1:11434/v1"
LARGE_TAG = "hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M"
MEDIUM_TAG = "gemma3:12b-tbv3ctx-64k"
LIVE_START = "01:00"
LIVE_END = "09:00"


def _settings(**values):
    import ai_summary
    import project_paths

    def _get(key, default=None):
        return values.get(key, default)

    class _Both:
        def __enter__(self):
            self._a = mock.patch.object(ai_summary, "get_local_setting", _get)
            self._b = mock.patch.object(project_paths, "get_local_setting", _get)
            self._a.start()
            self._b.start()
            return self

        def __exit__(self, *exc):
            self._b.stop()
            self._a.stop()
            return False

    return _Both()


def _live_settings(**extra):
    values = {
        "ai_local_endpoint_url": ENDPOINT,
        "ai_local_model_large": LARGE_TAG,
        "ai_local_model_medium": MEDIUM_TAG,
        "ai_local_context_tokens": 65536,
        "ai_offhours_start": LIVE_START,
        "ai_offhours_end": LIVE_END,
    }
    values.update(extra)
    return _settings(**values)


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _evidence(tmp_path: Path):
    import ai_summary

    overrides = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\n", encoding="utf-8")
        overrides[source_id] = path
    return ai_summary.build_evidence_package(["daily_report"], source_overrides=overrides)


def _reply_text(extra_rows=()):
    import ai_summary

    sections = {name: [] for name in ai_summary.MODEL_SUMMARY_SECTIONS}
    sections["what_is_working"] = [
        {
            "statement": "The week held its levels on four of five sessions.",
            "evidence_refs": ["daily.auto_report"],
            "confidence": "medium",
        },
        *extra_rows,
    ]
    return json.dumps({"executive_summary": "One measured week.", **sections})


def _body(text):
    return _Response(
        {
            "id": "chatcmpl-week",
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12_000, "completion_tokens": 400, "total_tokens": 12_400},
        }
    )


def _router(calls, *, refuse=(), text=None):
    """A fake endpoint that answers per MODEL TAG.

    `refuse` is the set of tags whose request raises the way a server that is
    not listening does.
    """

    def _post(url, **kwargs):
        model = kwargs["json"]["model"]
        calls.append((model, kwargs))
        if model in refuse:
            raise ConnectionError("[WinError 10061] No connection could be made")
        return _body(text or _reply_text())

    return _post


# ---------------------------------------------------------------------------
# which provider the week story asks for
# ---------------------------------------------------------------------------


def test_the_week_review_provider_defaults_to_the_large_local_model():
    """Decision 0021 answer 20, superseding TJ-5's OpenAI default."""
    from ai_jobs import provider

    assert provider.LOCAL_LARGE == "local_large"
    with _live_settings():
        assert provider.week_review_provider() == "local_large"


def test_openai_is_still_a_setting_and_it_is_off_until_it_is_typed():
    """"`openai` remains a setting, off." Both halves are the test: an
    unconfigured desk never names it, and a desk that sets it gets it."""
    from ai_jobs import provider

    assert provider.WEEK_REVIEW_PROVIDER_SETTING == "ai_week_review_provider"
    with _live_settings():
        assert provider.week_review_provider() != "openai"
    with _live_settings(ai_week_review_provider="openai"):
        assert provider.week_review_provider() == "openai"


# ---------------------------------------------------------------------------
# the large model answers, or the medium one does and the row says why
# ---------------------------------------------------------------------------


def test_the_large_model_answers_and_the_attribution_names_it(tmp_path):
    from ai_jobs import ledger, provider

    calls: list[tuple[str, dict]] = []
    with _live_settings():
        evidence = _evidence(tmp_path)
        outcome = provider.request_with_fallback(
            provider="local_large",
            evidence=evidence,
            post=_router(calls),
        )

    assert [model for model, _kwargs in calls] == [LARGE_TAG]
    assert outcome["status"] == ledger.STATUS_OK
    attribution = outcome["attribution"]
    assert attribution["model_asked"] == LARGE_TAG
    assert attribution["model_answered"] == LARGE_TAG
    assert attribution["fallback_reason"] == ""
    assert outcome["result"]["summary"]["executive_summary"]


def test_when_the_large_model_cannot_answer_the_medium_one_does_and_says_why(tmp_path):
    """The 27B has never run on this box. The first night it cannot load, the
    week story must still be written - and the record must not read as though
    the large model had been asked for and had answered."""
    import ai_summary
    from ai_jobs import ledger, provider

    calls: list[tuple[str, dict]] = []
    with _live_settings():
        evidence = _evidence(tmp_path)
        outcome = provider.request_with_fallback(
            provider="local_large",
            evidence=evidence,
            post=_router(calls, refuse={LARGE_TAG}),
        )

    assert [model for model, _kwargs in calls] == [LARGE_TAG, MEDIUM_TAG]
    assert outcome["status"] == ledger.STATUS_OK
    attribution = outcome["attribution"]
    assert attribution["model_asked"] == LARGE_TAG
    assert attribution["model_answered"] == MEDIUM_TAG
    assert attribution["fallback_reason"], "a fallback with no reason is a silent downgrade"
    assert ai_summary.is_endpoint_unreachable(
        ai_summary.LocalEndpointUnreachable(attribution["fallback_reason"])
    )


def test_the_fallback_sends_the_medium_model_the_same_closed_contract(tmp_path):
    """One contract, two models. A second shape would be a second place for the
    schema, the timeout and the citation rule to drift."""
    from ai_jobs import provider

    calls: list[tuple[str, dict]] = []
    with _live_settings():
        evidence = _evidence(tmp_path)
        provider.request_with_fallback(
            provider="local_large",
            evidence=evidence,
            post=_router(calls, refuse={LARGE_TAG}),
        )

    large_schema = calls[0][1]["json"]["response_format"]["json_schema"]["schema"]
    medium_schema = calls[1][1]["json"]["response_format"]["json_schema"]["schema"]
    assert large_schema == medium_schema
    assert large_schema.get("additionalProperties") is False
    assert '"maxLength": 2000' not in json.dumps(large_schema)


def test_both_models_fail_and_nothing_is_published(tmp_path):
    """A failure leaves the last verified file alone; it never publishes a
    half-answer under whichever model happened to be reachable."""
    from ai_jobs import ledger, provider

    calls: list[tuple[str, dict]] = []
    with _live_settings():
        evidence = _evidence(tmp_path)
        outcome = provider.request_with_fallback(
            provider="local_large",
            evidence=evidence,
            post=_router(calls, refuse={LARGE_TAG, MEDIUM_TAG}),
        )

    assert outcome["status"] == ledger.STATUS_FAILED
    assert outcome["result"] is None
    assert outcome["attribution"]["model_asked"] == LARGE_TAG
    assert outcome["attribution"]["model_answered"] == ""
    assert outcome["attribution"]["fallback_reason"]


def test_a_reply_that_cites_a_source_the_evidence_never_held_is_struck_out(tmp_path):
    """Ground rule 12.3: text fields cite ALLOWED `source_id`s. An invented one
    never reaches the document, whichever model wrote it."""
    from ai_jobs import provider

    invented = {
        "statement": "A source nobody packaged says the week was excellent.",
        "evidence_refs": ["daily.invented_source"],
        "confidence": "high",
    }
    calls: list[tuple[str, dict]] = []
    with _live_settings():
        evidence = _evidence(tmp_path)
        outcome = provider.request_with_fallback(
            provider="local_large",
            evidence=evidence,
            post=_router(calls, text=_reply_text(extra_rows=(invented,))),
        )

    published = json.dumps(outcome["result"]["summary"])
    assert "daily.invented_source" not in published
    assert "A source nobody packaged" not in published


# ---------------------------------------------------------------------------
# the ledger row is where the trader reads it
# ---------------------------------------------------------------------------


def test_a_slot_row_carries_which_model_was_asked_which_answered_and_why(tmp_path):
    """Driven through the REAL runner: the attribution has to survive the
    `extra` seam onto the row, because the ledger is the only place a reader
    can later tell a fallback night from a large-model night."""
    from ai_jobs import ledger, provider, runner, store, window

    led = tmp_path / "ledger.jsonl"
    calls: list[tuple[str, dict]] = []

    with _live_settings():
        evidence = _evidence(tmp_path)
        outcome = provider.request_with_fallback(
            provider="local_large",
            evidence=evidence,
            post=_router(calls, refuse={LARGE_TAG}),
        )

    attribution = outcome["attribution"]

    def _run(**kwargs):
        return {
            "status": ledger.STATUS_OK,
            "reason": "week story published",
            "model": attribution["model_answered"],
            "extra": {provider.LEDGER_FIELD: attribution},
        }

    slate = [
        replace(slot, run=_run)
        for slot in runner.default_slots()
        if slot.name == "market_story_narration"
    ]
    assert slate, "fixture drift: the narration slot is the stand-in for TJ-5's week slot"

    real_now = window.market_now
    with mock.patch.object(window, "market_now", lambda now=None: real_now(SATURDAY_NIGHT)):
        with mock.patch.object(store, "store_available", return_value=(True, "ready")):
            with _live_settings():
                runner.run_slots(slate, now=SATURDAY_NIGHT, ledger_path=led)

    row = [
        json.loads(line)
        for line in led.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][-1]
    assert row["model"] == MEDIUM_TAG
    recorded = row[provider.LEDGER_FIELD]
    assert recorded["model_asked"] == LARGE_TAG
    assert recorded["model_answered"] == MEDIUM_TAG
    assert recorded["fallback_reason"]


# ---------------------------------------------------------------------------
# guards: nothing here changes what a weeknight runs
# ---------------------------------------------------------------------------


def test_the_weeknight_slate_is_exactly_what_it_was_before_this_packet():
    """Pinned from the branch this packet starts on (`e8c04f88`: wave 1 + TJ-11F
    + TJ-3 + TJ-15 + TJ-14A). The week story is a SATURDAY slot; if TJ-13B adds
    anything to a weeknight, this is where it shows."""
    from ai_jobs import runner

    expected = (
        "journal_import",
        "journal_auto_tag",
        "veto_cohort_grading",
        "like_cohort_grading",
        "sidecar_completion",
        "pass_cohort_grading",
        "rejection_cohort_grading",
        "note_vocabulary_audit",
        "preference_trade_outcomes",
        "evidence_report",
        "daily_digest",
        "theta_pick_grading",
        "market_story_rollups",
        "miss_contrast",
        "measured_report",
        "ticker_briefs",
        "market_story_narration",
        "journal_enrichment",
        "review_policy_draft",
        "setup_research",
    )
    assert tuple(slot.name for slot in runner.slots_for("weeknight")) == expected


def test_every_registered_slot_still_declares_a_cap_and_a_reserve():
    """plan.md §12.3: a new or changed slot sets `max_attempts` (never 0) and a
    `reserve_minutes`, and declares `uses_model` honestly."""
    from ai_jobs import runner

    for slot in runner.default_slots() + runner.optional_slots():
        assert slot.max_attempts != 0, f"{slot.name} may retry to the end of the window"
        assert slot.reserve_minutes > 0, f"{slot.name} reserves nothing"
        assert isinstance(slot.uses_model, bool)


def test_the_shared_local_default_is_still_the_medium_model():
    """Guard on a SHARED function this packet must not repoint: every other
    caller of `default_model_for("local")` - the digest, the briefs, the
    enrichment - is a medium-tier job."""
    import ai_summary

    with _live_settings():
        assert ai_summary.default_model_for("local") == MEDIUM_TAG
