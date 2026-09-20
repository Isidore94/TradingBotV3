"""Which model writes the week story, and what the record says afterwards.

`plan.md` §12.4 TJ-13 item 7, decision 0021 answer 20: *"the week story is
written by the large LOCAL model … `ai_week_review_provider` defaults to
`local_large`, falls back to local medium and says so in the ledger; `openai`
remains a setting, off."*

Three things live here and nowhere else:

1. **The setting.** ``ai_week_review_provider``, default ``local_large``,
   ``openai`` available to anybody who types it and off until they do.
2. **The plan.** :func:`week_review_plan` answers "may the 27B run tonight, and
   what does it reserve?" from :mod:`ai_jobs.model_probe`'s recorded
   measurement. With no measurement it answers **no**, says why, and names the
   MEDIUM model instead - the trader wants a week story every Saturday
   (lead decision, 2026-09-19), so the absence of a measurement costs the large
   model, never the story.
3. **The request.** :func:`request_with_fallback` asks the large model, falls
   back to the medium one when it cannot answer, and returns an ATTRIBUTION
   saying which model was asked, which answered and why. The live ledger's
   ``model`` column is a single string: a row reading ``gemma3:12b-tbv3ctx-64k``
   cannot otherwise be told apart from a night when the large model was never
   attempted.

``local_large`` is a name in THIS seam, deliberately not in
``ai_summary.normalize_provider``: that vocabulary is shared with
``ai_credentials``, whose ``PROVIDER_ENV_KEYS`` knows only openai and anthropic
and RAISES on anything else. One vocabulary across both modules is a separate
piece of work, not a side effect of this one. Underneath, every call here is an
ordinary ``provider="local"`` request - one provider path, two tiers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import requests

from ai_jobs import ledger

_log = logging.getLogger(__name__)

#: The large local tier, as a provider NAME (this module's vocabulary only).
LOCAL_LARGE = "local_large"
#: The medium local tier, which is also what ``local_large`` falls back to.
LOCAL_MEDIUM = "local_medium"
#: A setting, and off until it is typed. No key is read and no request is built
#: for it here: decision 0021 answer 20 keeps it available, not wired.
OPENAI = "openai"

WEEK_REVIEW_PROVIDER_SETTING = "ai_week_review_provider"
WEEK_REVIEW_PROVIDERS = (LOCAL_LARGE, LOCAL_MEDIUM, OPENAI)

#: Row field the attribution is written under, by the SLOT that publishes.
LEDGER_FIELD = "model_attribution"

#: The same 900 s every other local session-scale call uses.
DEFAULT_TIMEOUT_SECONDS = 900


def week_review_provider() -> str:
    """Which provider the week story asks for. Default: the large local model."""
    import ai_summary

    raw = str(ai_summary.get_local_setting(WEEK_REVIEW_PROVIDER_SETTING, "") or "").strip().lower()
    if not raw:
        return LOCAL_LARGE
    if raw not in WEEK_REVIEW_PROVIDERS:
        _log.warning(
            "%s is set to %r, which this desk does not know; using %s.",
            WEEK_REVIEW_PROVIDER_SETTING,
            raw,
            LOCAL_LARGE,
        )
        return LOCAL_LARGE
    return raw


def week_review_plan(*, ledger_path=None) -> dict[str, Any]:
    """May the week story run on the large model tonight, and what does it cost?

    A REPORT: it reads the ledger and writes nothing. Reading what was measured
    must not itself look like a measurement.

    ``may_run_large`` is False until a probe row exists. ``reserve_minutes`` is
    then ``None`` rather than a default - see
    :func:`ai_jobs.model_probe.reserve_minutes_from_probe`.
    """
    import ai_summary

    from ai_jobs import model_probe

    provider = week_review_provider()
    medium = ai_summary.local_model("medium")

    if provider != LOCAL_LARGE:
        model = ai_summary.local_model("medium") if provider == LOCAL_MEDIUM else ""
        return {
            "provider": provider,
            "model": model,
            "reserve_minutes": None,
            "may_run_large": False,
            "reason": (
                f"{WEEK_REVIEW_PROVIDER_SETTING} is set to {provider!r}, so the large "
                "local model is not what this week story asks for"
            ),
        }

    reserve = model_probe.reserve_minutes_from_probe(tier="large", ledger_path=ledger_path)
    if reserve is None:
        return {
            "provider": provider,
            "model": medium,
            "reserve_minutes": None,
            "may_run_large": False,
            "reason": (
                "the large local model has never been measured on this desk, so "
                "there is no honest reserve for it: run "
                "`python scripts/run_ai_jobs.py --probe-model large` inside the "
                "night window with no AI job running. The week story still runs, "
                f"on the medium model ({medium})"
            ),
        }
    large = ai_summary.local_model("large")
    return {
        "provider": provider,
        "model": large,
        "reserve_minutes": reserve,
        "may_run_large": True,
        "reason": (
            f"{large} was measured on this desk; a slot on it reserves "
            f"{reserve} min ({model_probe.describe_measurement(tier='large', ledger_path=ledger_path)})"
        ),
    }


def _attribution(provider: str, model_asked: str) -> dict[str, str]:
    return {
        "provider": str(provider),
        "model_asked": str(model_asked),
        "model_answered": "",
        "fallback_reason": "",
    }


def request_with_fallback(
    *,
    provider: str = LOCAL_LARGE,
    evidence: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    schema_name: str = "tradingbot_ai_summary",
    post: Callable[..., Any] = requests.post,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ledger_path=None,
) -> dict[str, Any]:
    """Ask the large local model, fall back to the medium one, and write NO ledger row.

    The row is the calling SLOT's: it writes one, carrying ``attribution``
    under :data:`LEDGER_FIELD`. A row written here as well would be two records
    of one night. ``ledger_path`` is accepted for symmetry with the rest of
    this seam and is deliberately unused.

    Returns ``{"status", "result", "attribution"}``. ``result`` is
    ``ai_summary.request_ai_summary``'s envelope, or ``None`` when neither model
    answered - both failing publishes NOTHING, because a half-answer under
    whichever model happened to be reachable is worse than last week's file.

    Both models are sent the IDENTICAL contract: one schema, one timeout, one
    citation rule. A second shape would be a second place for all three to
    drift.

    Raises ``ValueError`` for ``openai`` - a setting that is off, refused
    before anything is sent - and for any name that is not a local tier. A slot
    calling this must catch it: a raise here is a FAILED row with a sentence on
    it, not a night that publishes half an answer.
    """
    import ai_summary

    if provider == OPENAI:
        raise ValueError(
            f"{WEEK_REVIEW_PROVIDER_SETTING} is set to 'openai'. It remains a "
            "setting and is off (decision 0021 answer 20): this seam speaks the "
            "local tiers only, so nothing was sent and nothing was spent"
        )
    if provider not in (LOCAL_LARGE, LOCAL_MEDIUM):
        raise ValueError(f"unknown week-review provider {provider!r}")

    medium = ai_summary.local_model("medium")
    model_asked = ai_summary.local_model("large") if provider == LOCAL_LARGE else medium
    attribution = _attribution(provider, model_asked)

    def _ask(model: str) -> dict[str, Any]:
        return ai_summary.request_ai_summary(
            provider="local",
            model=model,
            api_key="",
            evidence=evidence,
            timeout_seconds=timeout_seconds,
            post=post,
            schema=schema,
            schema_name=schema_name,
        )

    try:
        result = _ask(model_asked)
    except Exception as exc:
        first_failure = f"{model_asked} could not answer ({type(exc).__name__}: {exc})"
        _log.warning("week review: %s", first_failure)
        if provider != LOCAL_LARGE or not medium or medium == model_asked:
            attribution["fallback_reason"] = (
                f"{first_failure}; there is no other local tier to ask"
            )
            return {"status": ledger.STATUS_FAILED, "result": None, "attribution": attribution}
        try:
            result = _ask(medium)
        except Exception as second:
            attribution["fallback_reason"] = (
                f"{first_failure}; {medium} could not answer either "
                f"({type(second).__name__}: {second})"
            )
            _log.error("week review: %s", attribution["fallback_reason"])
            return {"status": ledger.STATUS_FAILED, "result": None, "attribution": attribution}
        attribution["model_answered"] = str(result.get("model") or medium)
        attribution["fallback_reason"] = (
            f"{first_failure}; the week story was written by {medium} instead"
        )
        return {"status": ledger.STATUS_OK, "result": result, "attribution": attribution}

    attribution["model_answered"] = str(result.get("model") or model_asked)
    return {"status": ledger.STATUS_OK, "result": result, "attribution": attribution}
