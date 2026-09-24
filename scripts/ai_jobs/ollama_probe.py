"""Night-start check that the local model answers (WISHLIST P1-3 packet 3e).

One one-token chat call to the medium model, capped at 30 s. It loads the model
if it is not loaded yet. The runner calls it before the first model slot of a
firing; on failure the firing runs deterministic work only and the ledger row
this writes is what the Health page and the phone digest read.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ai_jobs import ledger

#: Ledger job name of the probe row. Not a slot: it never counts as coverage work.
PROBE_JOB = "ollama_probe"
#: Hard cap for the one call, in seconds (connect + load + one token).
PROBE_TIMEOUT_SECONDS = 30.0
#: Local setting that overrides the cap if a cold model load needs longer.
PROBE_TIMEOUT_SETTING = "ai_ollama_probe_timeout_seconds"


def probe_local_model(
    *,
    post: Callable[..., Any] | None = None,
    timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[bool, str]:
    """(answered, one plain sentence). Never raises."""
    try:
        import ai_summary

        endpoint = ai_summary.local_endpoint_url()
        model = ai_summary.local_model("medium")
        if timeout_seconds is None:
            raw = ai_summary.get_local_setting(PROBE_TIMEOUT_SETTING, PROBE_TIMEOUT_SECONDS)
            try:
                timeout_seconds = float(raw) if float(raw) > 0 else PROBE_TIMEOUT_SECONDS
            except (TypeError, ValueError):
                timeout_seconds = PROBE_TIMEOUT_SECONDS
    except Exception as exc:  # noqa: BLE001 - an unreadable setting is a failed probe
        return False, f"local model settings unreadable ({type(exc).__name__}: {exc})"
    if not endpoint:
        return False, "no local model endpoint is set (ai_local_endpoint_url)"
    if post is None:
        import requests

        post = requests.post
    url = endpoint + ai_summary.LOCAL_CHAT_COMPLETIONS_PATH
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    started = clock()
    try:
        response = post(
            url,
            json=payload,
            headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure is "did not answer"
        return False, (
            f"{model} did not answer within {timeout_seconds:.0f} s "
            f"({type(exc).__name__}: {str(exc)[:160]})"
        )
    elapsed = clock() - started
    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code >= 400:
        detail = str(getattr(response, "text", "") or "")[:160]
        return False, f"{model} answered HTTP {status_code}: {detail}".rstrip(": ")
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {}
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        return False, f"{model} answered with no completion"
    return True, f"{model} answered one token in {elapsed:.1f} s"


def record_probe(
    answered: bool, detail: str, *, session_date: str, path=None
) -> dict[str, Any]:
    """Write the probe's one ledger row (ok or failed)."""
    return ledger.record(
        job=PROBE_JOB,
        status=ledger.STATUS_OK if answered else ledger.STATUS_FAILED,
        session_date=session_date,
        reason=detail if answered else f"Ollama probe failed: {detail}; the night runs deterministic work only",
        error="" if answered else detail,
        path=path,
    )
