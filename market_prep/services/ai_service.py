from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from market_prep.models import MarketPrepConfig


DEFAULT_AI_SETTINGS = {
    "enabled": True,
    "model": "gpt-5.2",
    "timeout_seconds": 30,
    "max_context_chars": 12000,
}


def build_market_prep_ai_brief(
    report: dict[str, Any],
    *,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    settings = get_market_prep_ai_settings(config)
    prompt = build_market_prep_ai_prompt(report, max_chars=_safe_int(settings.get("max_context_chars"), 12000))
    generated_at = datetime.now().isoformat(timespec="seconds")
    if not bool(settings.get("enabled", True)):
        return _fallback_payload("disabled", "Market Prep AI disabled in config.", prompt, generated_at)

    api_key = resolve_openai_api_key(config)
    if not api_key:
        return _fallback_payload("missing_key", "No OpenAI API key configured.", prompt, generated_at)

    try:
        from openai import OpenAI
    except Exception as exc:
        return _fallback_payload("missing_sdk", f"OpenAI SDK unavailable: {exc}", prompt, generated_at)

    try:
        client = OpenAI(api_key=api_key, timeout=_safe_int(settings.get("timeout_seconds"), 30))
        response = client.responses.create(
            model=str(settings.get("model") or "gpt-5.2"),
            instructions=(
                "You are a trading preparation assistant. Produce concise checklist-style market prep. "
                "Do not provide financial advice or tell the user to buy/sell. Separate confirmed scheduled "
                "catalysts from speculative thesis ideas."
            ),
            input=prompt,
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            text = "AI returned no text. Use the deterministic checklist below."
        return {
            "generated_at": generated_at,
            "source": "openai",
            "status": "ok",
            "status_label": "Ready",
            "model": str(settings.get("model") or "gpt-5.2"),
            "summary": text,
            "prompt": prompt,
            "warnings": [],
        }
    except Exception as exc:
        return _fallback_payload("failed", f"OpenAI brief failed: {exc}", prompt, generated_at)


def build_market_prep_ai_prompt(report: dict[str, Any], *, max_chars: int = 12000) -> str:
    report_type = str(report.get("report_type") or "market prep").lower()
    markdown = str(report.get("markdown") or "").strip()
    deterministic = _deterministic_context(report)
    context = (deterministic + "\n\n" + markdown).strip()
    if len(context) > max_chars:
        context = context[:max_chars].rstrip() + "\n\n[Context truncated]"
    if report_type == "weekly":
        task = (
            "Create a weekly thesis checklist with: market regime, main risk days, confirmed catalysts, "
            "watchlist names to review, no-swing/no-overnight windows, and speculative thesis ideas."
        )
    else:
        task = (
            "Create a start-of-day landmine checklist with: today's risk windows, confirmed catalysts, "
            "watchlist tickers to avoid blind holds in, market posture, and trade-management reminders."
        )
    return (
        f"{task}\n\n"
        "Rules: use only the context below, keep it concise, label speculation clearly, and do not give buy/sell instructions.\n\n"
        f"Context:\n{context}"
    )


def get_market_prep_ai_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_AI_SETTINGS)
    if config is not None and isinstance(getattr(config, "market_prep_ai", None), dict):
        settings.update(config.market_prep_ai)
    return settings


def resolve_openai_api_key(config: MarketPrepConfig | None) -> str:
    env_value = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_value:
        return env_value
    keys = getattr(config, "api_keys", {}) if config is not None else {}
    return str(keys.get("openai") or "").strip() if isinstance(keys, dict) else ""


def _deterministic_context(report: dict[str, Any]) -> str:
    blocks = []
    for title, key in (
        ("Daily landmine checklist", "daily_landmine_checklist"),
        ("Weekly thesis checklist", "weekly_thesis_checklist"),
        ("No-trade windows", "no_trade_windows"),
        ("Overnight hold warnings", "overnight_hold_warnings"),
        ("Swing conditions", "swing_trading_conditions"),
        ("Trading posture", "trading_posture"),
    ):
        rows = report.get(key)
        if isinstance(rows, list) and rows:
            blocks.append(title + ":\n" + "\n".join(f"- {row}" for row in rows if str(row).strip()))
    market = report.get("market_snapshot") if isinstance(report.get("market_snapshot"), dict) else {}
    classification = market.get("classification") if isinstance(market.get("classification"), dict) else {}
    if classification:
        blocks.append(
            "Market snapshot:\n"
            f"- {classification.get('label') or 'n/a'}: {classification.get('reason') or 'n/a'}"
        )
    return "\n\n".join(blocks)


def _fallback_payload(status: str, message: str, prompt: str, generated_at: str) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source": "openai",
        "status": status,
        "status_label": message,
        "model": "",
        "summary": "",
        "prompt": prompt,
        "warnings": [message] if message else [],
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
