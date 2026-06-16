from __future__ import annotations

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from market_prep.models import MarketPrepConfig


DEFAULT_AI_SETTINGS = {
    "enabled": True,
    "model": "gpt-5.2",
    "timeout_seconds": 30,
    "max_context_chars": 12000,
}
OPENAI_LOCAL_SETTING_KEY = "market_prep_openai_api_key"


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


def build_ticker_lookup_ai_brief(
    payload: dict[str, Any],
    *,
    config: MarketPrepConfig | None = None,
) -> dict[str, Any]:
    settings = get_market_prep_ai_settings(config)
    prompt = build_ticker_lookup_ai_prompt(payload, max_chars=_safe_int(settings.get("max_context_chars"), 12000))
    generated_at = datetime.now().isoformat(timespec="seconds")
    if not bool(settings.get("enabled", True)):
        return _fallback_payload("disabled", "Ticker Lookup AI disabled in config.", prompt, generated_at)

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
                "You are a trading risk-research assistant. Identify ticker-specific landmines, hidden exposure, "
                "scheduled catalysts, and context worth checking before a swing trade. Do not provide buy/sell "
                "instructions. Separate confirmed source-backed items from speculation or missing-information checks."
            ),
            input=prompt,
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            text = "AI returned no text. Review the deterministic lookup sections below."
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
        return _fallback_payload("failed", f"OpenAI ticker brief failed: {exc}", prompt, generated_at)


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
            "Create a start-of-day landmine checklist with: today's risk windows, the next 5 trading/calendar days roadmap, "
            "confirmed catalysts/news events, post-event data recaps, watchlist tickers to avoid blind holds in, "
            "market posture, and trade-management reminders."
        )
    return (
        f"{task}\n\n"
        "Rules: use only the context below, prioritize scheduled events and confirmed major news over generic headlines, "
        "keep it concise, label speculation clearly, and do not give buy/sell instructions.\n\n"
        f"Context:\n{context}"
    )


def build_ticker_lookup_ai_prompt(payload: dict[str, Any], *, max_chars: int = 12000) -> str:
    ticker = str(payload.get("ticker") or "").strip().upper() or "THE TICKER"
    markdown = str(payload.get("markdown") or "").strip()
    context = markdown or _ticker_lookup_context(payload)
    if len(context) > max_chars:
        context = context[:max_chars].rstrip() + "\n\n[Context truncated]"
    return (
        f"Create a ticker landmine brief for {ticker}.\n\n"
        "Include:\n"
        "- Start with exactly one swing safety rating: CLEAN, CAUTION, or AVOID/WAIT, with one sentence why.\n"
        "- Company/exposure snapshot, including strategic stakes, investments, subsidiaries, customers, suppliers, or partners if present.\n"
        "- Scheduled catalysts and timing risk over the lookup window.\n"
        "- Recent major announcements/news, upcoming investor events, earnings, guidance, product/customer events, and peer earnings that could move the ticker.\n"
        "- SEC, financing, dilution, debt, legal/regulatory, rating, guidance, and macro/industry risks.\n"
        "- Any hidden or second-order exposure that should be manually verified, such as AI/private-company stakes.\n"
        "- A short checklist of missing facts to verify before treating the ticker as clean.\n\n"
        "Rules: use only the context below, label speculation clearly, cite headline/source names when available, "
        "do not treat stale previews, analyst expectations, or thesis headlines as roadblocks unless the context shows a confirmed material event, "
        "do not call a ticker clean if earnings, high-risk SEC filings, financing/dilution, or major legal/regulatory risk are unresolved, "
        "and do not give buy/sell instructions.\n\n"
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
    local_value = _local_openai_api_key()
    if local_value:
        return local_value
    keys = getattr(config, "api_keys", {}) if config is not None else {}
    return str(keys.get("openai") or "").strip() if isinstance(keys, dict) else ""


def _local_openai_api_key() -> str:
    try:
        payload = json.loads(_local_settings_file().read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get(OPENAI_LOCAL_SETTING_KEY) or "").strip()


def _local_settings_file() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "TradingBotV3" / "local_settings.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TradingBotV3" / "local_settings.json"
    return Path.home() / ".local" / "share" / "TradingBotV3" / "local_settings.json"


def _ticker_lookup_context(payload: dict[str, Any]) -> str:
    ticker = str(payload.get("ticker") or "").strip().upper()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    lines = [
        f"Ticker: {ticker}",
        f"Company: {metadata.get('company_name') or 'n/a'}",
        f"Sector/industry: {metadata.get('sector') or 'n/a'} / {metadata.get('industry') or 'n/a'}",
        f"Market cap: {metadata.get('market_cap_fmt') or 'n/a'}",
        f"Peer tickers: {', '.join(payload.get('peer_tickers') or []) or 'n/a'}",
    ]
    swing_risk = payload.get("swing_risk") if isinstance(payload.get("swing_risk"), dict) else {}
    if swing_risk:
        lines.extend(
            [
                f"Swing verdict: {swing_risk.get('verdict') or 'n/a'}",
                f"Swing risk score: {swing_risk.get('risk_score', 0)}/100",
                f"Swing summary: {swing_risk.get('summary') or 'n/a'}",
            ]
        )
        risk_rows = swing_risk.get("risk_items") if isinstance(swing_risk.get("risk_items"), list) else []
        if risk_rows:
            lines.append("Deterministic road bumps:")
            for row in risk_rows[:15]:
                if isinstance(row, dict):
                    parts = [
                        str(row.get(key) or "")
                        for key in ("severity", "category", "date", "title", "reason")
                        if str(row.get(key) or "").strip()
                    ]
                    lines.append("- " + " | ".join(parts))
    for title, key in (
        ("Landmine headlines", "landmine_headlines"),
        ("Target headlines", "target_headlines"),
        ("Industry headlines", "industry_headlines"),
        ("Target earnings", "target_earnings"),
        ("Peer earnings", "peer_earnings"),
    ):
        rows = payload.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        lines.append("")
        lines.append(title + ":")
        for row in rows[:25]:
            if isinstance(row, dict):
                lines.append("- " + " | ".join(str(value) for value in row.values() if str(value).strip()))
    sec_payload = payload.get("sec_filings") if isinstance(payload.get("sec_filings"), dict) else {}
    sec_rows = sec_payload.get("filings") if isinstance(sec_payload.get("filings"), list) else []
    if sec_rows:
        lines.append("")
        lines.append("SEC filings:")
        for row in sec_rows[:20]:
            if isinstance(row, dict):
                lines.append("- " + " | ".join(str(value) for value in row.values() if str(value).strip()))
    return "\n".join(lines)



def _deterministic_context(report: dict[str, Any]) -> str:
    blocks = []
    future_roadmap = report.get("future_roadmap") if isinstance(report.get("future_roadmap"), dict) else {}
    roadmap_lines = _future_roadmap_context(future_roadmap)
    if roadmap_lines:
        blocks.append("Next 5 days roadmap:\n" + "\n".join(roadmap_lines))
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


def _future_roadmap_context(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    recaps = payload.get("recaps") if isinstance(payload.get("recaps"), list) else []
    for recap in recaps[:8]:
        if isinstance(recap, dict):
            parts = [
                "RECAP",
                str(recap.get("date") or ""),
                str(recap.get("bucket") or ""),
                str(recap.get("title") or ""),
                str(recap.get("detail") or ""),
            ]
            lines.append("- " + " | ".join(part for part in parts if part))
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for item in items[:18]:
        if isinstance(item, dict):
            parts = [
                "UPCOMING",
                str(item.get("date") or ""),
                str(item.get("time_et") or ""),
                str(item.get("bucket") or ""),
                str(item.get("priority") or ""),
                str(item.get("title") or ""),
                str(item.get("detail") or ""),
            ]
            lines.append("- " + " | ".join(part for part in parts if part))
    return lines


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
