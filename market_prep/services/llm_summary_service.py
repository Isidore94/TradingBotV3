from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import requests

from market_prep.config_loader import get_market_prep_openai_api_key
from market_prep.models import MarketPrepConfig

try:
    from lxml import html
except ImportError:  # pragma: no cover - lxml is in project requirements, but keep fallback-safe.
    html = None


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5-mini"
MACRO_TERMS = {
    "fed": 12,
    "fomc": 16,
    "powell": 10,
    "rate": 8,
    "rates": 8,
    "yield": 10,
    "yields": 10,
    "treasury": 12,
    "auction": 12,
    "cpi": 16,
    "ppi": 12,
    "pce": 16,
    "inflation": 14,
    "jobs": 10,
    "payroll": 14,
    "payrolls": 14,
    "unemployment": 10,
    "jobless": 8,
    "gdp": 8,
    "dollar": 6,
    "oil": 8,
    "crude": 8,
    "tariff": 9,
    "tariffs": 9,
    "export controls": 10,
    "semiconductor": 8,
    "semiconductors": 8,
    "nvidia": 7,
    "nvda": 7,
    "ai": 5,
    "earnings": 4,
    "guidance": 4,
}


def generate_market_prep_llm_summary(
    report: dict,
    *,
    config: MarketPrepConfig,
    api_key: str | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    if not isinstance(report, dict) or not report:
        return _status_payload(
            generated_at,
            "failed",
            "Failed",
            "Run Daily Prep or Weekly Prep before requesting an AI summary.",
        )

    key = str(api_key or get_market_prep_openai_api_key(config) or "").strip()
    if not key:
        return _status_payload(
            generated_at,
            "missing_key",
            "Missing API key",
            "Add an OpenAI API key in the Market Prep AI settings or set OPENAI_API_KEY.",
        )

    settings = config.llm_summary if isinstance(config.llm_summary, dict) else {}
    model = str(settings.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    max_output_tokens = _clamp_int(settings.get("max_output_tokens"), 120, 1000, 300)
    headline_limit = _clamp_int(settings.get("headline_limit"), 5, 40, 20)
    article_limit = _clamp_int(settings.get("article_limit"), 0, 8, 4)
    article_char_limit = _clamp_int(settings.get("article_char_limit"), 500, 5000, 2000)
    request_timeout = _clamp_int(settings.get("request_timeout_seconds"), 10, 120, 45)
    article_timeout = _clamp_int(settings.get("article_timeout_seconds"), 3, 30, 8)

    ranked_headlines = rank_market_prep_headlines(report, limit=headline_limit)
    article_rows, article_warnings = fetch_ranked_article_snippets(
        ranked_headlines,
        limit=article_limit,
        char_limit=article_char_limit,
        timeout=article_timeout,
    )
    digest = build_market_prep_llm_digest(
        report,
        headlines=ranked_headlines,
        articles=article_rows,
        headline_limit=headline_limit,
    )

    payload = {
        "model": model,
        "instructions": (
            "You are a concise market-prep assistant for an active trader. Use only the supplied "
            "calendar, earnings, watchlist, headline, and article-snippet data. Provide a brief "
            "macro economic outlook, why it matters for today's risk, and what catalysts deserve "
            "attention. Do not invent unsupplied facts, do not provide personalized financial advice, "
            "and keep the answer compact."
        ),
        "input": digest,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    try:
        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=request_timeout,
        )
    except Exception as exc:
        return _status_payload(generated_at, "failed", "Failed", f"OpenAI request failed: {exc}")

    if response.status_code >= 400:
        detail = _normalize_spaces(response.text)[:500]
        return _status_payload(
            generated_at,
            "failed",
            "Failed",
            f"OpenAI request failed with HTTP {response.status_code}: {detail}",
        )

    try:
        response_payload = response.json()
    except ValueError as exc:
        return _status_payload(generated_at, "failed", "Failed", f"OpenAI response was not JSON: {exc}")

    summary = _extract_response_text(response_payload)
    if not summary:
        return _status_payload(generated_at, "failed", "Failed", "OpenAI response did not include summary text.")

    return {
        "generated_at": generated_at,
        "status": "ok",
        "status_label": "Ready",
        "model": model,
        "summary": summary.strip(),
        "headline_count": len(ranked_headlines),
        "article_count": len(article_rows),
        "used_headlines": _public_headline_rows(ranked_headlines[:headline_limit]),
        "used_articles": [
            {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "char_count": len(str(item.get("text") or "")),
            }
            for item in article_rows
        ],
        "warnings": article_warnings,
        "prompt_char_count": len(digest),
    }


def build_market_prep_llm_digest(
    report: dict,
    *,
    headlines: list[dict[str, Any]] | None = None,
    articles: list[dict[str, Any]] | None = None,
    headline_limit: int = 20,
) -> str:
    lines: list[str] = [
        f"Report type: {report.get('report_type') or 'market_prep'}",
        f"Report date: {report.get('report_date') or ''}",
        f"Generated at: {report.get('generated_at') or ''}",
        "",
    ]
    posture = _string_rows(report.get("trading_posture") or report.get("swing_trading_conditions"), limit=5)
    if posture:
        lines.extend(["Trading posture already generated by rules:", *_bullet_lines(posture), ""])

    clock = _dict_rows(report.get("catalyst_clock"), limit=12)
    if clock:
        lines.extend(["Catalyst clock:", *(_bullet_lines(_format_clock_item(row) for row in clock)), ""])

    event_rows = _important_event_rows(report)
    if event_rows:
        lines.extend(["Macro/Fed/Treasury events:", *(_bullet_lines(_format_event(row) for row in event_rows[:16])), ""])

    earnings_rows = _important_earnings_rows(report)
    if earnings_rows:
        lines.extend(["Notable earnings/watchlist risk:", *(_bullet_lines(_format_risk_or_earnings(row) for row in earnings_rows[:14])), ""])

    headline_rows = headlines if headlines is not None else rank_market_prep_headlines(report, limit=headline_limit)
    if headline_rows:
        lines.extend(["Ranked headlines:", *(_bullet_lines(_format_headline(row) for row in headline_rows[:headline_limit])), ""])

    article_rows = articles or []
    if article_rows:
        lines.append("Article snippets:")
        for index, article in enumerate(article_rows, start=1):
            lines.append(
                f"{index}. {article.get('title') or 'Untitled'} ({article.get('source') or 'source unknown'})"
            )
            lines.append(f"URL: {article.get('url') or ''}")
            lines.append(str(article.get("text") or "").strip())
            lines.append("")

    lines.append(
        "Requested output: 1 short paragraph plus 3-5 bullets. Include a final compact 'Used:' line naming the source headlines/articles that mattered most."
    )
    return "\n".join(str(line).rstrip() for line in lines).strip()


def rank_market_prep_headlines(report: dict, *, limit: int = 20) -> list[dict[str, Any]]:
    payload = report.get("rss_headlines") if isinstance(report.get("rss_headlines"), dict) else {}
    headlines = payload.get("headlines") if isinstance(payload, dict) else []
    rows = [row for row in headlines if isinstance(row, dict)] if isinstance(headlines, list) else []
    rows.sort(key=lambda row: (-_headline_score(row), str(row.get("published") or ""), str(row.get("title") or "")))
    return rows[: max(0, int(limit))]


def fetch_ranked_article_snippets(
    headlines: list[dict[str, Any]],
    *,
    limit: int,
    char_limit: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    snippets: list[dict[str, Any]] = []
    warnings: list[str] = []
    for headline in headlines:
        if len(snippets) >= max(0, int(limit)):
            break
        url = str(headline.get("url") or "").strip()
        if not url:
            continue
        try:
            text = fetch_article_text(url, timeout=timeout, char_limit=char_limit)
        except Exception as exc:
            warnings.append(f"{headline.get('title') or url}: {exc}")
            continue
        if not text:
            continue
        snippets.append(
            {
                "title": str(headline.get("title") or "").strip(),
                "source": str(headline.get("source") or "").strip(),
                "url": url,
                "text": text,
            }
        )
    return snippets, warnings[:8]


def fetch_article_text(url: str, *, timeout: int, char_limit: int) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": "TradingBotV3 market prep article reader",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.7",
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}")
    content_type = str(response.headers.get("content-type") or "").lower()
    if "pdf" in content_type:
        raise RuntimeError("PDF article skipped")
    text = _extract_html_text(response.text)
    return text[: max(0, int(char_limit))].strip()


def _extract_html_text(raw_html: str) -> str:
    if html is None:
        return _normalize_spaces(re.sub(r"<[^>]+>", " ", raw_html))
    try:
        document = html.fromstring(raw_html)
    except Exception:
        return _normalize_spaces(re.sub(r"<[^>]+>", " ", raw_html))
    for node in document.xpath("//script|//style|//noscript|//svg|//header|//footer|//nav|//form"):
        try:
            node.drop_tree()
        except Exception:
            pass
    article_paragraphs = [_normalize_spaces(node.text_content()) for node in document.xpath("//article//p")]
    paragraphs = [text for text in article_paragraphs if len(text) >= 40]
    if sum(len(text) for text in paragraphs) < 600:
        paragraphs = [
            _normalize_spaces(node.text_content())
            for node in document.xpath("//p")
            if len(_normalize_spaces(node.text_content())) >= 40
        ]
    return "\n".join(_dedupe_strings(paragraphs[:24])).strip()


def _important_event_rows(report: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("todays_events", "next_7_events", "economic_calendar", "fed_calendar", "treasury_calendar"):
        payload = report.get(key) if isinstance(report.get(key), dict) else {}
        rows.extend(_dict_rows(payload.get("events"), limit=80))
    rows.sort(key=lambda row: (_event_priority_sort(row), str(row.get("date") or ""), str(row.get("time_et") or "")))
    return rows


def _important_earnings_rows(report: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("watchlist_risk", "watchlist_earnings_risk"):
        payload = report.get(key) if isinstance(report.get(key), dict) else {}
        rows.extend(_dict_rows(payload.get("risks"), limit=80))
    for key in ("today_tomorrow_earnings", "next_7_earnings", "major_earnings"):
        payload = report.get(key) if isinstance(report.get(key), dict) else {}
        rows.extend(_dict_rows(payload.get("earnings"), limit=80))
    rows.sort(key=lambda row: (_risk_priority_sort(row), str(row.get("date") or ""), str(row.get("ticker") or "")))
    return rows


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("output_text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def _headline_score(headline: dict[str, Any]) -> int:
    text = " ".join(
        [
            str(headline.get("title") or ""),
            str(headline.get("summary") or ""),
            str(headline.get("query") or ""),
            " ".join(headline.get("tags") or []),
        ]
    ).lower()
    score = 0
    for term, value in MACRO_TERMS.items():
        if term in text:
            score += value
    if headline.get("tags"):
        score += 8
    return score


def _public_headline_rows(headlines: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "title": str(row.get("title") or ""),
            "source": str(row.get("source") or ""),
            "url": str(row.get("url") or ""),
        }
        for row in headlines
    ]


def _format_clock_item(row: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(row.get("date") or ""),
            f"{row.get('time_et') or 'TBD'} ET",
            str(row.get("bucket") or "Catalyst"),
            str(row.get("priority") or ""),
            str(row.get("text") or ""),
        )
        if part
    )


def _format_event(row: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(row.get("date") or ""),
            f"{row.get('time_et') or 'TBD'} ET",
            f"[{row.get('priority')}]" if row.get("priority") else "",
            str(row.get("currency") or ""),
            str(row.get("event") or row.get("title") or ""),
            _event_stats(row),
        )
        if part
    )


def _format_risk_or_earnings(row: dict[str, Any]) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    label = str(row.get("classification") or row.get("importance") or "").strip()
    reason = str(row.get("reason") or row.get("notes") or row.get("company") or "").strip()
    date_value = str(row.get("date") or "").strip()
    return " | ".join(part for part in (date_value, ticker, label, reason) if part)


def _format_headline(row: dict[str, Any]) -> str:
    tags = ", ".join(row.get("tags") or [])
    parts = [
        str(row.get("title") or "").strip(),
        str(row.get("source") or "").strip(),
        f"tags={tags}" if tags else "",
        str(row.get("query") or "").strip(),
        str(row.get("url") or "").strip(),
    ]
    return " | ".join(part for part in parts if part)


def _event_stats(row: dict[str, Any]) -> str:
    stats = []
    for label, key in (("actual", "actual"), ("forecast", "forecast"), ("previous", "previous")):
        value = str(row.get(key) or "").strip()
        if value:
            stats.append(f"{label}={value}")
    return ", ".join(stats)


def _event_priority_sort(row: dict[str, Any]) -> int:
    value = str(row.get("priority") or "").upper()
    if value in {"HIGH", "MEGA"}:
        return 0
    if value == "MEDIUM":
        return 1
    return 2


def _risk_priority_sort(row: dict[str, Any]) -> int:
    value = str(row.get("classification") or row.get("importance") or "").upper()
    if "TODAY" in value or "MEGA" in value:
        return 0
    if "HIGH" in value:
        return 1
    if "MEDIUM" in value or "RISK" in value:
        return 2
    return 3


def _status_payload(generated_at: str, status: str, status_label: str, message: str) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "status": status,
        "status_label": status_label,
        "summary": "",
        "message": message,
        "warnings": [message] if status == "failed" else [],
    }


def _dict_rows(value, *, limit: int) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    return [row for row in rows if isinstance(row, dict)][: max(0, int(limit))]


def _string_rows(value, *, limit: int) -> list[str]:
    rows = value if isinstance(value, list) else []
    return [str(row).strip() for row in rows if str(row).strip()][: max(0, int(limit))]


def _bullet_lines(values) -> list[str]:
    return [f"- {str(value).strip()}" for value in values if str(value).strip()]


def _clamp_int(value, minimum: int, maximum: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, minimum), maximum)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        cleaned = _normalize_spaces(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped
