from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from market_prep.config_loader import load_market_prep_config
from market_prep.models import MarketPrepConfig
from market_prep.services.earnings_service import get_watchlist_earnings
from market_prep.services.ai_service import build_ticker_lookup_ai_brief
from market_prep.services.rss_news_service import fetch_rss_headlines
from market_prep.services.sec_service import get_sec_filing_risk
from market_prep.services.yfinance_service import get_ticker_metadata


DEFAULT_TICKER_LOOKUP_SETTINGS = {
    "days_ahead": 10,
    "news_limit": 80,
    "max_peer_tickers": 12,
    "include_sec_filings": True,
    "include_peer_earnings": True,
    "include_industry_news": True,
    "include_ai_brief": True,
    "queries": [
        "{ticker} earnings",
        "{ticker} guidance",
        "{ticker} analyst rating",
        "{ticker} price target",
        "{ticker} upgrade downgrade",
        "{ticker} investor day",
        "{ticker} conference",
        "{ticker} presentation",
        "{ticker} catalyst",
        "{ticker} announces",
        "{ticker} partnership",
        "{ticker} product launch",
        "{ticker} acquisition",
        "{ticker} contract",
        "{ticker} shipment",
        "{ticker} demand",
        "{ticker} regulation",
        "{ticker} lawsuit",
        "{ticker} offering",
        "{ticker} financing",
        "{ticker} debt",
        "{ticker} strategic investment",
        "{ticker} investment",
        "{ticker} stake",
        "{ticker} ownership",
        "{ticker} portfolio",
        "{ticker} subsidiary",
        "{ticker} joint venture",
        "{ticker} customer",
        "{ticker} supplier",
        "{ticker} revenue exposure",
        "{ticker} AI investment",
        "{ticker} Anthropic",
        "{ticker} OpenAI",
        "{ticker} insider selling",
    ],
}

INDUSTRY_PEER_TICKERS = {
    "semiconductor": ["NVDA", "AMD", "AVGO", "ASML", "QCOM", "INTC", "MU", "ARM", "MRVL", "LRCX", "AMAT"],
    "chip": ["NVDA", "AMD", "AVGO", "ASML", "QCOM", "INTC", "MU", "ARM", "MRVL", "LRCX", "AMAT"],
    "software": ["MSFT", "ORCL", "ADBE", "CRM", "NOW", "PANW", "SNOW", "DDOG"],
    "internet": ["GOOGL", "META", "AMZN", "NFLX", "UBER", "ABNB"],
    "auto": ["TSLA", "GM", "F", "TM", "RIVN", "LCID"],
    "bank": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
    "retail": ["WMT", "COST", "TGT", "HD", "LOW", "AMZN"],
    "biotech": ["AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA"],
    "pharma": ["LLY", "JNJ", "PFE", "MRK", "ABBV", "BMY"],
}

TICKER_PEER_OVERRIDES = {
    "TSM": ["NVDA", "AMD", "AVGO", "ASML", "QCOM", "AAPL", "INTC", "MU"],
    "ASML": ["TSM", "NVDA", "AMD", "AVGO", "INTC", "AMAT", "LRCX"],
    "AMD": ["NVDA", "TSM", "AVGO", "INTC", "QCOM", "MU", "ARM"],
    "NVDA": ["AMD", "TSM", "AVGO", "ASML", "QCOM", "MU", "ARM"],
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "TSM", "QCOM", "AVGO"],
    "SKM": ["KT", "VZ", "T", "TMUS", "ERIC", "NOK", "MSFT", "GOOGL", "AMZN", "NVDA"],
}

LANDMINE_KEYWORD_BUCKETS = {
    "AI/private exposure": ("anthropic", "openai", "artificial intelligence", "ai investment", "private company"),
    "strategic stake": ("stake", "investment", "invests", "invested", "ownership", "portfolio", "holding"),
    "subsidiary/JV": ("subsidiary", "joint venture", "jv", "affiliate", "spin off", "spinoff"),
    "customer/supplier": ("customer", "supplier", "supply", "contract", "deal", "partnership", "partner"),
    "financing/dilution": ("offering", "atm", "at-the-market", "shelf", "financing", "convertible", "dilution"),
    "debt/credit": ("debt", "credit", "loan", "refinancing", "liquidity", "default", "downgrade"),
    "legal/regulatory": ("lawsuit", "probe", "investigation", "regulator", "regulatory", "sec", "doj", "ftc"),
    "earnings/guidance": ("earnings", "guidance", "forecast", "profit warning", "revenue", "margin"),
    "M&A": ("merger", "acquisition", "acquires", "buyout", "takeover", "strategic alternatives"),
    "geopolitical": ("tariff", "sanction", "china", "export control", "korea", "taiwan", "trade restriction"),
    "analyst": ("downgrade", "upgrade", "rating", "price target", "initiates", "cuts target"),
}


def lookup_ticker_context(
    ticker: str,
    *,
    config: MarketPrepConfig | None = None,
    days_ahead: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    active_config = config or load_market_prep_config()
    settings = get_ticker_lookup_settings(active_config)
    symbol = normalize_lookup_ticker(ticker)
    if not symbol:
        raise ValueError("Enter a ticker symbol before running lookup.")

    window_days = _safe_int(days_ahead, _safe_int(settings.get("days_ahead"), 10))
    generated_at = datetime.now().isoformat(timespec="seconds")
    start = datetime.now().date()
    metadata = get_ticker_metadata(symbol, config=active_config)
    peer_tickers = peer_tickers_for_lookup(symbol, metadata, settings=settings)
    lookup_tickers = [symbol] + peer_tickers if bool(settings.get("include_peer_earnings", True)) else [symbol]

    earnings_payload = get_watchlist_earnings(
        lookup_tickers,
        days_ahead=window_days,
        start_date=start,
        config=active_config,
    )
    earnings_rows = earnings_payload.get("earnings") if isinstance(earnings_payload, dict) else []
    earnings_rows = [row for row in earnings_rows if isinstance(row, dict)] if isinstance(earnings_rows, list) else []
    target_earnings = [row for row in earnings_rows if normalize_lookup_ticker(row.get("ticker")) == symbol]
    peer_earnings = [
        row
        for row in earnings_rows
        if normalize_lookup_ticker(row.get("ticker")) in set(peer_tickers)
    ]

    sec_filings = (
        get_sec_filing_risk(active_config, tickers=[symbol], start_date=start, force_refresh=force_refresh)
        if bool(settings.get("include_sec_filings", True))
        else _disabled_payload("SEC EDGAR", "SEC filing lookup disabled for ticker lookup.")
    )
    news_payload = fetch_rss_headlines(
        limit=max(1, _safe_int(settings.get("news_limit"), 40)),
        config=_news_lookup_config(active_config, symbol, metadata, peer_tickers, settings),
        tickers=lookup_tickers,
        force_refresh=True,
    )
    headline_rows = news_payload.get("headlines") if isinstance(news_payload, dict) else []
    headline_rows = [row for row in headline_rows if isinstance(row, dict)] if isinstance(headline_rows, list) else []
    target_headlines, industry_headlines = split_lookup_headlines(
        symbol,
        peer_tickers,
        headline_rows,
        target_terms=target_terms_for_lookup(symbol, metadata),
    )
    landmine_headlines = rank_landmine_headlines(target_headlines + industry_headlines)

    payload = {
        "report_type": "ticker_lookup",
        "ticker": symbol,
        "report_date": start.isoformat(),
        "generated_at": generated_at,
        "window_days": window_days,
        "metadata": metadata,
        "peer_tickers": peer_tickers,
        "peer_reason": peer_reason(metadata),
        "target_earnings": target_earnings,
        "peer_earnings": peer_earnings,
        "earnings_payload": earnings_payload,
        "sec_filings": sec_filings,
        "news_headlines": news_payload,
        "target_headlines": target_headlines,
        "industry_headlines": industry_headlines,
        "landmine_headlines": landmine_headlines,
        "source_status": build_source_status(earnings_payload, sec_filings, news_payload),
    }
    payload["ai_swing_query"] = build_ai_swing_query(payload)
    payload["markdown"] = build_ticker_lookup_markdown(payload)
    payload["ai_brief"] = (
        build_ticker_lookup_ai_brief(payload, config=active_config)
        if bool(settings.get("include_ai_brief", True))
        else _disabled_payload("OpenAI", "Ticker Lookup AI disabled.")
    )
    payload["markdown"] = build_ticker_lookup_markdown(payload)
    return payload


def get_ticker_lookup_settings(config: MarketPrepConfig | None) -> dict[str, Any]:
    settings = dict(DEFAULT_TICKER_LOOKUP_SETTINGS)
    if config is not None and isinstance(getattr(config, "ticker_lookup", None), dict):
        settings.update(config.ticker_lookup)
    return settings


def normalize_lookup_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def peer_tickers_for_lookup(
    ticker: str,
    metadata: dict[str, Any] | None = None,
    *,
    settings: dict[str, Any] | None = None,
) -> list[str]:
    symbol = normalize_lookup_ticker(ticker)
    settings = settings or DEFAULT_TICKER_LOOKUP_SETTINGS
    limit = max(0, _safe_int(settings.get("max_peer_tickers"), 8))
    peers: list[str] = []
    for peer in TICKER_PEER_OVERRIDES.get(symbol, []):
        _append_unique_peer(peers, peer, symbol)

    metadata = metadata if isinstance(metadata, dict) else {}
    industry_text = f"{metadata.get('industry') or ''} {metadata.get('sector') or ''}".lower()
    for keyword, keyword_peers in INDUSTRY_PEER_TICKERS.items():
        if keyword not in industry_text:
            continue
        for peer in keyword_peers:
            _append_unique_peer(peers, peer, symbol)

    return peers[:limit]


def peer_reason(metadata: dict[str, Any] | None) -> str:
    metadata = metadata if isinstance(metadata, dict) else {}
    industry = str(metadata.get("industry") or "").strip()
    sector = str(metadata.get("sector") or "").strip()
    if industry and sector:
        return f"{industry} / {sector}"
    return industry or sector or "Static peer map"


def split_lookup_headlines(
    ticker: str,
    peer_tickers: list[str],
    headlines: list[dict[str, Any]],
    *,
    target_terms: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbol = normalize_lookup_ticker(ticker)
    peers = {normalize_lookup_ticker(peer) for peer in peer_tickers}
    target_tokens = {term.upper() for term in (target_terms or [symbol]) if str(term).strip()}
    target_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    for row in headlines:
        text = f"{row.get('title') or ''} {row.get('query') or ''}".upper()
        if any(term and term in text for term in target_tokens):
            target_rows.append(row)
        elif any(peer and peer in text for peer in peers):
            industry_rows.append(row)
        else:
            industry_rows.append(row)
    return target_rows, industry_rows


def target_terms_for_lookup(ticker: str, metadata: dict[str, Any] | None) -> list[str]:
    symbol = normalize_lookup_ticker(ticker)
    metadata = metadata if isinstance(metadata, dict) else {}
    terms = [symbol]
    company = str(metadata.get("company_name") or metadata.get("short_name") or "").strip()
    if company:
        terms.append(company)
        cleaned = company.replace(",", " ").replace(".", " ")
        words = [word for word in cleaned.split() if len(word) > 1]
        stop_words = {"INC", "CORP", "CO", "LTD", "PLC", "SA", "ADR", "THE", "COMPANY", "LIMITED"}
        core_words = [word for word in words if word.upper() not in stop_words]
        if len(core_words) >= 2:
            terms.append(" ".join(core_words[:2]))
        if core_words:
            terms.append(core_words[0])
    return _dedupe_text([term for term in terms if len(str(term).strip()) >= 2])


def rank_landmine_headlines(headlines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in headlines:
        if not isinstance(row, dict):
            continue
        text = " ".join(
            str(value or "")
            for value in (
                row.get("title"),
                row.get("summary"),
                row.get("query"),
                " ".join(row.get("tags") or []),
            )
        ).lower()
        buckets = []
        score = 0
        for bucket, keywords in LANDMINE_KEYWORD_BUCKETS.items():
            hits = [keyword for keyword in keywords if keyword in text]
            if not hits:
                continue
            buckets.append(bucket)
            score += 2 + min(3, len(hits))
        if not buckets:
            continue
        enriched = dict(row)
        enriched["landmine_tags"] = buckets
        enriched["landmine_score"] = score
        ranked.append(enriched)
    ranked.sort(key=lambda row: int(row.get("landmine_score") or 0), reverse=True)
    return ranked[:30]


def build_source_status(*payloads: dict[str, Any]) -> list[str]:
    statuses: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        source = str(payload.get("source") or "source").strip()
        status = str(payload.get("status_label") or payload.get("status") or payload.get("message") or "").strip()
        if status:
            _append_unique_text(statuses, f"{source}: {status}")
        nested = payload.get("yfinance_status")
        if isinstance(nested, dict):
            nested_status = str(nested.get("status_label") or nested.get("status") or "").strip()
            if nested_status:
                _append_unique_text(statuses, f"yfinance: {nested_status}")
        warnings = payload.get("warnings")
        if isinstance(warnings, list) and warnings:
            _append_unique_text(statuses, f"{source} warning: {warnings[0]}")
    return statuses


def build_ticker_lookup_markdown(payload: dict[str, Any]) -> str:
    ticker = str(payload.get("ticker") or "").strip().upper()
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    lines = [
        f"# Ticker Lookup - {ticker}",
        "",
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        f"Window: next {payload.get('window_days') or 'n/a'} day(s)",
        "",
        "## Company",
        "",
        f"- Name: {metadata.get('company_name') or 'n/a'}",
        f"- Sector: {metadata.get('sector') or 'n/a'}",
        f"- Industry: {metadata.get('industry') or 'n/a'}",
        f"- Market cap: {metadata.get('market_cap_fmt') or 'n/a'}",
        f"- Peer context: {payload.get('peer_reason') or 'n/a'}",
        f"- Peer tickers: {', '.join(payload.get('peer_tickers') or []) or 'None'}",
        "",
        "## Nearby Earnings",
        "",
    ]
    target_earnings = payload.get("target_earnings") if isinstance(payload.get("target_earnings"), list) else []
    lines.extend(_earnings_lines(target_earnings) or ["No nearby earnings found for this ticker."])
    lines.extend(["", "## Major Peer Earnings / Events", ""])
    peer_earnings = payload.get("peer_earnings") if isinstance(payload.get("peer_earnings"), list) else []
    lines.extend(_earnings_lines(peer_earnings[:20]) or ["No nearby peer earnings found."])
    lines.extend(["", "## SEC / Announcements", ""])
    sec_payload = payload.get("sec_filings") if isinstance(payload.get("sec_filings"), dict) else {}
    sec_rows = sec_payload.get("filings") if isinstance(sec_payload.get("filings"), list) else []
    lines.extend(_sec_lines([row for row in sec_rows if isinstance(row, dict)]) or [str(sec_payload.get("message") or "No SEC filing risk found.")])
    lines.extend(["", "## Landmine / Exposure Headlines", ""])
    landmine_headlines = payload.get("landmine_headlines") if isinstance(payload.get("landmine_headlines"), list) else []
    lines.extend(_headline_lines(landmine_headlines[:20]) or ["No landmine-tagged headlines found."])
    lines.extend(["", "## Ticker Headlines", ""])
    target_headlines = payload.get("target_headlines") if isinstance(payload.get("target_headlines"), list) else []
    lines.extend(_headline_lines(target_headlines[:15]) or ["No ticker-specific headlines found."])
    lines.extend(["", "## Industry / Big Player Headlines", ""])
    industry_headlines = payload.get("industry_headlines") if isinstance(payload.get("industry_headlines"), list) else []
    lines.extend(_headline_lines(industry_headlines[:25]) or ["No industry headlines found."])
    lines.extend(["", "## AI Brief", ""])
    ai_brief = payload.get("ai_brief") if isinstance(payload.get("ai_brief"), dict) else {}
    ai_summary = str(ai_brief.get("summary") or "").strip()
    if ai_summary:
        lines.append(ai_summary)
    elif ai_brief:
        lines.append(str(ai_brief.get("status_label") or ai_brief.get("status") or "AI brief unavailable."))
    else:
        lines.append("AI brief not generated yet.")
    lines.extend(["", "## AI Swing Query", ""])
    lines.append(str(payload.get("ai_swing_query") or build_ai_swing_query(payload)).strip())
    lines.extend(["", "## Source Status", ""])
    statuses = payload.get("source_status") if isinstance(payload.get("source_status"), list) else []
    lines.extend([f"- {status}" for status in statuses] or ["No source status available."])
    return "\n".join(lines).rstrip() + "\n"


def build_ai_swing_query(payload: dict[str, Any]) -> str:
    ticker = str(payload.get("ticker") or "").strip().upper() or "THIS STOCK"
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    peers = ", ".join(payload.get("peer_tickers") or []) or "n/a"
    return (
        f"Using the Market Prep context below for {ticker}, give a brief swing-trade read plus a ticker landmine and swing-risk read. "
        "Focus on catalysts, roadblocks, hidden exposure, strategic stakes/investments, customers/suppliers, "
        "financing/dilution, legal/regulatory risk, timing risk over the lookup window, and trade-management reminders. "
        "Separate confirmed facts from speculation and explicitly list missing facts to verify.\n\n"
        f"Ticker: {ticker}\n"
        f"Company: {metadata.get('company_name') or 'n/a'}\n"
        f"Sector/industry: {metadata.get('sector') or 'n/a'} / {metadata.get('industry') or 'n/a'}\n"
        f"Market cap: {metadata.get('market_cap_fmt') or 'n/a'}\n"
        f"Lookup window: next {payload.get('window_days') or 'n/a'} day(s)\n"
        f"Peer context: {payload.get('peer_reason') or 'n/a'}\n"
        f"Peer tickers: {peers}"
    )


def _news_lookup_config(
    config: MarketPrepConfig,
    ticker: str,
    metadata: dict[str, Any],
    peer_tickers: list[str],
    settings: dict[str, Any],
) -> MarketPrepConfig:
    queries = list(settings.get("queries") if isinstance(settings.get("queries"), list) else [])
    company = str(metadata.get("company_name") or "").strip()
    industry = str(metadata.get("industry") or "").strip()
    sector = str(metadata.get("sector") or "").strip()
    if company:
        queries.extend(
            [
                f"{company} earnings",
                f"{company} guidance",
                f"{company} analyst rating",
                f"{company} price target",
                f"{company} conference",
                f"{company} announces",
                f"{company} catalyst",
                f"{company} strategic investment",
                f"{company} investment",
                f"{company} stake",
                f"{company} ownership",
                f"{company} portfolio",
                f"{company} subsidiary",
                f"{company} joint venture",
                f"{company} partnership",
                f"{company} customer",
                f"{company} supplier",
                f"{company} contract",
                f"{company} revenue exposure",
                f"{company} financing",
                f"{company} debt",
                f"{company} lawsuit",
                f"{company} regulatory",
                f"{company} Anthropic",
                f"{company} OpenAI",
                f"{company} AI investment",
            ]
        )
    if bool(settings.get("include_industry_news", True)):
        if industry:
            queries.extend([f"{industry} news", f"{industry} earnings", f"{industry} demand", f"{industry} regulation"])
        if sector:
            queries.extend([f"{sector} sector news", f"{sector} analyst outlook"])
    for peer in peer_tickers[: max(0, _safe_int(settings.get("max_peer_tickers"), 8))]:
        queries.extend([f"{peer} earnings", f"{peer} conference", f"{peer} guidance", f"{peer} analyst rating"])

    news_settings = dict(config.google_news_rss if isinstance(config.google_news_rss, dict) else {})
    news_settings.update(
        {
            "enabled": True,
            "queries": _dedupe_text(queries),
            "max_watchlist_tickers": max(1, len(peer_tickers) + 1),
            "max_feeds": max(20, len(queries) + len(peer_tickers) + 4),
        }
    )
    return replace(config, google_news_rss=news_settings)


def _earnings_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        parts = [
            str(row.get("date") or "").strip(),
            str(row.get("ticker") or "").strip().upper(),
            str(row.get("company") or row.get("company_yfinance") or "").strip(),
            str(row.get("time") or "").strip().upper(),
            str(row.get("importance") or "").strip().upper(),
            str(row.get("market_cap_fmt") or "").strip(),
            str(row.get("notes") or "").strip(),
        ]
        lines.append("- " + " | ".join(part if part else "n/a" for part in parts))
    return lines


def _sec_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows[:20]:
        parts = [
            str(row.get("ticker") or "").strip().upper(),
            str(row.get("form") or "").strip(),
            str(row.get("filing_date") or "").strip(),
            str(row.get("risk_classification") or "").strip().upper(),
            ", ".join(row.get("matched_keywords") or []),
            str(row.get("url") or "").strip(),
        ]
        lines.append("- " + " | ".join(part if part else "n/a" for part in parts))
    return lines


def _headline_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        tags = ", ".join(row.get("tags") or [])
        prefix = f"[{tags}] " if tags else ""
        parts = [
            prefix + str(row.get("title") or "").strip(),
            str(row.get("source") or "").strip(),
            str(row.get("query") or "").strip(),
            str(row.get("published") or "").strip(),
            str(row.get("url") or "").strip(),
        ]
        lines.append("- " + " | ".join(part for part in parts if part))
    return lines


def _append_unique_peer(peers: list[str], value: Any, ticker: str) -> None:
    peer = normalize_lookup_ticker(value)
    if peer and peer != ticker and peer not in peers:
        peers.append(peer)


def _append_unique_text(values: list[str], value: str) -> None:
    text = str(value or "").strip()
    if text and text not in values:
        values.append(text)


def _dedupe_text(values: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            deduped.append(text)
    return deduped


def _disabled_payload(source: str, message: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "status": "disabled",
        "status_label": "Disabled",
        "message": message,
        "warnings": [],
    }


def _safe_int(value: Any, default: int | None = None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default or 0)
