from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime
from email.utils import parsedate_to_datetime
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
    "news_limit": 40,
    "headline_lookback_days": 14,
    "max_peer_tickers": 12,
    "include_sec_filings": True,
    "include_peer_earnings": True,
    "include_industry_news": True,
    "include_ai_brief": True,
    "queries": [
        "{ticker} earnings date",
        "{ticker} reports earnings",
        "{ticker} guidance",
        "{ticker} raises guidance",
        "{ticker} cuts guidance",
        "{ticker} announces",
        "{ticker} investor day",
        "{ticker} analyst day",
        "{ticker} acquisition",
        "{ticker} merger",
        "{ticker} partnership",
        "{ticker} contract",
        "{ticker} product launch",
        "{ticker} regulatory approval",
        "{ticker} lawsuit",
        "{ticker} investigation",
        "{ticker} offering",
        "{ticker} financing",
        "{ticker} debt",
        "{ticker} downgrade",
        "{ticker} strategic investment",
        "{ticker} stake",
        "{ticker} joint venture",
        "{ticker} customer",
        "{ticker} supplier",
        "{ticker} revenue exposure",
        "{ticker} AI investment",
        "{ticker} Anthropic",
        "{ticker} OpenAI",
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

CATALYST_KEYWORD_BUCKETS = {
    "earnings/guidance": ("earnings", "guidance", "eps", "revenue", "profit warning", "preannounce", "outlook"),
    "conference/investor event": (
        "conference",
        "investor day",
        "analyst day",
        "presentation",
        "fireside chat",
        "capital markets day",
    ),
    "product/customer catalyst": (
        "product launch",
        "launches",
        "shipment",
        "contract",
        "customer",
        "supplier",
        "partnership",
        "deal",
    ),
    "rating/price target": ("downgrade", "upgrade", "rating", "price target", "initiates", "cuts target"),
    "financing/legal/regulatory": (
        "offering",
        "atm",
        "shelf",
        "convertible",
        "lawsuit",
        "investigation",
        "regulatory",
        "probe",
    ),
    "macro/geopolitical exposure": ("tariff", "sanction", "export control", "china", "taiwan", "iran", "oil"),
}

MATERIAL_HEADLINE_BUCKETS = {
    "earnings/guidance": (
        "earnings date",
        "reports earnings",
        "reported earnings",
        "earnings results",
        "guidance",
        "raises guidance",
        "cuts guidance",
        "lowers guidance",
        "preannounce",
        "profit warning",
        "revenue warning",
    ),
    "SEC/financing": (
        "offering",
        "shelf",
        "atm",
        "at-the-market",
        "convertible",
        "debt",
        "credit facility",
        "refinancing",
        "liquidity",
        "bankruptcy",
        "restructuring",
    ),
    "legal/regulatory": (
        "lawsuit",
        "sues",
        "settlement",
        "probe",
        "investigation",
        "regulator",
        "regulatory",
        "sec",
        "doj",
        "ftc",
        "antitrust",
        "approval",
        "rejection",
        "recall",
    ),
    "M&A/strategic": (
        "merger",
        "acquisition",
        "acquires",
        "buyout",
        "takeover",
        "strategic alternatives",
        "strategic review",
        "joint venture",
        "stake",
        "ownership",
    ),
    "customer/contract": (
        "contract",
        "customer",
        "supplier",
        "supply agreement",
        "partnership",
        "partner",
        "deal",
        "order",
    ),
    "product/operations": (
        "product launch",
        "launches",
        "unveils",
        "shipment",
        "deliveries",
        "production halt",
        "halts production",
        "factory",
        "plant",
    ),
    "investor event": (
        "investor day",
        "analyst day",
        "capital markets day",
        "to present",
        "presentation",
        "conference",
        "fireside chat",
    ),
    "analyst action": (
        "downgrade",
        "downgrades",
        "upgrades",
        "upgrade",
        "cuts target",
        "raises target",
        "initiates",
        "resumes coverage",
    ),
    "AI/private exposure": (
        "anthropic",
        "openai",
        "ai investment",
        "artificial intelligence investment",
        "private company",
    ),
    "macro/geopolitical exposure": (
        "tariff",
        "sanction",
        "export control",
        "china",
        "taiwan",
        "trade restriction",
    ),
}

CONFIRMED_HEADLINE_ACTION_TERMS = (
    "announces",
    "announced",
    "reports",
    "reported",
    "posts",
    "posted",
    "files",
    "filed",
    "raises",
    "raised",
    "cuts",
    "cut",
    "lowers",
    "lowered",
    "launches",
    "launched",
    "unveils",
    "unveiled",
    "wins",
    "won",
    "receives",
    "received",
    "secures",
    "secured",
    "signs",
    "signed",
    "partners",
    "partnered",
    "acquires",
    "acquired",
    "merges",
    "merged",
    "approves",
    "approved",
    "rejects",
    "rejected",
    "sues",
    "sued",
    "settles",
    "settled",
    "downgrades",
    "downgraded",
    "upgrades",
    "upgraded",
    "initiates",
    "to present",
    "will present",
    "scheduled",
    "sets date",
    "expands",
    "expanded",
)
SPECULATIVE_HEADLINE_PHRASES = (
    "could",
    "may",
    "might",
    "would",
    "should",
    "rumor",
    "rumour",
    "speculation",
    "speculative",
    "prediction",
    "forecast",
    "expected to",
    "expects to",
    "analysts expect",
    "what to expect",
    "ahead of",
    "preview",
    "set to",
    "poised to",
    "looks to",
    "seeks to",
)
LOW_SIGNAL_HEADLINE_PHRASES = (
    "stock to watch",
    "stocks to watch",
    "why shares",
    "why the stock",
    "why stock",
    "is it time",
    "should you buy",
    "buy or sell",
    "market chatter",
    "options traders",
    "earnings preview",
    "conference preview",
    "transcript",
    "recap",
    "last month",
    "last week",
)
HARD_MATERIAL_TAGS = {
    "earnings/guidance",
    "SEC/financing",
    "legal/regulatory",
    "M&A/strategic",
    "customer/contract",
    "product/operations",
    "AI/private exposure",
    "macro/geopolitical exposure",
}
HARD_AVOID_TAGS = {"financing/dilution", "debt/credit", "legal/regulatory"}
STRONG_CAUTION_TAGS = {"earnings/guidance", "M&A", "geopolitical", "customer/supplier", "AI/private exposure"}
MAJOR_EARNINGS_IMPORTANCE = {"MEGA", "HIGH"}


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
    headline_lookback_days = max(1, _safe_int(settings.get("headline_lookback_days"), 14))
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
    target_headlines = focus_lookup_headlines(
        target_headlines,
        reference_date=start,
        lookback_days=headline_lookback_days,
    )
    industry_headlines = focus_lookup_headlines(
        industry_headlines,
        reference_date=start,
        lookback_days=headline_lookback_days,
    )
    landmine_headlines = rank_landmine_headlines(
        target_headlines + industry_headlines,
        reference_date=start,
        lookback_days=headline_lookback_days,
    )

    payload = {
        "report_type": "ticker_lookup",
        "ticker": symbol,
        "report_date": start.isoformat(),
        "generated_at": generated_at,
        "window_days": window_days,
        "headline_lookback_days": headline_lookback_days,
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
    payload["swing_risk"] = build_swing_risk_assessment(payload)
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


def focus_lookup_headlines(
    headlines: list[dict[str, Any]],
    *,
    reference_date: date | None = None,
    lookback_days: int = 14,
) -> list[dict[str, Any]]:
    focused: list[dict[str, Any]] = []
    for row in headlines:
        if not isinstance(row, dict):
            continue
        if not is_major_lookup_headline(row, reference_date=reference_date, lookback_days=lookback_days):
            continue
        enriched = dict(row)
        enriched["material_tags"] = material_tags_for_headline(row)
        focused.append(enriched)
    focused.sort(
        key=lambda row: (
            -_major_headline_score(row),
            _headline_age_days(row, reference_date),
            str(row.get("title") or ""),
        )
    )
    return focused


def is_major_lookup_headline(
    row: dict[str, Any],
    *,
    reference_date: date | None = None,
    lookback_days: int = 14,
) -> bool:
    if not _is_recent_headline(row, reference_date=reference_date, lookback_days=lookback_days):
        return False
    tags = material_tags_for_headline(row)
    if not tags:
        return False
    text = _headline_text(row)
    tag_set = set(tags)
    if _is_speculative_headline(text) and not _has_confirmed_headline_action(text):
        return False
    if _is_low_signal_headline(text) and not (tag_set & HARD_MATERIAL_TAGS):
        return False
    return True


def material_tags_for_headline(row: dict[str, Any]) -> list[str]:
    text = _headline_text(row)
    tags = []
    for bucket, keywords in MATERIAL_HEADLINE_BUCKETS.items():
        if _keyword_hits(text, keywords):
            tags.append(bucket)
    return tags


def rank_landmine_headlines(
    headlines: list[dict[str, Any]],
    *,
    reference_date: date | None = None,
    lookback_days: int = 14,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in headlines:
        if not isinstance(row, dict):
            continue
        if reference_date is not None and not is_major_lookup_headline(
            row,
            reference_date=reference_date,
            lookback_days=lookback_days,
        ):
            continue
        text = " ".join(
            str(value or "")
            for value in (
                row.get("title"),
                row.get("summary"),
                row.get("query"),
                " ".join(_text_list(row.get("tags"))),
            )
        ).lower()
        buckets = []
        score = 0
        for bucket, keywords in LANDMINE_KEYWORD_BUCKETS.items():
            hits = _keyword_hits(text, keywords)
            if not hits:
                continue
            buckets.append(bucket)
            score += 2 + min(3, len(hits))
        if not buckets:
            continue
        enriched = dict(row)
        enriched["landmine_tags"] = buckets
        enriched["material_tags"] = material_tags_for_headline(row)
        enriched["landmine_score"] = score + _major_headline_score(row)
        ranked.append(enriched)
    ranked.sort(
        key=lambda row: (
            -int(row.get("landmine_score") or 0),
            _headline_age_days(row, reference_date),
            str(row.get("title") or ""),
        )
    )
    return ranked[:30]


def build_swing_risk_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    ticker = normalize_lookup_ticker(payload.get("ticker"))
    report_day = _parse_date(payload.get("report_date")) or datetime.now().date()
    window_days = max(1, _safe_int(payload.get("window_days"), 10))
    headline_lookback_days = max(1, _safe_int(payload.get("headline_lookback_days"), 14))
    score = 0
    risk_items: list[dict[str, Any]] = []
    catalysts: list[dict[str, Any]] = []
    missing_checks: list[str] = []

    target_earnings = payload.get("target_earnings") if isinstance(payload.get("target_earnings"), list) else []
    for row in target_earnings:
        if not isinstance(row, dict):
            continue
        event_day = _parse_date(row.get("date"))
        days_until = (event_day - report_day).days if event_day else None
        importance = str(row.get("importance") or "").upper()
        severity = "HIGH" if importance in MAJOR_EARNINGS_IMPORTANCE else "MEDIUM"
        points = 45 if importance in MAJOR_EARNINGS_IMPORTANCE else 28
        if days_until is not None and days_until <= 3:
            points += 18
            severity = "HIGH"
        score += points
        item = {
            "severity": severity,
            "category": "Ticker earnings",
            "date": str(row.get("date") or ""),
            "title": _earnings_title(row),
            "source": "earnings calendar",
            "reason": "Known earnings/guidance timing risk inside the lookup window.",
            "days_until": days_until,
        }
        risk_items.append(item)
        catalysts.append(item)

    peer_earnings = payload.get("peer_earnings") if isinstance(payload.get("peer_earnings"), list) else []
    for row in peer_earnings:
        if not isinstance(row, dict):
            continue
        importance = str(row.get("importance") or "").upper()
        if importance not in {"MEGA", "HIGH", "MEDIUM"}:
            continue
        points = 18 if importance in MAJOR_EARNINGS_IMPORTANCE else 10
        score += points
        item = {
            "severity": "MEDIUM" if importance == "MEDIUM" else "HIGH",
            "category": "Peer earnings",
            "date": str(row.get("date") or ""),
            "title": _earnings_title(row),
            "source": "earnings calendar",
            "reason": "Major peer/big-player earnings can move the group.",
            "days_until": _days_until(row.get("date"), report_day),
        }
        risk_items.append(item)
        catalysts.append(item)

    sec_payload = payload.get("sec_filings") if isinstance(payload.get("sec_filings"), dict) else {}
    sec_rows = sec_payload.get("filings") if isinstance(sec_payload.get("filings"), list) else []
    for row in sec_rows:
        if not isinstance(row, dict):
            continue
        risk = str(row.get("risk_classification") or "").upper()
        if risk not in {"HIGH", "MEDIUM"}:
            continue
        points = 45 if risk == "HIGH" else 24
        score += points
        risk_items.append(
            {
                "severity": risk,
                "category": "SEC filing",
                "date": str(row.get("filing_date") or ""),
                "title": f"{ticker} {row.get('form') or ''} filing".strip(),
                "source": "SEC EDGAR",
                "reason": ", ".join(row.get("matched_keywords") or []) or "Potential filing risk.",
                "url": str(row.get("url") or ""),
            }
        )

    landmine_rows = payload.get("landmine_headlines") if isinstance(payload.get("landmine_headlines"), list) else []
    for row in landmine_rows:
        if not isinstance(row, dict):
            continue
        if not is_major_lookup_headline(
            row,
            reference_date=report_day,
            lookback_days=headline_lookback_days,
        ):
            continue
        tags = [str(tag) for tag in row.get("landmine_tags") or []]
        tag_set = set(tags)
        severity = "HIGH" if tag_set & HARD_AVOID_TAGS else "MEDIUM" if tag_set & STRONG_CAUTION_TAGS else "LOW"
        points = 30 if severity == "HIGH" else 16 if severity == "MEDIUM" else 6
        score += points
        item = {
            "severity": severity,
            "category": "Headline road bump",
            "date": str(row.get("published") or ""),
            "title": str(row.get("title") or ""),
            "source": str(row.get("source") or row.get("query") or "news"),
            "reason": ", ".join(tags) or "Landmine keyword match.",
            "url": str(row.get("url") or ""),
        }
        risk_items.append(item)
        if _is_catalyst_headline(row):
            catalysts.append(item)

    target_headline_rows = payload.get("target_headlines") if isinstance(payload.get("target_headlines"), list) else []
    for row in target_headline_rows:
        if not isinstance(row, dict):
            continue
        if not is_major_lookup_headline(
            row,
            reference_date=report_day,
            lookback_days=headline_lookback_days,
        ):
            continue
        catalyst_tags = catalyst_tags_for_headline(row)
        if not catalyst_tags:
            continue
        severity = "MEDIUM" if any(tag in {"earnings/guidance", "financing/legal/regulatory"} for tag in catalyst_tags) else "LOW"
        score += 12 if severity == "MEDIUM" else 6
        item = {
            "severity": severity,
            "category": "Possible upcoming event",
            "date": str(row.get("published") or ""),
            "title": str(row.get("title") or ""),
            "source": str(row.get("source") or row.get("query") or "news"),
            "reason": ", ".join(catalyst_tags),
            "url": str(row.get("url") or ""),
        }
        risk_items.append(item)
        catalysts.append(item)

    if not target_earnings:
        missing_checks.append(f"Verify {ticker} earnings date manually before holding through the full swing window.")
    if not sec_rows:
        missing_checks.append("No SEC rows found in this lookup; verify recent filings if position size is meaningful.")
    if not payload.get("target_headlines"):
        missing_checks.append("No ticker-specific headlines found; verify with broker/news terminal before assuming clean.")

    risk_items = _dedupe_risk_items(risk_items)
    catalysts = _dedupe_risk_items(catalysts)
    verdict = _swing_verdict(score, risk_items)
    confidence = _swing_confidence(payload, risk_items)
    return {
        "verdict": verdict,
        "risk_score": min(100, int(score)),
        "confidence": confidence,
        "summary": _swing_summary(verdict, risk_items, catalysts),
        "risk_items": risk_items[:20],
        "upcoming_catalysts": catalysts[:15],
        "missing_checks": missing_checks[:8],
    }


def catalyst_tags_for_headline(row: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(value or "")
        for value in (
            row.get("title"),
            row.get("summary"),
            row.get("query"),
            " ".join(_text_list(row.get("tags"))),
        )
    ).lower()
    tags = []
    for bucket, keywords in CATALYST_KEYWORD_BUCKETS.items():
        if _keyword_hits(text, keywords):
            tags.append(bucket)
    return tags


def _is_catalyst_headline(row: dict[str, Any]) -> bool:
    return bool(catalyst_tags_for_headline(row))


def _swing_verdict(score: int, risk_items: list[dict[str, Any]]) -> str:
    if any(str(item.get("severity") or "").upper() == "HIGH" for item in risk_items):
        return "AVOID / WAIT"
    if score >= 45:
        return "AVOID / WAIT"
    if score >= 18 or risk_items:
        return "CAUTION"
    return "CLEAN"


def _swing_confidence(payload: dict[str, Any], risk_items: list[dict[str, Any]]) -> str:
    source_count = 0
    if isinstance(payload.get("earnings_payload"), dict):
        source_count += 1
    if isinstance(payload.get("sec_filings"), dict):
        source_count += 1
    if isinstance(payload.get("news_headlines"), dict):
        source_count += 1
    if source_count >= 3 and risk_items:
        return "HIGH"
    if source_count >= 2:
        return "MEDIUM"
    return "LOW"


def _swing_summary(verdict: str, risk_items: list[dict[str, Any]], catalysts: list[dict[str, Any]]) -> str:
    if verdict == "CLEAN":
        return "No obvious scheduled or headline road bumps found in configured sources."
    top = risk_items[0] if risk_items else catalysts[0] if catalysts else {}
    title = str(top.get("title") or top.get("category") or "road bump").strip()
    reason = str(top.get("reason") or "").strip()
    detail = f": {title}" if title else ""
    if reason:
        detail += f" ({reason})"
    return f"{verdict}{detail}"


def _dedupe_risk_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped: list[dict[str, Any]] = []
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    for row in sorted(
        rows,
        key=lambda item: (
            severity_rank.get(str(item.get("severity") or "").upper(), 9),
            str(item.get("date") or ""),
            str(item.get("title") or ""),
        ),
    ):
        key = (
            str(row.get("category") or "").lower(),
            str(row.get("title") or "").lower(),
            str(row.get("date") or "")[:10],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def _days_until(value: Any, reference: date) -> int | None:
    parsed = _parse_date(value)
    return (parsed - reference).days if parsed else None


def _earnings_title(row: dict[str, Any]) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    company = str(row.get("company") or row.get("company_yfinance") or "").strip()
    when = " ".join(str(row.get(key) or "").strip() for key in ("date", "time")).strip()
    importance = str(row.get("importance") or "").strip().upper()
    parts = [part for part in (ticker, company, when, importance) if part]
    return " | ".join(parts)


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
        f"Headline focus: recent major announcements/news, last {payload.get('headline_lookback_days') or 'n/a'} day(s)",
        "",
        "## Swing Safety",
        "",
    ]
    lines.extend(_swing_risk_lines(payload.get("swing_risk") if isinstance(payload.get("swing_risk"), dict) else {}))
    lines.extend(
        [
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
    )
    target_earnings = payload.get("target_earnings") if isinstance(payload.get("target_earnings"), list) else []
    lines.extend(_earnings_lines(target_earnings) or ["No nearby earnings found for this ticker."])
    lines.extend(["", "## Major Peer Earnings / Events", ""])
    peer_earnings = payload.get("peer_earnings") if isinstance(payload.get("peer_earnings"), list) else []
    lines.extend(_earnings_lines(peer_earnings[:20]) or ["No nearby peer earnings found."])
    lines.extend(["", "## SEC / Announcements", ""])
    sec_payload = payload.get("sec_filings") if isinstance(payload.get("sec_filings"), dict) else {}
    sec_rows = sec_payload.get("filings") if isinstance(sec_payload.get("filings"), list) else []
    lines.extend(_sec_lines([row for row in sec_rows if isinstance(row, dict)]) or [str(sec_payload.get("message") or "No SEC filing risk found.")])
    lines.extend(["", "## Major Landmine / Exposure Headlines", ""])
    landmine_headlines = payload.get("landmine_headlines") if isinstance(payload.get("landmine_headlines"), list) else []
    lines.extend(_headline_lines(landmine_headlines[:20]) or ["No recent major landmine-tagged headlines found."])
    lines.extend(["", "## Major Ticker Headlines", ""])
    target_headlines = payload.get("target_headlines") if isinstance(payload.get("target_headlines"), list) else []
    lines.extend(_headline_lines(target_headlines[:15]) or ["No recent major ticker-specific headlines found."])
    lines.extend(["", "## Major Industry / Big Player Headlines", ""])
    industry_headlines = payload.get("industry_headlines") if isinstance(payload.get("industry_headlines"), list) else []
    lines.extend(_headline_lines(industry_headlines[:25]) or ["No recent major industry headlines found."])
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
    swing_risk = payload.get("swing_risk") if isinstance(payload.get("swing_risk"), dict) else {}
    top_risks = swing_risk.get("risk_items") if isinstance(swing_risk.get("risk_items"), list) else []
    top_catalysts = swing_risk.get("upcoming_catalysts") if isinstance(swing_risk.get("upcoming_catalysts"), list) else []
    return (
        f"Using the Market Prep context below for {ticker}, decide whether it looks CLEAN, CAUTION, or AVOID/WAIT for a swing. "
        "Focus on scheduled catalysts, confirmed roadblocks, hidden exposure, strategic stakes/investments, customers/suppliers, "
        "conferences/investor days, financing/dilution, legal/regulatory risk, rating/guidance risk, and timing risk over the lookup window. "
        "Ignore stale or speculative headlines unless they point to a confirmed material announcement, and explicitly list missing facts to verify.\n\n"
        f"Ticker: {ticker}\n"
        f"Company: {metadata.get('company_name') or 'n/a'}\n"
        f"Sector/industry: {metadata.get('sector') or 'n/a'} / {metadata.get('industry') or 'n/a'}\n"
        f"Market cap: {metadata.get('market_cap_fmt') or 'n/a'}\n"
        f"Lookup window: next {payload.get('window_days') or 'n/a'} day(s)\n"
        f"Peer context: {payload.get('peer_reason') or 'n/a'}\n"
        f"Peer tickers: {peers}\n"
        f"Deterministic verdict: {swing_risk.get('verdict') or 'n/a'} | score={swing_risk.get('risk_score') or 0} | confidence={swing_risk.get('confidence') or 'n/a'}\n"
        f"Top road bumps: {_compact_risk_items(top_risks[:5])}\n"
        f"Upcoming catalysts: {_compact_risk_items(top_catalysts[:5])}"
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
                f"{company} earnings date",
                f"{company} reports earnings",
                f"{company} guidance",
                f"{company} raises guidance",
                f"{company} cuts guidance",
                f"{company} announces",
                f"{company} analyst day",
                f"{company} investor day",
                f"{company} acquisition",
                f"{company} merger",
                f"{company} strategic investment",
                f"{company} stake",
                f"{company} joint venture",
                f"{company} partnership",
                f"{company} customer",
                f"{company} supplier",
                f"{company} contract",
                f"{company} revenue exposure",
                f"{company} regulatory approval",
                f"{company} financing",
                f"{company} debt",
                f"{company} lawsuit",
                f"{company} investigation",
                f"{company} Anthropic",
                f"{company} OpenAI",
                f"{company} AI investment",
            ]
        )
    if bool(settings.get("include_industry_news", True)):
        if industry:
            queries.extend([f"{industry} major news", f"{industry} earnings", f"{industry} regulation"])
        if sector:
            queries.extend([f"{sector} sector major news", f"{sector} sector regulation"])
    for peer in peer_tickers[: max(0, _safe_int(settings.get("max_peer_tickers"), 8))]:
        queries.extend(
            [
                f"{peer} earnings",
                f"{peer} earnings date",
                f"{peer} guidance",
                f"{peer} announces",
                f"{peer} acquisition",
                f"{peer} lawsuit",
                f"{peer} offering",
            ]
        )

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
        tags = ", ".join(
            _dedupe_text(
                _text_list(row.get("landmine_tags"))
                + _text_list(row.get("material_tags"))
                + _text_list(row.get("tags"))
            )
        )
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


def _swing_risk_lines(risk: dict[str, Any]) -> list[str]:
    if not risk:
        return ["No swing risk assessment generated."]
    lines = [
        f"- Verdict: {risk.get('verdict') or 'n/a'}",
        f"- Risk score: {risk.get('risk_score', 0)}/100",
        f"- Confidence: {risk.get('confidence') or 'n/a'}",
        f"- Read: {risk.get('summary') or 'n/a'}",
        "",
        "Road bumps:",
    ]
    risk_items = risk.get("risk_items") if isinstance(risk.get("risk_items"), list) else []
    if risk_items:
        lines.extend(_risk_item_line(item) for item in risk_items[:12])
    else:
        lines.append("- None found in configured sources.")
    lines.extend(["", "Upcoming catalysts / things to verify:"])
    catalysts = risk.get("upcoming_catalysts") if isinstance(risk.get("upcoming_catalysts"), list) else []
    if catalysts:
        lines.extend(_risk_item_line(item) for item in catalysts[:10])
    else:
        lines.append("- No explicit catalyst headlines or calendar events found.")
    missing = risk.get("missing_checks") if isinstance(risk.get("missing_checks"), list) else []
    if missing:
        lines.extend(["", "Missing checks:"])
        lines.extend(f"- {item}" for item in missing)
    return lines


def _risk_item_line(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("severity") or "").upper(),
        str(item.get("category") or "").strip(),
        str(item.get("date") or "").strip(),
        str(item.get("title") or "").strip(),
        str(item.get("reason") or "").strip(),
    ]
    return "- " + " | ".join(part for part in parts if part)


def _compact_risk_items(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "None found"
    compact = []
    for row in rows:
        title = str(row.get("title") or row.get("category") or "").strip()
        severity = str(row.get("severity") or "").strip().upper()
        date_text = str(row.get("date") or "").strip()[:10]
        compact.append(" / ".join(part for part in (severity, date_text, title) if part))
    return "; ".join(compact)


def _headline_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(value or "")
        for value in (
            row.get("title"),
            row.get("summary"),
            row.get("query"),
            " ".join(_text_list(row.get("tags"))),
        )
    ).lower()


def _major_headline_score(row: dict[str, Any]) -> int:
    tags = material_tags_for_headline(row)
    hard_tag_count = len(set(tags) & HARD_MATERIAL_TAGS)
    score = 4 * hard_tag_count + 2 * max(0, len(tags) - hard_tag_count)
    text = _headline_text(row)
    if _has_confirmed_headline_action(text):
        score += 3
    if _is_speculative_headline(text):
        score -= 4
    if _is_low_signal_headline(text):
        score -= 3
    return score


def _is_recent_headline(
    row: dict[str, Any],
    *,
    reference_date: date | None,
    lookback_days: int,
) -> bool:
    published = _headline_date(row)
    if published is None or reference_date is None:
        return True
    age_days = (reference_date - published).days
    return -1 <= age_days <= max(1, lookback_days)


def _headline_age_days(row: dict[str, Any], reference_date: date | None) -> int:
    published = _headline_date(row)
    if published is None or reference_date is None:
        return 9999
    return max(-1, (reference_date - published).days)


def _headline_date(row: dict[str, Any]) -> date | None:
    value = row.get("published") if isinstance(row, dict) else None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized[:10] if len(normalized) >= 10 else normalized).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def _is_speculative_headline(text: str) -> bool:
    return any(_headline_phrase_present(text, phrase) for phrase in SPECULATIVE_HEADLINE_PHRASES)


def _is_low_signal_headline(text: str) -> bool:
    return any(phrase in text for phrase in LOW_SIGNAL_HEADLINE_PHRASES)


def _headline_phrase_present(text: str, phrase: str) -> bool:
    term = str(phrase or "").strip().lower()
    if not term:
        return False
    if term == "may":
        return bool(re.search(r"\bmay\b(?!\s+\d{1,2}\b)", text))
    if re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(rf"\b{re.escape(term)}\b", text))
    return term in text


def _has_confirmed_headline_action(text: str) -> bool:
    return bool(_keyword_hits(text, CONFIRMED_HEADLINE_ACTION_TERMS))


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for keyword in keywords:
        term = str(keyword or "").strip().lower()
        if not term:
            continue
        if re.fullmatch(r"[a-z0-9]+", term):
            if re.search(rf"\b{re.escape(term)}\b", text):
                hits.append(term)
        elif term in text:
            hits.append(term)
    return hits


def _append_unique_peer(peers: list[str], value: Any, ticker: str) -> None:
    peer = normalize_lookup_ticker(value)
    if peer and peer != ticker and peer not in peers:
        peers.append(peer)


def _append_unique_text(values: list[str], value: str) -> None:
    text = str(value or "").strip()
    if text and text not in values:
        values.append(text)


def _text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


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
