from __future__ import annotations

from datetime import datetime

from .models import MarketPrepConfig

KEY_WEEKLY_RISK_EVENT_TERMS = ("CPI", "FOMC", "NFP", "NONFARM PAYROLLS", "PCE")
MEGA_CAP_TICKERS = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "TSLA",
    "AVGO",
    "BRK.B",
    "BRK-B",
    "JPM",
    "LLY",
    "V",
    "MA",
    "NFLX",
}
SECTOR_ROTATION_TICKERS = ("XLK", "XLF", "XLE", "XLU", "XLV", "XLY", "XLP", "XLI", "XLC", "XLB", "SMH", "ARKK")


def build_placeholder_report(action: str, config: MarketPrepConfig) -> str:
    return (
        "Market Prep tab loaded. Phase 1 skeleton ready.\n\n"
        f"{action}: infrastructure placeholder.\n"
        f"Timezone: {config.timezone}\n"
        f"Market timezone: {config.market_timezone}"
    )


def build_market_snapshot_report(snapshot: dict) -> str:
    classification = snapshot.get("classification") if isinstance(snapshot, dict) else {}
    classification = classification if isinstance(classification, dict) else {}
    rows = snapshot.get("rows") if isinstance(snapshot, dict) else []
    rows = rows if isinstance(rows, list) else []
    errors = snapshot.get("errors") if isinstance(snapshot, dict) else []
    errors = errors if isinstance(errors, list) else []

    lines = [
        "Market Prep Daily Snapshot",
        "=" * 80,
        f"Generated at: {snapshot.get('generated_at') or 'n/a'}",
        f"Source: {snapshot.get('source') or 'n/a'}",
        f"Regime: {classification.get('label') or 'n/a'}",
        f"Reason: {classification.get('reason') or 'n/a'}",
        "",
        (
            f"{'Ticker':<8} {'Close':>10} {'1D%':>8} {'5D%':>8} {'20D%':>8} "
            f"{'21SMA':>8} {'50SMA':>8} {'Vol/20D':>9}"
        ),
        "-" * 80,
    ]

    if rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"{str(row.get('ticker') or ''):<8} "
                f"{_fmt_number(row.get('last_close')):>10} "
                f"{_fmt_pct(row.get('return_1d_pct')):>8} "
                f"{_fmt_pct(row.get('return_5d_pct')):>8} "
                f"{_fmt_pct(row.get('return_20d_pct')):>8} "
                f"{_fmt_above(row.get('above_21_sma')):>8} "
                f"{_fmt_above(row.get('above_50_sma')):>8} "
                f"{_fmt_pct(row.get('volume_vs_20d_avg_pct')):>9}"
            )
    else:
        lines.append("No market snapshot rows available.")

    if errors:
        lines.extend(["", "Data Notes", "-" * 80])
        for error in errors[:12]:
            lines.append(f"- {error}")
        if len(errors) > 12:
            lines.append(f"- ... {len(errors) - 12} more")

    return "\n".join(lines).rstrip()


def build_daily_prep_report(
    todays_events: dict,
    next_7_events: dict | None = None,
    today_tomorrow_earnings: dict | None = None,
    next_7_earnings: dict | None = None,
    watchlist_risk: dict | None = None,
    rss_headlines: dict | None = None,
    youtube_links: dict | None = None,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
) -> str:
    return build_daily_report_object(
        todays_events,
        next_7_events or {},
        today_tomorrow_earnings or {},
        next_7_earnings or {},
        watchlist_risk or {},
        rss_headlines or {},
        youtube_links or {},
        fed_calendar=fed_calendar or {},
        treasury_calendar=treasury_calendar or {},
        sec_filings=sec_filings or {},
    )["markdown"]


def build_daily_report_object(
    todays_events: dict | None = None,
    next_7_events: dict | None = None,
    today_tomorrow_earnings: dict | None = None,
    next_7_earnings: dict | None = None,
    watchlist_risk: dict | None = None,
    rss_headlines: dict | None = None,
    youtube_links: dict | None = None,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
    *,
    report_date: str | None = None,
    generated_at: str | None = None,
) -> dict:
    report_date = report_date or datetime.now().date().isoformat()
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    todays_events = todays_events or {}
    next_7_events = next_7_events or {}
    today_tomorrow_earnings = today_tomorrow_earnings or {}
    next_7_earnings = next_7_earnings or {}
    watchlist_risk = watchlist_risk or {}
    rss_headlines = rss_headlines or {}
    youtube_links = youtube_links or {}
    fed_calendar = fed_calendar or {}
    treasury_calendar = treasury_calendar or {}
    sec_filings = sec_filings or {}
    trading_posture = build_daily_trading_posture(todays_events, today_tomorrow_earnings, watchlist_risk)
    report = {
        "report_type": "daily",
        "report_date": report_date,
        "export_prefix": "daily_market_prep",
        "generated_at": generated_at,
        "scheduled_landmines": build_scheduled_landmines(todays_events, today_tomorrow_earnings, watchlist_risk),
        "todays_events": todays_events,
        "next_7_events": next_7_events,
        "today_tomorrow_earnings": today_tomorrow_earnings,
        "next_7_earnings": next_7_earnings,
        "earnings_risk": today_tomorrow_earnings,
        "watchlist_risk": watchlist_risk,
        "rss_headlines": rss_headlines,
        "youtube_links": youtube_links,
        "fed_calendar": fed_calendar,
        "treasury_calendar": treasury_calendar,
        "sec_filings": sec_filings,
        "catalyst_clock": build_catalyst_clock(
            next_7_events,
            next_7_earnings,
            watchlist_risk,
            fed_calendar=fed_calendar,
            treasury_calendar=treasury_calendar,
            sec_filings=sec_filings,
        ),
        "trading_posture": trading_posture,
    }
    report["markdown"] = build_daily_markdown(report)
    return report


def build_weekly_report_object(
    economic_calendar: dict | None = None,
    earnings_calendar: dict | None = None,
    watchlist_risk: dict | None = None,
    rss_headlines: dict | None = None,
    youtube_links: dict | None = None,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
    *,
    report_date: str | None = None,
    generated_at: str | None = None,
) -> dict:
    report_date = report_date or datetime.now().date().isoformat()
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    economic_calendar = economic_calendar or {}
    earnings_calendar = earnings_calendar or {}
    watchlist_risk = watchlist_risk or {}
    rss_headlines = rss_headlines or {}
    youtube_links = youtube_links or {}
    fed_calendar = fed_calendar or {}
    treasury_calendar = treasury_calendar or {}
    sec_filings = sec_filings or {}
    major_earnings = _major_earnings_payload(earnings_calendar)
    week_risk = build_week_risk_level(
        economic_calendar,
        major_earnings,
        watchlist_risk,
        fed_calendar=fed_calendar,
        treasury_calendar=treasury_calendar,
        sec_filings=sec_filings,
    )
    swing_conditions = build_swing_trading_conditions(week_risk)
    report = {
        "report_type": "weekly",
        "report_date": report_date,
        "export_prefix": "weekly_market_prep",
        "generated_at": generated_at,
        "week_risk_level": week_risk,
        "economic_calendar": economic_calendar,
        "fed_calendar": fed_calendar,
        "treasury_calendar": treasury_calendar,
        "major_earnings": major_earnings,
        "watchlist_earnings_risk": watchlist_risk,
        "sec_filings": sec_filings,
        "rss_headlines": rss_headlines,
        "youtube_links": youtube_links,
        "swing_trading_conditions": swing_conditions,
    }
    report["markdown"] = build_weekly_markdown(report)
    return report


def build_daily_markdown(report: dict) -> str:
    report_date = str(report.get("report_date") or datetime.now().date().isoformat())
    scheduled_landmines = (
        report.get("scheduled_landmines")
        if isinstance(report.get("scheduled_landmines"), dict)
        else {}
    )
    todays_events = report.get("todays_events") if isinstance(report.get("todays_events"), dict) else {}
    next_7_events = report.get("next_7_events") if isinstance(report.get("next_7_events"), dict) else {}
    today_tomorrow_earnings = (
        report.get("today_tomorrow_earnings")
        if isinstance(report.get("today_tomorrow_earnings"), dict)
        else {}
    )
    next_7_earnings = report.get("next_7_earnings") if isinstance(report.get("next_7_earnings"), dict) else {}
    watchlist = report.get("watchlist_risk") if isinstance(report.get("watchlist_risk"), dict) else {}
    headlines = report.get("rss_headlines") if isinstance(report.get("rss_headlines"), dict) else {}
    youtube_links = report.get("youtube_links") if isinstance(report.get("youtube_links"), dict) else {}
    fed_calendar = report.get("fed_calendar") if isinstance(report.get("fed_calendar"), dict) else {}
    treasury_calendar = report.get("treasury_calendar") if isinstance(report.get("treasury_calendar"), dict) else {}
    sec_filings = report.get("sec_filings") if isinstance(report.get("sec_filings"), dict) else {}
    catalyst_clock = report.get("catalyst_clock") if isinstance(report.get("catalyst_clock"), list) else []
    posture = report.get("trading_posture") if isinstance(report.get("trading_posture"), list) else []

    lines = [
        f"# Daily Market Prep - {report_date}",
        "",
        "## 1. Highest Importance Focus",
        "",
    ]
    lines.extend(
        _daily_top_focus_markdown(
            todays_events,
            next_7_events,
            today_tomorrow_earnings,
            next_7_earnings,
            watchlist,
            fed_calendar=fed_calendar,
            treasury_calendar=treasury_calendar,
            sec_filings=sec_filings,
            headlines=headlines,
        )
    )
    lines.extend(
        [
            "",
            "## 2. Catalyst Clock",
            "",
        ]
    )
    lines.extend(_catalyst_clock_markdown(catalyst_clock))
    lines.extend(
        [
            "",
            "## 3. Scheduled Landmines Today",
            "",
        ]
    )
    lines.extend(_scheduled_landmines_markdown(scheduled_landmines))
    lines.extend(
        [
            "",
            "## 4. Economic Speedbumps",
            "",
            "Today:",
            "",
        ]
    )
    lines.extend(_economic_events_markdown(todays_events))
    lines.extend(["", "Next 7 Days:", ""])
    lines.extend(_economic_events_markdown(next_7_events))
    lines.extend(
        [
            "",
            "## 5. Fed Risk",
            "",
        ]
    )
    lines.extend(_calendar_risk_markdown(fed_calendar, empty_message="No Fed calendar risk found."))
    lines.extend(
        [
            "",
            "## 6. Treasury Auction Risk",
            "",
        ]
    )
    lines.extend(_calendar_risk_markdown(treasury_calendar, empty_message="No Treasury auction risk found."))
    lines.extend(
        [
            "",
            "## 7. Earnings Risk",
            "",
        ]
    )
    lines.extend(_daily_earnings_risk_markdown(today_tomorrow_earnings, next_7_earnings, watchlist))
    lines.extend(
        [
            "",
            "## 8. Watchlist Risk",
            "",
        ]
    )
    lines.extend(_watchlist_risk_markdown(watchlist, flagged_only=True))
    lines.extend(
        [
            "",
            "## 9. SEC Filing Risk",
            "",
        ]
    )
    lines.extend(_sec_filings_markdown(sec_filings))
    lines.extend(
        [
            "",
            "## 10. Sympathy Risk",
            "",
        ]
    )
    lines.extend(_sympathy_risk_markdown(watchlist))
    lines.extend(
        [
            "",
            "## 11. Google News/RSS Headline Risk",
            "",
        ]
    )
    lines.extend(_rss_headlines_markdown(headlines))
    lines.extend(
        [
            "",
            "## 12. YouTube Links",
            "",
        ]
    )
    lines.extend(_youtube_links_markdown(youtube_links))
    lines.extend(
        [
            "",
            "## 13. Trading Posture",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in posture)
    return "\n".join(lines).rstrip() + "\n"


def build_weekly_markdown(report: dict) -> str:
    report_date = str(report.get("report_date") or datetime.now().date().isoformat())
    week_risk = report.get("week_risk_level") if isinstance(report.get("week_risk_level"), dict) else {}
    economic = report.get("economic_calendar") if isinstance(report.get("economic_calendar"), dict) else {}
    fed_calendar = report.get("fed_calendar") if isinstance(report.get("fed_calendar"), dict) else {}
    treasury_calendar = report.get("treasury_calendar") if isinstance(report.get("treasury_calendar"), dict) else {}
    major_earnings = report.get("major_earnings") if isinstance(report.get("major_earnings"), dict) else {}
    watchlist = (
        report.get("watchlist_earnings_risk")
        if isinstance(report.get("watchlist_earnings_risk"), dict)
        else {}
    )
    sec_filings = report.get("sec_filings") if isinstance(report.get("sec_filings"), dict) else {}
    headlines = report.get("rss_headlines") if isinstance(report.get("rss_headlines"), dict) else {}
    youtube_links = report.get("youtube_links") if isinstance(report.get("youtube_links"), dict) else {}
    swing_conditions = (
        report.get("swing_trading_conditions")
        if isinstance(report.get("swing_trading_conditions"), list)
        else []
    )

    lines = [
        f"# Weekly Market Prep - Week of {report_date}",
        "",
        "## 1. Highest Importance Focus",
        "",
    ]
    lines.extend(
        _weekly_top_focus_markdown(
            economic,
            major_earnings,
            watchlist,
            fed_calendar=fed_calendar,
            treasury_calendar=treasury_calendar,
            sec_filings=sec_filings,
            headlines=headlines,
        )
    )
    lines.extend(
        [
            "",
            "## 2. Week Risk Level",
            "",
            f"- Level: {week_risk.get('level') or 'LOW'}",
            f"- Reason: {week_risk.get('reason') or 'No meaningful scheduled events found.'}",
            "",
            "## 3. Economic Calendar",
            "",
        ]
    )
    lines.extend(_economic_events_markdown(economic))
    lines.extend(
        [
            "",
            "## 4. Fed Calendar",
            "",
        ]
    )
    lines.extend(_calendar_risk_markdown(fed_calendar, empty_message="No Fed calendar risk found."))
    lines.extend(
        [
            "",
            "## 5. Treasury Calendar",
            "",
        ]
    )
    lines.extend(_calendar_risk_markdown(treasury_calendar, empty_message="No Treasury auction risk found."))
    lines.extend(
        [
            "",
            "## 6. Major Earnings",
            "",
        ]
    )
    lines.extend(_weekly_earnings_markdown(major_earnings))
    lines.extend(
        [
            "",
            "## 7. Watchlist Risks",
            "",
        ]
    )
    lines.extend(_watchlist_risk_markdown(watchlist, flagged_only=True))
    lines.extend(
        [
            "",
            "## 8. SEC Filing Risks",
            "",
        ]
    )
    lines.extend(_sec_filings_markdown(sec_filings))
    lines.extend(
        [
            "",
            "## 9. Sympathy Map Risks",
            "",
        ]
    )
    lines.extend(_sympathy_risk_markdown(watchlist))
    lines.extend(
        [
            "",
            "## 10. Google News/RSS Headline Risk",
            "",
        ]
    )
    lines.extend(_rss_headlines_markdown(headlines))
    lines.extend(
        [
            "",
            "## 11. YouTube Links",
            "",
        ]
    )
    lines.extend(_youtube_links_markdown(youtube_links))
    lines.extend(
        [
            "",
            "## 12. Swing Trading Conditions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in swing_conditions)
    return "\n".join(lines).rstrip() + "\n"


def _daily_top_focus_markdown(
    todays_events: dict,
    next_7_events: dict,
    today_tomorrow_earnings: dict,
    next_7_earnings: dict,
    watchlist: dict,
    *,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
    headlines: dict | None = None,
    limit: int = 12,
) -> list[str]:
    items: list[tuple[tuple, str]] = []
    today_event_keys = {_event_identity(event) for event in _payload_rows(todays_events, "events")}
    today_earnings_keys = {_earnings_identity(row) for row in _payload_rows(today_tomorrow_earnings, "earnings")}

    for row in _watchlist_rows(watchlist):
        if "Earnings" not in str(row.get("classification") or ""):
            continue
        items.append(((0, 0, str(row.get("ticker") or "")), "WATCHLIST: " + _watchlist_earnings_line(row)[2:]))

    for event in _sort_events_for_display(_payload_rows(todays_events, "events")):
        if str(event.get("priority") or "").upper() == "HIGH":
            items.append(((1, *_event_display_sort_key(event)), "TODAY MACRO: " + _economic_event_line(event)[2:]))

    for row in _sort_earnings_for_display(_payload_rows(today_tomorrow_earnings, "earnings")):
        if _is_high_or_mega_earnings(row):
            items.append(
                ((2, *_earnings_display_sort_key(row)), "TODAY/TOMORROW EARNINGS: " + _dated_earnings_line(row)[2:])
            )

    for event in _sort_events_for_display(_payload_rows(next_7_events, "events")):
        if _event_identity(event) in today_event_keys:
            continue
        if str(event.get("priority") or "").upper() == "HIGH":
            items.append(((3, *_event_display_sort_key(event)), "UPCOMING MACRO: " + _economic_event_line(event)[2:]))

    for event in _sort_events_for_display(_payload_rows(fed_calendar or {}, "events")):
        if str(event.get("priority") or "").upper() == "HIGH":
            items.append(((3, *_event_display_sort_key(event)), "FED: " + _economic_event_line(event)[2:]))

    for event in _sort_events_for_display(_payload_rows(treasury_calendar or {}, "events")):
        if str(event.get("priority") or "").upper() in {"HIGH", "MEDIUM"}:
            items.append(((4, *_event_display_sort_key(event)), "TREASURY: " + _economic_event_line(event)[2:]))

    for row in _sort_earnings_for_display(_payload_rows(next_7_earnings, "earnings")):
        if _earnings_identity(row) in today_earnings_keys:
            continue
        if _is_high_or_mega_earnings(row):
            items.append(((5, *_earnings_display_sort_key(row)), "UPCOMING EARNINGS: " + _dated_earnings_line(row)[2:]))

    for row in _sec_filing_rows(sec_filings or {}):
        if str(row.get("risk_classification") or "").upper() == "HIGH":
            items.append(((6, str(row.get("filing_date") or ""), str(row.get("ticker") or "")), "SEC: " + _sec_filing_line(row)[2:]))

    for row in _watchlist_rows(watchlist):
        if "Sympathy Risk" in str(row.get("classification") or ""):
            items.append(((7, 0, str(row.get("ticker") or "")), "SYMPATHY: " + _watchlist_earnings_line(row)[2:]))

    if not items:
        for event in _sort_events_for_display(_payload_rows(next_7_events, "events")):
            if str(event.get("priority") or "").upper() == "MEDIUM":
                items.append(((8, *_event_display_sort_key(event)), "MEDIUM MACRO: " + _economic_event_line(event)[2:]))
        for row in _sort_earnings_for_display(_payload_rows(next_7_earnings, "earnings")):
            if str(row.get("importance") or "").upper() == "MEDIUM":
                items.append(((9, *_earnings_display_sort_key(row)), "MEDIUM EARNINGS: " + _dated_earnings_line(row)[2:]))

    for headline in _top_headline_rows(headlines or {}, limit=3):
        items.append(((10, str(headline.get("published") or ""), str(headline.get("title") or "")), "HEADLINE: " + _headline_line(headline)[2:]))

    return _focus_lines(items, limit=limit)


def _weekly_top_focus_markdown(
    economic: dict,
    major_earnings: dict,
    watchlist: dict,
    *,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
    headlines: dict | None = None,
    limit: int = 15,
) -> list[str]:
    items: list[tuple[tuple, str]] = []
    for row in _watchlist_rows(watchlist):
        if "Earnings" in str(row.get("classification") or ""):
            items.append(((0, 0, str(row.get("ticker") or "")), "WATCHLIST: " + _watchlist_earnings_line(row)[2:]))

    for event in _sort_events_for_display(_payload_rows(economic, "events")):
        priority = str(event.get("priority") or "").upper()
        if priority == "HIGH" or _is_key_weekly_risk_event(event):
            items.append(((1, *_event_display_sort_key(event)), "MACRO: " + _economic_event_line(event)[2:]))

    for event in _sort_events_for_display(_payload_rows(fed_calendar or {}, "events")):
        if str(event.get("priority") or "").upper() == "HIGH":
            items.append(((1, *_event_display_sort_key(event)), "FED: " + _economic_event_line(event)[2:]))

    for event in _sort_events_for_display(_payload_rows(treasury_calendar or {}, "events")):
        if str(event.get("priority") or "").upper() in {"HIGH", "MEDIUM"}:
            items.append(((2, *_event_display_sort_key(event)), "TREASURY: " + _economic_event_line(event)[2:]))

    for row in _sort_earnings_for_display(_payload_rows(major_earnings, "earnings")):
        items.append(((3, *_earnings_display_sort_key(row)), "EARNINGS: " + _dated_earnings_line(row)[2:]))

    for row in _sec_filing_rows(sec_filings or {}):
        if str(row.get("risk_classification") or "").upper() == "HIGH":
            items.append(((4, str(row.get("filing_date") or ""), str(row.get("ticker") or "")), "SEC: " + _sec_filing_line(row)[2:]))

    for row in _watchlist_rows(watchlist):
        if "Sympathy Risk" in str(row.get("classification") or ""):
            items.append(((5, 0, str(row.get("ticker") or "")), "SYMPATHY: " + _watchlist_earnings_line(row)[2:]))

    if not items:
        for event in _sort_events_for_display(_payload_rows(economic, "events")):
            if str(event.get("priority") or "").upper() == "MEDIUM":
                items.append(((6, *_event_display_sort_key(event)), "MEDIUM MACRO: " + _economic_event_line(event)[2:]))

    for headline in _top_headline_rows(headlines or {}, limit=3):
        items.append(((7, str(headline.get("published") or ""), str(headline.get("title") or "")), "HEADLINE: " + _headline_line(headline)[2:]))

    return _focus_lines(items, limit=limit)


def _focus_lines(items: list[tuple[tuple, str]], *, limit: int) -> list[str]:
    if not items:
        return ["No high-importance configured catalysts in this window."]
    ordered = sorted(items, key=lambda item: item[0])
    lines = [f"- {line}" for _key, line in ordered[:limit]]
    hidden = max(0, len(ordered) - limit)
    if hidden:
        lines.append(f"- ... {hidden} more focus item(s) hidden; see the detailed sections below.")
    return lines


def build_trading_posture(snapshot: dict, todays_events: dict, watchlist_earnings: dict) -> list[str]:
    posture: list[str] = []
    if _has_high_priority_event(todays_events):
        posture.append("Reduce size around scheduled event.")

    label = str(_market_tone(snapshot).get("label") or "")
    if "Clean" in label:
        posture.append("Long side favored, prefer clean dips.")
    elif "Noisy" in label:
        posture.append("Wait for confirmation, avoid forcing trades.")

    if _has_major_watchlist_earnings_risk(watchlist_earnings):
        posture.append("Avoid blind overnight holds in flagged tickers.")

    if not posture:
        posture.append("Stay selective; let price confirm the setup.")
    return posture


def build_scheduled_landmines(
    todays_events: dict,
    today_tomorrow_earnings: dict,
    watchlist_risk: dict,
) -> dict[str, list[dict]]:
    high_priority_events = [
        event for event in _payload_rows(todays_events, "events")
        if str(event.get("priority") or "").upper() == "HIGH"
    ]
    earnings = _payload_rows(today_tomorrow_earnings, "earnings")
    watchlist_earnings = [
        row for row in _watchlist_rows(watchlist_risk)
        if "Earnings Today/Tomorrow" in str(row.get("classification") or "")
    ]
    return {
        "high_priority_events": high_priority_events,
        "earnings_today_tomorrow": earnings,
        "watchlist_earnings_today_tomorrow": watchlist_earnings,
    }


def build_daily_trading_posture(
    todays_events: dict,
    today_tomorrow_earnings: dict,
    watchlist_risk: dict,
) -> list[str]:
    posture: list[str] = []
    landmines = build_scheduled_landmines(todays_events, today_tomorrow_earnings, watchlist_risk)
    if landmines["high_priority_events"]:
        posture.append("Reduce size around scheduled macro event.")
    if landmines["watchlist_earnings_today_tomorrow"]:
        posture.append("Avoid blind overnight holds in flagged tickers.")
    if not any(landmines.values()):
        posture.append("No major scheduled landmines configured. Still verify manually.")
    posture.append("This report is based only on configured sources.")
    return posture


def build_week_risk_level(
    economic_calendar: dict,
    major_earnings: dict,
    watchlist_risk: dict | None = None,
    *,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
) -> dict[str, str]:
    events = _payload_rows(economic_calendar, "events")
    fed_events = _payload_rows(fed_calendar or {}, "events")
    treasury_events = _payload_rows(treasury_calendar or {}, "events")
    all_events = events + fed_events + treasury_events
    earnings = _payload_rows(major_earnings, "earnings")
    key_events = [event for event in all_events if _is_key_weekly_risk_event(event)]
    mega_earnings = [row for row in earnings if _is_major_mega_cap_earnings(row)]
    meaningful_events = [
        event
        for event in all_events
        if str(event.get("priority") or "").upper() in {"HIGH", "MEDIUM"}
    ]
    meaningful_earnings = [
        row
        for row in earnings
        if str(row.get("importance") or "").upper() in {"MEGA", "HIGH", "MEDIUM"} or _is_major_mega_cap_earnings(row)
    ]
    watchlist_earnings_count = len(
        [
            row for row in _watchlist_rows(watchlist_risk or {})
            if "Earnings" in str(row.get("classification") or "")
        ]
    )
    high_sec_count = len(
        [
            row for row in _sec_filing_rows(sec_filings or {})
            if str(row.get("risk_classification") or "").upper() == "HIGH"
        ]
    )
    sympathy_count = len(
        [
            row for row in _watchlist_rows(watchlist_risk or {})
            if "Sympathy Risk" in str(row.get("classification") or "")
        ]
    )

    if key_events or mega_earnings or high_sec_count:
        reasons = []
        if key_events:
            reasons.append("key macro events are scheduled")
        if mega_earnings:
            reasons.append("major mega-cap earnings are scheduled")
        if high_sec_count:
            reasons.append("high-risk SEC filings are present")
        return {
            "level": "HIGH",
            "reason": " and ".join(reasons).capitalize() + ".",
        }

    meaningful_count = len(meaningful_events) + len(meaningful_earnings) + sympathy_count
    if meaningful_count >= 2 or watchlist_earnings_count >= 3:
        return {
            "level": "MEDIUM",
            "reason": "Several medium/high scheduled events are present.",
        }
    if meaningful_count == 1:
        return {
            "level": "MEDIUM",
            "reason": "One meaningful scheduled event is present.",
        }
    return {
        "level": "LOW",
        "reason": "No meaningful scheduled events found.",
    }


def build_swing_trading_conditions(week_risk: dict) -> list[str]:
    risk_level = str(week_risk.get("level") or "LOW").upper()
    if risk_level == "HIGH":
        return ["Be careful initiating swing positions before scheduled catalysts."]
    if risk_level == "MEDIUM":
        return ["Swing trades are acceptable but event timing matters."]
    return ["No major configured weekly catalysts. Still verify manually."]


def build_catalyst_clock(
    economic_calendar: dict,
    earnings_calendar: dict,
    watchlist_risk: dict,
    *,
    fed_calendar: dict | None = None,
    treasury_calendar: dict | None = None,
    sec_filings: dict | None = None,
    limit: int = 60,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for event in _payload_rows(economic_calendar, "events"):
        items.append(
            _clock_item(
                date_value=str(event.get("date") or ""),
                time_value=str(event.get("time_et") or ""),
                bucket="Economic",
                priority=str(event.get("priority") or ""),
                text=_event_clock_text(event),
            )
        )
    for event in _payload_rows(fed_calendar or {}, "events"):
        items.append(
            _clock_item(
                date_value=str(event.get("date") or ""),
                time_value=str(event.get("time_et") or ""),
                bucket="Fed",
                priority=str(event.get("priority") or ""),
                text=_event_clock_text(event),
            )
        )
    for event in _payload_rows(treasury_calendar or {}, "events"):
        if str(event.get("priority") or "").upper() == "LOW":
            continue
        items.append(
            _clock_item(
                date_value=str(event.get("date") or ""),
                time_value=str(event.get("time_et") or ""),
                bucket="Treasury",
                priority=str(event.get("priority") or ""),
                text=_event_clock_text(event),
            )
        )
    watchlist_earnings_tickers = {
        str(row.get("ticker") or "").strip().upper()
        for row in _watchlist_rows(watchlist_risk)
        if "Earnings" in str(row.get("classification") or "")
    }
    for row in _payload_rows(earnings_calendar, "earnings"):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not (_is_notable_earnings(row) or ticker in watchlist_earnings_tickers):
            continue
        items.append(
            _clock_item(
                date_value=str(row.get("date") or ""),
                time_value=_earnings_clock_time(row),
                bucket="Earnings",
                priority=str(row.get("importance") or ""),
                text=_earnings_clock_text(row),
            )
        )
    for row in _sec_filing_rows(sec_filings or {}):
        if str(row.get("risk_classification") or "").upper() not in {"HIGH", "MEDIUM"}:
            continue
        items.append(
            _clock_item(
                date_value=str(row.get("filing_date") or ""),
                time_value="",
                bucket="SEC",
                priority=str(row.get("risk_classification") or ""),
                text=_sec_filing_clock_text(row),
            )
        )
    deduped = _dedupe_clock_items(items)
    deduped.sort(key=_clock_sort_key)
    return deduped[: max(0, int(limit))]


def build_economic_calendar_report(payload: dict, *, title: str = "Economic Calendar") -> str:
    events = payload.get("events") if isinstance(payload, dict) else []
    events = events if isinstance(events, list) else []
    lines = [
        title,
        "=" * 80,
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        f"Source: {payload.get('source') or 'manual'}",
        f"Window: {payload.get('start_date') or 'n/a'} to {payload.get('end_date') or 'n/a'}",
    ]
    status_lines = _forexfactory_status_markdown(payload)
    if status_lines:
        lines.extend(status_lines)
    lines.append("")

    if not events:
        lines.append(str(payload.get("message") or "No configured economic events found."))
        return "\n".join(lines).rstrip()

    lines.extend(
        [
            f"{'Date':<12} {'ET':<7} {'Priority':<8} {'CCY':<4} Event",
            "-" * 80,
        ]
    )
    for event in events:
        if not isinstance(event, dict):
            continue
        event_name = str(event.get("event") or "").strip()
        notes = str(event.get("notes") or "").strip()
        currency = str(event.get("currency") or "").strip()
        stats = _event_stats_text(event)
        line = (
            f"{str(event.get('date') or ''):<12} "
            f"{str(event.get('time_et') or ''):<7} "
            f"{str(event.get('priority') or ''):<8} "
            f"{currency:<4} "
            f"{event_name}"
        ).rstrip()
        if stats:
            line += f" | {stats}"
        if notes:
            line += f" | {notes}"
        lines.append(line)

    return "\n".join(lines).rstrip()


def build_earnings_report(payload: dict, *, title: str = "Earnings Calendar") -> str:
    earnings = payload.get("earnings") if isinstance(payload, dict) else []
    earnings = earnings if isinstance(earnings, list) else []
    lines = [
        title,
        "=" * 80,
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        f"Source: {payload.get('source') or 'manual'}",
        f"Window: {payload.get('start_date') or 'n/a'} to {payload.get('end_date') or 'n/a'}",
    ]
    yfinance_lines = _yfinance_status_markdown(payload)
    if yfinance_lines:
        lines.extend(yfinance_lines)
    lines.append("")

    if not earnings:
        lines.append(str(payload.get("message") or "No configured earnings found."))
        return "\n".join(lines).rstrip()

    focus_rows = [row for row in _sort_earnings_for_display(earnings) if _is_high_or_mega_earnings(row)]
    if focus_rows:
        lines.append("Market-moving focus:")
        for row in focus_rows[:12]:
            ticker = str(row.get("ticker") or "").strip().upper()
            company = str(row.get("company_yfinance") or row.get("company") or "").strip()
            when = " ".join(str(row.get(key) or "").strip() for key in ("date", "time")).strip()
            detail = " | ".join(
                part
                for part in (
                    ticker,
                    when,
                    str(row.get("importance") or "").strip(),
                    _market_cap_text(row),
                    company,
                )
                if part
            )
            lines.append(f"- {detail}")
        if len(focus_rows) > 12:
            lines.append(f"- ... {len(focus_rows) - 12} more high-impact earnings")
        lines.append("")

    lines.extend(
        [
            (
                f"{'Date':<12} {'Time':<6} {'Ticker':<8} {'Importance':<10} "
                f"{'Market Cap':<18} {'Sector':<24} Company / Notes"
            ),
            "-" * 80,
        ]
    )
    for row in earnings:
        if not isinstance(row, dict):
            continue
        company = str(row.get("company") or "").strip()
        notes = str(row.get("notes") or "").strip()
        detail = company
        if notes:
            detail = f"{detail} | {notes}" if detail else notes
        lines.append(
            f"{str(row.get('date') or ''):<12} "
            f"{str(row.get('time') or ''):<6} "
            f"{str(row.get('ticker') or ''):<8} "
            f"{str(row.get('importance') or ''):<10} "
            f"{_market_cap_text(row):<18} "
            f"{str(row.get('sector') or row.get('industry') or '')[:24]:<24} "
            f"{detail}".rstrip()
        )

    tickers = payload.get("tickers") if isinstance(payload, dict) else None
    if isinstance(tickers, list):
        lines.extend(["", f"Watchlist symbols checked: {len(tickers)}"])

    return "\n".join(lines).rstrip()


def build_watchlist_risk_report(payload: dict, *, title: str = "Watchlist Risk Scan") -> str:
    lines = [
        title,
        "=" * 80,
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        f"Window: {payload.get('start_date') or 'n/a'} to {payload.get('end_date') or 'n/a'}",
        "",
    ]
    missing_files = payload.get("missing_files") if isinstance(payload, dict) else []
    if isinstance(missing_files, list) and missing_files:
        lines.append("Missing watchlist files:")
        lines.extend(f"- {path}" for path in missing_files)
        lines.append("")
    yfinance_lines = _yfinance_status_markdown(payload)
    if yfinance_lines:
        lines.extend(yfinance_lines)
        lines.append("")
    lines.extend(_watchlist_risk_markdown(payload, flagged_only=False))
    return "\n".join(lines).rstrip()


def build_rss_news_report(payload: dict, *, title: str = "RSS Macro Headlines") -> str:
    lines = [
        title,
        "=" * 80,
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        "",
    ]
    warnings = payload.get("warnings") if isinstance(payload, dict) else []
    if isinstance(warnings, list) and warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(_rss_headlines_markdown(payload, limit=25))
    return "\n".join(lines).rstrip()


def build_youtube_links_report(payload: dict, *, title: str = "YouTube RSS Links") -> str:
    lines = [
        title,
        "=" * 80,
        f"Generated at: {payload.get('generated_at') or 'n/a'}",
        f"Output file: {payload.get('output_path') or 'n/a'}",
        "",
    ]
    warnings = payload.get("warnings") if isinstance(payload, dict) else []
    if isinstance(warnings, list) and warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(_youtube_links_markdown(payload, limit=25))
    return "\n".join(lines).rstrip()


def _fmt_number(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_above(value) -> str:
    if value is True:
        return "Above"
    if value is False:
        return "Below"
    return "n/a"


def _market_tone(snapshot: dict) -> dict:
    classification = snapshot.get("classification") if isinstance(snapshot, dict) else {}
    return classification if isinstance(classification, dict) else {}


def _market_snapshot_markdown_table(snapshot: dict) -> list[str]:
    rows = snapshot.get("rows") if isinstance(snapshot, dict) else []
    rows = rows if isinstance(rows, list) else []
    if not rows:
        return ["No market snapshot rows available."]

    lines = [
        "| Ticker | Close | 1D% | 5D% | 20D% | 21 SMA | 50 SMA | Vol vs 20D |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("ticker") or ""),
                    _fmt_number(row.get("last_close")),
                    _fmt_pct(row.get("return_1d_pct")),
                    _fmt_pct(row.get("return_5d_pct")),
                    _fmt_pct(row.get("return_20d_pct")),
                    _fmt_above(row.get("above_21_sma")),
                    _fmt_above(row.get("above_50_sma")),
                    _fmt_pct(row.get("volume_vs_20d_avg_pct")),
                ]
            )
            + " |"
        )
    return lines


def _economic_events_markdown(payload: dict) -> list[str]:
    events = payload.get("events") if isinstance(payload, dict) else []
    events = events if isinstance(events, list) else []
    lines = _forexfactory_status_markdown(payload)
    if not events:
        lines.append(str(payload.get("message") or "No configured economic events found."))
        return lines
    for event in _sort_events_for_display(events):
        if not isinstance(event, dict):
            continue
        lines.append(_economic_event_line(event))
    return lines


def _economic_event_line(event: dict) -> str:
    currency = str(event.get("currency") or "").strip()
    currency_text = f" {currency}" if currency else ""
    stats = _event_stats_text(event)
    date_value = str(event.get("date") or "").strip()
    time_value = str(event.get("time_et") or "").strip()
    time_text = f"{time_value} ET" if time_value else "TBD ET"
    line = (
        f"- {date_value} {time_text} "
        f"[{event.get('priority', '')}]{currency_text} {event.get('event', '')}"
    ).strip()
    if stats:
        line += f" - {stats}"
    if event.get("notes"):
        line += f" - {event.get('notes')}"
    return line


def _calendar_risk_markdown(payload: dict, *, empty_message: str) -> list[str]:
    lines = _source_status_markdown(payload)
    events = _payload_rows(payload, "events")
    if not events:
        lines.append(str(payload.get("message") or empty_message))
        return lines
    lines.extend(_economic_event_line(event) for event in _sort_events_for_display(events))
    return lines


def _source_status_markdown(payload: dict) -> list[str]:
    status_label = str(payload.get("status_label") or "").strip() if isinstance(payload, dict) else ""
    source = str(payload.get("source") or "").strip() if isinstance(payload, dict) else ""
    lines: list[str] = []
    if status_label:
        label = source or "Source"
        lines.append(f"{label}: {status_label}")
    warnings = payload.get("warnings") if isinstance(payload, dict) else []
    if isinstance(warnings, list) and warnings:
        label = source or "Source"
        lines.append(f"{label} warning: {warnings[0]}")
    return lines


def _catalyst_clock_markdown(items: list[dict[str, str]]) -> list[str]:
    if not items:
        return ["No catalyst clock items found."]
    return [_clock_line(item) for item in items]


def _clock_item(
    *,
    date_value: str,
    time_value: str,
    bucket: str,
    priority: str,
    text: str,
) -> dict[str, str]:
    return {
        "date": str(date_value or "").strip(),
        "time_et": str(time_value or "").strip(),
        "bucket": str(bucket or "").strip(),
        "priority": str(priority or "").strip().upper(),
        "text": str(text or "").strip(),
    }


def _clock_line(item: dict[str, str]) -> str:
    date_value = str(item.get("date") or "n/a").strip()
    time_value = str(item.get("time_et") or "TBD").strip()
    priority = str(item.get("priority") or "").strip().upper()
    priority_text = f" [{priority}]" if priority else ""
    return (
        f"- {date_value} {time_value} ET | "
        f"{item.get('bucket') or 'Catalyst'}{priority_text} | "
        f"{item.get('text') or 'n/a'}"
    )


def _clock_sort_key(item: dict[str, str]) -> tuple[str, int, str, str]:
    return (
        str(item.get("date") or "9999-99-99"),
        _time_sort_value(item.get("time_et")),
        str(item.get("bucket") or ""),
        str(item.get("text") or ""),
    )


def _dedupe_clock_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (
            str(item.get("date") or ""),
            str(item.get("time_et") or ""),
            str(item.get("bucket") or ""),
            str(item.get("text") or "").upper(),
        )
        if key in seen or not item.get("date") or not item.get("text"):
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _event_clock_text(event: dict) -> str:
    currency = str(event.get("currency") or "").strip()
    event_name = str(event.get("event") or "").strip()
    stats = _event_stats_text(event)
    notes = str(event.get("notes") or "").strip()
    parts = [part for part in (currency, event_name) if part]
    text = " ".join(parts)
    if stats:
        text += f" | {stats}"
    if notes:
        text += f" | {notes}"
    return text


def _earnings_clock_time(row: dict) -> str:
    label = str(row.get("time") or "").strip().upper()
    if label == "BMO":
        return "08:00"
    if label == "AMC":
        return "16:05"
    if re_match := _time_text_to_24h(label):
        return re_match
    return ""


def _earnings_clock_text(row: dict) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    company = str(row.get("company_yfinance") or row.get("company") or "").strip()
    time_value = str(row.get("time") or "").strip().upper()
    market_cap = _market_cap_text(row)
    parts = [ticker, company, time_value, market_cap]
    return " | ".join(part for part in parts if part)


def _sec_filing_clock_text(row: dict) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    form = str(row.get("form") or "").strip()
    risk = str(row.get("risk_classification") or "").strip().upper()
    keywords = ", ".join(row.get("matched_keywords") or [])
    text = f"{ticker} {form} filing"
    if risk:
        text += f" | {risk}"
    if keywords:
        text += f" | matched: {keywords}"
    return text


def _time_sort_value(value) -> int:
    text = str(value or "").strip()
    if not text:
        return 24 * 60 + 59
    parsed = _time_text_to_24h(text)
    if not parsed:
        return 24 * 60 + 59
    hour, minute = parsed.split(":", 1)
    return int(hour) * 60 + int(minute)


def _time_text_to_24h(value) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 5 and text[2] == ":" and text[:2].isdigit() and text[3:].isdigit():
        return text
    return ""


def _earnings_markdown(payload: dict) -> list[str]:
    earnings = payload.get("earnings") if isinstance(payload, dict) else []
    earnings = earnings if isinstance(earnings, list) else []
    lines = _yfinance_status_markdown(payload)
    if not earnings:
        lines.append(str(payload.get("message") or "No configured earnings found."))
        return lines
    lines.extend(_earnings_line(row) for row in _sort_earnings_for_display(earnings) if isinstance(row, dict))
    return lines


def _earnings_line(row: dict) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    company = str(row.get("company_yfinance") or row.get("company") or "").strip()
    time_value = str(row.get("time") or "").strip()
    market_cap = _market_cap_text(row)
    sector = str(row.get("sector") or row.get("industry") or "").strip()
    importance = str(row.get("importance") or "").strip().upper()
    source = _earnings_source_text(row)
    notes = str(row.get("notes") or "").strip()
    parts = [ticker, company, time_value, market_cap, sector, importance, source]
    line = "- " + " | ".join(part if part else "n/a" for part in parts)
    if notes:
        line += f" | {notes}"
    return line


def _dated_earnings_line(row: dict) -> str:
    dated = dict(row)
    date_value = str(row.get("date") or "").strip()
    line = _earnings_line(dated)
    return line.replace("- ", f"- {date_value} | ", 1) if date_value else line


def _daily_earnings_risk_markdown(
    today_tomorrow_payload: dict,
    next_7_payload: dict,
    watchlist_payload: dict,
    *,
    limit: int = 30,
) -> list[str]:
    today_rows = _sort_earnings_for_display(_payload_rows(today_tomorrow_payload, "earnings"))
    next_7_rows = _sort_earnings_for_display(_payload_rows(next_7_payload, "earnings"))
    watchlist_today_rows = [
        row for row in _watchlist_rows(watchlist_payload)
        if "Earnings Today/Tomorrow" in str(row.get("classification") or "")
    ]
    yfinance_lines = _dedupe_lines(
        _yfinance_status_markdown(today_tomorrow_payload)
        + _yfinance_status_markdown(next_7_payload)
        + _yfinance_status_markdown(watchlist_payload)
    )

    lines: list[str] = []
    lines.extend(yfinance_lines)
    if yfinance_lines:
        lines.append("")

    rendered = 0
    hidden = 0
    seen_keys: set[tuple[str, str, str]] = set()

    def add_group(title: str, rows: list[dict], *, watchlist_rows: bool = False) -> None:
        nonlocal rendered, hidden
        lines.append(f"{title}:")
        if not rows:
            lines.append("- None configured.")
            lines.append("")
            return
        for row in rows:
            if rendered >= limit:
                hidden += 1
                continue
            if watchlist_rows:
                lines.append(_watchlist_earnings_line(row))
            else:
                lines.append(_earnings_line(row))
                seen_keys.add(_earnings_identity(row))
            rendered += 1
        lines.append("")

    watchlist_tickers = {
        str(row.get("ticker") or "").strip().upper()
        for row in watchlist_today_rows
        if isinstance(row, dict)
    }
    mega_high_today = [
        row for row in today_rows
        if _earnings_identity(row) not in seen_keys
        and str(row.get("ticker") or "").strip().upper() not in watchlist_tickers
        and _is_high_or_mega_earnings(row)
    ]
    other_today = [
        row for row in today_rows
        if _earnings_identity(row) not in seen_keys
        and row not in mega_high_today
        and str(row.get("ticker") or "").strip().upper() not in watchlist_tickers
    ]
    next_notable = [
        row for row in next_7_rows
        if _earnings_identity(row) not in {_earnings_identity(today_row) for today_row in today_rows}
        and str(row.get("importance") or "").upper() in {"MEGA", "HIGH", "MEDIUM"}
    ]

    add_group("A. Watchlist earnings today/tomorrow", watchlist_today_rows, watchlist_rows=True)
    add_group("B. Mega/high market cap earnings today/tomorrow", mega_high_today)
    add_group("C. Other earnings today/tomorrow", other_today)
    add_group("D. Next 7 days notable earnings", next_notable)

    if hidden:
        lines.append(f"Additional earnings hidden: {hidden}. See full export/cache if needed.")
    while lines and lines[-1] == "":
        lines.pop()
    return lines or ["No configured earnings found."]


def _weekly_earnings_markdown(payload: dict, *, limit: int = 50) -> list[str]:
    rows = _sort_earnings_for_display(_payload_rows(payload, "earnings"))
    lines = _yfinance_status_markdown(payload)
    if not rows:
        lines.append(str(payload.get("message") or "No configured major earnings found."))
        return lines

    shown_rows = rows[:limit]
    hidden = max(0, len(rows) - len(shown_rows))
    if lines:
        lines.append("")
    lines.append("Top major earnings by market impact:")
    lines.extend(_dated_earnings_line(row) for row in shown_rows)
    if hidden:
        lines.append(f"Additional earnings hidden: {hidden}. See full export/cache if needed.")
    return lines


def _forexfactory_status_markdown(payload: dict) -> list[str]:
    status = payload.get("forexfactory_status") if isinstance(payload, dict) else None
    if not isinstance(status, dict):
        return []
    status_label = str(status.get("status_label") or status.get("status") or "").strip()
    if not status_label:
        return []
    lines = [f"ForexFactory: {status_label}"]
    warnings = status.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append(f"ForexFactory warning: {warnings[0]}")
    return lines


def _yfinance_status_markdown(payload: dict) -> list[str]:
    status = payload.get("yfinance_status") if isinstance(payload, dict) else None
    if not isinstance(status, dict):
        return []
    status_label = str(status.get("status_label") or status.get("status") or "").strip()
    if not status_label or status_label == "No metadata requested":
        return []
    lines = [f"yfinance: {status_label}"]
    warnings = status.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append(f"yfinance warning: {warnings[0]}")
    return lines


def _event_stats_text(event: dict) -> str:
    parts = []
    for label, key in (("Actual", "actual"), ("Forecast", "forecast"), ("Previous", "previous")):
        value = str(event.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return ", ".join(parts)


def _scheduled_landmines_markdown(payload: dict) -> list[str]:
    high_events = _payload_rows(payload, "high_priority_events")
    earnings = _payload_rows(payload, "earnings_today_tomorrow")
    watchlist_earnings = _payload_rows(payload, "watchlist_earnings_today_tomorrow")
    lines: list[str] = []
    if high_events:
        lines.append("High-priority economic events today:")
        lines.extend(_economic_events_markdown({"events": high_events}))
    if earnings:
        notable_earnings = [
            row for row in _sort_earnings_for_display(earnings)
            if _is_notable_earnings(row)
        ]
        visible_earnings = notable_earnings[:12]
        if not visible_earnings:
            visible_earnings = _sort_earnings_for_display(earnings)[:5]
        hidden = max(0, len(earnings) - len(visible_earnings))
        if lines:
            lines.append("")
        lines.append("Market-moving earnings today/tomorrow:")
        lines.extend(_earnings_markdown({"earnings": visible_earnings}))
        if hidden:
            lines.append(f"- ... {hidden} lower-priority earnings hidden; see Earnings Risk below.")
    if watchlist_earnings:
        if lines:
            lines.append("")
        lines.append("Watchlist earnings today/tomorrow:")
        lines.extend(_watchlist_risk_markdown({"risks": watchlist_earnings}, flagged_only=False))
    if not lines:
        lines.append("No major scheduled landmines configured. Still verify manually.")
    return lines


def _watchlist_risk_markdown(
    payload: dict,
    *,
    flagged_only: bool = False,
    earnings_only: bool = False,
) -> list[str]:
    rows = _watchlist_rows(payload)
    if flagged_only:
        rows = [
            row for row in rows
            if "Clean" not in str(row.get("classification") or "")
        ]
    if earnings_only:
        rows = [
            row for row in rows
            if "Earnings" in str(row.get("classification") or "")
        ]
    if not rows:
        return [str(payload.get("message") or "No flagged watchlist risk found.")]

    lines = [
        "| Ticker | Company | Source List | Market Cap | Sector | Classification | Reason |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(row.get("ticker")),
                    _escape_table_cell(row.get("company")),
                    _escape_table_cell(row.get("source_list") or ", ".join(row.get("source_lists") or [])),
                    _escape_table_cell(row.get("market_cap_fmt") or _market_cap_text(row)),
                    _escape_table_cell(row.get("sector") or row.get("industry")),
                    _escape_table_cell(row.get("classification")),
                    _escape_table_cell(row.get("reason")),
                ]
            )
            + " |"
        )
    return lines


def _sympathy_risk_markdown(payload: dict) -> list[str]:
    rows = payload.get("sympathy_risks") if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        rows = [
            row for row in _watchlist_rows(payload)
            if "Sympathy Risk" in str(row.get("classification") or "")
        ]
    if not rows:
        return ["No sympathy map risks found."]
    return _watchlist_risk_markdown({"risks": rows}, flagged_only=False)


def _sec_filings_markdown(payload: dict, *, limit: int = 20) -> list[str]:
    lines = _source_status_markdown(payload)
    rows = _sec_filing_rows(payload)
    if not rows:
        lines.append(str(payload.get("message") or "No SEC filing risk found."))
        return lines
    shown = rows[:limit]
    hidden = max(0, len(rows) - len(shown))
    lines.extend(
        [
            "| Ticker | Form | Filing Date | Risk | Matched Keywords | URL |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in shown:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(row.get("ticker")),
                    _escape_table_cell(row.get("form")),
                    _escape_table_cell(row.get("filing_date")),
                    _escape_table_cell(row.get("risk_classification")),
                    _escape_table_cell(", ".join(row.get("matched_keywords") or [])),
                    _escape_table_cell(row.get("url")),
                ]
            )
            + " |"
        )
    if hidden:
        lines.append(f"Additional SEC filings hidden: {hidden}. See cache/export if needed.")
    return lines


def _sec_filing_rows(payload: dict) -> list[dict]:
    rows = _payload_rows(payload, "filings")
    rows.sort(key=_sec_filing_sort_key)
    return rows


def _sec_filing_sort_key(row: dict) -> tuple[int, str, str, str]:
    risk = str(row.get("risk_classification") or "LOW").upper()
    risk_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(risk, 9)
    return (
        risk_rank,
        str(row.get("filing_date") or ""),
        str(row.get("ticker") or ""),
        str(row.get("form") or ""),
    )


def _sec_filing_line(row: dict) -> str:
    keywords = ", ".join(row.get("matched_keywords") or [])
    parts = [
        str(row.get("ticker") or "").strip().upper(),
        str(row.get("form") or "").strip(),
        str(row.get("filing_date") or "").strip(),
        str(row.get("risk_classification") or "").strip().upper(),
        keywords,
        str(row.get("url") or "").strip(),
    ]
    return "- " + " | ".join(part if part else "n/a" for part in parts)


def _rss_headlines_markdown(payload: dict, *, limit: int = 10) -> list[str]:
    headlines = payload.get("headlines") if isinstance(payload, dict) else []
    headlines = headlines if isinstance(headlines, list) else []
    if not headlines:
        return [str(payload.get("message") or "No RSS headlines found.")]
    lines = []
    for headline in headlines[:limit]:
        if not isinstance(headline, dict):
            continue
        lines.append(_headline_line(headline))
    return lines or [str(payload.get("message") or "No RSS headlines found.")]


def _top_headline_rows(payload: dict, *, limit: int) -> list[dict]:
    headlines = payload.get("headlines") if isinstance(payload, dict) else []
    rows = [row for row in headlines if isinstance(row, dict)] if isinstance(headlines, list) else []
    tagged = [row for row in rows if row.get("tags")]
    return (tagged or rows)[:limit]


def _headline_line(headline: dict) -> str:
    tags = ", ".join(headline.get("tags") or [])
    prefix = f"[{tags}] " if tags else ""
    source = str(headline.get("source") or "RSS").strip()
    title = str(headline.get("title") or "").strip()
    published = str(headline.get("published") or "").strip()
    query = str(headline.get("query") or "").strip()
    url = str(headline.get("url") or "").strip()
    detail = f"- {prefix}{title} ({source})"
    if query:
        detail += f" - query: {query}"
    if published:
        detail += f" - {published}"
    if url:
        detail += f" - {url}"
    return detail


def _youtube_links_markdown(payload: dict, *, limit: int = 10) -> list[str]:
    videos = payload.get("videos") if isinstance(payload, dict) else []
    videos = videos if isinstance(videos, list) else []
    if not videos:
        return [str(payload.get("message") or "No configured YouTube links found.")]
    lines = []
    for video in videos[:limit]:
        if not isinstance(video, dict):
            continue
        keywords = ", ".join(video.get("matched_keywords") or [])
        creator = str(video.get("creator") or "Creator").strip()
        title = str(video.get("title") or "").strip()
        published = str(video.get("published") or "").strip()
        url = str(video.get("url") or "").strip()
        detail = f"- {creator}: {title}"
        if published:
            detail += f" - {published}"
        if keywords:
            detail += f" - matched: {keywords}"
        if url:
            detail += f" - {url}"
        lines.append(detail)
    return lines or [str(payload.get("message") or "No configured YouTube links found.")]


def _sector_rotation_markdown(snapshot: dict) -> list[str]:
    rows = _payload_rows(snapshot, "rows")
    sector_rows = [
        row for row in rows
        if str(row.get("ticker") or "").upper() in SECTOR_ROTATION_TICKERS
    ]
    if not sector_rows:
        return ["No sector rotation snapshot available."]

    sector_rows.sort(
        key=lambda row: (
            _sort_float(row.get("return_5d_pct")),
            _sort_float(row.get("return_20d_pct")),
        ),
        reverse=True,
    )
    lines = [
        "| Ticker | 5D% | 20D% | 21 SMA | 50 SMA |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in sector_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("ticker") or ""),
                    _fmt_pct(row.get("return_5d_pct")),
                    _fmt_pct(row.get("return_20d_pct")),
                    _fmt_above(row.get("above_21_sma")),
                    _fmt_above(row.get("above_50_sma")),
                ]
            )
            + " |"
        )
    return lines


def _has_high_priority_event(payload: dict) -> bool:
    events = payload.get("events") if isinstance(payload, dict) else []
    return any(
        isinstance(event, dict) and str(event.get("priority") or "").upper() == "HIGH"
        for event in (events if isinstance(events, list) else [])
    )


def _has_major_watchlist_earnings_risk(payload: dict) -> bool:
    earnings = payload.get("earnings") if isinstance(payload, dict) else []
    return any(
        isinstance(row, dict) and str(row.get("importance") or "").upper() == "HIGH"
        for row in (earnings if isinstance(earnings, list) else [])
    )


def _payload_rows(payload: dict, key: str) -> list[dict]:
    rows = payload.get(key) if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _watchlist_rows(payload: dict) -> list[dict]:
    return _payload_rows(payload, "risks")


def _escape_table_cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _sort_events_for_display(rows: list[dict]) -> list[dict]:
    return sorted((row for row in rows if isinstance(row, dict)), key=_event_display_sort_key)


def _event_display_sort_key(event: dict) -> tuple[int, str, str, str]:
    priority = str(event.get("priority") or "LOW").upper()
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(priority, 9)
    return (
        priority_rank,
        str(event.get("date") or ""),
        str(event.get("time_et") or ""),
        str(event.get("event") or ""),
    )


def _event_identity(event: dict) -> tuple[str, str, str, str]:
    return (
        str(event.get("date") or "").strip(),
        str(event.get("time_et") or "").strip(),
        str(event.get("event") or "").strip().upper(),
        str(event.get("currency") or "").strip().upper(),
    )


def _sort_earnings_for_display(rows: list[dict]) -> list[dict]:
    return sorted((row for row in rows if isinstance(row, dict)), key=_earnings_display_sort_key)


def _earnings_display_sort_key(row: dict) -> tuple[int, int, int, str, str]:
    market_cap = _market_cap_value(row)
    importance = str(row.get("importance") or "UNKNOWN").upper()
    importance_rank = {"MEGA": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}.get(importance, 9)
    return (
        1 if market_cap <= 0 else 0,
        -market_cap,
        importance_rank,
        str(row.get("date") or ""),
        str(row.get("ticker") or ""),
    )


def _market_cap_value(row: dict) -> int:
    try:
        value = row.get("market_cap") if isinstance(row, dict) else None
        if value in (None, ""):
            return 0
        numeric = int(float(value))
        return numeric if numeric > 0 else 0
    except (TypeError, ValueError):
        return 0


def _market_cap_text(row: dict) -> str:
    explicit = str(row.get("market_cap_fmt") or "").strip() if isinstance(row, dict) else ""
    if explicit:
        return explicit
    market_cap = _market_cap_value(row)
    if market_cap <= 0:
        return "Market cap unknown"
    amount = float(market_cap)
    for suffix, divisor in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)):
        if abs(amount) >= divisor:
            return f"${amount / divisor:.2f}{suffix}"
    return f"${amount:,.0f}"


def _earnings_source_text(row: dict) -> str:
    raw_source = str(row.get("source") or "manual").strip() or "manual"
    source = raw_source.upper() if raw_source.lower() == "ibkr" else raw_source.capitalize()
    metadata_source = str(row.get("metadata_source") or "").strip()
    if metadata_source and metadata_source.lower() not in source.lower():
        source = f"{source}/{metadata_source}"
    return source


def _earnings_identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("date") or "").strip(),
        str(row.get("time") or "").strip().upper(),
        str(row.get("ticker") or "").strip().upper(),
    )


def _is_high_or_mega_earnings(row: dict) -> bool:
    return str(row.get("importance") or "").upper() in {"MEGA", "HIGH"} or _market_cap_value(row) >= 200_000_000_000


def _is_notable_earnings(row: dict) -> bool:
    return str(row.get("importance") or "").upper() in {"MEGA", "HIGH", "MEDIUM"} or _market_cap_value(row) >= 25_000_000_000


def _watchlist_earnings_line(row: dict) -> str:
    ticker = str(row.get("ticker") or "").strip().upper()
    company = str(row.get("company") or "").strip()
    source_list = str(row.get("source_list") or ", ".join(row.get("source_lists") or [])).strip()
    market_cap = str(row.get("market_cap_fmt") or _market_cap_text(row)).strip()
    sector = str(row.get("sector") or row.get("industry") or "").strip()
    classification = str(row.get("classification") or "").strip()
    reason = str(row.get("reason") or "").strip()
    return "- " + " | ".join(
        part if part else "n/a"
        for part in (ticker, company, source_list, market_cap, sector, classification, reason)
    )


def _dedupe_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for line in lines:
        text = str(line or "").strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def _sort_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _major_earnings_payload(payload: dict) -> dict:
    rows = _sort_earnings_for_display([
        row for row in _payload_rows(payload, "earnings")
        if _is_major_earnings(row)
    ])
    return {
        "generated_at": payload.get("generated_at") if isinstance(payload, dict) else "",
        "source": payload.get("source") if isinstance(payload, dict) else "manual",
        "start_date": payload.get("start_date") if isinstance(payload, dict) else "n/a",
        "end_date": payload.get("end_date") if isinstance(payload, dict) else "n/a",
        "earnings": rows,
        "yfinance_status": payload.get("yfinance_status") if isinstance(payload, dict) else {},
        "message": "" if rows else "No configured major earnings found.",
    }


def _is_major_earnings(row: dict) -> bool:
    return str(row.get("importance") or "").upper() in {"MEGA", "HIGH"} or _market_cap_value(row) >= 200_000_000_000 or _is_major_mega_cap_earnings(row)


def _is_major_mega_cap_earnings(row: dict) -> bool:
    ticker = str(row.get("ticker") or "").strip().upper()
    return ticker in MEGA_CAP_TICKERS or _market_cap_value(row) >= 1_000_000_000_000


def _is_key_weekly_risk_event(event: dict) -> bool:
    event_name = str(event.get("event") or "").upper()
    return any(term in event_name for term in KEY_WEEKLY_RISK_EVENT_TERMS)


def _clustered_major_event_dates(economic_calendar: dict, major_earnings: dict) -> list[str]:
    counts: dict[str, int] = {}
    for event in _payload_rows(economic_calendar, "events"):
        if _is_key_weekly_risk_event(event) or str(event.get("priority") or "").upper() == "HIGH":
            event_date = str(event.get("date") or "").strip()
            if event_date:
                counts[event_date] = counts.get(event_date, 0) + 1
    for row in _payload_rows(major_earnings, "earnings"):
        if _is_major_earnings(row):
            earnings_date = str(row.get("date") or "").strip()
            if earnings_date:
                counts[earnings_date] = counts.get(earnings_date, 0) + 1
    clustered = {date_value for date_value, count in counts.items() if count >= 2}
    dated_values = []
    for date_value in counts:
        try:
            dated_values.append((datetime.fromisoformat(date_value).date(), date_value))
        except ValueError:
            continue
    dated_values.sort()
    for index, (left_date, left_value) in enumerate(dated_values):
        for right_date, right_value in dated_values[index + 1:]:
            day_gap = (right_date - left_date).days
            if day_gap > 2:
                break
            clustered.add(left_value)
            clustered.add(right_value)
    return sorted(clustered)
