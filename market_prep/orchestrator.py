from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .config_loader import load_market_prep_config, set_forexfactory_enabled
from .logging_setup import get_market_prep_logger
from .report_builder import (
    build_daily_report_object,
    build_economic_calendar_report,
    build_earnings_report,
    build_placeholder_report,
    build_rss_news_report,
    build_watchlist_risk_report,
    build_weekly_report_object,
    build_youtube_links_report,
)
from .services.economic_calendar_service import (
    get_economic_events_for_date,
    get_upcoming_economic_events,
)
from .services.earnings_service import (
    get_earnings_for_today_and_tomorrow,
    get_upcoming_earnings,
)
from .services.fed_calendar_service import get_fed_calendar_events
from .services.ai_service import build_market_prep_ai_brief
from .services.prices_service import fetch_market_snapshot
from .services.rss_news_service import fetch_rss_headlines
from .services.sec_service import get_sec_filing_risk
from .services.treasury_calendar_service import get_treasury_calendar_events
from .services.watchlist_service import scan_watchlist_risk
from .services.youtube_rss_service import fetch_youtube_links


class MarketPrepOrchestrator:
    """Coordinator for no-paid-API Market Prep scheduled-risk workflows."""

    def __init__(self):
        self.config = load_market_prep_config()
        self.logger = get_market_prep_logger()

    def run_placeholder(self, action: str) -> dict[str, Any]:
        self.logger.info("Market Prep placeholder action invoked: %s", action)
        return {
            "action": action,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "report": build_placeholder_report(action, self.config),
        }

    def run_daily_prep(self) -> dict[str, Any]:
        self.logger.info("Market Prep daily prep started.")
        generated_at = datetime.now().isoformat(timespec="seconds")
        prep_date = resolve_daily_prep_date()
        next_7_events = self._load_upcoming_events(days=7, start_date=prep_date)
        fed_calendar = self._load_fed_calendar(days=7, start_date=prep_date)
        treasury_calendar = self._load_treasury_calendar(days=7, start_date=prep_date)
        todays_events = self._load_todays_events(target_date=prep_date)
        today_tomorrow_earnings = self._load_today_tomorrow_earnings(target_date=prep_date)
        next_7_earnings = self._load_upcoming_earnings(days=7, start_date=prep_date)
        watchlist_risk = self._load_watchlist_risk(
            todays_events=todays_events,
            today_tomorrow_earnings=today_tomorrow_earnings,
            upcoming_earnings=self._load_upcoming_earnings(days=14, start_date=prep_date),
            start_date=prep_date,
        )
        watchlist_tickers = _watchlist_tickers(watchlist_risk)
        sec_filings = self._load_sec_filings(start_date=prep_date, tickers=watchlist_tickers)
        rss_headlines = self._load_rss_headlines(limit=25, tickers=watchlist_tickers)
        youtube_links = self._load_youtube_links(limit=25)
        market_snapshot = self._load_market_snapshot()
        daily_report = build_daily_report_object(
            todays_events=todays_events,
            next_7_events=next_7_events,
            today_tomorrow_earnings=today_tomorrow_earnings,
            next_7_earnings=next_7_earnings,
            watchlist_risk=watchlist_risk,
            rss_headlines=rss_headlines,
            youtube_links=youtube_links,
            fed_calendar=fed_calendar,
            treasury_calendar=treasury_calendar,
            sec_filings=sec_filings,
            market_snapshot=market_snapshot,
            generated_at=generated_at,
            report_date=prep_date.isoformat(),
        )
        daily_report = self._attach_ai_brief(daily_report)
        return {
            "action": "Run Daily Prep",
            "generated_at": generated_at,
            "prep_date": prep_date.isoformat(),
            "todays_events": todays_events,
            "next_7_events": next_7_events,
            "fed_calendar": fed_calendar,
            "treasury_calendar": treasury_calendar,
            "today_tomorrow_earnings": today_tomorrow_earnings,
            "next_7_earnings": next_7_earnings,
            "watchlist_risk": watchlist_risk,
            "sec_filings": sec_filings,
            "rss_headlines": rss_headlines,
            "youtube_links": youtube_links,
            "market_snapshot": market_snapshot,
            "daily_report": daily_report,
            "report": daily_report["markdown"],
        }

    def run_weekly_prep(self) -> dict[str, Any]:
        self.logger.info("Market Prep weekly prep started.")
        generated_at = datetime.now().isoformat(timespec="seconds")
        weekly_window = resolve_weekly_prep_window()
        week_start = weekly_window["week_start"]
        window_start = weekly_window["window_start"]
        window_end = weekly_window["week_end"]
        days_remaining = max(0, (window_end - window_start).days)
        economic_calendar = self._load_upcoming_events(days=days_remaining, start_date=window_start)
        fed_calendar = self._load_fed_calendar(days=days_remaining, start_date=window_start)
        treasury_calendar = self._load_treasury_calendar(days=days_remaining, start_date=window_start)
        earnings_calendar = self._load_upcoming_earnings(days=days_remaining, start_date=window_start)
        todays_events = self._load_todays_events(target_date=window_start)
        today_tomorrow_earnings = self._load_today_tomorrow_earnings(target_date=window_start)
        watchlist_risk = self._load_watchlist_risk(
            todays_events=todays_events,
            today_tomorrow_earnings=today_tomorrow_earnings,
            upcoming_earnings=self._load_upcoming_earnings(days=14, start_date=window_start),
            start_date=window_start,
        )
        watchlist_tickers = _watchlist_tickers(watchlist_risk)
        sec_filings = self._load_sec_filings(tickers=watchlist_tickers, start_date=window_start)
        rss_headlines = self._load_rss_headlines(limit=25, tickers=watchlist_tickers)
        youtube_links = self._load_youtube_links(limit=25)
        market_snapshot = self._load_market_snapshot()
        weekly_report = build_weekly_report_object(
            economic_calendar=economic_calendar,
            earnings_calendar=earnings_calendar,
            watchlist_risk=watchlist_risk,
            rss_headlines=rss_headlines,
            youtube_links=youtube_links,
            fed_calendar=fed_calendar,
            treasury_calendar=treasury_calendar,
            sec_filings=sec_filings,
            market_snapshot=market_snapshot,
            report_date=week_start.isoformat(),
            generated_at=generated_at,
        )
        weekly_report = self._attach_ai_brief(weekly_report)
        return {
            "action": "Run Weekly Prep",
            "generated_at": generated_at,
            "week_start": week_start.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "economic_calendar": economic_calendar,
            "fed_calendar": fed_calendar,
            "treasury_calendar": treasury_calendar,
            "earnings_calendar": earnings_calendar,
            "watchlist_risk": watchlist_risk,
            "sec_filings": sec_filings,
            "rss_headlines": rss_headlines,
            "youtube_links": youtube_links,
            "market_snapshot": market_snapshot,
            "weekly_report": weekly_report,
            "report": weekly_report["markdown"],
        }

    def start_day_prep(self) -> dict[str, Any]:
        return self.run_daily_prep()

    def start_week_prep(self) -> dict[str, Any]:
        return self.run_weekly_prep()

    def refresh_economic_calendar(self, days: int = 7) -> dict[str, Any]:
        self.logger.info("Refreshing manual economic calendar for next %s day(s).", days)
        payload = self._load_upcoming_events(days=days, refresh_forexfactory=True)
        return {
            "action": "Refresh Economic Calendar",
            "generated_at": payload.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
            "economic_calendar": payload,
            "report": build_economic_calendar_report(payload),
        }

    def is_forexfactory_enabled(self) -> bool:
        return bool(self.config.features.get("forexfactory_calendar")) and bool(
            self.config.forexfactory.get("enabled")
        )

    def set_forexfactory_enabled(self, enabled: bool) -> dict[str, Any]:
        self.config = set_forexfactory_enabled(enabled)
        state = "enabled" if self.is_forexfactory_enabled() else "disabled"
        self.logger.info("ForexFactory calendar %s from GUI.", state)
        return {
            "enabled": self.is_forexfactory_enabled(),
            "state": state,
            "config_path": str(self.config.config_path or ""),
        }

    def refresh_earnings(self, days: int = 7) -> dict[str, Any]:
        self.logger.info("Refreshing earnings calendar for next %s day(s).", days)
        upcoming = self._load_upcoming_earnings(days=days)
        today_tomorrow = self._load_today_tomorrow_earnings()
        watchlist_risk = self._load_watchlist_risk(
            todays_events=self._load_todays_events(),
            today_tomorrow_earnings=today_tomorrow,
            upcoming_earnings=self._load_upcoming_earnings(days=14),
        )
        report = (
            build_earnings_report(today_tomorrow, title="Earnings Today/Tomorrow")
            + "\n\n"
            + build_earnings_report(upcoming, title="Earnings Next 7 Days")
            + "\n\n"
            + build_watchlist_risk_report(watchlist_risk, title="Watchlist Earnings Risk")
        ).rstrip()
        return {
            "action": "Refresh Earnings",
            "generated_at": upcoming.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
            "earnings": upcoming,
            "today_tomorrow_earnings": today_tomorrow,
            "watchlist_risk": watchlist_risk,
            "report": report,
        }

    def scan_watchlists(self) -> dict[str, Any]:
        self.logger.info("Scanning Market Prep watchlists for scheduled risk.")
        todays_events = self._load_todays_events()
        today_tomorrow_earnings = self._load_today_tomorrow_earnings()
        upcoming_earnings = self._load_upcoming_earnings(days=14)
        payload = self._load_watchlist_risk(
            todays_events=todays_events,
            today_tomorrow_earnings=today_tomorrow_earnings,
            upcoming_earnings=upcoming_earnings,
        )
        return {
            "action": "Scan Watchlists",
            "generated_at": payload.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
            "watchlist_risk": payload,
            "report": build_watchlist_risk_report(payload),
        }

    def refresh_news(self, limit: int = 25) -> dict[str, Any]:
        self.logger.info("Refreshing RSS macro headlines.")
        payload = self._load_rss_headlines(limit=limit, tickers=self._load_watchlist_tickers())
        return {
            "action": "Refresh News",
            "generated_at": payload.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
            "rss_headlines": payload,
            "report": build_rss_news_report(payload),
        }

    def refresh_youtube_links(self, limit: int = 25) -> dict[str, Any]:
        self.logger.info("Refreshing YouTube RSS links.")
        payload = self._load_youtube_links(limit=limit)
        return {
            "action": "Refresh YouTube Links",
            "generated_at": payload.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
            "youtube_links": payload,
            "report": build_youtube_links_report(payload),
        }

    def _load_todays_events(
        self,
        *,
        target_date=None,
        refresh_forexfactory: bool = False,
    ) -> dict[str, Any]:
        try:
            return get_economic_events_for_date(
                target_date=target_date,
                config=self.config,
                refresh_forexfactory=refresh_forexfactory,
            )
        except Exception as exc:
            self.logger.exception("Failed loading today's economic events.")
            today = datetime.now().date()
            return _event_error_payload(today, today, exc)

    def _load_upcoming_events(
        self,
        *,
        days: int,
        start_date=None,
        refresh_forexfactory: bool = False,
    ) -> dict[str, Any]:
        try:
            return get_upcoming_economic_events(
                start_date=start_date,
                days=days,
                config=self.config,
                refresh_forexfactory=refresh_forexfactory,
            )
        except Exception as exc:
            self.logger.exception("Failed loading upcoming economic events.")
            start = datetime.now().date()
            return _event_error_payload(start, start + timedelta(days=max(0, int(days))), exc)

    def _load_current_week_events(self, *, refresh_forexfactory: bool = False) -> dict[str, Any]:
        weekly_window = resolve_weekly_prep_window()
        days_remaining = max(0, (weekly_window["week_end"] - weekly_window["window_start"]).days)
        return self._load_upcoming_events(
            days=days_remaining,
            start_date=weekly_window["window_start"],
            refresh_forexfactory=refresh_forexfactory,
        )

    def _load_today_tomorrow_earnings(self, *, target_date=None) -> dict[str, Any]:
        try:
            return get_earnings_for_today_and_tomorrow(target_date=target_date, config=self.config)
        except Exception as exc:
            self.logger.exception("Failed loading today/tomorrow earnings.")
            start = target_date or datetime.now().date()
            return _earnings_error_payload(start, start + timedelta(days=1), exc)

    def _load_upcoming_earnings(self, *, days: int, start_date=None) -> dict[str, Any]:
        try:
            return get_upcoming_earnings(start_date=start_date, days=days, config=self.config)
        except Exception as exc:
            self.logger.exception("Failed loading upcoming earnings.")
            start = start_date or datetime.now().date()
            return _earnings_error_payload(start, start + timedelta(days=max(0, int(days))), exc)

    def _load_current_week_earnings(self) -> dict[str, Any]:
        weekly_window = resolve_weekly_prep_window()
        days_remaining = max(0, (weekly_window["week_end"] - weekly_window["window_start"]).days)
        return self._load_upcoming_earnings(days=days_remaining, start_date=weekly_window["window_start"])

    def _load_fed_calendar(self, *, days: int, start_date=None) -> dict[str, Any]:
        try:
            return get_fed_calendar_events(self.config, start_date=start_date, days_ahead=days)
        except Exception as exc:
            self.logger.exception("Failed loading Fed calendar.")
            start = start_date or datetime.now().date()
            return _calendar_error_payload("Federal Reserve", start, start + timedelta(days=max(0, int(days))), exc)

    def _load_treasury_calendar(self, *, days: int, start_date=None) -> dict[str, Any]:
        try:
            return get_treasury_calendar_events(self.config, start_date=start_date, days_ahead=days)
        except Exception as exc:
            self.logger.exception("Failed loading Treasury calendar.")
            start = start_date or datetime.now().date()
            return _calendar_error_payload(
                "U.S. Treasury FiscalData",
                start,
                start + timedelta(days=max(0, int(days))),
                exc,
            )

    def _load_sec_filings(self, *, tickers: list[str] | None = None, start_date=None) -> dict[str, Any]:
        try:
            return get_sec_filing_risk(self.config, tickers=tickers, start_date=start_date)
        except Exception as exc:
            self.logger.exception("Failed loading SEC filing risk.")
            end = start_date or datetime.now().date()
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "SEC EDGAR",
                "start_date": (end - timedelta(days=7)).isoformat(),
                "end_date": end.isoformat(),
                "tickers": tickers or [],
                "filings": [],
                "status": "failed",
                "status_label": "Failed",
                "warnings": [str(exc)],
                "message": f"SEC filing scan unavailable: {exc}",
            }

    def _load_watchlist_risk(
        self,
        *,
        todays_events: dict,
        today_tomorrow_earnings: dict,
        upcoming_earnings: dict,
        start_date=None,
    ) -> dict[str, Any]:
        try:
            return scan_watchlist_risk(
                self.config,
                todays_events=todays_events,
                today_tomorrow_earnings=today_tomorrow_earnings,
                upcoming_earnings=upcoming_earnings,
                start_date=start_date,
                days_ahead=14,
            )
        except Exception as exc:
            self.logger.exception("Failed scanning watchlist risk.")
            start = start_date or datetime.now().date()
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "manual",
                "start_date": start.isoformat(),
                "end_date": (start + timedelta(days=14)).isoformat(),
                "tickers": [],
                "risks": [],
                "missing_files": [],
                "message": f"No watchlist tickers found. Error: {exc}",
            }

    def _load_rss_headlines(self, *, limit: int, tickers: list[str] | None = None) -> dict[str, Any]:
        try:
            return fetch_rss_headlines(limit=limit, config=self.config, tickers=tickers)
        except Exception as exc:
            self.logger.exception("Failed refreshing RSS headlines.")
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "rss",
                "headlines": [],
                "warnings": [str(exc)],
                "message": f"No RSS headlines found. Error: {exc}",
            }

    def _load_watchlist_tickers(self) -> list[str]:
        payload = self._load_watchlist_risk(
            todays_events=self._load_todays_events(),
            today_tomorrow_earnings=self._load_today_tomorrow_earnings(),
            upcoming_earnings=self._load_upcoming_earnings(days=14),
        )
        return _watchlist_tickers(payload)

    def _load_youtube_links(self, *, limit: int) -> dict[str, Any]:
        try:
            return fetch_youtube_links(limit=limit)
        except Exception as exc:
            self.logger.exception("Failed refreshing YouTube RSS links.")
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "youtube_rss",
                "videos": [],
                "warnings": [str(exc)],
                "message": f"No configured YouTube links found. Error: {exc}",
            }

    def _load_market_snapshot(self) -> dict[str, Any]:
        try:
            return fetch_market_snapshot()
        except Exception as exc:
            self.logger.exception("Failed loading market snapshot.")
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "yfinance",
                "classification": {
                    "label": "Noisy",
                    "reason": f"Market snapshot unavailable: {exc}",
                },
                "rows": [],
                "errors": [str(exc)],
            }

    def _attach_ai_brief(self, report: dict[str, Any]) -> dict[str, Any]:
        try:
            ai_brief = build_market_prep_ai_brief(report, config=self.config)
        except Exception as exc:
            self.logger.exception("Failed building Market Prep AI brief.")
            ai_brief = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "openai",
                "status": "failed",
                "status_label": f"AI brief failed: {exc}",
                "summary": "",
                "prompt": "",
                "warnings": [str(exc)],
            }
        report = dict(report)
        report["ai_brief"] = ai_brief
        if str(report.get("report_type") or "").lower() == "weekly":
            from .report_builder import build_weekly_markdown

            report["markdown"] = build_weekly_markdown(report)
        else:
            from .report_builder import build_daily_markdown

            report["markdown"] = build_daily_markdown(report)
        return report

    def export_daily_markdown(self, daily_report: dict) -> dict[str, Any]:
        return self.export_markdown(daily_report)

    def export_markdown(self, report: dict) -> dict[str, Any]:
        if not isinstance(report, dict):
            raise ValueError("Run Daily Prep or Weekly Prep before exporting markdown.")
        markdown = str(report.get("markdown") or "").strip()
        if not markdown:
            raise ValueError("Run Daily Prep or Weekly Prep before exporting markdown.")

        report_date = str(report.get("report_date") or datetime.now().date().isoformat())
        export_prefix = str(report.get("export_prefix") or "").strip()
        if not export_prefix:
            report_type = str(report.get("report_type") or "daily").strip().lower()
            export_prefix = "weekly_market_prep" if report_type == "weekly" else "daily_market_prep"
        output_dir = self.config.resolved_paths().get("output_dir") or Path("output")
        output_path = output_dir / f"{export_prefix}_{report_date}.md"
        try:
            _write_text_atomic(output_path, markdown.rstrip() + "\n")
            self.logger.info("Exported Market Prep markdown to %s", output_path)
        except Exception:
            self.logger.exception("Failed exporting Market Prep markdown.")
            raise
        return {
            "path": str(output_path),
            "report_date": report_date,
            "report_type": str(report.get("report_type") or ""),
            "markdown": markdown,
        }


def resolve_daily_prep_date(reference=None):
    if reference is None:
        target = datetime.now().date()
    elif isinstance(reference, datetime):
        target = reference.date()
    else:
        target = reference

    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def resolve_weekly_prep_window(reference=None) -> dict[str, date]:
    if reference is None:
        target = datetime.now().date()
    elif isinstance(reference, datetime):
        target = reference.date()
    else:
        target = reference

    if target.weekday() >= 5:
        next_monday = target + timedelta(days=7 - target.weekday())
        return {
            "week_start": next_monday,
            "window_start": next_monday,
            "week_end": next_monday + timedelta(days=6),
        }

    week_start = target - timedelta(days=target.weekday())
    return {
        "week_start": week_start,
        "window_start": target,
        "week_end": week_start + timedelta(days=6),
    }


def _event_error_payload(start: datetime.date, end: datetime.date, exc: Exception) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "manual",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "events": [],
        "message": f"No configured economic events found. Error: {exc}",
    }


def _earnings_error_payload(start: datetime.date, end: datetime.date, exc: Exception) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "manual",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "earnings": [],
        "message": f"No configured earnings found. Error: {exc}",
    }


def _calendar_error_payload(source: str, start: datetime.date, end: datetime.date, exc: Exception) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "events": [],
        "status": "failed",
        "status_label": "Failed",
        "warnings": [str(exc)],
        "message": f"{source} calendar unavailable: {exc}",
    }


def _current_week_start(value):
    if value.weekday() == 6:
        return value
    return value - timedelta(days=value.weekday())


def _watchlist_tickers(payload: dict[str, Any]) -> list[str]:
    values = payload.get("tickers") if isinstance(payload, dict) else []
    if not isinstance(values, list):
        return []
    tickers = []
    seen = set()
    for value in values:
        ticker = str(value or "").strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def _parse_date_or_default(value, default):
    try:
        text = str(value or "").strip()
        return datetime.fromisoformat(text).date() if text else default
    except ValueError:
        return default


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
