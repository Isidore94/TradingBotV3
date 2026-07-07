from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from project_paths import DATA_DIR, EARNINGS_CALENDAR_HISTORY_FILE
except ImportError:  # pragma: no cover - used when imported as scripts.earnings_history
    from scripts.project_paths import DATA_DIR, EARNINGS_CALENDAR_HISTORY_FILE


SCHEMA_VERSION = 1
DEFAULT_HISTORY_FILE = EARNINGS_CALENDAR_HISTORY_FILE
CONFIRMED_SESSIONS = {"BMO", "AMC"}
SOURCE_PRIORITY = {"yfinance": 1, "nasdaq": 2, "manual": 3}
CONFIDENCE_PRIORITY = {"unknown": 0, "supplemental": 1, "inferred": 2, "confirmed": 3}
ACTIVE_FUTURE_SUPERSESSION_WINDOW_DAYS = 120


def normalize_release_session(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "TBD"
    if text in {"bmo", "before_market_open", "before market open", "pre", "pre-market", "premarket"}:
        return "BMO"
    if text in {"amc", "after_market_close", "after market close", "post", "post-market", "after-hours", "after hours"}:
        return "AMC"
    if "pre" in text or "before" in text:
        return "BMO"
    if "after" in text or "post" in text:
        return "AMC"
    if "not" in text and "supplied" in text:
        return "TBD"
    if text in {"tbd", "unknown", "time-not-supplied", "not supplied", "na", "n/a"}:
        return "TBD"
    return str(value or "").strip().upper() or "TBD"


def normalize_nasdaq_event(row: dict[str, Any], event_date: date | str) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    parsed_date = parse_date(event_date)
    ticker = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
    if parsed_date is None or not ticker:
        return None
    release_session = normalize_release_session(row.get("time"))
    market_cap = parse_market_cap(row.get("marketCap") or row.get("market_cap"))
    return normalize_event(
        {
            "ticker": ticker,
            "earnings_date": parsed_date.isoformat(),
            "release_session": release_session,
            "source": "nasdaq",
            "source_confidence": "confirmed" if release_session in CONFIRMED_SESSIONS else "unknown",
            "release_session_source": "nasdaq" if release_session in CONFIRMED_SESSIONS else "",
            "company": str(row.get("name") or row.get("company") or "").strip(),
            "market_cap": market_cap,
            "fiscal_quarter": str(row.get("fiscalQuarterEnding") or "").strip(),
            "eps_forecast": str(row.get("epsForecast") or "").strip(),
            "estimate_count": str(row.get("noOfEsts") or "").strip(),
            "last_year_report_date": str(row.get("lastYearRptDt") or "").strip(),
            "last_year_eps": str(row.get("lastYearEPS") or "").strip(),
            "raw_provider_fields": {"nasdaq": row},
        }
    )


def normalize_manual_event(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    parsed_date = parse_date(row.get("earnings_date") or row.get("date"))
    ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
    if parsed_date is None or not ticker:
        return None
    release_session = normalize_release_session(row.get("release_session") or row.get("time"))
    return normalize_event(
        {
            "ticker": ticker,
            "earnings_date": parsed_date.isoformat(),
            "release_session": release_session,
            "source": "manual",
            "source_confidence": "confirmed" if release_session in CONFIRMED_SESSIONS else "unknown",
            "release_session_source": "manual" if release_session in CONFIRMED_SESSIONS else "",
            "company": str(row.get("company") or "").strip(),
            "market_cap": parse_market_cap(row.get("market_cap")),
            "fiscal_quarter": str(row.get("fiscal_quarter") or row.get("fiscal_quarter_ending") or "").strip(),
            "eps_forecast": str(row.get("eps_forecast") or "").strip(),
            "estimate_count": str(row.get("estimate_count") or "").strip(),
            "last_year_report_date": str(row.get("last_year_report_date") or "").strip(),
            "last_year_eps": str(row.get("last_year_eps") or "").strip(),
            "raw_provider_fields": {"manual": row},
        }
    )


def normalize_yfinance_event(row: dict[str, Any], symbol: str | None = None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    parsed_date = parse_date(row.get("earnings_date") or row.get("date"))
    ticker = str(symbol or row.get("ticker") or row.get("symbol") or "").strip().upper()
    if parsed_date is None or not ticker:
        return None
    return normalize_event(
        {
            "ticker": ticker,
            "earnings_date": parsed_date.isoformat(),
            "release_session": normalize_release_session(row.get("release_session") or row.get("time")),
            "source": "yfinance",
            "source_confidence": "supplemental",
            "company": str(row.get("company") or "").strip(),
            "market_cap": parse_market_cap(row.get("market_cap")),
            "fiscal_quarter": str(row.get("fiscal_quarter") or row.get("fiscal_quarter_ending") or "").strip(),
            "eps_forecast": str(row.get("eps_forecast") or "").strip(),
            "raw_provider_fields": {"yfinance": row},
        }
    )


def normalize_event(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    parsed_date = parse_date(row.get("earnings_date") or row.get("date"))
    ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
    if parsed_date is None or not ticker:
        return None
    source = str(row.get("source") or "unknown").strip().lower() or "unknown"
    session = normalize_release_session(row.get("release_session") or row.get("time"))
    confidence = str(row.get("source_confidence") or "").strip().lower()
    if confidence not in CONFIDENCE_PRIORITY:
        confidence = "confirmed" if session in CONFIRMED_SESSIONS and source in {"manual", "nasdaq"} else "unknown"
    raw_fields = row.get("raw_provider_fields")
    if not isinstance(raw_fields, dict):
        raw_fields = {source: row.get("raw") or {}}
    return {
        "ticker": ticker,
        "earnings_date": parsed_date.isoformat(),
        "release_session": session if session in CONFIRMED_SESSIONS else "TBD",
        "source": source,
        "source_confidence": confidence,
        "release_session_source": str(row.get("release_session_source") or "").strip(),
        "inferred_release_session": str(row.get("inferred_release_session") or "").strip(),
        "company": str(row.get("company") or "").strip(),
        "market_cap": parse_market_cap(row.get("market_cap")),
        "fiscal_quarter": str(row.get("fiscal_quarter") or row.get("fiscal_quarter_ending") or "").strip(),
        "eps_forecast": str(row.get("eps_forecast") or "").strip(),
        "estimate_count": str(row.get("estimate_count") or "").strip(),
        "last_year_report_date": str(row.get("last_year_report_date") or "").strip(),
        "last_year_eps": str(row.get("last_year_eps") or "").strip(),
        "sources": sorted({source, *[str(value).lower() for value in row.get("sources", []) if str(value).strip()]}),
        "raw_provider_fields": raw_fields,
        "first_seen_at": str(row.get("first_seen_at") or "").strip(),
        "last_seen_at": str(row.get("last_seen_at") or "").strip(),
    }


def load_history(path: Path | None = None) -> dict[str, Any]:
    target = Path(path or DEFAULT_HISTORY_FILE)
    if not target.exists():
        return _empty_history()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_history()
    if not isinstance(payload, dict):
        return _empty_history()
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("generated_at", "")
    symbols = payload.get("symbols")
    payload["symbols"] = symbols if isinstance(symbols, dict) else {}
    return payload


def save_history(history: dict[str, Any], path: Path | None = None) -> Path:
    target = Path(path or DEFAULT_HISTORY_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_history(history)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
            temp_path = Path(handle.name)
        _replace_with_retries(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return target


# Deferred-save support: the shared history is tens of MB on a Drive-synced
# folder, and several callers record rows inside per-date/per-symbol loops.
# Without batching, every merge_events call pays a full parse + re-serialize +
# re-upload of the file. While a deferral is active, merges edit one in-memory
# copy per path and the file is rewritten once when the outermost scope ends.
# Reads via load_history stay disk-backed, so end the deferral before reading
# merged results back. Single-threaded by design (matches the calendar-cache
# deferral in master_avwap_lib).
_DEFERRED_SAVE_DEPTH = 0
_DEFERRED_HISTORIES: dict[str, dict[str, Any]] = {}
_DEFERRED_DIRTY_KEYS: set[str] = set()


def _history_key(path: Path | None) -> str:
    return str(Path(path or DEFAULT_HISTORY_FILE))


def begin_deferred_history_save() -> None:
    """Start batching history writes; reentrant, pair every call with end via try/finally."""
    global _DEFERRED_SAVE_DEPTH
    _DEFERRED_SAVE_DEPTH += 1


def end_deferred_history_save() -> None:
    """Leave one deferral scope; the outermost end flushes every dirty history once."""
    global _DEFERRED_SAVE_DEPTH
    if _DEFERRED_SAVE_DEPTH <= 0:
        return
    _DEFERRED_SAVE_DEPTH -= 1
    if _DEFERRED_SAVE_DEPTH:
        return
    for key in sorted(_DEFERRED_DIRTY_KEYS):
        history = _DEFERRED_HISTORIES.get(key)
        if history is None:
            continue
        try:
            save_history(history, Path(key))
        except Exception as exc:
            logging.warning("Deferred earnings-history save failed for %s: %s", key, exc)
    _DEFERRED_HISTORIES.clear()
    _DEFERRED_DIRTY_KEYS.clear()


def _load_history_for_update(path: Path | None) -> dict[str, Any]:
    if not _DEFERRED_SAVE_DEPTH:
        return load_history(path)
    key = _history_key(path)
    history = _DEFERRED_HISTORIES.get(key)
    if history is None:
        history = load_history(path)
        _DEFERRED_HISTORIES[key] = history
    return history


def _save_history_after_update(history: dict[str, Any], path: Path | None) -> None:
    if not _DEFERRED_SAVE_DEPTH:
        save_history(history, path)
        return
    _DEFERRED_DIRTY_KEYS.add(_history_key(path))


def _event_signature(event: dict[str, Any] | None) -> str | None:
    """Stable content signature ignoring the last_seen_at bookkeeping stamp.

    Must be taken BEFORE _merge_event runs: the merge mutates the existing
    event's raw_provider_fields dict in place, so an after-the-fact comparison
    would never see raw-field changes.
    """
    if not isinstance(event, dict):
        return None
    trimmed = {key: value for key, value in event.items() if key != "last_seen_at"}
    return json.dumps(trimmed, sort_keys=True, default=str)


def merge_events(events: list[dict[str, Any]], path: Path | None = None, now: datetime | None = None) -> list[dict[str, Any]]:
    normalized_events = [normalize_event(event) for event in events or []]
    normalized_events = [event for event in normalized_events if event is not None]
    if not normalized_events:
        return []
    now_value = now or datetime.now()
    now_text = now_value.isoformat(timespec="seconds")
    reference_date = now_value.date()
    history = _load_history_for_update(path)
    merged_events: list[dict[str, Any]] = []
    symbols = history.setdefault("symbols", {})
    changed = False
    for event in normalized_events:
        ticker = event["ticker"]
        symbol_entry = symbols.setdefault(ticker, {"ticker": ticker, "events": [], "updated_at": ""})
        existing_events = symbol_entry.get("events") if isinstance(symbol_entry.get("events"), list) else []
        existing_by_date = {
            str(existing.get("earnings_date") or ""): existing
            for existing in existing_events
            if isinstance(existing, dict)
        }
        count_before_drop = len(existing_by_date)
        _drop_superseded_active_events(existing_by_date, event, reference_date)
        event_changed = len(existing_by_date) != count_before_drop
        existing = existing_by_date.get(event["earnings_date"])
        existing_signature = _event_signature(existing)
        merged = _merge_event(existing, event, now_text)
        if _event_signature(merged) != existing_signature:
            event_changed = True
        merged_events.append(dict(merged))
        if not event_changed:
            # Re-merge of already-known data: leave the stored entry untouched
            # so the file is not rewritten just to bump last_seen_at stamps.
            continue
        changed = True
        existing_by_date[event["earnings_date"]] = merged
        symbol_entry["events"] = sorted(
            existing_by_date.values(),
            key=lambda item: str(item.get("earnings_date") or ""),
            reverse=True,
        )
        symbol_entry["updated_at"] = now_text
    if changed:
        history["generated_at"] = now_text
        _save_history_after_update(history, path)
    return merged_events


def record_nasdaq_rows(rows: list[dict[str, Any]], event_date: date | str, path: Path | None = None) -> list[dict[str, Any]]:
    events = [normalize_nasdaq_event(row, event_date) for row in rows or []]
    return merge_events([event for event in events if event is not None], path=path)


def record_manual_rows(rows: list[dict[str, Any]], path: Path | None = None) -> list[dict[str, Any]]:
    events = [normalize_manual_event(row) for row in rows or []]
    return merge_events([event for event in events if event is not None], path=path)


def record_yfinance_rows(rows: list[dict[str, Any]], symbol: str | None = None, path: Path | None = None) -> list[dict[str, Any]]:
    events = [normalize_yfinance_event(row, symbol=symbol) for row in rows or []]
    return merge_events([event for event in events if event is not None], path=path)


def record_inferred_release_session(
    symbol: str,
    earnings_date: date | str,
    release_session: str,
    path: Path | None = None,
) -> dict[str, Any] | None:
    parsed_date = parse_date(earnings_date)
    ticker = str(symbol or "").strip().upper()
    session = normalize_release_session(release_session)
    if parsed_date is None or not ticker or session not in CONFIRMED_SESSIONS:
        return None
    detail = "inferred_bmo" if session == "BMO" else "inferred_amc"
    merged = merge_events(
        [
            {
                "ticker": ticker,
                "earnings_date": parsed_date.isoformat(),
                "release_session": session,
                "source": "gap_inference",
                "source_confidence": "inferred",
                "release_session_source": "gap_inference",
                "inferred_release_session": detail,
            }
        ],
        path=path,
    )
    return merged[0] if merged else None


def get_events_for_symbols(
    symbols: list[str],
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    wanted = {str(symbol or "").strip().upper() for symbol in symbols or [] if str(symbol or "").strip()}
    history = load_history(path)
    start = parse_date(start_date) if start_date is not None else None
    end = parse_date(end_date) if end_date is not None else None
    result: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in sorted(wanted)}
    for symbol in sorted(wanted):
        entry = history.get("symbols", {}).get(symbol)
        events = entry.get("events") if isinstance(entry, dict) else []
        for event in events if isinstance(events, list) else []:
            parsed_date = parse_date(event.get("earnings_date"))
            if parsed_date is None:
                continue
            if start is not None and parsed_date < start:
                continue
            if end is not None and parsed_date > end:
                continue
            result.setdefault(symbol, []).append(dict(event))
        result[symbol].sort(key=lambda item: str(item.get("earnings_date") or ""), reverse=True)
    return result


def get_events_in_window(
    start_date: date | str,
    end_date: date | str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    if start is None or end is None:
        return []
    history = load_history(path)
    rows: list[dict[str, Any]] = []
    for entry in history.get("symbols", {}).values():
        events = entry.get("events") if isinstance(entry, dict) else []
        for event in events if isinstance(events, list) else []:
            parsed_date = parse_date(event.get("earnings_date"))
            if parsed_date is not None and start <= parsed_date <= end:
                rows.append(dict(event))
    rows.sort(key=lambda item: (str(item.get("earnings_date") or ""), str(item.get("ticker") or "")))
    return rows


def get_past_dates_by_symbol(
    symbols: list[str],
    *,
    as_of: date | str | None = None,
    limit: int = 8,
    path: Path | None = None,
) -> dict[str, list[str]]:
    reference = parse_date(as_of) or datetime.now().date()
    events_by_symbol = get_events_for_symbols(symbols, end_date=reference, path=path)
    result: dict[str, list[str]] = {}
    for symbol, events in events_by_symbol.items():
        dates = []
        for event in events:
            event_date = str(event.get("earnings_date") or "").strip()
            if event_date and event_date not in dates:
                dates.append(event_date)
        result[symbol] = dates[: max(1, int(limit))]
    return result


def get_future_dates_by_symbol(
    symbols: list[str],
    *,
    start_date: date | str | None = None,
    lookahead_days: int = 63,
    path: Path | None = None,
) -> dict[str, list[str]]:
    start = parse_date(start_date) or datetime.now().date()
    end = start + timedelta(days=max(0, int(lookahead_days)))
    events_by_symbol = get_events_for_symbols(symbols, start_date=start, end_date=end, path=path)
    result: dict[str, list[str]] = {}
    for symbol, events in events_by_symbol.items():
        dates = sorted({str(event.get("earnings_date") or "").strip() for event in events if event.get("earnings_date")})
        result[symbol] = dates
    return result


def get_latest_events_by_symbol(
    symbols: list[str],
    *,
    as_of: date | str | None = None,
    path: Path | None = None,
) -> dict[str, dict[str, Any] | None]:
    reference = parse_date(as_of) or datetime.now().date()
    events_by_symbol = get_events_for_symbols(symbols, end_date=reference, path=path)
    result: dict[str, dict[str, Any] | None] = {}
    for symbol, events in events_by_symbol.items():
        result[symbol] = events[0] if events else None
    return result


def shared_event_to_market_prep_row(event: dict[str, Any]) -> dict[str, Any]:
    row = {
        "date": str(event.get("earnings_date") or ""),
        "time": normalize_release_session(event.get("release_session")),
        "ticker": str(event.get("ticker") or "").strip().upper(),
        "company": str(event.get("company") or "").strip(),
        "source": str(event.get("source") or "").strip() or "shared",
        "market_cap": parse_market_cap(event.get("market_cap")),
        "eps_forecast": str(event.get("eps_forecast") or "").strip(),
        "estimate_count": str(event.get("estimate_count") or "").strip(),
        "fiscal_quarter_ending": str(event.get("fiscal_quarter") or "").strip(),
        "last_year_report_date": str(event.get("last_year_report_date") or "").strip(),
        "last_year_eps": str(event.get("last_year_eps") or "").strip(),
        "release_session_source": str(event.get("release_session_source") or "").strip(),
        "source_confidence": str(event.get("source_confidence") or "").strip(),
    }
    if row["time"] not in CONFIRMED_SESSIONS:
        row["time"] = "TBD"
    return row


def parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def parse_market_cap(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = int(float(value))
        return numeric if numeric >= 0 else None
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "NA", "--", "NONE"}:
        return None
    cleaned = text.replace("$", "").replace(",", "").strip().upper()
    multiplier = 1.0
    if cleaned.endswith("T"):
        multiplier = 1_000_000_000_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("B"):
        multiplier = 1_000_000_000.0
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    try:
        numeric = int(float(cleaned) * multiplier)
    except ValueError:
        return None
    return numeric if numeric >= 0 else None


def _empty_history() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "",
        "symbols": {},
    }


def _replace_with_retries(source: Path, target: Path, attempts: int = 6) -> None:
    last_error: OSError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.08 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _normalize_history(history: dict[str, Any]) -> dict[str, Any]:
    payload = _empty_history()
    if isinstance(history, dict):
        payload["generated_at"] = str(history.get("generated_at") or datetime.now().isoformat(timespec="seconds"))
        raw_symbols = history.get("symbols") if isinstance(history.get("symbols"), dict) else {}
        for raw_symbol, raw_entry in raw_symbols.items():
            symbol = str(raw_symbol or "").strip().upper()
            if not symbol or not isinstance(raw_entry, dict):
                continue
            events = [normalize_event(event) for event in raw_entry.get("events", []) if isinstance(event, dict)]
            events = [event for event in events if event is not None]
            if not events:
                continue
            payload["symbols"][symbol] = {
                "ticker": symbol,
                "updated_at": str(raw_entry.get("updated_at") or payload["generated_at"]),
                "events": sorted(events, key=lambda item: item["earnings_date"], reverse=True),
            }
    return payload


def _merge_event(existing: dict[str, Any] | None, new: dict[str, Any], now_text: str) -> dict[str, Any]:
    if not isinstance(existing, dict):
        merged = dict(new)
        merged["first_seen_at"] = now_text
        merged["last_seen_at"] = now_text
        return merged
    merged = dict(existing)
    merged["last_seen_at"] = now_text
    merged["sources"] = sorted(
        {
            *[str(value).lower() for value in merged.get("sources", []) if str(value).strip()],
            *[str(value).lower() for value in new.get("sources", []) if str(value).strip()],
            str(new.get("source") or "").strip().lower(),
            str(merged.get("source") or "").strip().lower(),
        }
        - {""}
    )
    raw_fields = merged.get("raw_provider_fields") if isinstance(merged.get("raw_provider_fields"), dict) else {}
    new_raw = new.get("raw_provider_fields") if isinstance(new.get("raw_provider_fields"), dict) else {}
    raw_fields.update({key: value for key, value in new_raw.items() if value not in ({}, None, "")})
    merged["raw_provider_fields"] = raw_fields

    for field in (
        "company",
        "market_cap",
        "fiscal_quarter",
        "eps_forecast",
        "estimate_count",
        "last_year_report_date",
        "last_year_eps",
    ):
        if new.get(field) not in (None, "") and (merged.get(field) in (None, "") or _new_source_is_better(merged, new)):
            merged[field] = new.get(field)

    if _new_session_is_better(merged, new):
        for field in ("release_session", "source", "source_confidence", "release_session_source", "inferred_release_session"):
            merged[field] = new.get(field, "")
    elif _new_source_is_better(merged, new) and merged.get("release_session") not in CONFIRMED_SESSIONS:
        merged["source"] = new.get("source", merged.get("source", ""))
        merged["source_confidence"] = new.get("source_confidence", merged.get("source_confidence", ""))

    merged.setdefault("first_seen_at", existing.get("first_seen_at") or now_text)
    return normalize_event(merged) or merged


def _drop_superseded_active_events(
    existing_by_date: dict[str, dict[str, Any]],
    new: dict[str, Any],
    reference_date: date,
) -> None:
    new_date = parse_date(new.get("earnings_date"))
    if new_date is None or new_date < reference_date:
        return
    for existing_date_text, existing in list(existing_by_date.items()):
        if existing_date_text == new.get("earnings_date"):
            continue
        if _new_event_supersedes_existing_active_date(existing, new, reference_date):
            existing_by_date.pop(existing_date_text, None)


def _new_event_supersedes_existing_active_date(
    existing: dict[str, Any],
    new: dict[str, Any],
    reference_date: date,
) -> bool:
    existing_date = parse_date(existing.get("earnings_date"))
    new_date = parse_date(new.get("earnings_date"))
    if existing_date is None or new_date is None:
        return False
    if existing_date < reference_date or new_date < reference_date:
        return False

    old_rank = _event_source_priority(existing)
    new_rank = _event_source_priority(new)
    if new_rank < old_rank:
        return False
    if new_rank == old_rank and _event_sources(existing).isdisjoint(_event_sources(new)):
        return False

    existing_quarter = str(existing.get("fiscal_quarter") or "").strip()
    new_quarter = str(new.get("fiscal_quarter") or "").strip()
    if existing_quarter and new_quarter:
        return existing_quarter == new_quarter

    return abs((existing_date - new_date).days) <= ACTIVE_FUTURE_SUPERSESSION_WINDOW_DAYS


def _event_sources(event: dict[str, Any]) -> set[str]:
    raw_sources = event.get("sources", [])
    if isinstance(raw_sources, str):
        source_values = [raw_sources]
    else:
        source_values = raw_sources if isinstance(raw_sources, (list, tuple, set)) else []
    sources = {
        str(event.get("source") or "").strip().lower(),
        *[str(value).strip().lower() for value in source_values if str(value).strip()],
    }
    raw_fields = event.get("raw_provider_fields")
    if isinstance(raw_fields, dict):
        sources.update(str(key).strip().lower() for key in raw_fields if str(key).strip())
    return sources - {""}


def _event_source_priority(event: dict[str, Any]) -> int:
    return max((SOURCE_PRIORITY.get(source, 0) for source in _event_sources(event)), default=0)


def _new_session_is_better(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    new_session = normalize_release_session(new.get("release_session"))
    existing_session = normalize_release_session(existing.get("release_session"))
    if new_session not in CONFIRMED_SESSIONS:
        return False
    if existing_session not in CONFIRMED_SESSIONS:
        return True
    new_conf = CONFIDENCE_PRIORITY.get(str(new.get("source_confidence") or "").lower(), 0)
    old_conf = CONFIDENCE_PRIORITY.get(str(existing.get("source_confidence") or "").lower(), 0)
    if new_conf != old_conf:
        return new_conf > old_conf
    return SOURCE_PRIORITY.get(str(new.get("source") or "").lower(), 0) > SOURCE_PRIORITY.get(
        str(existing.get("source") or "").lower(),
        0,
    )


def _new_source_is_better(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    new_source = str(new.get("source") or "").lower()
    old_source = str(existing.get("source") or "").lower()
    return SOURCE_PRIORITY.get(new_source, 0) > SOURCE_PRIORITY.get(old_source, 0)
