"""The week's names for the Saturday ticker briefs (WISHLIST P1-3 packet 3b).

Trader decision 2026-09-24: briefs run Saturday only, for names in the week's
picks, alerts and journal, with a 7-day reuse cache keyed by (symbol, week).
Read-only: every source is opened for reading and an unreadable one adds no
names and is named in ``unreadable``.

Order (the cap keeps the front): traded this week, claimed, liked, swing Focus,
then M5 alert names by how often they alerted this week.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")

#: Setting: most names briefed per week; the rest are counted as over the cap.
MAX_NAMES_SETTING = "ai_ticker_briefs_max_names"
DEFAULT_MAX_NAMES = 40
#: Reuse window of the (symbol, week) cache, in days.
WEEK_CACHE_DAYS = 7
WEEK_CACHE_FILENAME = "ticker_briefs_week_cache.jsonl"

REASON_TRADED = "traded"
REASON_CLAIMED = "claimed"
REASON_LIKED = "liked"
REASON_SWING_FOCUS = "swing_focus"
REASON_ALERTED = "alerted"


@dataclass
class WeekNames:
    """Names in brief order, why each is here, and which sources could not be read."""

    week: str
    ordered: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)

    def add(self, symbol: Any, reason: str) -> None:
        token = _symbol(symbol)
        if token and token not in self.reasons:
            self.reasons[token] = reason
            self.ordered.append(token)


def _symbol(value: Any) -> str:
    text = str(value or "").strip().upper().lstrip("$")
    text = text.split()[0] if text else ""
    return text if _SYMBOL.fullmatch(text) else ""


def week_start(session_date: str) -> date:
    day = date.fromisoformat(str(session_date)[:10])
    return day - timedelta(days=day.weekday())


def week_key(session_date: str) -> str:
    year, number, _ = date.fromisoformat(str(session_date)[:10]).isocalendar()
    return f"{year}-W{number:02d}"


def max_names() -> int:
    """The per-week cap from local settings (default 40; 0 or less means no cap)."""
    try:
        from ai_jobs import store

        raw = store._paths().get_local_setting(MAX_NAMES_SETTING, DEFAULT_MAX_NAMES)
        return int(raw)
    except Exception:  # noqa: BLE001 - an unreadable setting keeps the default
        return DEFAULT_MAX_NAMES


def default_sources() -> dict[str, Path]:
    import project_paths

    return {
        "journal": Path(project_paths.JOURNAL_DB_FILE),
        "claimed": Path(project_paths.CLAIMED_PICKS_FILE),
        "feedback": Path(project_paths.PICK_FEEDBACK_FILE),
        "swing_focus_longs": Path(project_paths.FOCUS_SWING_LONGS_FILE),
        "swing_focus_shorts": Path(project_paths.FOCUS_SWING_SHORTS_FILE),
        "alerts": Path(project_paths.INTRADAY_BOUNCES_FILE),
    }


def _in_week(value: Any, first: str, last: str) -> bool:
    text = str(value or "")[:10]
    return bool(text) and first <= text <= last


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def _traded(path: Path, first: str, last: str) -> list[str]:
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT symbol FROM trades WHERE substr(opened_at, 1, 10) BETWEEN ? AND ? "
            "OR substr(closed_at, 1, 10) BETWEEN ? AND ? OR trade_date BETWEEN ? AND ? "
            "ORDER BY opened_at",
            (first, last, first, last, first, last),
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def load_week_names(
    session_date: str, *, sources: Mapping[str, Path] | None = None
) -> WeekNames:
    """This session's week (Monday through ``session_date``) of picks, alerts and trades."""
    paths = dict(default_sources() if sources is None else sources)
    first = week_start(session_date).isoformat()
    last = str(session_date)[:10]
    names = WeekNames(week=week_key(session_date))

    def _read(name: str, reader) -> Any:
        path = paths.get(name)
        if path is None:
            return None
        try:
            return reader(Path(path))
        except Exception as exc:  # noqa: BLE001 - an unreadable source adds no names
            logging.warning("Ticker briefs: week source %s unreadable at %s (%s)", name, path, exc)
            names.unreadable.append(name)
            return None

    for symbol in _read("journal", lambda p: _traded(p, first, last)) or ():
        names.add(symbol, REASON_TRADED)
    for row in _read("claimed", _jsonl) or ():
        if str(row.get("action") or "") == "claim" and _in_week(row.get("session_date"), first, last):
            names.add(row.get("symbol"), REASON_CLAIMED)
    for row in _read("feedback", _jsonl) or ():
        if str(row.get("verdict") or "") == "like" and _in_week(row.get("trade_date"), first, last):
            names.add(row.get("symbol"), REASON_LIKED)
    for key in ("swing_focus_longs", "swing_focus_shorts"):
        text = _read(key, lambda p: p.read_text(encoding="utf-8", errors="replace"))
        for line in (text or "").splitlines():
            names.add(line.split("#", 1)[0], REASON_SWING_FOCUS)

    def _alert_counts(path: Path) -> Counter:
        counts: Counter = Counter()
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                if _in_week(row.get("trade_date"), first, last):
                    token = _symbol(row.get("symbol"))
                    if token:
                        counts[token] += 1
        return counts

    counts = _read("alerts", _alert_counts) or Counter()
    for symbol, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        names.add(symbol, REASON_ALERTED)
    return names


def week_cache_path(root: Path) -> Path:
    return Path(root) / WEEK_CACHE_FILENAME


def read_week_cache(
    path: Path, week: str, *, now: datetime | None = None
) -> dict[str, dict[str, Any]]:
    """Briefed entries for ``week`` recorded within the last 7 days, newest per symbol."""
    from diagnostics.artifact_io import read_jsonl

    try:
        rows = read_jsonl(Path(path))
    except (OSError, ValueError):
        return {}
    moment = (now or datetime.now()).astimezone()
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("week") or "") != week:
            continue
        if str(row.get("status") or "") != "briefed":
            continue
        try:
            recorded = datetime.fromisoformat(str(row.get("recorded_at") or ""))
        except ValueError:
            continue
        if recorded.tzinfo is None or moment - recorded > timedelta(days=WEEK_CACHE_DAYS):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            latest[symbol] = dict(row)
    return latest


def append_week_cache(path: Path, entry: Mapping[str, Any], week: str) -> None:
    from diagnostics.artifact_io import append_jsonl_rows

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl_rows(target, ({**dict(entry), "week": week},), fsync=True)
