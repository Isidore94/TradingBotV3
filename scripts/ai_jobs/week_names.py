"""The nightly ticker-brief roster (P1-3 packet 3b; every care name nightly 2026-09-30).

Trader's word 2026-09-30: every name the trader cares about is briefed every
night, uncapped - this week's trades, open journal positions, claims, liked picks
(``mentor_app.pick_jobs.liked_picks``), swing Focus and M5 Focus. Only the
alert-only names (the M5 alert flood) stay behind ``ai_ticker_briefs_max_names``.
A cached brief is reused only while the symbol's evidence hash is unchanged.
Read-only: every source is opened for reading and an unreadable one adds no
names and is named in ``unreadable``.

Order (the morning file's 48 KB cap keeps the front): traded this week, open
positions, claimed, liked, swing Focus, M5 Focus, then M5 alert names by how
often they alerted this week.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")

#: Setting: most ALERT-ONLY names briefed a night; care names are never capped.
MAX_NAMES_SETTING = "ai_ticker_briefs_max_names"
DEFAULT_MAX_NAMES = 40
#: Symbol -> content hash of its newest brief; a brief is reused while the hash holds.
EVIDENCE_CACHE_FILENAME = "ticker_briefs_evidence_cache.jsonl"
#: Rewrite the cache to one row per symbol once it holds more rows than this.
EVIDENCE_CACHE_COMPACT_ROWS = 5000

REASON_TRADED = "traded"
REASON_OPEN = "open_position"
REASON_CLAIMED = "claimed"
REASON_LIKED = "liked"
REASON_SWING_FOCUS = "swing_focus"
REASON_M5_FOCUS = "m5_focus"
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

    def roster(self, cap: int) -> tuple[list[str], int]:
        """Every care name, then alert-only names up to ``cap`` (<= 0: no cap); and how many were cut."""
        care = [name for name in self.ordered if self.reasons.get(name) != REASON_ALERTED]
        alerted = [name for name in self.ordered if self.reasons.get(name) == REASON_ALERTED]
        kept = alerted[:cap] if cap > 0 else alerted
        return care + kept, len(alerted) - len(kept)

    def reason_counts(self, symbols: list[str]) -> dict[str, int]:
        """How many of ``symbols`` came from each source, in first-seen order."""
        counts: dict[str, int] = {}
        for name in symbols:
            reason = self.reasons.get(name, "")
            counts[reason] = counts.get(reason, 0) + 1
        return counts


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
    """The alert-only cap from local settings (default 40; 0 or less means no cap)."""
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
        "favorites": Path(project_paths.SWING_FAVORITES_FILE),
        "swing_focus_longs": Path(project_paths.FOCUS_SWING_LONGS_FILE),
        "swing_focus_shorts": Path(project_paths.FOCUS_SWING_SHORTS_FILE),
        "m5_focus_longs": Path(project_paths.FOCUS_LONGS_FILE),
        "m5_focus_shorts": Path(project_paths.FOCUS_SHORTS_FILE),
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


def _open_positions(path: Path) -> list[str]:
    """Symbols of every open or half-exited journal trade, whenever it opened."""
    from journal_store import PARTLY_CLOSED_SPELLINGS, TRADE_STATUS_OPEN

    statuses = sorted({TRADE_STATUS_OPEN, *PARTLY_CLOSED_SPELLINGS})
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT symbol FROM trades WHERE upper(status) IN "
            f"({','.join('?' * len(statuses))}) ORDER BY opened_at",
            statuses,
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def _liked(paths: Mapping[str, Path], session_date: str) -> list[str]:
    """The Mentor app's liked set: claims after the fade, likes and swing favourites."""
    from mentor_app.pick_jobs import liked_picks

    picks = liked_picks(
        today=date.fromisoformat(str(session_date)[:10]),
        claims_path=paths["claimed"],
        feedback_path=paths["feedback"],
        favorites_path=paths["favorites"],
    )
    return [symbol for symbol, _side in picks]


def load_week_names(
    session_date: str, *, sources: Mapping[str, Path] | None = None, warn: bool = True
) -> WeekNames:
    """The night's roster: care names from every source, then this week's alert names."""
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
            if warn:
                logging.warning(
                    "Ticker briefs: roster source %s unreadable at %s (%s)", name, path, exc
                )
            if name not in names.unreadable:
                names.unreadable.append(name)
            return None

    for symbol in _read("journal", lambda p: _traded(p, first, last)) or ():
        names.add(symbol, REASON_TRADED)
    for symbol in _read("journal", _open_positions) or ():
        names.add(symbol, REASON_OPEN)
    for row in _read("claimed", _jsonl) or ():
        if str(row.get("action") or "") == "claim" and _in_week(row.get("session_date"), first, last):
            names.add(row.get("symbol"), REASON_CLAIMED)
    for row in _read("feedback", _jsonl) or ():
        if str(row.get("verdict") or "") == "like" and _in_week(row.get("trade_date"), first, last):
            names.add(row.get("symbol"), REASON_LIKED)
    if all(key in paths for key in ("claimed", "feedback", "favorites")):
        for symbol in _read("favorites", lambda _p: _liked(paths, last)) or ():
            names.add(symbol, REASON_LIKED)
    for key, reason in (
        ("swing_focus_longs", REASON_SWING_FOCUS),
        ("swing_focus_shorts", REASON_SWING_FOCUS),
        ("m5_focus_longs", REASON_M5_FOCUS),
        ("m5_focus_shorts", REASON_M5_FOCUS),
    ):
        text = _read(key, lambda p: p.read_text(encoding="utf-8", errors="replace"))
        for line in (text or "").splitlines():
            names.add(line.split("#", 1)[0], reason)

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


def content_hash(evidence: Mapping[str, Any]) -> str:
    """Hash of what the model is fed about the symbol: each projected source's id and content.

    The membership source and every read stamp are left out, so the hash moves only
    when the symbol's evidence moves.
    """
    from ai_jobs.briefs import MEMBERSHIP_SOURCE_ID

    payload = [
        {"source_id": str(source.get("source_id") or ""), "content": source.get("content")}
        for source in evidence.get("sources") or []
        if isinstance(source, Mapping)
        and str(source.get("source_id") or "") != MEMBERSHIP_SOURCE_ID
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_cache_path(root: Path) -> Path:
    return Path(root) / EVIDENCE_CACHE_FILENAME


def read_evidence_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Newest briefed row per symbol that carries a content hash."""
    from diagnostics.artifact_io import read_jsonl

    try:
        rows = read_jsonl(Path(path))
    except (OSError, ValueError):
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("status") or "") != "briefed":
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol and str(row.get("content_hash") or ""):
            latest[symbol] = dict(row)
    return latest


def append_evidence_cache(path: Path, entry: Mapping[str, Any], digest: str) -> None:
    from diagnostics.artifact_io import append_jsonl_rows

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl_rows(target, ({**dict(entry), "content_hash": str(digest)},), fsync=True)


def compact_evidence_cache(path: Path, *, max_rows: int = EVIDENCE_CACHE_COMPACT_ROWS) -> bool:
    """Rewrite the cache to its newest row per symbol once it holds more than ``max_rows``."""
    from diagnostics.artifact_io import read_jsonl

    target = Path(path)
    try:
        rows = read_jsonl(target)
    except (OSError, ValueError):
        return False
    if len(rows) <= max_rows:
        return False
    latest = read_evidence_cache(target)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in latest.values()),
        encoding="utf-8",
    )
    temp.replace(target)
    return True


def roster_size(session_date: str | None = None) -> int:
    """How many names tonight's run would brief (care names + capped alert names); 0 if unreadable."""
    try:
        day = str(session_date or date.today().isoformat())[:10]
        symbols, _over = load_week_names(day, warn=False).roster(max_names())
        return len(symbols)
    except Exception:  # noqa: BLE001 - a sizing read never decides the slate
        return 0
