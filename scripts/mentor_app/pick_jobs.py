"""Pick assessments as queue jobs: the 06:15 PT pass over every Focus name, hourly refreshes. Qt-free.

The morning pass narrates every Focus name whose cached card was not built today; an
hourly pass narrates only names whose pick pack hash changed. A card is cached in the
chat store under (symbol, pack hash), so an unchanged pack is never narrated twice.
A job yields (returns without a model call) when a chat turn needs the only slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_app.assess import CACHE_NAME, Assessment

PT = ZoneInfo("America/Los_Angeles")
MORNING_AT = time(6, 15)
REFRESH_EVERY = timedelta(hours=1)
KIND_MORNING = "morning"
KIND_HOURLY = "hourly"
KIND_LIVE = "live"


@dataclass
class PickSchedule:
    """When the next background pass is due. Marked only when a pass is actually queued."""

    last_morning: date | None = None
    last_run_utc: datetime | None = None

    def due(self, now: datetime) -> str:
        moment = now if now.tzinfo else now.astimezone()
        local = moment.astimezone(PT)
        if local.time() < MORNING_AT:
            return ""
        if self.last_morning != local.date():
            return KIND_MORNING
        if self.last_run_utc is None or moment - self.last_run_utc >= REFRESH_EVERY:
            return KIND_HOURLY
        return ""

    def mark(self, kind: str, now: datetime) -> None:
        moment = now if now.tzinfo else now.astimezone()
        if kind == KIND_MORNING:
            self.last_morning = moment.astimezone(PT).date()
        self.last_run_utc = moment.astimezone(timezone.utc)


def focus_names(focus: Mapping[str, Mapping[str, Iterable[str]]]) -> list[tuple[str, str]]:
    """``(symbol, SIDE)`` for every Focus name, swing first, each symbol once."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for category in ("swing", "m5"):
        for side in ("long", "short"):
            for raw in (focus.get(category) or {}).get(side) or ():
                symbol = str(raw or "").strip().upper()
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    out.append((symbol, side.upper()))
    return out


def cache_key(symbol: str, pack_hash: str) -> dict[str, str]:
    return {"symbol": str(symbol).upper(), "hash": str(pack_hash)}


def cached_assessment(store: Any, symbol: str, pack_hash: str) -> Assessment | None:
    row = store.get_pack(CACHE_NAME, cache_key(symbol, pack_hash))
    if not row:
        return None
    try:
        return Assessment.from_json(str(row["pack_json"]))
    except (ValueError, TypeError, KeyError):
        return None


def _built_on(assessment: Assessment) -> date | None:
    try:
        return datetime.fromisoformat(assessment.built_utc).astimezone(PT).date()
    except (TypeError, ValueError):
        return None


def needs_narration(cached: Assessment | None, kind: str, today_pt: date) -> bool:
    if cached is None or not cached.narrated:
        return True
    if kind == KIND_MORNING:
        return _built_on(cached) != today_pt
    return False


def run_pick_job(
    symbol: str,
    side: str,
    *,
    kind: str,
    store: Any,
    build_pack: Callable[[str, str], Any],
    pack_hash: Callable[[Any], str],
    narrate: Callable[[Any, str], Assessment],
    now: Callable[[], datetime],
    should_yield: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Build the pack, reuse the cached card when it still fits, else narrate and cache.

    Returns ``{symbol, side, pack, hash, assessment, narrated, yielded}``.
    """
    pack = build_pack(symbol, side)
    digest = pack_hash(pack)
    cached = cached_assessment(store, symbol, digest)
    result: dict[str, Any] = {
        "symbol": symbol, "side": side, "pack": pack, "hash": digest,
        "assessment": cached, "narrated": False, "yielded": False,
    }
    today = now().astimezone(PT).date()
    if not needs_narration(cached, kind, today):
        return result
    if should_yield():
        result["yielded"] = True  # one slot and a chat turn waits: the card comes next pass
        return result
    assessment = narrate(pack, digest)
    if assessment.narrated:
        store.put_pack(CACHE_NAME, cache_key(symbol, digest), assessment.to_json(), assessment.built_utc)
    result["assessment"] = assessment
    result["narrated"] = True
    return result


# ---------------------------------------------------------------- the liked set
LIKED_DAYS = 30
MAX_CHIPS = 24


def _norm_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in ("LONG", "SHORT") else ""


def liked_picks(
    *,
    today: date,
    claims_path: Any = None,
    feedback_path: Any = None,
    favorites_path: Any = None,
) -> list[tuple[str, str]]:
    """The trader's own picks, newest first, each symbol once: ``[(SYMBOL, SIDE)]``.

    A standing claim (``claimed_picks``), a like in ``pick_feedback`` within LIKED_DAYS,
    or a swing favourite still standing from a session within LIKED_DAYS. Read-only.
    """
    import claimed_picks
    import pick_feedback
    import swing_favorites

    since = (today - timedelta(days=LIKED_DAYS)).isoformat()
    seen: list[tuple[str, str, Any]] = []  # (when, symbol, side)
    for row in claimed_picks.active_claims(claims_path or claimed_picks.CLAIMED_PICKS_FILE, as_of=today):
        seen.append((str(row.get("claim_at") or row.get("session_date") or ""), str(row.get("symbol") or ""), row.get("side")))
    for row in pick_feedback.load_pick_feedback(feedback_path or pick_feedback.PICK_FEEDBACK_FILE):
        if row.get("verdict") == "like" and str(row.get("trade_date") or "")[:10] >= since:
            seen.append((str(row.get("ts") or ""), str(row.get("symbol") or ""), row.get("side")))
    standing: dict[tuple[str, str], str] = {}
    for row in swing_favorites.load_rows(favorites_path or swing_favorites.SWING_FAVORITES_FILE):
        if str(row.get("session_date") or "") < since:
            continue
        key = (str(row.get("symbol") or "").strip().upper(), _norm_side(row.get("side")))
        action = str(row.get("action") or "").strip().lower()
        if action == "remove":
            standing.pop(key, None)
        elif action == "add":
            standing[key] = str(row.get("event_at") or row.get("session_date") or "")
    seen.extend((when, symbol, side) for (symbol, side), when in standing.items())
    out: list[tuple[str, str]] = []
    done: set[str] = set()
    # Newest first by the wall-clock stamp (first 19 chars: offsets vary between stores).
    for _, symbol, side in sorted(seen, key=lambda item: item[0][:19], reverse=True):
        sym = str(symbol or "").strip().upper()
        if sym and sym not in done:
            done.add(sym)
            out.append((sym, _norm_side(side)))
    return out
