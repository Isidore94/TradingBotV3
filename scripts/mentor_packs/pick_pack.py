"""Pick pack: everything the desk knows about one liked name, every row with a citable id. Read-only.

Sections: Focus membership + claim + the trader's verdicts (30 d); the setup's
scoreboard cell (n, win rate, Wilson lower bound, avg return, avg closed R; the
n floor printed out loud); the name's own next earnings; industry peers reporting
within +-7 calendar days; the trading plan's lines; and the human-pick cohort's
1/3/5/10-session returns; and up to 5 stored news headlines from the last 3 days
(``news:<SYM>:<n>``, each with its URL). Files are read directly (the Focus and Journal store
classes write on construction, so they are never built here). A source that cannot be read
gives an "unknown" row, never a guess. Call :func:`build` on a worker. The pack hash
covers every row's id and text plus the plan file's hash, so any plan edit or a new
headline re-narrates the card at its next refresh.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "pick_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Everything the desk knows about one stock the trader likes: Focus membership and claim, "
            "his verdicts, the setup's scoreboard cell (n, win rate, Wilson lower bound), its own and its "
            "industry peers' earnings dates, his plan lines, and his pick cohort's returns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker, e.g. NVDA."},
                "side": {"type": "string", "description": "LONG or SHORT; blank = the side on Focus."},
            },
            "required": ["symbol"],
        },
    },
}

ET = ZoneInfo("America/New_York")
#: `evidence_stats.MIN_REPORTABLE_N`: below it a cell says "too few" and carries only n.
MIN_REPORTABLE_N = 30
#: `evidence_stats.SWING_HORIZON_SESSIONS`: the D1 cell's forward horizon.
CELL_HORIZON_SESSIONS = 5
FEEDBACK_DAYS = 30
FEEDBACK_MAX_ROWS = 10
PEER_WINDOW_DAYS = 7
PEER_MAX_ROWS = 12
COHORT_HORIZONS = (1, 3, 5, 10)
NEWS_DAYS = 3
NEWS_MAX_ROWS = 5
_VOLATILE_KINDS = frozenset({"asof"})


@dataclass(frozen=True)
class PickPaths:
    """Every file the pack reads; tests pass fixture paths, the app uses :func:`live_paths`."""

    focus_longs: Path
    focus_shorts: Path
    focus_swing_longs: Path
    focus_swing_shorts: Path
    pick_clocks: Path
    claimed_picks: Path
    pick_feedback: Path
    tier_outcomes: Path
    leaderboard: Path
    earnings_history: Path
    cohort_performance: Path
    plan: Path | None = None
    #: symbol -> industry context (``industry_context.load_industry_context_map`` shape).
    industry_map: Callable[[], Mapping[str, Mapping[str, Any]]] | None = field(default=None, compare=False)
    #: (symbol, since UTC ISO, limit) -> stored headlines (``news_pack.Reader``); None = no news source.
    news: Callable[[str, str, int], Any] | None = field(default=None, compare=False)
    #: symbol -> last good fetch UTC ISO / last feed failure (``news_pack`` readers); None = not known.
    news_stamps: Callable[[str], Any] | None = field(default=None, compare=False)
    news_errors: Callable[[str], Any] | None = field(default=None, compare=False)
    #: P13 M5 branch: the desk's M5 alert log (``INTRADAY_BOUNCES_FILE``) and its day-trade grades
    #: (``working_lately/setup_grades_latest.json``); None = not stored.
    m5_alerts: Path | None = None
    m5_grades: Path | None = None
    #: P15a: the ai_store ``briefs/`` root (``<year>/<session>/ticker_briefs_manifest.jsonl``); None = not read.
    briefs: Path | None = None


def live_paths() -> PickPaths:
    import project_paths as pp
    from mentor_packs import news_pack

    longs, shorts = Path(pp.FOCUS_LONGS_FILE), Path(pp.FOCUS_SHORTS_FILE)

    def industry_map() -> Mapping[str, Mapping[str, Any]]:
        from industry_context import load_industry_context_map

        return load_industry_context_map()

    return PickPaths(
        focus_longs=longs,
        focus_shorts=shorts,
        focus_swing_longs=longs.with_name("focus_swing_longs.txt"),
        focus_swing_shorts=shorts.with_name("focus_swing_shorts.txt"),
        pick_clocks=longs.with_name("focus_pick_clocks.json"),
        claimed_picks=Path(pp.CLAIMED_PICKS_FILE),
        pick_feedback=Path(pp.PICK_FEEDBACK_FILE),
        tier_outcomes=Path(pp.MASTER_AVWAP_TIER_OUTCOMES_FILE),
        leaderboard=Path(pp.MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE),
        earnings_history=Path(pp.EARNINGS_CALENDAR_HISTORY_FILE),
        cohort_performance=Path(pp.HUMAN_FOCUS_PERFORMANCE_FILE),
        plan=None,
        industry_map=industry_map,
        news=news_pack.live_reader(),
        news_stamps=news_pack.live_stamp_reader(),
        news_errors=news_pack.live_error_reader(),
        m5_alerts=Path(pp.INTRADAY_BOUNCES_FILE),
        m5_grades=Path(pp.LOCAL_SETTINGS_DIR) / "working_lately" / "setup_grades_latest.json",
        briefs=_live_briefs_root(),
    )


def _live_briefs_root() -> Path | None:
    try:
        from ai_jobs import store as ai_store

        return ai_store.briefs_dir(create=False)
    except Exception:  # noqa: BLE001 - no ai_store configured: the brief row says "not read"
        return None


# ---------------------------------------------------------------- small readers
def _sym(value: Any) -> str:
    return str(value or "").strip().upper()


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in ("LONG", "SHORT") else ""


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN is unknown


def _unknown(row_id: str, what: str, exc: BaseException) -> dict[str, Any]:
    return {"id": row_id, "kind": "unknown", "text": f"{what}: unknown ({type(exc).__name__})"}


_csv_cache: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}
_csv_lock = threading.Lock()


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _cached(kind: str, path: Path, load: Callable[[Path], Any]) -> Any:
    """``load(path)`` once per file version (mtime + size); the big CSVs are read once."""
    signature = _signature(path)
    key = (kind, str(path))
    with _csv_lock:
        hit = _csv_cache.get(key)
        if hit and hit[0] == signature:
            return hit[1]
    value = load(path)
    with _csv_lock:
        _csv_cache[key] = (signature, value)
    return value


def _tier_index(path: Path) -> dict[str, Any]:
    """Per (family, side) 5-session counts and per (symbol, side) the latest scan's family."""
    cells: dict[tuple[str, str], dict[str, float]] = {}
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            family = str(row.get("setup_family") or "").strip()
            side = _side(row.get("side"))
            symbol = _sym(row.get("symbol"))
            scan_date = str(row.get("scan_date") or "")[:10]
            if symbol and side and family and scan_date >= latest.get((symbol, side), ("", ""))[0]:
                latest[(symbol, side)] = (scan_date, family)
            if not family or not side or str(row.get("horizon_sessions") or "").strip() != str(CELL_HORIZON_SESSIONS):
                continue
            if str(row.get("stale_horizon") or "").strip().lower() == "true":
                continue  # the horizon spanned a data gap; not a clean 5-session outcome
            ret = _float(row.get("side_return_pct"))
            win = str(row.get("win") or "").strip().lower()
            if ret is None or win not in ("true", "false"):
                continue
            cell = cells.setdefault((family, side), {"n": 0, "wins": 0, "ret_sum": 0.0})
            cell["n"] += 1
            cell["wins"] += 1 if win == "true" else 0
            cell["ret_sum"] += ret
    return {"cells": cells, "latest": latest}


def _leaderboard_r(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Per (family, side) the closed-count-weighted avg closed R over priority buckets."""
    pooled: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("attribute_key") or "") != "setup.setup_family":
                continue
            side = _side(row.get("side"))
            family = str(row.get("value_label") or "").strip()
            closed = _float(row.get("closed_tradeable_setup_count"))
            avg_r = _float(row.get("avg_closed_r"))
            if not side or not family or not closed or avg_r is None:
                continue
            cell = pooled.setdefault((family, side), {"n": 0.0, "r_sum": 0.0})
            cell["n"] += closed
            cell["r_sum"] += avg_r * closed
    return pooled


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- sections
def _membership(symbol: str, today: date, paths: PickPaths) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The Focus row and the (category, side) memberships, swing first."""
    from focus_picks import load_focus_maps_by_category

    maps = load_focus_maps_by_category(
        focus_longs_path=paths.focus_longs,
        focus_shorts_path=paths.focus_shorts,
        focus_swing_longs_path=paths.focus_swing_longs,
        focus_swing_shorts_path=paths.focus_swing_shorts,
        today=today,
    )
    try:
        clocks = (_read_json(paths.pick_clocks) or {}).get("picks") or {}
    except (OSError, ValueError):
        clocks = {}
    found: list[tuple[str, str]] = []
    parts: list[str] = []
    for category in ("swing", "m5"):
        for side in ("long", "short"):
            if symbol not in maps.get(category, {}).get(side, ()):
                continue
            found.append((category, side.upper()))
            key = f"{symbol}|{side}" if category == "m5" else f"{symbol}|{side}|{category}"
            clock = clocks.get(key) if isinstance(clocks, Mapping) else None
            since = ""
            if isinstance(clock, Mapping) and clock.get("clock_from"):
                since = f" (pick clock from {clock['clock_from']}, {clock.get('reason') or 'added'})"
            parts.append(f"{category} {side}{since}")
    text = f"Focus: {'; '.join(parts)}" if parts else "Focus: not on a Focus list"
    return [{"id": f"pick:{symbol}:membership", "kind": "membership", "text": text}], found


def _claims(symbol: str, today: date, paths: PickPaths) -> list[dict[str, Any]]:
    import claimed_picks

    mine = [row for row in claimed_picks.active_claims(paths.claimed_picks, as_of=today) if _sym(row.get("symbol")) == symbol]
    if not mine:
        return [{"id": f"pick:{symbol}:claim", "kind": "claim", "text": "Claim: no standing claim"}]
    rows = []
    for claim in mine:
        side = _side(claim.get("side")) or "?"
        setup = str(claim.get("claimed_setup_id") or "").strip() or "no setup named"
        rows.append(
            {
                "id": f"pick:{symbol}:claim:{side}:{setup}",
                "kind": "claim",
                "side": side,
                "setup": setup,
                "claim_at": str(claim.get("claim_at") or ""),
                "text": f"Claim: {side} {setup} since session {claim.get('session_date') or '?'} ({claim.get('horizon') or '?'})",
            }
        )
    return rows


def _feedback(symbol: str, today: date, paths: PickPaths) -> tuple[list[dict[str, Any]], str]:
    """Verdict rows (newest first) and the latest like origin (for the cohort)."""
    from pick_feedback import load_pick_feedback

    since = (today - timedelta(days=FEEDBACK_DAYS)).isoformat()
    mine = [
        row for row in load_pick_feedback(paths.pick_feedback)
        if _sym(row.get("symbol")) == symbol and str(row.get("trade_date") or "")[:10] >= since
    ]
    mine.sort(key=lambda row: str(row.get("ts") or ""), reverse=True)
    origin = next((str(row.get("origin") or "") for row in mine if row.get("verdict") == "like" and row.get("origin")), "")
    if not mine:
        return [{"id": f"pick:{symbol}:fb", "kind": "feedback", "text": f"Verdicts: none in the last {FEEDBACK_DAYS} days"}], origin
    rows = []
    for row in mine[:FEEDBACK_MAX_ROWS]:
        reason = str(row.get("reason") or "").strip()
        rows.append(
            {
                "id": f"pick:{symbol}:fb:{str(row.get('ts') or '')[:19]}",
                "kind": "feedback",
                "text": (
                    f"Verdict {row.get('trade_date')}: {row.get('verdict')} {_side(row.get('side')) or ''} "
                    f"({row.get('category') or '-'}, from {row.get('origin') or '-'})"
                    + (f": {reason[:160]}" if reason else "")
                ).replace("  ", " "),
            }
        )
    return rows, origin


def _pct(value: float | None, digits: int = 1) -> str:
    return "unknown" if value is None else f"{value:+.{digits}f}%"


def is_m5_branch(found: list[tuple[str, str]], side: str, claims: list[dict[str, Any]]) -> bool:
    """An M5 Focus pick on ``side`` that is neither a swing Focus pick nor a D1 claim on that side."""
    return (bool(side) and ("m5", side) in found and ("swing", side) not in found
            and not any(row.get("side") == side for row in claims))


def _m5_branch_row(symbol: str, side: str) -> dict[str, Any]:
    return {"id": f"pick:{symbol}:branch", "kind": "branch", "branch": "m5",
            "text": (f"Setup cell branch: M5 (Focus m5 {side.lower()}, no D1 claim): the M5 bounce cell is this "
                     "pick's setup cell; any D1 cell row is context only")}


#: How far back the M5 alert log is searched for the name's latest alert.
M5_WINDOW_DAYS = 60
#: At most this many bounce types of one alert get a cell row.
M5_MAX_TYPES = 3


def _m5_latest_index(path: Path) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Per (symbol, side) the latest alert in the desk's M5 alert log: (trade date, time, bounce types)."""
    latest: dict[tuple[str, str], tuple[str, str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (_sym(row.get("symbol")), _side(row.get("direction")))
            stamp = (str(row.get("trade_date") or "")[:10], str(row.get("time_local") or ""))
            if key[0] and key[1] and stamp >= latest.get(key, ("", "", ""))[:2]:
                latest[key] = (*stamp, str(row.get("bounce_types") or ""))
    return latest


def _m5_grades(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    cells = {str(row.get("key") or ""): row for row in (payload or {}).get("daytrade") or () if isinstance(row, Mapping)}
    return {"as_of": str((payload or {}).get("as_of") or "?"), "cells": cells}


def _m5_cell_rows(symbol: str, side: str, today: date, paths: PickPaths) -> list[dict[str, Any]]:
    """The M5 setup cell: the desk's day-trade grade (``setup_grades_latest.json``) for each bounce type of the
    name's latest M5 alert (``intraday_bounces.csv``). Both are small files the desk already writes."""
    from held_run_score import bounce_components

    row_id = f"pick:{symbol}:m5cell"
    alerts, grades = paths.m5_alerts, paths.m5_grades
    if alerts is None or grades is None or not Path(alerts).is_file() or not Path(grades).is_file():
        return [{"id": row_id, "kind": "cell",
                 "text": "M5 setup cell: not stored (no M5 alert log or day-trade grades file on this desk)"}]
    latest = _cached("m5_alerts", Path(alerts), _m5_latest_index).get((symbol, side))
    since = (today - timedelta(days=M5_WINDOW_DAYS)).isoformat()
    if not latest or latest[0] < since:
        return [{"id": row_id, "kind": "cell", "n": 0,
                 "text": f"M5 setup cell: no M5 alert for {symbol} {side} in the last {M5_WINDOW_DAYS} days"}]
    day, clock, raw_types = latest
    graded = _cached("m5_grades", Path(grades), _m5_grades)
    types: list[str] = []
    for raw in raw_types.split(","):
        for part in bounce_components(raw.strip()):
            if part and part not in types:
                types.append(part)
    if not types:
        return [{"id": row_id, "kind": "cell", "text": f"M5 setup cell: the latest {symbol} alert ({day}) names no type"}]
    rows = []
    for family in types[:M5_MAX_TYPES]:
        head = (f"M5 setup cell {family} {side} (from the latest M5 alert, {day} {clock[:5]} PT; desk day-trade grade "
                f"as of {graded['as_of']}, 1:1 bracket)")
        cell = graded["cells"].get(f"{family}|{side}")
        n = int(_float((cell or {}).get("n")) or 0)
        if not cell or not n:
            text = f"{head}: no graded rows"
        elif n < MIN_REPORTABLE_N:
            text = f"{head}: too few, n={n} (floor {MIN_REPORTABLE_N})"
        else:
            win, bound, avg = (_float(cell.get(key)) for key in ("win_rate", "low_bound", "avg_r"))
            text = (f"{head}: grade {cell.get('grade') or '?'}, n={n} (floor {MIN_REPORTABLE_N}), win rate "
                    f"{'unknown' if win is None else f'{win:.0%}'}, low bound "
                    f"{'unknown' if bound is None else f'{bound:.2f}'}, avg R {'unknown' if avg is None else f'{avg:+.2f}'}")
        rows.append({"id": f"{row_id}:{family}", "kind": "cell", "setup": family, "n": n, "text": text})
    return rows


def _cell_rows(symbol: str, side: str, claims: list[dict[str, Any]], paths: PickPaths,
               *, m5: bool = False) -> list[dict[str, Any]]:
    import setup_grades

    if not side:
        return [{"id": f"pick:{symbol}:cell", "kind": "cell", "text": "Setup cell: side unknown, no cell read"}]
    index = _cached("tier", paths.tier_outcomes, _tier_index)
    setup, basis = "", ""
    claimed = [row for row in claims if row.get("side") == side and row.get("setup") and row["setup"] != "no setup named"]
    if claimed:
        claimed.sort(key=lambda row: row.get("claim_at") or "")
        setup, basis = claimed[-1]["setup"], "the trader's claim"
    else:
        latest = index["latest"].get((symbol, side))
        if latest:
            setup, basis = latest[1], f"the latest D1 scan row, {latest[0]}"
            if m5:
                basis += "; D1 context only, this is an M5 pick"
    if not setup:
        return [{"id": f"pick:{symbol}:cell", "kind": "cell", "text": f"Setup cell: no setup known for {symbol} {side}"}]
    cell = index["cells"].get((setup, side))
    head = f"Setup cell {setup} {side} (setup from {basis}), {CELL_HORIZON_SESSIONS}-session D1 outcomes"
    if not cell or not cell["n"]:
        rows = [{"id": f"pick:{symbol}:cell", "kind": "cell", "setup": setup, "n": 0, "text": f"{head}: no scoreboard rows"}]
    else:
        n, wins = int(cell["n"]), int(cell["wins"])
        if n < MIN_REPORTABLE_N:
            text = f"{head}: too few, n={n} (floor {MIN_REPORTABLE_N})"
        else:
            bound = setup_grades.wilson_lower_bound(wins, n)
            text = (
                f"{head}: n={n} (floor {MIN_REPORTABLE_N}), win rate {wins / n:.0%}, Wilson LB "
                f"{bound:.2f}, avg side return {_pct(cell['ret_sum'] / n, 2)}"
            )
        rows = [{"id": f"pick:{symbol}:cell", "kind": "cell", "setup": setup, "n": n, "wins": wins, "text": text}]
    try:
        pooled = _cached("leaderboard", paths.leaderboard, _leaderboard_r).get((setup, side))
    except OSError as exc:
        rows.append(_unknown(f"pick:{symbol}:cell_r", "Leaderboard avg closed R", exc))
        return rows
    if pooled and pooled["n"]:
        n_r = int(pooled["n"])
        text = (
            f"Leaderboard {setup} {side}: too few closed setups, n={n_r} (floor {MIN_REPORTABLE_N})"
            if n_r < MIN_REPORTABLE_N
            else f"Leaderboard {setup} {side}: avg closed R {pooled['r_sum'] / n_r:+.2f} over n={n_r} closed setups (floor {MIN_REPORTABLE_N})"
        )
    else:
        text = f"Leaderboard {setup} {side}: no closed setups listed"
    rows.append({"id": f"pick:{symbol}:cell_r", "kind": "cell", "text": text})
    return rows


def _earnings_dates(paths: PickPaths, since: date) -> dict[str, list[date]]:
    from earnings_warning import future_dates_from_history

    return future_dates_from_history(_read_json(paths.earnings_history), since=since)


def _day_word(days: int) -> str:
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days == -1:
        return "yesterday"
    return f"in {days} days" if days > 0 else f"{-days} days ago"


def _own_earnings(symbol: str, today: date, dates: dict[str, list[date]]) -> dict[str, Any]:
    upcoming = [value for value in dates.get(symbol, ()) if value >= today]
    if not upcoming:
        return {
            "id": f"pick:{symbol}:earn",
            "kind": "earnings",
            "text": "Own earnings: unknown (no upcoming date in the earnings calendar)",
        }
    nxt = upcoming[0]
    days = (nxt - today).days
    return {
        "id": f"pick:{symbol}:earn",
        "kind": "earnings",
        "date": nxt.isoformat(),
        "days": days,
        "text": f"Own earnings: {nxt:%a %Y-%m-%d}, {_day_word(days)} (long and short alike)",
    }


def _peer_rows(symbol: str, today: date, dates: dict[str, list[date]], paths: PickPaths) -> list[dict[str, Any]]:
    loader = paths.industry_map
    if loader is None:
        from industry_context import load_industry_context_map as loader
    context = (loader() or {}).get(symbol) or {}
    industry = str(context.get("industry") or "").strip()
    members = [_sym(peer) for peer in context.get("industry_member_symbols") or () if _sym(peer) and _sym(peer) != symbol]
    if not industry or not members:
        return [{"id": f"pick:{symbol}:peers", "kind": "peers", "text": "Peer earnings: industry unknown"}]
    lo, hi = today - timedelta(days=PEER_WINDOW_DAYS), today + timedelta(days=PEER_WINDOW_DAYS)
    near: list[tuple[int, int, str, date]] = []
    for peer in sorted(set(members)):
        inside = [value for value in dates.get(peer, ()) if lo <= value <= hi]
        if inside:
            best = min(inside, key=lambda value: (abs((value - today).days), value))
            days = (best - today).days
            near.append((abs(days), days, peer, best))
    near.sort()
    rows = [
        {
            "id": f"pick:{symbol}:industry",
            "kind": "peers",
            "text": (
                f"Industry: {industry} ({len(set(members))} peers); {len(near)} report within "
                f"+-{PEER_WINDOW_DAYS} calendar days" + (f", nearest {min(len(near), PEER_MAX_ROWS)} listed" if near else "")
            ),
        }
    ]
    for _, days, peer, when in near[:PEER_MAX_ROWS]:
        rows.append(
            {
                "id": f"pick:{symbol}:peer:{peer}",
                "kind": "peer_earnings",
                "peer": peer,
                "date": when.isoformat(),
                "days": days,
                "text": f"Peer {peer} earnings {when:%a %Y-%m-%d}, {_day_word(days)}",
            }
        )
    return rows


def _plan_rows(symbol: str, paths: PickPaths) -> list[dict[str, Any]]:
    from mentor_packs import plan_lines

    plan = plan_lines.build(path=paths.plan)
    if not plan.rows:
        return [{"id": f"pick:{symbol}:plan", "kind": "plan_empty", "text": f"Plan: {plan.empty_text or plan_lines.EMPTY_TEXT}"}]
    rows = []
    for line in plan.rows:
        plan_id = str(line["id"])
        rows.append(
            {
                "id": f"pick:{symbol}:plan:{plan_id.removeprefix('plan:')}",
                "kind": "plan_line",
                "plan_id": plan_id,
                "text": f"Plan [{plan_id}]: {line.get('text', '')}",
            }
        )
    return rows


def _cohort_rows(symbol: str, found: list[tuple[str, str]], side: str, origin: str, paths: PickPaths) -> list[dict[str, Any]]:
    category = next((cat for cat, s in found if s == side), found[0][0] if found else "")
    if not category or not side:
        return [{"id": f"pick:{symbol}:cohort", "kind": "cohort", "text": "Pick cohort: not on Focus, no cohort read"}]
    with paths.cohort_performance.open(newline="", encoding="utf-8-sig") as handle:
        table = {
            (str(row.get("cohort") or ""), _side(row.get("side")), str(row.get("horizon_sessions") or "").strip()): row
            for row in csv.DictReader(handle)
        }
    cohorts = [(f"human_focus_{category}", "")]
    if origin:
        cohorts.append((f"human_focus_{category}_{origin}", f"{origin}:"))
    rows = []
    for cohort, tag in cohorts:
        for horizon in COHORT_HORIZONS:
            row = table.get((cohort, side, str(horizon)))
            if row is None:
                if tag:
                    continue  # no origin sub-cohort rows yet; the base cohort speaks
                text = f"Cohort {cohort} {side} {horizon}d: no rows"
                n = 0
            else:
                n = int(_float(row.get("sample_count")) or 0)
                if n < MIN_REPORTABLE_N:
                    text = f"Cohort {cohort} {side} {horizon}d: too few, n={n} (floor {MIN_REPORTABLE_N})"
                else:
                    win = _float(row.get("win_rate"))
                    avg = _float(row.get("avg_side_return"))
                    text = (
                        f"Cohort {cohort} {side} {horizon}d: n={n} (floor {MIN_REPORTABLE_N}), win rate "
                        f"{'unknown' if win is None else f'{win:.0%}'}, avg side return "
                        f"{_pct(None if avg is None else avg * 100, 2)}"
                    )
            rows.append({"id": f"pick:{symbol}:cohort:{tag}{horizon}", "kind": "cohort", "n": n, "text": text})
    return rows


def _news_rows(symbol: str, moment: datetime, paths: PickPaths) -> list[dict[str, Any]]:
    """Up to 5 stored headlines from the last 3 days as ``pick:<SYM>:news:<n>``, each carrying its URL."""
    from mentor_packs import news_pack

    if paths.news is None:
        return [{"id": f"pick:{symbol}:news", "kind": "news_empty", "text": "News: not read (no news source)"}]
    found = news_pack.headline_rows(symbol, now=moment, days=NEWS_DAYS, limit=NEWS_MAX_ROWS, reader=paths.news)
    rows = [{**row, "id": f"pick:{symbol}:news:{row['id'].rsplit(':', 1)[-1]}", "news_id": row["id"]} for row in found]
    fetched, error = news_pack.read_status(symbol, paths.news_stamps, paths.news_errors)
    state = news_pack.news_state(fetched, error)
    if state == news_pack.STATE_UNKNOWN:
        # The last request failed on every feed: whatever is stored may be stale, and "none" is unknown.
        rows.append({"id": f"pick:{symbol}:news", "kind": "unknown", "text": news_pack.empty_text(state, error, NEWS_DAYS)})
    elif not rows and state == news_pack.STATE_NOT_FETCHED:
        rows = [{"id": f"pick:{symbol}:news", "kind": "news_not_fetched",
                 "text": news_pack.empty_text(state, error, NEWS_DAYS)}]
    elif not rows:
        rows = [{"id": f"pick:{symbol}:news", "kind": "news_empty",
                 "text": f"News: no stored headlines in the last {NEWS_DAYS} days"}]
    return rows


def _brief_rows(symbol: str, today: date, paths: PickPaths) -> list[dict[str, Any]]:
    """P15a: <= 3 lines of the symbol's newest night ticker brief (via the manifests), with its date."""
    from mentor_packs import night_pack

    if paths.briefs is None:
        return [{"id": f"pick:{symbol}:brief:none", "kind": "brief_none", "text": "Night brief: not read (no briefs store)"}]
    brief = night_pack.latest_briefs(paths.briefs, today).get(symbol)
    if not brief:
        return [{"id": f"pick:{symbol}:brief:none", "kind": "brief_none",
                 "text": f"Night brief: none for {symbol} in the last {night_pack.BRIEF_SESSIONS} brief nights"}]
    refs = list(brief["evidence_refs"])
    text = f"Night brief ({brief['session']}): " + " | ".join(brief["lines"])
    return [{"id": f"pick:{symbol}:brief", "kind": "brief", "session": brief["session"], "evidence_refs": refs,
             "text": text + (f" (src: {', '.join(refs[:3])})" if refs else "")}]


# ---------------------------------------------------------------- build
def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment


def build(symbol: str = "", side: str = "", *, now: datetime | None = None, paths: PickPaths | None = None) -> Pack:
    """Build the pick pack for ``symbol``. File reads: call it on a worker."""
    sym = _sym(symbol)
    if not sym or not sym.replace(".", "").replace("-", "").isalnum():
        return make_pack(NAME, (), empty_text="pick_pack needs a ticker, e.g. NVDA")
    moment = _now(now)
    today = moment.astimezone(ET).date()
    src = paths or live_paths()
    from mentor_packs.plan_lines import plan_digest

    try:
        plan_sha = plan_digest(src.plan)
    except Exception:  # noqa: BLE001 - an unreadable plan hashes as "unknown"
        plan_sha = "unknown"
    rows: list[dict[str, Any]] = [
        {
            "id": f"pick:{sym}:asof",
            "kind": "asof",
            "text": f"{sym} as of market date {today.isoformat()}",
            "at_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "plan_sha": plan_sha,
        }
    ]
    found: list[tuple[str, str]] = []
    try:
        membership, found = _membership(sym, today, src)
        rows.extend(membership)
    except Exception as exc:  # noqa: BLE001 - one broken source never blanks the others
        rows.append(_unknown(f"pick:{sym}:membership", "Focus membership", exc))
    claims: list[dict[str, Any]] = []
    try:
        claims = _claims(sym, today, src)
        rows.extend(claims)
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:claim", "Claim", exc))
    origin = ""
    try:
        feedback, origin = _feedback(sym, today, src)
        rows.extend(feedback)
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:fb", "Verdicts", exc))
    chosen = _side(side) or (found[0][1] if found else "") or next((c["side"] for c in claims if _side(c.get("side"))), "")
    rows.append(
        {"id": f"pick:{sym}:side", "kind": "side", "side": chosen, "text": f"Side assessed: {chosen or 'unknown'}"}
    )
    m5 = is_m5_branch(found, chosen, claims)
    if m5:
        # The D1 branch already says so in its cell row ("5-session D1 outcomes"); only M5 needs its own row.
        rows.append(_m5_branch_row(sym, chosen))
        try:
            rows.extend(_m5_cell_rows(sym, chosen, today, src))
        except Exception as exc:  # noqa: BLE001
            rows.append(_unknown(f"pick:{sym}:m5cell", "M5 setup cell", exc))
    try:
        rows.extend(_cell_rows(sym, chosen, claims, src, m5=m5))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:cell", "Setup cell", exc))
    dates: dict[str, list[date]] | None = None
    try:
        dates = _earnings_dates(src, today - timedelta(days=PEER_WINDOW_DAYS))
        rows.append(_own_earnings(sym, today, dates))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:earn", "Own earnings", exc))
    try:
        if dates is None:
            raise FileNotFoundError("earnings calendar unreadable")
        rows.extend(_peer_rows(sym, today, dates, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:peers", "Peer earnings", exc))
    try:
        rows.extend(_plan_rows(sym, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:plan", "Plan", exc))
    try:
        rows.extend(_cohort_rows(sym, found, chosen, origin, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:cohort", "Pick cohort", exc))
    try:
        rows.extend(_news_rows(sym, moment, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:news", "News", exc))
    try:
        rows.extend(_brief_rows(sym, today, src))
    except Exception as exc:  # noqa: BLE001
        rows.append(_unknown(f"pick:{sym}:brief", "Night brief", exc))
    return make_pack(NAME, rows)


def plan_sha_of(pack: Pack) -> str:
    """The plan file hash a pack was built with ("" for a pack built before it was carried)."""
    return next((str(row.get("plan_sha") or "") for row in pack.rows if "plan_sha" in row), "")


def pack_hash(pack: Pack) -> str:
    """A stable hash of what the pack SAYS (the as-of stamp aside) and of the plan file: same hash, same card."""
    stable = [(row.get("id"), row.get("text")) for row in pack.rows if row.get("kind") not in _VOLATILE_KINDS]
    body = json.dumps({"name": pack.name, "rows": stable, "empty": pack.empty_text, "plan": plan_sha_of(pack)},
                      sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def plan_ids(pack: Pack) -> set[str]:
    """The plan ids (``plan:...``) the pack carried, and their pick-row ids."""
    found: set[str] = set()
    for row in pack.rows:
        if row.get("kind") == "plan_line":
            found.update({str(row["plan_id"]), str(row["id"])})
    return found


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)  # Tue 2026-09-29, 07:00 PT

FIXTURE_TIER_HEADER = "symbol,side,setup_family,scan_date,horizon_sessions,win,side_return_pct,stale_horizon"
FIXTURE_LEADERBOARD_HEADER = "side,priority_bucket,attribute_key,value_label,closed_tradeable_setup_count,avg_closed_r"
FIXTURE_COHORT_HEADER = "cohort,side,horizon_sessions,sample_count,win_rate,avg_side_return"


def write_fixture_m5(root: Path | str) -> tuple[Path, Path]:
    """The M5 alert log and day-trade grades: AMD's latest SHORT alert (2026-09-25 07:05) fired
    ``vwap_lower_band, 10_candle_high``; ``vwap_lower_band|SHORT`` is graded B at n=40,
    ``10_candle|SHORT`` is thin (n=12). An older AMD alert (2026-08-01) fired ``ema_8``."""
    base = Path(root)
    alerts = base / "intraday_bounces.csv"
    alerts.write_text(
        "time_local,trade_date,symbol,direction,bounce_types,tier,composite_r\n"
        "06:40:00,2026-08-01,AMD,short,ema_8,B,0.1\n"
        "07:05:00,2026-09-25,AMD,short,\"vwap_lower_band, 10_candle_high\",A,0.3\n"
        "06:50:00,2026-09-25,NVDA,long,vwap,B,0.1\n",
        encoding="utf-8",
    )
    grades = base / "setup_grades_latest.json"
    grades.write_text(json.dumps({"schema": "setup_grades_v2", "as_of": "2026-09-28", "daytrade": [
        {"key": "vwap_lower_band|SHORT", "grade": "B", "n": 40, "wins": 22, "win_rate": 0.55, "low_bound": 0.40,
         "avg_r": 0.08, "bounce_type": "vwap_lower_band", "side": "SHORT"},
        {"key": "10_candle|SHORT", "grade": "New", "n": 12, "wins": 5, "win_rate": 0.42, "low_bound": 0.19,
         "avg_r": -0.1, "bounce_type": "10_candle", "side": "SHORT"},
    ]}), encoding="utf-8")
    return alerts, grades


def write_fixture_world(root: Path | str, *, plan_text: str | None = None) -> PickPaths:
    """A small, deterministic desk under ``root``: NVDA long, TSLA short, AMD thin, ZZZ no earnings.

    NVDA's peer AVGO reports tomorrow (2026-09-30). NVDA has two headlines in the last 3 days.
    """
    from mentor_packs import news_pack
    from mentor_packs.plan_lines import FIXTURE_PLAN

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    (base / "focus_longs.txt").write_text("", encoding="utf-8")
    (base / "focus_shorts.txt").write_text("AMD\n", encoding="utf-8")
    (base / "focus_swing_longs.txt").write_text("NVDA\nZZZ\n", encoding="utf-8")
    (base / "focus_swing_shorts.txt").write_text("TSLA\n", encoding="utf-8")
    (base / "focus_pick_clocks.json").write_text(
        json.dumps({"picks": {
            "NVDA|long|swing": {"symbol": "NVDA", "side": "long", "category": "swing", "clock_from": "2026-09-22", "reason": "added"},
            "TSLA|short|swing": {"symbol": "TSLA", "side": "short", "category": "swing", "clock_from": "2026-09-25", "reason": "added"},
        }}),
        encoding="utf-8",
    )
    (base / "claimed_picks.jsonl").write_text(
        json.dumps({"schema": "claimed_pick_v1", "action": "claim", "symbol": "NVDA", "side": "LONG", "horizon": "d1",
                    "claimed_setup_id": "avwap_breakout", "claim_at": "2026-09-29T06:40:00-07:00",
                    "session_date": "2026-09-29"}) + "\n",
        encoding="utf-8",
    )
    (base / "pick_feedback.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (
            {"ts": "2026-09-28T06:50:00", "trade_date": "2026-09-28", "symbol": "NVDA", "side": "LONG",
             "verdict": "like", "category": "swing", "origin": "d1", "reason": "", "context": ""},
            {"ts": "2026-08-01T06:50:00", "trade_date": "2026-08-01", "symbol": "NVDA", "side": "LONG",
             "verdict": "dislike", "category": "swing", "origin": "setups", "reason": "too old to show", "context": ""},
            {"ts": "2026-09-26T07:10:00", "trade_date": "2026-09-26", "symbol": "TSLA", "side": "SHORT",
             "verdict": "like", "category": "swing", "origin": "manual", "reason": "", "context": ""},
        )) + "\n",
        encoding="utf-8",
    )
    tier = [FIXTURE_TIER_HEADER]
    for index in range(40):  # avwap_breakout LONG: 24 wins of 40
        tier.append(f"X{index},LONG,avwap_breakout,2026-08-{1 + index % 28:02d},5,{index < 24},{2.0 if index < 24 else -1.5},False")
    for index in range(35):  # avwap_band_bounce SHORT: 14 wins of 35
        tier.append(f"Y{index},SHORT,avwap_band_bounce,2026-08-{1 + index % 28:02d},5,{index < 14},{1.0 if index < 14 else -1.0},False")
    for index in range(12):  # top_pattern SHORT: under the floor
        tier.append(f"W{index},SHORT,top_pattern,2026-08-{1 + index:02d},5,{index < 6},0.5,False")
    tier.append("X0,LONG,avwap_breakout,2026-08-01,5,True,90.0,True")  # stale: never counted
    tier.append("TSLA,SHORT,avwap_band_bounce,2026-09-26,1,False,-0.4,False")
    tier.append("AMD,SHORT,top_pattern,2026-09-25,1,False,-0.4,False")
    (base / "master_avwap_tier_outcomes.csv").write_text("\n".join(tier) + "\n", encoding="utf-8")
    (base / "master_avwap_setup_attribute_leaderboard.csv").write_text(
        "\n".join((
            FIXTURE_LEADERBOARD_HEADER,
            "LONG,favorite_setup,setup.setup_family,avwap_breakout,40,0.30",
            "LONG,near_favorite_zone,setup.setup_family,avwap_breakout,60,-0.10",
            "SHORT,near_favorite_zone,setup.setup_family,avwap_band_bounce,50,0.20",
            "SHORT,near_favorite_zone,setup.symbol,TSLA,50,9.99",
        )) + "\n",
        encoding="utf-8",
    )
    (base / "earnings_calendar_history.json").write_text(
        json.dumps({"schema_version": 1, "symbols": {
            "NVDA": {"events": [{"earnings_date": "2026-11-19"}, {"earnings_date": "2026-08-27"}]},
            "AVGO": {"events": [{"earnings_date": "2026-09-30"}]},
            "AMAT": {"events": [{"earnings_date": "2026-10-20"}]},
            "MU": {"events": [{"earnings_date": "2026-09-24"}]},
            "TSLA": {"events": [{"earnings_date": "2026-10-21"}]},
            "AMD": {"events": [{"earnings_date": "2026-11-03"}]},
        }}),
        encoding="utf-8",
    )
    cohort = [FIXTURE_COHORT_HEADER]
    for horizon in COHORT_HORIZONS:
        cohort.append(f"human_focus_swing,LONG,{horizon},{100 + horizon},0.52,0.004")
        cohort.append(f"human_focus_swing,SHORT,{horizon},{80 + horizon},0.47,-0.002")
        cohort.append(f"human_focus_swing_d1,LONG,{horizon},{10 + horizon},0.60,0.010")
        cohort.append(f"human_focus_m5,SHORT,{horizon},{200 + horizon},0.49,0.001")
    (base / "human_focus_performance.csv").write_text("\n".join(cohort) + "\n", encoding="utf-8")
    plan = base / "trading_plan.md"
    plan.write_text(FIXTURE_PLAN if plan_text is None else plan_text, encoding="utf-8")
    industries = {
        "NVDA": {"industry": "Semiconductors", "industry_member_symbols": ["NVDA", "AVGO", "AMAT", "MU", "AMD", "QCOM"]},
        "AMD": {"industry": "Semiconductors", "industry_member_symbols": ["NVDA", "AVGO", "AMAT", "MU", "AMD", "QCOM"]},
        "TSLA": {"industry": "Auto Manufacturers", "industry_member_symbols": ["TSLA", "F", "GM"]},
    }
    m5_alerts, m5_grades = write_fixture_m5(base)
    return PickPaths(
        m5_alerts=m5_alerts,
        m5_grades=m5_grades,
        focus_longs=base / "focus_longs.txt",
        focus_shorts=base / "focus_shorts.txt",
        focus_swing_longs=base / "focus_swing_longs.txt",
        focus_swing_shorts=base / "focus_swing_shorts.txt",
        pick_clocks=base / "focus_pick_clocks.json",
        claimed_picks=base / "claimed_picks.jsonl",
        pick_feedback=base / "pick_feedback.jsonl",
        tier_outcomes=base / "master_avwap_tier_outcomes.csv",
        leaderboard=base / "master_avwap_setup_attribute_leaderboard.csv",
        earnings_history=base / "earnings_calendar_history.json",
        cohort_performance=base / "human_focus_performance.csv",
        plan=plan,
        industry_map=lambda: industries,
        news=news_pack.fixture_reader(),
        news_stamps=news_pack.fixture_stamps,
        briefs=_write_fixture_briefs(base),
    )


def _write_fixture_briefs(base: Path) -> Path:
    """One night brief manifest (2026-09-28): NVDA briefed, TSLA membership only."""
    folder = base / "briefs" / "2026" / "2026-09-28"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"executive_summary": "NVDA held its AVWAP into the close.",
               "what_is_working": [{"statement": "Higher low on D1 above the anchor.", "evidence_refs": ["scan.tier_list"]}]}
    rows = ({"schema": "ai_ticker_brief_manifest_v2", "session_date": "2026-09-28", "symbol": "NVDA", "status": "briefed",
             "result": {"summary": summary}},
            {"schema": "ai_ticker_brief_manifest_v2", "session_date": "2026-09-28", "symbol": "TSLA",
             "status": "membership_only"})
    (folder / "ticker_briefs_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                                                          encoding="utf-8")
    return base / "briefs"


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build("NVDA", now=FIXTURE_NOW, paths=write_fixture_world(tmp))

