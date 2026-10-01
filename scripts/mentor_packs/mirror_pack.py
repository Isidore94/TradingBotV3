"""Mirror pack: the trader's own record in weekly cuts, each with n. Read-only, deterministic.

Cuts over the last ``weeks`` (default 6), every row with a stable id and ``weeks=``:

- ``mirror:liked:<side>:<h>``: claimed picks graded on daily closes (the D1 scan's own
  outcome row for the name, scanned up to 4 days before the claim) against every scan
  row of that side at the same horizon (5 and 10 sessions).
- ``mirror:veto:<reason>:<side>``: what the vetoed names did 10 sessions later (the veto
  cohort's side returns).
- ``mirror:journal:{kind,hold,hour,weekday,tax}:<k>``: closed journal decisions (option
  legs opened within 3 min paired as one spread): win rate, $ expectancy, net R where a
  planned stop exists ("R unknown" otherwise).
- ``mirror:regime:<r>``: the same, by the trader's structural regime on the entry day.
- ``mirror:asof``, ``mirror:weeks``, ``mirror:caveats``.

A rate carries its Wilson lower bound; under ``evidence_stats.MIN_REPORTABLE_N`` a cut
says "too few" and shows n only. Observations, never a rule proposal. Files are read
directly and the journal ``mode=ro``; call :func:`build` on a worker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs import journal_read
from mentor_packs.registry import Pack, make_pack

NAME = "mirror_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's own record over the last N weeks, in cuts with n: liked picks vs the scan by side "
            "(5 and 10 sessions), what vetoed names did by reason and side (10 sessions), journal results by "
            "kind, hold time, entry hour, weekday and account tax class (net R where a stop exists), and by "
            "his structural regime. Under n=30 a cut says 'too few'. Observations, never rules."
        ),
        "parameters": {
            "type": "object",
            "properties": {"weeks": {"type": "integer", "description": "How many weeks back (1-52); default 6."}},
            "required": [],
        },
    },
}

ET = ZoneInfo("America/New_York")
DEFAULT_WEEKS = 6
MAX_WEEKS = 52
LIKED_HORIZONS = (5, 10)
VETO_HORIZON = 10
#: A claim is graded by the name's scan row from the claim session or up to this many days before.
SETUP_LOOKBACK_DAYS = 4
UNCODED = "uncoded"
#: The id-safe slugs of ``journal_analytics.HOLD_TIME_BUCKETS``, in its order.
HOLD_BUCKETS = (
    ("under 5 min", 5.0), ("5-30 min", 30.0), ("30 min-2 h", 120.0), ("2 h+ same day", None),
    ("overnight, 1-5 days", 5 * 24 * 60.0), ("over 5 days", None),
)
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
CAVEATS = (
    "Journal times are read as ISO 8601 with their offsets; a time without one reads as New York.",
    "Option legs opened within 3 min in one account and underlying count as one spread.",
    "Picks are graded on daily closes only (intraday bar times mix zones).",
    "Claims carry no price at decision time (known_at_claim is empty): a pick is graded from the scan row's close.",
    "Only a few weeks of decision data exist: every cut prints n and says too few under 30.",
    "R needs a planned stop; without one a cut shows $ and says R unknown.",
)
UNHASHED_IDS = frozenset({"mirror:asof"})


def min_reportable_n() -> int:
    from evidence_stats import MIN_REPORTABLE_N

    return int(MIN_REPORTABLE_N)


@dataclass(frozen=True)
class MirrorPaths:
    """Every file the pack reads; tests pass fixture paths, the app uses :func:`live_paths`."""

    journal: Path
    claimed_picks: Path
    annotations: Path
    tier_outcomes: Path
    veto_outcomes: Path


def live_paths() -> MirrorPaths:
    import project_paths as pp

    return MirrorPaths(
        journal=Path(pp.JOURNAL_DB_FILE),
        claimed_picks=Path(pp.CLAIMED_PICKS_FILE),
        annotations=Path(pp.TRADER_ANNOTATIONS_FILE),
        tier_outcomes=Path(pp.MASTER_AVWAP_TIER_OUTCOMES_FILE),
        veto_outcomes=Path(pp.VETO_COHORT_OUTCOMES_FILE),
    )


# ---------------------------------------------------------------- small helpers
def _sym(value: Any) -> str:
    return str(value or "").strip().upper()


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return "SHORT" if text.startswith("SHORT") else "LONG" if text.startswith("LONG") else ""


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", str(value or "").strip()).strip("-").lower() or "unknown"


def _wilson(wins: int, n: int) -> float | None:
    import setup_grades

    return setup_grades.wilson_lower_bound(wins, n)


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value:+.2f}%"


def _money(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


_cache: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}
_cache_lock = threading.Lock()


def _cached(kind: str, path: Path, load: Callable[[Path], Any]) -> Any:
    """``load(path)`` once per file version (mtime + size); a missing file is loaded every time."""
    try:
        stat = path.stat()
    except OSError:
        return load(path)
    signature = (int(stat.st_mtime_ns), int(stat.st_size))
    with _cache_lock:
        hit = _cache.get((kind, str(path)))
        if hit and hit[0] == signature:
            return hit[1]
    value = load(path)
    with _cache_lock:
        _cache[(kind, str(path))] = (signature, value)
    return value


# ---------------------------------------------------------------- the scan outcomes (daily closes)
def _tier_rows(path: Path) -> dict[tuple[str, str, str, int], tuple[bool, float | None, str]]:
    """``{(SYMBOL, SIDE, scan_date, horizon): (win, side_return_pct, known_date)}``, clean rows, first per key."""
    out: dict[tuple[str, str, str, int], tuple[bool, float | None, str]] = {}
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError:
        return out
    wanted = {str(h) for h in LIKED_HORIZONS}
    with handle:
        for row in csv.DictReader(handle):
            horizon = str(row.get("horizon_sessions") or "").strip()
            if horizon not in wanted or str(row.get("stale_horizon") or "").strip().lower() == "true":
                continue
            win = str(row.get("win") or "").strip().lower()
            if win not in ("true", "false"):
                continue
            symbol, side = _sym(row.get("symbol")), _side(row.get("side"))
            scan = str(row.get("scan_date") or "").strip()[:10]
            if not symbol or not side or not scan:
                continue
            key = (symbol, side, scan, int(horizon))
            if key in out:
                continue
            known = str(row.get("future_scan_date") or "").strip()[:10] or scan
            out[key] = (win == "true", journal_read.num(row.get("side_return_pct")), known)
    return out


def _stat(results: list[tuple[bool, float | None]]) -> dict[str, Any]:
    n = len(results)
    wins = sum(1 for win, _ in results if win)
    values = [value for _, value in results if value is not None]
    return {"n": n, "wins": wins, "rate": wins / n if n else None, "lb": _wilson(wins, n) if n else None,
            "avg": sum(values) / len(values) if values else None}


def _stat_text(stat: Mapping[str, Any], floor: int, *, what: str = "win") -> str:
    if stat["n"] < floor:
        return f"n={stat['n']}: too few (floor {floor})"
    return (f"n={stat['n']}, {what} {stat['rate']:.0%} (LB {stat['lb']:.2f}), "
            f"avg side return {_pct(stat['avg'])}")


def _sessions_after(day: date, asof: date) -> int:
    """Weekdays after ``day`` up to ``asof`` (holidays ignored: a close enough count to say "not matured yet")."""
    count, cursor = 0, day
    while cursor < asof:
        cursor += timedelta(days=1)
        count += 1 if cursor.weekday() < 5 else 0
    return count


def liked_rows(claims: list[dict[str, Any]], tier: Mapping, start: date, asof: date, weeks: int,
               floor: int) -> list[dict[str, Any]]:
    import claimed_picks

    through = asof.isoformat()
    # A later drop ends the claim it names (exact symbol, side, setup); an expiry is a faded like, still a like.
    open_claims: dict[tuple[str, str, str], list[int]] = {}
    dropped: set[int] = set()
    for index, row in enumerate(claims):
        action = str(row.get("action") or "").strip().lower()
        key = claimed_picks.claim_key(row)
        if action == "claim":
            open_claims.setdefault(key, []).append(index)
        elif action == "drop" and open_claims.get(key):
            dropped.update(open_claims.pop(key))
    seen: set[tuple[str, str, str]] = set()
    picks: list[tuple[str, str, date]] = []
    for index, row in enumerate(claims):
        if str(row.get("action") or "") != "claim" or index in dropped:
            continue
        symbol, side, day = _sym(row.get("symbol")), _side(row.get("side")), _day(row.get("session_date"))
        if not symbol or not side or day is None or not start <= day <= asof:
            continue
        key = (day.isoformat(), symbol, side)
        if key not in seen:
            seen.add(key)
            picks.append((symbol, side, day))
    rows = []
    for side in ("LONG", "SHORT"):
        for horizon in LIKED_HORIZONS:
            graded: list[tuple[bool, float | None]] = []
            unmatched = maturing = 0
            for symbol, pick_side, day in picks:
                if pick_side != side:
                    continue
                hit = None
                for back in range(SETUP_LOOKBACK_DAYS + 1):
                    hit = tier.get((symbol, side, (day - timedelta(days=back)).isoformat(), horizon))
                    if hit is not None:
                        break
                if hit is None and _sessions_after(day, asof) < horizon:
                    maturing += 1  # the outcome cannot exist yet
                elif hit is None:
                    unmatched += 1
                elif hit[2] > through:
                    maturing += 1
                else:
                    graded.append((hit[0], hit[1]))
            base = _stat([(win, value) for (_s, sd, scan, h), (win, value, known) in tier.items()
                          if sd == side and h == horizon and start.isoformat() <= scan <= through and known <= through])
            mine = _stat(graded)
            base_text = (f"scan {side} baseline n={base['n']}, win {base['rate']:.0%} (LB {base['lb']:.2f}), "
                         f"avg side return {_pct(base['avg'])}" if base["n"] else f"scan {side} baseline n=0")
            extra = []
            if unmatched:
                extra.append(f"{unmatched} with no scan row")
            if maturing:
                extra.append(f"{maturing} not matured yet")
            rows.append({
                "id": f"mirror:liked:{side}:{horizon}", "kind": "liked", "side": side, "horizon": horizon,
                "n": mine["n"], "wins": mine["wins"], "lb": mine["lb"], "avg": mine["avg"], "weeks": weeks,
                "baseline_n": base["n"], "baseline_lb": base["lb"], "baseline_avg": base["avg"],
                "too_few": mine["n"] < floor,
                "text": (f"Liked {side} (claimed picks) at {horizon} sessions, daily closes, weeks={weeks}: "
                         f"{_stat_text(mine, floor)}" + (f" ({', '.join(extra)})" if extra else "")
                         + f"; {base_text}"),
            })
    return rows


# ---------------------------------------------------------------- vetoes
def veto_rows(annotations: list[dict[str, Any]], cohort: Mapping, start: date, asof: date, weeks: int,
              floor: int) -> list[dict[str, Any]]:
    through = asof.isoformat()
    ids = {str(row.get("event_id") or "") for row in annotations if row.get("event_id")}
    seen: set[tuple[str, str, str]] = set()
    cuts: dict[tuple[str, str], list[tuple[bool, float | None]]] = defaultdict(list)
    counted: dict[tuple[str, str], int] = defaultdict(int)
    for row in annotations:
        if row.get("event_type") != "veto" or str(row.get("supersedes") or "") in ids - {""}:
            continue  # a follow-up note on a click is not a second veto
        symbol, side, day = _sym(row.get("symbol")), _side(row.get("side")), _day(row.get("session_date"))
        if not symbol or not side or day is None or not start <= day <= asof:
            continue
        key = (day.isoformat(), symbol, side)
        if key in seen:
            continue  # first veto of a (session, name, side) wins, like the veto cohort
        seen.add(key)
        reason = str(row.get("reason_code") or "").strip().lower() or UNCODED
        counted[(reason, side)] += 1
        found = (cohort.get(key) or {}).get(VETO_HORIZON)
        if found is not None and found[0] <= through:
            cuts[(reason, side)].append((found[1] > 0, found[1] * 100.0))
    rows: list[dict[str, Any]] = []
    for reason, side in sorted(counted):
        stat = _stat(cuts.get((reason, side), []))
        pending = counted[(reason, side)] - stat["n"]
        rows.append({
            "id": f"mirror:veto:{_slug(reason)}:{side}", "kind": "veto", "reason": reason, "side": side,
            "n": stat["n"], "wins": stat["wins"], "lb": stat["lb"], "avg": stat["avg"], "weeks": weeks,
            "too_few": stat["n"] < floor,
            "text": (f"Vetoed {side} for {reason}, what the name did {VETO_HORIZON} sessions later, weeks={weeks}: "
                     f"{_stat_text(stat, floor, what='the name won')}"
                     + (f" ({pending} with no outcome yet)" if pending else "")),
        })
    return rank_vetoes(rows, mean=lambda row: row.get("avg"), measured=lambda row: not row["too_few"])


#: P16: rank 1 = the WORST veto record: the vetoed names' mean side return was highest, so the veto cost the
#: most; the last rank avoided the most loss. Only rows with n at or above the floor are ranked.
def rank_vetoes(rows: list[dict[str, Any]], *, mean: Any, measured: Any) -> list[dict[str, Any]]:
    """Stamp ``rank`` on the measured veto rows (1 = worst record) and start their text with it."""
    ranked = sorted((row for row in rows if measured(row) and mean(row) is not None), key=lambda row: -mean(row))
    total = len(ranked)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        word = ("worst #1" if index == 1 else f"#{index}") + f" of {total} veto records"
        how = "cost the most" if index == 1 else "avoided the most loss" if index == total and total > 1 else ""
        row["text"] = f"{word}{' (' + how + ')' if how else ''}: {row['text']}"
    return rows


# ---------------------------------------------------------------- the journal
def _hold_bucket(unit: journal_read.Unit) -> str | None:
    if unit.opened is None or unit.closed is None:
        return None
    minutes = max(0.0, (unit.closed - unit.opened).total_seconds() / 60.0)
    if unit.closed.date() == unit.opened.date():
        for label, bound in HOLD_BUCKETS[:4]:
            if bound is None or minutes < bound:
                return label
    label, bound = HOLD_BUCKETS[4]
    return label if minutes <= bound else HOLD_BUCKETS[5][0]


def _journal_stat(group: list[journal_read.Unit]) -> dict[str, Any]:
    values = [u.pnl for u in group if u.pnl is not None]
    rs = [u.r for u in group if u.r is not None]
    n = len(values)
    wins = sum(1 for v in values if v > 0)
    return {"n": n, "wins": wins, "rate": wins / n if n else None, "lb": _wilson(wins, n) if n else None,
            "expectancy": sum(values) / n if n else None, "net_r": sum(rs) if rs else None, "r_n": len(rs),
            "units": len(group)}


def _journal_text(title: str, stat: Mapping[str, Any], floor: int, weeks: int) -> str:
    head = f"Journal {title}, closed in the window, weeks={weeks}: "
    if stat["n"] < floor:
        return head + f"n={stat['n']}: too few (floor {floor})"
    text = (f"n={stat['n']}, win {stat['rate']:.0%} (LB {stat['lb']:.2f}), "
            f"expectancy {_money(stat['expectancy'])} per trade")
    if stat["r_n"]:
        text += f"; net R {stat['net_r']:+.1f} over {stat['r_n']} with a stop"
        if stat["r_n"] < stat["n"]:
            text += f" (R unknown for {stat['n'] - stat['r_n']})"
    else:
        text += "; R unknown (no planned stop on file)"
    return head + text


def _cut_rows(prefix: str, label: str, groups: Mapping[str, list[journal_read.Unit]], order: Iterable[str],
              floor: int, weeks: int, *, slug: Callable[[str], str] = _slug) -> list[dict[str, Any]]:
    rows = []
    for key in order:
        group = groups.get(key)
        if not group:
            continue
        stat = _journal_stat(group)
        rows.append({"id": f"{prefix}:{slug(key)}", "kind": prefix.split(":", 1)[1].replace(":", "_"), "key": key,
                     **{k: stat[k] for k in ("n", "wins", "lb", "expectancy", "net_r", "r_n")}, "weeks": weeks,
                     "too_few": stat["n"] < floor, "text": _journal_text(f"{label} {key}", stat, floor, weeks)})
    return rows


def journal_rows(trades: list[dict[str, Any]], accounts: list[Mapping[str, Any]], regime_rows: list[Mapping[str, Any]],
                 start: date, asof: date, weeks: int, floor: int) -> list[dict[str, Any]]:
    from mentor_packs.book_pack import tax_class

    decided = [u for u in journal_read.units(trades) if u.closed is not None and start <= u.closed.date() <= asof]
    by_account = {str(a.get("account_number") or ""): a for a in accounts}
    kinds: dict[str, list] = defaultdict(list)
    holds: dict[str, list] = defaultdict(list)
    hours: dict[str, list] = defaultdict(list)
    days: dict[str, list] = defaultdict(list)
    taxes: dict[str, list] = defaultdict(list)
    regimes: dict[str, list] = defaultdict(list)
    import structural_regime

    timeline = [(d, str(seg["regime"])) for seg in structural_regime.effective_segments(regime_rows)
                if (d := _day(seg.get("start_date"))) is not None]
    for unit in decided:
        kinds[unit.kind].append(unit)
        bucket = _hold_bucket(unit)
        if bucket:
            holds[bucket].append(unit)
        if unit.opened is not None:
            hours[f"{unit.opened.hour:02d}"].append(unit)
            days[WEEKDAYS[unit.opened.weekday()]].append(unit)
        first = unit.trades[0]
        account = by_account.get(unit.account_number) or {"account_label": first.get("account_label") or ""}
        taxes[tax_class(account)].append(unit)
        regime = "unknown"
        if unit.opened is not None:
            for begin, name in timeline:
                if begin <= unit.opened.date():
                    regime = name
        regimes[regime].append(unit)
    rows = _cut_rows("mirror:journal:kind", "kind", kinds, ("day", "swing", "option"), floor, weeks)
    rows += _cut_rows("mirror:journal:hold", "hold", holds, [label for label, _ in HOLD_BUCKETS], floor, weeks)
    rows += _cut_rows("mirror:journal:hour", "entry hour (ET)", hours, sorted(hours), floor, weeks, slug=str)
    rows += _cut_rows("mirror:journal:weekday", "weekday", days, WEEKDAYS, floor, weeks, slug=str)
    rows += _cut_rows("mirror:journal:tax", "account class", taxes, sorted(taxes), floor, weeks)
    rows += [{**row, "kind": "regime", "id": row["id"].replace("mirror:journal:regime", "mirror:regime"),
              "text": row["text"].replace(f"Journal regime {row['key']}",
                                          f"Journal in your regime {structural_regime.label(row['key'])}")}
             for row in _cut_rows("mirror:journal:regime", "regime", regimes,
                                  sorted(regimes, key=lambda k: (k == "unknown", k)), floor, weeks)]
    return rows


# ---------------------------------------------------------------- build
def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def _weeks_arg(weeks: Any) -> int | None:
    try:
        value = int(weeks)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= MAX_WEEKS else None


def _decision_span(claims: list[Mapping[str, Any]], annotations: list[Mapping[str, Any]], asof: date) -> tuple[str, str, int]:
    days = sorted(d for row in list(claims) + [r for r in annotations if r.get("event_type") in ("veto", "like_claim")]
                  if (d := _day(row.get("session_date"))) is not None and d <= asof)
    if not days:
        return "", "", 0
    return days[0].isoformat(), days[-1].isoformat(), (days[-1] - days[0]).days // 7 + 1


def build(weeks: Any = DEFAULT_WEEKS, *, now: datetime | None = None, paths: MirrorPaths | None = None) -> Pack:
    """Build the mirror pack. File and DB reads: call it on a worker."""
    import annotations_reader
    import claimed_picks

    span = _weeks_arg(weeks if weeks not in (None, "") else DEFAULT_WEEKS)
    if span is None:
        return make_pack(NAME, (), empty_text=f"mirror_pack needs weeks between 1 and {MAX_WEEKS}, not {weeks!r}")
    src = paths or live_paths()
    moment = _now(now)
    asof = moment.astimezone(ET).date()
    start = asof - timedelta(days=7 * span)
    floor = min_reportable_n()
    claims = claimed_picks.load_rows(src.claimed_picks)
    annotations = annotations_reader.read_rows(src.annotations)
    tier = _cached("mirror_tier", src.tier_outcomes, _tier_rows)
    cohort = _cached("mirror_veto", src.veto_outcomes, annotations_reader.veto_forward_returns)
    trades = journal_read.read_trades(src.journal, since=(start - timedelta(days=60)).isoformat())
    accounts = journal_read.read_accounts(src.journal)
    regime_rows = journal_read.read_regime_rows(src.journal)
    first, last, data_weeks = _decision_span(claims, annotations, asof)

    rows: list[dict[str, Any]] = [
        {"id": "mirror:asof", "kind": "asof", "asof_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
         "text": f"Mirror as of {moment.astimezone(ET):%a %Y-%m-%d %H:%M} ET"},
        {"id": "mirror:weeks", "kind": "weeks", "weeks": span, "data_weeks": data_weeks, "start": start.isoformat(),
         "end": asof.isoformat(),
         "text": (f"Window weeks={span} ({start.isoformat()} to {asof.isoformat()}); decision data on file: "
                  + (f"{data_weeks} week(s) ({first} to {last}), a thin record: read every n" if data_weeks
                     else "none (no claims or vetoes yet)"))},
    ]
    rows += liked_rows(claims, tier, start, asof, span, floor)
    rows += veto_rows(annotations, cohort, start, asof, span, floor)
    rows += journal_rows(trades, accounts, regime_rows, start, asof, span, floor)
    rows.append({"id": "mirror:caveats", "kind": "caveats", "lines": list(CAVEATS), "text": " | ".join(CAVEATS)})
    return make_pack(NAME, rows)


def pack_hash(pack: Pack) -> str:
    """A stable hash of what the pack SAYS, without the as-of clock."""
    stable = [(row.get("id"), row.get("text")) for row in pack.rows if str(row.get("id")) not in UNHASHED_IDS]
    body = json.dumps({"name": pack.name, "rows": stable, "empty": pack.empty_text}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 28, 13, 50, tzinfo=timezone.utc)  # Mon 2026-09-28, 06:50 PT
FIXTURE_TIER_HEADER = "symbol,side,setup_family,scan_date,future_scan_date,horizon_sessions,win,side_return_pct,stale_horizon"


def _fixture_journal(path: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE trades (trade_id TEXT, account_number TEXT, account_label TEXT, symbol TEXT, security_type TEXT, "
        "direction TEXT, status TEXT, opened_at TEXT, closed_at TEXT, quantity_opened REAL, quantity_closed REAL, "
        "average_entry_price REAL, average_exit_price REAL, net_pnl REAL, net_pnl_usd REAL);"
        "CREATE TABLE trade_legs (leg_id INTEGER PRIMARY KEY, trade_id TEXT, execution_uid TEXT, side TEXT, role TEXT, "
        "quantity REAL, price REAL, timestamp TEXT, commission REAL, fees REAL);"
        "CREATE TABLE trade_annotations (trade_id TEXT, planned_stop REAL, planned_risk REAL);"
        "CREATE TABLE accounts (broker TEXT, account_number TEXT, account_label TEXT, account_type TEXT, raw_json TEXT, "
        "tax_status TEXT);"
        "CREATE TABLE structural_regime (segment_id INTEGER PRIMARY KEY, start_date TEXT, regime TEXT, "
        "structure_note TEXT, supersedes INTEGER);"
        "INSERT INTO accounts VALUES ('QUESTRADE', 'M1', 'Margin 1', 'Margin', '{}', '');"
        "INSERT INTO accounts VALUES ('QUESTRADE', 'T1', 'TFSA 1', 'TFSA', '{}', 'TAX_FREE');"
        "INSERT INTO structural_regime VALUES (1, '2026-06-01', 'weekly_hh_then_compression', '', NULL);"
        "INSERT INTO structural_regime VALUES (2, '2026-09-01', 'bear_channel_lower_highs', '', NULL);"
    )
    rows = []
    first = date(2026, 8, 17)
    for index in range(36):  # 36 day trades in the margin account, 9:35-10:20 ET, 2 of 3 lose
        day = first + timedelta(days=(index % 30) + (index % 30) // 5 * 2)
        opened = datetime(day.year, day.month, day.day, 9, 35 + index % 10, tzinfo=ET)
        pnl = -50.0 if index % 3 else 80.0
        stop = 99.0 if index < 20 else None
        rows.append((f"D{index:02d}", "M1", "Margin 1", f"S{index % 7}", "STK", "LONG", "CLOSED", opened.isoformat(),
                     (opened + timedelta(minutes=12)).isoformat(), 100, 100, 100.0, 100.0 + pnl / 100, pnl, pnl, stop))
    for index in range(4):  # four swings in the TFSA, held 3 days
        opened = datetime(2026, 9, 1 + index, 15, 30, tzinfo=ET)
        rows.append((f"W{index}", "T1", "TFSA 1", "NVDA", "STK", "LONG", "CLOSED", opened.isoformat(),
                     (opened + timedelta(days=3)).isoformat(), 10, 10, 120.0, 125.0, 50.0, 50.0, 110.0))
    spread_open = datetime(2026, 9, 15, 10, 0, tzinfo=ET)  # a call spread: two legs 1 min apart = one decision
    for leg, (symbol, direction, pnl) in enumerate((("AMD261016C00150000", "LONG", 120.0),
                                                     ("AMD261016C00160000", "SHORT", -40.0))):
        rows.append((f"O{leg}", "M1", "Margin 1", symbol, "OPT", direction, "CLOSED",
                     (spread_open + timedelta(minutes=leg)).isoformat(),
                     (spread_open + timedelta(days=2)).isoformat(), 1, 1, 2.0, 3.0, pnl, pnl, None))
    rows.append(("OPEN1", "M1", "Margin 1", "TSLA", "STK", "SHORT", "OPEN", "2026-09-25T10:00:00-04:00", "",
                 10, 0, 250.0, None, None, None, None))
    rows.append(("OLD1", "M1", "Margin 1", "OLD", "STK", "LONG", "CLOSED", "2026-06-01T10:00:00", "2026-06-01T11:00:00",
                 10, 10, 10.0, 11.0, 10.0, 10.0, None))
    conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [r[:15] for r in rows])
    conn.executemany("INSERT INTO trade_annotations VALUES (?, ?, NULL)", [(r[0], r[15]) for r in rows])
    conn.commit()
    conn.close()


def write_fixture_world(root: Path | str) -> MirrorPaths:
    """A small desk covering every cut: 36 liked LONG picks (graded), a few SHORT ones (too few),
    40 LONG vetoes for 'compressed' (n over the floor) and 5 SHORT 'volume_dry' (too few), a journal of
    day trades, TFSA swings and one call spread, and two regimes."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    tier = [FIXTURE_TIER_HEADER]
    claims: list[str] = []
    notes: list[str] = []
    cohort = ["trade_date,symbol,side,source,h1_date,h1_return,h3_date,h3_return,h5_date,h5_return,h10_date,h10_return"]
    first = date(2026, 8, 17)
    for index in range(36):
        day = first + timedelta(days=index % 28)
        symbol = f"L{index:02d}"
        for horizon in LIKED_HORIZONS:
            known = day + timedelta(days=horizon + 4)
            tier.append(f"{symbol},LONG,avwap_breakout,{day},{known},{horizon},{index % 4 != 0},"
                        f"{1.5 if index % 4 else -2.0},False")
        claims.append(json.dumps({"action": "claim", "symbol": symbol, "side": "LONG", "session_date": day.isoformat(),
                                  "claimed_setup_id": "avwap_breakout", "horizon": "d1", "known_at_claim": {}}))
    for index in range(3):
        day = first + timedelta(days=index)
        tier.append(f"H{index},SHORT,general,{day},{day + timedelta(days=9)},5,True,1.0,False")
        claims.append(json.dumps({"action": "claim", "symbol": f"H{index}", "side": "SHORT",
                                  "session_date": day.isoformat(), "horizon": "d1", "known_at_claim": {}}))
    claims.append(json.dumps({"action": "drop", "symbol": "L00", "side": "LONG", "session_date": "2026-08-20"}))
    for index in range(200):  # the scan baselines
        day = first + timedelta(days=index % 30)
        for horizon in LIKED_HORIZONS:
            known = day + timedelta(days=horizon + 4)
            tier.append(f"B{index:03d},LONG,general,{day},{known},{horizon},{index % 2 == 0},{0.5 if index % 2 == 0 else -0.6},False")
            tier.append(f"C{index:03d},SHORT,general,{day},{known},{horizon},{index % 5 < 2},0.1,False")
    tier.append("L01,LONG,avwap_breakout,2026-08-18,2026-08-25,5,True,9.9,True")  # stale: never counted
    for index in range(40):
        day = first + timedelta(days=index % 25)
        symbol = f"V{index:02d}"
        notes.append(json.dumps({"event_type": "veto", "symbol": symbol, "side": "LONG", "session_date": day.isoformat(),
                                 "reason_code": "compressed", "event_id": f"v{index}"}))
        h10 = (day + timedelta(days=14)).isoformat()
        cohort.append(f"{day},{symbol},LONG,veto,{h10},0.01,{h10},0.01,{h10},0.01,{h10},{0.02 if index % 5 else -0.01}")
    notes.append(json.dumps({"event_type": "veto", "symbol": "V00", "side": "LONG", "session_date": "2026-08-17",
                             "reason_code": "compressed", "event_id": "v0-note", "supersedes": "v0", "note": "coiled"}))
    for index in range(5):
        day = first + timedelta(days=index)
        notes.append(json.dumps({"event_type": "veto", "symbol": f"X{index}", "side": "SHORT",
                                 "session_date": day.isoformat(), "reason_code": "volume_dry"}))
    (base / "master_avwap_tier_outcomes.csv").write_text("\n".join(tier) + "\n", encoding="utf-8")
    (base / "claimed_picks.jsonl").write_text("\n".join(claims) + "\n", encoding="utf-8")
    (base / "trader_annotations.jsonl").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (base / "veto_cohort_outcomes.csv").write_text("\n".join(cohort) + "\n", encoding="utf-8")
    _fixture_journal(base / "trade_journal.sqlite3")
    return MirrorPaths(journal=base / "trade_journal.sqlite3", claimed_picks=base / "claimed_picks.jsonl",
                       annotations=base / "trader_annotations.jsonl",
                       tier_outcomes=base / "master_avwap_tier_outcomes.csv",
                       veto_outcomes=base / "veto_cohort_outcomes.csv")


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build(now=FIXTURE_NOW, paths=write_fixture_world(tmp))
