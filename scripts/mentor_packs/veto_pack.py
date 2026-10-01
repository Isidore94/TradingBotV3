"""Veto pack: one session's vetoes and passes, each with its slice baseline. Read-only, deterministic.

For every veto the pack finds the D1 setup the name was scanned as that session
(the current tier list, else the tier-outcomes history, up to 4 calendar days back),
then the SLICE: the trader's earlier vetoes with the same (setup, side, reason),
joined to their 5-session D1 outcomes in ``master_avwap_tier_outcomes.csv`` (known by
the close of the vetoed session). The side's BASELINE is every clean 5-session row of
that side in the same CSV. A veto is a candidate challenge only when its slice has
``n >= MIN_REPORTABLE_N`` and its Wilson lower bound clears the side baseline's (guardrail 3);
below the floor the row says "too few" and is never a challenge. A pass is never a
challenge. The veto cohort's own side returns (1/3/5/10 sessions) ride along. Every row
has a stable id; call :func:`build` on a worker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "veto_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's vetoes and passes for one session (default: the last session before today), "
            "each with its slice: past vetoes of the same setup, side and reason, their 5-session win rate "
            "and Wilson lower bound (n printed, floor 30) against the side's baseline, and whether it is a "
            "candidate challenge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Session date YYYY-MM-DD; blank = the last session."},
                "scope": {"type": "string", "description": (
                    "'week', 'month', 'last_week' or 'last_month': instead of one session, every veto in that window "
                    "aggregated by reason and side (what the vetoed names did at 5 and 10 sessions, a verdict per "
                    "reason, and the vetoed set as a whole). Use it for 'did my vetoes work out', 'if I had followed "
                    "my vetoes', 'which veto reason has the worst record'.")},
            },
            "required": [],
        },
    },
}

ET = ZoneInfo("America/New_York")
SLICE_HORIZON_SESSIONS = 5
SETUP_LOOKBACK_DAYS = 4
COHORT_HORIZONS = (1, 3, 5, 10)
UNCODED = "uncoded"
KIND_VETO = "veto"
KIND_PASS = "pass"
KIND_SLICE = "slice"


def min_reportable_n() -> int:
    from evidence_stats import MIN_REPORTABLE_N

    return int(MIN_REPORTABLE_N)


@dataclass(frozen=True)
class VetoPaths:
    """Every file the pack reads; tests pass fixture paths, the app uses :func:`live_paths`."""

    annotations: Path
    tier_outcomes: Path
    tier_list: Path
    veto_outcomes: Path
    vocabularies: Path | None = None


def live_paths() -> VetoPaths:
    import project_paths as pp

    return VetoPaths(
        annotations=Path(pp.TRADER_ANNOTATIONS_FILE),
        tier_outcomes=Path(pp.MASTER_AVWAP_TIER_OUTCOMES_FILE),
        tier_list=Path(pp.MASTER_AVWAP_TIER_LIST_FILE),
        veto_outcomes=Path(pp.VETO_COHORT_OUTCOMES_FILE),
    )


# ---------------------------------------------------------------- small readers
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


def _tier_index(path: Path) -> dict[str, Any]:
    """Setups by (symbol, side, scan_date) and clean 5-session outcomes, from the tier-outcomes CSV."""
    setups: dict[tuple[str, str], dict[str, str]] = {}
    outcomes: dict[tuple[str, str, str], tuple[bool, str]] = {}
    baseline: dict[str, list[tuple[str, bool]]] = {"LONG": [], "SHORT": []}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            symbol, side = _sym(row.get("symbol")), _side(row.get("side"))
            scan = str(row.get("scan_date") or "").strip()[:10]
            family = str(row.get("setup_family") or "").strip()
            if not symbol or not side or not scan:
                continue
            if family:
                setups.setdefault((symbol, side), {})[scan] = family
            if str(row.get("horizon_sessions") or "").strip() != str(SLICE_HORIZON_SESSIONS):
                continue
            if str(row.get("stale_horizon") or "").strip().lower() == "true":
                continue  # the horizon spanned a data gap; not a clean outcome
            win = str(row.get("win") or "").strip().lower()
            if win not in ("true", "false"):
                continue
            known = str(row.get("future_scan_date") or "").strip()[:10] or scan
            outcomes[(symbol, side, scan)] = (win == "true", known)
            baseline[side].append((known, win == "true"))
    return {"setups": setups, "outcomes": outcomes, "baseline": baseline}


def _tier_list(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Setups from the current scan's tier list (rows that have not matured into outcomes yet)."""
    setups: dict[tuple[str, str], dict[str, str]] = {}
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError:
        return setups
    with handle:
        for row in csv.DictReader(handle):
            symbol, side = _sym(row.get("symbol")), _side(row.get("side"))
            scan = str(row.get("scan_date") or "").strip()[:10]
            family = str(row.get("setup_family") or "").strip()
            if symbol and side and scan and family:
                setups.setdefault((symbol, side), {})[scan] = family
    return setups


def _setup_for(symbol: str, side: str, session: str, *sources: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    """The setup family the name was scanned as on ``session``, else up to SETUP_LOOKBACK_DAYS before; "" = unknown."""
    day = _day(session)
    if day is None:
        return ""
    for back in range(SETUP_LOOKBACK_DAYS + 1):
        scan = (day - timedelta(days=back)).isoformat()
        for source in sources:
            family = (source.get((symbol, side)) or {}).get(scan)
            if family:
                return family
    return ""


def _aware(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp or "").strip())
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=ET)  # a naive stamp reads as New York


def _session_of(row: Mapping[str, Any]) -> str:
    from annotations_reader import row_decision_session

    return row_decision_session(row) or str(row.get("session_date") or "").strip()[:10]


def _reason(row: Mapping[str, Any]) -> str:
    return str(row.get("reason_code") or "").strip().lower() or UNCODED


def _events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Vetoes and passes with the trader's follow-up notes folded into the click they belong to."""
    by_id = {str(row.get("event_id") or ""): row for row in rows if row.get("event_id")}
    out: list[dict[str, Any]] = []
    for row in rows:
        target = by_id.get(str(row.get("supersedes") or ""))
        if target is not None and target is not row:
            if row.get("note"):
                target["_note"] = str(row["note"])
            continue
        out.append(row)
    return out


def _weeks(rows: list[dict[str, Any]], through: date) -> tuple[int, str, str]:
    days = sorted(
        day for row in rows
        if (day := _day(row.get("session_date") or row.get("created_at"))) is not None and day <= through
    )
    if not days:
        return 0, "", ""
    return (days[-1] - days[0]).days // 7 + 1, days[0].isoformat(), days[-1].isoformat()


def _wilson(wins: int, n: int) -> float | None:
    import setup_grades

    return setup_grades.wilson_lower_bound(wins, n)


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value * 100:+.2f}%"


# ---------------------------------------------------------------- the slice
def _slice(
    key: tuple[str, str, str],
    history: list[dict[str, Any]],
    tier: dict[str, Any],
    setup_sources: tuple[Mapping, ...],
    cohort: Mapping[tuple[str, str, str], Mapping[int, tuple[str, float]]],
    through: str,
) -> dict[str, Any]:
    """n, wins and LB for past vetoes of this (setup, side, reason), plus their cohort returns."""
    setup, side, reason = key
    n = wins = 0
    returns: dict[int, list[float]] = {horizon: [] for horizon in COHORT_HORIZONS}
    for row in history:
        if row["side"] != side or row["reason"] != reason:
            continue
        if _setup_for(row["symbol"], side, row["session"], *setup_sources) != setup:
            continue
        outcome = tier["outcomes"].get((row["symbol"], side, row["scan"])) if row.get("scan") else None
        if outcome is None:
            outcome = next(
                (tier["outcomes"][k] for back in range(SETUP_LOOKBACK_DAYS + 1)
                 if (k := (row["symbol"], side, (_day(row["session"]) - timedelta(days=back)).isoformat())) in tier["outcomes"]),
                None,
            )
        if outcome is not None and outcome[1] <= through:
            n += 1
            wins += 1 if outcome[0] else 0
        for horizon, (when, value) in (cohort.get((row["cohort_date"], row["symbol"], side)) or {}).items():
            if when <= through:
                returns[horizon].append(value)
    return {"n": n, "wins": wins, "lb": _wilson(wins, n) if n else None, "returns": returns}


def _baseline(tier: dict[str, Any], side: str, through: str) -> dict[str, Any]:
    known = [win for when, win in tier["baseline"].get(side, ()) if when <= through]
    n, wins = len(known), sum(1 for win in known if win)
    return {"n": n, "wins": wins, "lb": _wilson(wins, n) if n else None}


def _cohort_text(returns: Mapping[int, list[float]], floor: int) -> str:
    parts = []
    for horizon in COHORT_HORIZONS:
        values = returns.get(horizon) or []
        if len(values) < floor:
            parts.append(f"{horizon}d too few (n={len(values)})")
        else:
            parts.append(f"{horizon}d {_pct(sum(values) / len(values))} (n={len(values)})")
    return "veto cohort side return " + ", ".join(parts)


# ---------------------------------------------------------------- build
def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def target_session(date_arg: Any = "", *, now: datetime | None = None) -> date | None:
    """``date_arg`` when given, else the last session strictly before today (New York)."""
    if str(date_arg or "").strip():
        return _day(date_arg)
    today = _now(now).astimezone(ET).date()
    try:
        from market_calendar import previous_session

        return previous_session(today)
    except Exception:  # noqa: BLE001 - outside the calendar's range: the last weekday
        cursor = today - timedelta(days=1)
        while cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
        return cursor


def compare_lb(lb: float | None, baseline_lb: float | None, side: str, base: Mapping[str, Any]) -> str:
    """P16: both LBs and which is larger, in words ("LB=0.56 is BELOW the SHORT baseline LB=0.69"), so a
    comparison can never be read backwards."""
    if lb is None or baseline_lb is None:
        return f"LB unknown vs the {side} baseline (n={base.get('n', 0)})"
    word = "ABOVE" if lb > baseline_lb else "BELOW" if lb < baseline_lb else "EQUAL TO"
    rate = f"baseline win rate {base['wins'] / base['n']:.0%}, " if base.get("n") else ""
    return f"LB={lb:.2f} is {word} the {side} baseline LB={baseline_lb:.2f} ({rate}n={base.get('n', 0)})"


#: P16: the aggregate's forward horizons (sessions) and the verdict's horizon.
AGG_HORIZONS = (5, 10)
VERDICT_HORIZON = 5
VERDICT_WORDS = {
    "avoided": "vetoing this reason: avoided a losing cohort",
    "cost": "vetoing this reason: cost a winning cohort",
    "flat": "vetoing this reason: the cohort went nowhere (mean 0)",
    "too_few": "too few",
}


def _agg_verdict(values: list[float], floor: int) -> str:
    if len(values) < floor:
        return "too_few"
    mean = sum(values) / len(values)
    return "avoided" if mean < 0 else "cost" if mean > 0 else "flat"


def _agg_row(row_id: str, label: str, members: list[dict[str, Any]], tier: dict[str, Any], cohort: Mapping,
             through: str, floor: int, baselines: Mapping[str, dict[str, Any]], side: str) -> dict[str, Any]:
    """One aggregate row: n, mean side return, win rate and Wilson LB at 5 and 10 sessions, the D1 outcome vs the
    side baseline (``clears_baseline``) and a deterministic verdict on the 5-session cohort."""
    parts: list[str] = []
    out: dict[str, Any] = {"id": row_id, "kind": "aggregate", "vetoes": len(members)}
    for horizon in AGG_HORIZONS:
        values = [value for m in members
                  for h, (when, value) in (cohort.get((m["cohort_date"], m["symbol"], m["side"])) or {}).items()
                  if h == horizon and when <= through]
        n, wins = len(values), sum(1 for v in values if v > 0)
        lb = _wilson(wins, n) if n else None
        mean = sum(values) / n if n else None
        out[f"h{horizon}"] = {"n": n, "mean": mean, "win_rate": wins / n if n else None, "lb": lb,
                              "pending": len(members) - n}
        if n:
            parts.append(f"{horizon}d: n={n} ({len(members) - n} pending), mean side return {_pct(mean)}, "
                         f"win rate {wins / n:.0%}, LB={lb:.2f}")
        else:
            parts.append(f"{horizon}d: n=0 ({len(members)} pending)")
        if horizon == VERDICT_HORIZON:
            out["verdict"] = _agg_verdict(values, floor)
    d1_n = d1_wins = 0
    for m in members:
        outcome = tier["outcomes"].get((m["symbol"], m["side"], m["scan"])) if m.get("scan") else None
        if outcome is None:
            outcome = tier["outcomes"].get((m["symbol"], m["side"], m["session"]))
        if outcome is not None and outcome[1] <= through:
            d1_n += 1
            d1_wins += 1 if outcome[0] else 0
    d1_lb = _wilson(d1_wins, d1_n) if d1_n else None
    clears = "no"
    if side in ("LONG", "SHORT") and d1_n >= floor and d1_lb is not None and baselines[side]["lb"] is not None:
        clears = "yes" if d1_lb > baselines[side]["lb"] else "no"
        d1_text = (f"D1 {SLICE_HORIZON_SESSIONS}-session outcome n={d1_n}, wins {d1_wins} ({d1_wins / d1_n:.0%}), "
                   f"{compare_lb(d1_lb, baselines[side]['lb'], side, baselines[side])}")
    else:
        d1_text = f"D1 {SLICE_HORIZON_SESSIONS}-session outcome n={d1_n}" + (
            f" (too few vs the floor {floor})" if d1_n < floor else " (mixed sides: no single baseline)")
    verdict = VERDICT_WORDS[out["verdict"]]
    if row_id == "veto:agg:total":
        verdict = verdict.replace("vetoing this reason", "vetoing these names as a whole")
    out.update({"d1_n": d1_n, "d1_wins": d1_wins, "d1_lb": d1_lb, "clears_baseline": clears, "verdict_text": verdict})
    if out["verdict"] == "too_few":
        verdict = f"too few ({out['h5']['n']} measured at 5 sessions, floor {floor})"
    out["text"] = (f"{label}: {len(members)} veto(es); " + "; ".join(parts) + f"; {d1_text}; clears_baseline: "
                   f"{clears}; verdict: {verdict}")
    return out


def build_window(scope: str, *, now: datetime | None = None, paths: VetoPaths | None = None) -> Pack:
    """P16: every veto in a week or month, aggregated per (reason, side) and in total. File reads: on a worker."""
    import annotations_reader
    from mentor_packs import journal_pack

    today = _now(now).astimezone(ET).date()
    span = journal_pack.resolve(scope, today)
    if span is None or not str(scope).strip().lower().replace(" ", "_").endswith(("week", "month")):
        return make_pack(NAME, (), empty_text=f"veto_pack scope must be week, month, last_week or last_month, not {scope!r}")
    label, first, last = span
    src = paths or live_paths()
    through = min(last, today).isoformat()
    floor = min_reportable_n()
    all_rows = annotations_reader.read_rows(src.annotations)
    decisions = _events([dict(row) for row in all_rows if row.get("event_type") == KIND_VETO])
    try:
        tier = _cached("tier", src.tier_outcomes, _tier_index)
    except OSError as exc:
        tier = {"setups": {}, "outcomes": {}, "baseline": {"LONG": [], "SHORT": []}, "error": type(exc).__name__}
    cohort = _cached("veto_outcomes", src.veto_outcomes, annotations_reader.veto_forward_returns)
    members: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in decisions:
        day = _session_of(row)
        symbol, side = _sym(row.get("symbol")), _side(row.get("side"))
        when = _day(day)
        if not symbol or not side or when is None or not first <= when <= last:
            continue
        key = (day, symbol, side)
        if key in seen:
            continue  # first veto of a (session, name, side) wins, like the veto cohort
        seen.add(key)
        members.append({"session": day, "symbol": symbol, "side": side, "reason": _reason(row),
                        "scan": str(row.get("scan_date") or "").strip()[:10],
                        "cohort_date": str(row.get("session_date") or "").strip()[:10] or day})
    baselines = {side: _baseline(tier, side, through) for side in ("LONG", "SHORT")}
    window = f"{first.isoformat()} to {min(last, today).isoformat()}"
    rows: list[dict[str, Any]] = [{
        "id": f"veto:agg:{label}:asof", "kind": "asof", "window": label,
        "text": (f"Vetoes {window} ({label}): {len(members)} veto(es) by reason and side; returns are SIDE returns of "
                 f"the vetoed names (positive = the trade would have made money), measured only once known by "
                 f"{through}; a verdict needs n>={floor} at {VERDICT_HORIZON} sessions; mean < 0 = the veto avoided a "
                 f"losing cohort, mean > 0 = it cost a winning cohort"),
    }]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for m in members:
        groups.setdefault((m["reason"], m["side"]), []).append(m)
    agg: list[dict[str, Any]] = []
    for (reason, side), group in groups.items():
        label_text = annotations_reader.reason_label(reason, side=side, directory=src.vocabularies) if reason != UNCODED             else "no reason code"
        row = _agg_row(f"veto:agg:{reason}:{side}", f"{label_text or reason} ({reason}) {side}", group, tier, cohort,
                       through, floor, baselines, side)
        row.update({"reason_code": reason, "side": side})
        agg.append(row)

    def rank(row: Mapping[str, Any]) -> tuple[int, float]:
        # Worst record first: a veto that cost a winning cohort, by its mean side return; too few last.
        mean = row["h5"]["mean"]
        return (0 if row["verdict"] != "too_few" else 1, -(mean if mean is not None else 0.0))

    rows.extend(sorted(agg, key=rank))
    total_sides = {m["side"] for m in members}
    total = _agg_row("veto:agg:total", "All vetoes together (what the vetoed set did as a whole)", members, tier,
                     cohort, through, floor, baselines, next(iter(total_sides)) if len(total_sides) == 1 else "")
    rows.append(total)
    if tier.get("error"):
        rows.append({"id": "veto:agg:tier_error", "kind": "error",
                     "text": f"tier outcomes unknown ({tier['error']}): D1 outcomes are not measured"})
    return make_pack(NAME, rows)


def build(date: str = "", scope: str = "", *, now: datetime | None = None,  # noqa: A002 - the tool's argument name
          paths: VetoPaths | None = None) -> Pack:
    """Build the veto pack for one session, or (``scope``) a week's / month's aggregate. File reads: on a worker."""
    import annotations_reader

    if str(scope or "").strip():
        return build_window(scope, now=now, paths=paths)
    session = target_session(date, now=now)
    if session is None:
        return make_pack(NAME, (), empty_text=f"veto_pack needs a date like 2026-09-29, not {date!r}")
    src = paths or live_paths()
    through = session.isoformat()
    floor = min_reportable_n()
    all_rows = annotations_reader.read_rows(src.annotations)
    weeks, first, last = _weeks(all_rows, session)
    decisions = _events([dict(row) for row in all_rows if row.get("event_type") in (KIND_VETO, KIND_PASS)])
    try:
        tier = _cached("tier", src.tier_outcomes, _tier_index)
    except OSError as exc:
        tier = {"setups": {}, "outcomes": {}, "baseline": {"LONG": [], "SHORT": []}, "error": type(exc).__name__}
    current = _cached("tier_list", src.tier_list, _tier_list)
    setup_sources = (current, tier["setups"])
    cohort = _cached("veto_outcomes", src.veto_outcomes, annotations_reader.veto_forward_returns)

    history: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    today_rows: list[dict[str, Any]] = []
    for row in decisions:
        day = _session_of(row)
        symbol, side = _sym(row.get("symbol")), _side(row.get("side"))
        if day == through:
            today_rows.append(row)
        if row.get("event_type") != KIND_VETO or not symbol or not side or _day(day) is None or day >= through:
            continue
        key = (day, symbol, side)
        if key in seen:
            continue  # first veto of a (session, name, side) wins, like the veto cohort
        seen.add(key)
        history.append({"session": day, "symbol": symbol, "side": side, "reason": _reason(row),
                        "scan": str(row.get("scan_date") or "").strip()[:10],
                        "cohort_date": str(row.get("session_date") or "").strip()[:10]})
    today_rows.sort(key=lambda row: str(row.get("created_at") or ""))

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    slices: dict[tuple[str, str, str], dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    challenges = vetoes = passes = 0
    for row in today_rows:
        symbol = _sym(row.get("symbol"))
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
        row_id = f"veto:{through}:{symbol}:{counts[symbol]}"
        kind = str(row.get("event_type"))
        side = _side(row.get("side"))
        moment = _aware(row.get("created_at"))
        at = moment.isoformat(timespec="seconds") if moment else ""
        setup = _setup_for(symbol, side, through, *setup_sources) if side else ""
        if kind == KIND_PASS:
            passes += 1
            codes = [str(code).strip().lower() for code in row.get("reason_codes") or () if str(code).strip()]
            labels = [
                annotations_reader.reason_label(code, family=annotations_reader.PASS_FAMILY, version=row.get("vocab_version"),
                                                side=side, directory=src.vocabularies) or f"{code} (label unknown)"
                for code in codes
            ]
            reason_text = "; ".join(labels) or "no reason ticked"
            reason_code = ",".join(codes)
        else:
            vetoes += 1
            reason_code = _reason(row)
            label = "no reason code (not today)" if reason_code == UNCODED else (
                annotations_reader.reason_label(reason_code, version=row.get("vocab_version"), side=side,
                                                directory=src.vocabularies) or "label unknown")
            reason_text = f"{label} ({reason_code})"
        note = str(row.get("_note") or row.get("note") or "").strip()
        rows.append({
            "id": row_id, "kind": kind, "symbol": symbol, "side": side, "setup": setup, "reason_code": reason_code,
            "reason": reason_text, "at": at, "session": through,
            # The veto cohort keys trade_date on the row's own session_date (an after-close veto differs).
            "session_date": str(row.get("session_date") or "").strip()[:10] or through,
            "text": (
                f"{kind.upper()} {symbol} {side or 'side unknown'} {row.get('timeframe') or ''}: {reason_text}; "
                f"setup {setup or 'unknown'}; at {at or 'unknown time'}" + (f"; note: {note[:160]}" if note else "")
            ).replace("  ", " "),
        })
        slice_row: dict[str, Any] = {"id": f"{row_id}:slice", "kind": KIND_SLICE, "challenge": False, "weeks": weeks,
                                     "clears_baseline": "no"}
        if kind == KIND_PASS:
            slice_row["text"] = "A pass (the trader liked the day trade and passed on one issue): no D1 slice, never a challenge"
        elif not side:
            slice_row["text"] = "Side unknown: no slice, never a challenge"
        elif not setup:
            slice_row["text"] = (
                f"Setup unknown (no D1 scan row for {symbol} {side} within {SETUP_LOOKBACK_DAYS} days up to {through}): "
                "no slice, never a challenge"
            )
        else:
            key = (setup, side, reason_code)
            if key not in slices:
                slices[key] = _slice(key, history, tier, setup_sources, cohort, through)
            if side not in baselines:
                baselines[side] = _baseline(tier, side, through)
            cut, base = slices[key], baselines[side]
            head = f"Slice {setup} {side} vetoed for {reason_code} (past vetoes, {SLICE_HORIZON_SESSIONS}-session D1 outcomes known by {through})"
            slice_row.update({"setup": setup, "side": side, "reason_code": reason_code, "n": cut["n"], "wins": cut["wins"],
                              "lb": cut["lb"], "baseline_n": base["n"], "baseline_lb": base["lb"]})
            if cut["n"] < floor:
                verdict = f"too few (n={cut['n']}, floor {floor}); never a challenge"
            else:  # the slice's rows are baseline rows too, so the baseline is never thinner
                clears = cut["lb"] > base["lb"]
                slice_row["challenge"] = clears
                slice_row["clears_baseline"] = "yes" if clears else "no"
                challenges += 1 if clears else 0
                verdict = (
                    f"n={cut['n']}, wins {cut['wins']} ({cut['wins'] / cut['n']:.0%}), "
                    f"{compare_lb(cut['lb'], base['lb'], side, base)}: "
                    + ("clears the baseline, candidate challenge" if clears else "does not clear the baseline, no challenge")
                )
            slice_row["text"] = f"{head}: {verdict}; {_cohort_text(cut['returns'], floor)}; weeks={weeks}"
        rows.append(slice_row)

    summary = (
        f"Session {through}: {vetoes} veto{'es' if vetoes != 1 else ''}, {passes} pass{'es' if passes != 1 else ''}; "
        f"{challenges} candidate challenge{'s' if challenges != 1 else ''} "
        f"(a challenge needs n>={floor} and an LB above the side's baseline LB)"
    )
    if tier.get("error"):
        summary += f"; tier outcomes unknown ({tier['error']})"
    rows.append({"id": f"veto:{through}:summary", "kind": "summary", "vetoes": vetoes, "passes": passes,
                 "challenges": challenges, "text": summary})
    rows.append({
        "id": f"veto:{through}:weeks", "kind": "weeks", "weeks": weeks,
        "text": (f"Decision data: weeks={weeks} ({first} to {last}); a thin record, read every n"
                 if weeks else "Decision data: weeks=0 (no annotations up to this session)"),
    })
    return make_pack(NAME, rows)


def candidates(pack: Pack) -> list[dict[str, Any]]:
    """The candidate-challenge slice rows, each with its veto row: ``[{veto, slice}]``."""
    by_id = {str(row.get("id")): row for row in pack.rows}
    out = []
    for row in pack.rows:
        if row.get("kind") == KIND_SLICE and row.get("challenge"):
            veto = by_id.get(str(row["id"]).removesuffix(":slice"))
            if veto is not None:
                out.append({"veto": veto, "slice": row})
    return out


def pack_hash(pack: Pack) -> str:
    """A stable hash of what the pack SAYS: same hash, same card."""
    stable = [(row.get("id"), row.get("text")) for row in pack.rows]
    body = json.dumps({"name": pack.name, "rows": stable, "empty": pack.empty_text}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 30, 13, 45, tzinfo=timezone.utc)  # Wed 2026-09-30, 06:45 PT
FIXTURE_SESSION = "2026-09-29"
FIXTURE_TIER_HEADER = "symbol,side,setup_family,scan_date,future_scan_date,horizon_sessions,win,stale_horizon"


def _veto(symbol: str, side: str, session: str, reason: str, *, at: str = "12:00:00", event_id: str = "", **extra: Any) -> str:
    row = {"event_type": "veto", "symbol": symbol, "side": side, "session_date": session,
           "created_at": f"{session}T{at}-07:00", "decision_session": session,
           "decision_session_rule": "judged_session_v2", "timeframe": "D1", "vocab_version": 6,
           "event_id": event_id or f"{symbol}-{session}-{at}"}
    if reason:
        row["reason_code"] = reason
    row.update(extra)
    return json.dumps(row)


def write_fixture_world(root: Path | str) -> VetoPaths:
    """A small desk: one challenge (AAA), one above the floor without edge (BBB), one too few (CCC),
    one pass (DDD), one unknown setup (EEE), all on 2026-09-29."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    tier = [FIXTURE_TIER_HEADER]
    cohort = ["trade_date,symbol,side,source,h1_date,h1_return,h3_date,h3_return,h5_date,h5_return,h10_date,h10_return"]
    first = date(2026, 8, 3)
    for index in range(40):  # LONG avwap_breakout vetoed for compressed: 34 wins of 40
        day = (first + timedelta(days=index % 20)).isoformat()
        symbol = f"L{index:02d}"
        notes.append(_veto(symbol, "LONG", day, "compressed"))
        future = (_day(day) + timedelta(days=7)).isoformat()
        tier.append(f"{symbol},LONG,avwap_breakout,{day},{future},5,{index < 34},False")
        cohort.append(f"{day},{symbol},LONG,veto_v6_compressed,{future},0.01,{future},0.02,{future},0.03,{future},0.04")
    for index in range(35):  # SHORT avwap_band_bounce vetoed for too_extended_from_base: 15 wins of 35
        day = (first + timedelta(days=index % 20)).isoformat()
        symbol = f"S{index:02d}"
        notes.append(_veto(symbol, "SHORT", day, "too_extended_from_base"))
        tier.append(f"{symbol},SHORT,avwap_band_bounce,{day},{(_day(day) + timedelta(days=7)).isoformat()},5,{index < 15},False")
    for index in range(10):  # LONG top_pattern vetoed for volume_dry: under the floor
        day = (first + timedelta(days=index)).isoformat()
        notes.append(_veto(f"T{index:02d}", "LONG", day, "volume_dry"))
        tier.append(f"T{index:02d},LONG,top_pattern,{day},{(_day(day) + timedelta(days=7)).isoformat()},5,{index < 5},False")
    for index in range(200):  # the side baselines: LONG 45%, SHORT 50%
        day = (first + timedelta(days=index % 30)).isoformat()
        future = (_day(day) + timedelta(days=7)).isoformat()
        tier.append(f"BL{index:03d},LONG,general,{day},{future},5,{index % 20 < 9},False")
        tier.append(f"BS{index:03d},SHORT,general,{day},{future},5,{index % 2 == 0},False")
    tier.append("L00,LONG,avwap_breakout,2026-08-03,2026-08-10,5,True,True")  # stale: never counted
    tier.append("ZZZ,LONG,general,2026-09-20,2026-09-30,5,True,False")  # not known by 2026-09-29
    (base / "master_avwap_tier_outcomes.csv").write_text("\n".join(tier) + "\n", encoding="utf-8")
    (base / "master_avwap_tier_list.csv").write_text(
        "scan_date,symbol,side,setup_family\n"
        f"{FIXTURE_SESSION},AAA,LONG,avwap_breakout\n"
        f"{FIXTURE_SESSION},BBB,SHORT,avwap_band_bounce\n"
        f"2026-09-28,CCC,LONG,top_pattern\n",
        encoding="utf-8",
    )
    (base / "veto_cohort_outcomes.csv").write_text("\n".join(cohort) + "\n", encoding="utf-8")
    notes += [
        _veto("AAA", "LONG", FIXTURE_SESSION, "compressed", at="09:05:00", event_id="aaa-click"),
        _veto("AAA", "LONG", FIXTURE_SESSION, "compressed", at="09:05:07", event_id="aaa-note",
              supersedes="aaa-click", note="coiled under the 50"),
        _veto("BBB", "SHORT", FIXTURE_SESSION, "too_extended_from_base", at="09:10:00"),
        _veto("CCC", "LONG", FIXTURE_SESSION, "volume_dry", at="09:20:00"),
        json.dumps({"event_type": "pass", "symbol": "DDD", "side": "LONG", "session_date": FIXTURE_SESSION,
                    "created_at": f"{FIXTURE_SESSION}T10:00:00-07:00", "decision_session": FIXTURE_SESSION,
                    "decision_session_rule": "judged_session_v2", "timeframe": "M5", "event_id": "ddd",
                    "reason_codes": ["low_rvol"], "vocab_version": 1, "vocabulary_id": "pass_reasons"}),
        _veto("EEE", "LONG", FIXTURE_SESSION, "compressed", at="11:00:00"),
        _veto("AAA", "SHORT", FIXTURE_SESSION, "", at="12:30:00"),
        json.dumps({"event_type": "like_claim", "symbol": "FFF", "side": "LONG", "session_date": FIXTURE_SESSION,
                    "created_at": f"{FIXTURE_SESSION}T12:40:00-07:00", "claimed_setup_id": "avwap_breakout"}),
        "{torn row",
    ]
    (base / "trader_annotations.jsonl").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return VetoPaths(
        annotations=base / "trader_annotations.jsonl",
        tier_outcomes=base / "master_avwap_tier_outcomes.csv",
        tier_list=base / "master_avwap_tier_list.csv",
        veto_outcomes=base / "veto_cohort_outcomes.csv",
    )


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build(now=FIXTURE_NOW, paths=write_fixture_world(tmp))
