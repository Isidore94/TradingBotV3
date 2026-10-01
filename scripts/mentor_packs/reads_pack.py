"""Reads pack: the trader's own Market Journal reads, their predictions and how they graded. Read-only.

``reads_pack(n=10)``. Sources: the Market Journal ledger (``market_journal_entry_v1``; the trader's own rows only -
never a machine row or a pasted outside forecast), the read grades under ``DAY_REVIEW_READS_DIR`` (superseded rows
hidden; ``read_grades_mature`` closes the matured ones) and the trader's typed regime. Rows: the newest read first
(``kind`` ``current`` when it is today's), then the last ``n`` reads with their clicked call and grades
(``read:<entry_id>``); accuracy by horizon over CLICKED calls only (``read:acc:<horizon>``; an extracted stance is
never pooled with a click: ``read:acc:<horizon>:extracted``) and by the trader's regime at the read
(``read:acc:regime:<regime>``), each with n and the reporting floor; then the calls still awaiting a grade
(``read:open:<k>``). Missing files are "unknown", never "no reads".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "reads_pack"
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The trader's own market reads from his Market Journal: the current read, the last n with his clicked "
            "prediction (direction, horizon, confidence) and its grade (right/wrong/flat/pending), his accuracy by "
            "horizon and by regime (with n and the floor), and the predictions still awaiting a grade."
        ),
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "how many recent reads (default 10)"},
        }, "required": []},
    },
}

ET = ZoneInfo("America/New_York")
DEFAULT_N = 10
MAX_N = 40
#: How far back the ledger and the grade files are read.
LOOKBACK_DAYS = 120
TEXT_CHARS = 220
#: A clicked direction in the trader's words (the gate's conflict line uses them).
BIAS_WORDS = {"up": "bullish", "down": "bearish", "chop": "chop", "range": "range", "no_view": "no view"}
HORIZON_WORDS = {"rest_of_day": "rest of day", "next_5_sessions": "next 5 sessions"}


@dataclass(frozen=True)
class Sources:
    """Where each part reads from; tests pass fixtures, the app uses :func:`live_sources`."""

    entries: Callable[[], Iterable[Mapping[str, Any]]]
    grades: Callable[[], Iterable[Mapping[str, Any]]]
    regime_rows: Callable[[], Iterable[Mapping[str, Any]]] = field(default=lambda: [], compare=False)


def _live_entries() -> list[dict[str, Any]]:
    import market_journal
    from evidence_ledger import EvidenceLedger

    start = (datetime.now(ET).date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    ledger = EvidenceLedger(stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY)
    return list(ledger.read(start=start, event_types=("entry",)).rows)


def read_grade_files(folder: Path | str, *, since: str = "") -> list[dict[str, Any]]:
    """Every current grade row in ``<folder>/<session>.jsonl`` for sessions on or after ``since`` (read-only)."""
    import market_read_grades as grader

    out: list[dict[str, Any]] = []
    try:
        files = sorted(Path(folder).glob("*.jsonl"))
    except OSError:
        return out
    for path in files:
        if since and path.stem < since:
            continue
        out.extend(grader.current_grades(grader.read_grades(path.stem, root=Path(folder).parent)))
    return out


def _live_grades() -> list[dict[str, Any]]:
    from project_paths import DAY_REVIEW_READS_DIR

    since = (datetime.now(ET).date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    return read_grade_files(DAY_REVIEW_READS_DIR, since=since)


def _live_regime_rows() -> list[Mapping[str, Any]]:
    from mentor_packs import context_pack

    return context_pack._live_regime_rows()


def live_sources() -> Sources:
    return Sources(entries=_live_entries, grades=_live_grades, regime_rows=_live_regime_rows)


# ---------------------------------------------------------------- helpers
def _stamp(entry: Mapping[str, Any]) -> datetime | None:
    raw = str(entry.get("created_at") or "")
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def own_reads(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The trader's own current entries (no machine row, no outside forecast), newest first."""
    import market_journal

    picked = [row for row in market_journal.resolve_entries(rows)
              if not market_journal.is_machine_entry(row)
              and str(row.get("origin") or "") != market_journal.ORIGIN_EXTERNAL_FORECAST]
    picked.sort(key=lambda row: (_stamp(row) or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return picked


def call_of(entry: Mapping[str, Any]) -> dict[str, str] | None:
    """The clicked prediction as ``{direction, horizon, confidence, because}``; None when the read has no click."""
    import market_journal

    click = market_journal.prediction_of(entry)
    if click is None:
        return None
    return {"direction": click.direction, "horizon": click.horizon, "confidence": click.confidence,
            "because": click.because}


def call_text(call: Mapping[str, str]) -> str:
    words = f"{BIAS_WORDS.get(call['direction'], call['direction'])} {HORIZON_WORDS.get(call['horizon'], call['horizon'])}"
    return words + (f" ({call['confidence']})" if call.get("confidence") else "")


def _grade_text(grade: Mapping[str, Any]) -> str:
    verdict = str(grade.get("verdict") or "unknown")
    move = grade.get("move_atr")
    moved = f", moved {float(move):+.2f} ATR" if isinstance(move, (int, float)) else ""
    source = "" if str(grade.get("source") or "") == "click" else f", {grade.get('source') or 'unknown'} stance"
    return f"{HORIZON_WORDS.get(str(grade.get('horizon')), grade.get('horizon'))} {grade.get('direction') or '?'} -> {verdict}{moved}{source}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").lower()).strip("_") or "unknown"


def _floor_text(cell: Mapping[str, Any]) -> str:
    import evidence_stats

    n = int(cell.get("n") or 0)
    extra = (f"; {cell['pending']} pending" if cell.get("pending") else "") + (
        f"; {cell['unmeasured']} unmeasured" if cell.get("unmeasured") else "")
    if not cell.get("meets_floor"):
        # Under the floor no rate and no split is shown: a 1-of-2 is not a hit rate (review 2026-09-30).
        return f"too few, n={n} (floor {evidence_stats.MIN_REPORTABLE_N}){extra}"
    rate = f"{100 * cell['rate']:.0f}% right" if cell.get("rate") is not None else "no rate"
    return (f"{cell.get('right', 0)} right / {cell.get('wrong', 0)} wrong / {cell.get('flat', 0)} flat "
            f"(n={n}, {rate}){extra}")


def _cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The accuracy fields a row carries; the rate and its bound are withheld under the floor."""
    out = dict(cell)
    if not out.get("meets_floor"):
        out["rate"] = out["rate_lb"] = None
    return out


def _regime_on(rows: list[Mapping[str, Any]], day: str) -> str:
    import structural_regime

    try:
        found = structural_regime.regime_at(rows, day)
    except Exception:  # noqa: BLE001 - an unreadable regime table is "unknown"
        return "unknown"
    return str((found or {}).get("label") or (found or {}).get("regime") or "unknown")


# ---------------------------------------------------------------- build
def build(n: Any = DEFAULT_N, *, now: datetime | None = None, sources: Sources | None = None) -> Pack:
    """Build the reads pack. File reads: call it on a worker."""
    import market_read_grades as grader

    try:
        count = max(1, min(MAX_N, int(n)))
    except (TypeError, ValueError):
        count = DEFAULT_N
    src = sources or live_sources()
    moment = now or datetime.now(timezone.utc)
    today = (moment if moment.tzinfo else moment.astimezone()).astimezone(ET).date().isoformat()
    try:
        reads = own_reads(src.entries())
    except Exception as exc:  # noqa: BLE001 - an unreadable ledger is unknown, never "no reads"
        return make_pack(NAME, (), empty_text=f"the Market Journal could not be read ({type(exc).__name__}); unknown")
    try:
        grades = [dict(row) for row in src.grades()]
        grades_note = ""
    except Exception as exc:  # noqa: BLE001
        grades, grades_note = [], f"; grades could not be read ({type(exc).__name__}), unknown"
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for grade in grades:
        by_entry.setdefault(str(grade.get("entry_id") or ""), []).append(grade)
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(reads[:count]):
        entry_id = str(entry.get("entry_id") or "")
        stamp = _stamp(entry)
        when = stamp.astimezone(ET).strftime("%a %Y-%m-%d %H:%M") + " ET" if stamp else "time unknown"
        call = call_of(entry)
        mine = by_entry.get(entry_id, [])
        graded = "; ".join(_grade_text(g) for g in mine) if mine else "not graded yet"
        words = " ".join(str(entry.get("text") or "").split())
        if len(words) > TEXT_CHARS:
            words = words[:TEXT_CHARS - 1] + "..."
        symbols = ", ".join(entry.get("symbols") or ()) or "no symbol"
        current = index == 0 and str(entry.get("session_date") or "") == today
        rows.append({
            "id": f"read:{entry_id}", "kind": "current" if current else "read", "entry_id": entry_id,
            "session": str(entry.get("session_date") or ""), "at": stamp.isoformat() if stamp else "",
            "timeframe": str(entry.get("timeframe") or ""), "call": call, "verdicts": [g.get("verdict") for g in mine],
            "text": (f"{'Current read' if current else 'Read'} {when} {entry.get('timeframe') or '?'} ({symbols}): "
                     f"\"{words}\"; call: {call_text(call) if call else 'none clicked'}; grade: {graded}"),
        })
    clicked = [g for g in grades if str(g.get("source") or "") == grader.SOURCE_CLICK]
    extracted = [g for g in grades if str(g.get("source") or "") == grader.SOURCE_EXTRACTED]
    for pool, suffix in ((clicked, ""), (extracted, ":extracted")):
        for horizon in sorted({str(g.get("horizon") or "unknown") for g in pool}):
            cell = grader.accuracy([g for g in pool if str(g.get("horizon") or "unknown") == horizon])
            label = "your clicked calls" if not suffix else "stances read from your words (never pooled with clicks)"
            rows.append({"id": f"read:acc:{horizon}{suffix}", "kind": "accuracy", "horizon": horizon, **_cell(cell),
                         "text": f"Accuracy, {HORIZON_WORDS.get(horizon, horizon)}, {label}: {_floor_text(cell)}"})
    try:
        regime_rows = list(src.regime_rows())
    except Exception:  # noqa: BLE001
        regime_rows = []
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for grade in clicked:
        by_regime.setdefault(_regime_on(regime_rows, str(grade.get("session") or "")), []).append(grade)
    for regime in sorted(by_regime):
        cell = grader.accuracy(by_regime[regime])
        rows.append({"id": f"read:acc:regime:{_slug(regime)}", "kind": "accuracy_regime", "regime": regime, **_cell(cell),
                     "text": f"Accuracy in regime {regime} (your typed regime on the read's day), clicked calls: "
                             f"{_floor_text(cell)}"})
    waiting = 0
    for entry in reversed(reads):
        call = call_of(entry)
        if call is None or call["direction"] == "no_view":
            continue
        mine = by_entry.get(str(entry.get("entry_id") or ""), [])
        pending = [g for g in mine if str(g.get("verdict") or "").startswith(grader.PENDING_PREFIX)]
        if mine and not pending:
            continue
        waiting += 1
        stamp = _stamp(entry)
        when = stamp.astimezone(ET).strftime("%a %Y-%m-%d %H:%M") + " ET" if stamp else "time unknown"
        status = str(pending[0].get("verdict")) if pending else "no grade row yet"
        rows.append({"id": f"read:open:{waiting}", "kind": "open", "entry_id": str(entry.get("entry_id") or ""),
                     "text": f"Open prediction from {when}: {call_text(call)}, awaiting a grade ({status})"})
    if not rows:
        return make_pack(NAME, (), empty_text=f"no reads of yours in the Market Journal in the last {LOOKBACK_DAYS} "
                                              f"days{grades_note}")
    if grades_note:
        rows.append({"id": "read:grades:unknown", "kind": "note", "text": "Grades: unknown" + grades_note})
    return make_pack(NAME, rows)


def disagreement_run(grades: Iterable[Mapping[str, Any]], session: str, days: int = 2) -> list[dict[str, Any]]:
    """The last ``days`` sessions with a measured CLICKED rest-of-day call, ending on ``session``, when every
    measured call on each was graded wrong (his read and the tape disagreed); [] otherwise."""
    import market_read_grades as grader

    by_session: dict[str, list[Mapping[str, Any]]] = {}
    for grade in grades or ():
        day = str(grade.get("session") or "")[:10]
        if (str(grade.get("source") or "") == grader.SOURCE_CLICK and str(grade.get("horizon") or "") == "rest_of_day"
                and str(grade.get("verdict") or "") in (grader.VERDICT_RIGHT, grader.VERDICT_WRONG, grader.VERDICT_FLAT)
                and day and day <= session):
            by_session.setdefault(day, []).append(grade)
    last = sorted(by_session)[-days:]
    if len(last) < days or last[-1] != session:
        return []
    if not all(all(str(g.get("verdict")) == grader.VERDICT_WRONG for g in by_session[day]) for day in last):
        return []
    return [{"session": day, "entry_ids": sorted({str(g.get("entry_id") or "") for g in by_session[day]})}
            for day in last]


def conflict_text(pack: Pack, side: str, now: datetime) -> str:
    """The gate's line when today's read disagrees with the trade's side; "" when it does not (or no read today)."""
    today = now.astimezone(ET).date().isoformat()
    current = next((row for row in pack.rows if row.get("kind") == "current" and row.get("session") == today), None)
    call = (current or {}).get("call") or {}
    direction = call.get("direction")
    wanted = {"LONG": "down", "SHORT": "up"}.get(str(side or "").upper())
    if not current or not wanted or direction != wanted:
        return ""
    at = datetime.fromisoformat(current["at"]).astimezone(ET).strftime("%H:%M") if current.get("at") else "today's"
    return (f"Your {at} read said {BIAS_WORDS[direction]} {HORIZON_WORDS.get(call.get('horizon'), call.get('horizon'))}"
            f"; this is a {str(side).lower()}.")


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 30, 15, 30, tzinfo=timezone.utc)  # Wed 11:30 ET


def _entry(entry_id: str, session: str, created: str, text: str, *, timeframe: str = "M5",
           call: tuple[str, str, str] | None = None, origin: str = "trade_mentor") -> dict[str, Any]:
    mentor: dict[str, Any] = {}
    if call is not None:
        mentor = {"prediction": {"schema": "mentor_prediction_v1", "direction": call[0], "horizon": call[1],
                                 "confidence": call[2], "because": ""}}
    return {"event_type": "entry", "entry_id": entry_id, "session_date": session, "created_at": created,
            "timeframe": timeframe, "symbols": ["SPY"], "origin": origin, "text": text, "supersedes": "",
            "mentor": mentor}


def fixture_sources() -> Sources:
    entries = [
        _entry("mj-a", "2026-09-28", "2026-09-28T14:00:00+00:00", "SPY looks heavy, lower highs",
               call=("down", "rest_of_day", "medium")),
        _entry("mj-b", "2026-09-29", "2026-09-29T14:00:00+00:00", "Trend day up, buyers in control",
               call=("up", "rest_of_day", "high")),
        _entry("mj-c", "2026-09-29", "2026-09-29T12:00:00+00:00", "D1 still in a range", timeframe="D1",
               call=("range", "next_5_sessions", "low")),
        _entry("mj-d", "2026-09-30", "2026-09-30T15:00:00+00:00", "Bearish, failing at the AVWAP",
               call=("down", "rest_of_day", "medium")),
        _entry("mj-m", "2026-09-30", "2026-09-30T15:05:00+00:00", "Auto mode flipped", origin="auto_mode_flip"),
    ]
    grades = [
        {"grade_id": "gr-1", "entry_id": "mj-a", "session": "2026-09-28", "horizon": "rest_of_day",
         "direction": "down", "source": "click", "verdict": "right", "move_atr": -0.6, "supersedes": ""},
        {"grade_id": "gr-2", "entry_id": "mj-b", "session": "2026-09-29", "horizon": "rest_of_day",
         "direction": "up", "source": "click", "verdict": "wrong", "move_atr": -0.4, "supersedes": ""},
        {"grade_id": "gr-3", "entry_id": "mj-c", "session": "2026-09-29", "horizon": "next_5_sessions",
         "direction": "range", "source": "click", "verdict": "pending 2026-10-06", "supersedes": ""},
    ]
    regime = [{"segment_id": 1, "start_date": "2026-09-29", "regime": "chop", "entered_at": "2026-09-29T20:00:00Z"}]
    return Sources(entries=lambda: entries, grades=lambda: grades, regime_rows=lambda: regime)


def fixture() -> Pack:
    return build(n=10, now=FIXTURE_NOW, sources=fixture_sources())


__all__ = ["NAME", "SCHEMA", "Sources", "build", "conflict_text", "fixture", "live_sources", "read_grade_files"]
