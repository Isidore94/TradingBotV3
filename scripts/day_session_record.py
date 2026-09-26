"""Day Recap coach: one complete point-in-time record per session.

`DAY_SESSION_RECORDS_DIR/<date>.json` (schema `day_session_record_v1`) plus a
short `<date>.md` written from it, and `week-<YYYY-Www>.json` rollups. The files
are NEVER pruned (`day_review_index._prune` only walks `sessions/`).

Every item names its source store and source id. Missing values are `None` /
"unknown", never zero. What was known at the time sits beside `outcome` blocks
that were measured later (`measured_later: true`, at `outcomes_measured_at`).

`collect_inputs` does every store read (worker / night only); `build_record`,
`build_week` and `render_markdown` are pure. Writes are tmp + replace, skip an
unchanged record, and never replace a good record with a degraded one.
Evidence only: nothing reads these files to detect, score, rank, gate or alert.

CLI: `python scripts/day_session_record.py --date YYYY-MM-DD [--dry-run]`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import project_paths
from swallowed import note_swallowed

_log = logging.getLogger(__name__)

SCHEMA = project_paths.DAY_SESSION_RECORD_SCHEMA
WEEK_SCHEMA = "day_session_week_v1"

#: Below this many results a rollup row says "too few to tell".
MIN_N = 10
#: How many sessions the night rebuilds, ending at the closed target.
DEFAULT_REBUILD_SESSIONS = 5
#: How many rows per pick population the `.md` summary lists (the JSON has all).
MARKDOWN_PICKS = 8

SECTIONS: tuple[str, ...] = (
    "session_facts",
    "market_context",
    "trades",
    "alerts",
    "picks",
    "calls",
    "notes",
    "mood",
    "ai_ideas",
    "recap",
    "setup_grades",
)

PICK_POPULATIONS: tuple[str, ...] = ("liked_not_traded", "rejected", "traded_left_early", "claimed_d1")
PICK_DECISION_FIELDS: tuple[str, ...] = (
    "time", "symbol", "side", "category", "what_you_did", "reason", "instrument", "traded",
)
PICK_OUTCOME_FIELDS: tuple[str, ...] = (
    "state", "ran_after_pct", "held_at_close_pct", "against_first_pct", "at_close_pct",
    "ran_after_atr", "against_first_atr", "at_close_atr", "horizon_moves", "you_made",
    "left_on_table_pct", "real_miss", "sessions_held", "assignment", "not_judged_reason",
)
CALL_OUTCOME_FIELDS: tuple[str, ...] = ("verdict", "move_atr", "checkpoints", "grader_gap", "span")
TRADE_FIELDS: tuple[str, ...] = (
    "trade_id", "symbol", "direction", "status", "opened_at", "closed_at", "quantity_opened",
    "quantity_closed", "average_entry_price", "average_exit_price", "currency", "setup_tags",
    "auto_tag_summary", "planned_entry", "planned_stop", "planned_risk", "notes",
    "review_pnl_note", "label_provenance", "tag_status",
)
LEG_DROP = ("raw_json",)
ALERT_DROP = ("installation_id", "machine", "pid")

UNKNOWN = "unknown"

TIME_BUCKETS = (
    ("premarket", (0, 0), (9, 30)),
    ("open", (9, 30), (10, 30)),
    ("midday", (10, 30), (14, 0)),
    ("close", (14, 0), (16, 0)),
    ("after_hours", (16, 0), (24, 0)),
)


# ---------------------------------------------------------------------------
# plain values
# ---------------------------------------------------------------------------
def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return str(value)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _source(store: str, identifier: Any) -> dict[str, str]:
    return {"store": store, "id": _text(identifier) or UNKNOWN}


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _market_time(value: Any) -> datetime | None:
    """An aware stamp in New York time, or None (a naive stamp is unknown)."""
    import market_calendar

    moment = _moment(value)
    if moment is None or moment.tzinfo is None:
        return None
    return moment.astimezone(market_calendar.MARKET_TZ)


def time_bucket(value: Any) -> str:
    moment = _market_time(value)
    if moment is None:
        return UNKNOWN
    here = (moment.hour, moment.minute)
    for name, start, end in TIME_BUCKETS:
        if start <= here < end:
            return name
    return UNKNOWN


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _auto_environment(pack: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    shifts: list[dict[str, Any]] = []
    d1_label = ""
    for index, item in enumerate((pack or {}).get("environment") or ()):
        if not isinstance(item, Mapping):
            continue
        if item.get("kind") == "d1_label":
            d1_label = _text(item.get("label"))
            continue
        if item.get("kind") != "regime_shift":
            continue
        shifts.append({
            "label": _text(item.get("to_regime")) or UNKNOWN,
            "from_label": _text(item.get("from_regime")) or UNKNOWN,
            "at": _text(item.get("event_at")),
            "by": _text(item.get("source")) or UNKNOWN,
            "source": _source("evidence_ledgers/market_regime_shifts", item.get("source_id") or f"shift:{index}"),
        })
    shifts.sort(key=lambda row: row["at"])
    return shifts, d1_label


def auto_label_at(shifts: Sequence[Mapping[str, Any]], stamp: Any) -> str:
    """The auto regime in force at `stamp`, from the session's shifts, or unknown."""
    moment = _moment(stamp)
    if moment is None or moment.tzinfo is None or not shifts:
        return UNKNOWN
    label = _text(shifts[0].get("from_label")) or UNKNOWN
    for shift in shifts:
        at = _moment(shift.get("at"))
        if at is None or at.tzinfo is None:
            continue
        if at <= moment:
            label = _text(shift.get("label")) or UNKNOWN
    return label


def _index_closes(pack: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in (pack or {}).get("measured") or ():
        if not isinstance(cell, Mapping):
            continue
        close = _number(cell.get("close"))
        status = _text(cell.get("status"))
        rows.append({
            "symbol": _text(cell.get("symbol")) or UNKNOWN,
            "close": close,
            "change_pct": _number(cell.get("change_pct")),
            "status": status if status == "measured" and close is not None else UNKNOWN,
            "bars_through": _text(cell.get("bars_through")),
            "reason": _text(cell.get("reason")),
            "source": _source("market_story.measured", cell.get("source_id") or f"measured:{cell.get('symbol')}"),
        })
    return rows


def _market_context(session: str, inputs: Mapping[str, Any], recap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import recap_store

    shifts, d1_label = _auto_environment(inputs.get("pack") or {})
    manual = []
    for row in inputs.get("env_annotations") or ():
        if isinstance(row, Mapping):
            manual.append({
                "at": _text(row.get("ts")),
                "user_mode": _text(row.get("user_mode")),
                "auto_environment": _text(row.get("auto_environment")),
                "reason": _text(row.get("reason")),
                "source": _source("market_environment_annotations.jsonl", row.get("ts")),
            })
    verdicts = [row for row in recap_rows if row.get("kind") == recap_store.KIND_ENVIRONMENT_VERDICT]
    verdict = max(verdicts, key=lambda row: _text(row.get("recorded_at"))) if verdicts else None
    label = recap_store.environment_label_for(session, rows=recap_rows)
    if label["source"] == "none":
        label = {
            "label": shifts[-1]["label"] if shifts else UNKNOWN,
            "source": "auto_last" if shifts else "none",
            "verdict_id": "",
        }
    return {
        "index_closes": _index_closes(inputs.get("pack") or {}),
        "auto_environment": shifts,
        "d1_label": d1_label or UNKNOWN,
        "manual_selections": manual,
        "trader_verdict": dict(verdict) if verdict else None,
        "session_label": label,
    }


def _mentor_origin(answers: Iterable[Mapping[str, Any]]) -> str:
    origin = ""
    for row in answers or ():
        payload = row.get("payload") if isinstance(row, Mapping) else None
        if isinstance(payload, Mapping) and payload.get("mentor_question_kind") == "trade_origin":
            origin = _text(payload.get("trade_origin")) or origin
    return origin or UNKNOWN


def _exit_reason(*sources: Any) -> str:
    for fields in sources:
        if isinstance(fields, Mapping):
            why = fields.get("why")
            code = _text(why.get("code")) if isinstance(why, Mapping) else ""
            if code:
                return code
    return UNKNOWN


def setup_family(tags: Any) -> str:
    for part in re.split(r"[;,]", _text(tags)):
        if part.strip():
            return part.strip().lower()
    return UNKNOWN


def _trades(inputs: Mapping[str, Any], shifts: list[dict[str, Any]]) -> dict[str, Any]:
    from journal_analytics import counts_in_pnl, trade_r_multiple

    payload = inputs.get("payload") or {}
    reviews = {
        _text(row.get("trade_id")): row
        for row in payload.get("trade_reviews") or () if isinstance(row, Mapping)
    }
    legs_by_trade = inputs.get("trade_legs") or {}
    mentor_by_trade = inputs.get("mentor_answers") or {}
    rows: list[dict[str, Any]] = []
    for trade in payload.get("trades") or ():
        if not isinstance(trade, Mapping):
            continue
        trade_id = _text(trade.get("trade_id"))
        closed = _text(trade.get("status")).upper() == "CLOSED"
        counted = counts_in_pnl(dict(trade))
        pnl = _number(trade.get("net_pnl")) if closed else None
        pnl_cad = _number(trade.get("net_pnl_cad")) if closed else None
        # The journal's one R, native currency; `net_pnl_cad` stays as the money column.
        r_value = trade_r_multiple(dict(trade)) if closed else None
        review = reviews.get(trade_id) or {}
        answers = [_plain(row) for row in mentor_by_trade.get(trade_id) or ()]
        item = {name: _plain(trade.get(name)) for name in TRADE_FIELDS}
        item.update({
            "net_pnl": pnl,
            "net_pnl_cad": pnl_cad,
            "pnl_known": pnl is not None,
            "counted_in_pnl": counted,
            "r_multiple": r_value,
            "setup_family": setup_family(trade.get("setup_tags")),
            "grade": UNKNOWN,
            "time_bucket": time_bucket(trade.get("opened_at")),
            "auto_environment_at_open": auto_label_at(shifts, trade.get("opened_at")),
            "origin": _mentor_origin(answers),
            "exit_reason": _exit_reason(trade.get("exit_fields"), review.get("exit_fields")),
            "legs": [
                {key: _plain(value) for key, value in dict(leg).items() if key not in LEG_DROP}
                for leg in legs_by_trade.get(trade_id) or () if isinstance(leg, Mapping)
            ],
            "entry_words": _plain(review.get("entry_raw") or {}),
            "entry_answers": _plain(review.get("entry_answers") or {}),
            "exit_words": _plain(review.get("exit_raw") or {"text": _text(trade.get("exit_note"))}),
            "exit_fields": _plain(trade.get("exit_fields") or review.get("exit_fields") or {}),
            "mentor_answers": answers,
            "source": _source("trade_journal.sqlite3:trades", trade_id),
        })
        rows.append(item)
    # Totals add only trades whose entry is real; a made-up entry keeps its row.
    known = [row["net_pnl"] for row in rows if row["net_pnl"] is not None and row["counted_in_pnl"]]
    return {
        "n": len(rows),
        "n_pnl_known": len(known),
        "net_pnl": round(sum(known), 6) if known else None,
        "rows": rows,
    }


def _alerts(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in inputs.get("alert_reviews") or ():
        if not isinstance(row, Mapping):
            continue
        item = {key: _plain(value) for key, value in row.items() if key not in ALERT_DROP}
        item["source"] = _source("alert_review_events", row.get("review_record_id"))
        rows.append(item)
    rows.sort(key=lambda row: _text(row.get("ts")))
    actions = Counter(_text(row.get("action")) or UNKNOWN for row in rows)
    return {"n": len(rows), "by_action": dict(sorted(actions.items())), "rows": rows}


def _walk_rows(walkaway: Any, population: str) -> list[Any]:
    found = getattr(walkaway, population, None)
    if found is None and isinstance(walkaway, Mapping):
        found = walkaway.get(population)
    return list(found or ())


def _picks(inputs: Mapping[str, Any]) -> dict[str, Any]:
    walkaway = (inputs.get("payload") or {}).get("walkaway")
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for population in PICK_POPULATIONS:
        found = _walk_rows(walkaway, population)
        counts[population] = len(found)
        for raw in found:
            row = _plain(raw)
            if not isinstance(row, Mapping):
                continue
            identity = row.get("decision_id") or ()
            key = "|".join(str(part) for part in identity) if isinstance(identity, list) else _text(identity)
            item = {name: row.get(name) for name in PICK_DECISION_FIELDS}
            item["population"] = population
            item["outcome"] = {
                **{name: row.get(name) for name in PICK_OUTCOME_FIELDS},
                "measured_later": True,
            }
            item["source"] = _source("walkaway_day decisions", key)
            rows.append(item)
    return {"n": len(rows), "counts": counts, "rows": rows}


def _calls(inputs: Mapping[str, Any]) -> dict[str, Any]:
    payload = inputs.get("payload") or {}
    rows: list[dict[str, Any]] = []
    for read in payload.get("reads") or ():
        if not isinstance(read, Mapping):
            continue
        body = _plain(dict(read))
        outcome = {name: body.pop(name, None) for name in CALL_OUTCOME_FIELDS}
        outcome["measured_later"] = True
        body.pop("source_id", None)
        body["outcome"] = outcome
        session = _text(read.get("session"))[:10]
        body["source"] = _source(f"day_review/reads/{session or 'session'}.jsonl", read.get("read_id"))
        rows.append(body)
    congruence = [_plain(dict(line)) for line in payload.get("congruence") or () if isinstance(line, Mapping)]
    return {"n": len(rows), "rows": rows, "congruence": congruence}


def _after_close(session: str, stamp: Any) -> bool | None:
    import market_calendar

    moment = _moment(stamp)
    if moment is None or moment.tzinfo is None:
        return None
    return moment >= market_calendar.session_close(date.fromisoformat(session))


def _notes(session: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    import market_journal

    rows = []
    for entry in (inputs.get("payload") or {}).get("entries") or ():
        if not isinstance(entry, Mapping) or market_journal.is_machine_entry(entry):
            continue
        rows.append({
            "entry_id": _text(entry.get("entry_id")),
            "created_at": _text(entry.get("created_at")),
            "after_close": _after_close(session, entry.get("created_at")),
            "text": _text(entry.get("text")),
            "timeframe": _text(entry.get("timeframe")),
            "origin": _text(entry.get("origin")),
            "mentor": _plain(entry.get("mentor")) if entry.get("mentor") else None,
            "source": _source("market_journal", entry.get("entry_id")),
        })
    return {"n": len(rows), "rows": rows}


def _mood(inputs: Mapping[str, Any]) -> dict[str, Any]:
    section = (inputs.get("payload") or {}).get("mood") or {}
    recorded = []
    for row in (section.get("recorded") if isinstance(section, Mapping) else None) or ():
        if isinstance(row, Mapping):
            recorded.append({**_plain(dict(row)), "source": _source("market_journal", row.get("entry_id"))})
    return {"n": len(recorded), "recorded": recorded}


def _ideas(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in inputs.get("ideas") or ():
        if not isinstance(row, Mapping):
            continue
        item = _plain(dict(row))
        item["status"] = _text(row.get("status")) or "undecided"
        item["source"] = _source("ai_ideas.jsonl + ai_ideas_state.json", row.get("idea_id"))
        rows.append(item)
    return {"n": len(rows), "rows": rows}


def rule_reflections(mentor_by_trade: Any) -> list[dict[str, Any]]:
    """The Mentor's "kept or broke today's rule?" answers, one row per trade."""
    rows: list[dict[str, Any]] = []
    if not isinstance(mentor_by_trade, Mapping):
        return rows
    for trade_id, answers in mentor_by_trade.items():
        latest: Mapping[str, Any] | None = None
        for row in answers or ():
            payload = row.get("payload") if isinstance(row, Mapping) else None
            if isinstance(payload, Mapping) and payload.get("mentor_question_kind") == "rule_reflection":
                if latest is None or _text(payload.get("answered_at")) >= _text(latest.get("answered_at")):
                    latest = payload
        if latest is None:
            continue
        rows.append({
            "trade_id": _text(trade_id),
            "symbol": _text(latest.get("symbol")),
            "rule_tag": _text(latest.get("rule_tag")) or UNKNOWN,
            "answer": _text(latest.get("rule_kept")) or UNKNOWN,
            "answered_at": _text(latest.get("answered_at")),
            "source": _source("trade_journal.sqlite3:opportunity_events", trade_id),
        })
    rows.sort(key=lambda row: (row["answered_at"], row["trade_id"]))
    return rows


def _recap(inputs: Mapping[str, Any], recap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in recap_rows:
        rows.append({**row, "source": _source("day_recap_events.jsonl", row.get("id"))})
    counts = Counter(_text(row.get("kind")) for row in rows)
    rule = inputs.get("rule_for_session")
    streak = inputs.get("rule_streak")
    return {
        "n": len(rows),
        "counts": dict(sorted(counts.items())),
        "rows": rows,
        "rule_checked_today": _plain(dict(rule)) if isinstance(rule, Mapping) else None,
        "rule_streak": int(streak) if isinstance(streak, int) and not isinstance(streak, bool) else None,
        "rule_reflections": rule_reflections(inputs.get("mentor_answers")),
    }


def _setup_grades(sections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Grades that were stamped on a source row at the time. None are computed here."""
    rows = []
    for name in ("alerts", "picks", "trades"):
        for item in (sections.get(name) or {}).get("rows") or ():
            grade = _text(item.get("grade") or item.get("setup_grade"))
            if not grade or grade == UNKNOWN:
                continue
            rows.append({
                "grade": grade,
                "symbol": _text(item.get("symbol")),
                "side": _text(item.get("side") or item.get("direction")),
                "on": name,
                "source": dict(item.get("source") or {}),
            })
    return {"status": "recorded" if rows else "not_recorded", "rows": rows}


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
def _content_hash(record: Mapping[str, Any]) -> str:
    body = {
        key: value for key, value in record.items()
        if key not in ("built_at", "outcomes_measured_at", "content_hash")
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def build_record(session_date: str, inputs: Mapping[str, Any], *, built_at: datetime) -> dict[str, Any]:
    """One session's complete record from already-read inputs. Pure."""
    import market_calendar

    session = _text(session_date)[:10]
    day = date.fromisoformat(session)
    payload = inputs.get("payload") or {}
    recap_rows = [dict(row) for row in inputs.get("recap") or () if isinstance(row, Mapping)]
    market = _market_context(session, inputs, recap_rows)
    sections: dict[str, Any] = {
        "market_context": market,
        "trades": _trades(inputs, market["auto_environment"]),
        "alerts": _alerts(inputs),
        "picks": _picks(inputs),
        "calls": _calls(inputs),
        "notes": _notes(session, inputs),
        "mood": _mood(inputs),
        "ai_ideas": _ideas(inputs),
        "recap": _recap(inputs, recap_rows),
    }
    sections["setup_grades"] = _setup_grades(sections)
    # The grades the desk showed at the open, from the dated grade history (None before history began).
    sections["setup_grades"]["at_open"] = _plain(inputs.get("grades_at_open"))
    market["opening_regime"] = _plain(inputs.get("opening_regime"))
    unread = [str(item) for item in payload.get("pack_sources_unread") or ()]
    unread.extend(str(item) for item in inputs.get("unread") or ())
    facts = {
        "session_date": session,
        "weekday": day.strftime("%A"),
        "session_close": market_calendar.session_close(day).isoformat(),
        "provisional": bool(payload.get("provisional")),
        "sources_unread": sorted(set(unread)),
        "read_errors": _text(inputs.get("read_errors") or payload.get("error")),
        "counts": {name: sections[name].get("n", 0) for name in ("trades", "alerts", "picks", "calls", "notes", "ai_ideas", "recap")},
    }
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "session_date": session,
        "session_facts": facts,
        **sections,
        "outcomes_measured_at": built_at.isoformat(),
        "built_at": built_at.isoformat(),
    }
    record["content_hash"] = _content_hash(record)
    return record


def iter_items(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every sourced item in a record, for readers and tests."""
    items: list[Mapping[str, Any]] = []
    market = record.get("market_context") or {}
    for name in ("index_closes", "auto_environment", "manual_selections"):
        items.extend(market.get(name) or ())
    for name in ("trades", "alerts", "picks", "calls", "notes", "ai_ideas", "recap", "setup_grades"):
        items.extend((record.get(name) or {}).get("rows") or ())
    items.extend((record.get("mood") or {}).get("recorded") or ())
    return items


def _fmt(value: Any, *, places: int = 2, suffix: str = "") -> str:
    number = _number(value)
    return UNKNOWN if number is None else f"{number:.{places}f}{suffix}"


def render_markdown(record: Mapping[str, Any]) -> str:
    """A short human summary, written FROM the record and nothing else."""
    session = _text(record.get("session_date"))
    facts = record.get("session_facts") or {}
    market = record.get("market_context") or {}
    lines = [f"# Day record {session} ({facts.get('weekday', '')})", ""]
    if facts.get("sources_unread"):
        lines.append(f"Unread sources: {', '.join(facts['sources_unread'])}.")
    closes = ", ".join(
        f"{row['symbol']} {_fmt(row.get('close'))} ({_fmt(row.get('change_pct'), suffix='%')})"
        for row in market.get("index_closes") or ()
    )
    lines.append(f"Index closes: {closes or UNKNOWN}.")
    autos = " -> ".join(row["label"] for row in market.get("auto_environment") or ())
    lines.append(f"Auto environment: {autos or UNKNOWN}. D1 label: {market.get('d1_label', UNKNOWN)}.")
    label = market.get("session_label") or {}
    lines.append(f"Session label: {label.get('label', UNKNOWN)} ({label.get('source', 'none')}).")
    trades = record.get("trades") or {}
    lines += ["", f"## Trades ({trades.get('n', 0)}; P&L known on {trades.get('n_pnl_known', 0)})"]
    for row in trades.get("rows") or ():
        lines.append(
            f"- {row.get('symbol')} {row.get('direction')} {row.get('status')}: "
            f"P&L {_fmt(row.get('net_pnl'))} {row.get('currency') or ''}, R {_fmt(row.get('r_multiple'))}, "
            f"origin {row.get('origin')}, exit {row.get('exit_reason')}, {row.get('time_bucket')}"
        )
    picks = record.get("picks") or {}
    pick_rows = list(picks.get("rows") or ())
    counts = ", ".join(f"{name} {n}" for name, n in (picks.get("counts") or {}).items())
    lines += ["", f"## Picks, likes, vetoes, claims ({picks.get('n', 0)}: {counts or 'none'})"]
    if len(pick_rows) > MARKDOWN_PICKS:
        lines.append(f"The {MARKDOWN_PICKS} of each population in record order; the JSON holds all.")
        shown: dict[str, int] = {}
        kept = []
        for row in pick_rows:
            population = str(row.get("population"))
            shown[population] = shown.get(population, 0) + 1
            if shown[population] <= MARKDOWN_PICKS:
                kept.append(row)
        pick_rows = kept
    for row in pick_rows:
        outcome = row.get("outcome") or {}
        lines.append(
            f"- {row.get('population')}: {row.get('symbol')} {row.get('side')} "
            f"{row.get('reason') or ''} -> ran {_fmt(outcome.get('ran_after_pct'), suffix='%')} (measured later)"
        )
    calls = record.get("calls") or {}
    lines += ["", f"## Market calls ({calls.get('n', 0)})"]
    for row in calls.get("rows") or ():
        lines.append(
            f"- {row.get('stamp')}: {row.get('direction')} {row.get('horizon')} -> "
            f"{(row.get('outcome') or {}).get('verdict') or UNKNOWN}"
        )
    lines += ["", f"Alerts reviewed: {(record.get('alerts') or {}).get('n', 0)}. "
              f"Notes: {(record.get('notes') or {}).get('n', 0)}. Moods: {(record.get('mood') or {}).get('n', 0)}. "
              f"AI ideas: {(record.get('ai_ideas') or {}).get('n', 0)}."]
    recap = record.get("recap") or {}
    lines += ["", "## Recap"]
    for row in recap.get("rows") or ():
        kind = row.get("kind")
        if kind == "lesson":
            lines.append(f"- Keep: {row.get('keep') or '-'} | Stop: {row.get('stop') or '-'} | Try: {row.get('try') or '-'}")
        elif kind == "rule":
            lines.append(f"- Rule for next session: {row.get('text')} ({row.get('tag') or 'no tag'})")
        elif kind == "rule_check":
            lines.append(f"- Kept the rule: {row.get('answer')}")
        elif kind == "environment_verdict":
            lines.append(f"- Environment verdict: {row.get('verdict')} (auto was {row.get('auto_label')})")
        elif kind == "clue":
            lines.append(f"- Clue {row.get('symbol')} {row.get('timeframe')} {row.get('bar_time')}: {row.get('clue_tag')}")
    for row in recap.get("rule_reflections") or ():
        lines.append(f"- Rule on {row.get('symbol') or row.get('trade_id')}: {row.get('answer')}")
    streak = recap.get("rule_streak")
    lines.append(f"- Rule streak: {streak if streak is not None else UNKNOWN}")
    lines += ["", f"Outcomes measured later, at {record.get('outcomes_measured_at')}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the week
# ---------------------------------------------------------------------------
def week_key(session_date: Any) -> str:
    year, week, _ = date.fromisoformat(_text(session_date)[:10]).isocalendar()
    return f"{year}-W{week:02d}"


def _group(rows: Iterable[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for key, trade in rows:
        group = groups.setdefault(key, {
            "key": key, "n": 0, "pnl_known_n": 0, "pnl_unknown_n": 0, "_pnl": 0.0,
            "wins": 0, "losses": 0, "r_n": 0, "_r": 0.0,
        })
        group["n"] += 1
        pnl = _number(trade.get("net_pnl_cad"))
        if pnl is None:
            group["pnl_unknown_n"] += 1
        else:
            group["pnl_known_n"] += 1
            group["_pnl"] += pnl
            if pnl > 0:
                group["wins"] += 1
            elif pnl < 0:
                group["losses"] += 1
        r_value = _number(trade.get("r_multiple"))
        if r_value is not None:
            group["r_n"] += 1
            group["_r"] += r_value
    out = []
    for group in groups.values():
        out.append({
            "key": group["key"],
            "n": group["n"],
            "pnl_known_n": group["pnl_known_n"],
            "pnl_unknown_n": group["pnl_unknown_n"],
            "pnl_cad": round(group["_pnl"], 6) if group["pnl_known_n"] else None,
            "wins": group["wins"],
            "losses": group["losses"],
            "r_n": group["r_n"],
            "avg_r": round(group["_r"] / group["r_n"], 4) if group["r_n"] else None,
            "too_few_to_tell": group["pnl_known_n"] < MIN_N,
        })
    out.sort(key=lambda row: (-row["n"], row["key"]))
    return out


def _normal(text: Any) -> str:
    return re.sub(r"\s+", " ", _text(text).lower())


def _recurrence(values: Iterable[str], key: str) -> list[dict[str, Any]]:
    counts = Counter(value for value in values if value)
    return [{key: value, "n": n} for value, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def build_week(week: str, records: Sequence[Mapping[str, Any]], *, built_at: datetime) -> dict[str, Any]:
    """Roll one ISO week's day records up. Pure; sample sizes on every row."""
    records = sorted(
        (record for record in records if record.get("schema") == SCHEMA),
        key=lambda record: _text(record.get("session_date")),
    )
    trades: list[tuple[Mapping[str, Any], str]] = []
    recap_rows: list[Mapping[str, Any]] = []
    for record in records:
        label = (record.get("market_context") or {}).get("session_label") or {}
        trader = label.get("source") in ("trader_corrected", "trader_agreed")
        for trade in (record.get("trades") or {}).get("rows") or ():
            environment = label.get("label") if trader else trade.get("auto_environment_at_open")
            trades.append((trade, _text(environment) or UNKNOWN))
        recap_rows.extend((record.get("recap") or {}).get("rows") or ())

    def _by(field: str) -> list[dict[str, Any]]:
        return _group((_text(trade.get(field)) or UNKNOWN, trade) for trade, _env in trades)

    checks = Counter(_text(row.get("answer")) for row in recap_rows if row.get("kind") == "rule_check")
    checked = sum(checks.values())
    lessons = [row for row in recap_rows if row.get("kind") == "lesson"]
    rules = [row for row in recap_rows if row.get("kind") == "rule"]
    body: dict[str, Any] = {
        "schema": WEEK_SCHEMA,
        "week": week,
        "sessions": [_text(record.get("session_date")) for record in records],
        "min_n": MIN_N,
        "trades_n": len(trades),
        "by_setup_family": _by("setup_family"),
        "by_grade": _by("grade"),
        "by_time_of_day": _by("time_bucket"),
        "by_origin": _by("origin"),
        "by_exit_reason": _by("exit_reason"),
        "by_environment": _group((env, trade) for trade, env in trades),
        "lesson_recurrence": {
            part: _recurrence((_normal(row.get(part)) for row in lessons), "text")
            for part in ("keep", "stop", "try")
        },
        "rule_recurrence": {
            "by_tag": _recurrence((_text(row.get("tag")) for row in rules), "tag"),
            "by_text": _recurrence((_normal(row.get("text")) for row in rules), "text"),
        },
        "rule_kept": {
            "n": checked,
            "yes": checks.get("yes", 0),
            "partly": checks.get("partly", 0),
            "no": checks.get("no", 0),
            "rate": round(checks.get("yes", 0) / checked, 4) if checked else None,
            "too_few_to_tell": checked < MIN_N,
        },
        "sources": [
            _source("day_session_record", record.get("content_hash")) | {"session": record.get("session_date")}
            for record in records
        ],
    }
    body["content_hash"] = _content_hash(body)
    body["built_at"] = built_at.isoformat()
    return body


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def records_dir(root: Path | None = None) -> Path:
    """`DAY_SESSION_RECORDS_DIR`, read at CALL time so a test can redirect it."""
    return Path(root) if root is not None else Path(project_paths.DAY_SESSION_RECORDS_DIR)


def record_path(session_date: str, *, root: Path | None = None) -> Path:
    session = _text(session_date)[:10]
    date.fromisoformat(session)
    return records_dir(root) / f"{session}.json"


def week_path(week: str, *, root: Path | None = None) -> Path:
    if not re.fullmatch(r"\d{4}-W\d{2}", _text(week)):
        raise ValueError(f"{week!r} is not YYYY-Www")
    return records_dir(root) / f"week-{week}.json"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError as exc:
            note_swallowed("day session record temp file not removed", exc, quiet=True)
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def read_record(session_date: str, *, root: Path | None = None) -> dict[str, Any] | None:
    payload = _read_json(record_path(session_date, root=root))
    return payload if payload and payload.get("schema") == SCHEMA else None


def write_record(record: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Write `<date>.json` and `<date>.md`. Unchanged content keeps its bytes;
    a degraded build never replaces an existing good record."""
    path = record_path(_text(record.get("session_date")), root=root)
    markdown_path = path.with_suffix(".md")
    prior = _read_json(path)
    if prior and prior.get("content_hash") == record.get("content_hash"):
        if not markdown_path.is_file():
            _atomic_text(markdown_path, render_markdown(prior))
        return {"path": str(path), "changed": False, "reason": "unchanged"}
    unread = (record.get("session_facts") or {}).get("sources_unread") or ()
    if unread and prior and prior.get("schema") == SCHEMA:
        return {
            "path": str(path), "changed": False,
            "reason": f"kept the prior record; this build could not read {', '.join(unread)}",
        }
    _atomic_text(path, json.dumps(dict(record), sort_keys=True, separators=(",", ":"), default=str) + "\n")
    _atomic_text(markdown_path, render_markdown(record))
    return {"path": str(path), "changed": True, "reason": "written"}


def write_week(week: str, *, root: Path | None = None, built_at: datetime) -> dict[str, Any] | None:
    """Rebuild one week's rollup from the day records on disk. None if it has none."""
    base = records_dir(root)
    records = []
    for path in sorted(base.glob("????-??-??.json")):
        try:
            if week_key(path.stem) != week:
                continue
        except ValueError:
            continue
        payload = _read_json(path)
        if payload and payload.get("schema") == SCHEMA:
            records.append(payload)
    if not records:
        return None
    body = build_week(week, records, built_at=built_at)
    target = week_path(week, root=root)
    prior = _read_json(target)
    if prior and prior.get("content_hash") == body["content_hash"]:
        return {"path": str(target), "changed": False}
    _atomic_text(target, json.dumps(body, indent=2, sort_keys=True, default=str) + "\n")
    return {"path": str(target), "changed": True}


# ---------------------------------------------------------------------------
# reading the stores (worker / night only)
# ---------------------------------------------------------------------------
def _row_session(row: Mapping[str, Any]) -> str:
    return _text(row.get("trade_date") or row.get("session_date") or row.get("ts"))[:10]


def collect_inputs(session: str, *, service: Any = None, now: datetime | None = None) -> dict[str, Any]:
    """Every store read the record needs, each in its own guard. Worker only.

    The Day Review read itself must succeed; any other unreadable store is
    named in `unread` and its section stays empty (unknown).
    """
    import recap_store

    if service is None:
        from ui.services.day_review_service import DayReviewService

        service = DayReviewService()
    payload = service.read_day(session, now=now)
    pack = service.compose_pack(session, payload, now=now)
    unread: list[str] = []
    inputs: dict[str, Any] = {"payload": payload, "pack": pack, "unread": unread}

    def _guard(name: str, reader, default):
        try:
            return reader()
        except Exception as exc:  # noqa: BLE001 - one store costs one section
            _log.debug("Day record: %s unreadable.", name, exc_info=True)
            unread.append(f"{name} ({exc})")
            return default

    inputs["recap"] = _guard("day recap events", lambda: recap_store.records_for(session), [])
    inputs["rule_for_session"] = _guard("recap rule", lambda: recap_store.latest_rule_before(session), None)
    inputs["rule_streak"] = _guard("recap rule streak", lambda: recap_store.rule_streak(session), None)

    def _reviews():
        import review_events

        return [row for row in review_events.load_review_events() if _row_session(row) == session]

    inputs["alert_reviews"] = _guard("alert review events", _reviews, [])

    def _env():
        from market_environment_annotations import load_market_environment_annotations

        return [row for row in load_market_environment_annotations() if _row_session(row) == session]

    inputs["env_annotations"] = _guard("market environment annotations", _env, [])

    def _grades_at_open():
        import setup_grades_history
        from recap_findability import session_open

        opened = session_open(session)
        return setup_grades_history.grades_as_of(opened) if opened else None

    def _opening_regime():
        import opening_regime_history

        return opening_regime_history.opening_regime_for(session)

    inputs["grades_at_open"] = _guard("grade history", _grades_at_open, None)
    inputs["opening_regime"] = _guard("opening regime history", _opening_regime, None)
    trade_ids = [
        _text(row.get("trade_id")) for row in payload.get("trades") or ()
        if isinstance(row, Mapping) and _text(row.get("trade_id"))
    ]

    def _legs():
        from ui.services import journal_feed

        return {trade_id: journal_feed.trade_legs(trade_id) for trade_id in trade_ids}

    def _mentor():
        if not trade_ids:
            return {}
        from journal_store import JournalStore

        store = JournalStore()
        return {
            trade_id: [
                row for row in store.list_opportunity_events(trade_id=trade_id, event_type="NOTE", limit=10000)
                if _text(row.get("source")) == "trade_mentor"
            ]
            for trade_id in trade_ids
        }

    inputs["trade_legs"] = _guard("trade legs", _legs, {})
    inputs["mentor_answers"] = _guard("mentor answers", _mentor, {})

    def _ideas():
        from ai_jobs import improvement_ideas

        state = improvement_ideas.read_state()
        rows = []
        for row in improvement_ideas.read_ideas():
            if _text(row.get("session_date"))[:10] != session:
                continue
            record = state.get(_text(row.get("idea_id"))) or {}
            rows.append({**dict(row), "status": _text(record.get("status")), "decided_at": record.get("decided_at")})
        return rows

    inputs["ideas"] = _guard("AI ideas", _ideas, [])
    inputs["read_errors"] = _text(payload.get("error"))
    return inputs


# ---------------------------------------------------------------------------
# rebuilding
# ---------------------------------------------------------------------------
def _aware_now(now: datetime | None) -> datetime:
    from market_session import normalize_market_local_datetime

    return normalize_market_local_datetime(now)


def _closed(session: str, moment: datetime) -> tuple[bool, str]:
    import market_calendar

    try:
        day = date.fromisoformat(session)
    except ValueError:
        return False, f"{session!r} is not YYYY-MM-DD"
    if not market_calendar.is_session(day):
        return False, f"{session} is not an exchange session"
    last = market_calendar.last_completed_session(moment)
    if day > last:
        return False, f"{session} is not closed yet; last closed is {last.isoformat()}"
    return True, ""


def rebuild(
    session_date: str,
    *,
    service: Any = None,
    now: datetime | None = None,
    root: Path | None = None,
    dry_run: bool = False,
    write_weeks: bool = True,
) -> dict[str, Any]:
    """Build one closed session's record and (unless dry-run) write it."""
    session = _text(session_date)[:10]
    moment = _aware_now(now)
    closed, reason = _closed(session, moment)
    if not closed:
        return {"status": "skipped", "session": session, "reason": reason, "written": False}
    inputs = collect_inputs(session, service=service, now=moment)
    record = build_record(session, inputs, built_at=moment)
    if dry_run:
        return {"status": "ok", "session": session, "written": False, "record": record,
                "markdown": render_markdown(record)}
    result = write_record(record, root=root)
    week = None
    if write_weeks:
        week = write_week(week_key(session), root=root, built_at=moment)
    return {"status": "ok", "session": session, "written": result["changed"], "reason": result["reason"],
            "path": result["path"], "record": record, "week": week}


def rebuild_recent(
    end_session: str,
    *,
    sessions: int = DEFAULT_REBUILD_SESSIONS,
    service: Any = None,
    now: datetime | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Rebuild the last `sessions` closed sessions ending at `end_session`, then
    their weeks. One failure costs that session and keeps its prior file."""
    import market_calendar

    moment = _aware_now(now)
    cursor = date.fromisoformat(_text(end_session)[:10])
    days: list[str] = []
    for _ in range(max(1, int(sessions))):
        days.append(cursor.isoformat())
        cursor = market_calendar.previous_session(cursor)
    out: dict[str, Any] = {"written": [], "unchanged": [], "skipped": [], "failed": [], "weeks": []}
    for day in sorted(days):
        try:
            result = rebuild(day, service=service, now=moment, root=root, write_weeks=False)
        except Exception as exc:  # noqa: BLE001 - the prior record stays
            out["failed"].append({"session": day, "reason": str(exc)})
            continue
        if result["status"] != "ok":
            out["skipped"].append({"session": day, "reason": result.get("reason", "")})
        elif result["written"]:
            out["written"].append(day)
        else:
            out["unchanged"].append(day)
    for week in sorted({week_key(day) for day in days}):
        try:
            written = write_week(week, root=root, built_at=moment)
        except Exception as exc:  # noqa: BLE001
            out["failed"].append({"session": f"week {week}", "reason": str(exc)})
            continue
        if written is not None:
            out["weeks"].append(week)
    return out


def summary_line(result: Mapping[str, Any]) -> str:
    record = result.get("record") or {}
    facts = record.get("session_facts") or {}
    counts = facts.get("counts") or {}
    label = (record.get("market_context") or {}).get("session_label") or {}
    parts = [
        f"{result.get('session')}:",
        f"trades {counts.get('trades', 0)}",
        f"alerts {counts.get('alerts', 0)}",
        f"picks {counts.get('picks', 0)}",
        f"calls {counts.get('calls', 0)}",
        f"notes {counts.get('notes', 0)}",
        f"ideas {counts.get('ai_ideas', 0)}",
        f"recap {counts.get('recap', 0)}",
        f"label {label.get('label', UNKNOWN)} ({label.get('source', 'none')})",
    ]
    if facts.get("sources_unread"):
        parts.append(f"unread: {'; '.join(facts['sources_unread'])}")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one session's Day Recap record.")
    parser.add_argument("--date", required=True, help="the closed session, YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="build and print; write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = rebuild(args.date, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - a CLI says why and exits non-zero
        print(f"failed: {exc}")
        return 1
    if result["status"] != "ok":
        print(f"skipped: {result.get('reason')}")
        return 2
    if args.dry_run:
        print(f"dry run, nothing written. {summary_line(result)}")
        print(result["markdown"])
    else:
        state = "written" if result["written"] else f"not rewritten ({result.get('reason')})"
        print(f"{state}: {result['path']}. {summary_line(result)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
