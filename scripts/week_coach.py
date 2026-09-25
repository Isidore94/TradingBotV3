"""Day Recap coach, the week view and "Ask the AI".

Reads the `day_session_record` week files (and the day records they name) and
turns them into: your edge, your leaks, repeats, a 4-week trend and a month
rollup. Holds the one writer for the trader's plain-words questions
(`WEEK_QUESTIONS_FILE`), the evidence retrieval and citation check the night's
answer job uses, and the compact frontier digest (`records/week-<W>-frontier.md`).

Every number carries its n. Under `MIN_N` a row says "too few to tell"; a
missing value is unknown, never zero. Evidence only: nothing here detects,
scores, ranks alerts, gates or alerts. Store reads are for workers / the night.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import day_session_record as dsr
from swallowed import note_swallowed

MIN_N = dsr.MIN_N
UNKNOWN = dsr.UNKNOWN
QUESTION_SCHEMA = "week_question_v1"
ANSWER_SCHEMA = "week_answer_v1"
QUESTION_MAX = 500
TOO_FEW = "too few to tell"
#: P8-P5: rows with n from here up to MIN_N - 1 are shown as "thin (n)", never ranked.
THIN_MIN_N = 5
UNCITED_NOTE = "uncited — not shown"

STATUS_PENDING = "pending"
STATUS_ANSWERED = "answered"
STATUS_NO_CITED = "no_cited_answer"

#: The week-file groups, in the order the page names them.
GROUPS: tuple[tuple[str, str], ...] = (
    ("by_setup_family", "Setup"),
    ("by_grade", "Grade"),
    ("by_time_of_day", "Time of day"),
    ("by_origin", "Idea from"),
    ("by_exit_reason", "Exit reason"),
    ("by_environment", "Market"),
)
TREND_WEEKS = 4

#: Plain words in a question -> the week-file groups worth sending the model.
FIELD_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("by_time_of_day", ("morning", "afternoon", "open", "close", "midday", "lunch", "premarket", "time", "hour", "late", "early")),
    ("by_setup_family", ("setup", "family", "pattern", "bounce", "avwap", "breakout", "pullback")),
    ("by_grade", ("grade", "a+", "quality")),
    ("by_environment", ("market", "environment", "regime", "chop", "trend", "bear", "bull")),
    ("by_exit_reason", ("exit", "stop", "target", "sell", "cut")),
    ("by_origin", ("origin", "idea", "alert", "scan", "own", "mentor")),
)
RULE_WORDS = ("rule", "kept", "keep", "discipline", "lesson", "habit", "repeat")
CALL_WORDS = ("call", "read", "right", "wrong", "predict", "forecast")
MAX_CONTEXT_TRADES = 80
MAX_CONTEXT_CALLS = 40


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    return dsr._number(value)


def week_monday(week: str) -> date:
    year, number = _text(week).split("-W")
    return date.fromisocalendar(int(year), int(number), 1)


def shift_week(week: str, weeks: int) -> str:
    return dsr.week_key(week_monday(week) + timedelta(days=7 * int(weeks)))


def month_of(week: str) -> str:
    """The calendar month an ISO week belongs to (by its Thursday)."""
    return (week_monday(week) + timedelta(days=3)).strftime("%Y-%m")


def list_weeks(root: Path | None = None) -> list[str]:
    base = dsr.records_dir(root)
    out = []
    for path in base.glob("week-????-W??.json"):
        out.append(path.stem[len("week-"):])
    return sorted(out)


def load_week(week: str, root: Path | None = None) -> dict[str, Any] | None:
    payload = dsr._read_json(dsr.week_path(week, root=root))
    return payload if payload and payload.get("schema") == dsr.WEEK_SCHEMA else None


def load_records(sessions: Iterable[str], root: Path | None = None) -> list[dict[str, Any]]:
    out = []
    for session in sessions:
        try:
            record = dsr.read_record(_text(session), root=root)
        except ValueError:
            continue
        if record:
            out.append(record)
    return out


def group_id(week: str, group: str, key: str) -> str:
    return f"week:{week}:{group}:{key}"


# ---------------------------------------------------------------------------
# edge, leaks, month
# ---------------------------------------------------------------------------
def _rank_rows(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every week-file row with n >= MIN_N and a known metric. R wins over P&L."""
    week = _text(body.get("week"))
    rows = []
    for group, label in GROUPS:
        for row in body.get(group) or ():
            key = _text(row.get("key")) or UNKNOWN
            if key == UNKNOWN:
                continue
            n_known = int(row.get("pnl_known_n") or 0)
            if n_known < MIN_N:
                continue
            r_n = int(row.get("r_n") or 0)
            avg_r = _number(row.get("avg_r"))
            pnl = _number(row.get("pnl_cad"))
            if avg_r is not None and r_n >= MIN_N:
                metric, value, tier = "avg_r", avg_r, 0
            elif pnl is not None:
                metric, value, tier = "pnl_cad", pnl, 1
            else:
                continue
            rows.append({
                "group": group, "label": label, "key": key,
                "n": int(row.get("n") or 0), "pnl_known_n": n_known, "r_n": r_n,
                "wins": int(row.get("wins") or 0), "losses": int(row.get("losses") or 0),
                "avg_r": avg_r, "pnl_cad": pnl, "metric": metric, "value": value, "tier": tier,
                "id": group_id(week or "month", group, key),
            })
    return rows


def edge_and_leaks(body: Mapping[str, Any], *, k: int = 3) -> dict[str, Any]:
    """Top k (edge) and bottom k (leaks). An edge is positive, a leak negative."""
    ranked = _rank_rows(body)
    edge = sorted((row for row in ranked if row["value"] > 0), key=lambda row: (row["tier"], -row["value"], row["id"]))
    leaks = sorted((row for row in ranked if row["value"] < 0), key=lambda row: (row["tier"], row["value"], row["id"]))
    thin = sum(
        1 for group, _label in GROUPS for row in body.get(group) or ()
        if int(row.get("pnl_known_n") or 0) < MIN_N
    )
    return {"edge": edge[:k], "leaks": leaks[:k], "thin_rows": thin, "min_n": MIN_N}


def thin_rows(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rows with n THIN_MIN_N..MIN_N-1 and a known P&L, biggest first. Shown, never ranked."""
    rows = []
    for group, label in GROUPS:
        for row in body.get(group) or ():
            key = _text(row.get("key")) or UNKNOWN
            n_known = int(row.get("pnl_known_n") or 0)
            pnl = _number(row.get("pnl_cad"))
            if key == UNKNOWN or pnl is None or not THIN_MIN_N <= n_known < MIN_N:
                continue
            rows.append({
                "group": group, "label": label, "key": key, "pnl_known_n": n_known,
                "wins": int(row.get("wins") or 0), "losses": int(row.get("losses") or 0),
                "pnl_cad": pnl, "value": pnl,
            })
    rows.sort(key=lambda row: (-abs(row["value"]), row["group"], row["key"]))
    return rows


def thin_line(row: Mapping[str, Any]) -> str:
    words = str(row["key"]).replace("_", " ")
    return (
        f"{row['label']} {words}: {fmt_money(row.get('pnl_cad'))}, thin ({row['pnl_known_n']}), "
        f"{row['wins']} won / {row['losses']} lost"
    )


def rollup_view(weeks: Sequence[str], root: Path | None = None) -> dict[str, Any]:
    """Edge, leaks and thin rows over several week files, merged group by group. Worker only."""
    bodies = [body for body in (load_week(key, root) for key in weeks) if body]
    if not bodies:
        return {"weeks": [], "edge": [], "leaks": [], "thin": [], "thin_rows": 0}
    body = month_rollup(bodies)
    return {"weeks": list(body.get("weeks") or ()), **edge_and_leaks(body), "thin": thin_rows(body)}


def _merge_groups(bodies: Sequence[Mapping[str, Any]], group: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for body in bodies:
        for row in body.get(group) or ():
            key = _text(row.get("key")) or UNKNOWN
            out = merged.setdefault(key, {
                "key": key, "n": 0, "pnl_known_n": 0, "pnl_unknown_n": 0, "_pnl": 0.0,
                "wins": 0, "losses": 0, "r_n": 0, "_r": 0.0,
            })
            for name in ("n", "pnl_known_n", "pnl_unknown_n", "wins", "losses", "r_n"):
                out[name] += int(row.get(name) or 0)
            pnl = _number(row.get("pnl_cad"))
            if pnl is not None:
                out["_pnl"] += pnl
            avg_r = _number(row.get("avg_r"))
            if avg_r is not None:
                out["_r"] += avg_r * int(row.get("r_n") or 0)
    rows = []
    for out in merged.values():
        rows.append({
            "key": out["key"], "n": out["n"], "pnl_known_n": out["pnl_known_n"],
            "pnl_unknown_n": out["pnl_unknown_n"],
            "pnl_cad": round(out["_pnl"], 6) if out["pnl_known_n"] else None,
            "wins": out["wins"], "losses": out["losses"], "r_n": out["r_n"],
            "avg_r": round(out["_r"] / out["r_n"], 4) if out["r_n"] else None,
            "too_few_to_tell": out["pnl_known_n"] < MIN_N,
        })
    rows.sort(key=lambda row: (-row["n"], row["key"]))
    return rows


def _merge_counts(lists: Iterable[Iterable[Mapping[str, Any]]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for rows in lists:
        for row in rows or ():
            value = _text(row.get(key))
            if value:
                counts[value] = counts.get(value, 0) + int(row.get("n") or 0)
    return [{key: value, "n": n} for value, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def month_rollup(bodies: Sequence[Mapping[str, Any]], *, month: str = "") -> dict[str, Any]:
    """Several week files as one body in the week-file shape. Pure."""
    bodies = sorted((body for body in bodies if body), key=lambda body: _text(body.get("week")))
    kept = {"n": 0, "yes": 0, "partly": 0, "no": 0}
    for body in bodies:
        for name in kept:
            kept[name] += int((body.get("rule_kept") or {}).get(name) or 0)
    out: dict[str, Any] = {
        "schema": dsr.WEEK_SCHEMA,
        "week": "",
        "month": month,
        "weeks": [_text(body.get("week")) for body in bodies],
        "sessions": [session for body in bodies for session in body.get("sessions") or ()],
        "min_n": MIN_N,
        "trades_n": sum(int(body.get("trades_n") or 0) for body in bodies),
        "lesson_recurrence": {
            part: _merge_counts(((body.get("lesson_recurrence") or {}).get(part) or () for body in bodies), "text")
            for part in ("keep", "stop", "try")
        },
        "rule_recurrence": {
            "by_tag": _merge_counts(((body.get("rule_recurrence") or {}).get("by_tag") or () for body in bodies), "tag"),
            "by_text": _merge_counts(((body.get("rule_recurrence") or {}).get("by_text") or () for body in bodies), "text"),
        },
        "rule_kept": {
            **kept,
            "rate": round(kept["yes"] / kept["n"], 4) if kept["n"] else None,
            "too_few_to_tell": kept["n"] < MIN_N,
        },
        "sources": [source for body in bodies for source in body.get("sources") or ()],
    }
    for group, _label in GROUPS:
        out[group] = _merge_groups(bodies, group)
    return out


def month_weeks(week: str, available: Sequence[str]) -> list[str]:
    """The week files in the chosen week's calendar month (4 or 5 of them)."""
    target = month_of(week)
    return [item for item in available if month_of(item) == target]


# ---------------------------------------------------------------------------
# repeats, trend
# ---------------------------------------------------------------------------
def _recap_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for record in records for row in (record.get("recap") or {}).get("rows") or ()]


def repeats(body: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Lesson stop/keep items said 2+ times, and each rule with its kept rate."""
    lessons = []
    for part in ("stop", "keep"):
        for row in (body.get("lesson_recurrence") or {}).get(part) or ():
            if int(row.get("n") or 0) >= 2:
                lessons.append({"part": part, "text": _text(row.get("text")), "n": int(row["n"])})
    recap = _recap_rows(records)
    rule_text = {_text(row.get("id")): dsr._normal(row.get("text")) for row in recap if row.get("kind") == "rule"}
    checks: dict[str, dict[str, int]] = {}
    for row in recap:
        if row.get("kind") != "rule_check":
            continue
        text = rule_text.get(_text(row.get("rule_id"))) or dsr._normal(row.get("text"))
        if not text:
            continue
        cell = checks.setdefault(text, {"yes": 0, "partly": 0, "no": 0})
        answer = _text(row.get("answer"))
        if answer in cell:
            cell[answer] += 1
    written = {_text(row.get("text")): int(row.get("n") or 0) for row in (body.get("rule_recurrence") or {}).get("by_text") or ()}
    rules = []
    for text in sorted(set(written) | set(checks)):
        cell = checks.get(text) or {"yes": 0, "partly": 0, "no": 0}
        checked = sum(cell.values())
        rules.append({
            "text": text, "written_n": written.get(text, 0), "checked_n": checked, **cell,
            "rate": round(cell["yes"] / checked, 4) if checked else None,
            "too_few_to_tell": checked < MIN_N,
        })
    rules.sort(key=lambda row: (-row["checked_n"], -row["written_n"], row["text"]))
    return {"lessons": lessons, "rules": rules}


def calls_right(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Right / (right + wrong + flat) over finished calls. Pending is not a miss."""
    counts = {"right": 0, "wrong": 0, "flat": 0, "open": 0}
    for record in records:
        for row in (record.get("calls") or {}).get("rows") or ():
            verdict = _text((row.get("outcome") or {}).get("verdict")).lower()
            counts[verdict if verdict in ("right", "wrong", "flat") else "open"] += 1
    finished = counts["right"] + counts["wrong"] + counts["flat"]
    return {
        **counts, "n": finished,
        "rate": round(counts["right"] / finished, 4) if finished else None,
        "too_few_to_tell": finished < MIN_N,
    }


def week_pnl(body: Mapping[str, Any]) -> dict[str, Any]:
    rows = body.get("by_setup_family") or ()
    known = [row for row in rows if int(row.get("pnl_known_n") or 0)]
    return {
        "pnl_cad": round(sum(_number(row.get("pnl_cad")) or 0.0 for row in known), 2) if known else None,
        "pnl_known_n": sum(int(row.get("pnl_known_n") or 0) for row in rows),
        "trades_n": int(body.get("trades_n") or 0),
    }


def trend(week: str, root: Path | None = None, *, weeks: int = TREND_WEEKS) -> list[dict[str, Any]]:
    out = []
    for offset in range(weeks - 1, -1, -1):
        key = shift_week(week, -offset)
        body = load_week(key, root)
        if body is None:
            out.append({"week": key, "recorded": False})
            continue
        records = load_records(body.get("sessions") or (), root)
        kept = body.get("rule_kept") or {}
        out.append({
            "week": key, "recorded": True, **week_pnl(body),
            "rule_kept_rate": _number(kept.get("rate")), "rule_checks_n": int(kept.get("n") or 0),
            "calls": calls_right(records),
        })
    return out


# ---------------------------------------------------------------------------
# questions and answers (append-only)
# ---------------------------------------------------------------------------
class QuestionError(ValueError):
    """The question is invalid. Nothing was written."""


def _questions_path(path: Path | None) -> Path:
    import project_paths

    return Path(path) if path is not None else Path(project_paths.WEEK_QUESTIONS_FILE)


def _answers_path(path: Path | None) -> Path:
    import project_paths

    return Path(path) if path is not None else Path(project_paths.WEEK_ANSWERS_FILE)


def append_row(row: Mapping[str, Any], path: Path) -> None:
    """One JSON line, under the local writer lock, fsynced. Raises on failure."""
    from local_writer_lock import local_writer_lock, lock_key_for_path

    encoded = (json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with local_writer_lock(lock_key_for_path(path), timeout_seconds=2.0):
        torn = path.is_file() and path.stat().st_size > 0 and path.read_bytes()[-1:] != b"\n"
        with path.open("ab") as handle:
            handle.write((b"\n" if torn else b"") + encoded)
            handle.flush()
            os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def record_question(
    text: Any, *, week: str = "", now: datetime | None = None, path: Path | None = None,
) -> dict[str, Any]:
    """Save one plain-words question as pending. Worker only; raises if not saved."""
    words = re.sub(r"\s+", " ", _text(text))
    if not words:
        raise QuestionError("type a question first")
    if len(words) > QUESTION_MAX:
        raise QuestionError(f"a question is at most {QUESTION_MAX} characters")
    moment = now or datetime.now().astimezone()
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise QuestionError("asked_at must carry a timezone")
    scope = _text(week)
    if scope and not re.fullmatch(r"\d{4}-W\d{2}", scope):
        raise QuestionError(f"{week!r} is not YYYY-Www")
    row = {
        "schema": QUESTION_SCHEMA,
        "id": "q-" + uuid.uuid4().hex[:12],
        "asked_at": moment.isoformat(),
        "text": words,
        "status": STATUS_PENDING,
        "week": scope,
    }
    append_row(row, _questions_path(path))
    return row


def read_questions(
    *, questions_path: Path | None = None, answers_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Every question, newest first, with its latest answer folded in."""
    answers: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(_answers_path(answers_path)):
        if row.get("schema") == ANSWER_SCHEMA and _text(row.get("question_id")):
            answers[_text(row["question_id"])] = row
    out = []
    for row in _read_jsonl(_questions_path(questions_path)):
        if row.get("schema") != QUESTION_SCHEMA or not _text(row.get("id")):
            continue
        answer = answers.get(_text(row["id"]))
        out.append({**row, "status": _text(answer.get("status")) if answer else STATUS_PENDING, "answer": answer})
    out.sort(key=lambda row: _text(row.get("asked_at")), reverse=True)
    return out


def pending_questions(**paths: Any) -> list[dict[str, Any]]:
    rows = [row for row in read_questions(**paths) if row["status"] == STATUS_PENDING]
    return sorted(rows, key=lambda row: _text(row.get("asked_at")))


# ---------------------------------------------------------------------------
# retrieval and citations (the night job's pure half)
# ---------------------------------------------------------------------------
def question_fields(text: str) -> dict[str, Any]:
    words = _text(text).lower()
    groups = [group for group, keys in FIELD_WORDS if any(key in words for key in keys)]
    return {
        "groups": groups or [group for group, _label in GROUPS],
        "rules": any(key in words for key in RULE_WORDS),
        "calls": any(key in words for key in CALL_WORDS),
    }


def question_range(question: Mapping[str, Any], available: Sequence[str]) -> list[str]:
    """Weeks to read: those named by a date in the question, else its week and the 3 before."""
    named = sorted({dsr.week_key(day) for day in re.findall(r"\d{4}-\d{2}-\d{2}", _text(question.get("text"))) if _valid_day(day)})
    if named:
        return named
    end = _text(question.get("week")) or (available[-1] if available else "")
    if not end:
        return []
    return [shift_week(end, -offset) for offset in range(TREND_WEEKS - 1, -1, -1)]


def _valid_day(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


_TRADE_KEYS = ("symbol", "direction", "net_pnl_cad", "r_multiple", "setup_family", "grade", "time_bucket", "origin", "exit_reason", "auto_environment_at_open")


def build_evidence(question: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Only the record excerpts this question needs, each with a citable id."""
    fields = question_fields(_text(question.get("text")))
    weeks = question_range(question, list_weeks(root))
    allowed: dict[str, dict[str, Any]] = {}
    excerpts: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for week in weeks:
        body = load_week(week, root)
        if body is None:
            continue
        allowed[f"week:{week}"] = {"session": "", "n": int(body.get("trades_n") or 0)}
        excerpt: dict[str, Any] = {"id": f"week:{week}", "sessions": list(body.get("sessions") or ()), "trades_n": body.get("trades_n")}
        for group in fields["groups"]:
            rows = []
            for row in body.get(group) or ():
                ident = group_id(week, group, _text(row.get("key")) or UNKNOWN)
                allowed[ident] = {"session": "", "n": int(row.get("pnl_known_n") or 0)}
                rows.append({"id": ident, **{name: row.get(name) for name in ("key", "n", "pnl_known_n", "pnl_cad", "wins", "losses", "r_n", "avg_r", "too_few_to_tell")}})
            excerpt[group] = rows
        if fields["rules"]:
            excerpt["rule_kept"] = body.get("rule_kept")
            excerpt["rule_recurrence"] = (body.get("rule_recurrence") or {}).get("by_text")
            excerpt["lesson_recurrence"] = body.get("lesson_recurrence")
        excerpts.append(excerpt)
        for record in load_records(body.get("sessions") or (), root):
            session = _text(record.get("session_date"))
            allowed[f"session:{session}"] = {"session": session, "n": int((record.get("trades") or {}).get("n") or 0)}
            for trade in (record.get("trades") or {}).get("rows") or ():
                ident = f"trade:{_text(trade.get('trade_id'))}"
                allowed[ident] = {"session": session, "n": 1}
                trades.append({"id": ident, "session": session, **{name: trade.get(name) for name in _TRADE_KEYS}})
            if fields["calls"]:
                for read in (record.get("calls") or {}).get("rows") or ():
                    ident = f"read:{_text((read.get('source') or {}).get('id'))}"
                    allowed[ident] = {"session": session, "n": 1}
                    other.append({"id": ident, "session": session, "verdict": (read.get("outcome") or {}).get("verdict")})
            if fields["rules"]:
                for row in (record.get("recap") or {}).get("rows") or ():
                    if row.get("kind") in ("lesson", "rule", "rule_check"):
                        ident = f"recap:{_text(row.get('id'))}"
                        allowed[ident] = {"session": session, "n": 1}
                        other.append({"id": ident, "session": session, "kind": row.get("kind"), **{k: row.get(k) for k in ("keep", "stop", "text", "answer") if row.get(k)}})
    return {
        "question": _text(question.get("text")),
        "weeks": weeks,
        "min_n": MIN_N,
        "instructions": (
            "Answer the trader's question from these records only. Split the answer into short "
            "claims. Every claim must cite one or more ids copied exactly from allowed_source_ids. "
            f"A group with n under {MIN_N} is too few to tell: say so. Unknown is not zero. "
            "Never invent a number the records do not hold. If the records cannot answer, say so in one claim citing the weeks read."
        ),
        "allowed_source_ids": sorted(allowed),
        "weeks_read": excerpts,
        "trades": trades[:MAX_CONTEXT_TRADES],
        "trades_total": len(trades),
        "other": other[:MAX_CONTEXT_CALLS],
        "_allowed": allowed,
    }


def validate_claims(claims: Any, allowed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Keep a claim only if it cites at least one id that exists. Flag thin cites."""
    shown: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for claim in claims if isinstance(claims, list) else ():
        if not isinstance(claim, Mapping):
            continue
        text = _text(claim.get("text"))
        cited = [_text(item) for item in claim.get("citations") or () if _text(item)]
        valid = [item for item in dict.fromkeys(cited) if item in allowed]
        if not text or not valid:
            dropped.append({"text": text, "citations": cited, "note": UNCITED_NOTE})
            continue
        thin = [
            item for item in valid
            if item.startswith("week:") and item.count(":") >= 3 and int(allowed[item].get("n") or 0) < MIN_N
        ]
        shown.append({
            "text": text,
            "citations": [{"id": item, "session": _text(allowed[item].get("session"))} for item in valid],
            "flag": f"{TOO_FEW} (n under {MIN_N})" if thin else "",
        })
    return {"shown": shown, "dropped": dropped}


def answer_row(
    question: Mapping[str, Any], checked: Mapping[str, Any], *, model: str, answered_at: datetime,
    trades_total: int,
) -> dict[str, Any]:
    shown = list(checked.get("shown") or ())
    citations = []
    for claim in shown:
        for cite in claim["citations"]:
            if cite not in citations:
                citations.append(cite)
    return {
        "schema": ANSWER_SCHEMA,
        "question_id": _text(question.get("id")),
        "answered_at": answered_at.isoformat(),
        "model": model,
        "text": "\n".join(claim["text"] for claim in shown),
        "claims": shown,
        "citations": citations,
        "dropped_n": len(checked.get("dropped") or ()),
        "small_sample": trades_total < MIN_N,
        "trades_read": trades_total,
        "status": STATUS_ANSWERED if shown else STATUS_NO_CITED,
    }


# ---------------------------------------------------------------------------
# P8-P5: the journal's truth lines for the week and the 4-week rollup
# ---------------------------------------------------------------------------
def _load_journal_trades() -> list[dict[str, Any]]:
    from journal_store import JournalStore

    return list(JournalStore().list_trades())


def truth_view(
    weeks: Sequence[str], rollup_weeks: Sequence[str], *, trades_loader=None, grades_at=None,
) -> dict[str, Any]:
    """Stocks/options, longs/shorts and confirmed setups (CAD) for the chosen
    weeks and for the 4-week rollup, and the D-or-worse setups traded in the
    chosen weeks (bot grade as of each entry). Reads files: worker only."""
    import journal_truth

    try:
        trades = list((trades_loader or _load_journal_trades)())
    except Exception as exc:  # noqa: BLE001 - unread is unknown, never zero
        return {"error": f"the journal could not be read: {exc}"}

    def span(keys: Sequence[str]) -> list[dict[str, Any]]:
        keys = [key for key in keys if key]
        if not keys:
            return []
        first = week_monday(min(keys))
        last = week_monday(max(keys)) + timedelta(days=6)
        return journal_truth.in_window(trades, first, last)

    chosen, rollup = span(weeks), span(rollup_weeks)
    grades = journal_truth.bot_grades(chosen, grades_at or journal_truth.grade_reader())
    return {
        "weeks": list(weeks),
        "rollup_weeks": list(rollup_weeks),
        "lines": journal_truth.cad_truth_lines(chosen),
        "rollup_lines": journal_truth.cad_truth_lines(rollup),
        "worst_line": journal_truth.worst_setups_line(
            chosen, grades, span="this month" if len(weeks) > 1 else "this week"
        ),
    }


def rollup_weeks_for(week: str, *, weeks: int = TREND_WEEKS) -> list[str]:
    """The chosen week and the ones before it, oldest first."""
    return [shift_week(week, -offset) for offset in range(weeks - 1, -1, -1)]


# ---------------------------------------------------------------------------
# the page's one read
# ---------------------------------------------------------------------------
def read_view(
    week: str = "", *, month: bool = False, root: Path | None = None,
    questions_path: Path | None = None, answers_path: Path | None = None,
    trades_loader=None, grades_at=None,
) -> dict[str, Any]:
    """Everything the Week Review coach section shows. Worker only."""
    available = list_weeks(root)
    chosen = _text(week) or (available[-1] if available else dsr.week_key(date.today()))
    covered = month_weeks(chosen, available) if month else ([chosen] if chosen in available else [])
    bodies = [body for body in (load_week(item, root) for item in covered) if body]
    if month:
        body = month_rollup(bodies, month=month_of(chosen))
    else:
        body = bodies[0] if bodies else {}
    records = load_records(body.get("sessions") or (), root)
    return {
        "week": chosen,
        "month": month_of(chosen) if month else "",
        "available": available,
        "covered_weeks": [_text(item.get("week")) for item in bodies],
        "recorded": bool(bodies),
        "sessions": list(body.get("sessions") or ()),
        "trades_n": int(body.get("trades_n") or 0),
        **edge_and_leaks(body),
        "thin": thin_rows(body),
        # P8-P5: the same builders over the last 4 week files, so cells can reach n 10.
        "rollup": {} if month else rollup_view(rollup_weeks_for(chosen), root),
        "repeats": repeats(body, records),
        "rule_kept": dict(body.get("rule_kept") or {}),
        "calls": calls_right(records),
        # Month view: the trend ends at the month's last recorded week.
        "trend": trend(covered[-1] if month and covered else chosen, root),
        "questions": read_questions(questions_path=questions_path, answers_path=answers_path),
        "truth": truth_view(
            covered if month else [chosen],
            rollup_weeks_for(covered[-1] if month and covered else chosen),
            trades_loader=trades_loader,
            grades_at=grades_at,
        ),
    }


# ---------------------------------------------------------------------------
# plain words
# ---------------------------------------------------------------------------
def fmt_money(value: Any) -> str:
    number = _number(value)
    return UNKNOWN if number is None else f"{'+' if number >= 0 else '-'}${abs(number):,.0f}"


def fmt_rate(value: Any, n: int) -> str:
    number = _number(value)
    if number is None or not n:
        return f"{UNKNOWN} (n 0)"
    text = f"{number * 100:.0f}% (n {n})"
    return text + f", {TOO_FEW}" if n < MIN_N else text


def row_line(row: Mapping[str, Any]) -> str:
    if row.get("metric") == "avg_r":
        value = f"avg {row['avg_r']:+.2f}R (n {row['r_n']})"
    else:
        value = f"{fmt_money(row.get('pnl_cad'))} (n {row['pnl_known_n']})"
    words = str(row["key"]).replace("_", " ")  # plain words for ids like bullish_strong
    return f"{row['label']} {words}: {value}, {row['wins']} won / {row['losses']} lost"


# ---------------------------------------------------------------------------
# the frontier digest
# ---------------------------------------------------------------------------
def frontier_path(week: str, root: Path | None = None) -> Path:
    dsr.week_path(week, root=root)  # validates the key
    return dsr.records_dir(root) / f"week-{week}-frontier.md"


def render_frontier(view: Mapping[str, Any], week_body: Mapping[str, Any] | None) -> str:
    """A compact digest for a frontier model. Written only from the records."""
    week = _text(view.get("week"))
    lines = [
        f"# Week {week} - trading digest",
        "",
        "Source: day_session_record week and day files. Ids in [brackets] are citable. "
        f"Rows under n {MIN_N} say '{TOO_FEW}'. 'unknown' is not zero. Evidence only.",
        "",
        f"Sessions: {', '.join(f'[session:{s}]' for s in view.get('sessions') or ()) or 'none recorded'}",
        f"Trades: {view.get('trades_n', 0)}  [week:{week}]",
    ]
    if week_body:
        lines.append(f"Week record hash: {_text(week_body.get('content_hash'))[:16]}")
    for title, key in (("Edge", "edge"), ("Leaks", "leaks")):
        lines += ["", f"## {title}"]
        rows = view.get(key) or ()
        lines += [f"- {row_line(row)} [{row['id']}]" for row in rows] or [f"- none with n >= {MIN_N}"]
    lines.append(f"- {view.get('thin_rows', 0)} other row(s) are {TOO_FEW}.")
    rep = view.get("repeats") or {}
    lines += ["", "## Repeats"]
    lines += [f"- {item['part']}: \"{item['text']}\" said {item['n']}x" for item in rep.get("lessons") or ()] or ["- no lesson said twice"]
    lines += ["", "## Rules"]
    for rule in rep.get("rules") or ():
        lines.append(f"- \"{rule['text']}\" written {rule['written_n']}x, kept {fmt_rate(rule['rate'], rule['checked_n'])}")
    if not rep.get("rules"):
        lines.append("- no rule written or checked")
    kept = view.get("rule_kept") or {}
    calls = view.get("calls") or {}
    lines += [
        f"- all rules kept: {fmt_rate(kept.get('rate'), int(kept.get('n') or 0))}",
        f"- calls right: {fmt_rate(calls.get('rate'), int(calls.get('n') or 0))}; still open {calls.get('open', 0)}",
        "", "## 4-week trend",
    ]
    for row in view.get("trend") or ():
        if not row.get("recorded"):
            lines.append(f"- {row['week']}: no record")
            continue
        call = row.get("calls") or {}
        lines.append(
            f"- {row['week']}: P&L {fmt_money(row.get('pnl_cad'))} (n {row.get('pnl_known_n', 0)}), "
            f"rules kept {fmt_rate(row.get('rule_kept_rate'), row.get('rule_checks_n', 0))}, "
            f"calls right {fmt_rate(call.get('rate'), int(call.get('n') or 0))}"
        )
    lines += ["", "## Questions"]
    questions = [q for q in view.get("questions") or () if _text(q.get("week")) in ("", week) or q["status"] == STATUS_PENDING]
    for question in questions:
        lines.append(f"- [{question['id']}] {question['text']} ({question['status']})")
        answer = question.get("answer") or {}
        for claim in answer.get("claims") or ():
            cites = " ".join(f"[{cite['id']}]" for cite in claim.get("citations") or ())
            flag = f" ({claim['flag']})" if claim.get("flag") else ""
            lines.append(f"  - {claim['text']}{flag} {cites}")
    if not questions:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_frontier(
    week: str, *, root: Path | None = None, questions_path: Path | None = None,
    answers_path: Path | None = None,
) -> dict[str, Any] | None:
    """Write `week-<W>-frontier.md`. None when the week has no record. Same bytes skip."""
    body = load_week(week, root)
    if body is None:
        return None
    view = read_view(week, root=root, questions_path=questions_path, answers_path=answers_path)
    text = render_frontier(view, body)
    target = frontier_path(week, root)
    try:
        if target.read_text(encoding="utf-8") == text:
            return {"path": str(target), "changed": False}
    except OSError as exc:
        note_swallowed("week frontier file unreadable; rewriting it", exc, quiet=True)
    dsr._atomic_text(target, text)
    return {"path": str(target), "changed": True}
