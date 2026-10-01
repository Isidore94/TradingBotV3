"""`mentor_review` - the night reads the Trade Mentor app's day (mentor app Phase 4).

Deterministic half (always, also with ``ask=False``, a budget cut or the model down):
open ``MENTOR_CHAT_DB_FILE`` read-only, grade open challenges with
``mentor_app.challenge.grade_open`` (model-free; only 22:00-06:00 PT, when the night
owns grading), count the day's facts and publish them as ``mentor_day_facts`` to the
ai_store. The counts also ride on the ledger row.

Model half: one call (<= 600 output tokens) over an inputs pack with ids - the facts,
the day's profile notes, the day's challenges and up to 20 recent turns, plus (P15a) the
night's reads: the day's pick assessments, gate verdicts of the last 10 sessions with their
grades, the day's tilt observations, the mirror's top cuts, the daily digest's headline
values, the contrasts, the day review's verdicts and the improvement ideas. The citation
check is ``plan_review``'s rule: an id the pack does not carry rejects the reply whole;
an uncited item is dropped. Publishes ``mentor_day_digest`` (<= 5 items, <= 3 open
questions) by temp-and-rename, so a failed publish keeps the last good one.

Coach brief (P15a): the night's product for the day coach, ``mentor_coach_brief_<session>``.
The deterministic half finds recurring-issue candidates over the last 10 sessions (stable
``key``, ``first_seen`` carried from earlier briefs) and publishes the facts part. Only after
the digest succeeded and with time left in the slot's reserve, a second call (<= 500 tokens)
words <= 4 things to watch, <= 3 he may be missing, ranks and words the issues and writes
one line; ``check_brief`` rejects a foreign id and drops uncited items. Never a rule.

Hypotheses (P11): the reply may carry <= 3 ``hypotheses``, each a query into the shadow
permutation grid (the pack shows the newest report's vocabulary). The deterministic half
looks each one up (``mentor_packs.hypothesis_pack.find``: no new compute), records it as a
``hypothesis`` challenge (22:00-06:00 PT only, the night's write window) and publishes the
cell numbers or the miss reason in the digest as ``hyp:*`` lines. Nothing is applied.

The chat DB writes are the grading columns (``outcome_json``, ``graded_utc``) and the new
``hypothesis`` challenge rows, through :class:`NightChallengeStore`; digests are never
written back into the app's tables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time as clock
from contextlib import closing
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ai_jobs import ledger

_log = logging.getLogger(__name__)

PT = ZoneInfo("America/Los_Angeles")
PROMPT_VERSION = "mentor_review_v3"
SCHEMA_NAME = "tradingbot_mentor_review"
FACTS_STEM = "mentor_day_facts"
DIGEST_STEM = "mentor_day_digest"
FACTS_SCHEMA = "mentor_day_facts_v1"
DIGEST_SCHEMA = "mentor_day_digest_v1"
#: P15a: the night's product for the day coach, loaded first into the morning memory.
COACH_STEM = "mentor_coach_brief"
COACH_SCHEMA = "mentor_coach_brief_v1"
#: Every recurring-issue candidate by stable key (first_seen survives an issue left out of the brief).
REGISTRY_FILE = "mentor_issue_registry.json"
REGISTRY_SCHEMA = "mentor_issue_registry_v1"
REGISTRY_SESSIONS = 60
MAX_OUTPUT_TOKENS = 600
#: P15a coach brief: a second call (<= 500 tokens) only after the digest call succeeded with time to spare.
MAX_BRIEF_TOKENS = 500
BRIEF_MIN_SECONDS_LEFT = 240.0
MAX_WATCH = 4
MAX_MISSING = 3
MAX_ISSUES = 5
MAX_ONE_LINE = 200
#: Sessions the recurring-issue candidates and the gate verdicts look back over.
REVIEW_SESSIONS = 10
#: An issue is recurring when it shows at least this often in the window.
ISSUE_MIN_COUNT = 2
MIRROR_WEEKS = 6
MAX_MIRROR_CUTS = 5
MAX_IDEAS = 5
MAX_DIGEST_ITEMS = 5
MAX_OPEN_QUESTIONS = 3
MAX_HYPOTHESES = 3
MAX_ITEM_CHARS = 280
MAX_TURNS = 20
MAX_TURN_CHARS = 300
#: P18: every turn of the day rides (no 20-turn cap), each up to MAX_TURN_TEXT, the oldest dropped first to fit.
MAX_TURN_TEXT = 600
TURN_BUDGET_CHARS = 9000
#: P18: the day's journal lines, the oldest dropped first to fit; the habit counts see every line regardless.
JOURNAL_BUDGET_CHARS = 6000
MAX_LATENCY_VALUES = 500
TIMEOUT_SECONDS = 600
RESERVE_MINUTES = 10.0
BUSY_TIMEOUT_MS = 5000
EFFORT = "high"

INSTRUCTIONS = (
    "You are reviewing ONE day of a trader's conversation with his Trade Mentor app. "
    "Write at most five short digest items worth remembering tomorrow, and at most three "
    "open questions to ask him. Every item cites one or more ids copied exactly from "
    "allowed_evidence_ids. Copy numbers from the evidence; never compute one. Never "
    "suggest an order, a size, or a change to a detector, score or alert. night_reads carry what the "
    "night and the desk already found about this day and recent sessions; use them to say what is worth "
    "remembering. Say nothing "
    "rather than something the evidence does not carry. You may also propose at most three "
    "hypotheses: each is a query into the shadow permutation grid (population, horizon, family, "
    "side and one to three facets as 'name=value', using only names and values listed in "
    "hypothesis_vocabulary), with why and the ids it rests on. The desk looks each one up; "
    "you never state its numbers. Leave hypotheses empty when the vocabulary is empty."
)

_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "evidence_refs"],
    "properties": {
        "text": {"type": "string", "maxLength": MAX_ITEM_CHARS},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}
_QUERY = {
    "type": "object",
    "additionalProperties": False,
    "required": ["population", "horizon", "family", "side", "facets"],
    "properties": {
        "population": {"type": "string", "enum": ["swing", "m5"]},
        "horizon": {"type": "string"},
        "family": {"type": "string"},
        "side": {"type": "string", "enum": ["LONG", "SHORT"]},
        "facets": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    },
}
_HYPOTHESIS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "why", "evidence_refs"],
    "properties": {
        "query": _QUERY,
        "why": {"type": "string", "maxLength": MAX_ITEM_CHARS},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}
DIGEST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["digest", "open_questions", "hypotheses"],
    "properties": {
        "digest": {"type": "array", "maxItems": MAX_DIGEST_ITEMS, "items": _ITEM},
        "open_questions": {"type": "array", "maxItems": MAX_OPEN_QUESTIONS, "items": _ITEM},
        "hypotheses": {"type": "array", "maxItems": MAX_HYPOTHESES, "items": _HYPOTHESIS},
    },
}


class MentorReviewRejected(ValueError):
    """The reply cited an id tonight's pack does not carry, or had no arrays."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _utc(stamp: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(_text(stamp))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _pt_day(stamp: Any) -> str:
    moment = _utc(stamp)
    return moment.astimezone(PT).date().isoformat() if moment else ""


def _moment(now: datetime | None) -> datetime:
    stamp = now or datetime.now(timezone.utc)
    return stamp if stamp.tzinfo else stamp.astimezone()


def _live_journal() -> Path:
    import project_paths

    return Path(project_paths.JOURNAL_DB_FILE)


def _chat_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    import project_paths

    return Path(project_paths.MENTOR_CHAT_DB_FILE)


def _root(ai_root: Path | str | None) -> Path:
    if ai_root is not None:
        return Path(ai_root)
    from ai_jobs import store

    return store.digests_dir()


# ---------------------------------------------------------------------------
# the chat DB: read-only, plus the grading columns
# ---------------------------------------------------------------------------
def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def _rows(path: Path, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    with closing(_connect_ro(path)) as conn:
        try:
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise


class NightChallengeStore:
    """What ``challenge.grade_open`` needs: reads ``mode=ro``; the one write is a challenge's grade."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.graded: list[str] = []  # rows whose graded_utc this run set
        self.updated: list[str] = []  # every row whose outcome this run changed
        self.added: list[str] = []  # hypothesis rows this run inserted

    def challenges(self, *, kind: str | None = None, open_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM challenges WHERE 1 = 1"
        params: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if open_only:
            sql += " AND (graded_utc IS NULL OR graded_utc = '')"
        return _rows(self.path, sql + " ORDER BY issued_utc, id", params)

    def add_challenge(self, challenge_id: str, *, kind: str, symbol: str = "", claim: str = "",
                      evidence_ids: Sequence[str] = (), issued_utc: str = "",
                      outcome: Mapping[str, Any] | None = None) -> bool:
        """Insert one open challenge (the night's hypotheses); an id already issued is left alone."""
        try:
            with closing(sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)) as conn, conn:
                conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO challenges (id, kind, symbol, claim, evidence_ids_json, issued_utc, "
                    "graded_utc, outcome_json) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                    (str(challenge_id), str(kind), str(symbol or ""), str(claim),
                     json.dumps(list(evidence_ids)), str(issued_utc),
                     json.dumps(dict(outcome or {}), sort_keys=True, default=str)),
                )
        except sqlite3.Error:
            _log.exception("mentor_review: challenge %s could not be written", challenge_id)
            return False
        if cursor.rowcount:
            self.added.append(str(challenge_id))
        return bool(cursor.rowcount)

    def update_challenge(self, challenge_id: str, *, outcome: dict[str, Any], graded_utc: str | None = None) -> bool:
        try:
            with closing(sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)) as conn, conn:
                conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
                conn.execute(
                    "UPDATE challenges SET outcome_json = ?, graded_utc = ? WHERE id = ?",
                    (json.dumps(dict(outcome), sort_keys=True, default=str), graded_utc, str(challenge_id)),
                )
        except sqlite3.Error:
            _log.exception("mentor_review: grading %s could not be written", challenge_id)
            return False
        self.updated.append(str(challenge_id))
        if graded_utc:
            self.graded.append(str(challenge_id))
        return True


# ---------------------------------------------------------------------------
# the day's facts (deterministic)
# ---------------------------------------------------------------------------
def _percentile(values: Sequence[float], share: float) -> int | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round(share * (len(ordered) - 1)))))
    return int(ordered[index])


def _day_rows(path: Path, table: str, column: str, session: str) -> list[dict[str, Any]]:
    """Rows of ``table`` whose ``column`` (UTC ISO) falls on the PT day ``session``."""
    start = datetime.combine(datetime.fromisoformat(session).date(), time(0), PT).astimezone(timezone.utc)
    lo = (start - timedelta(days=1)).date().isoformat()
    hi = (start + timedelta(days=2)).date().isoformat()
    rows = _rows(path, f"SELECT * FROM {table} WHERE {column} >= ? AND {column} < ?", (lo, hi))
    return [row for row in rows if _pt_day(row.get(column)) == session]


def _json_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return list(value) if isinstance(value, list) else []


def _outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("outcome_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def day_facts(path: Path, session: str, *, grading: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Counts for one PT day of the chat DB. Missing data is 0 / None, never a guess."""
    turns = _day_rows(path, "turns", "ts_utc", session)
    user = [row for row in turns if row.get("role") == "user"]
    assistant = [row for row in turns if row.get("role") == "assistant"]
    tools: dict[str, int] = {}
    for row in assistant:
        for call in _json_list(row.get("tool_calls_json")):
            name = _text(call.get("name")) if isinstance(call, Mapping) else ""
            if name:
                tools[name] = tools.get(name, 0) + 1
    latencies = [int(row["latency_ms"]) for row in assistant if row.get("latency_ms") is not None]
    picks = [row for row in _day_rows(path, "pack_cache", "built_utc", session) if row.get("name") == "pick_assessment"]
    issued = _day_rows(path, "challenges", "issued_utc", session)
    graded_today = _day_rows(path, "challenges", "graded_utc", session)
    hits = [bool(_outcome(row)["hit"]) for row in graded_today if "hit" in _outcome(row)]
    notes = [row for row in _day_rows(path, "profile_notes", "ts_utc", session) if row.get("source") == "remember"]
    stats_rows = _rows(path, "SELECT value FROM app_state WHERE key = ?", (f"stats:{session}",))
    try:
        stats = json.loads(stats_rows[0]["value"]) if stats_rows else {}
    except ValueError:
        stats = {}
    return {
        "schema": FACTS_SCHEMA,
        "session_date": session,
        "turns": {"user": len(user), "assistant": len(assistant)},
        "tool_calls": dict(sorted(tools.items())),
        "picks_assessed": len(picks),
        "challenges": {
            "issued": len(issued),
            "graded": len(graded_today),
            "hit": sum(hits),
            "hit_n": len(hits),
        },
        "remember_notes": len(notes),
        "uncited_numbers": int(stats.get("uncited_numbers") or 0),
        "numbers": int(stats.get("numbers") or 0),
        "first_token_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "n": len(latencies),
            "values": latencies[:MAX_LATENCY_VALUES],
        },
        "brain_offline_min": round(float(stats.get("brain_offline_min") or 0), 1),
        "grading": dict(grading or {}),
    }


def fact_rows(facts: Mapping[str, Any]) -> list[dict[str, str]]:
    """The facts as citable rows (``fact:<name>``)."""
    latency = facts.get("first_token_ms") or {}
    chal = facts.get("challenges") or {}
    turns = facts.get("turns") or {}
    tools = ", ".join(f"{name} {count}" for name, count in (facts.get("tool_calls") or {}).items()) or "none"
    return [
        {"id": "fact:turns", "text": f"{turns.get('user', 0)} trader turns, {turns.get('assistant', 0)} mentor replies"},
        {"id": "fact:tools", "text": f"packs read: {tools}"},
        {"id": "fact:picks", "text": f"{facts.get('picks_assessed', 0)} pick assessments built"},
        {"id": "fact:challenges", "text": (
            f"challenges issued {chal.get('issued', 0)}, graded {chal.get('graded', 0)}, "
            f"hit {chal.get('hit', 0)} of n={chal.get('hit_n', 0)}")},
        {"id": "fact:notes", "text": f"{facts.get('remember_notes', 0)} /remember notes added"},
        {"id": "fact:uncited", "text": (
            f"{facts.get('uncited_numbers', 0)} uncited numbers of {facts.get('numbers', 0)} in replies")},
        {"id": "fact:latency", "text": (
            f"first token p50 {latency.get('p50')} ms, p95 {latency.get('p95')} ms (n={latency.get('n', 0)})")},
        {"id": "fact:offline", "text": f"brain offline {facts.get('brain_offline_min', 0)} minutes"},
    ]


# ---------------------------------------------------------------------------
# inputs, citation check, publish
# ---------------------------------------------------------------------------
def hypothesis_context(report: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """(the ``hyp:report:asof`` row, the report's vocabulary) for the model; empty without a report."""
    from mentor_packs import hypothesis_pack

    if report is None:
        return [], {}
    row = {"id": "hyp:report:asof",
           "text": f"newest permutation report: data date {report.asof or 'unknown'} ({Path(report.path).name})"}
    return [row], hypothesis_pack.vocabulary(report)


def _window_start(session: str, sessions: int = REVIEW_SESSIONS) -> str:
    """The first weekday of the last ``sessions`` weekdays ending on ``session`` (holidays count as sessions)."""
    day = datetime.fromisoformat(session).date()
    left = sessions - 1
    while left > 0:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            left -= 1
    return day.isoformat()


def _since_rows(path: Path, table: str, column: str, start: str, session: str) -> list[dict[str, Any]]:
    """Rows whose ``column`` falls on a PT day from ``start`` through ``session``."""
    lo = (datetime.fromisoformat(start) - timedelta(days=1)).date().isoformat()
    hi = (datetime.fromisoformat(session) + timedelta(days=2)).date().isoformat()
    rows = _rows(path, f"SELECT * FROM {table} WHERE {column} >= ? AND {column} < ?", (lo, hi))
    return [row for row in rows if start <= _pt_day(row.get(column)) <= session]


def _assessments(path: Path, session: str) -> list[dict[str, Any]]:
    """The day's pick assessments (newest per symbol): verdict, first bullet and any plan line flagged broken."""
    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(_day_rows(path, "pack_cache", "built_utc", session), key=lambda r: _text(r.get("built_utc"))):
        if row.get("name") != "pick_assessment":
            continue
        try:
            payload = json.loads(row.get("pack_json") or "{}")
        except ValueError:
            continue
        symbol = _text(payload.get("symbol")).upper()
        if symbol and _text(payload.get("verdict")):
            latest[symbol] = payload
    out = []
    for symbol, payload in sorted(latest.items()):
        bullets = [b for b in payload.get("bullets") or () if isinstance(b, Mapping)]
        broken = [_text(f.get("plan_id")) for f in payload.get("rule_flags") or ()
                  if isinstance(f, Mapping) and f.get("breaks")]
        text = f"{symbol}: {_text(payload.get('verdict'))}" + (f"; {_text(bullets[0].get('text'))}" if bullets else "")
        if broken:
            text += f"; breaks {', '.join(broken)}"
        out.append({"id": f"assess:{symbol}", "symbol": symbol, "text": text[:MAX_TURN_CHARS], "broken": broken})
    return out


def _challenge_row(row: Mapping[str, Any]) -> dict[str, Any]:
    outcome = _outcome(row)
    graded = ""
    if row.get("graded_utc"):
        parts = [f"{key} {outcome[key]}" for key in ("result", "r", "hit", "rest_pnl") if key in outcome]
        graded = "; graded: " + (", ".join(parts) or "done")
    return {"id": f"challenge:{row['id']}", "kind": _text(row.get("kind")), "symbol": _text(row.get("symbol")),
            "issued": _pt_day(row.get("issued_utc")),
            "text": (_text(row.get("claim"))[:200] + graded)[:MAX_TURN_CHARS],
            "status": _text(outcome.get("status")), "pattern": _text(outcome.get("pattern")),
            "hit": outcome.get("hit")}


def _mirror_rows(builder: Callable[[], Any] | None, moment: datetime) -> list[dict[str, Any]]:
    """The mirror's top cuts (n at or over its floor, biggest n first); never its as-of rows."""
    from mentor_packs import mirror_pack

    pack = builder() if builder is not None else mirror_pack.build(weeks=MIRROR_WEEKS, now=moment)
    floor = mirror_pack.min_reportable_n()
    cuts = [row for row in pack.rows if row.get("kind") not in ("asof", "weeks", "caveats")
            and isinstance(row.get("n"), int) and row["n"] >= floor]
    cuts.sort(key=lambda row: (-int(row["n"]), str(row["id"])))
    return [{"id": str(row["id"]), "text": _text(row.get("text"))[:MAX_TURN_CHARS]} for row in cuts[:MAX_MIRROR_CUTS]]


def _night_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"id": str(row["id"]), "text": _text(row.get("text"))[:MAX_TURN_CHARS]}


def night_inputs(path: Path, session: str, moment: datetime, *, night_paths: Any = None,
                 mirror_builder: Callable[[], Any] | None = None,
                 recap_paths: Any = None) -> dict[str, list[dict[str, Any]]]:
    """P15a: what the night and the app already know about the day, each row with its id.

    A section that cannot be read is left empty and named in ``unread``; it never costs the review.
    """
    from mentor_packs import night_pack

    paths = night_paths if night_paths is not None else night_pack.live_paths()
    day = datetime.fromisoformat(session).date()
    start = _window_start(session)
    out: dict[str, list[dict[str, Any]]] = {}
    unread: list[str] = []

    def section(name: str, read: Callable[[], list[dict[str, Any]]]) -> None:
        try:
            out[name] = read()
        except Exception as exc:  # noqa: BLE001 - one unreadable input never costs the review
            _log.debug("mentor_review: %s could not be read.", name, exc_info=True)
            out[name] = []
            unread.append(f"{name} ({type(exc).__name__})")

    section("pick_assessments", lambda: _assessments(path, session))
    section("gates", lambda: [_challenge_row(row) for row in _since_rows(path, "challenges", "issued_utc", start, session)
                              if row.get("kind") == "gate"])
    section("tilt", lambda: [_challenge_row(row) for row in _day_rows(path, "challenges", "issued_utc", session)
                             if row.get("kind") == "tilt"])
    section("mirror", lambda: _mirror_rows(mirror_builder, moment))
    section("digest_facts", lambda: [_night_row(r) for r in night_pack.digest_fact_rows(paths, session)[0]])
    section("contrasts", lambda: [_night_row(r) for r in (*night_pack.miss_rows(paths, day)[0],
                                                          *night_pack.prediction_rows(paths, day)[0])])
    section("day_review", lambda: [_night_row(r) for r in night_pack.day_review_rows(paths, day, 1)[0]])
    section("ideas", lambda: [_night_row(r) for r in night_pack.idea_rows(paths, day, limit=MAX_IDEAS)[0]])
    # P15b: the day recaps' recurrence table, so an issue's own row can be cited.
    section("recap_issues", lambda: [_night_row(r) for r in recap_issue_rows(session, recap_paths)])
    out["unread"] = [{"id": "", "text": name} for name in unread]
    return out


def _strip_src(text: str) -> str:
    return re.sub(r"\s*\(src: [^)]*\)$", "", _text(text))


def recap_issue_rows(session: str, recap_paths: Any = None) -> list[dict[str, Any]]:
    """P15b: the day recaps' recurrence table (``recap:issues:<key>``) over the last REVIEW_SESSIONS sessions."""
    from mentor_packs import recaps_pack

    day = datetime.fromisoformat(session).date()
    return recaps_pack.issue_rows(recap_paths, today=day, days=REVIEW_SESSIONS)


def issue_candidates(path: Path, session: str, *, night_paths: Any = None,
                     earlier: Sequence[Mapping[str, Any]] = (),
                     registry: Mapping[str, Mapping[str, Any]] | None = None,
                     recap_paths: Any = None) -> list[dict[str, Any]]:
    """Recurring problems over the last REVIEW_SESSIONS sessions, found by code (the model only words and ranks).

    Each has a stable ``key`` (so ``first_seen`` carries over from ``earlier`` coach briefs), an
    ``issue:<key>`` id, a count and the ids it rests on. Sources: the miss contrast's leader groups,
    vetoes graded as the name winning, repeated tilt patterns, plan lines flagged broken on pick
    assessments, reads the day reviews graded wrong, and (P15b) the day recaps' recurrence table
    (``recap:<key>``, count = sessions; its wrong-reads row is the one above, so it is not repeated).
    """
    from mentor_packs import night_pack

    paths = night_paths if night_paths is not None else night_pack.live_paths()
    day = datetime.fromisoformat(session).date()
    start = _window_start(session)
    found: list[dict[str, Any]] = []

    def add(key: str, text: str, count: int, refs: Sequence[str]) -> None:
        found.append({"key": key, "id": f"issue:{key}", "text": text[:MAX_ITEM_CHARS], "count": int(count),
                      "refs": [ref for ref in refs if ref][:5]})

    def guarded(read: Callable[[], None]) -> None:
        try:
            read()
        except Exception:  # noqa: BLE001 - one unreadable source never costs the others
            _log.debug("mentor_review: an issue source could not be read.", exc_info=True)

    def misses() -> None:
        for row in night_pack.miss_rows(paths, day)[0]:
            if row.get("group") and row["id"].split(":")[-1] in ("1", "2", "3"):
                add(f"miss:{row['group']}", _strip_src(row["text"]), 1, [row["id"]])

    def vetoes() -> None:
        won = [row for row in _since_rows(path, "challenges", "graded_utc", start, session)
               if row.get("kind") == "veto" and _outcome(row).get("hit") is True]
        if len(won) >= ISSUE_MIN_COUNT:
            names = sorted({_text(row.get("symbol")) for row in won if row.get("symbol")})
            add("veto_won", f"{len(won)} vetoed names won anyway in the last {REVIEW_SESSIONS} sessions"
                + (f" ({', '.join(names[:5])})" if names else ""), len(won), [f"challenge:{r['id']}" for r in won])

    def tilts() -> None:
        by_pattern: dict[str, list[dict[str, Any]]] = {}
        for row in _since_rows(path, "challenges", "issued_utc", start, session):
            if row.get("kind") == "tilt" and _text(_outcome(row).get("pattern")):
                by_pattern.setdefault(_text(_outcome(row)["pattern"]), []).append(row)
        for pattern, rows in sorted(by_pattern.items()):
            days = sorted({_pt_day(row.get("issued_utc")) for row in rows})
            if len(days) >= ISSUE_MIN_COUNT:
                graded = [_outcome(row).get("hit") for row in rows if "hit" in _outcome(row)]
                red = f"; rest of day red {sum(1 for hit in graded if hit)} of {len(graded)} graded" if graded else ""
                add(f"tilt:{pattern}", f"Tilt pattern '{pattern}' on {len(days)} of the last {REVIEW_SESSIONS} sessions"
                    + red, len(days), [f"challenge:{row['id']}" for row in rows])

    def rules() -> None:
        broken: dict[str, set[str]] = {}
        for row in _since_rows(path, "pack_cache", "built_utc", start, session):
            if row.get("name") != "pick_assessment":
                continue
            try:
                payload = json.loads(row.get("pack_json") or "{}")
            except ValueError:
                continue
            for flag in payload.get("rule_flags") or ():
                if isinstance(flag, Mapping) and flag.get("breaks") and _text(flag.get("plan_id")):
                    broken.setdefault(_text(flag["plan_id"]), set()).add(
                        f"{_pt_day(row.get('built_utc'))}:{_text(payload.get('symbol')).upper()}")
        for plan_id, picks in sorted(broken.items()):
            if len(picks) >= ISSUE_MIN_COUNT:
                add(f"rule:{plan_id}", f"Plan line {plan_id} flagged broken on {len(picks)} pick assessments in the "
                    f"last {REVIEW_SESSIONS} sessions", len(picks), [plan_id])

    def wrong_reads() -> None:
        rows = night_pack.day_review_rows(paths, day, REVIEW_SESSIONS)[0]
        wrong = [row for row in rows if row.get("verdict") == "wrong" and row.get("date", "") >= start]
        if len(wrong) >= ISSUE_MIN_COUNT:
            add("wrong_reads", f"{len(wrong)} of your reads were graded wrong by the day reviews of the last "
                f"{REVIEW_SESSIONS} sessions", len(wrong), [row["id"] for row in wrong])

    def recaps() -> None:
        for row in recap_issue_rows(session, recap_paths):
            if row["key"] != "wrong_reads":
                add(f"recap:{row['key']}", row["text"], row["count"], [row["id"]])
                found[-1]["first"] = row["first"]  # the recap's own first session dates the issue

    for read in (misses, vetoes, tilts, rules, wrong_reads, recaps):
        guarded(read)
    seen: dict[str, str] = {}
    for payload in earlier:
        for item in payload.get("issues") or ():
            if isinstance(item, Mapping) and _text(item.get("key")) and _text(item.get("first_seen")):
                key = _text(item["key"])
                seen[key] = min(seen.get(key, item["first_seen"]), _text(item["first_seen"]))
    # The issue registry remembers every candidate, also one that never made the top five of a brief.
    for key, entry in (registry or {}).items():
        if isinstance(entry, Mapping) and _text(entry.get("first_seen")):
            seen[key] = min(seen.get(key, entry["first_seen"]), _text(entry["first_seen"]))
    for item in found:
        item["first_seen"] = min(seen.get(item["key"], session), session, item.pop("first", "") or session)
    found.sort(key=lambda item: (-item["count"], item["first_seen"], item["key"]))
    return found


def registry_path(root: Path) -> Path:
    return Path(root) / REGISTRY_FILE


def read_issue_registry(root: Path) -> dict[str, dict[str, Any]]:
    """``{key: {first_seen, last_seen, nights_seen, sessions}}``; {} when none or unreadable."""
    issues = _read_json(registry_path(root)).get("issues")
    return {str(key): dict(value) for key, value in (issues or {}).items() if isinstance(value, Mapping)}


def update_issue_registry(root: Path, session: str, candidates: Sequence[Mapping[str, Any]],
                          registry: Mapping[str, Mapping[str, Any]], built_utc: str) -> Path:
    """Record every candidate seen on ``session`` (the facts half, every night): first_seen never moves later,
    a rerun of the same session counts once. Temp-and-rename; a failed write keeps the last good file."""
    issues = {key: dict(value) for key, value in registry.items()}
    for item in candidates:
        entry = issues.setdefault(item["key"], {})
        sessions = sorted({*entry.get("sessions", ()), session})[-REGISTRY_SESSIONS:]
        entry.update({
            "first_seen": min(_text(entry.get("first_seen")) or item["first_seen"], item["first_seen"], session),
            "last_seen": max(_text(entry.get("last_seen")) or session, session),
            "sessions": sessions, "nights_seen": max(int(entry.get("nights_seen") or 0), len(sessions)),
            "text": item["text"], "count": item["count"],
        })
    return _publish(registry_path(root), {"schema": REGISTRY_SCHEMA, "updated_utc": built_utc,
                                          "issues": dict(sorted(issues.items()))})


def budgeted(rows: Sequence[Mapping[str, Any]], budget: int) -> list[dict[str, Any]]:
    """The newest rows whose texts fit ``budget`` characters, in their order (the oldest go first)."""
    kept: list[dict[str, Any]] = []
    used = 0
    for row in reversed(rows):
        used += len(_text(row.get("text")))
        if used > budget and kept:
            break
        kept.append(dict(row))
    return kept[::-1]


def journal_rows(items: Sequence[Mapping[str, Any]], session: str) -> list[dict[str, Any]]:
    """P18: the day's journal lines as citable inputs (``journal:<id>``), budgeted oldest-first."""
    rows = [{"id": item["id"], "text": f"{item['bucket_et']} ET [{', '.join(item['tags']) or 'no tag'}]"
                                       + (" after a loss" if item.get("after_loss") else "") + f": {item['text']}"}
            for item in items if item["kind"] == "journal" and item["day"] == session]
    return budgeted(rows, JOURNAL_BUDGET_CHARS)


def read_rows(session: str, moment: datetime, sources: Any = None) -> tuple[list[dict[str, Any]], list[Any]]:
    """P18: the session's own Market Journal reads with their grades (``read:<entry_id>``), and every grade."""
    from mentor_packs import reads_pack

    src = sources if sources is not None else reads_pack.live_sources()
    pack = reads_pack.build(n=reads_pack.MAX_N, now=moment, sources=src)
    rows = [{"id": row["id"], "text": _text(row.get("text"))[:MAX_TURN_TEXT]} for row in pack.rows
            if row.get("kind") in ("current", "read") and row.get("session") == session]
    try:
        grades = list(src.grades())
    except Exception:  # noqa: BLE001 - unreadable grades: no disagreement run, never a guess
        grades = []
    return rows, grades


def read_tape_candidate(session: str, grades: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """P18: an issue when his clicked rest-of-day read and the tape disagreed two sessions running."""
    from mentor_packs import reads_pack

    run = reads_pack.disagreement_run(grades, session)
    if not run:
        return None
    refs = [f"read:{entry}" for day in run for entry in day["entry_ids"] if entry]
    days = ", ".join(day["session"] for day in run)
    return {"key": "read_vs_tape", "id": "issue:read_vs_tape", "count": len(run), "refs": refs[:5],
            "text": f"Your rest-of-day read and the tape disagreed {len(run)} sessions running ({days})"}


ROUTINES_FILE = "mentor_routines.json"
#: Calendar days of turns read for the routine count (enough for 10 session days).
ROUTINE_LOOKBACK_DAYS = 21


def publish_routines(root: Path, path: Path, session: str, built_utc: str) -> str:
    """P18 D: write ``mentor_routines.json`` from the asks view; returns the brief's line ("" unless it changed)."""
    from mentor_app import routines

    lo = (datetime.fromisoformat(session).date() - timedelta(days=ROUTINE_LOOKBACK_DAYS)).isoformat()
    rows = _rows(path, "SELECT id, ts_utc, role, tool_calls_json FROM turns WHERE ts_utc >= ? ORDER BY id", (lo,))
    table = routines.find_routines(routines.asks_from_turns(rows), session)
    target = Path(root) / ROUTINES_FILE
    old = routines.read_routines(target)
    if _text(old.get("session_date")) == session:  # a rerun keeps tonight's line unless the table moved again
        line = _text(old.get("line")) if old.get("routines") == table["routines"] else routines.routine_line(
            table, {"routines": old.get("previous") or []})
        previous = old.get("previous") or []
    else:
        line, previous = routines.routine_line(table, old), old.get("routines") or []
    _publish(target, {**table, "built_utc": built_utc, "line": line, "previous": previous})
    return line


def build_inputs(path: Path, session: str, facts: Mapping[str, Any], *, report: Any = None,
                 night: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Everything the model may see, with ids; ``inputs_hash`` ignores the clock."""
    # P15b: a feeling is cited by its trade (``feel:<trade_id>``, the newest one per trade); a note by its id.
    by_id: dict[str, dict[str, Any]] = {}
    for row in _day_rows(path, "profile_notes", "ts_utc", session):
        feeling = _text(row.get("kind")) == "feeling" and _text(row.get("trade_id"))
        key = f"feel:{_text(row.get('trade_id'))}" if feeling else f"note:{row['id']}"
        by_id.pop(key, None)
        by_id[key] = {"id": key, "text": _text(row.get("text"))[:MAX_TURN_CHARS]}
    notes = list(by_id.values())
    # The day's gate and tilt rows ride in their own sections (``night``), so each id appears once.
    split = night is not None
    challenges = [
        {"id": f"challenge:{row['id']}", "kind": _text(row.get("kind")), "symbol": _text(row.get("symbol")),
         "text": _text(row.get("claim"))[:MAX_TURN_CHARS], "status": _text(_outcome(row).get("status"))}
        for row in _day_rows(path, "challenges", "issued_utc", session)
        if not (split and row.get("kind") in ("gate", "tilt"))
    ]
    turns = budgeted([
        {"id": f"turn:{row['id']}", "role": _text(row.get("role")), "text": _text(row.get("text"))[:MAX_TURN_TEXT]}
        for row in _day_rows(path, "turns", "ts_utc", session)
        if row.get("role") in ("user", "assistant")
    ], TURN_BUDGET_CHARS)
    rows = fact_rows(facts)
    report_rows, vocab = hypothesis_context(report)
    sections = {key: [dict(row) for row in value] for key, value in (night or {}).items() if key != "unread"}
    ids = [row["id"] for row in (*rows, *notes, *challenges, *turns, *report_rows)]
    for value in sections.values():
        ids += [row["id"] for row in value if row.get("id") and row["id"] not in ids]
    body: dict[str, Any] = {
        "session_date": session,
        "facts": rows,
        "profile_notes": notes,
        "challenges": challenges,
        "turns": turns,
        "permutation_report": report_rows,
        "hypothesis_vocabulary": vocab,
        **({"night_reads": sections} if split else {}),
        "allowed_evidence_ids": ids,
    }
    body["inputs_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return body


def build_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in inputs.items() if key != "inputs_hash"}
    body["package_id"] = f"mentor-review:{_text(inputs.get('inputs_hash'))[:16]}"
    body["evidence_hash"] = _text(inputs.get("inputs_hash"))
    body["instructions"] = INSTRUCTIONS
    return body


def _refs(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("evidence_refs") or ()
    return [_text(ref) for ref in ([raw] if isinstance(raw, str) else raw) if _text(ref)]


def check_reply(reply: Any, inputs: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """(kept digest and questions, dropped count). Raises MentorReviewRejected for a foreign id."""
    if not isinstance(reply, Mapping):
        raise MentorReviewRejected("the reply was not an object")
    allowed = {_text(item) for item in inputs.get("allowed_evidence_ids") or ()}
    kept: dict[str, list[dict[str, Any]]] = {"digest": [], "open_questions": [], "hypotheses": []}
    dropped = 0
    for key, cap in (("digest", MAX_DIGEST_ITEMS), ("open_questions", MAX_OPEN_QUESTIONS)):
        rows = reply.get(key)
        if rows is None or not isinstance(rows, (list, tuple)):
            raise MentorReviewRejected(f"the reply carried no {key} array")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise MentorReviewRejected(f"{key} {index} was not an object")
            refs = _refs(row)
            for ref in refs:
                if ref not in allowed:
                    raise MentorReviewRejected(f"{key} {index} cited {ref!r}, which tonight does not carry")
            text = _text(row.get("text"))
            if not refs or not text or len(text) > MAX_ITEM_CHARS or len(kept[key]) >= cap:
                dropped += 1
                continue
            kept[key].append({"text": text, "evidence_refs": refs})
    hypotheses = reply.get("hypotheses")
    if hypotheses is not None and not isinstance(hypotheses, (list, tuple)):
        raise MentorReviewRejected("hypotheses was not an array")
    for index, row in enumerate(hypotheses or ()):
        if not isinstance(row, Mapping):
            raise MentorReviewRejected(f"hypotheses {index} was not an object")
        refs = _refs(row)
        for ref in refs:
            if ref not in allowed:
                raise MentorReviewRejected(f"hypotheses {index} cited {ref!r}, which tonight does not carry")
        why = _text(row.get("why"))
        query = row.get("query")
        if (not refs or not why or len(why) > MAX_ITEM_CHARS or not isinstance(query, Mapping)
                or len(kept["hypotheses"]) >= MAX_HYPOTHESES):
            dropped += 1
            continue
        kept["hypotheses"].append({"query": dict(query), "why": why, "evidence_refs": refs})
    return kept, dropped


def look_up_hypotheses(
    hypotheses: Sequence[Mapping[str, Any]],
    report: Any,
    *,
    session: str,
    issued_utc: str,
    night_store: "NightChallengeStore | None",
) -> list[dict[str, Any]]:
    """Look each hypothesis up in the report (no compute) and record it when the night may write."""
    from mentor_packs import hypothesis_pack

    out: list[dict[str, Any]] = []
    for index, item in enumerate(hypotheses, start=1):
        hyp_id = f"hyp:{session}:{index}"
        clean, _why = hypothesis_pack.normalise(item.get("query"))
        query = clean or dict(item.get("query") or {})
        record = hypothesis_pack.lookup_record(query, report)
        recorded = False
        if night_store is not None:
            recorded = night_store.add_challenge(
                hyp_id, kind=hypothesis_pack.KIND, claim=_text(item.get("why")),
                evidence_ids=list(item.get("evidence_refs") or ()), issued_utc=issued_utc,
                outcome={"status": "open", "query": query, "lookup": record},
            )
        text = hypothesis_pack.record_text(record)
        out.append({
            "id": hyp_id, "query": query, "why": _text(item.get("why")),
            "evidence_refs": list(item.get("evidence_refs") or ()), "lookup": record, "recorded": recorded,
            "line": f"[{hyp_id}] {hypothesis_pack.query_label(query)} -> {text}",
        })
    return out


def published_path(root: Path, stem: str, session: str) -> Path:
    return Path(root) / f"{stem}_{session}.json"


def _publish(path: Path, payload: Mapping[str, Any]) -> Path:
    from ai_jobs import digest

    return digest._publish(path, json.dumps(payload, sort_keys=True, indent=2, default=str))


def read_published(root: Path, stem: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """The newest ``limit`` publications of ``stem``, newest first; an unreadable file is skipped."""
    out: list[dict[str, Any]] = []
    try:
        paths = sorted(Path(root).glob(f"{stem}_????-??-??.json"), reverse=True)
    except OSError:
        return []
    for path in paths:
        if len(out) >= limit:
            break
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


# ---------------------------------------------------------------------------
# P15a: the coach brief - the night's product for the day coach
# ---------------------------------------------------------------------------
BRIEF_PROMPT_VERSION = "mentor_coach_brief_v2"
BRIEF_SCHEMA_NAME = "tradingbot_mentor_coach_brief"
BRIEF_INSTRUCTIONS = (
    "Write tomorrow morning's coach brief for the trader from the evidence below. watch: at most four "
    "things to watch today. missing: at most three things he may be missing, from the contrasts, ideas and "
    "day review. issues: rank and word the recurring issues listed in issue_candidates, each by its key; "
    "never invent an issue. one_line: one short sentence for the top of his day, with its own ids. Every item "
    "and the one line cite ids copied "
    "exactly from allowed_evidence_ids. Copy numbers; never compute one. Never suggest an order, a size, or a "
    "change to a rule, detector, score or alert; these are observations, never rules."
)
_KEYED = {
    "type": "object",
    "additionalProperties": False,
    "required": ["key", "text", "evidence_refs"],
    "properties": {
        "key": {"type": "string"},
        "text": {"type": "string", "maxLength": MAX_ITEM_CHARS},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}
#: Bounds are rechecked in ``check_brief`` (a schema is only a grammar hint, never a guard).
BRIEF_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["watch", "missing", "issues", "one_line"],
    "properties": {
        "watch": {"type": "array", "items": _ITEM},
        "missing": {"type": "array", "items": _ITEM},
        "issues": {"type": "array", "items": _KEYED},
        "one_line": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "evidence_refs"],
            "properties": {
                "text": {"type": "string", "maxLength": MAX_ONE_LINE},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}
#: The published one line when the model gave none or left it uncited (memory leads with the first watch item).
NO_ONE_LINE: dict[str, Any] = {"text": "", "evidence_refs": []}


def candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The issue candidates as citable input rows (``issue:<key>``)."""
    return [{"id": item["id"], "key": item["key"],
             "text": (f"{item['text']} (count {item['count']}, first seen {item['first_seen']}; rests on "
                      f"{', '.join(item['refs']) or 'its own count'})")[:MAX_TURN_CHARS]}
            for item in candidates]


def check_brief(reply: Any, inputs: Mapping[str, Any],
                candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], int]:
    """(kept brief, dropped count). A foreign id rejects the whole reply; uncited items and unknown keys drop."""
    if not isinstance(reply, Mapping):
        raise MentorReviewRejected("the brief was not an object")
    allowed = {_text(item) for item in inputs.get("allowed_evidence_ids") or ()}
    by_key = {item["key"]: item for item in candidates}
    kept: dict[str, Any] = {"watch": [], "missing": [], "issues": []}
    dropped = 0
    for key, cap in (("watch", MAX_WATCH), ("missing", MAX_MISSING), ("issues", MAX_ISSUES)):
        rows = reply.get(key)
        if rows is None or not isinstance(rows, (list, tuple)):
            raise MentorReviewRejected(f"the brief carried no {key} array")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise MentorReviewRejected(f"{key} {index} was not an object")
            refs = _refs(row)
            for ref in refs:
                if ref not in allowed:
                    raise MentorReviewRejected(f"{key} {index} cited {ref!r}, which tonight does not carry")
            text = _text(row.get("text"))
            issue = by_key.get(_text(row.get("key"))) if key == "issues" else None
            if (not refs or not text or len(text) > MAX_ITEM_CHARS or len(kept[key]) >= cap
                    or (key == "issues" and (issue is None or any(i["key"] == issue["key"] for i in kept[key])))):
                dropped += 1
                continue
            item: dict[str, Any] = {"text": text, "evidence_refs": refs}
            if issue is not None:
                item = {"key": issue["key"], **item, "first_seen": issue["first_seen"], "count": issue["count"]}
            kept[key].append(item)
    line = reply.get("one_line")
    if line is not None and not isinstance(line, Mapping):
        raise MentorReviewRejected("one_line was not an object with text and evidence_refs")
    refs = _refs(line or {})
    for ref in refs:
        if ref not in allowed:
            raise MentorReviewRejected(f"one_line cited {ref!r}, which tonight does not carry")
    text = _text((line or {}).get("text"))
    cited = bool(text and refs and len(text) <= MAX_ONE_LINE)
    kept["one_line"] = {"text": text, "evidence_refs": refs} if cited else dict(NO_ONE_LINE)
    if text and kept["one_line"] == NO_ONE_LINE:
        dropped += 1  # an uncited one line is model text with nothing under it
    # The model ranks and words; an issue it left out still stands, worded by the code, after its ranking.
    for issue in candidates:
        if len(kept["issues"]) >= MAX_ISSUES:
            break
        if all(item["key"] != issue["key"] for item in kept["issues"]):
            kept["issues"].append(_fact_issue(issue))
    return kept, dropped


def _fact_issue(issue: Mapping[str, Any]) -> dict[str, Any]:
    return {"key": issue["key"], "text": issue["text"], "evidence_refs": [issue["id"]],
            "first_seen": issue["first_seen"], "count": issue["count"]}


def brief_payload(session: str, built_utc: str, inputs: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                  *, kept: Mapping[str, Any] | None = None, model: str = "", dropped: int = 0,
                  habits: Sequence[Mapping[str, Any]] = (), routine: str = "") -> dict[str, Any]:
    """The published coach brief; without ``kept`` it is the facts part only (issues worded by code)."""
    body = kept or {"watch": [], "missing": [], "one_line": dict(NO_ONE_LINE),
                    "issues": [_fact_issue(issue) for issue in candidates[:MAX_ISSUES]]}
    return {
        "schema": COACH_SCHEMA, "session_date": session, "built_utc": built_utc, "worded": kept is not None,
        "model": model, "prompt_version": BRIEF_PROMPT_VERSION, "inputs_hash": _text(inputs.get("inputs_hash")),
        "one_line": dict(body.get("one_line") or NO_ONE_LINE), "watch": list(body.get("watch") or ()),
        "missing": list(body.get("missing") or ()), "issues": list(body.get("issues") or ()), "dropped": dropped,
        # P18: the top habits (observations, never rules), worded by the model when it answered, else by code.
        "habits": [dict(item) for item in habits],
        # P18: "You usually ask X at Y", only on the night the routine table changed (code, never model text).
        "routine": routine,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def publish_fact_brief(root: Path, session: str, built_utc: str, inputs: Mapping[str, Any],
                       candidates: Sequence[Mapping[str, Any]],
                       habits: Sequence[Mapping[str, Any]] = (), routine: str = "") -> Path | None:
    """The deterministic half: publish the facts-only brief unless a worded one already stands for the session."""
    path = published_path(root, COACH_STEM, session)
    if _read_json(path).get("worded"):
        return None
    return _publish(path, brief_payload(session, built_utc, inputs, candidates, habits=habits, routine=routine))


def ask_brief(request: Callable[..., Mapping[str, Any]], *, model: str, post: Callable[..., Any],
              inputs: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], str, int]:
    """One brief call (<= MAX_BRIEF_TOKENS). Returns (kept, answered model, dropped); raises on a bad reply."""
    evidence = build_evidence(inputs)
    evidence["instructions"] = BRIEF_INSTRUCTIONS
    evidence["package_id"] = f"mentor-coach-brief:{_text(inputs.get('inputs_hash'))[:16]}"
    result = request(
        provider="local", model=model, api_key="", evidence=evidence, timeout_seconds=TIMEOUT_SECONDS,
        post=capped_post(post, model=model, cap=MAX_BRIEF_TOKENS),
        schema=BRIEF_JSON_SCHEMA, schema_name=BRIEF_SCHEMA_NAME, prompt_version=BRIEF_PROMPT_VERSION,
    )
    kept, dropped = check_brief((result or {}).get("summary"), inputs, candidates)
    return kept, _text((result or {}).get("model")) or model, dropped


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def capped_post(post: Callable[..., Any], *, model: str, cap: int = MAX_OUTPUT_TOKENS) -> Callable[..., Any]:
    """Wrap ``post``: at most ``cap`` output tokens, and high reasoning effort for gpt-oss tags (Qt-free)."""

    def wrapped(url: str, **kwargs: Any) -> Any:
        payload = dict(kwargs.get("json") or {})
        payload["max_tokens"] = min(int(payload.get("max_tokens") or cap), cap)
        if str(model or "").strip().lower().startswith("gpt-oss"):
            payload["reasoning_effort"] = EFFORT
        kwargs["json"] = payload
        return post(url, **kwargs)

    return wrapped


def model_wanted(*, session_date: str = "", chat_db: Path | str | None = None, **_ignored: Any) -> bool:
    """False when the chat DB has no turn on this PT day, so no model is loaded to probe."""
    path = _chat_path(chat_db)
    if not path.exists():
        return False
    session = _text(session_date)[:10]
    try:
        if not session:
            return True
        # P18: a day of journal lines only (no chat) still has habits to word.
        return bool(_day_rows(path, "turns", "ts_utc", session)
                    or _rows(path, "SELECT id FROM journal_entries WHERE day_et = ? LIMIT 1", (session,)))
    except (sqlite3.Error, ValueError):
        return True


def _grade(path: Path, moment: datetime, veto_outcomes: Any, *, permutation_history: Any = None,
           permutation_report: Any = None) -> dict[str, Any]:
    from mentor_app import challenge

    if not challenge.night_owns_grading(moment):
        return {"owner": "app", "graded": 0, "updated": 0, "reason": "outside 22:00-06:00 PT the app grades"}
    night_store = NightChallengeStore(path)
    try:
        updated = challenge.grade_open(night_store, moment, veto_outcomes=veto_outcomes,
                                       permutation_history=permutation_history,
                                       permutation_report=permutation_report)
    except Exception as exc:  # noqa: BLE001 - a grading failure never costs the facts
        _log.debug("mentor_review grading failed.", exc_info=True)
        return {"owner": "night", "graded": len(night_store.graded), "updated": len(night_store.updated),
                "error": f"{type(exc).__name__}: {exc}"}
    return {"owner": "night", "graded": len(night_store.graded), "updated": int(updated)}


def run_mentor_review(
    *,
    session_date: str = "",
    now: datetime | None = None,
    chat_db: Path | str | None = None,
    ai_root: Path | str | None = None,
    veto_outcomes: Any = None,
    permutation_history: Path | str | None = None,
    permutation_report: Path | str | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    post: Callable[..., Any] | None = None,
    ask: bool = True,
    force: bool = False,
    night_paths: Any = None,
    mirror_builder: Callable[[], Any] | None = None,
    recap_paths: Any = None,
    journal_db: Path | str | None = None,
    reads_sources: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """One night's review of the Trade Mentor app's day. Never raises."""
    started = clock.monotonic()
    moment = _moment(now)
    session = _text(session_date)[:10] or moment.astimezone(PT).date().isoformat()
    path = _chat_path(chat_db)
    if not path.exists():
        return {"status": ledger.STATUS_SKIPPED, "model": "",
                "reason": "the Trade Mentor app has no chat store yet; nothing to review", "outputs": []}
    try:
        root = _root(ai_root)
    except Exception as exc:  # noqa: BLE001
        return {"status": ledger.STATUS_FAILED, "model": "", "reason": f"the ai_store is unavailable: {exc}",
                "outputs": []}

    grading = _grade(path, moment, veto_outcomes, permutation_history=permutation_history,
                     permutation_report=permutation_report)
    try:
        facts = day_facts(path, session, grading=grading)
    except Exception as exc:  # noqa: BLE001
        _log.debug("mentor_review facts failed.", exc_info=True)
        return {"status": ledger.STATUS_FAILED, "model": "", "reason": f"the chat store could not be read: {exc}",
                "outputs": [], "extra": {"grading": grading}}
    facts["built_utc"] = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    summary: dict[str, Any] = {
        "turns": facts["turns"], "challenges": facts["challenges"], "picks_assessed": facts["picks_assessed"],
        "remember_notes": facts["remember_notes"], "graded": grading.get("graded", 0),
        "updated": grading.get("updated", 0),
    }
    graded_note = f"{grading.get('graded', 0)} graded, {grading.get('updated', 0)} updated"
    try:
        facts_path = _publish(published_path(root, FACTS_STEM, session), facts)
    except OSError as exc:
        return {"status": ledger.STATUS_FAILED, "model": "",
                "reason": f"mentor_day_facts could not be published (the last good one is kept): {exc}",
                "outputs": [], "extra": summary}
    outputs = [str(facts_path)]

    # P15a: what the night and the app know about the day, the recurring-issue candidates and the
    # facts part of the coach brief - all deterministic, so they run with the model cut or down too.
    night = night_inputs(path, session, moment, night_paths=night_paths, mirror_builder=mirror_builder,
                         recap_paths=recap_paths)
    earlier = [payload for payload in read_published(root, COACH_STEM, limit=15)
               if _text(payload.get("session_date"))[:10] < session]
    registry = read_issue_registry(root)
    candidates = issue_candidates(path, session, night_paths=night_paths, earlier=earlier, registry=registry,
                                  recap_paths=recap_paths)
    # P18: the day's Market Journal reads + grades, and "read vs tape two sessions running" as an issue.
    try:
        reads, grades = read_rows(session, moment, reads_sources)
    except Exception:  # noqa: BLE001 - unreadable reads cost only their rows
        _log.debug("mentor_review: the reads could not be read.", exc_info=True)
        reads, grades = [], []
    if reads:
        night["reads"] = reads
    read_tape = read_tape_candidate(session, grades)
    if read_tape is not None:
        read_tape["first_seen"] = min(_text((registry.get("read_vs_tape") or {}).get("first_seen")) or session,
                                      session)
        candidates = sorted([*candidates, read_tape], key=lambda item: (-item["count"], item["first_seen"],
                                                                         item["key"]))
    # P18: every journal line and user turn of the last 30 days, counted into habits (deterministic).
    habits: list[dict[str, Any]] = []
    habits_note = ""
    try:
        from ai_jobs import mentor_habits

        first, last = mentor_habits.window(session)
        said = mentor_habits.said(path, first, last)
        journal_path = journal_db if journal_db is not None else _live_journal()
        habits = mentor_habits.find_habits(said, red_after=mentor_habits.red_after_from_journal(journal_path))
        lines = journal_rows(said, session)
        if lines:
            night["journal"] = lines
        if habits:
            night["habits"] = mentor_habits.habit_rows(habits)
        outputs.append(str(mentor_habits.publish_registry(root, session, habits, facts["built_utc"],
                                                          mentor_habits.day_facts(said, session))))
        summary["habits"] = len(habits)
    except Exception as exc:  # noqa: BLE001 - habits never cost the review; the last good registry stays
        _log.debug("mentor_review: habits could not be counted.", exc_info=True)
        habits_note = f"; habits not counted ({type(exc).__name__}: {exc})"
    # P18 D: the asks of the last session days counted into routines (deterministic); the brief's line on change.
    routine = ""
    try:
        routine = publish_routines(root, path, session, facts["built_utc"])
        outputs.append(str(Path(root) / ROUTINES_FILE))
    except Exception as exc:  # noqa: BLE001 - routines never cost the review; the last good file stays
        _log.debug("mentor_review: routines could not be counted.", exc_info=True)
        habits_note += f"; routines not counted ({type(exc).__name__}: {exc})"
    night["issues"] = candidate_rows(candidates)
    registry_note = ""
    try:
        update_issue_registry(root, session, candidates, registry, facts["built_utc"])
    except OSError as exc:
        registry_note = f"; the issue registry could not be updated (the last good one is kept): {exc}"
    counts = {key: len(value) for key, value in night.items() if key != "unread"}
    summary.update({"night_inputs": counts, "inputs": sum(counts.values()), "issues": len(candidates)})
    if night.get("unread"):
        summary["unread"] = [row["text"] for row in night["unread"]]
    try:
        from mentor_packs import hypothesis_pack

        report = hypothesis_pack.latest_report(permutation_history, permutation_report)
    except Exception:  # noqa: BLE001 - no report = no vocabulary, the digest still runs
        _log.debug("mentor_review could not read the permutation report.", exc_info=True)
        report = None
    inputs = build_inputs(path, session, facts, report=report, night=night)
    inputs_note = (f"{summary['inputs']} night input(s), {len(candidates)} issue candidate(s){registry_note}"
                   f"{habits_note}")
    fact_habits: list[dict[str, Any]] = []
    if habits:
        from ai_jobs import mentor_habits

        fact_habits = mentor_habits.fact_habits(habits)
    brief_path = published_path(root, COACH_STEM, session)
    try:
        if publish_fact_brief(root, session, facts["built_utc"], inputs, candidates, fact_habits,
                              routine) is not None:
            outputs.append(str(brief_path))
    except OSError as exc:
        inputs_note += f"; the facts-only coach brief could not be published ({exc})"
    if not ask:
        return {"status": ledger.STATUS_OK, "model": "", "outputs": outputs, "extra": summary,
                "reason": f"facts published for {session}; {graded_note}; {inputs_note}; no model asked"}
    if not (inputs["turns"] or inputs["profile_notes"] or facts["challenges"]["issued"] or night.get("journal")):
        return {"status": ledger.STATUS_OK, "model": "", "outputs": outputs, "extra": summary,
                "reason": f"no mentor conversation on {session}, so no model was loaded; {inputs_note}"}

    import ai_summary
    import requests

    if request is None:
        request = ai_summary.request_ai_summary
    try:
        model = ai_summary.local_model("medium")
    except Exception:  # noqa: BLE001 - a test's request needs no configured model
        model = ""
    post = post or requests.post

    def brief_step() -> str:
        """The second call: the coach brief, only with time left in the slot's reserve."""
        old = _read_json(brief_path)
        if (not force and old.get("worded") and old.get("inputs_hash") == inputs["inputs_hash"]
                and old.get("prompt_version") == BRIEF_PROMPT_VERSION):
            return "coach brief unchanged"
        if clock.monotonic() - started > RESERVE_MINUTES * 60 - BRIEF_MIN_SECONDS_LEFT:
            return "coach brief: no time left in the slot's reserve, facts part kept"
        try:
            kept_brief, brief_model, brief_dropped = ask_brief(request, model=model, post=post, inputs=inputs,
                                                                candidates=candidates)
        except MentorReviewRejected as exc:
            return f"coach brief rejected, facts part kept: {exc}"
        except Exception as exc:  # noqa: BLE001 - the facts part stands
            _log.debug("mentor_review could not ask for the coach brief.", exc_info=True)
            return f"coach brief: no local model answered, facts part kept: {exc}"
        worded_habits, habits_said = fact_habits, ""
        if habits:
            from ai_jobs import mentor_habits

            try:
                worded_habits, _habit_model, habit_dropped = mentor_habits.ask_habits(
                    request, model=model, post=post, habits=habits, session=session)
                habits_said = f", {len(worded_habits)} habit(s) worded ({habit_dropped} dropped)"
            except mentor_habits.HabitsRejected as exc:
                habits_said = f", habits rejected, code wording kept: {exc}"
            except Exception as exc:  # noqa: BLE001 - the code-worded habits stand
                habits_said = f", habits not worded, code wording kept: {exc}"
        try:
            _publish(brief_path, brief_payload(session, facts["built_utc"], inputs, candidates, kept=kept_brief,
                                               model=brief_model, dropped=brief_dropped, habits=worded_habits,
                                               routine=routine))
        except OSError as exc:
            return f"coach brief could not be published (the last good one is kept): {exc}"
        if str(brief_path) not in outputs:
            outputs.append(str(brief_path))
        summary.update({"brief_watch": len(kept_brief["watch"]), "brief_missing": len(kept_brief["missing"]),
                        "brief_issues": len(kept_brief["issues"]), "brief_dropped": brief_dropped})
        return (f"coach brief {len(kept_brief['watch'])} watch, {len(kept_brief['missing'])} missing, "
                f"{len(kept_brief['issues'])} issue(s){habits_said}")

    digest_path = published_path(root, DIGEST_STEM, session)
    if not force and digest_path.exists():
        old = _read_json(digest_path)
        if old.get("inputs_hash") == inputs["inputs_hash"] and old.get("prompt_version") == PROMPT_VERSION:
            note = brief_step()
            return {"status": ledger.STATUS_OK, "model": "", "outputs": [*outputs, str(digest_path)], "extra": summary,
                    "reason": f"the mentor day {session} is unchanged; no digest model was asked; {note}"}

    try:
        result = request(
            provider="local", model=model, api_key="", evidence=build_evidence(inputs),
            timeout_seconds=TIMEOUT_SECONDS,
            post=capped_post(post, model=model),
            schema=DIGEST_JSON_SCHEMA, schema_name=SCHEMA_NAME, prompt_version=PROMPT_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - the last digest stays
        _log.debug("mentor_review could not ask its model.", exc_info=True)
        return {"status": ledger.STATUS_DEGRADED, "model": "", "outputs": outputs, "extra": summary,
                "reason": f"facts published; no local model answered the mentor digest: {exc}"}
    answered = _text((result or {}).get("model")) or model
    try:
        kept, dropped = check_reply((result or {}).get("summary"), inputs)
    except MentorReviewRejected as exc:
        return {"status": ledger.STATUS_FAILED, "model": answered, "outputs": outputs, "extra": summary,
                "reason": f"the mentor digest was rejected and nothing was published: {exc}"}
    payload = {
        "schema": DIGEST_SCHEMA,
        "session_date": session,
        "built_utc": facts["built_utc"],
        "model": answered,
        "prompt_version": PROMPT_VERSION,
        "inputs_hash": inputs["inputs_hash"],
        "digest": kept["digest"],
        "open_questions": kept["open_questions"],
        "dropped": dropped,
    }
    from mentor_app import challenge

    # The night inserts rows only inside its window; a manual daytime run publishes the lookups unrecorded.
    night_store = NightChallengeStore(path) if challenge.night_owns_grading(moment) else None
    looked = look_up_hypotheses(kept["hypotheses"], report, session=session, issued_utc=facts["built_utc"],
                                night_store=night_store)
    payload |= {
        "permutation_report_asof": report.asof if report is not None else "",
        "hypotheses": looked,
        "hyp_lines": [item["line"] for item in looked],
    }
    try:
        _publish(digest_path, payload)
    except OSError as exc:
        return {"status": ledger.STATUS_FAILED, "model": answered, "outputs": outputs, "extra": summary,
                "reason": f"mentor_day_digest could not be published (the last good one is kept): {exc}"}
    summary.update({"digest_items": len(kept["digest"]), "open_questions": len(kept["open_questions"]),
                    "hypotheses": len(looked), "hypotheses_recorded": sum(1 for item in looked if item["recorded"]),
                    "dropped": dropped})
    note = brief_step()
    return {
        "status": ledger.STATUS_OK, "model": answered, "outputs": [*outputs, str(digest_path)], "extra": summary,
        "reason": (f"{len(kept['digest'])} digest item(s), {len(kept['open_questions'])} open question(s), "
                   f"{len(looked)} hypothesis lookup(s), {dropped} dropped for {session}; {graded_note}; "
                   f"{inputs_note}; {note}"),
    }


__all__ = [
    "BRIEF_JSON_SCHEMA",
    "COACH_STEM",
    "DIGEST_JSON_SCHEMA",
    "DIGEST_STEM",
    "FACTS_STEM",
    "MentorReviewRejected",
    "NightChallengeStore",
    "RESERVE_MINUTES",
    "build_inputs",
    "check_brief",
    "check_reply",
    "day_facts",
    "issue_candidates",
    "model_wanted",
    "night_inputs",
    "read_published",
    "run_mentor_review",
]
