"""`mentor_review` - the night reads the Trade Mentor app's day (mentor app Phase 4).

Deterministic half (always, also with ``ask=False``, a budget cut or the model down):
open ``MENTOR_CHAT_DB_FILE`` read-only, grade open challenges with
``mentor_app.challenge.grade_open`` (model-free; only 22:00-06:00 PT, when the night
owns grading), count the day's facts and publish them as ``mentor_day_facts`` to the
ai_store. The counts also ride on the ledger row.

Model half: one call (<= 600 output tokens) over an inputs pack with ids - the facts,
the day's profile notes, the day's challenges and up to 20 recent turns. The citation
check is ``plan_review``'s rule: an id the pack does not carry rejects the reply whole;
an uncited item is dropped. Publishes ``mentor_day_digest`` (<= 5 items, <= 3 open
questions) by temp-and-rename, so a failed publish keeps the last good one.

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
import sqlite3
from contextlib import closing
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ai_jobs import ledger

_log = logging.getLogger(__name__)

PT = ZoneInfo("America/Los_Angeles")
PROMPT_VERSION = "mentor_review_v2"
SCHEMA_NAME = "tradingbot_mentor_review"
FACTS_STEM = "mentor_day_facts"
DIGEST_STEM = "mentor_day_digest"
FACTS_SCHEMA = "mentor_day_facts_v1"
DIGEST_SCHEMA = "mentor_day_digest_v1"
#: P15a: the night's product for the day coach, loaded first into the morning memory.
COACH_STEM = "mentor_coach_brief"
COACH_SCHEMA = "mentor_coach_brief_v1"
MAX_OUTPUT_TOKENS = 600
MAX_DIGEST_ITEMS = 5
MAX_OPEN_QUESTIONS = 3
MAX_HYPOTHESES = 3
MAX_ITEM_CHARS = 280
MAX_TURNS = 20
MAX_TURN_CHARS = 300
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
    "suggest an order, a size, or a change to a detector, score or alert. Say nothing "
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


def build_inputs(path: Path, session: str, facts: Mapping[str, Any], *, report: Any = None) -> dict[str, Any]:
    """Everything the model may see, with ids; ``inputs_hash`` ignores the clock."""
    notes = [
        {"id": f"note:{row['id']}", "text": _text(row.get("text"))[:MAX_TURN_CHARS]}
        for row in _day_rows(path, "profile_notes", "ts_utc", session)
    ]
    challenges = [
        {"id": f"challenge:{row['id']}", "kind": _text(row.get("kind")), "symbol": _text(row.get("symbol")),
         "text": _text(row.get("claim"))[:MAX_TURN_CHARS], "status": _text(_outcome(row).get("status"))}
        for row in _day_rows(path, "challenges", "issued_utc", session)
    ]
    turns = [
        {"id": f"turn:{row['id']}", "role": _text(row.get("role")), "text": _text(row.get("text"))[:MAX_TURN_CHARS]}
        for row in _day_rows(path, "turns", "ts_utc", session)
        if row.get("role") in ("user", "assistant")
    ][-MAX_TURNS:]
    rows = fact_rows(facts)
    report_rows, vocab = hypothesis_context(report)
    ids = [row["id"] for row in (*rows, *notes, *challenges, *turns, *report_rows)]
    body: dict[str, Any] = {
        "session_date": session,
        "facts": rows,
        "profile_notes": notes,
        "challenges": challenges,
        "turns": turns,
        "permutation_report": report_rows,
        "hypothesis_vocabulary": vocab,
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
# the slot
# ---------------------------------------------------------------------------
def capped_post(post: Callable[..., Any], *, model: str) -> Callable[..., Any]:
    """Wrap ``post``: at most MAX_OUTPUT_TOKENS, and high reasoning effort for gpt-oss tags (Qt-free)."""

    def wrapped(url: str, **kwargs: Any) -> Any:
        payload = dict(kwargs.get("json") or {})
        payload["max_tokens"] = min(int(payload.get("max_tokens") or MAX_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS)
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
        return bool(_day_rows(path, "turns", "ts_utc", session)) if session else True
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
    **_ignored: Any,
) -> dict[str, Any]:
    """One night's review of the Trade Mentor app's day. Never raises."""
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
    summary = {
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
    if not ask:
        return {"status": ledger.STATUS_OK, "model": "", "outputs": outputs, "extra": summary,
                "reason": f"facts published for {session}; {graded_note}; no model asked"}

    try:
        from mentor_packs import hypothesis_pack

        report = hypothesis_pack.latest_report(permutation_history, permutation_report)
    except Exception:  # noqa: BLE001 - no report = no vocabulary, the digest still runs
        _log.debug("mentor_review could not read the permutation report.", exc_info=True)
        report = None
    inputs = build_inputs(path, session, facts, report=report)
    if not (inputs["turns"] or inputs["profile_notes"] or inputs["challenges"]):
        return {"status": ledger.STATUS_OK, "model": "", "outputs": outputs, "extra": summary,
                "reason": f"no mentor conversation on {session}, so no model was loaded"}
    digest_path = published_path(root, DIGEST_STEM, session)
    if not force and digest_path.exists():
        try:
            old = json.loads(digest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if old.get("inputs_hash") == inputs["inputs_hash"] and old.get("prompt_version") == PROMPT_VERSION:
            return {"status": ledger.STATUS_OK, "model": "", "outputs": [*outputs, str(digest_path)], "extra": summary,
                    "reason": f"the mentor day {session} is unchanged; no model was asked"}

    import ai_summary
    import requests

    if request is None:
        request = ai_summary.request_ai_summary
    try:
        model = ai_summary.local_model("medium")
    except Exception:  # noqa: BLE001 - a test's request needs no configured model
        model = ""
    try:
        result = request(
            provider="local", model=model, api_key="", evidence=build_evidence(inputs),
            timeout_seconds=TIMEOUT_SECONDS,
            post=capped_post(post or requests.post, model=model),
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
    return {
        "status": ledger.STATUS_OK, "model": answered, "outputs": [*outputs, str(digest_path)], "extra": summary,
        "reason": (f"{len(kept['digest'])} digest item(s), {len(kept['open_questions'])} open question(s), "
                   f"{len(looked)} hypothesis lookup(s), {dropped} dropped for {session}; {graded_note}"),
    }


__all__ = [
    "DIGEST_JSON_SCHEMA",
    "DIGEST_STEM",
    "FACTS_STEM",
    "MentorReviewRejected",
    "NightChallengeStore",
    "RESERVE_MINUTES",
    "build_inputs",
    "check_reply",
    "day_facts",
    "model_wanted",
    "read_published",
    "run_mentor_review",
]
