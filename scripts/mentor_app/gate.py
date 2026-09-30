"""The pre-trade gate (``/check``): one structured call over a gate pack, cited, advice only. Qt-free.

The model returns ``{verdict: go | wait | breaks a rule, bullets, rule_flags}`` through
``ai_summary.request_ai_summary``; ``assess.check_reply`` keeps cited bullets (a foreign
id rejects the reply) and rule flags that name a plan line the pack carried. The
verdict is advice: the card never sizes, orders or writes the journal. A narrated card
is one ``challenges`` row (kind ``gate``) that :func:`grade_open` grades later from the
journal: hit = a trade on that symbol/side opened within 1 session closed with R > 0.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_app import assess
from mentor_packs.registry import Pack

KIND = "gate"
VERDICTS = ("go", "wait", "breaks a rule")
SCHEMA_NAME = "trade_mentor_gate"
PROMPT_VERSION = "trade_mentor_gate_v1"
MAX_OUTPUT_TOKENS = 600
FOOTER = "*advice only; you click, it never orders*"
#: A gate with no matching trade is graded "not taken" after this many sessions.
NOT_TAKEN_AFTER_SESSIONS = 5
#: A trade counts for the gate when it opened within this many sessions of the check.
OPEN_WITHIN_SESSIONS = 1
ET = ZoneInfo("America/New_York")

SCHEMA: dict[str, Any] = {**assess.SCHEMA, "properties": {
    **assess.SCHEMA["properties"], "verdict": {"type": "string", "enum": list(VERDICTS)}}}

TASK = (
    "The trader is about to take the trade in the request row. Check it in at most 6 short bullets: "
    "the dollar risk against his setting, the pick and tape rows, his open book (same symbol, same "
    "industry) and his plan lines. Every bullet cites the source_id of each row it uses in evidence_refs; "
    "a bullet with no source_id is thrown away. Say n and the floor when you use a win rate. "
    "rule_flags: only for plan lines listed (their plan_id), breaks=true when this trade breaks it. "
    "Never size, never place an order, never suggest changing the plan, a detector, a score or an alert. "
    "A headline row is a title and a link only: cite its id, never say more than its title, and never "
    "mention news that is not a row here. "
    "verdict: 'go', 'wait' or 'breaks a rule'."
)


# ---------------------------------------------------------------- /check parsing
@dataclass(frozen=True)
class CheckRequest:
    side: str
    symbol: str
    size: float | None = None
    stop: float | None = None
    entry: float | None = None


_NUM = r"\$?(\d+(?:\.\d+)?|\.\d+)"


def parse_check(text: str) -> CheckRequest | None:
    """``short NVDA 400 stop 3.20 entry 3.05`` (any order after side and symbol; size is the bare number)."""
    words = str(text or "").replace(",", " ").split()
    if len(words) < 2:
        return None
    side = {"LONG": "LONG", "SHORT": "SHORT", "BUY": "LONG", "SELL": "SHORT"}.get(words[0].upper(), "")
    symbol = words[1].upper()
    if not side:  # tolerate `/check NVDA short`
        side = {"LONG": "LONG", "SHORT": "SHORT"}.get(symbol, "")
        symbol = words[0].upper() if side else ""
    if not side or not symbol or not symbol.replace(".", "").replace("-", "").isalnum() or symbol[0].isdigit():
        return None
    values: dict[str, float] = {}
    rest = words[2:]
    index = 0
    while index < len(rest):
        word = rest[index].lower().rstrip(":")
        if word in ("stop", "entry", "size", "sz", "@", "at") and index + 1 < len(rest):
            match = re.fullmatch(_NUM, rest[index + 1])
            if match is None:
                return None
            key = {"sz": "size", "@": "entry", "at": "entry"}.get(word, word)
            values[key] = float(match.group(1))
            index += 2
            continue
        match = re.fullmatch(_NUM + r"(?:sh|shares)?", word)
        if match is None or "size" in values:
            return None
        values["size"] = float(match.group(1))
        index += 1
    return CheckRequest(side, symbol, values.get("size"), values.get("stop"), values.get("entry"))


def request_hash(request: CheckRequest, pack_hash: str) -> str:
    """Never shared across different requests: the request row is in the key."""
    from mentor_packs.gate_pack import request_key

    key = request_key(request.side, request.symbol, request.size, request.stop, request.entry)
    return hashlib.sha256(f"{key}|{pack_hash}".encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- narration
def evidence_for(pack: Pack) -> dict[str, Any]:
    return {**assess.evidence_for(pack), "task": TASK}


def narrate(
    pack: Pack,
    *,
    symbol: str,
    pack_hash: str,
    model: str,
    endpoint: str,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> assess.Assessment:
    """One structured call over the gate pack; the same checks as a pick assessment."""
    import requests

    from mentor_packs.citations import CitationRejected

    out = assess.Assessment(symbol=symbol, pack_hash=pack_hash, model=model, effort=assess.EFFORT_LIVE,
                            built_utc=now().astimezone(timezone.utc).isoformat(timespec="seconds"),
                            pack_json=pack.as_json())
    if request is None:
        import ai_summary

        request = ai_summary.request_ai_summary
    try:
        result = request(
            provider="local", model=model, api_key="", evidence=evidence_for(pack),
            timeout_seconds=assess.TIMEOUT_SECONDS,
            post=assess.effort_post(post or requests.post, model=model, effort=assess.EFFORT_LIVE,
                                    max_tokens=MAX_OUTPUT_TOKENS),
            schema=SCHEMA, schema_name=SCHEMA_NAME, prompt_version=PROMPT_VERSION,
            endpoint=f"{endpoint.rstrip('/')}/v1",
        )
        reply = dict(result.get("summary") or {})
        out.bullets, out.rule_flags, out.dropped = assess.check_reply(reply, pack)
        verdict = str(reply.get("verdict") or "")
        out.verdict = verdict if verdict in VERDICTS else ""
        if not out.verdict:
            out.error = f"unknown verdict {verdict!r}"
        elif not out.bullets:
            out.error = "no cited bullet survived the check"
    except CitationRejected as exc:
        out.error = f"reply rejected: {exc}"
        out.dropped.append({"kind": "reply", "reason": str(exc)})
    except Exception as exc:  # noqa: BLE001 - a failed narration leaves the pack, never a guess
        out.error = f"{type(exc).__name__}: {exc}"
    return out


def card_markdown(assessment: assess.Assessment, request: CheckRequest, pack: Pack | None = None) -> str:
    """The /check card. No verdict: the pack's rows are shown instead. Always the advice-only footer."""
    head = f"**Check {request.side} {request.symbol}**"
    if not assessment.narrated:
        lines = [f"{head}: no verdict ({assessment.error or 'no narration'})", ""]
        for row in (pack.rows if pack is not None else ()):
            if row.get("kind") != "asof":
                lines.append(f"- [{row['id']}] {row.get('text', '')}")
        return "\n".join(lines + ["", FOOTER])
    lines = [f"{head}: **{assessment.verdict}**", ""]
    card_pack = pack if pack is not None else assessment.pack()
    for bullet in assessment.bullets:
        lines.append(assess.bullet_line(bullet, card_pack))
    for flag in assessment.rule_flags:
        word = "breaks" if flag.get("breaks") else "keeps"
        lines.append(f"- Plan {word} [{flag['plan_id']}]: {flag.get('text', '')}".rstrip(": "))
    uncited = sum(1 for drop in assessment.dropped if drop.get("reason") == "uncited")
    if uncited:
        lines.append(f"- *({uncited} uncited point{'s' if uncited != 1 else ''} dropped)*")
    return "\n".join(lines + ["", FOOTER])


def record(store: Any, assessment: assess.Assessment, request: CheckRequest, digest: str) -> bool:
    """One ``challenges`` row for a narrated gate; a card with no verdict is not a claim."""
    if not assessment.narrated:
        return False
    first = assessment.bullets[0].get("text", "") if assessment.bullets else ""
    refs: list[str] = []
    for bullet in assessment.bullets:
        refs.extend(ref for ref in bullet.get("evidence_refs") or () if ref not in refs)
    issued = assessment.built_utc
    session = datetime.fromisoformat(issued).astimezone(ET).date().isoformat()
    seed = {"status": "open", "session": session, "side": request.side, "entry": request.entry,
            "stop": request.stop, "size": request.size, "verdict": assessment.verdict}
    return bool(store.add_challenge(
        f"gate:{session}:{request.symbol}:{digest}", kind=KIND, symbol=request.symbol,
        claim=f"{assessment.verdict}: {first}".strip(), evidence_ids=refs, issued_utc=issued, outcome=seed,
    ))


# ---------------------------------------------------------------- grading (no model)
def _sessions_between(start: date, end: date) -> int:
    """Trading sessions after ``start`` up to and including ``end`` (weekdays when the calendar cannot say)."""
    from datetime import timedelta

    try:
        from market_calendar import is_session
    except Exception:  # noqa: BLE001
        def is_session(day: date) -> bool:
            return day.weekday() < 5
    count, day = 0, start
    while day < end:
        day += timedelta(days=1)
        try:
            count += 1 if is_session(day) else 0
        except Exception:  # noqa: BLE001
            count += 1 if day.weekday() < 5 else 0
    return count


def read_trades(path: Path | str, symbol: str, side: str) -> list[dict[str, Any]]:
    """Journal trades for one symbol/side with their planned stop (``mode=ro``)."""
    db = Path(path)
    if not db.exists():
        return []
    conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_annotations'").fetchone()
        stop = "a.planned_stop" if has else "NULL"
        join = "LEFT JOIN trade_annotations a ON a.trade_id = t.trade_id" if has else ""
        sql = (f"SELECT t.trade_id, t.status, t.opened_at, t.closed_at, t.direction, t.quantity_closed, "
               f"t.average_entry_price, t.average_exit_price, {stop} AS planned_stop FROM trades t {join} "
               "WHERE UPPER(t.symbol) = ? AND UPPER(t.direction) = ? ORDER BY t.opened_at")
        return [{k: row[k] for k in row.keys()} for row in conn.execute(sql, (symbol.upper(), side.upper()))]
    finally:
        conn.close()


def _trade_r(trade: Mapping[str, Any], side: str) -> float | None:
    try:
        entry, exit_, stop = (float(trade[k]) for k in ("average_entry_price", "average_exit_price", "planned_stop"))
    except (TypeError, ValueError, KeyError):
        return None
    risk = abs(entry - stop)
    if not risk:
        return None
    return ((exit_ - entry) if side == "LONG" else (entry - exit_)) / risk


def grade_open(store: Any, now: datetime, *, journal: Path | str | None = None) -> int:
    """Grade each open gate from the journal. Returns rows updated.

    A trade on the symbol/side opened within 1 session of the check: open -> waits; closed
    -> hit = R > 0 (R from its planned stop; no stop = R unknown, no hit field). No trade
    after 5 sessions -> graded "not taken".
    """
    if journal is None:
        import project_paths

        journal = project_paths.JOURNAL_DB_FILE
    moment = now if now.tzinfo else now.astimezone()
    today = moment.astimezone(ET).date()
    updated = 0
    for row in store.challenges(kind=KIND, open_only=True):
        try:
            outcome = json.loads(row.get("outcome_json") or "{}")
        except ValueError:
            outcome = {}
        side = str(outcome.get("side") or "").upper()
        try:
            session = date.fromisoformat(str(outcome.get("session") or ""))
        except ValueError:
            continue
        symbol = str(row.get("symbol") or "").upper()
        try:
            issued = datetime.fromisoformat(str(row.get("issued_utc") or ""))
            issued = issued if issued.tzinfo else issued.replace(tzinfo=timezone.utc)
        except ValueError:
            issued = datetime.combine(session, datetime.min.time(), tzinfo=ET)
        taken = None
        for trade in read_trades(journal, symbol, side):
            try:
                opened = datetime.fromisoformat(str(trade["opened_at"]))
            except ValueError:
                continue
            opened = opened if opened.tzinfo else opened.replace(tzinfo=ET)
            opened_day = opened.astimezone(ET).date()
            # Only a trade opened at or after the check, on its day or the next session, answers it.
            if opened >= issued and _sessions_between(session, opened_day) <= OPEN_WITHIN_SESSIONS:
                taken = trade
                break
        new = dict(outcome)
        graded = None
        if taken is None:
            if _sessions_between(session, today) >= NOT_TAKEN_AFTER_SESSIONS:
                new.update(status="graded", result="not taken")
                graded = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
            else:
                new.update(status="open", reason="no trade yet")
        elif str(taken.get("status") or "").upper() != "CLOSED":
            new.update(status="open", trade_id=taken["trade_id"], reason="trade still open")
        else:
            r_value = _trade_r(taken, side)
            new.update(status="graded", result="taken", trade_id=taken["trade_id"])
            new.pop("reason", None)
            if r_value is None:
                new["r"] = None
            else:
                new["r"] = round(r_value, 4)
                new["hit"] = r_value > 0
            graded = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
        if new == outcome:
            continue
        if store.update_challenge(row["id"], outcome=new, graded_utc=graded):
            updated += 1
    return updated
