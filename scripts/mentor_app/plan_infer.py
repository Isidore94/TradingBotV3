"""Plan inference: the trader's own chat words become ``[ai YYYY-MM-DD]`` plan lines, live. Qt-free.

After a chat turn, one structured side call to the chat model reads the trader's recent turns
(with store ids) and the plan's lines, and returns ``{"ops": [{op, plan_id, section, text,
turn_ids}]}``. The citation check: a turn id or plan id the input does not carry rejects the
whole reply; an op that cites no turn, cites no NEW turn, has a bad section or bad text, or
targets the trader's own line is dropped. Only the trader's turns are evidence.

Writes go through ``trading_plan``'s AI-line functions (its lock, its history), on the window's
one IO thread, which also owns the day's count: at most MAX_WRITES_PER_DAY add/update writes per
PT day. Nothing here reaches a detector, score, alert, watchlist, Focus, queue or review policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import trading_plan
from mentor_app import brain, rule_gate

#: The recent trader turns one inference reads.
TURN_LIMIT = 8
#: The most ops one reply may carry; the rest are dropped.
MAX_OPS_PER_REPLY = 3
#: Add/update plan writes per PT day (the trader's cap, 2026-09-30).
MAX_WRITES_PER_DAY = 5
MAX_OUTPUT_TOKENS = 600
TIMEOUT_SECONDS = 120
MAX_TURN_CHARS = 600
QUOTE_CHARS = 100
OPS = ("add", "update", "retire")
SECTIONS = tuple(heading for heading in trading_plan.HEADINGS if heading != trading_plan.DECISIONS)
WRITES_KEY = "plan_infer:writes:{day}"
CAPPED_KEY = "plan_infer:capped:{day}"
SOURCE_KEY = "plan_src:{digest}"
#: The rule text an id last showed (in /plan or an Added/Updated line); /drop must still find it there.
SHOWN_KEY = "plan_shown:{plan_id}"
#: Never equal to a rule: a /drop of an id the app never showed is refused as moved.
_NEVER_SHOWN = "\x00never shown"

DROP_NOT_AN_OBJECT = "not_an_object"
DROP_BAD_OP = "bad_op"
DROP_NO_TURN = "no_turn"
DROP_NO_NEW_TURN = "no_new_turn"
DROP_BAD_SECTION = "bad_section"
DROP_BAD_TEXT = "bad_text"
DROP_NOT_AI_LINE = "not_ai_line"
DROP_TOO_MANY = "too_many"

INSTRUCTIONS = (
    "You keep ONE trader's trading plan in step with what he says in chat. Only his own words in "
    "trader_turns are evidence; the Mentor's replies are not. Return {\"ops\": []} unless a turn marked "
    "new clearly states a rule he follows or has decided to follow: a limit, a time, a goal, a setup he "
    "trades, a risk rule, or something he is testing. A question, a hypothetical, a maybe, a feeling, a "
    "one-off plan for one trade, or the Mentor's own suggestion is not a rule. Say nothing when unsure.\n"
    "- add: a new rule. section is one of allowed_sections; text is the rule in his words, one short line "
    "of at most 140 characters, with no dates or ids.\n"
    "- update: he changed or clarified a rule on a line where ai is true (for example \"no, I meant 1 PM\"): "
    "copy its plan_id exactly and give the new text.\n"
    "- retire: he dropped a rule on a line where ai is true: copy its plan_id exactly.\n"
    "Never target a line where ai is false: those lines are his own. Never add a rule already in plan_lines "
    "or in dropped_rules unless he states it again. Every op cites turn_ids copied exactly from "
    "trader_turns, at least one of them new. At most three ops."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OPS)},
                    "plan_id": {"type": "string"},
                    "section": {"type": "string"},
                    "text": {"type": "string"},
                    "turn_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["op", "turn_ids"],
            },
        }
    },
    "required": ["ops"],
}

ADDED = "Added to your plan: {text} [{plan_id}] — /drop {plan_id} to undo"
UPDATED = "Updated in your plan: {text} [{plan_id}] — /drop {plan_id} to undo"
DROPPED = "Dropped from your plan: {text} (a Decisions line keeps it)"
NOT_SAVED = "Not saved to your plan: {text} ({error})"
CAPPED = "I changed your plan {cap} times today, the daily cap; I will not add more until tomorrow."
TRADERS_OWN = "{plan_id} is your own line, so I will not drop it. Edit trading_plan.md to change it."
MOVED = "That rule moved, so I did not drop {plan_id}. Run /plan and try again."
REMEMBER_CAPPED = ("Kept as a note, but not added to your plan today: I already changed your plan {cap} times "
                   "today, the daily cap.")


class InferRejected(ValueError):
    """The reply cited a turn or plan line the input did not carry: nothing is written."""


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def turn_ref(value: Any) -> str:
    """``12``, ``"12"`` or ``"turn:12"`` -> ``"turn:12"``; anything else as given."""
    text = _clean(value)
    if isinstance(value, int) or text.isdigit():
        return f"turn:{int(text)}"
    return text


# ---------------------------------------------------------------- inputs and the call
def build_inputs(turns: Iterable[Mapping[str, Any]], parsed: Mapping[str, Any], *, after: int) -> dict[str, Any]:
    """What one inference may see: the trader's last TURN_LIMIT turns (``new`` = id > after) and the plan."""
    mine = [row for row in turns if str(row.get("role")) == "user" and _clean(row.get("text"))][-TURN_LIMIT:]
    trader_turns = [
        {"id": f"turn:{int(row['id'])}", "text": _clean(row.get("text"))[:MAX_TURN_CHARS], "new": int(row["id"]) > int(after)}
        for row in mine
    ]
    plan_lines = [
        {"id": str(row["id"]), "section": str(row["section"]), "text": str(row.get("rule") or row["text"]),
         "ai": bool(row.get("ai"))}
        for row in trading_plan.plan_lines(parsed)
        if row.get("section") != trading_plan.DECISIONS
    ]
    dropped = [
        _clean(str(row.get("text"))[len(trading_plan.AI_DROPPED_PREFIX):])
        for row in (parsed or {}).get("decisions") or ()
        if str(row.get("text") or "").startswith(trading_plan.AI_DROPPED_PREFIX)
    ]
    return {
        "trader_turns": trader_turns,
        "plan_lines": plan_lines,
        "dropped_rules": dropped,
        "allowed_sections": list(SECTIONS),
    }


def request(
    inputs: Mapping[str, Any],
    *,
    endpoint: str,
    model: str,
    post: brain.Post = brain.default_post,
    keep_alive: Any = -1,
    num_ctx: int | None = None,
) -> Any:
    """One non-streamed structured call; the parsed JSON reply. Raises on any failure."""
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": json.dumps(dict(inputs), sort_keys=True)},
    ]
    payload = brain.chat_payload(model, messages, stream=False, keep_alive=keep_alive, num_ctx=num_ctx,
                                 fmt=SCHEMA, max_tokens=MAX_OUTPUT_TOKENS,
                                 think=False if brain.thinks_unless_told(model) else None)
    reply = post(f"{endpoint.rstrip('/')}/api/chat", payload, TIMEOUT_SECONDS)
    return brain.json_reply(reply)


# ---------------------------------------------------------------- the citation check
def check_reply(reply: Any, inputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The ops to write and the drop counts. A foreign turn id or plan id raises InferRejected."""
    if not isinstance(reply, Mapping) or not isinstance(reply.get("ops"), list):
        raise InferRejected("the reply is not {ops: [...]}")
    turns = {row["id"]: row for row in inputs.get("trader_turns") or ()}
    lines = {row["id"]: row for row in inputs.get("plan_lines") or ()}
    items = list(reply["ops"])
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        for ref in item.get("turn_ids") or ():
            if turn_ref(ref) not in turns:
                raise InferRejected(f"op {index} cited {ref!r}, which the input does not carry")
        plan_id = _clean(item.get("plan_id"))
        if plan_id and plan_id not in lines:
            raise InferRejected(f"op {index} named plan line {plan_id!r}, which the input does not carry")
    drops: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for item in items:
        reason, op = _check_one(item, turns, lines)
        if not reason and len(out) >= MAX_OPS_PER_REPLY:
            reason = DROP_TOO_MANY
        if reason:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        out.append(op)
    return out, drops


def _check_one(item: Any, turns: Mapping[str, Any], lines: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(item, Mapping):
        return DROP_NOT_AN_OBJECT, {}
    kind = _clean(item.get("op")).lower()
    if kind not in OPS:
        return DROP_BAD_OP, {}
    refs = list(dict.fromkeys(turn_ref(ref) for ref in item.get("turn_ids") or () if _clean(ref)))
    if not refs:
        return DROP_NO_TURN, {}
    if not any(turns[ref]["new"] for ref in refs):
        return DROP_NO_NEW_TURN, {}
    section = ""
    if _clean(item.get("section")):
        section = next((name for name in SECTIONS if trading_plan.slug(name) == trading_plan.slug(item["section"])), "")
        if not section:
            return DROP_BAD_SECTION, {}
    op: dict[str, Any] = {"op": kind, "plan_id": "", "section": section, "text": "", "turn_ids": refs,
                          "quote": turns[refs[0]]["text"][:QUOTE_CHARS], "expect_text": None}
    if kind == "add":
        if not section:
            return DROP_BAD_SECTION, {}
    else:
        line = lines.get(_clean(item.get("plan_id")))
        if line is None or not line.get("ai"):
            return DROP_NOT_AI_LINE, {}
        op.update(plan_id=line["id"], section=line["section"], expect_text=line["text"])
    if kind != "retire":
        try:
            op["text"] = trading_plan.clean_ai_text(item.get("text"))
        except trading_plan.PlanLineRefused:
            return DROP_BAD_TEXT, {}
    return "", op


# ---------------------------------------------------------------- writing (the window's IO thread only)
def _source_key(text: str) -> str:
    return SOURCE_KEY.format(digest=hashlib.sha1(trading_plan.normalize_rule(text).encode("utf-8")).hexdigest()[:16])


def apply(
    ops: Sequence[Mapping[str, Any]],
    *,
    store: Any,
    day: str,
    now: datetime | None = None,
    path: Path | None = None,
    say_duplicates: bool = False,
) -> list[str]:
    """Write the checked ops; the chat lines to show. A failed plan write is one loud line."""
    shown: list[str] = []
    key = WRITES_KEY.format(day=day)
    written = int(store.get_state(key) or 0)
    for op in ops:
        kind, text = op["op"], str(op.get("text") or "")
        if kind in ("add", "update") and written >= MAX_WRITES_PER_DAY:
            logging.info("Trade Mentor: plan %s skipped, %d writes today is the cap: %s", kind, written, text)
            if not store.get_state(CAPPED_KEY.format(day=day)):
                store.set_state(CAPPED_KEY.format(day=day), "1")
                shown.append(CAPPED.format(cap=MAX_WRITES_PER_DAY))
            continue
        try:
            if kind == "add":
                result = trading_plan.add_ai_line(text, section=op["section"], now=now, day=day, path=path)
            elif kind == "update":
                result = trading_plan.update_ai_line(op["plan_id"], text, expect_text=op.get("expect_text"),
                                                     now=now, day=day, path=path)
            else:
                result = trading_plan.retire_ai_line(op["plan_id"], expect_text=op.get("expect_text"),
                                                     now=now, day=day, path=path)
        except trading_plan.PlanLineRefused as exc:
            logging.info("Trade Mentor: plan %s refused: %s", kind, exc)
            continue
        except trading_plan.PlanWriteError as exc:
            logging.warning("Trade Mentor: plan %s not saved: %s", kind, exc)
            shown.append(NOT_SAVED.format(text=text or op.get("plan_id") or "", error=exc))
            break
        if not result.get("changed"):
            if result.get("duplicate") and say_duplicates:
                shown.append(f"Already in your plan: [{result['duplicate']}].")
            continue
        if kind == "retire":
            shown.append(DROPPED.format(text=result.get("text") or ""))
            continue
        written += 1
        store.set_state(key, str(written))
        store.set_state(_source_key(text), json.dumps(
            {"turn_ids": list(op.get("turn_ids") or ()), "quote": str(op.get("quote") or "")[:QUOTE_CHARS]}))
        template = ADDED if kind == "add" else UPDATED
        plan_id = result.get("plan_id") or op.get("plan_id")
        store.set_state(SHOWN_KEY.format(plan_id=plan_id), text)
        shown.append(template.format(text=text, plan_id=plan_id))
    return shown


def rule_from_note(note: Any) -> str:
    """``rule: I stop after two losses`` -> ``I stop after two losses``; "" for any other note."""
    text = _clean(note)
    return text[5:].strip() if text.lower().startswith("rule:") else ""


def remember_rule(note: Any, *, store: Any, day: str, now: datetime | None = None, path: Path | None = None) -> list[str]:
    """A ``/remember rule: ...`` note is also an AI plan line (same cap and dedupe); its chat lines."""
    words = rule_from_note(note)
    if not words:
        return []
    try:
        text = trading_plan.clean_ai_text(words)
    except trading_plan.PlanLineRefused as exc:
        return [f"Kept as a note only, not in your plan: {exc}."]
    if int(store.get_state(WRITES_KEY.format(day=day)) or 0) >= MAX_WRITES_PER_DAY:
        return [REMEMBER_CAPPED.format(cap=MAX_WRITES_PER_DAY)]
    op = {"op": "add", "section": "Rules", "text": text, "turn_ids": [], "quote": _clean(note)[:QUOTE_CHARS]}
    return apply([op], store=store, day=day, now=now, path=path, say_duplicates=True)


def drop(plan_id: str, *, store: Any, day: str, now: datetime | None = None, path: Path | None = None) -> str:
    """``/drop <plan_id>``: retire the AI line that id last showed; the trader's own lines are refused.

    The line must still say the rule the id showed in /plan or an Added line (checked under the plan's
    lock), so a stale id never drops a different rule.
    """
    shown = store.get_state(SHOWN_KEY.format(plan_id=plan_id)) or _NEVER_SHOWN
    try:
        result = trading_plan.retire_ai_line(plan_id, expect_text=shown, now=now, day=day, path=path)
    except trading_plan.PlanLineRefused as exc:
        kind = getattr(exc, "kind", "")
        if kind == "trader":
            return TRADERS_OWN.format(plan_id=plan_id)
        if kind == "moved":
            return MOVED.format(plan_id=plan_id)
        return f"There is no AI line {plan_id}. `/plan` shows the ids."
    except trading_plan.PlanWriteError as exc:
        return NOT_SAVED.format(text=f"dropping {plan_id}", error=exc)
    store.set_state(SHOWN_KEY.format(plan_id=plan_id), "")
    return DROPPED.format(text=result.get("text") or "")


def listing(*, store: Any, path: Path | None = None) -> str:
    """``/plan``: every line by section with its id; an AI line is flagged with the words it came from."""
    plan = trading_plan.read_plan(create=False, snapshot=False, path=path)
    if plan.get("error"):
        return f"Your plan could not be read: {plan['error']}"
    rows = trading_plan.plan_lines(plan.get("parsed") or {})
    if not rows:
        return "Your plan has no lines yet. Tell me your rules in plain words and I will add them."
    out = ["**Your plan** (lines ending in [ai date] I inferred from your words; `/drop <id>` undoes one)"]
    for heading in trading_plan.HEADINGS:
        mine = [row for row in rows if row["section"] == heading]
        if not mine:
            continue
        out += ["", f"**{heading}**", ""]
        for row in mine:
            line = f"- [{row['id']}] {row['text']}"
            if row.get("ai"):
                store.set_state(SHOWN_KEY.format(plan_id=row["id"]), row["rule"])
                try:
                    source = json.loads(store.get_state(_source_key(row["rule"])) or "{}")
                except ValueError:
                    source = {}
                quote = str(source.get("quote") or "")
                line += f' *(AI, from: "{quote}")*' if quote else " *(AI)*"
            out.append(line)
    return "\n".join(out)


def infer(
    turns: Sequence[Mapping[str, Any]],
    *,
    after: int,
    endpoint: str,
    model: str,
    post: brain.Post = brain.default_post,
    keep_alive: Any = -1,
    num_ctx: int | None = None,
    path: Path | None = None,
    read_plan: Callable[..., Mapping[str, Any]] | None = None,
    gate: Callable[[str], float | None] | None = None,
    gate_mode: str = "off",
    on_gate: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Worker thread: read the plan, ask the model, check the reply. [] when anything fails (logged).

    With a ``gate`` in mode ``on``/``shadow`` the new turns are scored first; ``on`` skips the call
    when the best score is under the cut, ``shadow`` never skips. ``on_gate`` gets one record.
    """
    plan = (read_plan or trading_plan.read_plan)(create=False, snapshot=False, path=path)
    if plan.get("error"):
        logging.info("Trade Mentor: plan inference skipped, the plan could not be read (%s)", plan["error"])
        return []
    inputs = build_inputs(turns, plan.get("parsed") or {}, after=after)
    if not any(row["new"] for row in inputs["trader_turns"]):
        return []
    record = _gate(inputs, gate, gate_mode)
    if record is not None and record["skipped"]:
        logging.info("Trade Mentor: plan inference skipped by the rule gate (score %.2f)", record["score"])
        _record(record, on_gate)
        return []
    ops: list[dict[str, Any]] = []
    try:
        reply = request(inputs, endpoint=endpoint, model=model, post=post, keep_alive=keep_alive, num_ctx=num_ctx)
        ops, drops = check_reply(reply, inputs)
    except Exception as exc:  # noqa: BLE001 - model down, tunnel down or a bad reply: nothing is written
        logging.info("Trade Mentor: plan inference skipped (%s: %s)", type(exc).__name__, exc)
        if record is not None:
            _record(record, on_gate)
        return []
    if drops:
        logging.info("Trade Mentor: plan inference dropped %s", drops)
    if record is not None:
        _record({**record, "ops": len(ops)}, on_gate)
    return ops


def _gate(inputs: Mapping[str, Any], gate: Callable[[str], float | None] | None, gate_mode: str) -> dict[str, Any] | None:
    """Score the new turns; the record, or None when the gate is off. Only ever a skip, never an op."""
    if gate is None or gate_mode not in ("shadow", "on"):
        return None
    new = [row for row in inputs["trader_turns"] if row["new"]]
    scores = []
    for row in new:
        try:
            scores.append(gate(row["text"]))
        except Exception:  # noqa: BLE001 - a broken gate is "no answer"
            scores.append(None)
    top = None if any(value is None for value in scores) else max(scores)
    if top is None:
        logging.info("Trade Mentor: rule gate gave no answer; plan inference runs as before")
    would_skip = top is not None and top < rule_gate.CUT
    return {"turn_ids": [row["id"] for row in new], "score": top, "mode": gate_mode, "cut": rule_gate.CUT,
            "would_skip": would_skip, "skipped": would_skip and gate_mode == "on", "ops": None}


def _record(record: dict[str, Any], on_gate: Callable[[dict[str, Any]], Any] | None) -> None:
    score = "none" if record["score"] is None else f"{record['score']:.2f}"
    logging.info("Trade Mentor: rule gate (%s) score %s on %s: %s, plan inference returned %s ops",
                 record["mode"], score, ",".join(record["turn_ids"]),
                 "skip" if record["would_skip"] else "ask", "no" if record["ops"] is None else record["ops"])
    if on_gate is not None:
        try:
            on_gate(record)
        except Exception:  # noqa: BLE001 - a lost record never blocks plan inference
            logging.warning("Trade Mentor: the rule gate record was not kept", exc_info=True)
