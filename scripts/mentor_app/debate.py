"""Bull/bear debate (``/debate SYM``): two persona calls on ONE pick pack, then a scoreboard line. Qt-free.

The pick pack is built once. Two sequential structured calls go through
``ai_summary.request_ai_summary`` with ``persona_instruction`` (the base citation rules
plus one role paragraph): "bull" argues FOR the pick, "bear" AGAINST it, each returning
``{case: [{text, evidence_refs}], weakest_point: {text, evidence_refs}}``. Each reply goes
through ``citations.check_citations``: a foreign id rejects that side (shown as rejected,
never replaced), an uncited bullet is dropped. The scoreboard line is computed here, never
by the model. A clean pair is cached in ``pack_cache`` by (symbol, side, pack hash, prompt).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from mentor_packs.citations import CitationRejected, check_citations
from mentor_packs.registry import Pack, pack_from_json

SIDES = ("bull", "bear")
CACHE_NAME = "debate_card"
SCHEMA_NAME = "trade_mentor_debate"
PROMPT_VERSION = "trade_mentor_debate_v1"
MAX_OUTPUT_TOKENS = 400
EFFORT = "medium"
TIMEOUT_SECONDS = 180
FOOTER = "*Two arguments from the same evidence; you decide.*"

ROLES = {
    "bull": (
        "Your role in this debate: the BULL. Make the strongest honest case FOR this pick, using only the rows "
        "in this package. Then name the weakest point of your own case."
    ),
    "bear": (
        "Your role in this debate: the BEAR. Make the strongest honest case AGAINST this pick, using only the rows "
        "in this package. Then name the weakest point of your own case."
    ),
}

_LINE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "evidence_refs"],
    "properties": {
        "text": {"type": "string", "maxLength": 220},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case", "weakest_point"],
    "properties": {"case": {"type": "array", "maxItems": 4, "items": _LINE}, "weakest_point": _LINE},
}
TASK = (
    "Argue your side in at most 4 short 'case' bullets, then one 'weakest_point' of your own case. Every line "
    "cites the source_id of each row it uses in evidence_refs; a line with no source_id is thrown away and a "
    "source_id not listed here throws away your whole side. Say n with every rate; a row that says 'too few' "
    "is not evidence. Never size or place an order, never propose changing the plan, a detector, a score or an "
    "alert."
)


@dataclass
class SideReply:
    """One persona's checked reply. ``error`` set = this side is shown as rejected or not run."""

    side: str
    case: list[dict[str, Any]] = field(default_factory=list)
    weakest_point: dict[str, Any] | None = None
    dropped: list[dict[str, Any]] = field(default_factory=list)
    reply: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def argued(self) -> bool:
        return bool(self.case) and not self.error

    @property
    def rejected(self) -> bool:
        """The model answered but cited an id the pack never carried (or a bad shape)."""
        return self.error.startswith("rejected")

    def cited(self) -> set[str]:
        ids = {ref for row in self.case for ref in row.get("evidence_refs") or ()}
        if self.weakest_point:
            ids.update(self.weakest_point.get("evidence_refs") or ())
        return ids


@dataclass
class Debate:
    symbol: str
    side: str
    pack_hash: str
    bull: SideReply = field(default_factory=lambda: SideReply("bull"))
    bear: SideReply = field(default_factory=lambda: SideReply("bear"))
    error: str = ""
    model: str = ""
    built_utc: str = ""
    pack_json: str = ""

    @property
    def clean(self) -> bool:
        """Both sides argued: the only pair worth caching."""
        return not self.error and self.bull.argued and self.bear.argued

    def pack(self) -> Pack | None:
        return pack_from_json(self.pack_json) if self.pack_json else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Debate":
        payload = json.loads(text)
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        for side in SIDES:
            raw = known.get(side)
            if isinstance(raw, Mapping):
                known[side] = SideReply(**{k: raw[k] for k in SideReply.__dataclass_fields__ if k in raw})
        return cls(**known)


def evidence_for(pack: Pack, side: str) -> dict[str, Any]:
    """The same rows for both personas; only the role in the system text differs."""
    return {
        "task": TASK,
        "debate_side": side,
        "pack": pack.name,
        "rows": [{"source_id": str(row["id"]), "text": str(row.get("text") or "")} for row in pack.rows],
    }


def allowed_ids(pack: Pack) -> set[str]:
    from mentor_packs import pick_pack

    return set(pack.ids) | pick_pack.plan_ids(pack)


def check_side(side: str, reply: Any, pack: Pack) -> SideReply:
    """Keep cited case bullets and a cited weakest point. Raises CitationRejected for a foreign id or a bad shape."""
    if not isinstance(reply, Mapping):
        raise CitationRejected("the reply was not an object")
    allowed = allowed_ids(pack)
    out = SideReply(side=side, reply=dict(reply))
    case = check_citations({"bullets": reply.get("case")}, allowed)
    out.case = list(case.kept)
    out.dropped = [{"kind": "case", "reason": "uncited", **row} for row in case.dropped]
    weakest = reply.get("weakest_point")
    if weakest is not None:
        if not isinstance(weakest, Mapping):
            raise CitationRejected("weakest_point was not an object")
        checked = check_citations({"bullets": [weakest]}, allowed)
        if checked.kept:
            out.weakest_point = checked.kept[0]
        else:
            out.dropped.append({"kind": "weakest_point", "reason": "uncited", **dict(weakest)})
    if not out.case:
        out.error = "no cited point survived the check"
    return out


def argue(
    side: str,
    pack: Pack,
    *,
    model: str,
    endpoint: str,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
) -> SideReply:
    """One persona call. A failed call or a rejected reply is a SideReply with ``error``, never a guess."""
    import requests

    import ai_summary
    from mentor_app.assess import effort_post

    if request is None:
        request = ai_summary.request_ai_summary
    raw: Any = None
    try:
        result = request(
            provider="local", model=model, api_key="", evidence=evidence_for(pack, side),
            timeout_seconds=TIMEOUT_SECONDS,
            post=effort_post(post or requests.post, model=model, effort=EFFORT, max_tokens=MAX_OUTPUT_TOKENS),
            schema=SCHEMA, schema_name=SCHEMA_NAME, prompt_version=PROMPT_VERSION,
            endpoint=f"{endpoint.rstrip('/')}/v1",
            system_instruction=ai_summary.persona_instruction(ROLES[side]),
        )
        raw = result.get("summary")
        return check_side(side, raw, pack)
    except CitationRejected as exc:
        return SideReply(side=side, error=f"rejected: {exc}", dropped=[{"kind": "reply", "reason": str(exc)}],
                         reply=dict(raw) if isinstance(raw, Mapping) else {"raw": str(raw)[:2000]})
    except Exception as exc:  # noqa: BLE001 - a failed side is shown as failed, never replaced
        return SideReply(side=side, error=f"{type(exc).__name__}: {exc}")


def debate(
    pack: Pack,
    *,
    symbol: str,
    side: str,
    pack_hash: str,
    model: str,
    endpoint: str,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Debate:
    """Bull then bear on the same pack; Stop between the two leaves the bear side "not run: stopped"."""
    out = Debate(symbol=symbol, side=side, pack_hash=pack_hash, model=model, pack_json=pack.as_json(),
                 built_utc=now().astimezone(timezone.utc).isoformat(timespec="seconds"))
    for name in SIDES:
        if cancelled():
            setattr(out, name, SideReply(side=name, error="not run: stopped"))
            continue
        setattr(out, name, argue(name, pack, model=model, endpoint=endpoint, post=post, request=request))
    drops = {name: getattr(out, name).dropped for name in SIDES if getattr(out, name).dropped}
    if drops:
        logging.info("Trade Mentor debate %s: dropped %s", symbol, json.dumps(drops, default=str)[:1000])
    return out


# ---------------------------------------------------------------- cache
def cache_key(symbol: str, side: str, pack_hash: str) -> dict[str, str]:
    return {"symbol": str(symbol).upper(), "side": str(side or "").upper(), "hash": str(pack_hash),
            "prompt": PROMPT_VERSION}


def cached_debate(store: Any, symbol: str, side: str, pack_hash: str) -> Debate | None:
    row = store.get_pack(CACHE_NAME, cache_key(symbol, side, pack_hash))
    if not row:
        return None
    try:
        return Debate.from_json(str(row["pack_json"]))
    except (ValueError, TypeError, KeyError):
        return None


def run_debate_job(
    symbol: str,
    side: str,
    *,
    store: Any,
    build_pack: Callable[[str, str], Pack],
    pack_hash: Callable[[Pack], str],
    run: Callable[[Pack, str], Debate] | None,
) -> dict[str, Any]:
    """Build the pack once; reuse a cached clean pair, else run (when given) and cache a clean pair.

    ``run`` None = the brain is off: the pack alone. Returns ``{symbol, side, pack, hash, debate, ran}``.
    """
    pack = build_pack(symbol, side)
    digest = pack_hash(pack)
    out: dict[str, Any] = {"symbol": symbol, "side": side, "pack": pack, "hash": digest, "debate": None, "ran": False}
    found = cached_debate(store, symbol, side, digest)
    if found is not None and found.clean:
        out["debate"] = found
        return out
    if run is None:
        return out
    fresh = run(pack, digest)
    if fresh.clean:
        store.put_pack(CACHE_NAME, cache_key(symbol, side, digest), fresh.to_json(), fresh.built_utc)
    out["debate"], out["ran"] = fresh, True
    return out


# ---------------------------------------------------------------- scoreboard + card
def _cell_row(pack: Pack) -> dict[str, Any] | None:
    return next((row for row in pack.rows if row.get("kind") == "cell" and str(row.get("id", "")).endswith(":cell")),
                None)


def cell_text(pack: Pack) -> str:
    """The pack's cell row (n, LB) restated by the app; below the floor it says too few."""
    import setup_grades
    from mentor_packs.pick_pack import MIN_REPORTABLE_N

    row = _cell_row(pack)
    if row is None:
        return "cell: not in the pack"
    ref = f"[{row['id']}]"
    n = row.get("n")
    if not isinstance(n, int) or n <= 0:
        return f"cell {ref}: {row.get('text', 'no read')}"
    if n < MIN_REPORTABLE_N:
        return f"cell {ref}: n={n}, too few (floor {MIN_REPORTABLE_N})"
    wins = int(row.get("wins") or 0)
    return f"cell {ref}: n={n}, LB {setup_grades.wilson_lower_bound(wins, n):.2f}"


def scoreboard_line(result: Debate, pack: Pack) -> str:
    """Counts, shared ids and the cell: computed here; the model only argues."""
    parts = []
    for name in SIDES:
        reply = getattr(result, name)
        if reply.argued:
            parts.append(f"{name} {len(reply.case)} cited")
        else:
            parts.append(f"{name} {'rejected' if reply.rejected else 'none'}")
    shared = sorted(result.bull.cited() & result.bear.cited()) if result.bull.argued and result.bear.argued else []
    both = " ".join(f"[{ref}]" for ref in shared) or "none"
    return f"**Scoreboard:** {', '.join(parts)}; cited by both: {both}; {cell_text(pack)}"


def _cell(text: str) -> str:
    return str(text or "").replace("|", "/").replace("\n", " ").strip()


def _line(row: Mapping[str, Any]) -> str:
    refs = " ".join(f"[{ref}]" for ref in row.get("evidence_refs") or ())
    return _cell(f"{row.get('text', '')} {refs}")


def _column(reply: SideReply) -> tuple[list[str], str]:
    """(case cells, weakest-point cell) for one side; a rejected side says so and nothing else."""
    if reply.error:
        word = "rejected" if reply.rejected else "not argued"
        reason = reply.error.removeprefix("rejected: ")
        return [_cell(f"*{reply.side} case {word}: {reason}*")], ""
    weakest = f"*Weakest point:* {_line(reply.weakest_point)}" if reply.weakest_point else "*Weakest point: none cited*"
    return [_line(row) for row in reply.case], weakest


def card_markdown(result: Debate | None, pack: Pack, *, symbol: str, side: str = "", pack_hash: str = "",
                  brain_reason: str = "") -> str:
    """Two columns (bull | bear), each weakest point under its column, the scoreboard, the footer.

    No debate (brain off, a failed build): the pack's rows alone, never a made-up side.
    """
    head = f"**Debate {symbol}{' ' + side if side else ''}**"
    link = f"[show evidence](evidence:{symbol}:{pack_hash})" if pack_hash else ""
    replies = [getattr(result, name) for name in SIDES] if result is not None else []
    if result is None or result.error or not any(reply.argued or reply.rejected for reply in replies):
        if result is None:
            why = f"brain off: {brain_reason}" if brain_reason else "brain off"
        else:
            why = result.error or "; ".join(f"{reply.side}: {reply.error}" for reply in replies)
        lines = [f"{head}: no debate ({why}). The evidence:", ""]
        lines += [f"- {row.get('text', '')} [{row['id']}]" for row in pack.rows if row.get("id")]
        if link:
            lines += ["", link]
        return "\n".join(lines)
    bull_cells, bull_weak = _column(result.bull)
    bear_cells, bear_weak = _column(result.bear)
    lines = [head, "", "| Bull | Bear |", "|---|---|"]
    for index in range(max(len(bull_cells), len(bear_cells))):
        left = bull_cells[index] if index < len(bull_cells) else ""
        right = bear_cells[index] if index < len(bear_cells) else ""
        lines.append(f"| {left} | {right} |")
    if bull_weak or bear_weak:
        lines.append(f"| {bull_weak} | {bear_weak} |")
    lines += ["", scoreboard_line(result, pack)]
    uncited = sum(1 for name in SIDES for drop in getattr(result, name).dropped if drop.get("reason") == "uncited")
    if uncited:
        lines.append(f"*({uncited} uncited point{'s' if uncited != 1 else ''} dropped)*")
    lines += ["", FOOTER]
    if link:
        lines += ["", link]
    return "\n".join(lines)


def turn_tool_calls(done: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What the turn log keeps: the pack call plus both replies with their drops and errors."""
    result: Debate | None = done.get("debate")
    calls: list[dict[str, Any]] = [{"name": "pick_pack", "arguments": {"symbol": done.get("symbol"),
                                                                      "side": done.get("side")},
                                    "hash": done.get("hash")}]
    if result is not None:
        for name in SIDES:
            reply = getattr(result, name)
            calls.append({"name": f"debate_{name}", "prompt_version": PROMPT_VERSION, "effort": EFFORT,
                          "reply": reply.reply, "kept": reply.case, "weakest_point": reply.weakest_point,
                          "dropped": reply.dropped, "error": reply.error})
    return calls
