"""Veto challenges: word the veto pack's candidate rows, record them, grade them later. Qt-free.

The pack decides WHICH vetoes are challenged (n >= 30 and a slice LB above the side
baseline); the model only words them, in one structured call over the candidate rows
(``{challenges: [{veto_id, claim, evidence_refs, n, lb}]}``). The citation check: a
foreign id rejects the whole reply; an item for a veto that is not a candidate, one that
does not cite its slice, or one whose n / lb differ from the pack's is dropped. Each
surviving challenge is one ``challenges`` row (kind ``veto``). A challenge is a note,
never a rule proposal: rules go through the plan (``plan_challenges``).

Grading is deterministic (no model): :func:`grade_open` fills each open challenge's
side returns at 1/3/5/10 sessions from the veto cohort's graded outcomes as they mature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.citations import CitationRejected, check_citations
from mentor_packs.registry import Pack, pack_from_json

PT = ZoneInfo("America/Los_Angeles")
KIND = "veto"
SCHEMA_NAME = "trade_mentor_veto_challenges"
PROMPT_VERSION = "trade_mentor_veto_challenges_v1"
CACHE_NAME = "veto_card"
MAX_OUTPUT_TOKENS = 600
EFFORT = "high"
TIMEOUT_SECONDS = 180
MORNING_AT = time(6, 45)
GRADE_HORIZONS = (1, 3, 5, 10)
HIT_HORIZON = 5
FOOTER = "*This is a note, not a rule: rules go through the plan.*"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["challenges"],
    "properties": {
        "challenges": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["veto_id", "claim", "evidence_refs", "n", "lb"],
                "properties": {
                    "veto_id": {"type": "string"},
                    "claim": {"type": "string", "maxLength": 240},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "n": {"type": "integer"},
                    "lb": {"type": "number"},
                },
            },
        },
    },
}

TASK = (
    "Each candidate below is a veto whose slice (past vetoes of the same setup, side and reason) won "
    "more often than the side's baseline. Word ONE short challenge per candidate for the trader: say "
    "what he vetoed and that vetoes like it have won, with n and the Wilson lower bound exactly as given "
    "(copy n and lb, never compute). Cite the veto id and its slice id in evidence_refs. Never suggest "
    "changing a rule, the plan, a detector, a score or an alert, and never size or place an order: a "
    "challenge is a note, not a rule."
)


@dataclass
class VetoCard:
    session: str
    pack_hash: str
    challenges: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    model: str = ""
    built_utc: str = ""
    pack_json: str = ""
    worded: bool = False

    @property
    def done(self) -> bool:
        """True when the card is final: every candidate had its chance to be worded, or there were none."""
        return self.worded and not self.error

    def pack(self) -> Pack | None:
        return pack_from_json(self.pack_json) if self.pack_json else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "VetoCard":
        payload = json.loads(text)
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


def _lb(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def candidates(pack: Pack) -> list[dict[str, Any]]:
    from mentor_packs import veto_pack

    return veto_pack.candidates(pack)


def evidence_for(pack: Pack) -> dict[str, Any]:
    """The package the model sees: the task and only the candidate rows (plus the session's weeks line)."""
    items = []
    for item in candidates(pack):
        veto, cut = item["veto"], item["slice"]
        items.append({
            "veto_id": veto["id"], "n": int(cut["n"]), "lb": _lb(cut["lb"]),
            "rows": [{"source_id": veto["id"], "text": veto["text"]}, {"source_id": cut["id"], "text": cut["text"]}],
        })
    weeks = next((row for row in pack.rows if row.get("kind") == "weeks"), None)
    return {
        "task": TASK,
        "pack": pack.name,
        "candidates": items,
        "context": [{"source_id": weeks["id"], "text": weeks["text"]}] if weeks else [],
    }


def check_reply(reply: Any, pack: Pack) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept challenges, drops). Raises CitationRejected for a malformed reply or a foreign id."""
    if not isinstance(reply, Mapping) or not isinstance(reply.get("challenges"), (list, tuple)):
        raise CitationRejected("the reply carried no challenges array")
    wanted = {item["veto"]["id"]: item for item in candidates(pack)}
    items = [dict(row) if isinstance(row, Mapping) else {"_bad": str(row)[:200]} for row in reply["challenges"]]
    # Same rule as every structured reply: a foreign id anywhere rejects it all (raises).
    check_citations(
        {"bullets": [{"text": row.get("claim"), "evidence_refs": row.get("evidence_refs") or ()} for row in items]},
        pack.ids,
    )
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in items:
        raw = row.get("evidence_refs") or ()
        refs = [str(ref).strip() for ref in ([raw] if isinstance(raw, str) else raw) if str(ref).strip()]
        claim = str(row.get("claim") or "").strip()
        if not refs or not claim:
            drops.append({"reason": "uncited", **row})
            continue
        veto_id = str(row.get("veto_id") or "").strip()
        item = wanted.get(veto_id)
        if item is None:
            drops.append({"reason": "not a candidate veto", **row})
            continue
        if veto_id in seen:
            drops.append({"reason": "a second challenge for one veto", **row})
            continue
        cut = item["slice"]
        if cut["id"] not in refs:
            drops.append({"reason": "does not cite its slice", **row})
            continue
        try:
            n_given = int(row.get("n"))
        except (TypeError, ValueError):
            n_given = None
        if n_given != int(cut["n"]) or _lb(row.get("lb")) != _lb(cut["lb"]):
            drops.append({"reason": "n or lb differ from the pack", **row})
            continue
        seen.add(veto_id)
        kept.append({
            "veto_id": veto_id, "symbol": item["veto"]["symbol"], "side": item["veto"]["side"],
            "session": item["veto"].get("session", ""), "claim": claim, "evidence_refs": refs,
            "n": int(cut["n"]), "lb": _lb(cut["lb"]),
        })
    return kept, drops


def word(
    pack: Pack,
    *,
    pack_hash: str,
    model: str,
    endpoint: str,
    post: Callable[..., Any] | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> VetoCard:
    """Word the pack's candidates. No candidate = no model call. A failed call keeps the pack, never a guess."""
    session = next((str(row["id"]).split(":")[1] for row in pack.rows if row.get("kind") == "summary"), "")
    card = VetoCard(
        session=session, pack_hash=pack_hash, model=model,
        built_utc=now().astimezone(timezone.utc).isoformat(timespec="seconds"), pack_json=pack.as_json(),
    )
    if not candidates(pack):
        card.worded = True
        return card
    if request is None:
        import ai_summary

        request = ai_summary.request_ai_summary
    import requests

    from mentor_app.assess import effort_post

    try:
        result = request(
            provider="local",
            model=model,
            api_key="",
            evidence=evidence_for(pack),
            timeout_seconds=TIMEOUT_SECONDS,
            post=effort_post(post or requests.post, model=model, effort=EFFORT, max_tokens=MAX_OUTPUT_TOKENS),
            schema=SCHEMA,
            schema_name=SCHEMA_NAME,
            prompt_version=PROMPT_VERSION,
            endpoint=f"{endpoint.rstrip('/')}/v1",
        )
        card.challenges, card.dropped = check_reply(dict(result.get("summary") or {}), pack)
        card.worded = True
    except CitationRejected as exc:
        card.error = f"reply rejected: {exc}"
        card.dropped.append({"reason": str(exc)})
    except Exception as exc:  # noqa: BLE001 - a failed wording leaves the pack's rows, never a guess
        card.error = f"{type(exc).__name__}: {exc}"
    if card.dropped:
        logging.info("Trade Mentor vetoes %s: dropped %s", session, json.dumps(card.dropped, default=str)[:1000])
    return card


def record(store: Any, card: VetoCard) -> int:
    """One ``challenges`` row per worded challenge (a veto is challenged once). Returns rows written."""
    written = 0
    for item in card.challenges:
        seed = {"status": "open", "session": item.get("session") or card.session, "side": item.get("side"),
                "n": item.get("n"), "lb": item.get("lb")}
        if store.add_challenge(
            item["veto_id"], kind=KIND, symbol=item.get("symbol", ""), claim=item["claim"],
            evidence_ids=item.get("evidence_refs") or (), issued_utc=card.built_utc, outcome=seed,
        ):
            written += 1
    return written


def _slice_verdict(text: str) -> str:
    """The slice row's verdict part (after the head), for the compact card line."""
    return text.split("): ", 1)[1] if "): " in text else text


def card_markdown(card: VetoCard) -> str:
    """The /vetoes card: every veto with its slice line, then the worded challenges or why there are none."""
    pack = card.pack()
    rows = list(pack.rows) if pack is not None else []
    summary = next((row for row in rows if row.get("kind") == "summary"), None)
    weeks = next((row for row in rows if row.get("kind") == "weeks"), None)
    lines = [f"**Vetoes, session {card.session or '?'}**", ""]
    if summary:
        lines.append(f"{summary['text']} [{summary['id']}]")
    if weeks:
        lines.append(f"{weeks['text']} [{weeks['id']}]")
    lines.append("")
    by_id = {str(row.get("id")): row for row in rows}
    for row in rows:
        if row.get("kind") not in ("veto", "pass"):
            continue
        cut = by_id.get(f"{row['id']}:slice")
        lines.append(f"- {row['text']} [{row['id']}]")
        if cut:
            lines.append(f"  - {_slice_verdict(str(cut['text']))} [{cut['id']}]")
    lines.append("")
    if card.challenges:
        lines.append("**Challenges**")
        for item in card.challenges:
            refs = " ".join(f"[{ref}]" for ref in item.get("evidence_refs") or ())
            lines.append(f"- {item['symbol']}: {item['claim']} (n={item['n']}, LB={item['lb']:.2f}) {refs}".rstrip())
    elif card.error:
        count = len(candidates(pack)) if pack is not None else 0
        lines.append(f"**Challenges**: {count} candidate(s) not worded ({card.error}). The slice lines above stand.")
    else:
        measured = [row for row in rows if row.get("kind") == "slice" and "n" in row]
        thin = sum(1 for row in measured if int(row.get("n") or 0) < 30)
        lines.append(
            f"**No challenge**: n too small ({thin} slice{'s' if thin != 1 else ''} under 30) / "
            f"no edge over baseline ({len(measured) - thin})."
        )
    dropped = sum(1 for drop in card.dropped if drop.get("reason"))
    if dropped and card.challenges:
        lines.append(f"- *({dropped} unsupported challenge{'s' if dropped != 1 else ''} dropped)*")
    lines += ["", FOOTER]
    return "\n".join(lines)


def inbox_line(card: VetoCard) -> str:
    count = len(card.challenges)
    return f"Yesterday's vetoes: {count} challenge{'s' if count != 1 else ''}"


# ---------------------------------------------------------------- the 06:45 card
@dataclass
class VetoSchedule:
    """The morning card is due once a PT day from 06:45; marked when it is queued."""

    last_day: Any = None

    def due(self, now: datetime) -> bool:
        local = (now if now.tzinfo else now.astimezone()).astimezone(PT)
        return local.time() >= MORNING_AT and self.last_day != local.date()

    def mark(self, now: datetime) -> None:
        self.last_day = (now if now.tzinfo else now.astimezone()).astimezone(PT).date()


def is_session_day(now: datetime) -> bool:
    """True when today (New York) is a trading session; a calendar that cannot answer says weekday."""
    today = (now if now.tzinfo else now.astimezone()).astimezone(ZoneInfo("America/New_York")).date()
    try:
        from market_calendar import is_session

        return bool(is_session(today))
    except Exception:  # noqa: BLE001
        return today.weekday() < 5


# ---------------------------------------------------------------- grading (no model)
def _veto_outcomes_path() -> Path:
    import project_paths

    return Path(project_paths.VETO_COHORT_OUTCOMES_FILE)


def grade_open(store: Any, now: datetime, *, veto_outcomes: Path | str | None = None) -> int:
    """Fill each open veto challenge's matured side returns; ``graded_utc`` once all four horizons are in.

    A challenge with no cohort row yet stays open, its outcome saying why. Returns rows updated.
    """
    import annotations_reader

    moment = now if now.tzinfo else now.astimezone()
    today = moment.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    returns = annotations_reader.veto_forward_returns(veto_outcomes or _veto_outcomes_path())
    updated = 0
    for row in store.challenges(kind=KIND, open_only=True):
        try:
            outcome = json.loads(row.get("outcome_json") or "{}")
        except ValueError:
            outcome = {}
        parts = str(row["id"]).split(":")
        session = str(outcome.get("session") or (parts[1] if len(parts) > 2 else ""))
        side = str(outcome.get("side") or "")
        symbol = str(row.get("symbol") or (parts[2] if len(parts) > 2 else "")).upper()
        found = returns.get((session, symbol, side))
        if found is None:
            new = {**outcome, "status": "open", "reason": "no veto cohort row yet"}
        else:
            matured = {str(h): round(value, 6) for h, (when, value) in found.items() if when <= today}
            new = {**outcome, "returns": matured}
            new.pop("reason", None)
            if str(HIT_HORIZON) in matured:
                new["hit"] = matured[str(HIT_HORIZON)] > 0  # the vetoed trade would have won
            new["status"] = "graded" if len(matured) == len(GRADE_HORIZONS) else "maturing"
            if not matured:
                new["reason"] = "no horizon matured yet"
        if new == outcome:
            continue
        graded = moment.astimezone(timezone.utc).isoformat(timespec="seconds") if new.get("status") == "graded" else None
        if store.update_challenge(row["id"], outcome=new, graded_utc=graded):
            updated += 1
    return updated


def scorecard(store: Any, *, floor: int | None = None) -> str:
    """Issued / graded veto challenges and the 5-session hit rate with n; "too few" under the floor."""
    if floor is None:
        from evidence_stats import MIN_REPORTABLE_N

        floor = int(MIN_REPORTABLE_N)
    rows = store.challenges(kind=KIND)
    issued = len(rows)
    graded = sum(1 for row in rows if row.get("graded_utc"))
    hits = []
    for row in rows:
        try:
            outcome = json.loads(row.get("outcome_json") or "{}")
        except ValueError:
            continue
        if "hit" in outcome:
            hits.append(bool(outcome["hit"]))
    head = f"**Scorecard: veto challenges**\n\n- issued {issued}, fully graded {graded}"
    if len(hits) < floor:
        return head + f"\n- {HIT_HORIZON}-session hit rate: too few (n={len(hits)}, floor {floor})"
    return head + f"\n- {HIT_HORIZON}-session hit rate {sum(hits) / len(hits):.0%} (n={len(hits)}, floor {floor})"
