"""Setup keys narration (WISHLIST P1-4 / 4d): three cited sentences per family.

A Saturday slot. It hands the local model the permutation report as FACTS
ONLY - each fact a short line with an id - and asks for at most three
sentences per family, each citing fact ids of that family. The pattern is
`improvement_ideas`:

* The JSON schema is a grammar hint; :func:`check_narration` re-checks every
  bound against the input after the answer comes back and rejects the answer
  WHOLE on any fabricated citation or foreign family.
* Nothing to narrate -> no model load (a SKIPPED row, not a failure).
* An unchanged report -> no second ask (``force`` re-spends only that skip).
* The output file is replaced only by a checked answer; a failure leaves the
  last good narration in place.

Shadow only: nothing reads the sentences for a score, a filter or an alert.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import project_paths
from ai_jobs import ledger

_log = logging.getLogger(__name__)

PROMPT_VERSION = "setup_keys_narration_v1"
SCHEMA = "setup_keys_narration_v1"
SCHEMA_NAME = "tradingbot_setup_keys_narration"
MAX_SENTENCES_PER_FAMILY = 3
MAX_SENTENCE_CHARS = 240
MAX_CITES_PER_SENTENCE = 4
#: Families one night may carry; the rest are COUNTED and said, never silently dropped.
MAX_FAMILIES = 16
MAX_KEYS_PER_FAMILY = 3
TIMEOUT_SECONDS = 900
RESERVE_MINUTES = 10.0

EVIDENCE_KEYS = ("package_id", "evidence_hash", "instructions", "report_generated_at", "families",
                 "families_left_out", "allowed_family_ids")

INSTRUCTIONS = (
    "For each family below write at most three short sentences that say what the facts say: "
    "which facet key held up on the hold-out, how its win rate compares with the family baseline, "
    "and how many episodes and sessions stand behind it - or that no key was found. Every sentence "
    "must cite at least one fact id copied exactly from that family's facts. Do not calculate a new "
    "statistic, do not recommend a trade, and do not name a symbol. Say nothing rather than saying "
    "something the facts do not carry."
)

NARRATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sentences"],
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["family_id", "text", "cites"],
                "properties": {
                    "family_id": {"type": "string"},
                    "text": {"type": "string", "maxLength": MAX_SENTENCE_CHARS},
                    "cites": {"type": "array", "maxItems": MAX_CITES_PER_SENTENCE,
                              "items": {"type": "string", "maxLength": 200}},
                },
            },
        }
    },
}


class NarrationRejected(ValueError):
    """The answer broke a bound; nothing is stored."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "unmeasured"


def _stats(block: Mapping[str, Any] | None) -> str:
    block = block or {}
    return (f"win rate {_pct(block.get('win_rate'))}, low bound {_pct(block.get('wilson_lb'))}, "
            f"n={block.get('n', 0)} over {block.get('sessions', 0)} sessions")


def report_path() -> Path:
    return Path(project_paths.SETUP_PERMUTATION_REPORT_FILE)


def output_path() -> Path:
    return Path(project_paths.SETUP_KEYS_NARRATION_FILE)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_inputs(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Facts per family, each with an id. Keys found first, then the largest baselines."""
    families = []
    for population, pop in sorted(((report or {}).get("populations") or {}).items()):
        for horizon, block in sorted((pop.get("horizons") or {}).items(), key=lambda item: int(item[0])):
            for name, family in sorted((block.get("families") or {}).items()):
                family_id = f"{population}|h{horizon}|{name}"
                verdict = _text(family.get("verdict"))
                facts = [
                    {"id": f"{family_id}|verdict", "text": f"verdict: {verdict.replace('_', ' ')}"},
                    {"id": f"{family_id}|baseline",
                     "text": f"baseline before the hold-out: {_stats(family.get('baseline'))}"},
                    {"id": f"{family_id}|holdout_baseline",
                     "text": f"baseline on the last 20 sessions: {_stats(family.get('holdout_baseline'))}"},
                ]
                for key in list(family.get("keys") or ())[:MAX_KEYS_PER_FAMILY]:
                    rank = key.get("rank")
                    facts.append({
                        "id": f"{family_id}|key{rank}",
                        "text": (f"key #{rank} ({key.get('label')}): {_stats(key.get('selection'))}, "
                                 f"lift {key.get('lift_pp')} points; hold-out {_stats(key.get('holdout'))}"),
                    })
                families.append({
                    "family_id": family_id, "population": population, "horizon": horizon,
                    "family": name, "verdict": verdict, "facts": facts,
                    "_order": (0 if family.get("keys") else 1,
                               -int((family.get("baseline") or {}).get("n") or 0)),
                })
    families.sort(key=lambda item: item["_order"])
    kept, left_out = families[:MAX_FAMILIES], max(0, len(families) - MAX_FAMILIES)
    for family in kept:
        family.pop("_order", None)
    digest = hashlib.sha256(json.dumps(kept, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "report_generated_at": _text((report or {}).get("generated_at")),
        "families": kept,
        "families_left_out": left_out,
        "allowed_family_ids": [family["family_id"] for family in kept],
        "inputs_hash": digest,
    }


def build_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    body = {name: inputs.get(name) for name in EVIDENCE_KEYS if name in inputs}
    body["package_id"] = f"setup-keys-narration:{_text(inputs.get('inputs_hash'))[:16]}"
    body["evidence_hash"] = _text(inputs.get("inputs_hash"))
    body["instructions"] = INSTRUCTIONS
    return body


def _validate(payload: Any) -> dict[str, Any]:
    import ai_summary

    body = ai_summary.validate_structured_output(payload, NARRATION_JSON_SCHEMA, name="setup_keys")
    item_schema = json.loads(json.dumps(NARRATION_JSON_SCHEMA["properties"]["sentences"]["items"]))
    for spec in item_schema["properties"].values():
        spec.pop("maxLength", None)
    body["sentences"] = [
        ai_summary.validate_structured_output(row, item_schema, name=f"setup_keys.sentences[{index}]")
        for index, row in enumerate(list(body.get("sentences") or ()))
    ]
    return body


def check_narration(body: Any, inputs: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every bound, against THIS input. Returns ``{family_id: [sentence]}`` or raises."""
    if not isinstance(body, Mapping) or not isinstance(body.get("sentences"), (list, tuple)):
        raise NarrationRejected("the answer carried no sentences array")
    facts = {family["family_id"]: {fact["id"] for fact in family["facts"]} for family in inputs.get("families") or ()}
    out: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(body["sentences"]):
        if not isinstance(row, Mapping):
            raise NarrationRejected(f"sentence {index} was not an object")
        family_id = _text(row.get("family_id"))
        if family_id not in facts:
            raise NarrationRejected(f"sentence {index} names family {family_id!r}, which tonight does not carry")
        text = _text(row.get("text"))
        if not text or len(text) > MAX_SENTENCE_CHARS:
            raise NarrationRejected(f"sentence {index} is empty or longer than {MAX_SENTENCE_CHARS} characters")
        cites = [_text(cite) for cite in row.get("cites") or () if _text(cite)]
        if not cites or len(cites) > MAX_CITES_PER_SENTENCE:
            raise NarrationRejected(f"sentence {index} cites {len(cites)} facts; 1-{MAX_CITES_PER_SENTENCE} allowed")
        for cite in cites:
            if cite not in facts[family_id]:
                raise NarrationRejected(f"sentence {index} cited {cite!r}, which is not one of that family's facts")
        out.setdefault(family_id, []).append({"text": text, "cites": cites})
        if len(out[family_id]) > MAX_SENTENCES_PER_FAMILY:
            raise NarrationRejected(f"family {family_id!r} got more than {MAX_SENTENCES_PER_FAMILY} sentences")
    return out


def _write(payload: Mapping[str, Any]) -> Path:
    target = output_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temp, target)
    return target


def run_setup_keys_narration(
    *,
    session_date: str = "",
    now: datetime | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    force: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    """One Saturday's narration. Never raises."""
    report = _read_json(report_path())
    if not isinstance(report, Mapping):
        return {"status": ledger.STATUS_SKIPPED, "model": "", "outputs": [],
                "reason": "no setup-keys report to narrate; no model was loaded"}
    inputs = build_inputs(report)
    if not inputs["families"]:
        return {"status": ledger.STATUS_SKIPPED, "model": "", "outputs": [],
                "reason": "the setup-keys report holds no family; no model was loaded"}
    previous = _read_json(output_path())
    if not force and isinstance(previous, Mapping) and previous.get("inputs_hash") == inputs["inputs_hash"]:
        return {"status": ledger.STATUS_OK, "model": "", "outputs": [str(output_path())],
                "reason": "the setup-keys report is unchanged; no model was asked"}

    import ai_summary

    model = ""
    try:
        model = ai_summary.local_model("medium")
    except Exception:  # noqa: BLE001 - a test's request needs no configured model
        model = ""
    ask = request or ai_summary.request_ai_summary
    try:
        result = ask(provider="local", model=model, api_key="", evidence=build_evidence(inputs),
                     timeout_seconds=TIMEOUT_SECONDS, schema=NARRATION_JSON_SCHEMA, schema_name=SCHEMA_NAME,
                     prompt_version=PROMPT_VERSION)
    except Exception as exc:  # noqa: BLE001 - the last good narration stands
        _log.debug("The setup-keys narration could not ask its model.", exc_info=True)
        return {"status": ledger.STATUS_DEGRADED, "model": "", "outputs": [],
                "reason": f"no local model answered the setup-keys narration: {exc}"}
    answered = _text((result or {}).get("model")) or model
    try:
        sentences = check_narration(_validate((result or {}).get("summary")), inputs)
    except Exception as exc:  # noqa: BLE001 - a breach rejects the answer WHOLE
        return {"status": ledger.STATUS_FAILED, "model": "", "outputs": [],
                "reason": f"the setup-keys narration was rejected and nothing was stored: {exc}"}
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    payload = {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "written_at": stamp,
        "session_date": _text(session_date)[:10],
        "model": answered,
        "inputs_hash": inputs["inputs_hash"],
        "report_generated_at": inputs["report_generated_at"],
        "families": [
            {"family_id": family["family_id"], "verdict": family["verdict"],
             "sentences": sentences.get(family["family_id"], []), "facts": family["facts"]}
            for family in inputs["families"]
        ],
        "families_left_out": inputs["families_left_out"],
        "note": "Shadow only: nothing ranks, filters or alerts on these sentences.",
    }
    try:
        path = _write(payload)
    except OSError as exc:
        return {"status": ledger.STATUS_FAILED, "model": answered, "outputs": [],
                "reason": f"the setup-keys narration could not be written: {exc}"}
    narrated = sum(1 for family in payload["families"] if family["sentences"])
    return {"status": ledger.STATUS_OK, "model": answered, "outputs": [str(path)],
            "reason": f"{narrated} of {len(payload['families'])} famil(ies) narrated"}


__all__ = [
    "INSTRUCTIONS",
    "MAX_SENTENCES_PER_FAMILY",
    "NARRATION_JSON_SCHEMA",
    "NarrationRejected",
    "PROMPT_VERSION",
    "RESERVE_MINUTES",
    "build_evidence",
    "build_inputs",
    "check_narration",
    "run_setup_keys_narration",
]
