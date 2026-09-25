"""The night's challenges to the trading plan, and the trader's answers (P1-7 7b).

Two append-only JSONL files with different writers:

* `PLAN_CHALLENGES_FILE` - the night's half, written only by
  `ai_jobs.plan_review`: one ``challenge`` row per new challenge and one
  ``night`` row per asked night.
* `PLAN_CHALLENGE_ANSWERS_FILE` - the trader's half, written only by
  :func:`answer_challenge` from the Mentor card. Accept also appends a dated
  line under the plan's Decisions (which snapshots the plan); reject keeps the
  reason.

A challenge nobody answers expires :data:`EXPIRY_DAYS` days after it was
written. Expiry is worked out on read and never written.

Evidence only: nothing here detects, scores, ranks, gates or alerts, and no
nightly job may call :func:`answer_challenge`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_log = logging.getLogger(__name__)

SCHEMA = "plan_challenge_v1"
ANSWER_SCHEMA = "plan_challenge_answer_v1"
ROW_CHALLENGE = "challenge"
ROW_NIGHT = "night"

EXPIRY_DAYS = 7

STATUS_OPEN = "open"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_EXPIRED = "expired"

#: The Mentor clicks. Any other answer state files a reject with that state as the reason.
ACCEPT = "accept"
REJECT_OPTIONS = ("reject_too_few", "reject_wrong_evidence", "reject_disagree")

#: The Mentor kind and the key its answer is filed under.
MENTOR_KIND = "plan_challenge"
ANSWER_KEY = "decision"


class PlanChallengeError(RuntimeError):
    """An answer that could not be filed."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    else:
        text = _text(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        return None
    return moment


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo is not None else moment.astimezone()


def challenges_path() -> Path:
    import project_paths

    return Path(project_paths.PLAN_CHALLENGES_FILE)


def answers_path() -> Path:
    import project_paths

    return Path(project_paths.PLAN_CHALLENGE_ANSWERS_FILE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every object row. A torn or foreign line costs that line only. Never creates the file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _append(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Append-only, one JSON object per line, flushed to disk. Raises OSError."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def mint_challenge_id(plan_line_text: Any, evidence: Any, text: Any) -> str:
    """One id per (plan line words, evidence id, challenge words)."""
    basis = "\n".join(" ".join(_text(part).lower().split()) for part in (plan_line_text, evidence, text))
    return "pc-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def read_challenges(path: Path | None = None) -> list[dict[str, Any]]:
    """Every challenge row, first sighting of each id wins."""
    seen: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path if path is not None else challenges_path()):
        if _text(row.get("kind")) != ROW_CHALLENGE:
            continue
        key = _text(row.get("challenge_id"))
        if key and key not in seen:
            seen[key] = row
    return list(seen.values())


def read_nights(path: Path | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in _read_jsonl(path if path is not None else challenges_path())
        if _text(row.get("kind")) == ROW_NIGHT
    ]


def read_answers(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The trader's answer per challenge id; the first answer stands."""
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path if path is not None else answers_path()):
        key = _text(row.get("challenge_id"))
        if key and key not in out and _text(row.get("decision")):
            out[key] = row
    return out


def status_of(challenge: Mapping[str, Any], answer: Mapping[str, Any] | None, now: datetime) -> str:
    """open, accepted, rejected or expired. An unreadable creation time never expires."""
    decision = _text((answer or {}).get("decision"))
    if decision in (STATUS_ACCEPTED, STATUS_REJECTED):
        return decision
    created = _aware(challenge.get("created_at"))
    if created is not None and _now(now) - created >= timedelta(days=EXPIRY_DAYS):
        return STATUS_EXPIRED
    return STATUS_OPEN


def challenge_status(challenge_id: Any, *, now: datetime | None = None,
                     challenges: Iterable[Mapping[str, Any]] | None = None,
                     answers: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """The status of one challenge; reads the trader's ``decision``. ``""`` when unknown."""
    rows = list(challenges) if challenges is not None else read_challenges()
    replies = dict(answers) if answers is not None else read_answers()
    wanted = _text(challenge_id)
    for row in rows:
        if _text(row.get("challenge_id")) == wanted:
            reply = replies.get(wanted) or {}
            if _text(reply.get("decision")):
                return _text(reply["decision"])
            return status_of(row, None, _now(now))
    return ""


def with_status(now: datetime | None = None, *, challenges_file: Path | None = None,
                answers_file: Path | None = None) -> list[dict[str, Any]]:
    """Every challenge with its status and the trader's answer fields, oldest first."""
    moment = _now(now)
    replies = read_answers(answers_file)
    out = []
    for row in read_challenges(challenges_file):
        reply = replies.get(_text(row.get("challenge_id"))) or {}
        out.append(
            {
                **row,
                "status": status_of(row, reply, moment),
                "decision_reason": _text(reply.get("reason")),
                "answered_at": _text(reply.get("answered_at")),
            }
        )
    return out


def open_challenges(now: datetime | None = None) -> list[dict[str, Any]]:
    return [row for row in with_status(now) if row["status"] == STATUS_OPEN]


def closed_keys(now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """``{"plan_challenge:<id>": {"answered_at": ...}}`` for every closed challenge.

    The Mentor host merges this into its ``answered`` lane, so a carried
    question that was answered or expired is never asked again.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in with_status(now):
        if row["status"] == STATUS_OPEN:
            continue
        stamp = row.get("answered_at") or row.get("created_at") or ""
        out[f"{MENTOR_KIND}:{_text(row.get('challenge_id'))}"] = {"answered_at": _text(stamp)}
    return out


def mentor_prompt(row: Mapping[str, Any]) -> str:
    return (
        f"Your plan says: \"{_text(row.get('plan_line_text'))}\". "
        f"The night AI: {_text(row.get('text'))} "
        f"(evidence {_text(row.get('evidence'))}). Accept adds it to Decisions."
    )


def answer_challenge(
    challenge_id: Any,
    state: Any,
    *,
    now: datetime | None = None,
    plan_path: Path | None = None,
    challenges_file: Path | None = None,
    answers_file: Path | None = None,
) -> dict[str, Any]:
    """File the trader's answer. Accept writes the plan; anything else is a reject with its reason.

    Raises :class:`PlanChallengeError` when nothing could be filed. The desk's
    Mentor card is the only caller.
    """
    import trading_plan

    moment = _now(now)
    wanted = _text(challenge_id)
    chosen = _text(state)
    if not chosen:
        raise PlanChallengeError("no answer was chosen")
    rows = {_text(row.get("challenge_id")): row for row in read_challenges(challenges_file)}
    challenge = rows.get(wanted)
    if challenge is None:
        raise PlanChallengeError(f"no challenge {wanted!r} was found")
    replies = read_answers(answers_file)
    status = status_of(challenge, replies.get(wanted), moment)
    if status != STATUS_OPEN:
        raise PlanChallengeError(f"challenge {wanted} is already {status}")
    decision = STATUS_ACCEPTED if chosen == ACCEPT else STATUS_REJECTED
    plan_line = ""
    if decision == STATUS_ACCEPTED:
        plan_line = (
            f"Accepted the night AI's challenge to \"{_text(challenge.get('plan_line_text'))}\": "
            f"{_text(challenge.get('text'))} ({_text(challenge.get('evidence'))})"
        )
        try:
            trading_plan.append_decision(plan_line, now=moment, path=plan_path)
        except trading_plan.PlanWriteError as exc:
            raise PlanChallengeError(str(exc)) from exc
    row = {
        "schema": ANSWER_SCHEMA,
        "challenge_id": wanted,
        "decision": decision,
        "reason": "" if decision == STATUS_ACCEPTED else chosen,
        "answered_at": moment.isoformat(),
        "plan_line_written": plan_line,
    }
    try:
        _append([row], answers_file if answers_file is not None else answers_path())
    except OSError as exc:
        raise PlanChallengeError(f"the answer was not saved: {exc}") from exc
    return row


def week_rows(end: Any, *, days: int = 7, now: datetime | None = None,
              as_of: datetime | None = None, challenges_file: Path | None = None,
              answers_file: Path | None = None) -> list[dict[str, Any]]:
    """Challenges written for sessions in the `days` calendar days ending `end`.

    With `as_of`, rows and answers written after that moment are hidden and
    status is worked out at `as_of`.
    """
    last = end if isinstance(end, date) else date.fromisoformat(_text(end)[:10])
    first = last - timedelta(days=max(1, int(days)) - 1)
    cutoff = as_of
    moment = cutoff or _now(now)
    replies = {
        key: reply
        for key, reply in read_answers(answers_file).items()
        if cutoff is None or ((_aware(reply.get("answered_at")) or moment) <= cutoff)
    }
    out = []
    for row in read_challenges(challenges_file):
        session = _text(row.get("session_date"))[:10]
        try:
            day = date.fromisoformat(session)
        except ValueError:
            continue
        if not (first <= day <= last):
            continue
        created = _aware(row.get("created_at"))
        if cutoff is not None and (created is None or created > cutoff):
            continue
        reply = replies.get(_text(row.get("challenge_id"))) or {}
        out.append(
            {
                **row,
                "status": status_of(row, reply, moment),
                "decision_reason": _text(reply.get("reason")),
                "answered_at": _text(reply.get("answered_at")),
            }
        )
    return out


__all__ = [
    "ACCEPT",
    "ANSWER_KEY",
    "EXPIRY_DAYS",
    "MENTOR_KIND",
    "PlanChallengeError",
    "REJECT_OPTIONS",
    "STATUS_ACCEPTED",
    "STATUS_EXPIRED",
    "STATUS_OPEN",
    "STATUS_REJECTED",
    "answer_challenge",
    "challenge_status",
    "closed_keys",
    "mentor_prompt",
    "mint_challenge_id",
    "open_challenges",
    "read_answers",
    "read_challenges",
    "read_nights",
    "status_of",
    "week_rows",
    "with_status",
]
