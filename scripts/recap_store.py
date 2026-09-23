"""Day Recap coach: the ONE writer for the trader's recap inputs.

Six record kinds, append-only, in `project_paths.DAY_RECAP_EVENTS_FILE`:
`card_answer`, `lesson`, `rule`, `rule_check`, `clue`, `environment_verdict`.
Every row has `id`, `kind`, `session_date`, `recorded_at` (tz-aware), `schema`
and `supersedes`. A correction is a NEW row that supersedes the old one; readers
fold superseded rows away and nothing is ever rewritten.

A card answer for a Trade Mentor question (a `mentor_subject` is handed in) goes
through `mentor_questions.record_answer` into the Mentor's own store instead.

A failed write raises `RecapWriteError`: these are the trader's own words, so the
page must say "not saved". No trade or journal row is touched here.
Evidence only: nothing reads this file to detect, score, rank, gate or alert.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "day_recap_event_v1"

KIND_CARD_ANSWER = "card_answer"
KIND_LESSON = "lesson"
KIND_RULE = "rule"
KIND_RULE_CHECK = "rule_check"
KIND_CLUE = "clue"
KIND_ENVIRONMENT_VERDICT = "environment_verdict"
KINDS = (
    KIND_CARD_ANSWER,
    KIND_LESSON,
    KIND_RULE,
    KIND_RULE_CHECK,
    KIND_CLUE,
    KIND_ENVIRONMENT_VERDICT,
)

#: The Walk's card kinds.
CARD_KINDS = ("trade", "miss", "good_pass", "call", "environment", "ai_pattern", "lesson")

#: What a card's subject may name.
SUBJECT_KEYS = ("trade_id", "symbol", "side", "call_id", "pick_id", "idea_id", "clue_id")

RULE_TAGS = (
    "hold_winners",
    "wait_for_confirmation",
    "respect_stop",
    "size_down_in_chop",
    "no_trade_first_15m",
    "one_trade_at_a_time",
    "only_a_plus_setups",
    "other",
)

RULE_CHECK_ANSWERS = ("yes", "partly", "no")

CLUE_TAGS = (
    "volume_dry_up",
    "volume_surge",
    "vwap_reclaim",
    "vwap_reject",
    "level_break",
    "level_hold",
    "rs_vs_spy",
    "sector_move",
    "news",
    "gap",
    "trendline_break",
    "sma_reclaim",
    "other",
)

CLUE_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "D1", "W1")

#: The auto environment's own labels (`bounce_bot_lib.legacy.MARKET_ENVIRONMENTS`,
#: pinned by a test without importing the bot).
ENVIRONMENT_LABELS = (
    "bearish_strong",
    "bearish_weak",
    "bullish_strong",
    "bullish_weak",
    "neutral_chop",
)
VERDICT_AGREE = "agree"

SHORT_TEXT_MAX = 280
TEXT_MAX = 2000
MAX_ROW_BYTES = 16 * 1024


class RecapError(ValueError):
    """The record itself is invalid. Nothing was written."""


class RecapWriteError(RuntimeError):
    """A valid record did not reach disk. The trader must be told."""


# ---------------------------------------------------------------------------
# small cleaners
# ---------------------------------------------------------------------------
def _default_path() -> Path:
    import project_paths

    return Path(project_paths.DAY_RECAP_EVENTS_FILE)


def _text(value: Any, *, limit: int = TEXT_MAX, field: str = "text") -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise RecapError(f"{field} is {len(text)} characters; the limit is {limit}")
    return text


def _session(value: Any) -> str:
    text = str(value or "").strip()[:10]
    try:
        day = date.fromisoformat(text)
    except ValueError as exc:
        raise RecapError(f"session_date {value!r} is not YYYY-MM-DD") from exc
    import market_calendar

    try:
        is_session = market_calendar.is_session(day)
    except Exception as exc:  # noqa: BLE001 - outside the calendar's range
        raise RecapError(f"session_date {text} is outside the calendar: {exc}") from exc
    if not is_session:
        raise RecapError(f"{text} is not an exchange session")
    return text


def _aware(value: Any, *, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise RecapError(f"{field} {value!r} is not an ISO time") from exc
    if not isinstance(value, datetime):
        raise RecapError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise RecapError(f"{field} must carry a timezone")
    return value


def _choice(value: Any, allowed: Iterable[str], *, field: str, upper: bool = False) -> str:
    text = str(value or "").strip()
    text = text.upper() if upper else text.lower()
    if text not in tuple(allowed):
        raise RecapError(f"{field} {value!r} is not one of {tuple(allowed)}")
    return text


def _after_close(session: str, moment: datetime) -> bool:
    import market_calendar

    return moment >= market_calendar.session_close(date.fromisoformat(session))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def load_records(*, path: Path | None = None) -> list[dict[str, Any]]:
    """Every valid row in append order. A torn or foreign line is skipped."""
    target = Path(path) if path is not None else _default_path()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("schema") == SCHEMA and row.get("kind") in KINDS:
            rows.append(row)
    return rows


def _superseded_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("supersedes") or "") for row in rows if row.get("supersedes")}


def effective(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows no later row supersedes, in append order."""
    rows = [dict(row) for row in rows]
    gone = _superseded_ids(rows)
    return [row for row in rows if str(row.get("id") or "") not in gone]


def records_for(
    session_date: Any,
    kinds: Iterable[str] | None = None,
    *,
    path: Path | None = None,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """One session's records, optionally narrowed to `kinds`, oldest first."""
    session = str(session_date or "").strip()[:10]
    wanted = set(kinds) if kinds is not None else None
    rows = load_records(path=path)
    if not include_superseded:
        rows = effective(rows)
    return [
        row for row in rows
        if row.get("session_date") == session and (wanted is None or row.get("kind") in wanted)
    ]


def latest_rule_before(session_date: Any, *, path: Path | None = None) -> dict[str, Any] | None:
    """The newest effective rule written for an EARLIER session, or None."""
    session = str(session_date or "").strip()[:10]
    rules = [
        row for row in effective(load_records(path=path))
        if row.get("kind") == KIND_RULE and str(row.get("session_date") or "") < session
    ]
    if not rules:
        return None
    return max(rules, key=lambda row: (str(row.get("session_date")), str(row.get("recorded_at"))))


def _checks_by_session(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row.get("kind") != KIND_RULE_CHECK:
            continue
        session = str(row.get("session_date") or "")
        stamp = str(row.get("recorded_at") or "")
        if session not in out or stamp >= out[session][0]:
            out[session] = (stamp, str(row.get("answer") or ""))
    return {session: answer for session, (_stamp, answer) in out.items()}


def rule_streak(until: Any, *, path: Path | None = None) -> int:
    """Consecutive sessions the rule was kept ("yes"), ending at the last
    checked session on or before `until`. A partly, a no or an unchecked
    session ends the run."""
    import market_calendar

    limit = str(until or "").strip()[:10]
    checks = _checks_by_session(effective(load_records(path=path)))
    checked = sorted(session for session in checks if session <= limit)
    if not checked:
        return 0
    cursor = date.fromisoformat(checked[-1])
    streak = 0
    while checks.get(cursor.isoformat()) == "yes":
        streak += 1
        cursor = market_calendar.previous_session(cursor)
    return streak


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _check_supersedes(kind: str, session: str, supersedes: str, path: Path) -> None:
    if not supersedes:
        return
    rows = load_records(path=path)
    target = next((row for row in rows if row.get("id") == supersedes), None)
    if target is None:
        raise RecapError(f"supersedes {supersedes!r} names no recap record")
    if target.get("kind") != kind or target.get("session_date") != session:
        raise RecapError("a correction must be the same kind and session as the row it replaces")
    if supersedes in _superseded_ids(rows):
        raise RecapError(f"{supersedes!r} was already corrected; correct the newest row")


def _tail_is_torn(target: Path) -> bool:
    try:
        if target.stat().st_size == 0:
            return False
    except FileNotFoundError:
        return False
    with target.open("rb") as probe:
        probe.seek(-1, os.SEEK_END)
        return probe.read(1) != b"\n"


def _append(row: Mapping[str, Any], path: Path) -> None:
    encoded = (json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_ROW_BYTES:
        raise RecapError(f"row is {len(encoded)} bytes; the cap is {MAX_ROW_BYTES}")
    from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with local_writer_lock(lock_key_for_path(path), timeout_seconds=2.0):
            if _tail_is_torn(path):
                encoded = b"\n" + encoded
            with path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, LocalLockUnavailable) as exc:
        raise RecapWriteError(f"the recap record was not saved: {exc}") from exc


def _write(
    kind: str,
    session_date: Any,
    fields: Mapping[str, Any],
    *,
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    session = _session(session_date)
    moment = _aware(now if now is not None else datetime.now().astimezone(), field="recorded_at")
    target = Path(path) if path is not None else _default_path()
    replaced = str(supersedes or "").strip()
    _check_supersedes(kind, session, replaced, target)
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "id": f"rc-{uuid.uuid4().hex}",
        "kind": kind,
        "session_date": session,
        "recorded_at": moment.isoformat(),
        "recorded_after_close": _after_close(session, moment),
        "supersedes": replaced,
        **dict(fields),
    }
    _append(row, target)
    return row


def record_card_answer(
    *,
    session_date: Any,
    card_id: Any,
    card_kind: Any,
    subject: Mapping[str, Any],
    option: Any,
    text: Any = "",
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
    mentor_subject: Any = None,
    journal_store: Any = None,
) -> dict[str, Any]:
    """One clicked card option. A Mentor question is filed by the Mentor's writer."""
    kind = _choice(card_kind, CARD_KINDS, field="card_kind")
    clicked = _text(option, limit=80, field="option")
    if not clicked:
        raise RecapError("option is empty")
    words = _text(text)
    card = _text(card_id, limit=120, field="card_id")
    if not card:
        raise RecapError("card_id is empty")
    clean_subject: dict[str, str] = {}
    for key, value in dict(subject or {}).items():
        if key not in SUBJECT_KEYS:
            raise RecapError(f"subject key {key!r} is not one of {SUBJECT_KEYS}")
        item = _text(value, limit=120, field=f"subject.{key}")
        clean_subject[key] = item.upper() if key in ("symbol", "side") else item
    if mentor_subject is not None:
        import mentor_questions

        _session(session_date)
        moment = _aware(now if now is not None else datetime.now().astimezone(), field="recorded_at")
        try:
            result = mentor_questions.record_answer(
                mentor_subject, {"state": clicked, "text": words}, store=journal_store, now=moment,
            )
        except Exception as exc:  # noqa: BLE001 - a journal write fails loudly
            raise RecapWriteError(f"the Mentor answer was not saved: {exc}") from exc
        if not result.get("ok"):
            raise RecapWriteError(f"the Mentor answer was not saved: {result.get('reason', '')}")
        return {"routed_to": "mentor", "card_id": card, "result": result}
    return _write(
        KIND_CARD_ANSWER,
        session_date,
        {"card_id": card, "card_kind": kind, "subject": clean_subject, "option": clicked, "text": words},
        supersedes=supersedes, now=now, path=path,
    )


def record_lesson(
    *,
    session_date: Any,
    keep: Any = "",
    stop: Any = "",
    try_: Any = "",
    mood: Any = None,
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Keep / Stop / Try (each optional) plus a mood on the Market Journal's scale."""
    import market_journal

    fields = {
        "keep": _text(keep, limit=SHORT_TEXT_MAX, field="keep"),
        "stop": _text(stop, limit=SHORT_TEXT_MAX, field="stop"),
        "try": _text(try_, limit=SHORT_TEXT_MAX, field="try"),
        "mood": None,
    }
    if mood is not None:
        if isinstance(mood, bool) or not isinstance(mood, int) or mood not in market_journal.MOOD_SCALE:
            raise RecapError(f"mood {mood!r} is not on the scale {market_journal.MOOD_SCALE}")
        fields["mood"] = mood
    if not (fields["keep"] or fields["stop"] or fields["try"] or fields["mood"] is not None):
        raise RecapError("a lesson needs keep, stop, try or a mood")
    return _write(KIND_LESSON, session_date, fields, supersedes=supersedes, now=now, path=path)


def record_rule(
    *,
    session_date: Any,
    text: Any,
    tag: Any = "",
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """One rule for the NEXT session, written in this session's recap."""
    words = _text(text, limit=SHORT_TEXT_MAX, field="text")
    if not words:
        raise RecapError("a rule needs text")
    chosen = _choice(tag, RULE_TAGS, field="tag") if str(tag or "").strip() else ""
    return _write(
        KIND_RULE, session_date, {"text": words, "tag": chosen},
        supersedes=supersedes, now=now, path=path,
    )


def record_rule_check(
    *,
    session_date: Any,
    answer: Any,
    rule_id: Any = "",
    text: Any = "",
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Did I keep the rule on `session_date`? yes, partly or no."""
    return _write(
        KIND_RULE_CHECK,
        session_date,
        {
            "answer": _choice(answer, RULE_CHECK_ANSWERS, field="answer"),
            "rule_id": _text(rule_id, limit=80, field="rule_id"),
            "text": _text(text),
        },
        supersedes=supersedes, now=now, path=path,
    )


def record_clue(
    *,
    session_date: Any,
    symbol: Any,
    timeframe: Any,
    bar_time: Any,
    price: Any,
    clue_tag: Any,
    text: Any = "",
    card_id: Any = "",
    trade_id: Any = "",
    pick_id: Any = "",
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """A hindsight chart clue: what the trader sees now on one bar."""
    ticker = _text(symbol, limit=20, field="symbol").upper()
    if not ticker:
        raise RecapError("a clue needs a symbol")
    try:
        level = float(price)
    except (TypeError, ValueError) as exc:
        raise RecapError(f"price {price!r} is not a number") from exc
    if level != level or level <= 0:
        raise RecapError(f"price {price!r} is not a positive number")
    return _write(
        KIND_CLUE,
        session_date,
        {
            "symbol": ticker,
            "timeframe": _choice(timeframe, CLUE_TIMEFRAMES, field="timeframe", upper=True),
            "bar_time": _aware(bar_time, field="bar_time").isoformat(),
            "price": level,
            "clue_tag": _choice(clue_tag, CLUE_TAGS, field="clue_tag"),
            "text": _text(text),
            "links": {
                "card_id": _text(card_id, limit=120, field="card_id"),
                "trade_id": _text(trade_id, limit=120, field="trade_id"),
                "pick_id": _text(pick_id, limit=120, field="pick_id"),
            },
        },
        supersedes=supersedes, now=now, path=path,
    )


def record_environment_verdict(
    *,
    session_date: Any,
    auto_label: Any,
    verdict: Any,
    clue_ids: Iterable[Any] = (),
    text: Any = "",
    supersedes: Any = "",
    now: datetime | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """The trader's verdict on the auto environment: agree, or a corrected label."""
    return _write(
        KIND_ENVIRONMENT_VERDICT,
        session_date,
        {
            "auto_label": _text(auto_label, limit=80, field="auto_label").lower(),
            "verdict": _choice(verdict, (VERDICT_AGREE, *ENVIRONMENT_LABELS), field="verdict"),
            "clue_ids": [_text(item, limit=80, field="clue_id") for item in (clue_ids or ()) if str(item or "").strip()],
            "text": _text(text),
        },
        supersedes=supersedes, now=now, path=path,
    )


def environment_label_for(session_date: Any, *, path: Path | None = None, rows=None) -> dict[str, Any]:
    """The session's trader-corrected environment, when a verdict exists.

    ``{"label", "source", "verdict_id"}``; `source` is `trader_corrected`,
    `trader_agreed` or `none`.
    """
    session = str(session_date or "").strip()[:10]
    source_rows = rows if rows is not None else records_for(session, [KIND_ENVIRONMENT_VERDICT], path=path)
    verdicts = [
        row for row in source_rows
        if row.get("kind") == KIND_ENVIRONMENT_VERDICT and row.get("session_date") == session
    ]
    if not verdicts:
        return {"label": "", "source": "none", "verdict_id": ""}
    latest = max(verdicts, key=lambda row: str(row.get("recorded_at") or ""))
    if latest.get("verdict") == VERDICT_AGREE:
        return {"label": str(latest.get("auto_label") or ""), "source": "trader_agreed", "verdict_id": latest["id"]}
    return {"label": str(latest.get("verdict") or ""), "source": "trader_corrected", "verdict_id": latest["id"]}
