"""Review my day: the walk's cards, and the one reader that feeds them.

The Day Recap coach's guided walk (trader, 2026-09-23: "5 mins is great").
One card at a time: trades, real misses, a good pass, the market calls, the
market environment, then the lesson. About ten cards, best teaching value
first, the lesson always last.

Two halves:

* **Pure** (:func:`build_cards` and its helpers): no file, no clock, no Qt.
  It turns the Day Review payload, the findability answer, the environment
  summary and the Mentor's pending questions into plain card dicts.
* **The reader** (:func:`load_walk_inputs`, below the line): every store read
  the walk needs, each in its own guard. Worker thread only, never the Qt
  thread.

Missing is "not known", never a zero. Evidence only: nothing here reaches a
detector, score, alert, watchlist, Focus or ``review_policy.json``.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
NOT_KNOWN = "not known"

#: About ten cards, the lesson included.
WALK_CAP = 10
MISS_MAX = 3
GOOD_PASS_MAX = 2
#: Rough seconds each kind takes, for "~N min left".
SECONDS_PER_CARD = {
    "trade": 40, "miss": 30, "good_pass": 15, "call": 20, "environment": 40, "lesson": 60,
}
#: The Mentor question kinds a trade card may carry. `pending` decides which are
#: awake; the walk only narrows to the ones about ONE trade.
WALK_MENTOR_KINDS = ("trade_origin", "open_position_check", "exit_draft_review")

MISS_OPTIONS = (("good_pass", "Good pass"), ("real_miss", "Real miss"))
GOOD_PASS_OPTIONS = (("good_pass", "Right call"), ("lucky", "Lucky - it could have run"))
CALL_OPTIONS = (("read_right", "I read it right"), ("misread", "I misread it"), ("too_early", "Too early to tell"))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value or "").strip()


def _float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _side(value: Any) -> str:
    text = _text(value).upper()
    if text in ("LONG", "BUY", "L"):
        return "LONG"
    if text in ("SHORT", "SELL", "S", "SELL_SHORT"):
        return "SHORT"
    return text


def _moment(value: Any, zone: Any = ET) -> datetime | None:
    """An aware moment. A naive stamp is market-local (the bar convention)."""
    if isinstance(value, datetime):
        stamp = value
    else:
        text = _text(value)
        if not text:
            return None
        try:
            stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=ET)


def clock(value: Any, zone: Any = ET) -> str:
    """HH:MM in `zone`, or "" when the stamp is unreadable."""
    stamp = _moment(value)
    if stamp is None:
        return ""
    return stamp.astimezone(zone or ET).strftime("%H:%M")


def _money(value: float | None) -> str:
    if value is None:
        return NOT_KNOWN
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def _r(value: float | None) -> str:
    return NOT_KNOWN if value is None else f"{value:+.2f}R"


def _words(value: Any) -> str:
    return _text(value).replace("_", " ")


# ---------------------------------------------------------------------------
# a trade: R, and "what if"
# ---------------------------------------------------------------------------
def trade_r(trade: Mapping[str, Any]) -> float | None:
    """Net over planned risk; else price move over the planned stop distance."""
    net = _float(trade.get("net_pnl"))
    risk = _float(trade.get("planned_risk"))
    if net is not None and risk:
        return net / abs(risk)
    entry = _float(trade.get("average_entry_price"))
    exit_ = _float(trade.get("average_exit_price"))
    stop = _float(trade.get("planned_stop"))
    sign = 1.0 if _side(trade.get("direction")) == "LONG" else -1.0 if _side(trade.get("direction")) == "SHORT" else 0.0
    if not sign or entry is None or not exit_ or stop is None or entry == stop:
        return None
    return (exit_ - entry) * sign / abs(entry - stop)


def what_if(trade: Mapping[str, Any], bars: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Held to the close, held to the stop, and MFE / MAE, from the session's M5 bars.

    Each value is per share (and in R when the stop is known). A value the
    tape or the trade cannot answer is None and reads "not known".
    """
    out: dict[str, Any] = {"held_to_close": None, "held_to_stop": None, "mfe": None, "mae": None, "unit": "$/sh"}
    entry = _float(trade.get("average_entry_price"))
    exit_ = _float(trade.get("average_exit_price"))
    stop = _float(trade.get("planned_stop"))
    side = _side(trade.get("direction"))
    sign = 1.0 if side == "LONG" else -1.0 if side == "SHORT" else 0.0
    opened = _moment(trade.get("opened_at"))
    closed = _moment(trade.get("closed_at"))
    rows = []
    for bar in bars or ():
        if not isinstance(bar, Mapping):
            continue
        start = _moment(bar.get("dt"))
        high, low, close = _float(bar.get("high")), _float(bar.get("low")), _float(bar.get("close"))
        if start is None or high is None or low is None or close is None:
            continue
        rows.append((start, high, low, close))
    rows.sort(key=lambda item: item[0])
    if not sign or not entry or opened is None or not rows or opened < rows[0][0] - _five_minutes():
        # A swing opened before this tape: the tape cannot say what it did from entry.
        return out
    risk = abs(entry - stop) if stop is not None and stop != entry else None
    unit = (lambda v: v / risk) if risk else (lambda v: v)
    out["unit"] = "R" if risk else "$/sh"
    in_trade = [r for r in rows if r[0] + _five_minutes() > opened and (closed is None or r[0] <= closed)]
    if in_trade:
        # Long: best is the highest high, worst the lowest low. Short: the mirror.
        best = [(h - entry) if sign > 0 else (entry - lo) for _s, h, lo, _c in in_trade]
        worst = [(lo - entry) if sign > 0 else (entry - h) for _s, h, lo, _c in in_trade]
        out["mfe"] = unit(max(best))
        out["mae"] = unit(min(worst))
    last_close = rows[-1][3]
    out["held_to_close"] = unit((last_close - entry) * sign)
    if stop is not None and closed is not None and exit_:
        after = [r for r in rows if r[0] >= closed]
        stopped = next(
            (r for r in after if (sign > 0 and r[2] <= stop) or (sign < 0 and r[1] >= stop)), None
        )
        out["held_to_stop"] = unit((stop - entry) * sign) if stopped else unit((last_close - entry) * sign)
    return out


def _five_minutes() -> timedelta:
    return timedelta(minutes=5)


def what_if_words(values: Mapping[str, Any]) -> str:
    unit = values.get("unit") or "$/sh"

    def fmt(key: str) -> str:
        value = values.get(key)
        if value is None:
            return NOT_KNOWN
        return f"{value:+.2f}R" if unit == "R" else f"{value:+.2f} $/sh"

    return (
        f"What if: held to the close {fmt('held_to_close')} · held to the stop {fmt('held_to_stop')}"
        f" · best move (MFE) {fmt('mfe')} · worst move (MAE) {fmt('mae')}"
    )


# ---------------------------------------------------------------------------
# teaching value
# ---------------------------------------------------------------------------
def teaching_value(card: Mapping[str, Any]) -> float:
    """Higher teaches more. The lesson is not ranked: it is always last.

    * trade: 60, +15 when the Mentor still has a question, +5 for a loss, + up to 20 for size (|R|).
    * real miss 70 / pass that ran 65, + up to 20 for how far it ran (%).
    * market environment 58 (the verdict is the day's frame).
    * calls 45 when at least one was measured, else 25.
    * good pass 35 + up to 10 for how far it fell away.
    """
    kind = card.get("kind")
    if kind == "trade":
        r = card.get("r")
        pnl = card.get("pnl")
        value = 60.0
        if card.get("mentor") or card.get("exit_draft") or card.get("exit_note"):
            value += 15
        if (pnl is not None and pnl < 0) or (r is not None and r < 0):
            value += 5
        if r is not None:
            value += min(20.0, 5.0 * abs(r))
        return value
    if kind == "miss":
        ran = abs(_float(card.get("ran_after_pct")) or 0.0)
        base = 70.0 if card.get("category") == "real_miss" else 65.0
        return base + min(20.0, ran)
    if kind == "environment":
        return 58.0
    if kind == "call":
        return 45.0 if card.get("measured") else 25.0
    if kind == "good_pass":
        return 35.0 + min(10.0, abs(_float(card.get("ran_after_pct")) or 0.0))
    return 0.0


# ---------------------------------------------------------------------------
# the cards
# ---------------------------------------------------------------------------
def _chart(payload: Mapping[str, Any], symbol: str) -> dict[str, Any] | None:
    charts = payload.get("name_charts") or {}
    chart = charts.get(symbol.upper()) if isinstance(charts, Mapping) else None
    if not isinstance(chart, Mapping):
        return None
    bars = [b for b in chart.get("bars") or () if isinstance(b, Mapping) and b.get("dt") is not None]
    if not bars:
        return None
    return {"symbol": symbol.upper(), "bars": bars, "markers": tuple(chart.get("markers") or ())}


def _env_words(traits: Mapping[str, Any] | None) -> str:
    if not traits:
        return NOT_KNOWN
    import recap_findability as rf

    d1 = (traits.get("env_d1") or {}).get("label", rf.UNKNOWN)
    intra = (traits.get("env_intraday") or {}).get("label", rf.UNKNOWN)
    parts = []
    if d1 != rf.UNKNOWN:
        parts.append(f"D1 {rf.D1_ENV_WORDS.get(d1, d1)}")
    if intra != rf.UNKNOWN:
        parts.append(f"tape {rf.INTRADAY_ENV_WORDS.get(intra, intra)}")
    return ", ".join(parts) or NOT_KNOWN


def _subject_options(subject: Any) -> list[tuple[str, str]]:
    import mentor_questions

    return [
        (str(option), _words(option))
        for option in tuple(getattr(subject, "options", ()) or ())
        if str(option) != mentor_questions.STOP_ASKING
    ]


def trade_card(
    trade: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    traits: Mapping[str, Any] | None = None,
    mentor: Sequence[Any] = (),
    zone: Any = ET,
) -> dict[str, Any]:
    """One trade: chart, R / P&L, grade and environment at entry, source alert, what-if, Mentor questions."""
    trade_id = _text(trade.get("trade_id"))
    symbol = _text(trade.get("symbol")).upper()
    side = _side(trade.get("direction"))
    pnl = _float(trade.get("net_pnl"))
    r = trade_r(trade)
    chart = _chart(payload, symbol)
    reviews = {
        _text(row.get("trade_id")): row for row in payload.get("trade_reviews") or () if isinstance(row, Mapping)
    }
    review = reviews.get(trade_id) or {}
    opened, closed = clock(trade.get("opened_at"), zone), clock(trade.get("closed_at"), zone)
    lines = [
        f"Opened {opened or NOT_KNOWN} · " + (f"closed {closed}" if closed else "still open"),
        f"Result: {_money(pnl)} · {_r(r)}",
    ]
    grade = _text((traits or {}).get("grade")) or "unknown"
    lines.append(
        f"Grade at entry: {NOT_KNOWN if grade == 'unknown' else grade} · market at entry: {_env_words(traits)}"
    )
    kind = _text((traits or {}).get("alert_kind"))
    if kind and kind != "unknown":
        tier = _text((traits or {}).get("alert_tier"))
        at = clock((traits or {}).get("alert_at"), zone)
        lines.append(
            f"Source alert: M5 {_words(kind)}"
            + (f" tier {tier}" if tier and tier != "unknown" else "")
            + (f" at {at}" if at else "")
        )
    else:
        lines.append(f"Source alert: {NOT_KNOWN}")
    whatif = what_if(trade, chart["bars"] if chart else ())
    exit_raw = review.get("exit_raw") or {}
    said_exit = _text(exit_raw.get("text")) or _text(exit_raw.get("answer_state"))
    questions = []
    draft = None
    for subject in mentor:
        if getattr(subject, "kind", "") == "exit_draft_review":
            if draft is None:
                detail = dict(getattr(subject, "detail", {}) or {})
                draft = {
                    "key": detail.get("key") or getattr(subject, "subject_id", ""),
                    "trade_id": detail.get("trade_id") or trade_id,
                    "exit_session": detail.get("exit_session", ""),
                    "note_id": detail.get("note_id", ""),
                    "fields": dict(detail.get("fields") or {}),
                    "raw_text": detail.get("raw_text", ""),
                    "sentence": draft_sentence(detail.get("fields") or {}),
                }
            continue
        questions.append({
            "subject": subject,
            "kind": getattr(subject, "kind", ""),
            "prompt": _text(getattr(subject, "prompt", "")) or _words(getattr(subject, "kind", "")),
            "options": _subject_options(subject),
        })
    exit_note = None
    session = _text(payload.get("session_date"))[:10]
    if (
        draft is None and not said_exit and closed
        and _text(trade.get("status")).upper() == "CLOSED"
        and _text(trade.get("closed_at"))[:10] == session
    ):
        exit_note = {"trade_id": trade_id, "exit_session": session}
    card = {
        "id": f"trade:{trade_id}",
        "kind": "trade",
        "title": f"{symbol} {side}".strip() + f" · {_money(pnl)} · {_r(r)}",
        "subject": {k: v for k, v in (("trade_id", trade_id), ("symbol", symbol), ("side", side)) if v},
        "symbol": symbol,
        "trade_id": trade_id,
        "chart": chart,
        "chart_missing": f"No {symbol} M5 tape for this session." if chart is None else "",
        "lines": lines,
        "what_if": what_if_words(whatif),
        "pnl": pnl,
        "r": r,
        "mentor": questions,
        "exit_draft": draft,
        "exit_note": exit_note,
    }
    card["value"] = teaching_value(card)
    return card


def draft_sentence(fields: Mapping[str, Any]) -> str:
    """The night's exit reading as "Looks like ... - right?"."""
    parts: list[str] = []
    why = fields.get("why") if isinstance(fields, Mapping) else None
    if isinstance(why, Mapping) and _text(why.get("code")):
        code = _text(why.get("code"))
        try:
            import exit_reasons

            code = exit_reasons.label_for(code) or code
        except Exception:  # noqa: BLE001 - the code reads fine on its own
            pass
        parts.append(f"you exited because: {_words(code)}")
    felt = [
        _text((row or {}).get("code")) for row in (fields.get("felt") or () if isinstance(fields, Mapping) else ())
        if isinstance(row, Mapping) and _text(row.get("code"))
    ]
    if felt:
        parts.append("you felt " + ", ".join(_words(code) for code in felt))
    watching = [
        _text((row or {}).get("quote")) for row in (fields.get("watching") or () if isinstance(fields, Mapping) else ())
        if isinstance(row, Mapping) and _text(row.get("quote"))
    ]
    if watching:
        parts.append("you were watching " + "; ".join(watching))
    return ("Looks like " + " - ".join(parts) + " - right?") if parts else ""


def _pick_line(pick: Mapping[str, Any], zone: Any) -> str:
    surfaced = pick.get("surfaced") or {}
    first = surfaced.get("surfaces") or ()
    if surfaced.get("first_at"):
        where = _text(first[0].get("surface")) if first else "the desk"
        return f"The desk showed it at {clock(surfaced['first_at'], zone) or '?'} ({where})."
    return "It never surfaced on the desk."


def miss_card(pick: Mapping[str, Any], payload: Mapping[str, Any], *, zone: Any = ET) -> dict[str, Any]:
    """A real miss or a pass that ran: the recipe, how it did lately, where to look."""
    symbol = _text(pick.get("symbol")).upper()
    side = _side(pick.get("side"))
    lately = pick.get("recipe_lately") or {}
    core = (lately.get("core") or {}).get("words") or "no measured rows lately"
    in_env = lately.get("core_in_this_environment") or {}
    lines = [
        f"What you did: {_text(pick.get('what_you_did')) or NOT_KNOWN}",
        f"Your pass reason: {_text(pick.get('reason')) or 'none written'}",
        f"Recipe: {_text(pick.get('recipe')) or 'no known traits'}",
        f"How that recipe did lately: {core}"
        + (f" · in this environment ({_words(in_env.get('environment'))}): {in_env.get('words')}" if in_env else ""),
        _pick_line(pick, zone),
    ]
    look = [str(line) for line in pick.get("where_to_look") or ()]
    card = {
        "id": f"miss:{symbol}:{side}",
        "kind": "miss",
        "category": _text(pick.get("category")),
        "title": f"{symbol} {side} · {pick.get('category_words') or _words(pick.get('category'))} · {_text(pick.get('result'))}",
        "subject": {k: v for k, v in (("symbol", symbol), ("side", side)) if v},
        "symbol": symbol,
        "chart": _chart(payload, symbol),
        "chart_missing": f"No {symbol} M5 tape for this session." if _chart(payload, symbol) is None else "",
        "lines": lines,
        "where_to_look": look,
        "options": list(MISS_OPTIONS),
        "ran_after_pct": _ran_after(pick),
    }
    card["value"] = teaching_value(card)
    return card


def _ran_after(pick: Mapping[str, Any]) -> float | None:
    text = _text(pick.get("result"))
    if "ran " in text:
        try:
            return float(text.split("ran ", 1)[1].split("%", 1)[0])
        except ValueError:
            return None
    return None


def good_pass_card(pick: Mapping[str, Any], payload: Mapping[str, Any], *, zone: Any = ET) -> dict[str, Any]:
    symbol = _text(pick.get("symbol")).upper()
    side = _side(pick.get("side"))
    card = {
        "id": f"good_pass:{symbol}:{side}",
        "kind": "good_pass",
        "title": f"{symbol} {side} · good pass · {_text(pick.get('result'))}",
        "subject": {k: v for k, v in (("symbol", symbol), ("side", side)) if v},
        "symbol": symbol,
        "chart": _chart(payload, symbol),
        "chart_missing": "",
        "lines": [
            f"Your pass reason: {_text(pick.get('reason')) or 'none written'}",
            "You passed and it did not run. That was the right call - keep doing it.",
        ],
        "options": list(GOOD_PASS_OPTIONS),
        "ran_after_pct": _ran_after(pick),
    }
    card["value"] = teaching_value(card)
    return card


def calls_card(payload: Mapping[str, Any], *, zone: Any = ET) -> dict[str, Any] | None:
    reads = [row for row in payload.get("reads") or () if isinstance(row, Mapping)]
    if not reads:
        return None
    lines = []
    measured = 0
    for row in reads:
        verdict = _text(row.get("verdict")) or "unmeasured"
        if not verdict.startswith("unmeasured"):
            measured += 1
        shown = "not measured yet" if verdict.startswith("unmeasured") else _words(verdict)
        lines.append(
            f"{clock(row.get('stamp'), zone) or '--:--'} · {_words(row.get('horizon')) or '?'} · "
            f"{_words(row.get('direction')) or 'no view'}"
            + (f" ({_words(row.get('confidence'))})" if _text(row.get("confidence")) else "")
            + f" → {shown}"
        )
    card = {
        "id": "calls",
        "kind": "call",
        "title": f"Your market calls · {measured} of {len(reads)} measured",
        "subject": {},
        "chart": None,
        "lines": lines,
        "options": list(CALL_OPTIONS),
        "measured": measured,
    }
    card["value"] = teaching_value(card)
    return card


def environment_card(environment: Mapping[str, Any] | None, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The auto environment in plain words, how setups did in it lately, and the verdict."""
    import recap_findability as rf
    import recap_store

    env = dict(environment or {})
    auto = _text(env.get("main_intraday_label")) or rf.UNKNOWN
    lines = [_text(item.get("words")) for item in env.get("timeline") or () if _text(item.get("words"))]
    if not lines:
        lines = ["The desk recorded no environment for this session."]
    lately = env.get("lately_in_this_environment") or {}
    for key, trait, label in (("m5", "by_alert_kind", "M5 alerts"), ("d1", "by_family", "D1 setups")):
        block = lately.get(key) or {}
        rows = list(block.get(trait) or ())[:3]
        if rows:
            lines.append(
                f"{label} lately in {_words(block.get('environment'))}: "
                + "; ".join(f"{_words(row.get('value'))} {row.get('words')}" for row in rows)
            )
        elif block.get("why"):
            lines.append(f"{label} lately: {block['why']}")
    options: list[tuple[str, str]] = []
    if auto != rf.UNKNOWN:
        options.append((recap_store.VERDICT_AGREE, f"Agree: {rf.INTRADAY_ENV_WORDS.get(auto, _words(auto))}"))
    for label in recap_store.ENVIRONMENT_LABELS:
        if label != auto:
            options.append((label, f"It was {rf.INTRADAY_ENV_WORDS.get(label, _words(label))}"))
    bars = [b for b in payload.get("spy_m5_bars") or () if isinstance(b, Mapping) and b.get("dt") is not None]
    card = {
        "id": "environment",
        "kind": "environment",
        "title": "The market today · the desk said "
        + (rf.INTRADAY_ENV_WORDS.get(auto, _words(auto)) if auto != rf.UNKNOWN else "nothing it could name"),
        "subject": {"symbol": "SPY"},
        "symbol": "SPY",
        "auto_label": auto,
        "chart": {"symbol": "SPY", "bars": bars, "markers": tuple(payload.get("spy_markers") or ())} if bars else None,
        "chart_missing": "No SPY M5 tape for this session." if not bars else "",
        "lines": lines,
        "options": options,
        "clue_tags": list(recap_store.CLUE_TAGS),
    }
    card["value"] = teaching_value(card)
    return card


def lesson_card(rule: Mapping[str, Any] | None, streak: int) -> dict[str, Any]:
    import market_journal
    import recap_store

    return {
        "id": "lesson",
        "kind": "lesson",
        "title": "Your lesson",
        "subject": {},
        "chart": None,
        "lines": [],
        "rule": dict(rule) if rule else None,
        "rule_options": [(a, a.capitalize()) for a in recap_store.RULE_CHECK_ANSWERS] if rule else [],
        "streak": int(streak or 0),
        "moods": list(market_journal.MOOD_SCALE),
        "rule_tags": list(recap_store.RULE_TAGS),
        "options": [(a, a.capitalize()) for a in recap_store.RULE_CHECK_ANSWERS] if rule else [],
        "value": -1.0,
    }


def build_cards(
    session: str,
    payload: Mapping[str, Any],
    *,
    findability: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    traits_by_trade: Mapping[str, Mapping[str, Any]] | None = None,
    mentor_by_trade: Mapping[str, Sequence[Any]] | None = None,
    rule: Mapping[str, Any] | None = None,
    streak: int = 0,
    zone: Any = ET,
    cap: int = WALK_CAP,
) -> list[dict[str, Any]]:
    """Every candidate card, ranked by teaching value, capped, the lesson last."""
    payload = dict(payload or {})
    payload.setdefault("session_date", session)
    candidates: list[dict[str, Any]] = []
    for trade in payload.get("trades") or ():
        if not isinstance(trade, Mapping) or not _text(trade.get("trade_id")):
            continue
        trade_id = _text(trade.get("trade_id"))
        candidates.append(trade_card(
            trade, payload,
            traits=(traits_by_trade or {}).get(trade_id),
            mentor=tuple((mentor_by_trade or {}).get(trade_id) or ()),
            zone=zone,
        ))
    picks = [p for p in (findability or {}).get("picks") or () if isinstance(p, Mapping)]
    misses = [miss_card(p, payload, zone=zone) for p in picks if p.get("category") in ("real_miss", "pass_ran")]
    misses.sort(key=lambda c: -c["value"])
    candidates.extend(misses[:MISS_MAX])
    goods = [good_pass_card(p, payload, zone=zone) for p in picks if p.get("category") == "good_pass_failed"]
    goods.sort(key=lambda c: -c["value"])
    candidates.extend(goods[:GOOD_PASS_MAX])
    call = calls_card(payload, zone=zone)
    if call is not None:
        candidates.append(call)
    candidates.append(environment_card(environment, payload))
    # Stable: equal value keeps the order above (trades, misses, passes, calls, market).
    ranked = sorted(candidates, key=lambda c: -c["value"])
    keep = max(0, int(cap) - 1)
    return ranked[:keep] + [lesson_card(rule, streak)]


def minutes_left(cards: Sequence[Mapping[str, Any]], index: int) -> str:
    seconds = sum(SECONDS_PER_CARD.get(str(c.get("kind")), 30) for c in cards[max(0, index):])
    if seconds < 60:
        return "under a minute left"
    return f"~{math.ceil(seconds / 60)} min left"


def progress_text(cards: Sequence[Mapping[str, Any]], index: int) -> str:
    return f"{index + 1} of {len(cards)} · {minutes_left(cards, index)}"


# ---------------------------------------------------------------------------
# the Mentor, narrowed to one trade (pure over a loaded state)
# ---------------------------------------------------------------------------
def mentor_by_trade(state: Mapping[str, Any], trade_ids: Iterable[str]) -> dict[str, list[Any]]:
    """The Mentor's owed questions about each trade: `pending`'s asked plus carried.

    `pending` already drops dormant kinds (no reader yet), answered and retired
    subjects. The walk keeps only the trade-scoped kinds, per trade.
    """
    import mentor_questions

    wanted = {str(t) for t in trade_ids if str(t or "").strip()}
    result = mentor_questions.pending(state, None)
    out: dict[str, list[Any]] = {}
    for subject in tuple(result.asked) + tuple(result.carried):
        if subject.kind not in WALK_MENTOR_KINDS:
            continue
        detail = dict(subject.detail or {})
        trade_id = _text(detail.get("trade_id")) or _text(subject.subject_id).split("@", 1)[0]
        if trade_id in wanted:
            out.setdefault(trade_id, []).append(subject)
    return out


# ===========================================================================
# the reader - worker thread only, never the Qt thread
# ===========================================================================
def read_mentor_state(
    session: str,
    trades: Sequence[Mapping[str, Any]],
    store: Any,
    *,
    today: str = "",
) -> dict[str, Any]:
    """Every lane `mentor_questions.pending` reads for the walk's trades, loaded once.

    The same lanes the desk's Mentor card reads (`ui.app._mentor_question_state`),
    bounded to this session. Each lane in its own guard: an unreadable lane asks
    nothing extra and never costs the walk.
    """
    import day_report_card
    import mentor_questions

    day = _text(session)[:10]
    decisions: list[Any] = []
    try:
        from project_paths import TRADER_ANNOTATIONS_FILE
        from ui.annotations.store import EVENT_LIKE_CLAIM, load_annotations

        decisions = load_annotations(TRADER_ANNOTATIONS_FILE, session_date=day, event_types=(EVENT_LIKE_CLAIM,))
    except Exception:  # noqa: BLE001
        logging.debug("Walk: decision lane unreadable.", exc_info=True)
    claims: list[Any] = []
    try:
        import claimed_picks

        claims = [row for row in claimed_picks.load_rows() if _text(row.get("session_date"))[:10] == day]
    except Exception:  # noqa: BLE001
        logging.debug("Walk: claim lane unreadable.", exc_info=True)
    loaded = {"decisions": decisions, "claims": claims}
    lanes = {
        name: loaded.get(name, ()) if name in day_report_card.DESK_ORIGIN_LANES_READ else ()
        for name in day_report_card.ORIGIN_LANES
    }
    exit_drafts: list[Any] = []
    try:
        import trade_mentor_trade_check as check

        exit_drafts = list(check.waiting_exit_drafts(store, (today or day)[:10]))
    except Exception:  # noqa: BLE001
        logging.debug("Walk: exit-draft lane unreadable.", exc_info=True)
    answered: dict[str, dict[str, str]] = {}
    for trade_id in dict.fromkeys(_text(t.get("trade_id")) for t in trades):
        if not trade_id:
            continue
        try:
            rows = store.list_opportunity_events(
                trade_id=trade_id, event_type=mentor_questions.EVENT_MENTOR_ANSWER, limit=1000,
            )
        except Exception:  # noqa: BLE001
            logging.debug("Walk: Mentor answers unreadable for %s.", trade_id, exc_info=True)
            continue
        for row in rows:
            body = row.get("payload") or {}
            kind, subject_id = _text(body.get("mentor_question_kind")), _text(body.get("subject_id"))
            if kind and subject_id:
                answered[f"{kind}:{subject_id}"] = {"answered_at": _text(row.get("occurred_at"))[:10]}
    retired: list[str] = []
    try:
        import json
        from pathlib import Path

        from project_paths import TRADE_MENTOR_SLOTS_FILE

        state = json.loads(Path(TRADE_MENTOR_SLOTS_FILE).read_text(encoding="utf-8"))
        retired = [str(item) for item in state.get("retired_subjects") or () if str(item or "").strip()]
    except Exception:  # noqa: BLE001 - no file means nothing retired
        pass
    rows = [dict(t) for t in trades if isinstance(t, Mapping)]
    return {
        "session": day,
        "auto_mode": "",
        "trades": rows,
        "open_positions": [t for t in rows if _text(t.get("status")).upper() != "CLOSED"],
        "likes": (),
        "grader_gaps": (),
        **lanes,
        "exit_drafts": exit_drafts,
        "answered": answered,
        "retired": retired,
        "carried": (),
    }


def load_walk_inputs(
    session: str,
    payload: Mapping[str, Any],
    *,
    journal_store: Any = None,
    today: str = "",
    zone: Any = ET,
) -> dict[str, Any]:
    """Read everything the walk needs and build its cards. Worker thread only.

    Returns ``{"session", "cards", "unread"}``. One unreadable store costs its
    part of the walk and is named in `unread`.
    """
    import recap_findability as rf
    import recap_store

    day = _text(session)[:10]
    payload = dict(payload or {})
    unread: list[str] = []
    trades = [t for t in payload.get("trades") or () if isinstance(t, Mapping)]
    inputs: dict[str, Any] = {}
    findability: dict[str, Any] = {}
    environment: dict[str, Any] = {}
    traits: dict[str, Any] = {}
    try:
        inputs = rf.read_inputs(day, payload=payload, extra_symbols=[_text(t.get("symbol")) for t in trades])
        unread.extend(inputs.get("unread") or ())
        findability = rf.findability_for_session(day, inputs)
        environment = rf.environment_summary(day, inputs)
        for trade in trades:
            pick = {
                "symbol": trade.get("symbol"),
                "side": _side(trade.get("direction")),
                "timeframe": "M5" if "day_trade" in _text(trade.get("auto_tag_summary")) else "D1",
                "pick_at": trade.get("opened_at"),
            }
            traits[_text(trade.get("trade_id"))] = rf.traits_for(pick, {**inputs, "session": day})
    except Exception as exc:  # noqa: BLE001
        unread.append(f"findability: {exc}")
    mentor: dict[str, list[Any]] = {}
    if trades:
        try:
            store = journal_store
            if store is None:
                from journal_store import JournalStore

                store = JournalStore()
            state = read_mentor_state(day, trades, store, today=today)
            mentor = mentor_by_trade(state, [_text(t.get("trade_id")) for t in trades])
        except Exception as exc:  # noqa: BLE001
            unread.append(f"trade mentor: {exc}")
    rule = None
    streak = 0
    try:
        rule = recap_store.latest_rule_before(day)
        streak = recap_store.rule_streak(day)
    except Exception as exc:  # noqa: BLE001
        unread.append(f"recap rules: {exc}")
    cards = build_cards(
        day, payload,
        findability=findability, environment=environment,
        traits_by_trade=traits, mentor_by_trade=mentor,
        rule=rule, streak=streak, zone=zone,
    )
    if any(card.get("exit_draft") for card in cards):
        reasons: list[tuple[str, str]] = []
        try:
            import exit_reasons

            reasons = [(code, exit_reasons.label_for(code) or _words(code)) for code in exit_reasons.codes()]
        except Exception as exc:  # noqa: BLE001 - Fix then offers only the note words
            unread.append(f"exit reasons: {exc}")
        for card in cards:
            if card.get("exit_draft"):
                card["exit_draft"]["reasons"] = reasons
    return {"session": day, "cards": cards, "unread": unread}


__all__ = [
    "GOOD_PASS_MAX",
    "MISS_MAX",
    "NOT_KNOWN",
    "WALK_CAP",
    "WALK_MENTOR_KINDS",
    "build_cards",
    "load_walk_inputs",
    "mentor_by_trade",
    "progress_text",
    "read_mentor_state",
    "teaching_value",
    "what_if",
]
