"""Day Recap coach: the "one rule for tomorrow" loop, in front of the trader.

Reads the rule the last recap set (`recap_store.latest_rule_before`) and its
streak, words it for the prep page, the desk chip and the Mentor, and decides
which of the session's closed trades the rule's tag can honestly be checked on.

`load_today_rule` reads files: call it on a worker. Everything else is pure.
Evidence only: nothing here detects, scores, ranks, gates or alerts.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, time
from typing import Any, Iterable, Mapping, Sequence

#: The Mentor kind, and the key its answer is filed under.
REFLECTION_KIND = "rule_reflection"
REFLECTION_ANSWER_KEY = "rule_kept"
REFLECTION_OPTIONS = ("kept", "broke", "not_relevant")

#: At most this many reflections a session; the first closed trades win.
MAX_REFLECTIONS_PER_DAY = 3

#: Tags a closed trade can be checked against. Every other tag is never asked.
CHECKABLE_TAGS = ("hold_winners", "respect_stop", "no_trade_first_15m", "size_down_in_chop")

#: A median of fewer winners is not a day's median; the trade is not checked.
MIN_WINNERS_FOR_MEDIAN = 3
#: A size baseline needs this many earlier trades with a known size.
MIN_SIZE_BASELINE = 5
#: How far back the size baseline looks, in calendar days.
SIZE_BASELINE_DAYS = 30
#: The first minutes of the session `no_trade_first_15m` covers.
FIRST_MINUTES = 15
CHOP_LABEL = "neutral_chop"

TAG_WORDS = {
    "hold_winners": "hold winners",
    "wait_for_confirmation": "wait for confirmation",
    "respect_stop": "respect your stop",
    "size_down_in_chop": "size down in chop",
    "no_trade_first_15m": "no trades in the first 15 minutes",
    "one_trade_at_a_time": "one trade at a time",
    "only_a_plus_setups": "only A+ setups",
    "other": "",
}

CHIP_MAX = 48


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or _text(value) == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _aware(value: Any) -> datetime | None:
    """An aware moment, or None. A naive stamp is unknown, never guessed."""
    if isinstance(value, datetime):
        moment = value
    else:
        text = _text(value)
        if len(text) < 16:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None or moment.utcoffset() is None:
        return None
    return moment


def _market_tz():
    import market_calendar

    return market_calendar.MARKET_TZ


def market_date(now: datetime | None = None) -> date:
    """Today's market-local calendar date."""
    moment = now if now is not None else datetime.now().astimezone()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(_market_tz()).date()


# ---------------------------------------------------------------------------
# reading (worker only)
# ---------------------------------------------------------------------------
def load_today_rule(now: datetime | None = None, *, path: Any = None) -> dict[str, Any] | None:
    """The rule in force for today's market date, with its streak, or None.

    File reads: worker or night only.
    """
    return rule_for_date(market_date(now), path=path)


def rule_for_date(day: Any, *, path: Any = None) -> dict[str, Any] | None:
    """The rule in force on `day` (set by an earlier recap), with its streak, or None."""
    import recap_store

    for_date = _text(day.isoformat() if isinstance(day, date) else day)[:10]
    rule = recap_store.latest_rule_before(for_date, path=path)
    if not rule or not _text(rule.get("text")):
        return None
    try:
        streak = int(recap_store.rule_streak(for_date, path=path))
    except Exception:  # noqa: BLE001 - an unreadable streak is unknown
        streak = None
    return {
        "for_date": for_date,
        "rule_id": _text(rule.get("id")),
        "set_on": _text(rule.get("session_date")),
        "text": _text(rule.get("text")),
        "tag": _text(rule.get("tag")),
        "streak": streak,
    }


# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------
def prep_line(info: Mapping[str, Any] | None) -> str:
    """`Today's rule: ... (streak N)`, or "" when there is no rule."""
    if not isinstance(info, Mapping) or not _text(info.get("text")):
        return ""
    streak = info.get("streak")
    tail = f" (streak {streak})" if isinstance(streak, int) and not isinstance(streak, bool) else ""
    return f"Today's rule: {_text(info.get('text'))}{tail}"


def chip_text(info: Mapping[str, Any] | None) -> str:
    """`Rule: ...`, shortened for the status bar, or "" when there is no rule."""
    if not isinstance(info, Mapping) or not _text(info.get("text")):
        return ""
    words = _text(info.get("text"))
    if len(words) > CHIP_MAX:
        words = words[: CHIP_MAX - 3].rstrip() + "..."
    return f"Rule: {words}"


def chip_tooltip(info: Mapping[str, Any] | None) -> str:
    if not isinstance(info, Mapping) or not _text(info.get("text")):
        return ""
    streak = info.get("streak")
    kept = (
        f"Kept {streak} session(s) in a row."
        if isinstance(streak, int) and not isinstance(streak, bool)
        else "Streak unknown."
    )
    return (
        f"Today's rule: {_text(info.get('text'))}\n{kept}\n"
        f"Set in the {_text(info.get('set_on'))} recap."
    )


def rule_words(info: Mapping[str, Any]) -> str:
    tag = _text(info.get("tag"))
    return TAG_WORDS.get(tag) or _text(info.get("text"))


# ---------------------------------------------------------------------------
# which closed trades the rule can be checked on (pure)
# ---------------------------------------------------------------------------
def _side(row: Mapping[str, Any]) -> str:
    text = _text(row.get("direction")).upper()
    if text.startswith("SHORT") or text in {"SELL", "S"}:
        return "SHORT"
    if text.startswith("LONG") or text in {"BUY", "B"}:
        return "LONG"
    return ""


def _closed_in(row: Mapping[str, Any], session: date) -> datetime | None:
    if _text(row.get("status")).upper() != "CLOSED":
        return None
    closed = _aware(row.get("closed_at"))
    if closed is None or closed.astimezone(_market_tz()).date() != session:
        return None
    return closed


def trade_r(row: Mapping[str, Any]) -> float | None:
    """`net_pnl_cad / |planned_risk|`, the journal's one R, or None."""
    risk = _number(row.get("planned_risk"))
    pnl = _number(row.get("net_pnl_cad"))
    if risk is None or pnl is None or abs(risk) < 1e-9:
        return None
    return pnl / abs(risk)


def trade_pct(row: Mapping[str, Any]) -> float | None:
    """The price move captured, in percent, signed by the side, or None."""
    entry = _number(row.get("average_entry_price"))
    exit_ = _number(row.get("average_exit_price"))
    side = _side(row)
    if not entry or not exit_ or entry <= 0 or exit_ <= 0 or not side:
        return None
    move = (exit_ - entry) / entry * 100.0
    return move if side == "LONG" else -move


def trade_size(row: Mapping[str, Any]) -> float | None:
    """Entry notional (shares x average entry), or None."""
    qty = _number(row.get("quantity_opened"))
    entry = _number(row.get("average_entry_price"))
    if not qty or not entry or entry <= 0:
        return None
    return abs(qty) * entry


def size_baseline(rows: Iterable[Mapping[str, Any]], session: date) -> float | None:
    """Median entry notional of trades opened BEFORE `session`, or None."""
    import trade_origin

    sizes = []
    for row in rows or ():
        opened = trade_origin.trade_session(row)
        size = trade_size(row)
        if opened is None or size is None or opened >= session:
            continue
        sizes.append(size)
    if len(sizes) < MIN_SIZE_BASELINE:
        return None
    return float(statistics.median(sizes))


def regime_at(timeline: Sequence[tuple[datetime, str]] | None, moment: datetime | None) -> str:
    """The auto regime the desk last saw at or before `moment`, same market day, or ""."""
    if moment is None or not timeline:
        return ""
    day = moment.astimezone(_market_tz()).date()
    label = ""
    for seen_at, env_key in timeline:
        at = _aware(seen_at)
        if at is None or at.astimezone(_market_tz()).date() != day:
            continue
        if at <= moment:
            label = _text(env_key)
    return label


def _winner_pcts(rows: Sequence[Mapping[str, Any]], session: date) -> list[float]:
    out = []
    for row in rows:
        if _closed_in(row, session) is None:
            continue
        pnl = _number(row.get("net_pnl"))
        pct = trade_pct(row)
        if pnl is not None and pnl > 0 and pct is not None:
            out.append(pct)
    return out


def _check(
    tag: str,
    row: Mapping[str, Any],
    *,
    session: date,
    winners: Sequence[float],
    size_median: float | None,
    timeline: Sequence[tuple[datetime, str]] | None,
) -> str:
    """What happened, in words, when this trade is worth a reflection; "" otherwise."""
    symbol = _text(row.get("symbol")).upper() or "this trade"
    if tag == "hold_winners":
        pnl = _number(row.get("net_pnl"))
        if pnl is None or pnl <= 0:
            return ""
        r_value = trade_r(row)
        if r_value is not None:
            return f"You closed {symbol} at {r_value:+.1f}R" if r_value < 1.0 else ""
        pct = trade_pct(row)
        if pct is None or len(winners) < MIN_WINNERS_FOR_MEDIAN:
            return ""
        median = statistics.median(winners)
        if pct < median:
            return f"You closed {symbol} at {pct:+.1f}%, under the day's median winner ({median:+.1f}%)"
        return ""
    if tag == "respect_stop":
        r_value = trade_r(row)
        if r_value is None or r_value >= -1.0:
            return ""
        return f"You closed {symbol} at {r_value:+.1f}R, past your planned risk"
    if tag == "no_trade_first_15m":
        import trade_origin

        opened = trade_origin.first_fill_at(row)
        if opened is None:
            return ""
        local = opened.astimezone(_market_tz())
        if local.date() != session:
            return ""
        start = datetime.combine(session, time(9, 30), tzinfo=_market_tz())
        minutes = (local - start).total_seconds() / 60.0
        if 0 <= minutes < FIRST_MINUTES:
            return f"You entered {symbol} at {local:%H:%M}"
        return ""
    if tag == "size_down_in_chop":
        import trade_origin

        size = trade_size(row)
        if size is None or size_median is None or size <= size_median:
            return ""
        opened = trade_origin.first_fill_at(row)
        if regime_at(timeline, opened) != CHOP_LABEL:
            return ""
        return (
            f"You sized {symbol} at ${size:,.0f} (your usual is ${size_median:,.0f}) "
            "while the auto regime read neutral chop"
        )
    return ""


def reflection_rows(
    rule: Mapping[str, Any] | None,
    trades: Iterable[Mapping[str, Any]],
    *,
    session: Any,
    size_median: float | None = None,
    regime_timeline: Sequence[tuple[datetime, str]] | None = None,
) -> list[dict[str, Any]]:
    """The session's closed trades the rule can be checked on, first three by close.

    A rule for another session, an unchecked tag, or unknown data asks nothing.
    """
    if not isinstance(rule, Mapping):
        return []
    tag = _text(rule.get("tag"))
    if tag not in CHECKABLE_TAGS:
        return []
    try:
        day = date.fromisoformat(_text(session)[:10])
    except ValueError:
        return []
    if _text(rule.get("for_date"))[:10] != day.isoformat():
        return []
    rows = [row for row in trades or () if isinstance(row, Mapping)]
    winners = _winner_pcts(rows, day)
    closed: list[tuple[datetime, str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        trade_id = _text(row.get("trade_id"))
        moment = _closed_in(row, day)
        if not trade_id or trade_id in seen or moment is None:
            continue
        seen.add(trade_id)
        closed.append((moment, trade_id, row))
    closed.sort(key=lambda item: (item[0], item[1]))
    out: list[dict[str, Any]] = []
    for _moment, trade_id, row in closed:
        what = _check(
            tag, row, session=day, winners=winners,
            size_median=size_median, timeline=regime_timeline,
        )
        if not what:
            continue
        words = rule_words(rule)
        out.append({
            "trade_id": trade_id,
            "symbol": _text(row.get("symbol")).upper(),
            "tag": tag,
            "rule_id": _text(rule.get("rule_id")),
            "session": day.isoformat(),
            "prompt": f"Your rule today: {words}. {what} — kept it or broke it?",
        })
        if len(out) >= MAX_REFLECTIONS_PER_DAY:
            break
    return out


__all__ = [
    "CHECKABLE_TAGS",
    "MAX_REFLECTIONS_PER_DAY",
    "REFLECTION_ANSWER_KEY",
    "REFLECTION_KIND",
    "REFLECTION_OPTIONS",
    "chip_text",
    "chip_tooltip",
    "load_today_rule",
    "market_date",
    "prep_line",
    "reflection_rows",
    "regime_at",
    "size_baseline",
]
