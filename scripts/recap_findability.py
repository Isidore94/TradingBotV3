"""Find it next time: the traits of a session's notable picks, and how they do lately.

Day Recap "coach" (trader, 2026-09-23): *"focusing on not just great picks but
also how to find them next time, combining them with setup tracker and day
trade tracker to get my eyes to what's working"* and *"combine with auto market
environment to help me clarify the market environment."*

Two answers per session, both plain JSON:

* :func:`findability_for_session` - each notable name's traits AT THE TIME it
  was picked, a plain-words recipe, how that recipe did lately, where the desk
  shows such names, and whether the desk surfaced this one before the pick.
* :func:`environment_summary` - every environment label the desk gave the
  session and when, the evidence behind each, and how setups did lately in it.

Rules:

* **PURE.** No file, no clock, no Qt. :func:`read_inputs` (below the line) is
  the one reader; it runs on a worker or in the CLI, never on the Qt thread.
* **Point in time.** A trait or label is used for a pick only when it was
  recorded AT OR BEFORE the pick time. Anything later is ``unknown``.
* **Missing is unknown**, never a zero or a guess.
* **Read-only.** Nothing here reaches a detector, score, alert, the setup
  tracker, the grades, a watchlist, Focus or ``review_policy.json``.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

SCHEMA = "recap_findability_v1"
ENV_SCHEMA = "recap_environment_v1"
UNKNOWN = "unknown"

#: The trailing window for "lately", in exchange sessions (evidence_stats.LATELY_SESSIONS).
WINDOW_SESSIONS = 20
#: A cohort under this many measured rows says "too few to tell".
MIN_COHORT_N = 10
#: The D1 ruler: the scan row's side return this many sessions after the scan.
D1_HORIZON = 5

ET = ZoneInfo("America/New_York")

#: The notable-name categories, in priority order (a name lands in the first).
GREAT_TRADE = "great_trade"
REAL_MISS = "real_miss"
PASS_RAN = "pass_ran"
GOOD_PASS = "good_pass_failed"
LIKED = "liked"
CATEGORIES = (GREAT_TRADE, REAL_MISS, PASS_RAN, GOOD_PASS, LIKED)
#: The categories that get a "find it next time" recipe.
RECIPE_CATEGORIES = (GREAT_TRADE, REAL_MISS, PASS_RAN)

#: Walk-away "category" values that name a decision lane, not a setup.
NOT_A_SETUP = frozenset({"m5", "trade", "chart_review", ""})

CATEGORY_WORDS = {
    GREAT_TRADE: "great trade",
    REAL_MISS: "real miss (liked, not traded, it ran)",
    PASS_RAN: "pass that ran",
    GOOD_PASS: "good pass (it did not run)",
    LIKED: "liked pick",
}

#: The D1 environment vocabulary (`indicators.d1_environment.LABELS`).
D1_ENV_WORDS = {
    "trending_up": "trend-up",
    "trending_down": "trend-down",
    "compressed": "compressed",
    "mixed": "mixed",
    UNKNOWN: "unknown",
}
#: The intraday auto-regime vocabulary (`bounce_bot_lib` MARKET_ENVIRONMENTS keys).
INTRADAY_ENV_WORDS = {
    "bullish_strong": "Bullish Strong",
    "bullish_weak": "Bullish Weak",
    "bearish_strong": "Bearish Strong",
    "bearish_weak": "Bearish Weak",
    "neutral_chop": "Neutral / Chop",
    UNKNOWN: "unknown",
}

#: Time-of-day buckets in New York time: (name, start, end, words).
TOD_BUCKETS = (
    ("pre_open", time(0, 0), time(9, 30), "before the open"),
    ("first_hour", time(9, 30), time(10, 30), "first hour"),
    ("late_morning", time(10, 30), time(12, 0), "late morning"),
    ("midday", time(12, 0), time(14, 0), "midday"),
    ("afternoon", time(14, 0), time(15, 0), "afternoon"),
    ("last_hour", time(15, 0), time(16, 0), "last hour"),
    ("after_close", time(16, 0), time(23, 59, 59, 999999), "after the close"),
)
TOD_WORDS = {name: words for name, _s, _e, words in TOD_BUCKETS} | {UNKNOWN: "unknown time"}

INTRADAY_RULE_TEXT = (
    "Auto regime: SPY against its session VWAP and 1-sigma band once 12 five-minute "
    "bars exist (held above/below the band = strong); before that, SPY's day % "
    "against yesterday's close (|0.5%| or more = strong)."
)
D1_RULE_TEXT = (
    "D1 label (d1_environment_v1): SPY's last 10 sessions' range and 20-day SMA slope "
    "in ATR units - range under 3 ATR = compressed, slope beyond 0.5 ATR = trending, "
    "else mixed; written once per session after its close."
)


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _side(value: Any) -> str:
    text = _text(value).upper()
    return {"BUY": "LONG", "SELL": "SHORT"}.get(text, text)


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _flag(value: Any) -> bool | None:
    text = _text(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _tz(inputs: Mapping[str, Any] | None) -> ZoneInfo:
    name = _text((inputs or {}).get("local_tz")) or "America/New_York"
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - an unknown zone falls back to the exchange's
        return ET


def moment(value: Any, local_tz: Any = ET) -> datetime | None:
    """An aware datetime, or None. A naive stamp is desk-local time (attached, not converted)."""
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
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=local_tz)
    return stamp


def _iso(stamp: datetime | None) -> str:
    return stamp.astimezone(ET).isoformat(timespec="seconds") if stamp else ""


def _et_clock(stamp: datetime | None) -> str:
    return stamp.astimezone(ET).strftime("%H:%M ET") if stamp else ""


def tod_bucket(stamp: datetime | None) -> str:
    """The New York time-of-day bucket of an aware moment, or unknown."""
    if stamp is None:
        return UNKNOWN
    clock = stamp.astimezone(ET).time()
    for name, start, end, _words in TOD_BUCKETS:
        if start <= clock < end or (name == "after_close" and clock >= start):
            return name
    return UNKNOWN


def session_close(session: str) -> datetime | None:
    """16:00 New York on the session date (early closes are not modelled here)."""
    try:
        day = date.fromisoformat(_text(session)[:10])
    except ValueError:
        return None
    return datetime.combine(day, time(16, 0), tzinfo=ET)


def session_open(session: str) -> datetime | None:
    try:
        day = date.fromisoformat(_text(session)[:10])
    except ValueError:
        return None
    return datetime.combine(day, time(9, 30), tzinfo=ET)


# ---------------------------------------------------------------------------
# environment, point in time
# ---------------------------------------------------------------------------
def d1_env_at(rows: Iterable[Mapping[str, Any]], at: datetime | None, *, benchmark: str = "SPY") -> dict[str, Any]:
    """The newest D1 label WRITTEN at or before ``at``. Unknown when none was."""
    if at is None:
        return {"label": UNKNOWN, "why": "no pick time"}
    best: Mapping[str, Any] | None = None
    for row in rows or ():
        if _text(row.get("benchmark")).upper() != benchmark:
            continue
        written = moment(row.get("written_at"))
        if written is None or written > at:
            continue
        if best is None or _text(row.get("session")) > _text(best.get("session")):
            best = row
    if best is None:
        return {"label": UNKNOWN, "why": "no D1 label was written before this time"}
    label = _text(best.get("label")) or UNKNOWN
    return {
        "label": label if label in D1_ENV_WORDS else UNKNOWN,
        "session": _text(best.get("session")),
        "written_at": _iso(moment(best.get("written_at"))),
        "range_atr": _float(best.get("range_atr")),
        "slope_atr": _float(best.get("slope_atr")),
        "reason": _text(best.get("reason")),
        "rule_version": _text(best.get("rule_version")),
    }


def intraday_env_at(
    shifts: Iterable[Mapping[str, Any]],
    session: str,
    at: datetime | None,
    *,
    opening: Mapping[str, Any] | None = None,
    local_tz: Any = ET,
) -> dict[str, Any]:
    """The auto regime in force at ``at``: the last shift recorded at or before it.

    Falls back to the session's opening reading when it was recorded before ``at``.
    A shift's ``from_regime`` is NOT used for earlier moments: it was recorded later.
    """
    if at is None:
        return {"label": UNKNOWN, "why": "no pick time"}
    best: tuple[datetime, Mapping[str, Any]] | None = None
    for row in shifts or ():
        if _text(row.get("session_date"))[:10] != session:
            continue
        stamp = moment(row.get("event_at"))
        if stamp is None or stamp > at:
            continue
        if best is None or stamp >= best[0]:
            best = (stamp, row)
    if best is not None:
        label = _text(best[1].get("to_regime")) or UNKNOWN
        return {
            "label": label if label in INTRADAY_ENV_WORDS else UNKNOWN,
            "since": _iso(best[0]),
            "source": _text(best[1].get("source")) or UNKNOWN,
            "basis": "regime_shift",
        }
    if opening and _text(opening.get("date"))[:10] == session:
        stamp = moment(opening.get("recorded_at"), local_tz)
        label = _text(opening.get("env"))
        if stamp is not None and stamp <= at and label in INTRADAY_ENV_WORDS:
            return {"label": label, "since": _iso(stamp), "source": "auto", "basis": "opening_environment"}
    return {"label": UNKNOWN, "why": "no auto regime was recorded before this time"}


# ---------------------------------------------------------------------------
# traits of one pick
# ---------------------------------------------------------------------------
def _latest_before(
    rows: Iterable[Mapping[str, Any]], at: datetime | None, stamp_key: str, local_tz: Any
) -> tuple[datetime, Mapping[str, Any]] | None:
    best: tuple[datetime, Mapping[str, Any]] | None = None
    for row in rows:
        stamp = moment(row.get(stamp_key), local_tz)
        if stamp is None or (at is not None and stamp > at):
            continue
        if best is None or stamp >= best[0]:
            best = (stamp, row)
    return best


def _earliest(rows: Iterable[Mapping[str, Any]], stamp_key: str, local_tz: Any) -> datetime | None:
    stamps = [s for s in (moment(r.get(stamp_key), local_tz) for r in rows) if s is not None]
    return min(stamps) if stamps else None


def d1_setup_at(
    events: Iterable[Mapping[str, Any]], symbol: str, side: str, at: datetime | None
) -> dict[str, Any]:
    """The newest Setup Tracker row open on (symbol, side) as recorded at or before ``at``."""
    mine = [
        row for row in events or ()
        if _text(row.get("symbol")).upper() == symbol
        and _side(row.get("side") or row.get("state_side")) == side
        and _text(row.get("event_type")) in {"initial", "reopened", "transition"}
    ]
    hit = _latest_before(
        [row for row in mine if _text(row.get("state_setup_status")).upper() in {"", "OPEN"}],
        at, "event_at", ET,
    )
    if hit is None:
        return {}
    stamp, row = hit
    setup_id = _text(row.get("setup_id"))
    first = _earliest(
        [r for r in mine if _text(r.get("setup_id")) == setup_id and _text(r.get("event_type")) == "initial"],
        "event_at", ET,
    )
    return {
        "setup_id": setup_id,
        "family": _text(row.get("state_setup_family")) or UNKNOWN,
        "bucket": _text(row.get("state_priority_bucket")) or UNKNOWN,
        "scan_date": _text(row.get("scan_date")),
        "recorded_at": _iso(stamp),
        "first_seen_at": _iso(first),
    }


def m5_alerts_for(
    alerts: Iterable[Mapping[str, Any]], symbol: str, side: str, session: str, local_tz: Any
) -> list[tuple[datetime, Mapping[str, Any]]]:
    """This session's M5 alerts on (symbol, side), oldest first, with aware times."""
    out = []
    for row in alerts or ():
        if _text(row.get("trade_date"))[:10] != session:
            continue
        if _text(row.get("symbol")).upper() != symbol or _side(row.get("direction")) != side:
            continue
        stamp = moment(f"{session}T{_text(row.get('time_local'))}", local_tz)
        if stamp is not None:
            out.append((stamp, row))
    out.sort(key=lambda item: item[0])
    return out


def _grade_key(bounce_type: Any, side: str) -> str:
    return f"{_text(bounce_type).lower()}|{side}"


def m5_grade_at(
    events: Iterable[Mapping[str, Any]], bounce_types: str, side: str, session: str
) -> dict[str, Any]:
    """The day-trade grade from outcomes of sessions BEFORE ``session`` (setup_grades' ladder)."""
    import setup_grades

    before = [e for e in events or () if _text(e.get("trade_date")) < session and e.get("result")]
    if not before:
        return {"grade": UNKNOWN, "why": "no earlier day-trade outcomes in the window"}
    cells = setup_grades.daytrade_cells(
        {"trade_date": e["trade_date"], "side": _side(e.get("side")), "bounce_type": e.get("bounce_type"),
         "result": e["result"]}
        for e in before
    )
    lookup = {str(cell["key"]): cell for cell in cells}
    parts = "-".join(p.strip() for p in _text(bounce_types).replace(",", "-").split("-") if p.strip())
    grade = setup_grades.daytrade_grade_for_alert(lookup, parts, side)
    return {"grade": grade, "basis": "outcomes before the session"}


def swing_grade_at(grades: Mapping[str, Any] | None, side: str, bucket: str, family: str, session: str) -> dict[str, Any]:
    """The swing grade the desk showed: only a grades file built BEFORE the session counts."""
    payload = grades or {}
    as_of = _text(payload.get("as_of"))[:10]
    if UNKNOWN in (bucket, family) or not payload:
        return {"grade": UNKNOWN, "grade_now": UNKNOWN, "as_of": as_of,
                "why": "no Setup Tracker row or no grades file to grade it by"}
    key = "|".join((side, bucket.lower(), (family or "general").lower()))
    cell = next((c for c in payload.get("swing") or () if _text(c.get("key")) == key), None)
    now = _text(cell.get("grade")) if cell else "New"
    if as_of and as_of < session:
        return {"grade": now, "grade_now": now, "as_of": as_of, "basis": "grades file built before the session"}
    return {
        "grade": UNKNOWN,
        "grade_now": now,
        "as_of": as_of,
        "why": "the desk keeps only the latest grades file, built after this pick",
    }


def _grades_then(reader: Any, at: datetime | None) -> Mapping[str, Any] | None:
    """The newest grades snapshot written at or before `at`, when a history reader is given."""
    if not callable(reader) or at is None:
        return None
    try:
        payload = reader(at)
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def traits_for(pick: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Every trait of one pick as known AT its pick time. Missing is unknown."""
    tz = _tz(inputs)
    session = _text(inputs.get("session"))[:10]
    symbol = _text(pick.get("symbol")).upper()
    side = _side(pick.get("side"))
    timeframe = _text(pick.get("timeframe")).upper() or UNKNOWN
    at = moment(pick.get("pick_at"), tz)
    traits: dict[str, Any] = {
        "timeframe": timeframe,
        "side": side or UNKNOWN,
        "pick_at": _iso(at),
        "time_of_day": tod_bucket(at),
        "sector": _text((inputs.get("sectors") or {}).get(symbol)) or UNKNOWN,
        "family": UNKNOWN,
        "claim_setup": UNKNOWN,
        "bucket": UNKNOWN,
        "setup_id": "",
        "grade": UNKNOWN,
        "grade_now": UNKNOWN,
        "alert_kind": UNKNOWN,
        "alert_tier": UNKNOWN,
        "rs_vs_spy": None,
        "rs_signal": UNKNOWN,
    }
    traits["env_d1"] = d1_env_at(inputs.get("d1_environment") or (), at)
    traits["env_intraday"] = intraday_env_at(
        inputs.get("regime_shifts") or (), session, at,
        opening=inputs.get("opening_environment"), local_tz=tz,
    )

    setup = d1_setup_at(inputs.get("tracker_events") or (), symbol, side, at)
    claimed_family = _text(pick.get("family"))
    if setup:
        traits.update(family=setup["family"], bucket=setup["bucket"], setup_id=setup["setup_id"])
        traits["setup_recorded_at"] = setup["recorded_at"]
    if claimed_family and claimed_family not in NOT_A_SETUP:
        traits["claim_setup"] = claimed_family  # the claim row recorded it at pick time
    if timeframe == "D1":
        g = swing_grade_at(inputs.get("grades_now"), side, traits["bucket"], traits["family"], session)
        then = _grades_then(inputs.get("grades_as_of"), at)
        # The dated history is the primary source; the latest file is only a fallback.
        if then:
            past = swing_grade_at({**then, "as_of": "0000-00-00"}, side, traits["bucket"], traits["family"], session)
            if past["grade"] != UNKNOWN:
                g = {**g, "grade": past["grade"], "basis": f"grade history written {then.get('written_at', '')}"}
        traits["grade"], traits["grade_now"] = g["grade"], g.get("grade_now", UNKNOWN)
        traits["grade_basis"] = g.get("basis") or g.get("why", "")

    alerts = m5_alerts_for(inputs.get("m5_alerts") or (), symbol, side, session, tz)
    before = [(s, r) for s, r in alerts if at is None or s <= at]
    if before:
        stamp, row = before[-1]
        traits["alert_kind"] = _text(row.get("bounce_types")) or UNKNOWN
        traits["alert_tier"] = _text(row.get("tier")) or UNKNOWN
        traits["alert_at"] = _iso(stamp)
    event = _m5_event_at(inputs.get("m5_events") or (), symbol, side, session, at, tz)
    if event:
        traits["rs_vs_spy"] = event.get("rrs_spy")
        traits["rs_signal"] = event.get("rs_signal") or UNKNOWN
        if traits["sector"] == UNKNOWN:
            traits["sector"] = event.get("sector") or UNKNOWN
        if traits["env_intraday"]["label"] == UNKNOWN and event.get("env_intraday") in INTRADAY_ENV_WORDS:
            traits["env_intraday"] = {
                "label": event["env_intraday"], "since": event.get("entry_at", ""),
                "source": "alert context", "basis": "m5_alert_context",
            }
        if traits["alert_kind"] == UNKNOWN:
            traits["alert_kind"] = event.get("bounce_type") or UNKNOWN
    if timeframe == "M5" and traits["alert_kind"] != UNKNOWN:
        g = m5_grade_at(inputs.get("m5_events") or (), traits["alert_kind"], side, session)
        traits["grade"] = g["grade"]
        traits["grade_basis"] = g.get("basis") or g.get("why", "")
    return traits


def _m5_event_at(events, symbol, side, session, at, tz) -> Mapping[str, Any] | None:
    mine = [
        e for e in events
        if _text(e.get("trade_date")) == session and _text(e.get("symbol")).upper() == symbol
        and _side(e.get("side")) == side
    ]
    hit = _latest_before(mine, at, "entry_at", tz)
    return hit[1] if hit else None


def _nearest_join(rows, at, close, stamp_key, tz):
    """The newest add at or before the pick; else the first add after it, up to the close."""
    hit = _latest_before(rows, at, stamp_key, tz) if at is not None else None
    if hit is not None:
        return hit
    later = [(s, r) for r in rows if (s := moment(r.get(stamp_key), tz)) is not None and (close is None or s <= close)]
    return min(later, key=lambda item: item[0]) if later else None


def surfaced(pick: Mapping[str, Any], traits: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Where the desk put this name in front of the trader, first time per surface."""
    tz = _tz(inputs)
    session = _text(inputs.get("session"))[:10]
    symbol = _text(pick.get("symbol")).upper()
    side = _side(pick.get("side"))
    at = moment(pick.get("pick_at"), tz)
    close = session_close(session)
    found: list[dict[str, Any]] = []

    def add(surface: str, stamp: datetime | None, detail: str = "") -> None:
        if stamp is not None and close is not None and stamp <= close:
            found.append({"surface": surface, "at": _iso(stamp), "detail": detail, "_t": stamp})

    setup = d1_setup_at(inputs.get("tracker_events") or (), symbol, side, close)
    if setup:
        add("scan row (Setup Tracker)", moment(setup.get("first_seen_at") or setup.get("recorded_at")),
            f"{setup['family']} / {setup['bucket']}")
    alerts = m5_alerts_for(inputs.get("m5_alerts") or (), symbol, side, session, tz)
    if alerts:
        add("M5 alert", alerts[0][0], f"{_text(alerts[0][1].get('bounce_types'))} tier {_text(alerts[0][1].get('tier'))}")
    focus = [
        r for r in inputs.get("focus_events") or ()
        if _text(r.get("symbol")).upper() == symbol and _side(r.get("side")) == side
        and _text(r.get("event_type")) == "joined"
    ]
    hit = _nearest_join(focus, at, close, "event_at", tz)
    if hit:
        add("Focus", hit[0], _text(hit[1].get("category")))
    watch = [
        r for r in inputs.get("watchlist_events") or ()
        if _text(r.get("symbol")).upper() == symbol and _text(r.get("action")) == "add"
        and (not _text(r.get("side")) or _side(r.get("side")) == side)
    ]
    hit = _nearest_join(watch, at, close, "ts", tz)
    if hit:
        add("watchlist", hit[0], _text(hit[1].get("list")))
    found.sort(key=lambda item: item["_t"])
    first = found[0]["_t"] if found else None
    for item in found:
        item.pop("_t")
    opening = session_open(session)
    return {
        "surfaces": found,
        "first_at": _iso(first),
        "before_pick": (first <= at) if (first and at) else None,
        "before_open": (first <= opening) if (first and opening) else None,
        "words": (
            f"first on the desk {found[0]['surface']} at {_et_clock(first)}"
            if found else "the desk has no record of surfacing it"
        ),
    }


def saw_it(pick: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Did the trader see it? A decision or trade is a yes; else the review events say."""
    tz = _tz(inputs)
    session = _text(inputs.get("session"))[:10]
    symbol = _text(pick.get("symbol")).upper()
    # A `hidden_by_show` row records an alert the Show filter kept OFF the feed
    # (P8b B6); it is evidence the trader did not see it, never that they did.
    rows = [
        r for r in inputs.get("review_events") or ()
        if _text(r.get("symbol")).upper() == symbol and _text(r.get("trade_date"))[:10] == session
        and _text(r.get("action")) != "hidden_by_show"
    ]
    first = _earliest(rows, "ts", tz)
    actions = sorted({_text(r.get("action")) for r in rows if _text(r.get("action"))})
    if _text(pick.get("source")) in {"trade", "rejected", "liked_not_traded", "claimed_d1"}:
        return {"saw": "yes", "basis": _text(pick.get("what_you_did")) or _text(pick.get("source")),
                "first_review_at": _iso(first), "review_actions": actions}
    if rows:
        return {"saw": "yes", "basis": "review events", "first_review_at": _iso(first), "review_actions": actions}
    return {"saw": UNKNOWN, "basis": "no decision and no review event", "first_review_at": "", "review_actions": []}


# ---------------------------------------------------------------------------
# lately: observations and cohorts
# ---------------------------------------------------------------------------
def m5_event_summaries(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One summary per M5 alert from the outcome log: context at entry, bracket result, final R."""
    import setup_grades

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows or ():
        event_id = _text(row.get("event_id"))
        if event_id:
            grouped.setdefault(event_id, []).append(row)
    results = {r["event_id"]: r for r in setup_grades.bracket_results(
        row for group in grouped.values() for row in group
    )}
    out = []
    for event_id, group in grouped.items():
        first = group[0]
        context: Mapping[str, Any] = {}
        for row in group:
            try:
                parsed = json.loads(row.get("context_json") or "{}")
            except (TypeError, ValueError):
                parsed = {}
            if isinstance(parsed, Mapping) and parsed:
                context = parsed
                break
        final = [r for r in group if _text(r.get("event_type")).lower() == "final"]
        result = results.get(event_id, {}).get("result")
        out.append({
            "event_id": event_id,
            "trade_date": _text(first.get("trade_date")),
            "symbol": _text(first.get("symbol")).upper(),
            "side": _side(first.get("direction")),
            "bounce_type": results.get(event_id, {}).get("bounce_type", ""),
            "entry_at": _text(first.get("entry_time")),
            "env_intraday": _text(context.get("market_environment")) or UNKNOWN,
            "sector": _text(context.get("sector")) or UNKNOWN,
            "rrs_spy": _float(context.get("rrs_spy")),
            "rs_signal": _text(context.get("rrs_spy_signal")) or UNKNOWN,
            "result": result if result in (setup_grades.WIN, setup_grades.LOSS) else "",
            "close_r": _float(final[-1].get("close_r")) if final else None,
        })
    return out


def m5_observations(events: Iterable[Mapping[str, Any]], inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Measured M5 alerts in the window: +1R before -1R is a hit, final R is the value."""
    from held_run_score import bounce_components

    tz = _tz(inputs)
    window = set(inputs.get("window_sessions") or ())
    d1_rows = inputs.get("d1_environment") or ()
    out = []
    for e in events or ():
        if window and _text(e.get("trade_date")) not in window:
            continue
        if e.get("result") not in ("win", "loss"):
            continue
        at = moment(e.get("entry_at"), tz)
        out.append({
            "timeframe": "M5",
            "session": _text(e.get("trade_date")),
            "symbol": _text(e.get("symbol")),
            "side": _side(e.get("side")),
            "alert_kind": tuple(bounce_components(e.get("bounce_type"))) or (UNKNOWN,),
            "time_of_day": tod_bucket(at),
            "sector": _text(e.get("sector")) or UNKNOWN,
            "rs_signal": _text(e.get("rs_signal")) or UNKNOWN,
            "env_intraday": _text(e.get("env_intraday")) if e.get("env_intraday") in INTRADAY_ENV_WORDS else UNKNOWN,
            "env_d1": d1_env_at(d1_rows, at)["label"],
            "hit": e["result"] == "win",
            "value": _float(e.get("close_r")),
        })
    return out


def d1_observations(rows: Iterable[Mapping[str, Any]], inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Measured D1 scan rows in the window at the D1 horizon: favorable is a hit, side return % the value."""
    window = set(inputs.get("window_sessions") or ())
    sectors = inputs.get("sectors") or {}
    d1_rows = inputs.get("d1_environment") or ()
    seen: set[str] = set()
    out = []
    for row in rows or ():
        scan_date = _text(row.get("scan_date"))[:10]
        if window and scan_date not in window:
            continue
        if _text(row.get("horizon_sessions")) != str(D1_HORIZON) or _flag(row.get("measured")) is not True:
            continue
        favorable = _flag(row.get("favorable"))
        if favorable is None:
            continue
        key = _text(row.get("observation_id")) or f"{row.get('scan_row_id')}:{D1_HORIZON}"
        if key in seen:
            continue
        seen.add(key)
        symbol = _text(row.get("symbol")).upper()
        out.append({
            "timeframe": "D1",
            "session": scan_date,
            "symbol": symbol,
            "side": _side(row.get("side")),
            "family": _text(row.get("setup_family")) or UNKNOWN,
            "bucket": _text(row.get("priority_bucket")) or UNKNOWN,
            "sector": _text(sectors.get(symbol)) or UNKNOWN,
            "env_d1": d1_env_at(d1_rows, session_close(scan_date))["label"],
            "hit": favorable,
            "value": _float(row.get("side_return_pct")),
        })
    return out


def _matches(obs: Mapping[str, Any], trait: str, value: Any) -> bool:
    have = obs.get(trait)
    if isinstance(have, tuple):
        return value in have
    return have == value


def cohort(obs: Sequence[Mapping[str, Any]], *, split: bool = True) -> dict[str, Any]:
    """n, hits, hit rate and median value; "too few to tell" under MIN_COHORT_N."""
    n = len(obs)
    hits = sum(1 for o in obs if o.get("hit"))
    values = [o["value"] for o in obs if o.get("value") is not None]
    unit = "R" if obs and obs[0].get("timeframe") == "M5" else "%"
    stats: dict[str, Any] = {
        "n": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "median": round(statistics.median(values), 3) if values else None,
        "unit": unit,
        "too_few": n < MIN_COHORT_N,
    }
    stats["words"] = _cohort_words(stats)
    if split:
        for env_key in ("env_d1", "env_intraday"):
            labels = sorted({_text(o.get(env_key)) for o in obs if o.get(env_key)})
            if labels:
                stats[f"by_{env_key}"] = {
                    label: cohort([o for o in obs if o.get(env_key) == label], split=False) for label in labels
                }
    return stats


def _cohort_words(stats: Mapping[str, Any]) -> str:
    n = stats["n"]
    if n == 0:
        return "no measured rows lately"
    if stats["too_few"]:
        return f"too few to tell (n {n})"
    median = stats["median"]
    tail = f", median {median:+.2f}{stats['unit']}" if median is not None else ""
    return f"{stats['hits']} of {n} worked ({stats['hit_rate'] * 100:.0f}%){tail}"


D1_TRAITS = ("family", "side", "bucket", "sector", "env_d1")
M5_TRAITS = ("alert_kind", "side", "time_of_day", "sector", "rs_signal", "env_intraday", "env_d1")


def trait_tables(obs: Sequence[Mapping[str, Any]], traits: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Per trait, per value: the cohort, best measured first, too-few rows last."""
    tables: dict[str, list[dict[str, Any]]] = {}
    for trait in traits:
        values: set[str] = set()
        for o in obs:
            have = o.get(trait)
            values.update(have if isinstance(have, tuple) else (have,))
        rows = []
        for value in sorted(v for v in values if v):
            stats = cohort([o for o in obs if _matches(o, trait, value)])
            rows.append({"value": value, **stats})
        rows.sort(key=lambda r: (r["too_few"], -(r["hit_rate"] or 0.0), -r["n"], r["value"]))
        tables[trait] = rows
    return tables


# ---------------------------------------------------------------------------
# find it next time
# ---------------------------------------------------------------------------
def recipe_parts(traits: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """``(trait, value, words)`` for each KNOWN trait of the recipe, in reading order."""
    parts: list[tuple[str, str, str]] = []
    tf = traits.get("timeframe")
    if tf == "D1":
        setup = traits.get("claim_setup", UNKNOWN)
        fam = traits.get("family", UNKNOWN)
        if setup != UNKNOWN:
            parts.append(("claim_setup", setup, f"D1 {setup.replace('_', ' ')}"))
        if fam != UNKNOWN and fam != setup:
            parts.append(("family", fam, f"tracker family {fam.replace('_', ' ')}"))
        if setup == UNKNOWN and fam == UNKNOWN:
            parts.append(("timeframe", "D1", "D1 swing"))
    elif tf == "M5":
        kind = traits.get("alert_kind")
        parts.append(("alert_kind", kind, f"M5 {kind.replace('_', ' ')}") if kind != UNKNOWN else ("timeframe", "M5", "M5 day trade"))
    if traits.get("side") not in (None, UNKNOWN, ""):
        parts.append(("side", traits["side"], traits["side"].lower()))
    if traits.get("grade") not in (None, UNKNOWN):
        parts.append(("grade", traits["grade"], f"grade {traits['grade']}"))
    d1 = (traits.get("env_d1") or {}).get("label", UNKNOWN)
    if d1 != UNKNOWN:
        parts.append(("env_d1", d1, f"{D1_ENV_WORDS[d1]} market"))
    intraday = (traits.get("env_intraday") or {}).get("label", UNKNOWN)
    if intraday != UNKNOWN:
        parts.append(("env_intraday", intraday, f"{INTRADAY_ENV_WORDS[intraday]} tape"))
    if traits.get("time_of_day") not in (None, UNKNOWN):
        parts.append(("time_of_day", traits["time_of_day"], TOD_WORDS[traits["time_of_day"]]))
    if traits.get("sector") not in (None, UNKNOWN, ""):
        parts.append(("sector", traits["sector"], traits["sector"]))
    if traits.get("rs_signal") not in (None, UNKNOWN, ""):
        parts.append(("rs_signal", traits["rs_signal"], f"{traits['rs_signal']} vs SPY"))
    return parts


def recipe_stats(traits: Mapping[str, Any], obs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How the recipe did lately: its core (setup + side), the core in this environment, each trait alone."""
    parts = recipe_parts(traits)
    # The trader's claimed setup is looked up as a tracker family when the tracker had none.
    usable = []
    has_family = any(t == "family" for t, _v, _w in parts)
    for t, v, w in parts:
        if t in ("grade", "timeframe"):
            continue
        if t == "claim_setup":
            if has_family:
                continue
            t = "family"
        usable.append((t, v, w))
    per_trait = {}
    for trait, value, words in usable:
        if obs and trait in obs[0]:
            per_trait[trait] = {"value": value, "words": words, **cohort([o for o in obs if _matches(o, trait, value)], split=False)}
    core_traits = [(t, v) for t, v, _w in usable if t in ("family", "alert_kind", "side")]
    core = [o for o in obs if all(_matches(o, t, v) for t, v in core_traits)] if core_traits else []
    env_key = "env_d1" if traits.get("timeframe") == "D1" else "env_intraday"
    env_label = (traits.get(env_key) or {}).get("label", UNKNOWN)
    in_env = [o for o in core if o.get(env_key) == env_label] if env_label != UNKNOWN else []
    return {
        "core": {"traits": dict(core_traits), **cohort(core)} if core_traits else None,
        "core_in_this_environment": (
            {"environment": env_label, **cohort(in_env, split=False)} if env_label != UNKNOWN and core_traits else None
        ),
        "per_trait": per_trait,
    }


def _bucket_label(bucket: str) -> str:
    """The Bucket filter's own label for a bucket key (`ui.models.setup`, an import-light dict)."""
    try:
        from ui.models.setup import SETUP_BUCKET_LABELS
    except Exception:  # noqa: BLE001 - the key reads fine on its own
        SETUP_BUCKET_LABELS = {}
    key = bucket.strip().lower()
    return SETUP_BUCKET_LABELS.get(key, key.replace("_", " ").title())


def where_to_look(traits: Mapping[str, Any], surfaced_info: Mapping[str, Any]) -> list[str]:
    """The existing desk filter/sort/page that shows such names. Says so when none exists."""
    out: list[str] = []
    side = traits.get("side", UNKNOWN)
    sector = traits.get("sector", UNKNOWN)
    if traits.get("timeframe") == "D1":
        bucket = traits.get("bucket", UNKNOWN)
        line = "Master AVWAP page (Setup Tracker scan): sort by the Grade badge (best first)"
        if side in ("LONG", "SHORT"):
            line += f", Side = {side}"
        if bucket != UNKNOWN:
            line += f", Bucket = {_bucket_label(bucket)}"
        if sector != UNKNOWN:
            line += f", type '{sector}' in 'Filter symbol, tag, level'"
        out.append(line)
        if traits.get("grade_now") not in (None, UNKNOWN):
            out.append(f"Today this family grades {traits['grade_now']} on that page.")
        out.append("No setup-family filter exists on that page.")
    elif traits.get("timeframe") == "M5":
        tier = traits.get("alert_tier", UNKNOWN)
        line = "Trading Desk M5 alerts column: tick 'Prioritise what is working' (Working lately strip) to sort by grade"
        out.append(line)
        if tier in ("S", "A", "B"):
            label = {"S": "S tier / PROVEN only", "A": "A tier and above", "B": "B tier and above"}[tier]
            out.append(f"Alert Center 'Alerts' tab: Min tier = '{label}' keeps alerts like this one")
        out.append("The 'Working now' strip shows how today's alerts of this kind are doing, in R.")
    if any(s.get("surface") == "Focus" for s in surfaced_info.get("surfaces") or ()):
        out.append("It was on Focus: Alert Center 'D1 Focus' tab / the Focus lists.")
    out.append("No desk filter splits by market environment or time of day yet.")
    return out


def _result_words(pick: Mapping[str, Any]) -> str:
    made = _float(pick.get("you_made"))
    if made is not None:
        return f"you made {made:+.2f}"
    ran = _float(pick.get("ran_after_pct"))
    verdict = _text(pick.get("verdict"))
    if ran is not None:
        return f"ran {ran:+.1f}% after ({verdict or 'unmeasured'})"
    return verdict or _text(pick.get("state")) or UNKNOWN


def findability_for_session(session_date: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The session's notable names: traits then, recipe, lately stats, where to look, surfaced when."""
    session = _text(session_date)[:10]
    scoped = dict(inputs or {})
    scoped["session"] = session
    d1_obs = d1_observations(scoped.get("d1_scan_outcomes") or (), scoped)
    m5_obs = m5_observations(scoped.get("m5_events") or (), scoped)
    picks = []
    for pick in scoped.get("notable") or ():
        traits = traits_for(pick, scoped)
        where = surfaced(pick, traits, scoped)
        item = {
            "symbol": _text(pick.get("symbol")).upper(),
            "side": _side(pick.get("side")),
            "category": _text(pick.get("category")),
            "category_words": CATEGORY_WORDS.get(_text(pick.get("category")), ""),
            "timeframe": traits["timeframe"],
            "result": _result_words(pick),
            "what_you_did": _text(pick.get("what_you_did")),
            "reason": _text(pick.get("reason")),
            "traits": traits,
            "saw_it": saw_it(pick, scoped),
            "surfaced": where,
        }
        if item["category"] in RECIPE_CATEGORIES:
            obs = d1_obs if traits["timeframe"] == "D1" else m5_obs if traits["timeframe"] == "M5" else []
            item["recipe"] = " · ".join(w for _t, _v, w in recipe_parts(traits)) or "no known traits"
            item["recipe_lately"] = recipe_stats(traits, obs)
            item["where_to_look"] = where_to_look(traits, where)
        picks.append(item)
    picks.sort(key=lambda p: (CATEGORIES.index(p["category"]) if p["category"] in CATEGORIES else 99, p["symbol"]))
    return {
        "schema": SCHEMA,
        "session": session,
        "window_sessions": list(scoped.get("window_sessions") or ()),
        "min_cohort_n": MIN_COHORT_N,
        "counts": {c: sum(1 for p in picks if p["category"] == c) for c in CATEGORIES},
        "picks": picks,
        "lately": {
            "d1": {"horizon_sessions": D1_HORIZON, "n": len(d1_obs), "tables": trait_tables(d1_obs, D1_TRAITS)},
            "m5": {"n": len(m5_obs), "tables": trait_tables(m5_obs, M5_TRAITS)},
        },
        "unread": list(scoped.get("unread") or ()),
        "gaps": list(GAPS),
    }


#: Traits the desk does not record at pick time; said, never guessed.
GAPS = (
    "Swing grade at pick time: only the latest setup_grades_latest.json is kept, so a pick is graded only "
    "when that file was built before its session.",
    "Sector is today's symbol classification, not a dated one.",
    "RS vs SPY is recorded only on M5 alerts (alert context); D1 picks have none.",
    "The auto regime before a session's first shift is not recorded at the time; it reads unknown.",
    "M5 alert tier is on the alert log only, not on the outcome log, so the lately tables have no tier split.",
    "Early closes are not modelled in time-of-day buckets.",
)


# ---------------------------------------------------------------------------
# environment clarity
# ---------------------------------------------------------------------------
def environment_vocabulary() -> dict[str, list[dict[str, str]]]:
    """Every label the auto environment can give, for an "agree / it was actually X" control."""
    return {
        "d1": [{"label": k, "words": v} for k, v in D1_ENV_WORDS.items()],
        "intraday": [{"label": k, "words": v} for k, v in INTRADAY_ENV_WORDS.items()],
    }


def _in_environment(obs: Sequence[Mapping[str, Any]], env_key: str, label: str, trait: str) -> dict[str, Any]:
    """Setups lately in ONE environment label, best first, with how many rows each label has."""
    coverage: dict[str, int] = {}
    for o in obs:
        coverage[_text(o.get(env_key)) or UNKNOWN] = coverage.get(_text(o.get(env_key)) or UNKNOWN, 0) + 1
    rows = trait_tables([o for o in obs if o.get(env_key) == label], (trait,)).get(trait, []) if label != UNKNOWN else []
    out: dict[str, Any] = {"environment": label, f"by_{trait}": rows, "coverage": dict(sorted(coverage.items()))}
    if label == UNKNOWN:
        out["why"] = "the environment is unknown, so there is nothing to compare"
    elif not rows:
        out["why"] = f"no measured rows lately were labelled {label} at the time"
    return out


def environment_summary(session_date: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """What the auto environment said for a session and when, why, and how setups did in it lately."""
    session = _text(session_date)[:10]
    scoped = dict(inputs or {})
    scoped["session"] = session
    tz = _tz(scoped)
    d1_rows = scoped.get("d1_environment") or ()
    opening = session_open(session)
    timeline: list[dict[str, Any]] = []
    known_at_open = d1_env_at(d1_rows, opening)
    if known_at_open["label"] != UNKNOWN:
        timeline.append({
            "at": _iso(opening), "kind": "d1", "label": known_at_open["label"],
            "words": f"D1 label known at the open: {D1_ENV_WORDS[known_at_open['label']]} "
                     f"(for {known_at_open.get('session')}, written {known_at_open.get('written_at')})",
            "evidence": {k: known_at_open.get(k) for k in ("range_atr", "slope_atr", "reason", "rule_version")},
        })
    open_env = scoped.get("opening_environment") or {}
    if _text(open_env.get("date"))[:10] == session and _text(open_env.get("env")) in INTRADAY_ENV_WORDS:
        stamp = moment(open_env.get("recorded_at"), tz)
        timeline.append({
            "at": _iso(stamp), "kind": "intraday", "label": open_env["env"], "source": "auto",
            "words": f"opening auto regime {INTRADAY_ENV_WORDS[open_env['env']]}",
            "evidence": {},
        })
    shifts = sorted(
        (r for r in scoped.get("regime_shifts") or () if _text(r.get("session_date"))[:10] == session),
        key=lambda r: moment(r.get("event_at")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    close = session_close(session)
    for row in shifts:
        stamp = moment(row.get("event_at"))
        label = _text(row.get("to_regime")) or UNKNOWN
        timeline.append({
            "at": _iso(stamp), "kind": "intraday", "label": label, "source": _text(row.get("source")) or UNKNOWN,
            "after_close": bool(stamp and close and stamp > close),
            "words": f"{_et_clock(stamp)} {INTRADAY_ENV_WORDS.get(_text(row.get('from_regime')), _text(row.get('from_regime')) or '?')}"
                     f" -> {INTRADAY_ENV_WORDS.get(label, label)} ({_text(row.get('source')) or 'unknown'})",
            "evidence": {"spy_day_pct": _float(row.get("spy_day_pct")), "detail": _text(row.get("detail"))},
        })
    own = next(
        (r for r in d1_rows if _text(r.get("benchmark")).upper() == "SPY" and _text(r.get("session"))[:10] == session),
        None,
    )
    if own is not None:
        timeline.append({
            "at": _iso(moment(own.get("written_at"))), "kind": "d1_after", "label": _text(own.get("label")) or UNKNOWN,
            "words": f"what kind of day it was (written after): {D1_ENV_WORDS.get(_text(own.get('label')), UNKNOWN)}",
            "evidence": {k: own.get(k) for k in ("range_atr", "slope_atr", "reason", "rule_version", "bars_through")},
        })
    timeline.sort(key=lambda item: item["at"] or "")
    intraday_labels = [t["label"] for t in timeline if t["kind"] == "intraday"]
    time_in: dict[str, float] = {}
    marks = [(moment(t["at"]), t["label"]) for t in timeline if t["kind"] == "intraday"]
    for index, (stamp, label) in enumerate(marks):
        if stamp is None or close is None:
            continue
        start = max(stamp, opening) if opening else stamp
        end = marks[index + 1][0] if index + 1 < len(marks) and marks[index + 1][0] else close
        end = min(end, close)
        if end > start:
            time_in[label] = time_in.get(label, 0.0) + (end - start).total_seconds() / 60.0
    d1_obs = d1_observations(scoped.get("d1_scan_outcomes") or (), scoped)
    m5_obs = m5_observations(scoped.get("m5_events") or (), scoped)
    d1_label = known_at_open["label"]
    main_intraday = max(time_in, key=time_in.get) if time_in else UNKNOWN
    return {
        "schema": ENV_SCHEMA,
        "session": session,
        "timeline": timeline,
        "d1_known_at_open": known_at_open,
        "d1_for_the_session": ({"label": _text(own.get("label")), "written_at": _iso(moment(own.get("written_at")))}
                               if own is not None else {"label": UNKNOWN, "why": "no D1 row for this session yet"}),
        "intraday_labels": intraday_labels,
        "intraday_minutes_by_label": {k: round(v) for k, v in sorted(time_in.items())},
        "main_intraday_label": main_intraday,
        "shift_count": len(shifts),
        "rules": {"d1": D1_RULE_TEXT, "intraday": INTRADAY_RULE_TEXT},
        "lately_in_this_environment": {
            "d1": _in_environment(d1_obs, "env_d1", d1_label, "family"),
            "m5": _in_environment(m5_obs, "env_intraday", main_intraday, "alert_kind"),
        },
        "vocabulary": environment_vocabulary(),
    }


# ===========================================================================
# the reader - worker thread or CLI only, never the Qt thread
# ===========================================================================
def notable_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The session's notable names from a Day Review payload (`DayReviewService.read_day`)."""
    session = _text(payload.get("session_date") or payload.get("session"))[:10]
    walkaway = payload.get("walkaway")
    out: dict[tuple[str, str], dict[str, Any]] = {}

    def put(key, row):
        have = out.get(key)
        if have is None or CATEGORIES.index(row["category"]) < CATEGORIES.index(have["category"]):
            out[key] = row

    for trade in payload.get("trades") or ():
        pnl = _float(trade.get("net_pnl_usd", trade.get("net_pnl")))
        opened = _text(trade.get("opened_at"))
        closed = _text(trade.get("closed_at"))
        if pnl is None or pnl <= 0 or session not in (opened[:10], closed[:10]):
            continue
        side = _side(trade.get("direction"))
        put((_text(trade.get("symbol")).upper(), side), {
            "symbol": trade.get("symbol"), "side": side, "category": GREAT_TRADE,
            "timeframe": "M5" if "day_trade" in _text(trade.get("auto_tag_summary")) else "D1",
            "pick_at": opened, "you_made": pnl, "source": "trade",
            "what_you_did": "traded" if opened[:10] == session else f"closed a trade opened {opened[:10]}",
        })

    def rows(name):
        return tuple(getattr(walkaway, name, ()) or ()) if walkaway is not None else ()

    for name in ("liked_not_traded", "claimed_d1", "rejected"):
        for row in rows(name):
            verdict = _text(getattr(row, "real_miss", ""))
            if name == "rejected":
                if verdict == "run":
                    category = PASS_RAN
                elif verdict == "no_run":
                    category = GOOD_PASS
                else:
                    continue
            else:
                category = REAL_MISS if verdict == "run" and _text(getattr(row, "traded", "no")) != "yes" else LIKED
            ident = getattr(row, "decision_id", ()) or ()
            stamp = getattr(row, "time", None)
            put((_text(row.symbol).upper(), _side(row.side)), {
                "symbol": row.symbol, "side": _side(row.side), "category": category,
                "timeframe": _text(ident[5]).upper() if len(ident) > 5 else UNKNOWN,
                "pick_at": stamp.isoformat() if isinstance(stamp, datetime) else _text(stamp),
                "family": _text(getattr(row, "category", "")),
                "verdict": verdict, "ran_after_pct": getattr(row, "ran_after_pct", None),
                "state": _text(getattr(row, "state", "")), "source": name,
                "what_you_did": _text(getattr(row, "what_you_did", "")), "reason": _text(getattr(row, "reason", "")),
            })
    return list(out.values())


def _read_csv(path: Path, keep=None) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return [dict(r) for r in csv.DictReader(handle) if keep is None or keep(r)]
    except OSError:
        return []


def read_inputs(
    session_date: str,
    *,
    payload: Mapping[str, Any] | None = None,
    extra_symbols: Iterable[str] = (),
) -> dict[str, Any]:
    """Build ``inputs`` from the `project_paths` stores. Read-only; each store in its own guard.

    `extra_symbols` widens the per-name reads (setup tracker, Focus, watchlist) past the
    notable names, so a caller can ask `traits_for` about a losing trade too.
    """
    import project_paths as pp
    import walkaway_day

    session = _text(session_date)[:10]
    unread: list[str] = []
    window = list(walkaway_day.earlier_sessions(session, count=WINDOW_SESSIONS - 1)) + [session]
    first = window[0]
    inputs: dict[str, Any] = {"session": session, "window_sessions": window, "unread": unread}

    def guard(name, fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one unreadable store costs one input
            unread.append(f"{name}: {exc}")
            return default

    try:
        from market_session import get_market_local_timezone

        inputs["local_tz"] = get_market_local_timezone()[1]
    except Exception:  # noqa: BLE001
        inputs["local_tz"] = "America/New_York"

    if payload is None:
        def _day():
            import day_review_bars
            from ui.services.day_review_service import DayReviewService

            original = day_review_bars.fetch_session_bars
            day_review_bars.fetch_session_bars = lambda *a, **k: {}  # never fetch from the recap reader
            try:
                return DayReviewService().read_day(session)
            finally:
                day_review_bars.fetch_session_bars = original
        payload = guard("day review", _day, {})
    inputs["notable"] = guard("notable names", lambda: notable_from_payload(payload or {}), [])
    symbols = {_text(p.get("symbol")).upper() for p in inputs["notable"]}
    symbols |= {_text(s).upper() for s in extra_symbols or () if _text(s)}

    def _d1_env():
        import d1_environment_store

        return [r for r in d1_environment_store.read_rows() if _text(r.get("benchmark")).upper() == "SPY"]
    inputs["d1_environment"] = guard("d1 environment", _d1_env, [])

    def _ledger(stream, schema, types=None, start=first):
        from evidence_ledger import EvidenceLedger

        return EvidenceLedger(stream=stream, schema=schema).read(start=start, end=session, event_types=types).rows

    inputs["regime_shifts"] = guard(
        "regime shifts", lambda: _ledger("market_regime_shifts", "market_regime_shift_v1"), []
    )
    def _opening():
        # The per-session history keeps every day's first read; the live file keeps only the latest day.
        import opening_regime_history

        row = opening_regime_history.opening_regime_for(session)
        if row and row.get("label"):
            return {"date": row["session_date"], "env": row["label"], "recorded_at": row["written_at"]}
        return json.loads(Path(pp.AUTO_OPENING_ENV_FILE).read_text(encoding="utf-8"))

    inputs["opening_environment"] = guard("opening environment", _opening, {})

    def _grades_as_of():
        import setup_grades_history

        return lambda when: setup_grades_history.grades_as_of(when)

    inputs["grades_as_of"] = guard("grades history", _grades_as_of, None)
    inputs["tracker_events"] = guard("setup tracker events", lambda: [
        r for r in _ledger("setup_tracker_events", "setup_tracker_event_v1",
                           ("initial", "reopened", "transition"),
                           start=(date.fromisoformat(session) - timedelta(days=45)).isoformat())
        if _text(r.get("symbol")).upper() in symbols
    ], [])
    inputs["focus_events"] = guard("focus events", lambda: [
        r for r in _ledger("focus_membership_events", "focus_membership_event_v1", ("joined",),
                           start=(date.fromisoformat(session) - timedelta(days=45)).isoformat())
        if _text(r.get("symbol")).upper() in symbols
    ], [])

    def _watch():
        import watchlist_intent_events

        return [r for r in watchlist_intent_events.read_events()
                if _text(r.get("symbol")).upper() in symbols and _text(r.get("market_date"))[:10] <= session]
    inputs["watchlist_events"] = guard("watchlist events", _watch, [])

    def _reviews():
        import review_events

        return [r for r in review_events.load_review_events() if _text(r.get("trade_date"))[:10] == session]
    inputs["review_events"] = guard("review events", _reviews, [])
    inputs["m5_alerts"] = guard("m5 alerts", lambda: _read_csv(
        Path(pp.INTRADAY_BOUNCES_FILE), lambda r: _text(r.get("trade_date"))[:10] == session
    ), [])

    def _events():
        import held_run_score

        rows = held_run_score.read_outcome_rows(
            Path(pp.INTRADAY_BOUNCE_OUTCOMES_FILE), sessions=WINDOW_SESSIONS + 1, as_of=session
        )
        return m5_event_summaries(r for r in rows if _text(r.get("trade_date")) <= session)
    inputs["m5_events"] = guard("m5 outcomes", _events, [])
    inputs["d1_scan_outcomes"] = guard("d1 scan outcomes", lambda: _read_csv(
        Path(pp.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE),
        lambda r: first <= _text(r.get("scan_date"))[:10] <= session and _text(r.get("horizon_sessions")) == str(D1_HORIZON),
    ), [])

    def _sectors():
        import industry_scanner

        wanted = symbols | {_text(r.get("symbol")).upper() for r in inputs["d1_scan_outcomes"]}
        return {s: row["sector"] for s, row in industry_scanner.load_symbol_classifications().items()
                if s in wanted and row.get("sector")}
    inputs["sectors"] = guard("sectors", _sectors, {})
    inputs["grades_now"] = guard("grades", lambda: json.loads(
        (Path(pp.LOCAL_SETTINGS_DIR) / "working_lately" / "setup_grades_latest.json").read_text(encoding="utf-8")
    ), {})
    return inputs


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Print a session's findability and environment summary as JSON.")
    parser.add_argument("--date", required=True, help="session date, YYYY-MM-DD")
    parser.add_argument("--part", choices=("both", "findability", "environment"), default="both")
    args = parser.parse_args(argv)
    try:
        session = date.fromisoformat(args.date).isoformat()
    except ValueError:
        parser.error("--date must be YYYY-MM-DD")
    inputs = read_inputs(session)
    out: dict[str, Any] = {}
    if args.part in ("both", "findability"):
        out["findability"] = findability_for_session(session, inputs)
    if args.part in ("both", "environment"):
        out["environment"] = environment_summary(session, inputs)
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
