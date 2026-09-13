"""A Setup Tracker for the theta plays - packet WS-TH (WISHLIST item 6).

The trader's brief: *"Track, per pick and per day it appears: symbol, scan date,
support set, the score and rank, the chosen strike/expiry/premium. Grade at the
sold put's expiry and at 5/10/20 sessions: did price hold above the strike, max
adverse excursion in ATR, which supports broke first."*

**SHADOW ONLY, and the word is load-bearing.** Nothing in this module reaches
the theta scan, `_theta_support_quality`, `_theta_support_entry`, the option
ranker, the report, a detector, an alert, a watchlist, Focus, the review queue
or `review_policy.json`. It READS the rows the scan already built and writes an
evidence file beside the tracker's own. `legacy.py` is not edited by this packet.

**What the real rows taught this module** (recon + the tester's fixtures,
2026-09-12). Each of these is a place a hand-written recorder would be wrong:

* `SMA_20` is BUILT as a theta support and then DROPPED -
  `_is_valid_theta_support_entry` (legacy.py:21082) refuses it - so a recorded
  support set names SMA_50/100/200 and never SMA_20. The store records what the
  ROW carries, never what the builder attempted.
* **`held` is `level <= close` and is NEVER derived from `distance_atr`.**
  `_theta_support_entry` keeps a level up to `THETA_SUPPORT_ABOVE_TOL_ATR`
  (0.05 ATR) ABOVE the close and CLAMPS the negative distance to `0.0`, so a
  level 0.05 ATR overhead and a level sitting exactly on price record the same
  distance. Only the level separates them.
* **A sold put carries `strike`; a put credit spread carries `short_strike` and
  `long_strike`.** Both shapes are recorded in full - `strike` is the SOLD leg
  either way, so a spread is never a NULL strike.
* `_apply_best_option_to_theta_row` REPLACES `score` with the option's
  `rank_score` and keeps the support score as `base_score`. The report ranks on
  the former, so the store records BOTH and the grade line grades the former.
* IB hands expirations back as `YYYYMMDD`; the store keeps an ISO session date
  so the grader can ask the calendar about it without re-parsing a broker
  format.

**The grain, and the two counts that are never added together** (lead ruling
(b), 2026-09-12): one store row per `(symbol, scan_date, play_type)` - every
appearance is an observation - and the readout's cohort grain is the FIRST
appearance. A name the scan finds eight days running is ONE observation for the
hold rate and eight days of interest beside it, so `repeat_days` sits next to
`n` and is never summed into it.

**The exchange calendar answers "5 sessions", not a weekday walk.** A scan on
2026-06-01 has its 20th session on 2026-06-30, because Juneteenth (06-19) is
not one; twenty business days lands on 06-29. The marks are taken at the exact
session endpoint through the same `_nth_session_after` walk
`session_horizon_outcomes` uses, and a session the calendar has not reached is
`pending` with a reason - never a break.

**Completed bars only.** `as_of` is the last COMPLETED session; a mark, a MAE
window or an expiry grade past it is not read at all.

**RS is `not_measured`, never invented** (lead ruling (d)): today's theta
scoring has no relative-strength term - `_theta_support_quality` takes source
and `distance_atr` and nothing else - so the RS cut is a column that says so.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import market_calendar
from evidence_stats import MIN_REPORTABLE_N
from master_avwap_lib.session_horizon_outcomes import (
    REASON_IMMATURE,
    REASON_NO_TARGET_BAR,
    REASON_TARGET_OUT_OF_RANGE,
)
from project_paths import MASTER_AVWAP_THETA_OUTCOMES_FILE, THETA_PICKS_FILE
from swing_headline import wilson_lower_bound

_log = logging.getLogger(__name__)

#: The session marks every pick is measured at. `HEADLINE_MARK` is the one the
#: readout leads with, and it is stated in words on every surface - a hold rate
#: whose horizon is implied is a hold rate nobody can check.
SESSION_MARKS: tuple[int, ...] = (5, 10, 20)
HEADLINE_MARK = 20

#: Why a pick could not be graded, beyond the three the session-horizon module
#: already names. A pick the desk was never offered an option on has no strike,
#: so "did price hold above the strike" has no answer - grading it False would
#: score a play that did not exist.
REASON_NO_OPTION_QUOTE = "no_option_quote"
REASON_EXPIRY_IMMATURE = "expiry_session_not_complete"
REASON_NO_EXPIRY_BAR = "no_bar_for_expiry_session"
REASON_NO_EXPIRY = "no_expiry_recorded"

#: Ruling (d): the RS cut is a column that says it was never measured.
RS_FLAG_NOT_MEASURED = "not_measured"
RS_NOTE = "theta scoring carries no relative-strength term (_theta_support_quality)"

#: `first_support_broken` when the path never crossed a support that was
#: holding on the scan date. A word, not a blank: a blank reads as "we did not
#: look" and the nearest support would read as a break that never happened.
NO_SUPPORT_BROKEN = "none"

STATUS_MEASURED = "measured"
STATUS_PENDING = "pending"
STATUS_UNMEASURED = "unmeasured"

#: The outcome CSV header, in order. The readout joins on it and the Setup
#: Tracker's Theta tab reads it; a column is appended, never reordered.
THETA_OUTCOME_COLUMNS: tuple[str, ...] = (
    "symbol",
    "scan_date",
    "first_seen_scan_date",
    "bar_date",
    "play_type",
    "rank",
    "support_combo",
    "supports_held",
    "score",
    "base_score",
    "strike",
    "short_strike",
    "long_strike",
    "expiry",
    "premium",
    "atr",
    "close",
    "held_5",
    "held_10",
    "held_20",
    "held_at_expiry",
    "mae_atr",
    "first_support_broken",
    "status",
    "unmeasured_reason",
    "rs_flag",
    "rs_note",
)

#: What the Theta tab says before the first grade has been written.
THETA_NO_EXPORT_SENTENCE = (
    "No export yet: the theta picks the scan records are graded at "
    f"{HEADLINE_MARK} sessions, and none has matured. Nothing on this tab is a "
    "recommendation."
)


# ---------------------------------------------------------------------------
# small readers, shared by the recorder, the grader and the readout
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A float, or None. A blank, a NaN and an unparseable string are all None."""
    if value is None or value is True or value is False:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _int(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(round(number))


def _iso_date(value: Any) -> str:
    """`YYYY-MM-DD` from a date, a datetime or a string. "" when unreadable."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return ""


def _parse_date(value: Any) -> date | None:
    text = _iso_date(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _expiry_iso(value: Any) -> str | None:
    """An ISO session date from IB's `YYYYMMDD`, an ISO string or a date.

    `_format_option_expiration` hands back `20260717`; the store keeps
    `2026-07-17` so the grader asks the exchange calendar a question about a
    date rather than about a broker's wire format.
    """
    if isinstance(value, (date, datetime)):
        return _iso_date(value) or None
    text = str(value or "").strip()
    if not text:
        return None
    iso = _iso_date(text)
    if iso:
        return iso
    if len(text) == 8 and text.isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:])).isoformat()
        except ValueError:
            return None
    return None


def _tristate(value: Any) -> bool | None:
    """True / False / None, off a native bool OR a CSV cell.

    The export is a CSV, so `True` arrives as the STRING `"True"` and an
    unmeasured cell arrives as the empty string. A reader that tests
    ``value is True`` calls every exported row a break; one that tests
    truthiness calls `""` a hold. Both have happened on this desk.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every readable row. An unreadable store is an EMPTY one, never a raise."""
    rows: list[dict[str, Any]] = []
    try:
        if not Path(path).is_file():
            return rows
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        _log.debug("Theta pick store unreadable at %s", path, exc_info=True)
    return rows


def read_theta_picks(path: Path | None = None) -> list[dict[str, Any]]:
    """The recorded picks. Read-only; a missing store is an empty list."""
    return _read_jsonl(Path(path) if path is not None else THETA_PICKS_FILE)


# ---------------------------------------------------------------------------
# Item 1 - the recorder
# ---------------------------------------------------------------------------


def _pick_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol") or "").strip().upper(),
        str(row.get("scan_date") or "").strip(),
        str(row.get("play_type") or "").strip().lower(),
    )


def _support_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The scan's own support entries, plus the one fact it does not carry.

    `held` is `level <= close`. It is NOT `distance_atr == 0`: the scan keeps a
    level up to 0.05 ATR overhead and clamps that negative distance to zero, so
    an overhead level and a level on price are indistinguishable by distance.
    """
    close = _number(row.get("last_close"))
    out: list[dict[str, Any]] = []
    for entry in row.get("supports") or []:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        level = _number(entry.get("level"))
        out.append(
            {
                "label": label,
                "source": str(entry.get("source") or "").strip(),
                "level": level,
                "distance_atr": _number(entry.get("distance_atr")),
                "held": bool(level is not None and close is not None and level <= close),
            }
        )
    return out


def _ranked_report_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """The rows in the order the REPORT prints them, so `rank` is what it shows.

    `_sort_theta_report_rows` is the report's own sort and is called read-only.
    A failure to reach it falls back to the order the scan handed the rows over
    - a rank the report disagrees with is worth less than no store at all.
    """
    dict_rows = [row for row in rows or [] if isinstance(row, dict)]
    if not dict_rows:
        return []
    try:
        from master_avwap_lib.legacy import _sort_theta_report_rows

        return list(_sort_theta_report_rows(dict_rows))
    except Exception:  # noqa: BLE001 - never let a sort cost the store
        _log.debug("Theta report sort unavailable; recording in scan order", exc_info=True)
        return dict_rows


def _build_pick_row(
    row: Mapping[str, Any],
    *,
    scan_date: str,
    default_play_type: str,
    rank: int,
    now: Any,
) -> dict[str, Any] | None:
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    play_type = str(row.get("play_type") or default_play_type).strip().lower()
    play_type = play_type or default_play_type
    option = row.get("best_option")
    option = option if isinstance(option, Mapping) else {}
    short_strike = _number(option.get("short_strike"))
    long_strike = _number(option.get("long_strike"))
    # The SOLD leg, whichever shape the play is. A recorder that reads `strike`
    # on both writes None for every credit spread.
    strike = short_strike if play_type == "pcs" else _number(option.get("strike"))
    if strike is None:
        strike = _number(option.get("strike")) if play_type == "pcs" else short_strike
    supports = _support_rows(row)
    return {
        "symbol": symbol,
        "scan_date": scan_date,
        # The session the supports and the close were read off. On the desk it
        # is normally the scan date; when the scan runs before a session
        # completes they differ, and the grade's horizons are counted from the
        # BAR, because it is that bar's close the strike is defended against.
        "bar_date": _iso_date(row.get("last_trade_date")) or scan_date,
        "play_type": play_type,
        "rank": int(rank),
        # `_apply_best_option_to_theta_row` replaces `score` with the option's
        # rank score and keeps the support score as `base_score`. Both, always.
        "score": _int(row.get("score")),
        "base_score": _int(row.get("base_score")),
        "option_status": str(row.get("option_status") or "").strip(),
        "supports": supports,
        "support_combo": "+".join(sorted(entry["label"] for entry in supports)),
        "supports_held": sum(1 for entry in supports if entry["held"]),
        "strike": strike,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "expiry": _expiry_iso(option.get("expiration")),
        "premium": _number(option.get("credit")),
        "atr": _number(row.get("atr20")),
        "close": _number(row.get("last_close")),
        "rs_flag": RS_FLAG_NOT_MEASURED,
        "recorded_at": now.isoformat(timespec="seconds")
        if isinstance(now, datetime)
        else str(now or ""),
        "first_seen_scan_date": scan_date,
    }


def record_theta_picks(
    theta_rows: Iterable[Any],
    pcs_rows: Iterable[Any],
    scan_date: Any,
    now: Any,
    *,
    path: Path | None = None,
) -> int:
    """Append one row per `(symbol, scan_date, play_type)`. Returns rows written.

    Called from the RUNNER right after `write_theta_put_report` (lead ruling
    (c)) - the scan's own output pass, not `legacy.py`'s tracker save.

    **A row already present for that key is not rewritten.** A scan that reruns,
    or a deferred option pass that calls this twice, must not double the n.

    **A repeat appearance is its own row and keeps `first_seen_scan_date`**, so
    the readout can count first appearances and repeat days separately without
    a second pass over the file.

    **It never raises into the scan.** Evidence stores are never allowed to cost
    the thing they record: a blocked path, an unwritable directory or a
    malformed row loses the row, never the scan. A malformed entry in the list
    is skipped individually, so one bad row does not cost the good ones.
    """
    store_path = Path(path) if path is not None else Path(THETA_PICKS_FILE)
    written = 0
    try:
        scan_text = _iso_date(scan_date)
        if not scan_text:
            _log.warning("Theta picks not recorded: unreadable scan date %r", scan_date)
            return 0
        existing = _read_jsonl(store_path)
        seen = {_pick_key(row) for row in existing}
        first_seen: dict[tuple[str, str], str] = {}
        for row in existing:
            symbol, day, play = _pick_key(row)
            if not symbol or not day:
                continue
            recorded = str(row.get("first_seen_scan_date") or day)
            key = (symbol, play)
            current = first_seen.get(key)
            if current is None or recorded < current:
                first_seen[key] = recorded

        fresh: list[dict[str, Any]] = []
        for rows, default_play_type in (
            (theta_rows, "sold_put"),
            (pcs_rows, "pcs"),
        ):
            for rank, row in enumerate(_ranked_report_rows(rows), start=1):
                try:
                    built = _build_pick_row(
                        row,
                        scan_date=scan_text,
                        default_play_type=default_play_type,
                        rank=rank,
                        now=now,
                    )
                except Exception:  # noqa: BLE001 - one bad row never costs the rest
                    _log.debug("Theta pick row unreadable; skipped", exc_info=True)
                    continue
                if built is None:
                    continue
                key = _pick_key(built)
                if key in seen:
                    continue
                seen.add(key)
                identity = (built["symbol"], built["play_type"])
                built["first_seen_scan_date"] = min(
                    first_seen.get(identity, scan_text), scan_text
                )
                first_seen[identity] = built["first_seen_scan_date"]
                fresh.append(built)

        if not fresh:
            return 0
        store_path.parent.mkdir(parents=True, exist_ok=True)
        with store_path.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        written = len(fresh)
    except Exception:  # noqa: BLE001 - the scan's next statement must still run
        _log.warning("Theta picks could not be recorded at %s", store_path, exc_info=True)
        return 0
    return written


# ---------------------------------------------------------------------------
# Item 2 - the grader
# ---------------------------------------------------------------------------


def _nth_session_after(day: date, horizon: int, calendar) -> date | None:
    """The `horizon`-th session strictly after `day`, holidays skipped.

    The same walk `session_horizon_outcomes._nth_session_after` does, kept here
    so the grader does not reach into a private of another module. A calendar
    that refuses to answer for a day gives None, and the mark is unmeasured
    with `target_session_outside_calendar_range` rather than guessed.
    """
    cursor = day
    remaining = int(horizon)
    for _ in range(remaining * 5 + 30):
        cursor = cursor + timedelta(days=1)
        try:
            answer = bool(calendar.is_session(cursor))
        except Exception:  # noqa: BLE001 - outside the validated range is "cannot say"
            return None
        if answer:
            remaining -= 1
            if remaining <= 0:
                return cursor
    return None


def _bar_prices(bar: Any) -> tuple[float | None, float | None]:
    """`(close, low)` off one day's bar. A mapping, never a bare float.

    The grader's `closes_for` hands back `{date: {"close", "low", "high"}}`
    rather than `{date: close}`: MAE and `first_support_broken` are questions a
    close-only series CANNOT answer - a session that traded through a support
    and closed back above it is a break the closes never see.
    """
    if isinstance(bar, Mapping):
        return _number(bar.get("close")), _number(bar.get("low"))
    return _number(bar), None


def _grade_one(
    pick: Mapping[str, Any],
    *,
    bars: Mapping[date, Any],
    calendar,
    as_of: date,
) -> dict[str, Any]:
    symbol = str(pick.get("symbol") or "").strip().upper()
    scan_date = str(pick.get("scan_date") or "").strip()
    entry_day = _parse_date(pick.get("bar_date")) or _parse_date(scan_date)
    strike = _number(pick.get("strike"))
    atr = _number(pick.get("atr"))
    close = _number(pick.get("close"))
    supports = [entry for entry in pick.get("supports") or [] if isinstance(entry, Mapping)]

    reasons: list[str] = []
    marks: dict[int, bool | None] = {mark: None for mark in SESSION_MARKS}

    def _note(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if strike is None:
        _note(REASON_NO_OPTION_QUOTE)
    if entry_day is None:
        _note(REASON_TARGET_OUT_OF_RANGE)

    endpoints: dict[int, date | None] = {}
    for mark in SESSION_MARKS:
        endpoint = _nth_session_after(entry_day, mark, calendar) if entry_day else None
        endpoints[mark] = endpoint
        if strike is None or entry_day is None:
            continue
        if endpoint is None:
            _note(REASON_TARGET_OUT_OF_RANGE)
            continue
        if endpoint > as_of:
            # A session that has not happened is not a break. COMPLETED BARS
            # ONLY: the bar is not read at all, even when the caller holds one.
            _note(REASON_IMMATURE)
            continue
        target_close, _low = _bar_prices(bars.get(endpoint))
        if target_close is None:
            _note(REASON_NO_TARGET_BAR)
            continue
        # A put at the strike is not assigned, so "held" is close >= strike.
        marks[mark] = bool(target_close >= strike)

    # ---- the path: MAE and the first support to break -------------------
    # The window is the sessions after the entry, out to the widest mark, and
    # never past `as_of`.
    window_end = endpoints.get(HEADLINE_MARK)
    window_end = min(window_end, as_of) if window_end else as_of
    walked: list[tuple[date, float | None, float | None]] = []
    if entry_day is not None:
        for day, bar in bars.items():
            day_value = day if isinstance(day, date) else _parse_date(day)
            if day_value is None or not (entry_day < day_value <= window_end):
                continue
            bar_close, bar_low = _bar_prices(bar)
            walked.append((day_value, bar_close, bar_low))
        walked.sort(key=lambda item: item[0])

    lows = [low for _day, _close, low in walked if low is not None]
    mae_atr: float | None = None
    if lows and close is not None and atr is not None and atr > 0:
        mae_atr = max(0.0, (close - min(lows)) / atr)

    # Only a support that was HOLDING on the scan date can break. A level the
    # scan kept 0.05 ATR overhead was never defending the strike, and counting
    # it would report a break on day one for every stacked pick.
    holding = [
        entry
        for entry in supports
        if bool(entry.get("held")) and _number(entry.get("level")) is not None
    ]
    first_support_broken = NO_SUPPORT_BROKEN if holding else ""
    for _day, _bar_close, low in walked:
        if low is None:
            continue
        broken = [
            entry for entry in holding if low < float(_number(entry.get("level")) or 0.0)
        ]
        if not broken:
            continue
        # Ties inside ONE session are resolved by the level price crossed first
        # on the way down - the highest one - and then by label, so the answer
        # is deterministic and is the support the trader watched fail.
        broken.sort(
            key=lambda entry: (
                -(float(_number(entry.get("level")) or 0.0)),
                str(entry.get("label") or ""),
            )
        )
        first_support_broken = str(broken[0].get("label") or "")
        break

    # ---- the expiry grade, only once the expiry session is complete ------
    expiry = _parse_date(pick.get("expiry"))
    held_at_expiry: bool | None = None
    if strike is None:
        pass
    elif expiry is None:
        _note(REASON_NO_EXPIRY)
    elif expiry > as_of:
        _note(REASON_EXPIRY_IMMATURE)
    else:
        expiry_close, _low = _bar_prices(bars.get(expiry))
        if expiry_close is None:
            _note(REASON_NO_EXPIRY_BAR)
        else:
            held_at_expiry = bool(expiry_close >= strike)

    if strike is None:
        status = STATUS_UNMEASURED
    elif all(marks[mark] is not None for mark in SESSION_MARKS):
        status = STATUS_MEASURED
    elif REASON_IMMATURE in reasons:
        status = STATUS_PENDING
    else:
        status = STATUS_UNMEASURED

    row: dict[str, Any] = {
        "symbol": symbol,
        "scan_date": scan_date,
        "first_seen_scan_date": str(pick.get("first_seen_scan_date") or scan_date),
        "bar_date": entry_day.isoformat() if entry_day else "",
        "play_type": str(pick.get("play_type") or "").strip().lower(),
        "rank": _int(pick.get("rank")),
        "support_combo": str(pick.get("support_combo") or ""),
        "supports_held": _int(pick.get("supports_held")),
        "score": _int(pick.get("score")),
        "base_score": _int(pick.get("base_score")),
        "strike": strike,
        "short_strike": _number(pick.get("short_strike")),
        "long_strike": _number(pick.get("long_strike")),
        "expiry": expiry.isoformat() if expiry else "",
        "premium": _number(pick.get("premium")),
        "atr": atr,
        "close": close,
        "held_at_expiry": held_at_expiry,
        "mae_atr": mae_atr,
        "first_support_broken": first_support_broken,
        "status": status,
        "unmeasured_reason": ";".join(reasons),
        "rs_flag": RS_FLAG_NOT_MEASURED,
        "rs_note": RS_NOTE,
    }
    for mark in SESSION_MARKS:
        row[f"held_{mark}"] = marks[mark]
    return row


def grade_theta_picks(
    store: Iterable[Mapping[str, Any]],
    *,
    closes_for: Callable[[str], Mapping[date, Any] | None],
    calendar=market_calendar,
    as_of: date | datetime,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Grade every recorded pick and write `master_avwap_theta_outcomes.csv`.

    `closes_for(symbol)` is supplied by the caller and returns
    ``{date: {"close": float, "low": float, "high": float}}`` for the completed
    daily bars already in hand. **It is a MAPPING per day, not a bare close**
    (lead ruling, 2026-09-12): the maximum adverse excursion and
    `first_support_broken` both ask what price TRADED, and a session that cut
    through a support and closed back above it is a break no close-only series
    can see. The parameter keeps the name `closes_for` that
    `session_horizon_outcomes` established; only its value shape is wider.

    **It never fetches.** A symbol with no bars in hand produces an unmeasured
    row with a reason, never a network call inside an export (ground rule 8).

    `as_of` is the last COMPLETED session. A mark, a MAE window or an expiry
    past it is not read at all - a session that has not happened is `pending`,
    never a break.

    Returns the graded rows and writes them to `path` (default
    `MASTER_AVWAP_THETA_OUTCOMES_FILE`) through a temp-and-rename, so a failed
    write never destroys the last good export.
    """
    as_of_day = as_of.date() if isinstance(as_of, datetime) else as_of
    cache: dict[str, Mapping[date, Any]] = {}
    rows: list[dict[str, Any]] = []
    for pick in store or []:
        if not isinstance(pick, Mapping):
            continue
        symbol = str(pick.get("symbol") or "").strip().upper()
        if symbol not in cache:
            try:
                cache[symbol] = closes_for(symbol) or {}
            except Exception:  # noqa: BLE001 - a caller's lookup never fails an export
                _log.debug("closes_for(%s) raised", symbol, exc_info=True)
                cache[symbol] = {}
        rows.append(
            _grade_one(pick, bars=cache[symbol], calendar=calendar, as_of=as_of_day)
        )
    _write_outcome_csv(
        Path(path) if path is not None else Path(MASTER_AVWAP_THETA_OUTCOMES_FILE), rows
    )
    return rows


def _write_outcome_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(THETA_OUTCOME_COLUMNS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if row.get(key) is None else row.get(key))
                    for key in THETA_OUTCOME_COLUMNS
                }
            )
    tmp.replace(path)


def closes_from_daily_bars(daily_bars_dir: Path | None = None):
    """A `closes_for` over the durable daily-bar store. READS, never fetches.

    The same parquet-per-symbol store the veto and like cohorts grade from
    (`human_focus_tracking._load_durable_daily_frame`). Memoised per symbol; a
    symbol with no file gives an empty mapping, which grades as unmeasured with
    a reason rather than as a break.
    """
    cache: dict[str, dict[date, dict[str, float]]] = {}

    def closes_for(symbol: str) -> dict[date, dict[str, float]]:
        key = str(symbol or "").strip().upper()
        if key in cache:
            return cache[key]
        bars: dict[date, dict[str, float]] = {}
        try:
            import pandas as pd

            from human_focus_tracking import (
                MASTER_AVWAP_DAILY_BARS_DIR,
                _load_durable_daily_frame,
            )

            frame = _load_durable_daily_frame(
                key, Path(daily_bars_dir or MASTER_AVWAP_DAILY_BARS_DIR)
            )
            if frame is not None and not getattr(frame, "empty", True):
                work = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
                stamps = work["datetime"] if "datetime" in work.columns else work[work.columns[0]]
                for index, stamp in enumerate(stamps):
                    day = pd.to_datetime(stamp, errors="coerce")
                    if day is None or pd.isna(day):
                        continue
                    entry: dict[str, float] = {}
                    for name in ("close", "low", "high"):
                        if name not in work.columns:
                            continue
                        value = _number(work[name].iloc[index])
                        if value is not None:
                            entry[name] = value
                    if "close" in entry:
                        bars[day.date()] = entry
        except Exception:  # noqa: BLE001 - a bad store is "no bars", never a failed export
            _log.debug("Daily bars unreadable for %s", key, exc_info=True)
            bars = {}
        cache[key] = bars
        return bars

    return closes_for


# ---------------------------------------------------------------------------
# Item 3 - the readout
# ---------------------------------------------------------------------------


def _readout_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    symbol = str(row.get("symbol") or "").strip().upper()
    scan_date = str(row.get("scan_date") or "").strip()
    if not symbol or not scan_date:
        return None
    first_seen = str(row.get("first_seen_scan_date") or "").strip() or scan_date
    return {
        "symbol": symbol,
        "scan_date": scan_date,
        "first_seen_scan_date": first_seen,
        # THE COHORT GRAIN. A repeat appearance is an observation of interest,
        # never a second observation of the outcome.
        "is_first_appearance": first_seen == scan_date,
        "play_type": str(row.get("play_type") or "").strip().lower(),
        "support_combo": str(row.get("support_combo") or ""),
        "score": _number(row.get("score")),
        "held_20": _tristate(row.get(f"held_{HEADLINE_MARK}")),
        "held_at_expiry": _tristate(row.get("held_at_expiry")),
        "status": str(row.get("status") or "").strip(),
    }


@dataclass
class ThetaReadout:
    """The Theta tab's cells and its two sentences. Pure; nothing here writes."""

    cells: list[dict[str, Any]] = field(default_factory=list)
    n_first_appearances: int = 0
    n_rows: int = 0
    n_scan_dates: int = 0
    n_expiry_grades: int = 0
    _graded: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def population_sentence(self) -> str:
        """What the table IS, counted at its own grain, above the numbers."""
        if not self.n_rows:
            return THETA_NO_EXPORT_SENTENCE
        return (
            f"{self.n_first_appearances} first appearance"
            f"{'' if self.n_first_appearances == 1 else 's'} of {self.n_rows} theta "
            f"pick{'' if self.n_rows == 1 else 's'} over {self.n_scan_dates} scan "
            f"date{'' if self.n_scan_dates == 1 else 's'}; graded at {HEADLINE_MARK} "
            f"sessions; expiry grades: {self.n_expiry_grades}"
        )

    def grade_sentence(self) -> str:
        """Does a higher theta score hold better? Terciles, in the point system's words.

        The wording and the cells are `setup_points_evidence.Cell` - the same
        `X% of N (>= L%)` line and the same `lift +N pts` the Points grade uses -
        so two grade lines on this desk never say the same thing two ways. The
        refusal under `MIN_REPORTABLE_N` per third is that module's too: a
        tercile lift on nine picks is a coin flip with a decimal point.
        """
        if not self.n_rows:
            return ""
        from setup_points_evidence import Cell

        ordered = sorted(self._graded, key=lambda row: -(row["score"] or 0.0))
        third = len(ordered) // 3
        top_rows = ordered[:third] if third else []
        bottom_rows = ordered[len(ordered) - third :] if third else []
        top = Cell("top third", len(top_rows), sum(1 for r in top_rows if r["held_20"]))
        bottom = Cell(
            "bottom third", len(bottom_rows), sum(1 for r in bottom_rows if r["held_20"])
        )
        head = (
            f"Theta score grade over {len(ordered)} graded pick"
            f"{'' if len(ordered) == 1 else 's'} at {HEADLINE_MARK} sessions: "
        )
        if not third or min(top.n, bottom.n) < MIN_REPORTABLE_N:
            return head + f"not enough per third yet ({top.line()}; {bottom.line()})."
        lift = (top.win_rate or 0.0) - (bottom.win_rate or 0.0)
        verdict = (
            "a higher theta score DID hold better"
            if lift > 0
            else "a higher theta score did NOT hold better"
        )
        return head + f"{verdict} - {top.line()} vs {bottom.line()}, lift {lift * 100:+.0f} pts."


def theta_readout(rows: Iterable[Mapping[str, Any]]) -> ThetaReadout:
    """Support-combo x play-type cells, hold rate first, sorted by the bound.

    **Win rate leads and the sort is the BOUND**, exactly as the Controls and
    Studies tabs do it: a 100% on two picks is not better than a 60% on forty,
    and the Wilson lower bound is the only ordering that says so. `n` counts
    FIRST APPEARANCES with a measured 20-session mark; `repeat_days` counts the
    later appearances beside it and is never summed into `n`.

    Reads a native row and an exported CSV row identically - `True` arrives as
    the string `"True"` and an unmeasured cell as `""`.
    """
    prepared = [_readout_row(row) for row in rows or []]
    prepared = [row for row in prepared if row is not None]

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in prepared:
        key = (row["support_combo"], row["play_type"])
        cell = buckets.setdefault(
            key,
            {
                "support_combo": key[0],
                "play_type": key[1],
                "n": 0,
                "wins": 0,
                "repeat_days": 0,
                "n_unmeasured": 0,
                "n_expiry_graded": 0,
                "n_expiry_held": 0,
            },
        )
        if not row["is_first_appearance"]:
            cell["repeat_days"] += 1
            continue
        if row["held_20"] is None:
            cell["n_unmeasured"] += 1
        else:
            cell["n"] += 1
            cell["wins"] += 1 if row["held_20"] else 0
        if row["held_at_expiry"] is not None:
            cell["n_expiry_graded"] += 1
            cell["n_expiry_held"] += 1 if row["held_at_expiry"] else 0

    cells: list[dict[str, Any]] = []
    for cell in buckets.values():
        n = int(cell["n"])
        wins = int(cell["wins"])
        expiry_n = int(cell["n_expiry_graded"])
        cells.append(
            {
                **cell,
                "hold_rate": (wins / n) if n else None,
                "hold_rate_lb": wilson_lower_bound(wins, n) if n else None,
                "expiry_hold_rate": (cell["n_expiry_held"] / expiry_n) if expiry_n else None,
                "meets_floor": n >= MIN_REPORTABLE_N,
                "horizon_sessions": HEADLINE_MARK,
            }
        )
    cells.sort(
        key=lambda cell: (
            cell["hold_rate_lb"] is None,
            -(cell["hold_rate_lb"] or 0.0),
            -int(cell["n"]),
            str(cell["support_combo"]),
            str(cell["play_type"]),
        )
    )

    graded = [
        row
        for row in prepared
        if row["is_first_appearance"]
        and row["held_20"] is not None
        and row["score"] is not None
    ]
    return ThetaReadout(
        cells=cells,
        n_first_appearances=sum(1 for row in prepared if row["is_first_appearance"]),
        n_rows=len(prepared),
        n_scan_dates=len({row["scan_date"] for row in prepared}),
        n_expiry_grades=sum(1 for row in prepared if row["held_at_expiry"] is not None),
        _graded=graded,
    )
