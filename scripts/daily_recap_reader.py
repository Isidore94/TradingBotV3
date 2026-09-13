"""The Daily Recap's session reader - WISHLIST 10F + 5F, packet WS-DR.

**The recap reads the day from the STORES, not the process.** The AWAY Recap
page is handed `center._alerts` + `center._d1_alerts` (`ui/app.py`
`_feed_away_recap`): a process-scoped list capped at 250 / 100 items. A desk
restarted mid-session, or left running across midnight, therefore reported what
the PROCESS saw rather than what the SESSION produced. Everything here takes its
input as PATHS (`RecapSources`) and a clock, so the same session read twice - in
two different interpreters, a week apart - is the same answer.

Descriptive learning, not a ranking model. Nothing in this module reaches a
detector, a score, a gate, an alert, a watchlist, Focus, the review queue or
`review_policy.json`; nothing here loads the 1.1 GB setup tracker.

Four rules the numbers obey, each of which a naive reader gets wrong:

1. **The measured columns are already side-adjusted.** `intraday_bounce_outcomes`
   stores `mfe_pct` / `eod_move_pct` from the trade's own side, and its side
   column is named `direction`. A short that closed 5% below its entry is a
   `+5.00`; recomputing `(close - entry) / entry` would file it beside the long
   that lost 5%.
2. **A decision's credit starts at its own timestamp.** A like clicked at 12:30
   is never credited with the 10:00 high. Where the bars behind the decision are
   reachable (a day-trade pass writes an M5 sidecar) the post-decision excursion
   is MEASURED; where they are not, the cell is `unavailable` - never the day's
   number, and never zero.
3. **Present-and-empty is not zero.** An unfinalized outcome row, an immature
   horizon and a session nobody labelled are each `None` with a named reason.
4. **Every verdict is its own fact.** A veto, a dislike, a pass, a not-today, a
   quick like and a claimed like are separate rows even on one symbol; repeated
   clicks on ONE opportunity link into one row with an occurrence count;
   `unfavorite` is never graded at all (CLAUDE.md, P5).

Timestamps: naive stamps get the DESK ZONE ATTACHED (`pass_bars.desk_zone`);
an aware stamp keeps the offset it was written with. Nothing is ever stripped.
Every emitted moment is then normalised to its own fixed offset so a session
read is byte-identical between two processes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import project_paths

#: What a cell reads when nobody measured it. `unknown` is its own answer.
UNKNOWN = "unknown"

#: The four views, in the packet's order.
VIEW_NAMES: tuple[str, ...] = (
    "worked_today",
    "recent_swings",
    "my_decisions",
    "rejected_that_worked",
)

#: The verdicts that are a REFUSAL. They are the only ones view 4 considers, and
#: they are never pooled with an endorsement. `unfavorite` is in neither list:
#: taking a name off a list is not a verdict on the setup.
REJECT_VERDICTS: tuple[str, ...] = (
    "veto",
    "pass",
    "not_today",
    "dislike",
    "m5_click_away",
)

#: The endorsements.
ENDORSE_VERDICTS: tuple[str, ...] = ("like", "swing_favorite")

#: Verdict -> the `preference_trade_outcomes` channel that carries it. The join
#: is on the CHANNEL as well as the name, so a symbol vetoed AND thrown back for
#: the day does not inherit the other statement's journal match.
_CHANNEL_BY_SOURCE_VERDICT: dict[tuple[str, str], str] = {
    ("annotations", "like"): "annotation:like_claim",
    ("annotations", "veto"): "annotation:veto",
    ("annotations", "pass"): "annotation:pass",
    ("swing_favorites", "swing_favorite"): "swing_favorite",
    ("pick_feedback", "like"): "pick_feedback:like",
    ("pick_feedback", "dislike"): "pick_feedback:dislike",
    ("pick_feedback", "not_today"): "pick_feedback:not_today",
    ("review_events", "m5_click_away"): "review_event:m5_click_away",
}

#: The measures each view declares. A sort control may offer THESE and nothing
#: else: a recap orders by what it measured, never by a score it made up (the
#: refusal gate #43 puts on the narration view).
SORT_KEYS: dict[str, tuple[str, ...]] = {
    "worked_today": ("mfe_pct", "eod_move_pct"),
    "recent_swings": ("selected_end_pct", "next_close_pct", "first_favorable_pct"),
    "my_decisions": ("day_mfe_pct", "mfe_pct_after_decision", "journal_r"),
    "rejected_that_worked": (
        "favorable_pct",
        "favorable_pct_after_decision",
        "adverse_pct",
    ),
}

#: What each view is a population OF, said in one clause so every table can
#: print its cohort without inventing a sentence of its own.
COHORTS: dict[str, str] = {
    "worked_today": "every intraday M5 outcome row logged for the session",
    "recent_swings": "every swing observation scanned inside the lookback window",
    "my_decisions": "every verdict the trader recorded in the session",
    "rejected_that_worked": (
        "the refusals whose later path went the way the trader turned down"
    ),
}


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def _working_lately_default() -> Path:
    """`snapshot_latest.json` without importing the Qt service to find it."""
    return project_paths.LOCAL_SETTINGS_DIR / "working_lately" / "snapshot_latest.json"


def _environment_default() -> Path:
    import d1_environment_store

    return d1_environment_store.default_path()


@dataclass(frozen=True)
class RecapSources:
    """Every durable store a session read touches, as PATHS.

    Paths rather than a feed on purpose: this is the whole difference from
    `_feed_away_recap`. A reader whose input is a set of files can be re-run
    after a restart, on an old date, or in another process, and give the same
    answer - which is the property WISHLIST 10F asks for.
    """

    intraday_outcomes: Path = project_paths.INTRADAY_BOUNCE_OUTCOMES_FILE
    tier_outcomes: Path = project_paths.MASTER_AVWAP_TIER_OUTCOMES_FILE
    session_horizon_outcomes: Path = (
        project_paths.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE
    )
    annotations: Path = project_paths.TRADER_ANNOTATIONS_FILE
    pick_feedback: Path = project_paths.PICK_FEEDBACK_FILE
    swing_favorites: Path = project_paths.SWING_FAVORITES_FILE
    human_focus_outcomes: Path = project_paths.HUMAN_FOCUS_OUTCOMES_FILE
    review_events: Path = project_paths.ALERT_REVIEW_EVENTS_FILE
    preference_report: Path = field(
        default_factory=lambda: _preference_report_default()
    )
    staged_picks: Path = project_paths.AUTO_POPULATE_PENDING_FILE
    environment_labels: Path = field(default_factory=_environment_default)
    working_lately: Path = field(default_factory=_working_lately_default)


def _preference_report_default() -> Path:
    import preference_trade_outcomes

    return preference_trade_outcomes.REPORT_FILE


#: The order coverage is reported in. Declared once so the reported set cannot
#: drift from the fields above.
SOURCE_NAMES: tuple[str, ...] = (
    "intraday_outcomes",
    "tier_outcomes",
    "session_horizon_outcomes",
    "annotations",
    "pick_feedback",
    "swing_favorites",
    "human_focus_outcomes",
    "review_events",
    "preference_report",
    "staged_picks",
    "environment_labels",
    "working_lately",
)


# ---------------------------------------------------------------------------
# rows, views, the session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceCoverage:
    """What one store gave this read, and what it could not give.

    `rows` counts EVERY row on disk, not the ones that survived the session
    filter: the question the coverage line answers is "was the file there and
    how much of it did I see", and a session with no rows in a full file is a
    different fact from a file that would not open.
    """

    name: str
    path: str
    rows: int
    oldest: datetime | None
    newest: datetime | None
    unavailable_reason: str = ""


@dataclass(frozen=True)
class RecapRow:
    """One observation or one decision, with its identity and its numbers.

    `measures` carries exactly the view's declared measures, `None` where a
    thing was not measured; `unavailable` names the reason for every `None`.
    The two never collapse into one field, because "0.00%" and "nobody looked"
    are different answers.
    """

    symbol: str
    side: str
    source: str
    category: str
    capture_id: str
    observed_at: datetime | None
    measures: dict[str, float | None]
    unavailable: dict[str, str]
    detail: dict[str, Any]
    pick_key: tuple[str, str, str, str]
    d1_environment: str = UNKNOWN
    occurrences: int = 1
    pending: bool = False


@dataclass(frozen=True)
class RecapView:
    """One table: its population, its window, its rows and its sort control."""

    name: str
    cohort: str
    window: tuple[str, str]
    rows: tuple[RecapRow, ...]
    pending: tuple[RecapRow, ...]
    sort_key: str
    sort_keys: tuple[str, ...]
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.rows)

    def sorted_by(self, key: str) -> tuple[RecapRow, ...]:
        """Rows by one DECLARED measure, best first, unmeasured last.

        A key the view never measured raises rather than sorting by nothing:
        a sort control that silently ignores its argument is a control that
        lies about what the table is ordered by.
        """
        if key not in self.sort_keys:
            raise ValueError(
                f"{self.name} measures {list(self.sort_keys)}; it cannot sort by {key!r}"
            )
        return tuple(sorted(self.rows, key=lambda row: _sort_key(row, key)))

    def population_sentence(self) -> str:
        """Cohort, window, n, pending - said out loud above the table."""
        first, last = self.window
        window_text = first if first == last else f"{first} to {last}"
        return (
            f"{self.cohort}. Window {first} → {last} ({window_text}); "
            f"n = {self.n}; {len(self.pending)} pending"
            + (f". {self.note}" if self.note else ".")
        )


@dataclass(frozen=True)
class RecapSession:
    """One session, read from the files. Nothing here is process-scoped."""

    session_date: str
    lookback_sessions: int
    provisional: bool
    now: datetime
    coverage: dict[str, SourceCoverage]
    worked_today: RecapView
    recent_swings: RecapView
    my_decisions: RecapView
    rejected_that_worked: RecapView
    staged_picks: dict[str, tuple[str, ...]]
    working_lately: dict[str, Any]

    def view(self, name: str) -> RecapView:
        if name not in VIEW_NAMES:
            raise ValueError(f"no such view: {name!r}")
        return getattr(self, name)


def _sort_key(row: RecapRow, key: str) -> tuple:
    """Descending by the measure, unmeasured last, ties by symbol.

    A tuple rather than a `reverse=True` sort, because `None` has to land at
    the END either way round: "not measured" is not "worst".
    """
    value = row.measures.get(key)
    if value is None:
        return (1, 0.0, row.symbol, row.capture_id)
    return (0, -float(value), row.symbol, row.capture_id)


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------


def _desk_zone():
    from ui.annotations.pass_bars import desk_zone

    return desk_zone()


def _aware(moment: datetime | None) -> datetime | None:
    """Attach the desk zone to a naive moment; never strip an aware one (N1).

    The result is then pinned to its own FIXED offset. Two things ride on that:
    the offset the desk resolved is the one the reader reports, and the repr of
    a session read is identical in two processes - which is how the durability
    property is provable at all.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_desk_zone())
    offset = moment.utcoffset()
    if offset is None:
        return moment
    return moment.astimezone(timezone(offset))


def _parse_moment(value: Any) -> datetime | None:
    """An ISO stamp (aware or naive), a date, or `None`."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        pass
    try:
        day = date.fromisoformat(text[:10])
    except ValueError:
        return None
    return _session_moment(day)


def _session_moment(day: date) -> datetime | None:
    """A date-only stamp reads as that session's close - aware, in market time."""
    try:
        import market_calendar

        return _aware(market_calendar.session_close(day))
    except Exception:  # noqa: BLE001 - a coverage bound never costs the read
        return None


def _float_or_none(value: Any) -> float | None:
    """A present-and-EMPTY column is `None`, never 0.0."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "L"}:
        return "LONG"
    if text in {"SHORT", "SELL", "S"}:
        return "SHORT"
    return text


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _session_text(value: Any) -> str:
    return str(value or "").strip()[:10]


# ---------------------------------------------------------------------------
# reading one store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Store:
    """The rows of one source plus the coverage entry that describes them."""

    rows: tuple[dict[str, Any], ...]
    coverage: SourceCoverage


def _coverage(
    name: str,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    moments: Sequence[datetime | None],
    reason: str = "",
) -> SourceCoverage:
    stamps = sorted(m for m in moments if m is not None)
    return SourceCoverage(
        name=name,
        path=str(path),
        rows=len(rows),
        oldest=stamps[0] if stamps else None,
        newest=stamps[-1] if stamps else None,
        unavailable_reason=reason,
    )


def _unreadable(name: str, path: Path, exc: Exception) -> SourceCoverage:
    """A missing source is NAMED, and it costs no other view.

    The reason names the FILE, because "a source was unavailable" tells a
    trader nothing they can act on and "intraday_bounce_outcomes.csv is not
    there" tells them exactly what to look at.
    """
    return SourceCoverage(
        name=name,
        path=str(path),
        rows=0,
        oldest=None,
        newest=None,
        unavailable_reason=f"{Path(path).name} could not be read: {exc}",
    )


def _read_csv(name: str, path: Path, clock_field: str) -> _Store:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return _Store((), _unreadable(name, target, exc))
    rows = tuple(csv.DictReader(text.splitlines()))
    moments = [_parse_moment(row.get(clock_field)) for row in rows]
    return _Store(rows, _coverage(name, target, rows, moments))


def _read_jsonl(name: str, path: Path, clock_field: str) -> _Store:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return _Store((), _unreadable(name, target, exc))
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # One torn line never makes the rest of the record unreadable.
            continue
        if isinstance(row, dict):
            rows.append(row)
    moments = [_parse_moment(row.get(clock_field)) for row in rows]
    return _Store(tuple(rows), _coverage(name, target, rows, moments))


# ---------------------------------------------------------------------------
# the session calendar
# ---------------------------------------------------------------------------


def default_session(now: datetime | None = None) -> str:
    """The last COMPLETED exchange session - the recap's default day."""
    import market_calendar

    return market_calendar.last_completed_session(now or datetime.now()).isoformat()


def _lookback_window(session_date: str, lookback_sessions: int) -> tuple[str, str]:
    """The `lookback_sessions` sessions BEFORE the selected one.

    Counted on the exchange calendar, never in calendar days: Labor Day 2026-09-07
    is not a session, so the third prior session of 2026-09-10 is 09-04 and a
    day-counted window would have said 09-07 and swept in another scan date.
    """
    import market_calendar

    try:
        cursor = date.fromisoformat(session_date)
    except ValueError:
        return (session_date, session_date)
    sessions: list[str] = []
    for _ in range(max(1, int(lookback_sessions))):
        try:
            cursor = market_calendar.previous_session(cursor)
        except Exception:  # noqa: BLE001 - a calendar refusal narrows, never widens
            break
        sessions.append(cursor.isoformat())
    if not sessions:
        return (session_date, session_date)
    return (sessions[-1], sessions[0])


def _is_provisional(session_date: str, now: datetime) -> bool:
    """True while the selected session is still open (or still in the future)."""
    try:
        selected = date.fromisoformat(session_date)
    except ValueError:
        return False
    try:
        import market_calendar

        return selected > market_calendar.last_completed_session(now)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# the intraday outcome index - the day's measured path, per name
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Outcome:
    """One M5 outcome row, reduced to what the recap reports."""

    event_id: str
    trade_date: str
    symbol: str
    side: str
    entry_time: datetime | None
    status: str
    mfe_pct: float | None
    mae_pct: float | None
    eod_move_pct: float | None


def _outcomes(store: _Store) -> tuple[_Outcome, ...]:
    out: list[_Outcome] = []
    for row in store.rows:
        out.append(
            _Outcome(
                event_id=str(row.get("event_id") or ""),
                trade_date=_session_text(row.get("trade_date")),
                symbol=_symbol(row.get("symbol")),
                # The live column is `direction`, not `side`.
                side=_side(row.get("direction")),
                entry_time=_parse_moment(row.get("entry_time")),
                status=str(row.get("status") or "").strip(),
                mfe_pct=_float_or_none(row.get("mfe_pct")),
                mae_pct=_float_or_none(row.get("mae_pct")),
                eod_move_pct=_float_or_none(row.get("eod_move_pct")),
            )
        )
    return tuple(out)


def _outcome_for(
    outcomes: Sequence[_Outcome], session_date: str, symbol: str, side: str
) -> _Outcome | None:
    """The session's outcome row for one name and side, or `None`.

    Side is part of the key: the same symbol can carry a long and a short row
    on one day, and they are opposite readings of the same tape.
    """
    for row in outcomes:
        if row.trade_date != session_date or row.symbol != symbol:
            continue
        if side and row.side and row.side != side:
            continue
        return row
    return None


def _unmeasured_reason(outcome: _Outcome | None, column: str) -> str:
    if outcome is None:
        return (
            f"no intraday outcome row measured {column} for this name on this session"
        )
    return (
        f"{outcome.event_id or 'the outcome row'} is {outcome.status or 'unfinalised'}"
        f" - its {column} column is present and empty"
    )


# ---------------------------------------------------------------------------
# view 1 - what worked today
# ---------------------------------------------------------------------------


def _worked_today_view(
    session_date: str,
    outcomes: Sequence[_Outcome],
    labels: Mapping[str, str],
) -> RecapView:
    rows: list[RecapRow] = []
    for outcome in outcomes:
        if outcome.trade_date != session_date:
            continue
        measures: dict[str, float | None] = {
            "mfe_pct": outcome.mfe_pct,
            "eod_move_pct": outcome.eod_move_pct,
        }
        unavailable = {
            key: _unmeasured_reason(outcome, key)
            for key, value in measures.items()
            if value is None
        }
        rows.append(
            RecapRow(
                symbol=outcome.symbol,
                side=outcome.side,
                source="intraday_outcomes",
                category="m5",
                capture_id=outcome.event_id,
                observed_at=outcome.entry_time,
                measures=measures,
                unavailable=unavailable,
                detail={
                    "status": outcome.status,
                    "verdict": "",
                    "reason": "",
                },
                pick_key=(session_date, outcome.symbol, outcome.side, "m5"),
                d1_environment=labels.get(outcome.trade_date, UNKNOWN),
            )
        )
    view = RecapView(
        name="worked_today",
        cohort=COHORTS["worked_today"],
        window=(session_date, session_date),
        rows=tuple(rows),
        pending=(),
        sort_key=SORT_KEYS["worked_today"][0],
        sort_keys=SORT_KEYS["worked_today"],
        note=(
            "Best available movement, already side-adjusted by the store - "
            "never money earned."
        ),
    )
    return _with_rows(view, view.sorted_by(view.sort_key))


def _with_rows(view: RecapView, rows: tuple[RecapRow, ...]) -> RecapView:
    return RecapView(
        name=view.name,
        cohort=view.cohort,
        window=view.window,
        rows=rows,
        pending=view.pending,
        sort_key=view.sort_key,
        sort_keys=view.sort_keys,
        note=view.note,
    )


# ---------------------------------------------------------------------------
# view 2 - recent swing picks that followed through
# ---------------------------------------------------------------------------


def _recent_swings_view(
    window: tuple[str, str],
    lookback_sessions: int,
    horizon_store: _Store,
    outcomes: Sequence[_Outcome],
    labels: Mapping[str, str],
) -> RecapView:
    """One row per swing observation inside the window.

    Three separate columns, never blended: the next session's CLOSE (the
    instant follow-through read), the first favorable move on that next session
    (from the name's own M5 outcome row) and the SELECTED 1-3-session end. The
    lookback control selects both the window and the horizon reported, because
    "how far back" and "how far forward" are one question for a swing pick.
    """
    first, last = window
    horizon = max(1, int(lookback_sessions))
    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for row in horizon_store.rows:
        scan_date = _session_text(row.get("scan_date"))
        if not (first <= scan_date <= last):
            continue
        try:
            horizon_sessions = int(str(row.get("horizon_sessions") or "").strip())
        except ValueError:
            continue
        key = (scan_date, _symbol(row.get("symbol")), _side(row.get("side")))
        grouped.setdefault(key, {})[horizon_sessions] = dict(row)

    rows: list[RecapRow] = []
    pending: list[RecapRow] = []
    for key in sorted(grouped):
        scan_date, symbol, side = key
        by_horizon = grouped[key]
        next_row = by_horizon.get(1)
        selected_row = by_horizon.get(horizon)

        next_close = _measured_return(next_row)
        selected_end = _measured_return(selected_row)
        target_session = _session_text((next_row or {}).get("target_session"))
        outcome = (
            _outcome_for(outcomes, target_session, symbol, side)
            if target_session
            else None
        )
        first_favorable = outcome.mfe_pct if outcome is not None else None

        measures: dict[str, float | None] = {
            "selected_end_pct": selected_end,
            "next_close_pct": next_close,
            "first_favorable_pct": first_favorable,
        }
        unavailable: dict[str, str] = {}
        if selected_end is None:
            unavailable["selected_end_pct"] = (
                f"the {horizon}-session end has not been measured for the "
                f"{scan_date} observation"
                if selected_row is not None
                else f"no {horizon}-session horizon row exists for {scan_date}"
            )
        if next_close is None:
            unavailable["next_close_pct"] = (
                f"no measured next-session close for the {scan_date} observation"
            )
        if first_favorable is None:
            unavailable["first_favorable_pct"] = (
                "no intraday outcome row measured this name's excursion on "
                f"{target_session or 'the next session'}"
            )

        is_pending = selected_end is None
        row = RecapRow(
            symbol=symbol,
            side=side,
            source="session_horizon_outcomes",
            category="swing",
            capture_id=str((selected_row or next_row or {}).get("scan_row_id") or ""),
            observed_at=_session_moment_or_none(scan_date),
            measures=measures,
            unavailable=unavailable,
            detail={
                "scan_date": scan_date,
                "target_session": target_session,
                "horizon_sessions": horizon,
                "setup_family": str(
                    (selected_row or next_row or {}).get("setup_family") or ""
                ),
                "maturity": str((selected_row or {}).get("maturity") or ""),
                "verdict": "",
                "reason": "",
            },
            pick_key=(scan_date, symbol, side, "swing"),
            d1_environment=labels.get(scan_date, UNKNOWN),
            pending=is_pending,
        )
        (pending if is_pending else rows).append(row)

    view = RecapView(
        name="recent_swings",
        cohort=COHORTS["recent_swings"],
        window=window,
        rows=tuple(rows),
        pending=tuple(pending),
        sort_key=SORT_KEYS["recent_swings"][0],
        sort_keys=SORT_KEYS["recent_swings"],
        note=(
            f"Horizon {horizon} session(s); an end that has not arrived is "
            "pending, never a zero."
        ),
    )
    return _with_rows(view, view.sorted_by(view.sort_key))


def _session_moment_or_none(session_date: str) -> datetime | None:
    try:
        return _session_moment(date.fromisoformat(session_date))
    except ValueError:
        return None


def _measured_return(row: Mapping[str, Any] | None) -> float | None:
    """`side_return_pct`, but only when the row SAYS it was measured.

    An immature horizon writes `measured` and `side_return_pct` present and
    EMPTY. Reading the blank as 0.0 would file a pick that has not finished as
    a pick that went nowhere.
    """
    if row is None:
        return None
    measured = str(row.get("measured") or "").strip().lower()
    if measured not in {"true", "1", "yes"}:
        return None
    return _float_or_none(row.get("side_return_pct"))


# ---------------------------------------------------------------------------
# view 3 - my decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Decision:
    """One statement the trader made, before any measurement is attached."""

    symbol: str
    side: str
    verdict: str
    source: str
    category: str
    capture_id: str
    observed_at: datetime | None
    reason: str
    detail: dict[str, Any]


def _like_mode(row: Mapping[str, Any]) -> str:
    """P9: absence reads `claimed`, because a claim was REQUIRED until P9."""
    mode = str(row.get("like_mode") or "").strip().lower()
    return mode if mode in {"quick", "claimed"} else "claimed"


def _decisions(
    session_date: str,
    annotations: _Store,
    pick_feedback: _Store,
    swing_favorites: _Store,
    review_events: _Store,
) -> tuple[_Decision, ...]:
    """Every verdict recorded in the session, from four stores, unpooled.

    A note is neither an endorsement nor a refusal and is not a verdict on the
    setup, so it is not here; `unfavorite` is never graded at all.
    """
    out: list[_Decision] = []

    for row in annotations.rows:
        if _session_text(row.get("session_date")) != session_date:
            continue
        kind = str(row.get("event_type") or "").strip()
        moment = _parse_moment(row.get("created_at"))
        symbol = _symbol(row.get("symbol"))
        side = _side(row.get("side"))
        if kind == "like_claim":
            out.append(
                _Decision(
                    symbol=symbol,
                    side=side,
                    verdict="like",
                    source="annotations",
                    category="chart_review",
                    capture_id=str(row.get("event_id") or ""),
                    observed_at=moment,
                    reason=str(row.get("note") or ""),
                    detail={"like_mode": _like_mode(row)},
                )
            )
        elif kind == "veto":
            out.append(
                _Decision(
                    symbol=symbol,
                    side=side,
                    verdict="veto",
                    source="annotations",
                    category="chart_review",
                    capture_id=str(row.get("event_id") or ""),
                    observed_at=moment,
                    reason=str(row.get("reason_code") or ""),
                    detail={"vocab_version": row.get("vocab_version")},
                )
            )
        elif kind == "pass":
            codes = tuple(
                str(code or "").strip()
                for code in (row.get("reason_codes") or ())
                if str(code or "").strip()
            )
            out.append(
                _Decision(
                    symbol=symbol,
                    side=side,
                    verdict="pass",
                    source="annotations",
                    category="chart_review",
                    capture_id=str(row.get("event_id") or ""),
                    observed_at=moment,
                    reason=", ".join(codes),
                    # ONE pass, two codes. The cohorts overlap and are never
                    # summed: this is one decision, not two.
                    detail={
                        "reason_codes": codes,
                        "m5_bars_ref": str(row.get("m5_bars_ref") or ""),
                    },
                )
            )

    for row in pick_feedback.rows:
        if _session_text(row.get("trade_date")) != session_date:
            continue
        verdict = str(row.get("verdict") or "").strip()
        if verdict not in {"like", "dislike", "not_today"}:
            # `unfavorite` is never graded (CLAUDE.md, P5).
            continue
        out.append(
            _Decision(
                symbol=_symbol(row.get("symbol")),
                side=_side(row.get("side")),
                verdict=verdict,
                source="pick_feedback",
                category=str(row.get("category") or "").strip() or "pick",
                capture_id="",
                # Every pick-feedback `ts` on the desk is NAIVE. The zone is
                # attached; the row is never dropped for want of an offset.
                observed_at=_parse_moment(row.get("ts")),
                reason=str(row.get("reason") or ""),
                detail={
                    "origin": str(row.get("origin") or ""),
                    # A ★ on a board carries no P9 mode.
                    "like_mode": "" if verdict != "like" else "claimed",
                },
            )
        )

    live: dict[tuple[str, str], dict[str, Any]] = {}
    for row in swing_favorites.rows:
        if _session_text(row.get("session_date")) != session_date:
            continue
        key = (_symbol(row.get("symbol")), _side(row.get("side")))
        action = str(row.get("action") or "").strip().lower()
        if action == "remove":
            # A retraction removes the favorite. It is not a favorite.
            live.pop(key, None)
        elif action == "add":
            live.pop(key, None)
            live[key] = dict(row)
    for (symbol, side), row in live.items():
        out.append(
            _Decision(
                symbol=symbol,
                side=side,
                verdict="swing_favorite",
                source="swing_favorites",
                category="swing",
                capture_id="",
                observed_at=_parse_moment(row.get("event_at")),
                reason=str(row.get("origin") or ""),
                detail={"origin": str(row.get("origin") or "")},
            )
        )

    for row in review_events.rows:
        if _session_text(row.get("trade_date")) != session_date:
            continue
        detail = row.get("detail")
        reason = str((detail or {}).get("reason") or "") if isinstance(detail, dict) else ""
        if reason != "clicked_away_from_m5_alert":
            # An impression is not a decision.
            continue
        out.append(
            _Decision(
                symbol=_symbol(row.get("symbol")),
                side=_side(row.get("side")),
                verdict="m5_click_away",
                source="review_events",
                category="m5",
                capture_id=str(row.get("event_id") or ""),
                observed_at=_parse_moment(row.get("ts")),
                # A click away IS a pass (trader, 2026-09-01).
                reason="clicked away from the M5 alert",
                detail={"trigger": str(row.get("trigger") or "")},
            )
        )
    return tuple(out)


def _decision_rows(
    session_date: str,
    decisions: Sequence[_Decision],
    outcomes: Sequence[_Outcome],
    annotations: _Store,
    preference: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    labels: Mapping[str, str],
    annotations_path: Path,
) -> tuple[RecapRow, ...]:
    """Link repeated clicks on ONE opportunity, then attach the measurements.

    The grain is `(trade_date, symbol, side, category slot)` plus the verdict -
    `human_focus_tracking._pick_key` with the statement kept separate, because a
    symbol vetoed AND thrown back for the day is two statements about one name
    and pooling them would count one decision twice. Two clicks on the same
    statement are one row with `occurrences = 2`, and the credit starts at the
    FIRST one.
    """
    grouped: dict[tuple[str, str, str, str, str], list[_Decision]] = {}
    order: list[tuple[str, str, str, str, str]] = []
    for decision in decisions:
        key = (
            session_date,
            decision.symbol,
            decision.side,
            decision.category,
            decision.verdict,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(decision)

    sidecar_bars = _pass_bar_index(annotations, session_date, annotations_path)

    rows: list[RecapRow] = []
    for key in order:
        members = sorted(
            grouped[key],
            key=lambda item: (
                item.observed_at is None,
                item.observed_at or datetime.min.replace(tzinfo=timezone.utc),
                item.capture_id,
            ),
        )
        first = members[0]
        outcome = _outcome_for(outcomes, session_date, first.symbol, first.side)
        after, after_reason = _after_decision_favorable_pct(
            first, outcome, sidecar_bars
        )
        joined = preference.get(
            (
                session_date,
                first.symbol,
                first.side,
                _CHANNEL_BY_SOURCE_VERDICT.get((first.source, first.verdict), ""),
            )
        )
        journal_r = _float_or_none((joined or {}).get("journal_r"))
        measures: dict[str, float | None] = {
            "day_mfe_pct": outcome.mfe_pct if outcome is not None else None,
            "mfe_pct_after_decision": after,
            "journal_r": journal_r,
        }
        unavailable: dict[str, str] = {}
        if measures["day_mfe_pct"] is None:
            unavailable["day_mfe_pct"] = _unmeasured_reason(outcome, "mfe_pct")
        if after is None:
            unavailable["mfe_pct_after_decision"] = after_reason
        if journal_r is None:
            unavailable["journal_r"] = (
                "no journal trade is joined to this statement "
                f"({(joined or {}).get('match_state') or 'not in the preference report'})"
            )
        detail: dict[str, Any] = {
            "verdict": first.verdict,
            "reason": first.reason,
            "source": first.source,
            "like_mode": str(first.detail.get("like_mode") or ""),
            "reason_codes": tuple(first.detail.get("reason_codes") or ()),
            "channel": _CHANNEL_BY_SOURCE_VERDICT.get(
                (first.source, first.verdict), ""
            ),
            "match_state": str((joined or {}).get("match_state") or ""),
            "journal_net_pnl": _float_or_none((joined or {}).get("journal_net_pnl")),
            "trade_id": str((joined or {}).get("trade_id") or ""),
            "occurrence_ids": tuple(
                member.capture_id for member in members if member.capture_id
            ),
        }
        for name, value in first.detail.items():
            detail.setdefault(name, value)
        rows.append(
            RecapRow(
                symbol=first.symbol,
                side=first.side,
                source=first.source,
                category=first.category,
                capture_id=first.capture_id,
                observed_at=first.observed_at,
                measures=measures,
                unavailable=unavailable,
                detail=detail,
                pick_key=(session_date, first.symbol, first.side, first.category),
                d1_environment=labels.get(session_date, UNKNOWN),
                occurrences=len(members),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# credit starts at the decision's own timestamp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Bar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float


def _pass_bar_index(
    annotations: _Store, session_date: str, annotations_path: Path
) -> dict[str, tuple[_Bar, ...]]:
    """The M5 bars behind each day-trade pass, keyed by its event id.

    A capture sidecar's `dt` is NAIVE desk-local wall time (621 of the desk's
    622 rows), so the zone is ATTACHED here rather than the decision's offset
    being stripped - N1's rule, on the same seam that produced it.
    """
    from ui.annotations import pass_bars as pass_bars_module

    index: dict[str, tuple[_Bar, ...]] = {}
    for row in annotations.rows:
        if str(row.get("event_type") or "") != "pass":
            continue
        if _session_text(row.get("session_date")) != session_date:
            continue
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        try:
            payload = pass_bars_module.read_pass_bars(
                row, annotations_path=annotations_path
            )
        except Exception:  # noqa: BLE001 - a lost chart never costs the recap
            payload = {}
        bars: list[_Bar] = []
        for bar in payload.get("bars") or ():
            moment = _parse_moment(bar.get("dt"))
            if moment is None:
                continue
            try:
                bars.append(
                    _Bar(
                        dt=moment,
                        open=float(bar.get("open")),
                        high=float(bar.get("high")),
                        low=float(bar.get("low")),
                        close=float(bar.get("close")),
                    )
                )
            except (TypeError, ValueError):
                continue
        if bars:
            index[event_id] = tuple(sorted(bars, key=lambda item: item.dt))
    return index


def _after_decision_favorable_pct(
    decision: _Decision,
    outcome: _Outcome | None,
    sidecar_bars: Mapping[str, tuple[_Bar, ...]],
) -> tuple[float | None, str]:
    """The part of the move the trader could still have had.

    A decision made at 12:30 may not take credit for the 10:00 high. Where the
    bars behind the decision exist - a day-trade pass writes an M5 sidecar - the
    reference is the LAST COMPLETED bar at the decision and the excursion is
    measured forward from it, side-adjusted. Where they do not, the answer is
    `unavailable`: the day's MFE would be a claim about timing nothing recorded,
    and 0.0 would be a claim that nothing happened.
    """
    moment = decision.observed_at
    if moment is None:
        return None, "the decision carries no timestamp, so nothing can be measured from it"
    bars = sidecar_bars.get(decision.capture_id) or ()
    if not bars:
        return None, (
            "no bar series is reachable for this decision - the stores record "
            "WHAT moved on the session but not WHEN, so the part of it available "
            f"after {moment.strftime('%H:%M')} is unmeasured"
        )
    before = [bar for bar in bars if bar.dt < moment]
    after = [bar for bar in bars if bar.dt >= moment]
    if not after:
        return None, "the captured bars all pre-date the decision"
    reference = before[-1].close if before else after[0].open
    if not reference:
        return None, "the bar at the decision carries no usable price"
    side = decision.side or (outcome.side if outcome is not None else "")
    if side == "SHORT":
        extreme = min(bar.low for bar in after)
        return (reference - extreme) / reference * 100.0, ""
    extreme = max(bar.high for bar in after)
    return (extreme - reference) / reference * 100.0, ""


# ---------------------------------------------------------------------------
# view 4 - my rejected picks that worked
# ---------------------------------------------------------------------------


def _rejected_that_worked_view(
    session_date: str,
    decision_rows: Sequence[RecapRow],
    outcomes: Sequence[_Outcome],
) -> RecapView:
    """The refusals whose later path went the way the trader turned down.

    Both numbers travel and the trader's own words travel with them: a later
    rise ALONE does not prove a timing or risk veto wrong, so the adverse
    movement the trader would have sat through is beside the favorable one. The
    credited move - the part still available after the decision - is a THIRD
    column and is never blended with the day's path.
    """
    rows: list[RecapRow] = []
    for row in decision_rows:
        if row.detail.get("verdict") not in REJECT_VERDICTS:
            continue
        outcome = _outcome_for(outcomes, session_date, row.symbol, row.side)
        favorable = outcome.mfe_pct if outcome is not None else None
        if favorable is None or favorable <= 0:
            continue
        measures: dict[str, float | None] = {
            "favorable_pct": favorable,
            "favorable_pct_after_decision": row.measures.get("mfe_pct_after_decision"),
            "adverse_pct": outcome.mae_pct if outcome is not None else None,
        }
        unavailable: dict[str, str] = {}
        if measures["favorable_pct_after_decision"] is None:
            unavailable["favorable_pct_after_decision"] = row.unavailable.get(
                "mfe_pct_after_decision", "not measured"
            )
        if measures["adverse_pct"] is None:
            unavailable["adverse_pct"] = _unmeasured_reason(outcome, "mae_pct")
        rows.append(
            RecapRow(
                symbol=row.symbol,
                side=row.side,
                source=row.source,
                category=row.category,
                capture_id=row.capture_id,
                observed_at=row.observed_at,
                measures=measures,
                unavailable=unavailable,
                detail=dict(row.detail),
                pick_key=row.pick_key,
                d1_environment=row.d1_environment,
                occurrences=row.occurrences,
            )
        )
    view = RecapView(
        name="rejected_that_worked",
        cohort=COHORTS["rejected_that_worked"],
        window=(session_date, session_date),
        rows=tuple(rows),
        pending=(),
        sort_key=SORT_KEYS["rejected_that_worked"][0],
        sort_keys=SORT_KEYS["rejected_that_worked"],
        note=(
            "A later rise alone does not prove a timing or risk refusal wrong - "
            "the adverse move is beside it."
        ),
    )
    return _with_rows(view, view.sorted_by(view.sort_key))


# ---------------------------------------------------------------------------
# the side stores
# ---------------------------------------------------------------------------


def _preference_index(
    store: _Store,
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    """WS-5B's symmetric report, keyed by session, name, side and CHANNEL."""
    index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in store.rows:
        key = (
            _session_text(row.get("session_date")),
            _symbol(row.get("symbol")),
            _side(row.get("side")),
            str(row.get("channel") or "").strip(),
        )
        index.setdefault(key, dict(row))
    return index


def _staged_picks(path: Path, now: datetime) -> tuple[dict[str, tuple[str, ...]], SourceCoverage]:
    """AWAY's staged picks, day-scoped by the desk's own reader.

    The block moves onto this page unchanged: AWAY still STAGES and never
    adopts, and nothing here adopts for it.
    """
    target = Path(path)
    try:
        from autopilot_core import load_auto_populate_pending_picks

        payload = load_auto_populate_pending_picks(target, now=now)
    except Exception as exc:  # noqa: BLE001
        return {"long": (), "short": ()}, _unreadable("staged_picks", target, exc)
    pending = payload.get("pending") or {}
    staged = {
        side: tuple(sorted(_symbol(name) for name in (pending.get(side) or {})))
        for side in ("long", "short")
    }
    rows = [{"side": side, "symbol": name} for side in staged for name in staged[side]]
    return staged, _coverage("staged_picks", target, rows, [_parse_moment(payload.get("date"))])


def _environment_labels(path: Path) -> tuple[dict[str, str], SourceCoverage]:
    """WS-ENV's session labels. A session nobody labelled reads `unknown`."""
    target = Path(path)
    try:
        import d1_environment_store

        rows = d1_environment_store.read_rows(target)
        labels = d1_environment_store.labels_by_session(path=target)
    except Exception as exc:  # noqa: BLE001
        return {}, _unreadable("environment_labels", target, exc)
    if not target.exists():
        return {}, _unreadable(
            "environment_labels", target, FileNotFoundError(str(target))
        )
    moments = [_parse_moment(row.get("written_at")) for row in rows]
    return dict(labels), _coverage("environment_labels", target, rows, moments)


def _working_lately(path: Path) -> tuple[dict[str, Any], SourceCoverage]:
    """The desk's ONE Working-lately snapshot, read - never rebuilt."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, _unreadable("working_lately", target, exc)
    if not isinstance(payload, dict):
        return {}, _unreadable(
            "working_lately", target, ValueError("the snapshot is not an object")
        )
    cells = payload.get("cells") or []
    return payload, _coverage(
        "working_lately",
        target,
        cells if isinstance(cells, list) else [],
        [_parse_moment(payload.get("as_of"))],
    )


# ---------------------------------------------------------------------------
# the read
# ---------------------------------------------------------------------------


def read_session(
    session_date: str,
    *,
    lookback_sessions: int = 3,
    now: datetime | None = None,
    sources: RecapSources | None = None,
) -> RecapSession:
    """One session, read from the durable stores. Pure; worker-only.

    Its whole input is a session, a lookback, a clock and a set of PATHS - no
    process-scoped feed, no in-memory alert list, no tracker. Call it on a
    worker: it opens nine files.
    """
    session_date = _session_text(session_date)
    lookback_sessions = max(1, int(lookback_sessions))
    now = now or datetime.now()
    sources = sources or RecapSources()

    intraday = _read_csv("intraday_outcomes", sources.intraday_outcomes, "logged_at")
    tier = _read_csv("tier_outcomes", sources.tier_outcomes, "run_timestamp")
    horizon = _read_csv(
        "session_horizon_outcomes", sources.session_horizon_outcomes, "scan_date"
    )
    human_focus = _read_csv(
        "human_focus_outcomes", sources.human_focus_outcomes, "updated_at"
    )
    preference_store = _read_csv(
        "preference_report", sources.preference_report, "generated_at"
    )
    annotations = _read_jsonl("annotations", sources.annotations, "created_at")
    feedback = _read_jsonl("pick_feedback", sources.pick_feedback, "ts")
    favorites = _read_jsonl("swing_favorites", sources.swing_favorites, "event_at")
    events = _read_jsonl("review_events", sources.review_events, "ts")

    labels, environment_coverage = _environment_labels(sources.environment_labels)
    staged, staged_coverage = _staged_picks(sources.staged_picks, now)
    snapshot, snapshot_coverage = _working_lately(sources.working_lately)

    coverage: dict[str, SourceCoverage] = {}
    for store in (
        intraday,
        tier,
        horizon,
        annotations,
        feedback,
        favorites,
        human_focus,
        events,
        preference_store,
    ):
        coverage[store.coverage.name] = store.coverage
    coverage["staged_picks"] = staged_coverage
    coverage["environment_labels"] = environment_coverage
    coverage["working_lately"] = snapshot_coverage
    coverage = {name: coverage[name] for name in SOURCE_NAMES if name in coverage}

    outcomes = _outcomes(intraday)
    worked_today = _worked_today_view(session_date, outcomes, labels)
    recent_swings = _recent_swings_view(
        _lookback_window(session_date, lookback_sessions),
        lookback_sessions,
        horizon,
        outcomes,
        labels,
    )
    decision_rows = _decision_rows(
        session_date,
        _decisions(session_date, annotations, feedback, favorites, events),
        outcomes,
        annotations,
        _preference_index(preference_store),
        labels,
        Path(sources.annotations),
    )
    my_decisions = RecapView(
        name="my_decisions",
        cohort=COHORTS["my_decisions"],
        window=(session_date, session_date),
        rows=decision_rows,
        pending=(),
        sort_key=SORT_KEYS["my_decisions"][0],
        sort_keys=SORT_KEYS["my_decisions"],
        note=(
            "Likes, claimed likes, swing favorites, passes, vetoes and "
            "not-today are separate facts; a repeated click links rather than "
            "counting twice."
        ),
    )
    my_decisions = _with_rows(my_decisions, my_decisions.sorted_by(my_decisions.sort_key))
    rejected = _rejected_that_worked_view(session_date, my_decisions.rows, outcomes)

    return RecapSession(
        session_date=session_date,
        lookback_sessions=lookback_sessions,
        provisional=_is_provisional(session_date, now),
        now=now,
        coverage=coverage,
        worked_today=worked_today,
        recent_swings=recent_swings,
        my_decisions=my_decisions,
        rejected_that_worked=rejected,
        staged_picks=staged,
        working_lately=snapshot,
    )


__all__ = [
    "COHORTS",
    "ENDORSE_VERDICTS",
    "RecapRow",
    "RecapSession",
    "RecapSources",
    "RecapView",
    "REJECT_VERDICTS",
    "SORT_KEYS",
    "SOURCE_NAMES",
    "SourceCoverage",
    "UNKNOWN",
    "VIEW_NAMES",
    "default_session",
    "read_session",
]
