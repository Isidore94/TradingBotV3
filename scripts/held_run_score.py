"""Did the level hold, and then how far did it run — V1 item 2. SHADOW ONLY.

Decision 0016 answer 4, in the trader's words: *"the intraday level holds, then
the name runs. Rank by maximum favourable excursion — the most the move offered —
not by any exit; exiting well is the trader's job."* And: *"the best day trades
are also swings, so an M5 alert on a name that also carries a D1 setup outranks
the same alert on a name that does not."*

    held_run_score = P(level held) x trimmed-mean MFE_R of the ones that held

**A SECOND score, never a replacement.** The champion tier comes from
`bounce_bot_lib.learning`'s `production_r`, a blend of end-of-day entry quality
and 60-minute quick production, and it gates alerts. Nothing here touches it:
the tier gate, the mutes and the PROVEN stamp are untouched, and the existing
bounce goldens are byte-identical. This answers a different question — *was the
level worth trusting, and what did the move offer* — and it is displayed beside
the tier, never instead of it.

**Why the two halves multiply.** A segment where the level almost always holds
but the move offers 0.3R is not a good alert, and neither is one that offers 4R
on the one occasion in ten it holds. The product is the expected offer per alert,
which is the thing the trader is choosing between when two alerts fire at once.

**"Held" is a MEASURED 30-minute question** (packet Q1, 2026-09-04). An episode
is `measured_held`, `measured_broken`, `pending` or `unmeasured`, and only the
first is held; the others are counted and shown beside the headline (`n_measured`
/ `n` and `coverage`), never assumed. The window is the shared exchange-session
"lately" window with its gaps reported (`window_report`). The D1 dimension keeps
the setup's SIDE (`aligned` / `opposed` / `none` / `unknown`) and its basis is
retrospective: the snapshot carries no time of day.

**"Held" is a 30-minute question**, not an end-of-day one. The trader's day trade
lasts minutes to hours, and a level that gives way an hour later gave the trade
its chance first. `stop_hit` inside the first 30 minutes is the failure; anything
after it is the exit, which is the trader's job.

Deterministic: no model, no network, no fetch. Nothing here reaches a detector, a
score that gates, an alert, a watchlist, Focus, the review queue or
`review_policy.json`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

#: The window in which the level has to hold. Minutes, from the entry.
HELD_WINDOW_MINUTES = 30

#: Measurement states (packet Q1, process review 2026-09-04 finding 1). "Held"
#: used to be `not broke_early`, so an episode nothing had ever followed up read
#: as held: 979 of 8,161 recent episodes on 2026-09-04. Only MEASURED_HELD is
#: held now; the rest are counted and shown, never assumed.
MEASURED_HELD = "measured_held"
MEASURED_BROKEN = "measured_broken"
PENDING = "pending"
UNMEASURED = "unmeasured"
MEASUREMENT_STATES = (MEASURED_HELD, MEASURED_BROKEN, PENDING, UNMEASURED)

#: Why an episode is pending / unmeasured.
REASON_NO_FOLLOW_UP = "no_follow_up"
REASON_WINDOW_NOT_REACHED = "window_not_reached"
#: The outcome log carries `stop_hit` as a boolean over ALL bars since entry and
#: no first-break time (`legacy.py` `BOUNCE_OUTCOME_COLUMNS`), so a stop first
#: reported past the window with no earlier row bracketing the window may have
#: gone at minute 13 or minute 90. Adding `stop_hit_at` to the producer is a
#: `legacy.py` change under the file-scoped ask-first rule - owed, not built.
REASON_BREAK_TIME_UNKNOWN = "break_time_unknown"

#: The D1 overlap (finding 2). The scanner's scoring snapshot carries a `side`
#: per setup, and the join used to drop it: a SHORT swing setup marked a long
#: M5 alert as "carrying a D1 setup" (8 of 2,646 live episodes). Only ALIGNED
#: carries the D1 privilege; OPPOSED, NONE and UNKNOWN never do, and V4's
#: priority switch inherits that. The snapshot carries `scan_date` and no time
#: of day, so this join can never claim the setup was KNOWN when the alert
#: fired - its basis is retrospective and every summary says so.
D1_ALIGNED = "aligned"
D1_OPPOSED = "opposed"
D1_NONE = "none"
D1_UNKNOWN = "unknown"
D1_BASIS = "same_session_retrospective"

#: The rolling window the score is measured over. Decision 0016 answer 6:
#: "lately" is about 20 sessions and carries NO regime label.
#:
#: V3 item 3: ONE CONSTANT, and it lives in `evidence_stats` with the rest of the
#: desk's statistics contract. This name is kept as an alias so nothing that
#: imports it breaks, but the NUMBER is no longer written here - two modules that
#: each own a "lately" eventually disagree about the trader's own word.
from evidence_stats import LATELY_SESSIONS as ROLLING_SESSIONS  # noqa: E402

#: Time buckets - THE CHAMPION'S OWN, read from `bounce_bot_lib.learning`
#: rather than restated (R4 fix round 1).
#:
#: This module used to declare four of its own (`open_30m`, `morning`, `midday`,
#: `power_hour`) and compute them by comparing raw wall-clock hours against
#: Eastern cutoffs. Two things were wrong with that. The desk is on Pacific time
#: and `entry_time` in the outcome log is DESK-LOCAL, so 10:40 PT was judged
#: against a 10:00/11:30/15:00 boundary set that only means anything in New York
#: - which is precisely the defect `time_bucket_for`'s own docstring records
#: itself as having fixed ("on a Pacific machine that mislabeled nearly the
#: entire session"). And a second vocabulary meant the Daytrade Tracker's Time of
#: Day tab could not join: measured on the live stores, **2 of 10 rows matched**,
#: and the other eight went blank for a spelling reason while the docs called the
#: tab measurable.
TIME_BUCKETS = (
    "opening_drive",
    "late_morning",
    "midday",
    "afternoon",
    "closing_window",
)

#: The market environments the alert context already records. Passed through
#: verbatim - this module never re-derives a regime, and decision 0016 answer 6
#: says "lately" needs no regime label of its own.
UNKNOWN = "unknown"


def time_bucket(entry_time: Any) -> str:
    """Which part of the session an entry sits in. `unknown` when unreadable.

    ONE DEFINITION, and it is the champion's: `bounce_bot_lib.learning`'s public
    `time_bucket_for`, measured in ELAPSED MINUTES OF ITS OWN SESSION rather than
    against wall-clock hours. Called, never copied - a drift-tested copy is the
    right shape when the source might be absent (that is why `group_rrs` has
    one), and this source ships beside us and the tracker panel imports it
    already.

    Reading it rather than restating it is what lets the Daytrade Tracker join
    this module's cells to the aggregator's rows at all; see :data:`TIME_BUCKETS`
    for the two defects the private copy carried.

    A row whose time cannot be read is bucketed `unknown` rather than dropped -
    it still counts toward what was NOT measurable, which is the honest
    denominator.
    """
    stamp = _as_datetime(entry_time)
    if stamp is None:
        return UNKNOWN
    try:
        from bounce_bot_lib.learning import time_bucket_for
    except Exception:  # pragma: no cover - the module ships beside this one
        return UNKNOWN
    return str(time_bucket_for(stamp) or UNKNOWN)


#: A `10_candle_high` and a `10_candle_low` are one bounce TYPE to the
#: aggregator, which records both as `10_candle`. The module reads its type off
#: the event id, which keeps the side.
_TYPE_ALIASES = ("10_candle",)


def bounce_components(bounce_type: Any) -> tuple[str, ...]:
    """The individual bounce types inside one episode's label.

    `bounce_type_from_event_id` joins multiple types with `-`
    (`eod_vwap-impulse_retest_vwap_eod-vwap`), and the aggregator's `bounce_type`
    dimension counts an episode under EACH of its types - which is what
    `evaluate_bounce_quality` does with `bounce_types` too. Splitting here is
    what took the live Bounce Types join from 28 of 36 rows to 36 of 36: the
    eight it missed were the ones that only ever appear inside a combination.
    """
    parts = []
    for raw in str(bounce_type or "").split("-"):
        part = raw.strip()
        if not part:
            continue
        for alias in _TYPE_ALIASES:
            if part.startswith(alias):
                part = alias
                break
        parts.append(part)
    return tuple(parts)


def bounce_combo(bounce_type: Any) -> str:
    """The whole combination, spelled the way the aggregator spells it.

    The aggregator joins with `+`; the event id joins with `-`. That single
    separator is why the Combos tab matched **0 of 59** live rows while the docs
    called it unanswerable - it took a normalisation, not a missing measurement.
    58 of 59 match now, and the one that does not is a combination that did not
    fire inside the rolling window, which is an honest absence.
    """
    parts = bounce_components(bounce_type)
    return "+".join(parts) if parts else UNKNOWN


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for cut in (19, 16, 10):
        try:
            return datetime.fromisoformat(text[:cut])
        except ValueError:
            continue
    return None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


@dataclass
class Episode:
    """One alert, followed far enough to answer both halves of the question."""

    event_id: str
    trade_date: str
    symbol: str
    direction: str
    bounce_type: str
    entry_time: str
    market_environment: str = UNKNOWN
    #: `aligned` / `opposed` / `none` / `unknown` - see `d1_alignment`.
    d1_alignment: str = D1_UNKNOWN
    #: One of `MEASUREMENT_STATES`; the only held one is `MEASURED_HELD`.
    measurement: str = PENDING
    measurement_reason: str = REASON_NO_FOLLOW_UP
    #: The best MFE seen on any followed row for this event.
    mfe_r: float | None = None
    # Accumulators for the classification; `_finalize` turns them into a state.
    _broke_inside: bool = field(default=False, repr=False, compare=False)
    _held_past_window: bool = field(default=False, repr=False, compare=False)
    _stop_after_window: bool = field(default=False, repr=False, compare=False)
    _follow_up_rows: int = field(default=0, repr=False, compare=False)
    _has_final: bool = field(default=False, repr=False, compare=False)

    @property
    def held(self) -> bool:
        """MEASURED held, and nothing else - never the absence of a break."""
        return self.measurement == MEASURED_HELD

    @property
    def broke_early(self) -> bool:
        return self.measurement == MEASURED_BROKEN

    @property
    def measured(self) -> bool:
        return self.measurement in (MEASURED_HELD, MEASURED_BROKEN)

    @property
    def d1_setup_present(self) -> bool:
        """Aligned only. An opposed, absent or unknown setup carries no privilege."""
        return self.d1_alignment == D1_ALIGNED

    def segment(self) -> tuple[str, str, str, str]:
        return (
            self.bounce_type or UNKNOWN,
            time_bucket(self.entry_time),
            self.market_environment or UNKNOWN,
            self.d1_alignment or D1_UNKNOWN,
        )

    def _finalize(self, as_of: str) -> None:
        """Turn the accumulated rows into ONE measurement state.

        Order matters and is the review's rule: a stop placed inside the window
        is broken; a no-stop row that reached the window is held; a stop first
        seen past the window with nothing bracketing it is unknown; and an
        episode no row answered is pending on its own session, unmeasured after.
        """
        if self._broke_inside:
            self.measurement, self.measurement_reason = MEASURED_BROKEN, ""
        elif self._held_past_window:
            self.measurement, self.measurement_reason = MEASURED_HELD, ""
        elif self._stop_after_window:
            self.measurement, self.measurement_reason = UNMEASURED, REASON_BREAK_TIME_UNKNOWN
        else:
            reason = REASON_NO_FOLLOW_UP if not self._follow_up_rows else REASON_WINDOW_NOT_REACHED
            still_open = (not self._has_final) and bool(self.trade_date) and self.trade_date >= as_of
            self.measurement = PENDING if still_open else UNMEASURED
            self.measurement_reason = reason


@dataclass
class Segment:
    """One (bounce_type, time_bucket, environment, d1_setup) cell."""

    key: tuple[str, str, str, str]
    episodes: int = 0
    held: int = 0
    broken: int = 0
    pending: int = 0
    unmeasured: int = 0
    mfe_of_held: list[float] = field(default_factory=list)

    @property
    def measured(self) -> int:
        return self.held + self.broken

    @property
    def hold_rate(self) -> float | None:
        """held / MEASURED. Never held / episodes: the unmeasured are not holds."""
        return (self.held / self.measured) if self.measured else None

    @property
    def coverage(self) -> float | None:
        return (self.measured / self.episodes) if self.episodes else None

    def add(self, episode: "Episode") -> None:
        self.episodes += 1
        if episode.measurement == MEASURED_HELD:
            self.held += 1
            if episode.mfe_r is not None:
                self.mfe_of_held.append(episode.mfe_r)
        elif episode.measurement == MEASURED_BROKEN:
            self.broken += 1
        elif episode.measurement == PENDING:
            self.pending += 1
        else:
            self.unmeasured += 1

    def summary(self, *, min_n: int | None = None) -> dict[str, Any]:
        """The cell as the desk reports it, floors included.

        `evidence_stats` is the desk's ONE statistics contract, so the trimmed
        mean, the floor and the discovery label come from it rather than from a
        second arithmetic written here.
        """
        import evidence_stats

        stats = evidence_stats.summarize(
            self.mfe_of_held,
            min_n=evidence_stats.MIN_REPORTABLE_N if min_n is None else min_n,
        )
        trimmed = (stats.get("clipped") or {}).get("trimmed_mean")
        hold_rate = self.hold_rate
        score = None
        if hold_rate is not None and trimmed is not None:
            score = hold_rate * trimmed
        bounce_type, bucket, environment, d1 = self.key
        return {
            "bounce_type": bounce_type,
            "time_bucket": bucket,
            "market_environment": environment,
            "d1_alignment": d1,
            "d1_setup_present": d1 == D1_ALIGNED,
            "d1_basis": D1_BASIS,
            "n": self.episodes,
            "n_held": self.held,
            "n_broken": self.broken,
            "n_measured": self.measured,
            "n_pending": self.pending,
            "n_unmeasured": self.unmeasured,
            "coverage": self.coverage,
            "hold_rate": hold_rate,
            "mean_mfe_r_of_held": trimmed,
            "held_run_score": score,
            # The floor is on the HELD episodes, because the MFE half of the
            # score is measured only over those - a cell with 40 alerts of which
            # 3 held has three readings of the run, not forty.
            "meets_floor": bool(stats.get("meets_n_floor")),
            "evidence_label": stats.get("evidence_label", "discovery"),
        }


def build_episodes(
    rows: Iterable[Mapping[str, Any]],
    *,
    d1_setups_by_session: Mapping[str, Mapping[str, set]] | None = None,
    as_of: Any = None,
) -> list[Episode]:
    """Fold the outcome log's many rows per event into one episode each.

    The log carries a `registered` row, a stream of `update`s, the milestone
    rows and a `final` for every alert. What this needs from them is narrow:
    the identity and context from whichever row carries it, whether the stop was
    hit inside the first thirty minutes, and the best MFE reached at any point.

    `d1_setups_by_session` is `{trade_date: {SYMBOL: {"LONG", "SHORT"}}}` read
    from the scanner's own output files by the caller. **Never fetched** - a
    study that reached for a quote would be a study that could not be re-run.
    `as_of` (a date or ISO string, default today) decides whether an unanswered
    episode is still PENDING or UNMEASURED for good.
    """
    from setup_scoreboard import bounce_type_from_event_id

    setups = d1_setups_by_session  # None = no snapshot = UNKNOWN everywhere
    as_of_text = _as_of_text(as_of)
    episodes: dict[str, Episode] = {}
    for row in rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        episode = episodes.get(event_id)
        if episode is None:
            trade_date = str(row.get("trade_date") or "").strip()
            symbol = str(row.get("symbol") or "").strip().upper()
            episode = Episode(
                event_id=event_id,
                trade_date=trade_date,
                symbol=symbol,
                direction=str(row.get("direction") or "").strip().lower(),
                bounce_type=str(bounce_type_from_event_id(event_id) or UNKNOWN),
                entry_time=str(row.get("entry_time") or ""),
            )
            episode.d1_alignment = d1_alignment(setups, trade_date, symbol, episode.direction)
            episodes[event_id] = episode

        if not episode.entry_time:
            episode.entry_time = str(row.get("entry_time") or "")
        if episode.market_environment == UNKNOWN:
            environment = _environment_of(row)
            if environment:
                episode.market_environment = environment

        mfe = _as_float(row.get("mfe_r"))
        if mfe is not None and (episode.mfe_r is None or mfe > episode.mfe_r):
            episode.mfe_r = mfe

        kind = str(row.get("event_type") or "").strip().lower()
        if kind == "final":
            episode._has_final = True
        if kind != "registered":
            episode._follow_up_rows += 1
        stop_hit = _as_bool(row.get("stop_hit"))
        if stop_hit:
            if _within_hold_window(episode, row):
                episode._broke_inside = True
            else:
                episode._stop_after_window = True
        else:
            # Rule 2 needs a row that MEASURED bars. A `registered` row saw
            # none (`bars_elapsed=0`, blank `minutes_elapsed`) and its
            # `logged_at` is a replay/backfill write time - on the live log a
            # median 1,013 minutes after the entry - so reading that gap as
            # "the window passed with the stop intact" called 728 unmeasured
            # episodes held (Q1 review, 2026-09-04). Rule 1 keeps its
            # fallback: a stop we cannot place is still a stop.
            minutes = _measured_minutes(episode, row, kind)
            if minutes is not None and minutes >= HELD_WINDOW_MINUTES:
                episode._held_past_window = True
    for episode in episodes.values():
        episode._finalize(as_of_text)
    return list(episodes.values())


def _as_of_text(value: Any) -> str:
    """ISO date for the as-of session. Today, market-local, when not given."""
    if value is None or value == "":
        from datetime import date as _date

        return _date.today().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    return str(value).strip()[:10]


def _environment_of(row: Mapping[str, Any]) -> str:
    try:
        context = json.loads(str(row.get("context_json") or "") or "{}")
    except ValueError:
        return ""
    return str((context or {}).get("market_environment") or "").strip()


def _elapsed_minutes(episode: Episode, row: Mapping[str, Any]) -> float | None:
    """Minutes from the entry to what this row measured, or None if unplaceable.

    `minutes_elapsed` when the row carries one, else the gap between the row's
    own timestamp and the entry. Note this is the LAST bar the row knew about,
    never the time a stop went - the log has no such column.
    """
    minutes = _as_float(row.get("minutes_elapsed"))
    if minutes is not None:
        return minutes
    entry = _as_datetime(episode.entry_time)
    stamp = _as_datetime(row.get("logged_at"))
    if entry is None or stamp is None:
        return None
    return (stamp - entry) / timedelta(minutes=1)


def _measured_minutes(episode: Episode, row: Mapping[str, Any], kind: str) -> float | None:
    """Minutes of BARS this row measured, or None for a row that measured none.

    `minutes_elapsed` when present. Without it, the `logged_at` gap counts only
    for a non-`registered` row that reports `bars_elapsed > 0`; a registration
    or a bar-less row is not a measurement of anything.
    """
    if kind == "registered":
        return None
    minutes = _as_float(row.get("minutes_elapsed"))
    if minutes is not None:
        return minutes
    bars = _as_float(row.get("bars_elapsed"))
    if bars is None or bars <= 0:
        return None
    return _elapsed_minutes(episode, row)


def _within_hold_window(episode: Episode, row: Mapping[str, Any]) -> bool:
    """Whether this row's stop hit happened inside the hold window.

    A row that says neither `minutes_elapsed` nor a usable `logged_at` is
    treated as INSIDE the window: a stop we cannot place is a stop, and calling
    it late would quietly improve every hold rate on the board.
    """
    minutes = _elapsed_minutes(episode, row)
    if minutes is None:
        return True
    return minutes <= HELD_WINDOW_MINUTES


def window_bounds(*, sessions: int = ROLLING_SESSIONS, as_of: Any = None) -> tuple[str, str]:
    """`(start, end)` ISO dates of the shared "lately" window - ONE definition.

    `evidence_stats.lately_window` walks the exchange calendar, so this module
    and the swing path agree on the trader's own word (V3 item 3). Until packet
    Q1 this module kept "the last N distinct dates present in the file", which
    silently widened on sparse data.
    """
    import evidence_stats

    return evidence_stats.lately_window(end=_as_of_text(as_of), sessions=sessions)


def _in_window(trade_date: str, bounds: tuple[str, str]) -> bool:
    return bool(trade_date) and bounds[0] <= trade_date <= bounds[1]


def recent_sessions(
    episodes: Iterable[Episode], *, sessions: int = ROLLING_SESSIONS, as_of: Any = None
) -> set:
    """The trade dates present in the data that fall inside the lately window."""
    bounds = window_bounds(sessions=sessions, as_of=as_of)
    return {
        episode.trade_date
        for episode in episodes
        if _in_window(episode.trade_date, bounds)
    }


def window_report(
    episodes: Iterable[Episode], *, sessions: int = ROLLING_SESSIONS, as_of: Any = None
) -> dict[str, Any]:
    """The window and its gaps, for a status line: which sessions carry no data.

    Exchange sessions come from `market_calendar`; a date the calendar refuses
    is neither counted nor reported missing. A weekend row, if one ever exists,
    is kept as data and is not a session.
    """
    from datetime import date as _date, timedelta as _timedelta

    bounds = window_bounds(sessions=sessions, as_of=as_of)
    present = recent_sessions(episodes, sessions=sessions, as_of=as_of)
    session_days: list[str] = []
    try:
        from market_calendar import is_session

        cursor = _date.fromisoformat(bounds[0])
        last = _date.fromisoformat(bounds[1])
        while cursor <= last:
            try:
                if is_session(cursor):
                    session_days.append(cursor.isoformat())
            except Exception:  # noqa: BLE001 - outside the validated range: not a session we can name
                pass
            cursor += _timedelta(days=1)
    except Exception:  # noqa: BLE001 - a report is never worth a blank readout
        session_days = []
    with_data = [day for day in session_days if day in present]
    return {
        "start": bounds[0],
        "end": bounds[1],
        "sessions": len(session_days),
        "sessions_with_data": len(with_data),
        "missing_sessions": [day for day in session_days if day not in present],
    }


def build_segments(
    episodes: Iterable[Episode],
    *,
    sessions: int = ROLLING_SESSIONS,
    min_n: int | None = None,
    as_of: Any = None,
) -> list[dict[str, Any]]:
    """Every segment, ranked by score, with the unmeasurable ones still listed.

    A cell under the floor is REPORTED with its n and its discovery label rather
    than dropped: the trader needs to know a segment exists and is thin, which is
    a different fact from a segment that has never fired.
    """
    episodes = list(episodes)
    bounds = window_bounds(sessions=sessions, as_of=as_of)
    cells: dict[tuple, Segment] = {}
    for episode in episodes:
        if not _in_window(episode.trade_date, bounds):
            continue
        key = episode.segment()
        cell = cells.get(key)
        if cell is None:
            cell = cells[key] = Segment(key=key)
        cell.add(episode)
    summaries = [cell.summary(min_n=min_n) for cell in cells.values()]
    summaries.sort(
        key=lambda cell: (
            cell["held_run_score"] is None,
            -(cell["held_run_score"] or 0.0),
        )
    )
    return summaries


def segment_index(summaries: Iterable[Mapping[str, Any]]) -> dict[tuple, Mapping[str, Any]]:
    """`{segment key: summary}` for a per-alert lookup."""
    return {
        (
            str(cell.get("bounce_type") or UNKNOWN),
            str(cell.get("time_bucket") or UNKNOWN),
            str(cell.get("market_environment") or UNKNOWN),
            str(cell.get("d1_alignment") or D1_UNKNOWN),
        ): cell
        for cell in summaries
    }


def alert_cell(
    index: Mapping[tuple, Mapping[str, Any]] | None,
    *,
    bounce_type: Any,
    entry_time: Any,
    market_environment: Any = UNKNOWN,
    d1_setup_present: Any = None,
    d1_alignment: Any = None,
) -> Mapping[str, Any] | None:
    """One alert's cell out of `segment_index`, or None.

    `segment_index` said it existed "for a per-alert lookup" and no caller ever
    built the key, so the alert row had nothing to read (R4 A10). The key is
    built HERE rather than at each call site: four positional strings that must
    agree with `Episode.segment()` is exactly the sort of thing that drifts.
    """
    if not index:
        return None
    if d1_alignment is not None:
        alignment = str(d1_alignment).strip().lower() or D1_UNKNOWN
    elif d1_setup_present is None:
        alignment = D1_UNKNOWN
    else:
        # A bool caller cannot tell "no setup" from "no snapshot"; False is read
        # as NONE. A caller that can tell should pass the string.
        alignment = D1_ALIGNED if d1_setup_present else D1_NONE
    return index.get(
        (
            str(bounce_type or UNKNOWN).strip() or UNKNOWN,
            time_bucket(entry_time),
            str(market_environment or UNKNOWN).strip() or UNKNOWN,
            alignment,
        )
    )


def alert_suffix(cell: Mapping[str, Any] | None) -> str:
    """"held 71% / ran 1.9R" for the M5 alert row, or "" below the floor.

    BLANK below the floor, never a number in brackets. A row that showed
    "held 100% / ran 3.2R (n=2)" would be read as a strong segment by anyone
    glancing at a list, and a glance is exactly what this row is for.
    """
    if not cell or not cell.get("meets_floor"):
        return ""
    hold_rate = cell.get("hold_rate")
    ran = cell.get("mean_mfe_r_of_held")
    if hold_rate is None or ran is None:
        return ""
    return f"held {hold_rate * 100:.0f}% / ran {ran:.1f}R"


def read_outcome_rows(
    path: Path, *, sessions: int = ROLLING_SESSIONS, as_of: Any = None
) -> list[dict]:
    """Stream the outcome CSV, keeping only the lately window's rows.

    STREAMED, and filtered on the way in. The live file is ~325,000 rows and
    ~300 MB; materialising it to build a 20-session score would put the whole
    year in memory to answer a question about a month. ONE pass: the window's
    bounds come from the exchange calendar (`window_bounds`), not from a first
    pass over the dates present.
    """
    target = Path(path)
    if not target.exists():
        return []
    bounds = window_bounds(sessions=sessions, as_of=as_of)
    rows: list[dict] = []
    with target.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _in_window(str(row.get("trade_date") or "").strip(), bounds):
                rows.append(dict(row))
    return rows


def d1_alignment(
    setups: Mapping[str, Mapping[str, set]] | None, session: str, symbol: str, direction: Any
) -> str:
    """`aligned` / `opposed` / `none` / `unknown` for one M5 alert.

    `unknown` when there is no snapshot, the session is not in it, or the alert
    names no side; `none` when the session is known and the symbol is absent;
    `aligned` when the alert's side is among the symbol's setup sides; else
    `opposed`. Long <-> LONG, short <-> SHORT.
    """
    if setups is None:
        return D1_UNKNOWN
    side = {"long": "LONG", "short": "SHORT"}.get(str(direction or "").strip().lower())
    if side is None:
        return D1_UNKNOWN
    by_symbol = setups.get(str(session or "").strip())
    if by_symbol is None:
        return D1_UNKNOWN
    sides = by_symbol.get(str(symbol or "").strip().upper())
    if not sides:
        return D1_NONE
    return D1_ALIGNED if side in sides else D1_OPPOSED


def d1_setups_by_session(
    rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, dict[str, set]] | None:
    """`{session: {SYMBOL: {"LONG", "SHORT"}}}` from the scanner's own tracker output.

    The caller supplies the rows; this only shapes them. Decision 0016 answer 4
    makes the D1 setup a SEGMENT DIMENSION - *"an M5 alert on a name that also
    carries a D1 setup outranks the same alert on a name that does not"* - so
    whether the name also had a swing setup that day has to travel with the
    episode, and it is read from files the scan already wrote rather than fetched.
    """
    if rows is None:
        return None  # no snapshot: UNKNOWN everywhere, never "no setup"
    wanted = {"favorite_setup", "near_favorite_zone"}
    by_session: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        bucket = str(row.get("bucket") or row.get("priority_bucket") or "").strip()
        if bucket not in wanted:
            continue
        session = str(row.get("scan_date") or row.get("session_date") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if session and symbol and side:
            by_session[session][symbol].add(side)
    return {session: dict(symbols) for session, symbols in by_session.items()}

#: The tracker dimensions this module can measure, in the AGGREGATOR'S OWN
#: SPELLING so the join is an equality rather than a hope.
#:
#: Measured against the live stores after the fix round: `bounce_type` 36 of 36
#: rows, `bounce_combo` 58 of 59, `time_bucket` 10 of 10, `market_environment`
#: 10 of 10. Before it the same four read 28/36, 0/59, 2/10 and 10/10, and the
#: three that missed were missing for SPELLING reasons - a `-` where the
#: aggregator writes `+`, an unsplit combination, and a second time-bucket
#: vocabulary - while the docs called them unanswerable.
#:
#: The five that stay blank are two different things, and saying which is the
#: point. `master_avwap_focus`, `master_avwap_priority_bucket`,
#: `master_avwap_setup_family` and `master_avwap_swing_trait` are NOT in
#: `intraday_bounce_outcomes.csv` at all - not in a column and not in
#: `context_json` - so this module cannot be asked. `rrs_alignment` IS reachable
#: (`context_json` carries `rrs_spy`) and is simply not derived here yet; it is
#: named as owed rather than filed under "cannot".
MEASURABLE_DIMENSIONS = (
    "bounce_type",
    "bounce_combo",
    "time_bucket",
    "market_environment",
)

#: Reachable from the outcome log's `context_json` and not derived yet. Kept
#: separate from "cannot be measured" because the two are different promises.
UNDERIVED_DIMENSIONS = ("rrs_alignment",)

#: The direction slot for a cell that pools both sides - R4 B4.
#:
#: `review_preference_state.json` records what the trader took and passed per
#: SEGMENT and does not carry a side within a dimension, so a "My Decisions" row
#: has no direction to join on. The pooled cell exists so that row can still show
#: the day-trade headline. It is accumulated from the EPISODES like every other
#: cell and summarised by the same `Segment.summary` - never by averaging the
#: long cell and the short cell, because a mean of trimmed means is not a
#: trimmed mean.
#:
#: `"all"` cannot collide with a real direction: the outcome log writes `long`
#: and `short`.
ALL_DIRECTIONS = "all"


def dimension_summaries(
    episodes: Iterable[Episode],
    *,
    sessions: int = ROLLING_SESSIONS,
    min_n: int | None = None,
    as_of: Any = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """`{(dimension, direction, value): summary}` - the tracker's own join key.

    The Daytrade Tracker groups by ONE dimension at a time and by side; this
    module's native cell is the four-way cross. So the marginal is built here,
    with the SAME arithmetic - `Segment.summary` - rather than by adding up the
    cross-cells, because a mean of trimmed means is not a trimmed mean.

    A dimension outside :data:`MEASURABLE_DIMENSIONS` simply has no key, and the
    caller shows a blank. Blank is the honest answer for a question the outcome
    log cannot be asked.
    """
    episodes = list(episodes)
    bounds = window_bounds(sessions=sessions, as_of=as_of)
    cells: dict[tuple[str, str, str], Segment] = {}
    for episode in episodes:
        if not _in_window(episode.trade_date, bounds):
            continue
        direction = str(episode.direction or "").strip().lower()
        # An episode counts under EVERY bounce type it carries, which is what
        # the aggregator does and what makes the two comparable; it counts once
        # under its whole combination, its time bucket and its environment.
        values: dict[str, tuple[str, ...]] = {
            "bounce_type": bounce_components(episode.bounce_type) or (UNKNOWN,),
            "bounce_combo": (bounce_combo(episode.bounce_type),),
            "time_bucket": (time_bucket(episode.entry_time),),
            "market_environment": (episode.market_environment or UNKNOWN,),
        }
        for dimension in MEASURABLE_DIMENSIONS:
            for raw in values.get(dimension, (UNKNOWN,)):
                value = str(raw or UNKNOWN)
                # Both the sided cell and the pooled one, accumulated from the
                # SAME episode rather than from each other (R4 B4).
                for slot in (direction, ALL_DIRECTIONS):
                    key = (dimension, slot, value)
                    cell = cells.get(key)
                    if cell is None:
                        # The Segment key is only used for its `summary()`
                        # labels, which the caller does not read here - the join
                        # key above is what identifies the row.
                        cell = cells[key] = Segment(key=(value, value, value, D1_UNKNOWN))
                    cell.add(episode)
    return {key: cell.summary(min_n=min_n) for key, cell in cells.items()}


def d1_setup_rows(path: Path) -> list[dict[str, str]] | None:
    """The scanner's own snapshot, reduced to what the D1 dimension needs.

    R4 A9. `d1_setup_present` had no caller anywhere: every one of the live
    segments read False, so decision 0016 answer 4 - *"an M5 alert on a name that
    also carries a D1 setup outranks the same alert on a name that does not"* -
    was a dimension in the schema and a constant in the data.

    Read from `master_avwap_tracker_scoring_snapshot.json`, which is the
    scanner's own output and carries `scan_date`, `symbol` and `priority_bucket`
    per setup. NOT from `master_avwap_setup_tracker.json`, which holds the same
    three fields and is 1.1 GB - `json.loads` on that file is one of the three
    measured causes of the 10 GB desk (2026-08-27) and it must never be read to
    answer a question a 19 MB sibling already answers.

    **Never fetched.** A study that reached for a quote is a study that cannot be
    re-run. A missing or unreadable snapshot yields None, and every episode
    then reads `d1_alignment="unknown"` (packet Q1) - not False: an absent file
    is a question that was not asked, never an answer of "no setup".
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    setups = payload.get("setups") if isinstance(payload, Mapping) else None
    if isinstance(setups, Mapping):
        entries = list(setups.values())
    elif isinstance(setups, list):
        entries = setups
    else:
        return None
    rows: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "scan_date": str(entry.get("scan_date") or ""),
                "symbol": str(entry.get("symbol") or ""),
                "side": str(entry.get("side") or ""),
                "priority_bucket": str(entry.get("priority_bucket") or ""),
            }
        )
    return rows


def load_episodes(
    *,
    outcomes_path: Path | None = None,
    setups_path: Path | None = None,
    sessions: int = ROLLING_SESSIONS,
    as_of: Any = None,
) -> list[Episode]:
    """The whole build path, in one call, so no caller re-assembles it.

    R4 A9/A10: the D1 dimension was never fed and the tracker computed its own
    version of the score. One entry point means one answer.

    Both paths default to the live stores through `project_paths`, addressed by
    their NAMED CONSTANTS - resolving a home-folder store by name under the wrong
    root shipped a blank page for six days.
    """
    from project_paths import (
        INTRADAY_BOUNCE_OUTCOMES_FILE,
        MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE,
    )

    outcomes = Path(outcomes_path or INTRADAY_BOUNCE_OUTCOMES_FILE)
    setups = Path(setups_path or MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE)
    rows = read_outcome_rows(outcomes, sessions=sessions, as_of=as_of)
    return build_episodes(
        rows,
        d1_setups_by_session=d1_setups_by_session(d1_setup_rows(setups)),
        as_of=as_of,
    )
