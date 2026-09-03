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

#: The rolling window the score is measured over. Decision 0016 answer 6:
#: "lately" is about 20 sessions and carries NO regime label.
#:
#: V3 item 3: ONE CONSTANT, and it lives in `evidence_stats` with the rest of the
#: desk's statistics contract. This name is kept as an alias so nothing that
#: imports it breaks, but the NUMBER is no longer written here - two modules that
#: each own a "lately" eventually disagree about the trader's own word.
from evidence_stats import LATELY_SESSIONS as ROLLING_SESSIONS  # noqa: E402

#: Time buckets, market-local. The open and the last hour behave differently
#: enough from the middle of the day that pooling them hides both.
TIME_BUCKETS = ("open_30m", "morning", "midday", "power_hour")

#: The market environments the alert context already records. Passed through
#: verbatim - this module never re-derives a regime, and decision 0016 answer 6
#: says "lately" needs no regime label of its own.
UNKNOWN = "unknown"


def time_bucket(entry_time: Any) -> str:
    """Which part of the session an entry sits in. `unknown` when unreadable.

    Boundaries are the trader's own working day: the first half hour, then the
    morning to 11:30, the middle to 15:00, and the last hour. A row whose time
    cannot be read is bucketed `unknown` rather than dropped - it still counts
    toward what was NOT measurable, which is the honest denominator.
    """
    stamp = _as_datetime(entry_time)
    if stamp is None:
        return UNKNOWN
    minutes = stamp.hour * 60 + stamp.minute
    if minutes < 10 * 60:
        return "open_30m"
    if minutes < 11 * 60 + 30:
        return "morning"
    if minutes < 15 * 60:
        return "midday"
    return "power_hour"


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
    d1_setup_present: bool = False
    #: Whether the stop was hit inside `HELD_WINDOW_MINUTES` of the entry.
    broke_early: bool = False
    #: The best MFE seen on any followed row for this event.
    mfe_r: float | None = None

    @property
    def held(self) -> bool:
        return not self.broke_early

    def segment(self) -> tuple[str, str, str, bool]:
        return (
            self.bounce_type or UNKNOWN,
            time_bucket(self.entry_time),
            self.market_environment or UNKNOWN,
            bool(self.d1_setup_present),
        )


@dataclass
class Segment:
    """One (bounce_type, time_bucket, environment, d1_setup) cell."""

    key: tuple[str, str, str, bool]
    episodes: int = 0
    held: int = 0
    mfe_of_held: list[float] = field(default_factory=list)

    @property
    def hold_rate(self) -> float | None:
        return (self.held / self.episodes) if self.episodes else None

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
            "d1_setup_present": d1,
            "n": self.episodes,
            "n_held": self.held,
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
    d1_setups_by_session: Mapping[str, set] | None = None,
) -> list[Episode]:
    """Fold the outcome log's many rows per event into one episode each.

    The log carries a `registered` row, a stream of `update`s, the milestone
    rows and a `final` for every alert. What this needs from them is narrow:
    the identity and context from whichever row carries it, whether the stop was
    hit inside the first thirty minutes, and the best MFE reached at any point.

    `d1_setups_by_session` is `{trade_date: {SYMBOL, ...}}` read from the
    scanner's own output files by the caller. **Never fetched** - a study that
    reached for a quote would be a study that could not be re-run.
    """
    from setup_scoreboard import bounce_type_from_event_id

    setups = d1_setups_by_session or {}
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
                d1_setup_present=symbol in (setups.get(trade_date) or set()),
            )
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

        if _as_bool(row.get("stop_hit")) and _within_hold_window(episode, row):
            episode.broke_early = True
    return list(episodes.values())


def _environment_of(row: Mapping[str, Any]) -> str:
    try:
        context = json.loads(str(row.get("context_json") or "") or "{}")
    except ValueError:
        return ""
    return str((context or {}).get("market_environment") or "").strip()


def _within_hold_window(episode: Episode, row: Mapping[str, Any]) -> bool:
    """Whether this row's stop hit happened inside the hold window.

    `minutes_elapsed` when the row carries one, else the gap between the row's
    own timestamp and the entry. A row that says neither is treated as INSIDE
    the window: a stop we cannot place is a stop, and calling it late would
    quietly improve every hold rate on the board.
    """
    minutes = _as_float(row.get("minutes_elapsed"))
    if minutes is not None:
        return minutes <= HELD_WINDOW_MINUTES
    entry = _as_datetime(episode.entry_time)
    stamp = _as_datetime(row.get("logged_at"))
    if entry is None or stamp is None:
        return True
    return stamp - entry <= timedelta(minutes=HELD_WINDOW_MINUTES)


def recent_sessions(episodes: Iterable[Episode], *, sessions: int = ROLLING_SESSIONS) -> set:
    """The last `sessions` trade dates present in the data, newest first."""
    dates = sorted({episode.trade_date for episode in episodes if episode.trade_date})
    return set(dates[-sessions:]) if dates else set()


def build_segments(
    episodes: Iterable[Episode],
    *,
    sessions: int = ROLLING_SESSIONS,
    min_n: int | None = None,
) -> list[dict[str, Any]]:
    """Every segment, ranked by score, with the unmeasurable ones still listed.

    A cell under the floor is REPORTED with its n and its discovery label rather
    than dropped: the trader needs to know a segment exists and is thin, which is
    a different fact from a segment that has never fired.
    """
    episodes = list(episodes)
    wanted = recent_sessions(episodes, sessions=sessions)
    cells: dict[tuple, Segment] = {}
    for episode in episodes:
        if wanted and episode.trade_date not in wanted:
            continue
        key = episode.segment()
        cell = cells.get(key)
        if cell is None:
            cell = cells[key] = Segment(key=key)
        cell.episodes += 1
        if episode.held:
            cell.held += 1
            if episode.mfe_r is not None:
                cell.mfe_of_held.append(episode.mfe_r)
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
            bool(cell.get("d1_setup_present")),
        ): cell
        for cell in summaries
    }


def alert_cell(
    index: Mapping[tuple, Mapping[str, Any]] | None,
    *,
    bounce_type: Any,
    entry_time: Any,
    market_environment: Any = UNKNOWN,
    d1_setup_present: Any = False,
) -> Mapping[str, Any] | None:
    """One alert's cell out of `segment_index`, or None.

    `segment_index` said it existed "for a per-alert lookup" and no caller ever
    built the key, so the alert row had nothing to read (R4 A10). The key is
    built HERE rather than at each call site: four positional strings that must
    agree with `Episode.segment()` is exactly the sort of thing that drifts.
    """
    if not index:
        return None
    return index.get(
        (
            str(bounce_type or UNKNOWN).strip() or UNKNOWN,
            time_bucket(entry_time),
            str(market_environment or UNKNOWN).strip() or UNKNOWN,
            bool(d1_setup_present),
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


def read_outcome_rows(path: Path, *, sessions: int = ROLLING_SESSIONS) -> list[dict]:
    """Stream the outcome CSV, keeping only the recent sessions' rows.

    STREAMED, and filtered on the way in. The live file is 307,908 rows and
    ~90 MB; materialising it to build a 20-session score would put the whole
    year in memory to answer a question about a month.

    A two-pass read is deliberate: the first pass learns which dates exist so
    "the last 20 sessions" is measured rather than assumed from a calendar, and
    the second keeps only those rows.
    """
    target = Path(path)
    if not target.exists():
        return []
    dates: set[str] = set()
    with target.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = str(row.get("trade_date") or "").strip()
            if stamp:
                dates.add(stamp)
    wanted = set(sorted(dates)[-sessions:])
    if not wanted:
        return []
    rows: list[dict] = []
    with target.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("trade_date") or "").strip() in wanted:
                rows.append(dict(row))
    return rows


def d1_setups_by_session(rows: Iterable[Mapping[str, Any]]) -> dict[str, set]:
    """`{session: {SYMBOL}}` from the scanner's own tracker output.

    The caller supplies the rows; this only shapes them. Decision 0016 answer 4
    makes the D1 setup a SEGMENT DIMENSION - *"an M5 alert on a name that also
    carries a D1 setup outranks the same alert on a name that does not"* - so
    whether the name also had a swing setup that day has to travel with the
    episode, and it is read from files the scan already wrote rather than fetched.
    """
    wanted = {"favorite_setup", "near_favorite_zone"}
    by_session: dict[str, set] = defaultdict(set)
    for row in rows:
        bucket = str(row.get("bucket") or row.get("priority_bucket") or "").strip()
        if bucket not in wanted:
            continue
        session = str(row.get("scan_date") or row.get("session_date") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if session and symbol:
            by_session[session].add(symbol)
    return dict(by_session)

#: The tracker dimensions this module can measure, and the only ones. They are
#: the three the OUTCOME LOG itself carries; every other tab on the Daytrade
#: Tracker (combos, RRS, the four Swing ones) is derived from alert context that
#: `intraday_bounce_outcomes.csv` does not record, so this module answers BLANK
#: for them rather than a number computed some other way. A second formula under
#: the same column heading is the failure R4 A10 removed.
MEASURABLE_DIMENSIONS = ("bounce_type", "time_bucket", "market_environment")


def dimension_summaries(
    episodes: Iterable[Episode],
    *,
    sessions: int = ROLLING_SESSIONS,
    min_n: int | None = None,
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
    wanted = recent_sessions(episodes, sessions=sessions)
    cells: dict[tuple[str, str, str], Segment] = {}
    for episode in episodes:
        if wanted and episode.trade_date not in wanted:
            continue
        direction = str(episode.direction or "").strip().lower()
        values = {
            "bounce_type": episode.bounce_type or UNKNOWN,
            "time_bucket": time_bucket(episode.entry_time),
            "market_environment": episode.market_environment or UNKNOWN,
        }
        for dimension in MEASURABLE_DIMENSIONS:
            value = str(values.get(dimension) or UNKNOWN)
            key = (dimension, direction, value)
            cell = cells.get(key)
            if cell is None:
                # The Segment key is only used for its `summary()` labels, which
                # the caller does not read here - the join key above is what
                # identifies the row.
                cell = cells[key] = Segment(key=(value, value, value, False))
            cell.episodes += 1
            if episode.held:
                cell.held += 1
                if episode.mfe_r is not None:
                    cell.mfe_of_held.append(episode.mfe_r)
    return {key: cell.summary(min_n=min_n) for key, cell in cells.items()}


def d1_setup_rows(path: Path) -> list[dict[str, str]]:
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
    re-run. A missing or unreadable snapshot yields no rows, and every episode
    then reads `d1_setup_present=False` - which is what happened before this
    existed, so an absent file degrades to the old behaviour rather than to an
    error.
    """
    target = Path(path)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    setups = payload.get("setups") if isinstance(payload, Mapping) else None
    if isinstance(setups, Mapping):
        entries = list(setups.values())
    elif isinstance(setups, list):
        entries = setups
    else:
        return []
    rows: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "scan_date": str(entry.get("scan_date") or ""),
                "symbol": str(entry.get("symbol") or ""),
                "priority_bucket": str(entry.get("priority_bucket") or ""),
            }
        )
    return rows


def load_episodes(
    *,
    outcomes_path: Path | None = None,
    setups_path: Path | None = None,
    sessions: int = ROLLING_SESSIONS,
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
    rows = read_outcome_rows(outcomes, sessions=sessions)
    return build_episodes(
        rows, d1_setups_by_session=d1_setups_by_session(d1_setup_rows(setups))
    )
