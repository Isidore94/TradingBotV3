r"""`miss_contrast` - what the misses had in common (TJ-15 items 2-4).

Trader, 2026-09-19: *"if they said no to a bunch of stocks that went on to have
great moves that day or the next day, then I want to know about it."* TJ-11 put
those names on the Day Review page. This slot asks the next question of the same
rows - **what did they have in common?** - and answers it with arithmetic only.

Six rules it is built around:

* **D1 only** (fix round, 2026-09-19). Its features are the D1 scan's and its
  ruler is the D1 swing ruler, so it judges D1 decisions and nothing else.
  Measured over the 20 sessions ending 2026-09-18 the population was 1,026 M5
  against 1,013 D1: pooling them printed one `like` rate over 320 D1 and 358 M5
  decisions, and named an all-M5 `not_today` group the night's top finding.
  Every other timeframe is COUNTED in ``excluded_by_timeframe`` and named in the
  pack's sentence; a row with no timeframe is counted separately and never read
  as D1.
* **Point in time.** The features are the ones the D1 scan had AT the decision:
  the LAST scan row for ``(symbol, side)`` at or before the decision's own
  stamp, from the decision's own session or the ONE exchange session before it
  (:data:`MAX_SCAN_AGE_SESSIONS`) - never older, never later - with the age
  carried on the row and counted in the group. A decision with no such row is
  counted under ``no_point_in_time_scan`` and contributes no feature at all.
  ``run_timestamp`` in `d1_features_history.csv` is NAIVE desk wall time
  (`master_avwap_lib/runner.py` stamps ``datetime.now()``) and an annotation's
  stamp is ZONED, so the desk zone is ATTACHED to the naive side through
  `ui.annotations.pass_bars.attach_desk_zone` and the aware side is never
  stripped.
* **Two floors, and they are different floors.** `MIN_REPORTABLE_N` gates the
  group's RATE; `evidence_contrast.MIN_CONTRAST_SIDE_N` gates a FEATURE, whose
  population is only the decisions that also carried a scan row. A group is a
  leader only with a reportable rate AND at least one feature over the feature
  floor; one with a reportable rate and no such feature keeps its row and its
  rate and says ``no feature had enough rows on both sides``.
* **Streamed, never materialised.** That file is 709 MB and 264 columns wide on
  the live desk. :func:`stream_feature_rows` is a lazy `csv` walk filtered by
  ``run_date``; nothing here builds a list of the file, and no `pandas` frame of
  it is ever constructed.
* **One rule for "did it run", and one mapping for "which session".**
  `real_miss.verdict` is CALLED through its module attribute, and which session
  a decision belongs to is `walkaway_day._row_session` - the same seam Day
  Review's own walk-away asks, so the table under the misses and the misses
  cannot disagree. A second copy of either would drift the day one is tuned.
* **It never costs the night.** A missing, locked or unreadable features file,
  an unreadable annotation store and an unreachable daily-bar store are each a
  recorded REASON and an `ok` row with a zero-group pack - never an exception
  into the runner, never a rewritten pack. Every write is temp-and-rename onto a
  superseding sibling (`ai_jobs.digest.superseding_path`), so a correction is a
  new file and the pack believed on the day survives it.

Deterministic: `uses_model=False`, no inference, seconds of work. Nothing it
writes reaches a detector, score, alert, watchlist, Focus, the review queue,
`review_policy.json` or `WISHLIST.md`; it is REPORTED evidence and a reader.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import evidence_contrast
import evidence_stats
import market_calendar
import real_miss
import walkaway_day

_log = logging.getLogger(__name__)

#: The pack's own name. A later shape is a new version, never a re-reading.
PACK_SCHEMA = "miss_contrast_v1"

#: What every pack file is called, before any superseding sibling.
PACK_PREFIX = "miss_contrast"

#: Columns of `d1_features_history.csv` that identify a scan row rather than
#: describe a name. They are never contrasted.
IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {
        "feature_history_schema_version",
        "run_id",
        "run_timestamp",
        "run_date",
        "symbol",
        "side",
    }
)

#: The two labels each half of a group carries. A veto that ran is a MISS; a
#: like that ran is exactly what the trader wanted, so the words are not shared.
VETO_LABELS = (evidence_contrast.LABEL_A, evidence_contrast.LABEL_B)
LIKE_LABELS = ("real_run", "dud")

#: How many reason groups the pack's own sentence may NAME. A bounded view, and
#: the number over the floor is stated beside it.
LEADER_LIMIT = 3

#: Calendar days either side of the window the default decision reader asks the
#: store for. A decision made after Friday's close is STAMPED on the Saturday,
#: and one made on a Sunday maps forward into Monday; `_row_session` decides
#: which session each belongs to, so the reader only has to not miss the row.
_STAMP_SLACK_DAYS = 4


# ---------------------------------------------------------------------------
# the streamed feature history
# ---------------------------------------------------------------------------


def stream_feature_rows(
    path: Any, *, sessions: Iterable[str]
) -> Iterator[dict[str, Any]]:
    """Lazily yield the rows of a `d1_features_history.csv` for ``sessions``.

    A GENERATOR on purpose. The live file is 709 MB; a reader that returned a
    list would hold about 1.4 GB of dicts to answer a question about six of
    them, in a process that also owns the night. The caller keeps what it needs
    and the rest is freed as the walk moves on.

    An unreadable file yields nothing - the slot records the reason instead.
    """
    wanted = {str(value)[:10] for value in (sessions or ()) if str(value or "").strip()}
    if not wanted:
        return
    target = Path(path)
    try:
        handle = target.open("r", encoding="utf-8-sig", newline="", errors="replace")
    except OSError as exc:
        _log.info("miss_contrast: feature history unreadable at %s: %s", target, exc)
        return
    with handle:
        try:
            for row in csv.DictReader(handle):
                if str(row.get("run_date") or "")[:10] in wanted:
                    yield row
        except (OSError, csv.Error) as exc:
            _log.info("miss_contrast: feature history stopped early: %s", exc)
            return


def _feature_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    """One scan row with its identity columns removed."""
    return {
        str(key): value
        for key, value in row.items()
        if str(key) not in IDENTITY_COLUMNS and str(key or "").strip()
    }


# ---------------------------------------------------------------------------
# stamps
# ---------------------------------------------------------------------------


def _aware(value: Any) -> datetime | None:
    """A stamp as an AWARE moment: the desk zone is attached, never stripped."""
    moment = value if isinstance(value, datetime) else None
    if moment is None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is not None:
        return moment
    # THE ONE SEAM (N1): a naive desk stamp gets the desk zone ATTACHED.
    from ui.annotations.pass_bars import attach_desk_zone

    return attach_desk_zone(moment)


def _window_sessions(session: str, count: int) -> tuple[str, ...]:
    """The ``count`` exchange sessions ENDING at ``session``, oldest first."""
    size = max(1, int(count or 1))
    earlier = walkaway_day.earlier_sessions(session, count=size - 1)
    return tuple(earlier) + (session,)


# ---------------------------------------------------------------------------
# the population
# ---------------------------------------------------------------------------


def _group_key(verdict: str, reason: str) -> tuple[str, str]:
    """``(verdict, reason_code)`` - and only a VETO has a reason code.

    A veto's ``reason`` is a code from the versioned veto vocabulary, so vetoes
    group by it, pooled ACROSS vocabulary versions: a code is never reused, so a
    code means one thing (lead decision, 2026-09-19).

    Every other verdict's ``reason`` is FREE TEXT - the trader's own note - and
    grouping on it makes one group per sentence. On the live 2026-09-18 window
    `dislike` alone split into ``"[other] rejecting 1stdev"`` and ``"... the
    1stdev"``, two groups of one, each immune to any floor because a group of
    one is never compared with anything (reviewer, 2026-09-19). So they group by
    the verdict alone and carry an empty reason code.
    """
    name = str(verdict or "").strip().lower()
    if name == "veto":
        return ("veto", str(reason or "").strip())
    return (name, "")


#: The ONE timeframe this pack judges. Its features are the D1 scan's and its
#: ruler is TJ-11's five-exchange-session swing horizon.
JUDGED_TIMEFRAME = "D1"


def _decision_rows(
    decisions: Iterable[Mapping[str, Any]] | None, window: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """``(judged rows, excluded by timeframe, rows with no timeframe)``.

    **This pack judges D1 decisions only** (fix round, 2026-09-19). The first
    build pooled every timeframe and the population was not what it was assumed
    to be: measured over the 20 sessions ending 2026-09-18 it was **1,026 M5
    against 1,013 D1** (plus 4 H1 and 2 "5M"), `like` was 320 D1 and 358 M5
    printed as ONE rate, and the pack's top-named leader - `not_today` - was
    **317 of 317 M5 decisions**, every one of them judged on a five-session
    swing horizon with D1-scan features it never had.

    Nothing is assumed away: the other timeframes are COUNTED in
    ``excluded_by_timeframe`` and the pack says the number in a sentence, and a
    row with no timeframe at all is counted separately rather than read as D1.
    """
    inside = {str(value)[:10] for value in window}
    out: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    no_timeframe = 0
    for row in decisions or ():
        if not isinstance(row, Mapping):
            continue
        verdict = str(row.get("verdict") or "").strip().lower()
        if verdict not in walkaway_day.REJECTS and verdict not in walkaway_day.LIKES:
            continue
        # The desk's ONE mapping, asked - never a second copy of the rule.
        # TJ-11F is reversing it under this call and the pack follows it there.
        session = walkaway_day._row_session(row)
        if session not in inside:
            continue
        timeframe = str(row.get("timeframe") or "").strip().upper()
        if not timeframe:
            no_timeframe += 1
            continue
        if timeframe != JUDGED_TIMEFRAME:
            excluded[timeframe] = excluded.get(timeframe, 0) + 1
            continue
        out.append({**row, "_session": session})
    return out, excluded, no_timeframe


#: How many exchange sessions back a point-in-time scan row may come from.
#:
#: ZERO would be the strictest reading of "the scan at the decision", and it is
#: what the first build did - but the scan does not run when the trader clicks.
#: Measured on the live 2026-09-18 window: scans ran at roughly 07:00-07:50,
#: 10:01, 12:45 and 13:00 desk time, and on some sessions only once, so an
#: evening or pre-market decision has no same-session scan at all and **305 of
#: 435 likes had no features** (reviewer, 2026-09-19). One session back is what
#: the trader was actually looking at in the evening or before the open. Two
#: would be a different day's picture, so it is refused.
MAX_SCAN_AGE_SESSIONS = 1


def _previous_session(session: str) -> str:
    try:
        return market_calendar.previous_session(date.fromisoformat(str(session)[:10])).isoformat()
    except (ValueError, market_calendar.SessionCalendarError):
        return ""


def _point_in_time_features(
    rows: Sequence[Mapping[str, Any]], features: Any
) -> dict[int, tuple[dict[str, Any], int]]:
    """``{decision index: (feature mapping, scan age in sessions)}``.

    The LAST scan row for ``(symbol, side)`` at or before the decision's own
    stamp, from the decision's own session or the ONE exchange session before
    it - never older, never later. The age is carried so the pack can say which
    it used rather than leaving the reader to assume.

    One streamed pass over the history; the window simply starts one session
    earlier. Nothing is held but the best row found so far per decision, so the
    memory cost is the POPULATION, never the file.
    """
    if features is None:
        return {}
    wanted: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    stamps: dict[int, datetime] = {}
    previous: dict[str, str] = {}
    for index, row in enumerate(rows):
        moment = _aware(row.get("stamp") or row.get("created_at"))
        if moment is None:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        session = str(row.get("_session") or "")[:10]
        if not symbol or not session:
            continue
        stamps[index] = moment
        if session not in previous:
            previous[session] = _previous_session(session)
        for age, run_date in enumerate((session, previous[session])):
            if not run_date or age > MAX_SCAN_AGE_SESSIONS:
                continue
            wanted.setdefault((symbol, side, run_date), []).append((index, age))
    if not wanted:
        return {}

    sessions = {key[2] for key in wanted}
    if isinstance(features, (str, Path)):
        stream: Iterable[Mapping[str, Any]] = stream_feature_rows(features, sessions=sessions)
    else:
        stream = (
            row
            for row in features
            if isinstance(row, Mapping) and str(row.get("run_date") or "")[:10] in sessions
        )

    best: dict[int, tuple[datetime, dict[str, Any], int]] = {}
    for scan in stream:
        key = (
            str(scan.get("symbol") or "").strip().upper(),
            str(scan.get("side") or "").strip().upper(),
            str(scan.get("run_date") or "")[:10],
        )
        indexes = wanted.get(key)
        if not indexes:
            continue
        ran_at = _aware(scan.get("run_timestamp"))
        if ran_at is None:
            continue
        mapping: dict[str, Any] | None = None
        for index, age in indexes:
            if ran_at > stamps[index]:
                # It did not exist when the trader clicked.
                continue
            held = best.get(index)
            # The LATEST scan at or before the stamp wins, which is also what
            # prefers a same-session row over yesterday's without a second rule.
            if held is not None and held[0] >= ran_at:
                continue
            if mapping is None:
                mapping = _feature_mapping(scan)
            best[index] = (ran_at, mapping, age)
    return {index: (mapping, age) for index, (_at, mapping, age) in best.items()}


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------


def _timeframe_sentence(excluded: Mapping[str, int], no_timeframe: int) -> str:
    """What this pack did NOT judge, said in words rather than left to a field."""
    parts = [
        f"{count} {name}"
        for name, count in sorted(excluded.items(), key=lambda pair: (-pair[1], pair[0]))
        if count
    ]
    if no_timeframe:
        parts.append(f"{no_timeframe} with no timeframe recorded")
    if not parts:
        return f"Every decision in the window was {JUDGED_TIMEFRAME}."
    return (
        f"{', '.join(parts)} decision(s) are not judged here: this pack uses "
        f"{JUDGED_TIMEFRAME}-scan features and the {JUDGED_TIMEFRAME} swing ruler."
    )


def build_pack(
    session_date: str,
    *,
    now: datetime | None = None,
    decisions: Iterable[Mapping[str, Any]] | None = None,
    daily_bars: Mapping[str, Any] | None = None,
    features: Any = None,
    window_sessions: int = evidence_stats.LATELY_SESSIONS,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """The contrast pack for one session's window. Pure: no store, no clock.

    ``features`` is a path to a file in `d1_features_history.csv` shape or an
    iterable of row mappings. ``daily_bars`` is ``{symbol: [daily bar, ...]}``
    from the durable daily store; a symbol that is absent is ``unmeasured``.
    """
    session = str(session_date or "")[:10]
    moment = now or datetime.now()
    window = _window_sessions(session, window_sessions)
    rows, excluded_by_timeframe, no_timeframe = _decision_rows(decisions, window)
    bars = daily_bars or {}
    last_session = walkaway_day._last_completed(moment)

    by_symbol: dict[str, list[tuple[date, Mapping[str, Any]]]] = {}

    def _daily(symbol: str) -> list[tuple[date, Mapping[str, Any]]]:
        if symbol not in by_symbol:
            by_symbol[symbol] = walkaway_day._completed_daily(
                bars.get(symbol) or (), last_session
            )
        return by_symbol[symbol]

    features_by_index = _point_in_time_features(rows, features)

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        verdict_name = str(row.get("verdict") or "").strip().lower()
        key = _group_key(verdict_name, row.get("reason"))
        bucket = groups.setdefault(
            key,
            {
                "verdict": key[0],
                "reason_code": key[1],
                "n": 0,
                "measured": 0,
                "pending": 0,
                "unmeasured": 0,
                "no_point_in_time_scan": 0,
                "scan_same_session": 0,
                "scan_prior_session": 0,
                "misses": 0,
                "correct": 0,
                "_a": [],
                "_b": [],
            },
        )
        bucket["n"] += 1

        try:
            decision_day = date.fromisoformat(str(row.get("_session") or "")[:10])
        except ValueError:
            decision_day = None
        if decision_day is None:
            bucket["unmeasured"] += 1
            continue

        daily = _daily(symbol)
        atr = walkaway_day._daily_atr([pair for pair in daily if pair[0] <= decision_day])
        # The D1 ruler for a D1 decision. Every other timeframe was excluded
        # above rather than pooled under it.
        reading = walkaway_day._d1_reading(
            daily, decision_day, side, atr=atr, last_session=last_session
        )
        if reading.pool == walkaway_day.POOL_PENDING:
            bucket["pending"] += 1
            continue
        if reading.pool != walkaway_day.POOL_MEASURED:
            bucket["unmeasured"] += 1
            continue
        bucket["measured"] += 1
        ran = reading.verdict == real_miss.RUN
        bucket["misses" if ran else "correct"] += 1
        joined = features_by_index.get(index)
        if joined is None:
            bucket["no_point_in_time_scan"] += 1
            continue
        mapping, age = joined
        bucket["scan_prior_session" if age else "scan_same_session"] += 1
        bucket["_a" if ran else "_b"].append(mapping)

    built: list[dict[str, Any]] = []
    for key in sorted(groups):
        bucket = groups[key]
        labels = LIKE_LABELS if key[0] in walkaway_day.LIKES else VETO_LABELS
        comparison = evidence_contrast.contrast(
            bucket.pop("_a"), bucket.pop("_b"), label_a=labels[0], label_b=labels[1]
        )
        cell = evidence_contrast.rate(
            bucket["misses"], bucket["measured"], pending=bucket["pending"]
        )
        built.append(
            {
                **bucket,
                "name": key[1] or key[0],
                "label_a": comparison["label_a"],
                "label_b": comparison["label_b"],
                # The two CONTRAST populations, as fields at this level too: the
                # group's `n` is decisions and these are the rows that actually
                # carried features, and a reader must never have to infer one
                # from the other (reviewer, 2026-09-19).
                "n_a": comparison["n_a"],
                "n_b": comparison["n_b"],
                "rate": cell["rate"],
                "low": cell["low"],
                "high": cell["high"],
                "reportable": cell["reportable"],
                "compared": comparison["compared"],
                "thin": comparison["thin"],
                "min_side": comparison["min_side"],
                "min_total": comparison["min_total"],
                "top": comparison["top"],
                "statement": comparison["statement"],
                "features": comparison["features"],
                "thin_features": comparison["thin_features"],
                "unmeasured_features": list(comparison["unmeasured_features"]),
                "floor_note": (
                    "" if cell["reportable"] else f"too few to call (n={bucket['measured']})"
                ),
                "feature_note": (
                    ""
                    if comparison["features"]
                    else "no feature had enough rows on both sides"
                ),
            }
        )

    over_floor = [group for group in built if group["reportable"]]
    # A leader needs BOTH floors: a reportable RATE and at least one feature
    # that cleared the feature floor. `sma_incoming` cleared the first and was
    # named off four rows against one (reviewer, 2026-09-19).
    eligible = [group for group in over_floor if group["features"]]
    ranked = sorted(
        eligible,
        key=lambda group: (
            -max(
                (abs(float(row["auc"]) - 0.5) for row in group["features"] if row["auc"] is not None),
                default=0.0,
            ),
            str(group["name"]),
        ),
    )
    leaders = tuple(group["name"] for group in ranked[:LEADER_LIMIT])

    if leaders:
        headline = (
            f"observational, not causal: top {len(leaders)} of {len(eligible)} group(s) "
            f"with both a reportable rate ({evidence_stats.MIN_REPORTABLE_N} measured "
            f"decisions) and at least one feature over the feature floor "
            f"({evidence_contrast.MIN_CONTRAST_SIDE_N} rows a side) - " + ", ".join(leaders)
        )
    elif over_floor:
        headline = (
            f"observational, not causal: {len(over_floor)} group(s) reached the floor of "
            f"{evidence_stats.MIN_REPORTABLE_N} measured decisions, but no feature had "
            "enough rows on both sides, so no reason is named and nothing is ranked"
        )
    else:
        headline = (
            "observational, not causal: no group has reached the floor of "
            f"{evidence_stats.MIN_REPORTABLE_N} measured decisions, so no reason is "
            "named and nothing is ranked"
        )
    statement = (
        f"{headline}. {len(built)} group(s) over {len(window)} session(s) ending "
        f"{session}; a group or a feature under its floor keeps its row and its "
        "counts and is never ranked. A rate counts CLOSED horizons only; open "
        f"ones are printed as pending and are in neither half. {_timeframe_sentence(excluded_by_timeframe, no_timeframe)}"
    )

    return {
        "schema": PACK_SCHEMA,
        "session_date": session,
        "built_at": moment.isoformat(),
        "window_sessions": int(window_sessions),
        "window_first_session": window[0],
        "real_miss_rule": real_miss.REAL_MISS_V1,
        "ruler": "d1_sessions",
        "timeframe": JUDGED_TIMEFRAME,
        "horizon_sessions": max(walkaway_day.HORIZONS),
        "decisions": len(rows),
        "excluded_by_timeframe": dict(sorted(excluded_by_timeframe.items())),
        "no_timeframe": int(no_timeframe),
        "max_scan_age_sessions": MAX_SCAN_AGE_SESSIONS,
        "feature_floor": {
            "min_side": evidence_contrast.MIN_CONTRAST_SIDE_N,
            "min_total": evidence_stats.MIN_REPORTABLE_N,
        },
        "pooling": (
            "VETO reason codes are pooled ACROSS vocabulary versions: a code is "
            "never reused, so a code means one thing. Every other verdict groups "
            "by the verdict alone - its reason is the trader's free text, not a code"
        ),
        "fundamentals": (
            "the desk's own fundamentals only - earnings dates and the pasted "
            "forecast; nothing here is fetched"
        ),
        "groups": built,
        "leaders": leaders,
        "statement": statement,
        "notes": [str(note) for note in notes if str(note or "").strip()],
    }


# ---------------------------------------------------------------------------
# publishing and reading
# ---------------------------------------------------------------------------


def _pack_dir(root: Any = None, *, create: bool = True) -> Path:
    if root is not None:
        target = Path(root)
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target
    from ai_jobs import store

    return store.digests_dir(create=create)


def pack_path(session_date: str, root: Any = None) -> Path:
    return _pack_dir(root) / f"{PACK_PREFIX}-{str(session_date)[:10]}.json"


def _publish(pack: Mapping[str, Any], root: Any = None) -> Path:
    """Write ONE pack, temp-and-rename, onto a superseding sibling (D6)."""
    from ai_jobs.digest import _publish as publish, superseding_path

    target = superseding_path(pack_path(str(pack.get("session_date") or ""), root))
    return publish(target, json.dumps(pack, indent=1, sort_keys=True, default=str) + "\n")


def read_latest(session_date: str, *, root: Any = None) -> dict[str, Any] | None:
    """The newest pack for ``session_date``, or ``None``. Never raises.

    TJ-5's Week Review table reads through here; this packet ships the reader
    and leaves the page hook to that packet (lead decision 4, 2026-09-19).
    """
    session = str(session_date or "")[:10]
    if not session:
        return None
    try:
        folder = _pack_dir(root, create=False)
        candidates = sorted(folder.glob(f"{PACK_PREFIX}-{session}*.json"))
    except (OSError, ValueError):
        return None

    def _index(path: Path) -> int:
        tail = path.stem[len(f"{PACK_PREFIX}-{session}") :].lstrip(".")
        try:
            return int(tail)
        except ValueError:
            return 0

    newest: dict[str, Any] | None = None
    best = -1
    for path in candidates:
        order = _index(path)
        if order < best:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, Mapping):
            newest, best = dict(payload), order
    return newest


# ---------------------------------------------------------------------------
# the default readers (only used when the caller supplies nothing)
# ---------------------------------------------------------------------------


def _stamped_dates(window: Sequence[str]) -> list[str]:
    """Every calendar date a decision in ``window`` could be STAMPED with.

    The window's own sessions plus the non-session days around them: an
    after-close Friday call carries the Saturday, and a Sunday call carries the
    Sunday. `_row_session` is what decides which session each row belongs to -
    this only has to not miss the row.
    """
    days = [date.fromisoformat(str(value)[:10]) for value in window if str(value or "").strip()]
    if not days:
        return []
    cursor = min(days) - timedelta(days=_STAMP_SLACK_DAYS)
    end = max(days) + timedelta(days=_STAMP_SLACK_DAYS)
    out: list[str] = []
    while cursor <= end:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _read_decisions(window: Sequence[str]) -> tuple[list[dict[str, Any]], str]:
    """Day Review's own decision rows for the window, or a reason. Never raises."""
    try:
        import daily_recap_reader

        sources = daily_recap_reader.RecapSources()
        annotations = daily_recap_reader._read_jsonl(
            "annotations", sources.annotations, "created_at"
        )
        feedback = daily_recap_reader._read_jsonl("pick_feedback", sources.pick_feedback, "ts")
        favorites = daily_recap_reader._read_jsonl(
            "swing_favorites", sources.swing_favorites, "event_at"
        )
        events = daily_recap_reader._read_jsonl("review_events", sources.review_events, "ts")
        rows: list[dict[str, Any]] = []
        for stamped in _stamped_dates(window):
            for decision in daily_recap_reader._decisions(
                stamped, annotations, feedback, favorites, events
            ):
                rows.append(
                    {
                        "session_date": stamped,
                        "symbol": decision.symbol,
                        "side": decision.side,
                        "category": decision.category,
                        "verdict": decision.verdict,
                        "source": decision.source,
                        "timeframe": decision.timeframe,
                        "stamp": getattr(decision.observed_at, "isoformat", lambda: "")(),
                        "capture_id": decision.capture_id,
                        "reason": decision.reason,
                        # LEFT EMPTY on purpose: `_row_session` maps the stamp
                        # through the desk's own rule rather than this reader
                        # writing a second answer beside it.
                        "decision_session": "",
                    }
                )
        return rows, ""
    except Exception as exc:  # noqa: BLE001 - an unreadable store is a REASON
        _log.debug("miss_contrast: decisions unreadable.", exc_info=True)
        return [], f"decisions unreadable: {type(exc).__name__}: {exc}"


def _read_daily_bars(symbols: Iterable[str]) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Durable daily bars for the named symbols, or a reason. Never raises."""
    wanted = sorted({str(value or "").strip().upper() for value in symbols if str(value or "").strip()})
    if not wanted:
        return {}, ""
    try:
        import pandas as pd

        from human_focus_tracking import MASTER_AVWAP_DAILY_BARS_DIR, _load_durable_daily_frame
    except Exception as exc:  # noqa: BLE001
        return {}, f"daily bars unreadable: {type(exc).__name__}: {exc}"

    out: dict[str, list[dict[str, Any]]] = {}
    missing = 0
    for symbol in wanted:
        try:
            frame = _load_durable_daily_frame(symbol, Path(MASTER_AVWAP_DAILY_BARS_DIR))
        except Exception:  # noqa: BLE001 - a bad file is "no bars", never a failure
            frame = None
        if frame is None or getattr(frame, "empty", True):
            missing += 1
            continue
        work = frame.rename(columns={name: str(name).strip().lower() for name in frame.columns})
        stamps = work["datetime"] if "datetime" in work.columns else work[work.columns[0]]
        bars: list[dict[str, Any]] = []
        for index, stamp in enumerate(stamps):
            day = pd.to_datetime(stamp, errors="coerce")
            if day is None or pd.isna(day):
                continue
            bar: dict[str, Any] = {"dt": day.to_pydatetime().replace(tzinfo=None).isoformat()}
            for name in ("open", "high", "low", "close"):
                if name in work.columns:
                    bar[name] = work[name].iloc[index]
            bars.append(bar)
        if bars:
            out[symbol] = bars
        else:
            missing += 1
    reason = f"{missing} symbol(s) had no durable daily bars" if missing else ""
    return out, reason


def _features_path() -> tuple[Path | None, str]:
    try:
        from project_paths import D1_FEATURES_HISTORY_FILE

        path = Path(D1_FEATURES_HISTORY_FILE)
        if not path.exists():
            return None, f"no feature history at {path}"
        return path, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"feature history unreachable: {type(exc).__name__}: {exc}"


def run_miss_contrast(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Any = None,
    decisions: Iterable[Mapping[str, Any]] | None = None,
    features: Any = None,
    daily_bars: Mapping[str, Any] | None = None,
    window_sessions: int | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The nightly slot. Deterministic, model-free, and never fails the night.

    Every input is read defensively and a missing one becomes a recorded reason
    on an `ok` row with a zero-group pack - an evidence job is never allowed to
    cost the thing it records, and a session with nothing to contrast is a fact
    about the session, not a failure of the job.
    """
    notes: list[str] = []
    session = str(session_date or "")[:10]
    moment = now or datetime.now()
    size = int(window_sessions or evidence_stats.LATELY_SESSIONS)

    # A pack keyed to nothing is worse than no pack: it published
    # `miss_contrast-.json`, one file that every later session would supersede
    # and no reader could date (reviewer, 2026-09-19). Refuse, loudly, and write
    # nothing - the runner records the failure and the next firing retries.
    try:
        date.fromisoformat(session)
    except ValueError:
        return {
            "status": "failed",
            "model": "",
            "reason": (
                f"refusing to build a contrast pack for session_date "
                f"{session_date!r}: a pack must be keyed to a readable date"
            ),
            "outputs": [],
        }

    try:
        window = _window_sessions(session, size)
    except (ValueError, market_calendar.SessionCalendarError) as exc:
        return {
            "status": "ok",
            "model": "",
            "reason": f"no contrast: the session window could not be built ({exc})",
            "outputs": [],
        }

    if decisions is None:
        decisions, note = _read_decisions(window)
        if note:
            notes.append(note)
    rows, _excluded, _no_timeframe = _decision_rows(decisions, window)

    if features is None:
        path, note = _features_path()
        if note:
            notes.append(note)
        features = path

    if daily_bars is None:
        daily_bars, note = _read_daily_bars(
            str(row.get("symbol") or "") for row in rows
        )
        if note:
            notes.append(note)

    try:
        pack = build_pack(
            session,
            now=moment,
            decisions=decisions,
            daily_bars=daily_bars,
            features=features,
            window_sessions=size,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 - arithmetic must not cost the night
        _log.exception("miss_contrast: the pack could not be built.")
        return {
            "status": "ok",
            "model": "",
            "reason": f"no contrast: {type(exc).__name__}: {exc}",
            "outputs": [],
        }

    try:
        written = _publish(pack, root)
    except OSError as exc:
        return {
            "status": "failed",
            "model": "",
            "reason": f"the contrast pack could not be published: {exc}",
            "outputs": [],
        }

    skipped = sum(pack["excluded_by_timeframe"].values()) + pack["no_timeframe"]
    reason = (
        f"contrasted {pack['decisions']} {JUDGED_TIMEFRAME} decision(s) in "
        f"{len(pack['groups'])} group(s) over {size} session(s) ending {session}; "
        f"{skipped} decision(s) on other timeframes not judged"
    )
    if notes:
        reason = f"{reason}; {'; '.join(notes)}"
    return {
        "status": "ok",
        "model": "",
        "reason": reason,
        "outputs": [str(written)],
        "extra": {"groups": len(pack["groups"]), "leaders": list(pack["leaders"])},
    }


__all__ = [
    "IDENTITY_COLUMNS",
    "PACK_SCHEMA",
    "build_pack",
    "pack_path",
    "read_latest",
    "run_miss_contrast",
    "stream_feature_rows",
]
