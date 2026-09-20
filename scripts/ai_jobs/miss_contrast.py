r"""`miss_contrast` - what the misses had in common (TJ-15 items 2-4).

Trader, 2026-09-19: *"if they said no to a bunch of stocks that went on to have
great moves that day or the next day, then I want to know about it."* TJ-11 put
those names on the Day Review page. This slot asks the next question of the same
rows - **what did they have in common?** - and answers it with arithmetic only.

Four rules it is built around:

* **Point in time.** The features are the ones the D1 scan had AT the decision:
  the LAST scan row at or before the decision's own stamp, on the decision's own
  session. A decision with no such row is counted under
  ``no_point_in_time_scan`` and contributes no feature at all - never a later
  scan's, which is a number the trader could not have been looking at.
  ``run_timestamp`` in `d1_features_history.csv` is NAIVE desk wall time
  (`master_avwap_lib/runner.py` stamps ``datetime.now()``) and an annotation's
  stamp is ZONED, so the desk zone is ATTACHED to the naive side through
  `ui.annotations.pass_bars.attach_desk_zone` and the aware side is never
  stripped.
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
    """(verdict, reason_code) for a rejection; ONE group for the likes.

    A like has no reason code - its ``reason`` is the trader's free-text note -
    so the likes pool into one group. Rejections pool by ``reason_code`` ACROSS
    vocabulary versions: a code is never reused, so a code means one thing, and
    the pack says the versions are pooled (lead decision, 2026-09-19).
    """
    name = str(verdict or "").strip().lower()
    if name in walkaway_day.LIKES:
        return ("like", "")
    return (name, str(reason or "").strip())


def _decision_rows(
    decisions: Iterable[Mapping[str, Any]] | None, window: Sequence[str]
) -> list[dict[str, Any]]:
    """The decisions that BELONG to the window, in Day Review's own mapping."""
    inside = {str(value)[:10] for value in window}
    out: list[dict[str, Any]] = []
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
        out.append({**row, "_session": session})
    return out


def _point_in_time_features(
    rows: Sequence[Mapping[str, Any]], features: Any
) -> dict[int, dict[str, Any]]:
    """``{decision index: feature mapping}`` from the LAST scan at or before it.

    One streamed pass over the history. Nothing is held but the best row found
    so far per decision, so the memory cost is the POPULATION, never the file.
    """
    if features is None:
        return {}
    wanted: dict[tuple[str, str, str], list[int]] = {}
    stamps: dict[int, datetime] = {}
    for index, row in enumerate(rows):
        moment = _aware(row.get("stamp") or row.get("created_at"))
        if moment is None:
            continue
        key = (
            str(row.get("symbol") or "").strip().upper(),
            str(row.get("side") or "").strip().upper(),
            str(row.get("_session") or "")[:10],
        )
        if not key[0] or not key[2]:
            continue
        stamps[index] = moment
        wanted.setdefault(key, []).append(index)
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

    best: dict[int, tuple[datetime, dict[str, Any]]] = {}
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
        for index in indexes:
            if ran_at > stamps[index]:
                # It did not exist when the trader clicked.
                continue
            held = best.get(index)
            if held is not None and held[0] >= ran_at:
                continue
            if mapping is None:
                mapping = _feature_mapping(scan)
            best[index] = (ran_at, mapping)
    return {index: mapping for index, (_at, mapping) in best.items()}


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------


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
    rows = _decision_rows(decisions, window)
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
        # The D1 ruler, whatever the decision was made on: the features are the
        # D1 scan's and the horizon is TJ-11's five exchange sessions. Said in
        # the pack's `ruler` field rather than assumed by a reader.
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
        mapping = features_by_index.get(index)
        if mapping is None:
            bucket["no_point_in_time_scan"] += 1
            continue
        bucket["_a" if ran else "_b"].append(mapping)

    built: list[dict[str, Any]] = []
    for key in sorted(groups):
        bucket = groups[key]
        labels = LIKE_LABELS if key[0] == "like" else VETO_LABELS
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
                "rate": cell["rate"],
                "low": cell["low"],
                "high": cell["high"],
                "reportable": cell["reportable"],
                "compared": comparison["compared"],
                "top": comparison["top"],
                "statement": comparison["statement"],
                "features": comparison["features"],
                "unmeasured_features": list(comparison["unmeasured_features"]),
                "floor_note": (
                    "" if cell["reportable"] else f"too few to call (n={bucket['measured']})"
                ),
            }
        )

    over_floor = [group for group in built if group["reportable"]]
    ranked = sorted(
        over_floor,
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
            f"observational, not causal: top {len(leaders)} of {len(over_floor)} group(s) "
            f"over the floor of {evidence_stats.MIN_REPORTABLE_N} measured decisions - "
            + ", ".join(leaders)
        )
    else:
        headline = (
            "observational, not causal: no group has reached the floor of "
            f"{evidence_stats.MIN_REPORTABLE_N} measured decisions, so no reason is "
            "named and nothing is ranked"
        )
    statement = (
        f"{headline}. {len(built)} group(s) over {len(window)} session(s) ending "
        f"{session}; a group under the floor keeps its row and its counts and is "
        "never ranked. A rate counts CLOSED horizons only; open ones are printed "
        "as pending and are in neither half."
    )

    return {
        "schema": PACK_SCHEMA,
        "session_date": session,
        "built_at": moment.isoformat(),
        "window_sessions": int(window_sessions),
        "window_first_session": window[0],
        "real_miss_rule": real_miss.REAL_MISS_V1,
        "ruler": "d1_sessions",
        "horizon_sessions": max(walkaway_day.HORIZONS),
        "decisions": len(rows),
        "pooling": (
            "reason codes are pooled ACROSS vocabulary versions: a code is never "
            "reused, so a code means one thing"
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
    rows = _decision_rows(decisions, window)

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

    reason = (
        f"contrasted {pack['decisions']} decision(s) in {len(pack['groups'])} group(s) "
        f"over {size} session(s) ending {session}"
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
