"""The one owner of the Working-lately snapshot and its leader-change events.

Packet ST6.3, 2026-09-06. Four surfaces on this desk answered "what is working"
- the Setup Tracker banner, Weekend Prep's verdict card, the AWAY Recap and now
the strip above the M5 list - and until this service nothing tied their answers
to ONE reading. Two of them could name different families on the same afternoon
and there was no way to tell which was older.

**Everything expensive is on the worker.** `build_payload` runs the three reads
(`read_recent_rows`, `read_favorable_read`, `read_held_run_summaries`), builds
the snapshot, writes it and appends the events. `_on_payload_ready` is the GUI
slot and it does exactly one thing: emit `snapshotChanged`. A test monkeypatches
the three readers to RAISE and drives the slot, so "the slot does not read" is
proven rather than promised.

**Four triggers, one reaction.** Once after the window shows, on the day roll,
after every persisted tracker export, and on a thirty-minute timer - all four
route through ONE `SignalCoalescer` (`ui.timer_utils`, 200 ms leading edge), so
a day roll that lands beside an export is one build, not two. A build already in
flight is not queued behind itself: `refresh` is single-flight.

**A restart replays nothing.** Every event is deduplicated on
`(kind, prior_leader, new_leader, new_snapshot_id)` against the file itself, so
reopening the desk on the same evidence appends zero lines. The snapshot is
written temp-and-rename; a half-written reading is worse than yesterday's.

Nothing here reaches a detector, a score, a rank that gates, an alert, a
watchlist, Focus, the review queue or `review_policy.json`. It decides what four
screens SAY.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import QObject, QThread, QTimer, Signal

import setup_grades_history
import working_lately
from working_lately import EvidenceSnapshot, build_snapshot, leader_name

#: How often the desk re-reads the three sources on its own. Half an hour is
#: slower than anything that can change inside a session (the tracker writes at
#: the close slot; the outcome log grows all day) and fast enough that a
#: leader change is on screen before the trader next looks for it.
REFRESH_INTERVAL_MS = 30 * 60 * 1000

#: The setup grades (PROVEN/A/B/C/D/New) written beside the snapshot. Kept OUT
#: of the snapshot itself so its id and size cap are untouched.
GRADES_FILE_NAME = "setup_grades_latest.json"

#: The event file's schema, named so a later shape cannot be read as this one.
EVENT_SCHEMA = "working_lately_leader_change_v1"


def default_store_dir() -> Path:
    """`%LOCALAPPDATA%\\TradingBotV3\\working_lately`. Per-machine, small.

    A per-MACHINE store on purpose: the snapshot is a reading of this desk's
    own files at this desk's own clock, and the DAS is the durable tier for
    evidence, not for a display cache.
    """
    from project_paths import LOCAL_SETTINGS_DIR

    return Path(LOCAL_SETTINGS_DIR) / "working_lately"


# ---------------------------------------------------------------------------
# the three reads - module level so a test can replace them by name
# ---------------------------------------------------------------------------


def _recent_stats_path() -> Path:
    from project_paths import MASTER_AVWAP_SETUP_STATS_FILE

    return Path(MASTER_AVWAP_SETUP_STATS_FILE).with_name(
        "master_avwap_setup_type_recent_stats.csv"
    )


def read_recent_rows() -> list[dict[str, str]]:
    """ST2.1's recent setup-type rows, straight off the CSV.

    A missing export is ZERO ROWS, not a raised reader: the tracker writes it at
    the close slot and the desk starts before that every morning.
    """
    path = _recent_stats_path()
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def read_favorable_read() -> Any:
    """ST1's ONE eligible-row read of the tier outcomes file, or None.

    None means the file could not be read at all, which the snapshot renders as
    `rows is None` and the verdict as `source unavailable`. A zero would be an
    answer; this is a question that was not asked.
    """
    from project_paths import MASTER_AVWAP_TIER_OUTCOMES_FILE
    import swing_evidence

    path = Path(MASTER_AVWAP_TIER_OUTCOMES_FILE)
    if not path.is_file():
        return None
    try:
        return swing_evidence.read_eligible_rows(path, swing_evidence.POLICY_SCANROW_V1)
    except Exception:  # noqa: BLE001 - an unreadable source is absent, not fatal
        logging.debug("Working-lately favorable read failed", exc_info=True)
        return None


#: The outcome window streamed ONCE per build and shared by the two readers
#: that need it (held x ran and the setup grades). Set and cleared by
#: `build_payload` on the worker; never held between builds - the window is
#: hundreds of MB of dicts.
_OUTCOME_ROWS_THIS_BUILD: list[dict] | None = None


def _outcome_rows() -> list[dict]:
    global _OUTCOME_ROWS_THIS_BUILD
    if _OUTCOME_ROWS_THIS_BUILD is not None:
        return _OUTCOME_ROWS_THIS_BUILD
    import held_run_score
    from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE

    return held_run_score.read_outcome_rows(Path(INTRADAY_BOUNCE_OUTCOMES_FILE))


def read_held_run_summaries() -> Any:
    """`held_run_score.dimension_summaries` over the rolling window, or None."""
    try:
        import held_run_score

        episodes = held_run_score.load_episodes(rows=_outcome_rows())
        if not episodes:
            return None
        return held_run_score.dimension_summaries(episodes)
    except Exception:  # noqa: BLE001 - the day-trade half is absent, not fatal
        logging.debug("Working-lately held-run read failed", exc_info=True)
        return None


def read_setup_grades(recent_rows: Any) -> dict[str, Any] | None:
    """`setup_grades.build_payload` over the tracker rows and the outcome window.

    The swing cells also get their wins vs SPY and cum R (`read_swing_tape`),
    over the tracker's own window. None when it could not be built; the
    surfaces then keep arrival order.
    """
    try:
        import setup_grades

        as_of = _last_completed_session().isoformat()
        try:
            swing_tape = read_swing_tape(_tracker_reference(recent_rows), as_of=as_of)
        except Exception:  # noqa: BLE001 - tape and cum R are then unknown
            logging.warning("Setup grades tape read failed", exc_info=True)
            swing_tape = None
        return setup_grades.build_payload(
            recent_rows=recent_rows or (),
            outcome_rows=_outcome_rows(),
            as_of=as_of,
            swing_tape=swing_tape,
        )
    except Exception:  # noqa: BLE001 - grades are presentation, never fatal
        logging.debug("Setup grades build failed", exc_info=True)
        return None


def _horizon_outcomes_path() -> Path:
    from project_paths import MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE

    return Path(MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE)


def _spy_bars_path() -> Path:
    from project_paths import MASTER_AVWAP_DAILY_BARS_DIR

    return Path(MASTER_AVWAP_DAILY_BARS_DIR) / "SPY.parquet"


#: The horizon-file columns the tape join reads; the rest are dropped on read.
_HORIZON_COLUMNS = (
    "symbol", "side", "scan_date", "target_session", "horizon_sessions",
    "side_return_pct", "measured", "maturity", "outcome_kind",
)


def _horizon_index() -> dict:
    """The 5-session horizon rows by `(SYMBOL, SIDE, scan_date)`, `{}` when unreadable."""
    import setup_grades

    path = _horizon_outcomes_path()

    def build() -> dict:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                return setup_grades.horizon_index(
                    {name: row.get(name) for name in _HORIZON_COLUMNS}
                    for row in csv.DictReader(handle)
                )
        except OSError:
            return {}

    return _cached("horizon_index", _file_key(path), build, keep=path.is_file())


def read_spy_closes() -> dict[str, float]:
    """SPY's daily closes from the durable bar store, `{iso date: close}`."""
    path = _spy_bars_path()

    def build() -> dict[str, float]:
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - no SPY is unknown, never a guess
            return {}
        column = next((c for c in ("datetime", "date") if c in frame.columns), None)
        if column is None or "close" not in frame.columns:
            return {}
        days = pd.to_datetime(frame[column], errors="coerce")
        closes = pd.to_numeric(frame["close"], errors="coerce")
        return {
            day.date().isoformat(): float(close)
            for day, close in zip(days, closes, strict=False)
            if not pd.isna(day) and not pd.isna(close)
        }

    return _cached("spy_closes", _file_key(path), build, keep=path.is_file())


def _swing_tape_for(setups: Mapping[str, Any], reference: date, *, as_of: str) -> dict[str, Any]:
    """`setup_grades.swing_tape_stats` over the tracker's window ending `reference`.

    The window is the tracker's own (scan dates 0..lookback calendar days
    back), and the picks are its selected episodes, so the tape cell describes
    the same picks as the plain one.
    """
    import looking_back
    import setup_grades
    from master_avwap_lib import legacy

    start = reference - timedelta(days=int(legacy.RECENT_SETUP_TYPE_LOOKBACK_DAYS))
    in_window = {
        key: setup
        for key, setup in (setups or {}).items()
        if isinstance(setup, Mapping)
        and start.isoformat() <= str(setup.get("scan_date") or "")[:10] <= reference.isoformat()
    }
    picks = looking_back.swing_pick_results(in_window, reference=reference)
    return setup_grades.swing_tape_stats(picks, _horizon_index(), read_spy_closes(), as_of=as_of)


def _tape_inputs_key() -> tuple:
    return (
        _file_key(_scoring_snapshot_path()),
        _file_key(_horizon_outcomes_path()),
        _file_key(_spy_bars_path()),
    )


def read_swing_tape(reference: date, *, as_of: str) -> dict[str, Any] | None:
    """Per swing key: wins vs SPY and cum R over the tracker window, or None.

    None when the tracker's scoring snapshot is unreadable: every swing cell's
    tape and cum R are then unknown. THE WORKER SIDE; cached on its inputs.
    """

    def build() -> dict[str, Any] | None:
        setups = _build_setups()
        return _swing_tape_for(setups, reference, as_of=as_of) if setups else None

    return _cached("swing_tape", (_tape_inputs_key(), reference, as_of), build)


#: The looking-back reading (P2-9): pick equity curves and the hold-out window.
#: Its own file, so the snapshot and the grades files stay byte-identical.
LOOKING_BACK_FILE_NAME = "looking_back_latest.json"

#: Results that cannot change inside a session, keyed by what they were read
#: from: `{name: (key, value)}`. Small derived values only, never the raw rows.
_LOOKING_BACK_CACHE: dict[str, tuple[Any, Any]] = {}


def _cached(name: str, key: Any, build, *, keep: bool = True) -> Any:
    """`build()` once per `key`. `keep=False` never stores (e.g. a missing file)."""
    hit = _LOOKING_BACK_CACHE.get(name)
    if hit is not None and hit[0] == key:
        return hit[1]
    value = build()
    if keep:
        _LOOKING_BACK_CACHE[name] = (key, value)
    else:
        _LOOKING_BACK_CACHE.pop(name, None)
    return value


def _file_key(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0)
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _scoring_snapshot_path() -> Path:
    from project_paths import MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE

    return Path(MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE)


def _scoring_setups() -> dict[str, Any]:
    """The tracker's compact scoring snapshot `setups`, `{}` when unreadable."""
    try:
        payload = json.loads(_scoring_snapshot_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    setups = payload.get("setups") if isinstance(payload, dict) else None
    return setups if isinstance(setups, dict) else {}


#: The scoring snapshot parsed ONCE per build and shared by the tape read and
#: the hold-out read: None outside a build, `[]` inside one before the first
#: read, `[setups]` after it. Cleared by `build_payload`; never kept between builds.
_SETUPS_THIS_BUILD: list | None = None


def _build_setups() -> dict[str, Any]:
    """`_scoring_setups()`, parsed at most once per `build_payload`."""
    holder = _SETUPS_THIS_BUILD
    if holder is None:
        return _scoring_setups()
    if not holder:
        holder.append(_scoring_setups())
    return holder[0]


def _outcome_log_path() -> Path:
    from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE

    return Path(INTRADAY_BOUNCE_OUTCOMES_FILE)


def _stream_outcome_rows(window: tuple[str, str]) -> list[dict]:
    """The outcome log's rows inside `window`, streamed in one pass."""
    path = _outcome_log_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = str(row.get("trade_date") or "").strip()
            if stamp and window[0] <= stamp <= window[1]:
                rows.append(dict(row))
    return rows


def _prior_m5(prior: tuple[str, str]) -> dict[str, Any]:
    """The prior window's M5 results, grades and held x ran cells.

    The same builders as the live cells, over the prior dates only. Cached on
    the window and the outcome log's mtime and size; a missing log is never
    cached, so the first build after it appears reads it.
    """
    import looking_back
    import setup_grades

    def build() -> dict[str, Any]:
        rows = _stream_outcome_rows(prior)
        held: list[Any] = []
        try:
            import held_run_score

            episodes = held_run_score.load_episodes(rows=rows, as_of=prior[1])
            if episodes:
                held = working_lately.daytrade_held_run_cells(
                    held_run_score.dimension_summaries(episodes, as_of=prior[1])
                )
        except Exception:  # noqa: BLE001 - the held x ran column is absent, not fatal
            logging.warning("Looking-back prior held-run read failed", exc_info=True)
        return {
            "results": looking_back.m5_alert_results(rows),
            "grades": setup_grades.daytrade_cells(setup_grades.bracket_results(rows)),
            "held": held,
        }

    path = _outcome_log_path()
    return _cached("prior_m5", (tuple(prior), _file_key(path)), build, keep=path.is_file())


def _tracker_reference(recent_rows: Any) -> date:
    """The day the tracker's recent family rows were built for."""
    stamps = []
    for row in recent_rows or ():
        text = str((row or {}).get("tracker_saved_at") or "")[:10]
        try:
            stamps.append(date.fromisoformat(text))
        except ValueError:
            continue
    return max(stamps) if stamps else _last_completed_session()


def _swing(recent_reference: date) -> dict[str, Any]:
    """Swing picks for the curve, and the tracker's family rows for the prior window.

    One read of the compact scoring snapshot, cached on its mtime. The prior
    rows are the tracker's own `build_recent_tracker_setup_family_rows` at an
    earlier reference date: the same statistic, split by date.
    """
    import looking_back
    import setup_grades
    from master_avwap_lib import legacy

    lookback = int(legacy.RECENT_SETUP_TYPE_LOOKBACK_DAYS)
    prior_reference = looking_back.tracker_prior_reference(recent_reference, lookback)

    as_of = _last_completed_session().isoformat()

    def build() -> dict[str, Any]:
        setups = _build_setups()
        prior_tape = _swing_tape_for(setups, prior_reference, as_of=as_of) if setups else None
        prior_rows = legacy.build_recent_tracker_setup_family_rows(
            setups,
            reference_date=prior_reference,
            lookback_days=lookback,
            current_regime_label=None,
        ) if setups else []
        for row in prior_rows:
            row["namespace"] = "live"
        return {
            "picks": looking_back.swing_pick_results(setups),
            "grades": setup_grades.swing_cells(prior_rows, prior_tape),
            "trade_r": working_lately.bucketed_trade_r_cells(prior_rows),
            "windows": {
                "recent": [
                    (recent_reference - timedelta(days=lookback)).isoformat(),
                    recent_reference.isoformat(),
                ],
                "prior": [
                    (prior_reference - timedelta(days=lookback)).isoformat(),
                    prior_reference.isoformat(),
                ],
            },
        }

    return _cached("swing", (_tape_inputs_key(), prior_reference, as_of), build)


def _prior_favorable() -> dict[str, Any]:
    """The favorable-direction cells over the prior window, cached per window."""
    import looking_back
    import swing_evidence
    from project_paths import MASTER_AVWAP_TIER_OUTCOMES_FILE

    policy = swing_evidence.POLICY_SCANROW_V1
    windows = looking_back.split_windows(sessions=int(policy.window_sessions))
    path = Path(MASTER_AVWAP_TIER_OUTCOMES_FILE)

    def build() -> dict[str, Any]:
        cells: list[Any] = []
        if path.is_file():
            read = swing_evidence.read_eligible_rows(path, policy, window=windows["prior"])
            cells = working_lately.swing_favorable_cells(read)
        return {"cells": cells, "windows": {k: list(v) for k, v in windows.items()}}

    return _cached("prior_favorable", (_file_key(path), windows["prior"]), build)


#: Which namespaces each kind has a prior-window source for. The tracker's
#: compact snapshot holds live setups only, so a study cell has no prior.
PRIOR_SOURCES = {
    "swing_trade_r": ("live",),
    "swing_favorable": ("live",),
    "daytrade_held_run": ("live",),
}


def recent_m5_results() -> list[dict[str, Any]]:
    """This build's recent-window M5 results, taken while its rows are held."""
    import looking_back

    window = looking_back.split_windows()["recent"]
    return looking_back.m5_alert_results(
        row
        for row in (_outcome_rows() or ())
        if looking_back.in_window(row.get("trade_date"), window)
    )


def read_looking_back(
    *,
    recent_rows: Any = (),
    snapshot: Any = None,
    grades: Mapping[str, Any] | None = None,
    recent_m5: Any = None,
) -> dict[str, Any] | None:
    """Pick equity curves and the hold-out columns, or None when not built.

    THE WORKER SIDE. `recent_m5` is this build's recent window, taken before
    its rows were freed (`recent_m5_results`); every prior-window read is
    cached. The recent side of each hold-out row is the value this build
    already published: the same builders over the same rows.
    """
    try:
        import looking_back
        import setup_grades

        windows = looking_back.split_windows()
        swing = _swing(_tracker_reference(recent_rows))
        if recent_m5 is None:
            recent_m5 = recent_m5_results()
        prior_m5 = _prior_m5(windows["prior"])
        favorable = _prior_favorable()
        cells = list(getattr(snapshot, "cells", ()) or ())
        graded = grades or {}

        def of_kind(*kinds: str) -> list[Any]:
            return [cell for cell in cells if cell.kind in kinds]

        holdout = {
            "swing": {
                "windows": swing["windows"],
                "favorable_windows": favorable["windows"],
                "grades": setup_grades.holdout_view(graded.get("swing") or (), swing["grades"]),
                # Trade-R cells come with their row's bucket: the snapshot cell
                # carries none, and two buckets of a family are two populations.
                "working_lately": working_lately.holdout_view(
                    working_lately.bucketed_trade_r_cells(recent_rows)
                    + of_kind("swing_favorable"),
                    list(swing["trade_r"]) + list(favorable["cells"]),
                    prior_sources=PRIOR_SOURCES,
                ),
            },
            "day": {
                "windows": {k: list(v) for k, v in windows.items()},
                "grades": setup_grades.holdout_view(graded.get("daytrade") or (), prior_m5["grades"]),
                "working_lately": working_lately.holdout_view(
                    of_kind("daytrade_held_run"), prior_m5["held"], prior_sources=PRIOR_SOURCES
                ),
            },
        }
        return looking_back.build_payload(
            swing_results=swing["picks"],
            m5_results=list(prior_m5["results"]) + list(recent_m5),
            holdout=holdout,
            as_of=_last_completed_session().isoformat(),
        )
    except Exception:  # noqa: BLE001 - a display reading, never fatal
        logging.warning("Looking-back build failed", exc_info=True)
        return None


def read_persisted_looking_back(store_dir: Any = None) -> dict[str, Any]:
    """The last published looking-back reading, `{}` when absent."""
    path = Path(store_dir) if store_dir is not None else default_store_dir()
    try:
        payload = json.loads((path / LOOKING_BACK_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_persisted_grades(store_dir: Any = None) -> dict[str, Any]:
    """The last published grades, `{}` when absent."""
    path = Path(store_dir) if store_dir is not None else default_store_dir()
    try:
        payload = json.loads((path / GRADES_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_persisted_snapshot(store_dir: Any = None) -> dict[str, Any]:
    """The last published snapshot, straight off the file. `{}` when absent.

    A module function on purpose: the AWAY Recap builds on a worker thread and
    needs the reading, not the service. Constructing a `QObject` with a QTimer
    child off the GUI thread to read two files would be a Qt object with no
    owner and no event loop, which is a crash waiting for a slow day.
    """
    path = Path(store_dir) if store_dir is not None else default_store_dir()
    try:
        payload = json.loads((path / "snapshot_latest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_persisted_events(session: str = "", store_dir: Any = None) -> list[dict[str, Any]]:
    """The leader-change events, optionally narrowed to ONE session's `as_of`."""
    path = Path(store_dir) if store_dir is not None else default_store_dir()
    wanted = str(session or "")[:10]
    out: list[dict[str, Any]] = []
    try:
        text = (path / "leader_change_events.jsonl").read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if wanted and str(row.get("as_of") or "")[:10] != wanted:
            continue
        out.append(row)
    return out


def _last_completed_session() -> date:
    """The session the snapshot is ABOUT. Today when the calendar refuses.

    Falling back to today only ever makes the freshness test stricter, never
    looser, so an unreadable calendar can withhold a leader and can never invent
    one.
    """
    import market_calendar

    try:
        return market_calendar.last_completed_session(
            datetime.now(market_calendar.MARKET_TZ)
        )
    except Exception:  # noqa: BLE001
        return datetime.now().date()


class _BuildWorker(QThread):
    """One build, off the Qt thread. A failure arrives as a signal."""

    built = Signal(object)
    failed = Signal(str)

    def __init__(self, work, parent=None) -> None:
        super().__init__(parent)
        self._work = work

    def run(self) -> None:  # pragma: no cover - exercised through its signals
        try:
            self.built.emit(self._work())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class WorkingLatelyService(QObject):
    """Builds the snapshot on a worker, persists it, and says what changed."""

    snapshotChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self, parent=None, *, store_dir: Any = None) -> None:
        super().__init__(parent)
        self._dir = Path(store_dir) if store_dir is not None else default_store_dir()
        self._lock = threading.Lock()
        self._worker: _BuildWorker | None = None
        self._inflight = False
        self._started = False
        # ONE reaction for four triggers (fluidity rule: coalesce at the
        # LISTENER). The store and the day roll keep signalling per event.
        from ui.timer_utils import SignalCoalescer

        self._coalescer = SignalCoalescer(self._refresh_now, parent=self)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self._coalescer.request)

    # -- identity ----------------------------------------------------------

    @property
    def store_dir(self) -> Path:
        return self._dir

    @property
    def snapshot_path(self) -> Path:
        return self._dir / "snapshot_latest.json"

    @property
    def events_path(self) -> Path:
        return self._dir / "leader_change_events.jsonl"

    # -- the four triggers -------------------------------------------------

    def start(self) -> None:
        """Trigger (a): once, after the window has shown. Idempotent."""
        if self._started:
            return
        self._started = True
        self._timer.start()
        self._coalescer.request()

    def on_day_roll(self) -> None:
        """Trigger (b): a new session. The window moved; the reading has not."""
        self._coalescer.request()

    def on_tracker_export(self) -> None:
        """Trigger (c): a persisted tracker write landed - new recent rows."""
        self._coalescer.request()

    def request_refresh(self) -> None:
        """Any caller's "please re-read", folded into the same window."""
        self._coalescer.request()

    def shutdown(self) -> None:
        self._timer.stop()
        worker = self._worker
        if worker is not None:
            from ui.read_worker import join_worker

            join_worker(worker)
        self._worker = None

    # -- the worker --------------------------------------------------------

    def _refresh_now(self) -> None:
        if self._inflight:
            # Single-flight: a second request while a build is running is
            # already answered by the build that is running.
            return
        self._inflight = True
        worker = _BuildWorker(self.build_payload, parent=self)
        worker.built.connect(self._on_payload_ready)
        worker.built.connect(self._clear_inflight)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _clear_inflight(self, _payload: object = None) -> None:
        self._inflight = False

    def _on_failed(self, message: str) -> None:
        self._inflight = False
        # Last good survives: nothing is cleared, and the failure is said out
        # loud rather than shown as an empty strip.
        self.statusChanged.emit(f"Working lately: the reading failed ({message}).")

    def build_payload(self) -> dict[str, Any]:
        """THE WORKER SIDE. Reads, builds, publishes, returns the payload.

        The outcome window is streamed once and shared by held x ran and the
        setup grades. The grades ride on the emitted dict under `setup_grades`
        and in their own file; the persisted snapshot is unchanged.
        """
        global _SETUPS_THIS_BUILD
        _SETUPS_THIS_BUILD = []
        try:
            return self._build_payload()
        finally:
            _SETUPS_THIS_BUILD = None

    def _build_payload(self) -> dict[str, Any]:
        global _OUTCOME_ROWS_THIS_BUILD
        try:
            try:
                _OUTCOME_ROWS_THIS_BUILD = _outcome_rows()
            except Exception:  # noqa: BLE001 - each reader reports its own absence
                _OUTCOME_ROWS_THIS_BUILD = None
            recent_rows = read_recent_rows()
            snapshot = build_snapshot(
                recent_rows=recent_rows,
                favorable_read=read_favorable_read(),
                held_run_summaries=read_held_run_summaries(),
                last_completed_session=_last_completed_session(),
                previous_verdicts=self.previous_verdicts(),
            )
            grades = read_setup_grades(recent_rows)
            try:
                recent_m5 = recent_m5_results()
            except Exception:  # noqa: BLE001 - read_looking_back reports its own failure
                logging.warning("Looking-back recent M5 read failed", exc_info=True)
                recent_m5 = None
        finally:
            _OUTCOME_ROWS_THIS_BUILD = None
        # After the recent window's rows are freed: the prior stream never
        # holds both windows at once.
        looking = (
            read_looking_back(
                recent_rows=recent_rows, snapshot=snapshot, grades=grades, recent_m5=recent_m5
            )
            if recent_m5 is not None
            else None
        )
        self.publish(snapshot)
        payload = snapshot.to_payload()
        if grades is not None:
            self._write_grades(grades)
        else:
            grades = read_persisted_grades(self._dir) or None
        if grades:
            payload["setup_grades"] = grades
        if looking is not None:
            self._write_json(LOOKING_BACK_FILE_NAME, looking)
        else:
            looking = read_persisted_looking_back(self._dir) or None
        if looking:
            payload["looking_back"] = looking
        return payload

    def _write_json(self, name: str, value: Mapping[str, Any]) -> None:
        """Temp-and-rename; a failed write keeps the last good file."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._dir / name
            temp = target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(value, default=str), encoding="utf-8")
            os.replace(temp, target)
        except OSError:
            logging.warning("Writing %s failed", name, exc_info=True)

    def _write_grades(self, grades: Mapping[str, Any]) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._dir / GRADES_FILE_NAME
            temp = target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(grades, default=str), encoding="utf-8")
            os.replace(temp, target)
        except OSError:
            logging.debug("Setup grades write failed", exc_info=True)
            return
        # Dated point-in-time copy for the Day Recap; a failure loses this line only.
        try:
            setup_grades_history.append_snapshot(grades, history_dir=self._history_dir())
        except Exception:  # noqa: BLE001 - evidence write, never fatal
            logging.warning("Setup grades history write failed", exc_info=True)

    def _history_dir(self) -> Path:
        """The live history constant for the default store, else beside `store_dir`."""
        from project_paths import SETUP_GRADES_HISTORY_DIR

        default = Path(SETUP_GRADES_HISTORY_DIR)
        if self._dir == default.parent:
            return default
        return self._dir / default.name

    def _on_payload_ready(self, payload: dict) -> None:
        """THE GUI SLOT. It formats nothing and reads nothing - it emits."""
        self.snapshotChanged.emit(dict(payload or {}))

    # -- persistence -------------------------------------------------------

    def load_snapshot(self) -> dict[str, Any] | None:
        return read_persisted_snapshot(self._dir) or None

    def previous_verdicts(self) -> dict[str, Any]:
        """`{kind: LeaderVerdict}` from the persisted snapshot, `{}` when none."""
        return working_lately.verdicts_from_payload(self.load_snapshot())

    def events(self) -> list[dict[str, Any]]:
        """Every leader-change event ever written here, in file order."""
        return read_persisted_events(store_dir=self._dir)

    def events_for_session(self, session: str) -> list[dict[str, Any]]:
        """That session's events only - what the AWAY Recap prints."""
        return read_persisted_events(session, store_dir=self._dir)

    def publish(self, snapshot: EvidenceSnapshot) -> list[dict[str, Any]]:
        """Persist the snapshot and append the events THIS call produced.

        The prior payload is read BEFORE the new one is written, because the
        `corrected_data` cause is a comparison between the two.
        """
        with self._lock:
            prior = self.load_snapshot()
            candidates = self._events_for(snapshot, prior)
            self._write_snapshot(snapshot)
            return self._append_events(candidates)

    def _write_snapshot(self, snapshot: EvidenceSnapshot) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        # COMPACT, and the reason is that the size cap is about this file rather
        # than about a shape (re-review round 2): `indent=2` cost 12,876 bytes on
        # live data - a quarter of the file - purely in leading spaces, so the
        # test measured 40,045 while the desk wrote 52,921 and blew the 48 KB
        # gate. Nothing reads this by eye that cannot pipe it through a
        # formatter; every surface reads it through `json.loads`.
        payload = json.dumps(snapshot.to_payload(), default=str)
        temp = self.snapshot_path.with_suffix(".json.tmp")
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, self.snapshot_path)

    def _append_events(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        seen = {self._dedupe_key(row) for row in self.events()}
        fresh = []
        for row in candidates:
            key = self._dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            fresh.append(row)
        if not fresh:
            return []
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8", newline="\n") as handle:
            for row in fresh:
                handle.write(json.dumps(row, default=str) + "\n")
        return fresh

    @staticmethod
    def _dedupe_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("kind") or ""),
            str(row.get("prior_leader") or ""),
            str(row.get("new_leader") or ""),
            str(row.get("new_snapshot_id") or ""),
        )

    # -- what changed, and why ---------------------------------------------

    def _events_for(
        self, snapshot: EvidenceSnapshot, prior: Mapping[str, Any] | None
    ) -> list[dict[str, Any]]:
        """One candidate event per kind whose leadership CHANGED.

        Two things count as a change (lead decision, 2026-09-06): the leader
        NAME moved, or the STATE moved while at least one side named somebody.
        The second clause is what makes a lost-coverage event exist at all - it
        carries the SAME name in both slots, because nothing new was learned;
        the desk simply stopped being able to read it. The first-ever build,
        where both names are empty, is not news and writes nothing.
        """
        # Advisory 6: MARKET-local and AWARE, the same clock `as_of` is on.
        stamp = working_lately.market_local_now()
        prior_id = str((prior or {}).get("snapshot_id") or "")
        events: list[dict[str, Any]] = []
        for kind in sorted(snapshot.verdicts):
            verdict = snapshot.verdicts[kind]
            prior_name = str(verdict.coverage.get("prior_leader") or "")
            prior_state = str(verdict.coverage.get("prior_state") or "")
            new_name = leader_name(verdict)
            state_moved = verdict.state != prior_state and bool(prior_name or new_name)
            if new_name == prior_name and not state_moved:
                continue
            events.append(
                {
                    "schema": EVENT_SCHEMA,
                    "ts": stamp,
                    "kind": kind,
                    "prior_leader": prior_name,
                    "new_leader": new_name,
                    "prior_snapshot_id": prior_id,
                    "new_snapshot_id": snapshot.snapshot_id,
                    "cause": self._cause(kind, snapshot, prior, prior_name),
                    "as_of": snapshot.as_of,
                    "prior_state": prior_state,
                    "new_state": verdict.state,
                }
            )
        return events

    def _cause(
        self,
        kind: str,
        snapshot: EvidenceSnapshot,
        prior: Mapping[str, Any] | None,
        prior_name: str,
    ) -> str:
        """REFUSAL FIRST, then correction, then the calendar, then news.

        The packet listed `window_rollover` first, but `as_of` moves on nearly
        every build, so checking it first would make `corrected_data` and
        `lost_coverage` unreachable - the two causes that are worth reading.
        """
        verdict = snapshot.verdicts[kind]
        if verdict.state in {"last_reliable_reading", "no_evidence"} and prior_name:
            return "lost_coverage"
        if prior and self._restated(kind, snapshot, prior):
            return "corrected_data"
        if prior and str(prior.get("as_of") or "") != snapshot.as_of:
            return "window_rollover"
        return "new_outcomes"

    @staticmethod
    def _restated(
        kind: str, snapshot: EvidenceSnapshot, prior: Mapping[str, Any]
    ) -> bool:
        """Did rows for a session at or before the PREVIOUS `as_of` change?

        A restatement is not news about the market, and an events file that
        called it `new_outcomes` would teach the trader that the desk found
        something.

        The one subtlety, and it is deliberate: a bucket that merely EMPTIED
        while the source's total held steady is a window moving forward, not a
        correction - the recent-types export re-states one row per family under
        a new `latest_measured_session` every close slot, so every previous
        session's bucket empties on a perfectly ordinary day. A bucket that
        GAINED rows, or one that lost them while the source shrank, is a
        restatement and is labelled as one.
        """
        prior_as_of = str(prior.get("as_of") or "")
        if not prior_as_of:
            return False
        prior_entry = ((prior.get("sources") or {}).get(kind) or {})
        new_entry = snapshot.sources.get(kind) or {}
        prior_map = dict(prior_entry.get("rows_by_session") or {})
        new_map = dict(new_entry.get("rows_by_session") or {})
        prior_total = prior_entry.get("rows")
        new_total = new_entry.get("rows")
        shrank = (
            isinstance(prior_total, int)
            and isinstance(new_total, int)
            and new_total < prior_total
        )
        for session in set(prior_map) | set(new_map):
            if str(session)[:10] > prior_as_of:
                continue
            before = int(prior_map.get(session) or 0)
            after = int(new_map.get(session) or 0)
            if before == after:
                continue
            if after > 0 or shrank:
                return True
        return False
