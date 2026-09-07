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
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import QObject, QThread, QTimer, Signal

import working_lately
from working_lately import EvidenceSnapshot, build_snapshot, leader_name

#: How often the desk re-reads the three sources on its own. Half an hour is
#: slower than anything that can change inside a session (the tracker writes at
#: the close slot; the outcome log grows all day) and fast enough that a
#: leader change is on screen before the trader next looks for it.
REFRESH_INTERVAL_MS = 30 * 60 * 1000

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


def read_held_run_summaries() -> Any:
    """`held_run_score.dimension_summaries` over the rolling window, or None."""
    try:
        import held_run_score

        episodes = held_run_score.load_episodes()
        if not episodes:
            return None
        return held_run_score.dimension_summaries(episodes)
    except Exception:  # noqa: BLE001 - the day-trade half is absent, not fatal
        logging.debug("Working-lately held-run read failed", exc_info=True)
        return None


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
        """THE WORKER SIDE. Reads, builds, publishes, returns the payload."""
        snapshot = build_snapshot(
            recent_rows=read_recent_rows(),
            favorable_read=read_favorable_read(),
            held_run_summaries=read_held_run_summaries(),
            last_completed_session=_last_completed_session(),
            previous_verdicts=self.previous_verdicts(),
        )
        self.publish(snapshot)
        return snapshot.to_payload()

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
