"""The one owner of the weekend routine's state and its refreshes (R8 §2/§3).

**No QTimer.** Not "a timer that is usually off" — none exists, and a test
asserts that structurally rather than trusting this sentence. The weekend
quiet-hours gate already refuses automatic work on a Saturday; the design basis
is that manual buttons are the carve-out, so this service starts nothing until
something is pressed. If it owned a timer, the trader's weekend would start
fetching by itself, which is the behaviour the whole gate exists to prevent.

**Single-flight per action.** Each refreshable thing (three boards, walk-away,
week-ahead) has its own in-flight flag, so pressing Refresh twice does not run
twice, and refreshing H1 does not block D1.

**Last good survives a failure.** A failed fetch keeps the board that was there
and puts the error in the status line. An empty board after a network blip
reads as "nothing is strong this week", which is a different and much worse
statement than "the fetch failed".
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

import market_calendar
import weekend_strength
from project_paths import WEEKEND_PREP_STATE_FILE

STEP_IDS: tuple[str, ...] = ("week_review", "focus_review", "walkaway", "discovery", "week_ahead")
STEP_LABELS = {
    "week_review": "Week in review",
    "focus_review": "Focus pick review",
    "walkaway": "Walk-away",
    "discovery": "Discovery",
    "week_ahead": "Week ahead",
}
STEP_STATUSES = ("pending", "done", "skipped")

#: Weekends kept in the state file. Eight is a season's worth of context and a
#: file small enough to read by eye when something looks wrong.
KEEP_WEEKENDS = 8

STATE_VERSION = 1


def weekend_id(now: datetime | None = None) -> str:
    """The Friday of the week containing the last completed session.

    Anchored to the session calendar, not the wall clock: opening the tab on
    Sunday evening and again on Saturday morning must land on the *same*
    weekend, or the routine would silently start over halfway through. A short
    week ending on Thursday still identifies by its Friday date, which keeps the
    id a simple calendar fact rather than a lookup.
    """
    moment = now or datetime.now()
    try:
        session = market_calendar.last_completed_session(moment)
    except Exception:
        session = (moment - timedelta(days=1)).date()
    # Monday=0 ... Friday=4. Walk forward to that week's Friday.
    return (session + timedelta(days=4 - session.weekday())).isoformat()


def week_bounds(weekend: str) -> tuple[date, date]:
    """(Monday, Friday) of the reviewed week."""
    friday = date.fromisoformat(str(weekend)[:10])
    return friday - timedelta(days=4), friday


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _empty_weekend() -> dict[str, Any]:
    return {
        "steps": {step: {"status": "pending", "at": ""} for step in STEP_IDS},
        "boards": {},
        "adopted": [],
        "tag_review": {"confirmed": [], "corrected": {}},
        "week_ahead": {"ran_at": ""},
    }


class _Worker(QThread):
    """One background action. Errors arrive as a signal, never as a crash."""

    done = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, action: str, fn: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._action = action
        self._fn = fn

    def run(self) -> None:  # pragma: no cover - exercised on the desk
        try:
            self.done.emit(self._action, self._fn())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._action, str(exc))


class WeekendPrepService(QObject):
    """State, steps, and manually-triggered refreshes for the weekend routine."""

    stateChanged = Signal()
    boardChanged = Signal(str)          # timeframe key
    statusChanged = Signal(str)
    weekAheadReady = Signal(str)

    def __init__(self, parent=None, *, state_path: Path | None = None, now: datetime | None = None) -> None:
        super().__init__(parent)
        self._path = Path(state_path or WEEKEND_PREP_STATE_FILE)
        self._lock = threading.Lock()
        self._inflight: set[str] = set()
        self._workers: dict[str, _Worker] = {}
        self._boards: dict[str, weekend_strength.WeekendBoard] = {}
        self._board_sides: dict[str, dict[str, weekend_strength.WeekendBoard]] = {}
        self._week_ahead_markdown = ""
        self._now_provider = (lambda: now) if now is not None else datetime.now
        self._weekend = weekend_id(self._now_provider())
        self._state = self._load()

    # -- identity ----------------------------------------------------------

    @property
    def weekend(self) -> str:
        self._weekend = weekend_id(self._now_provider())
        return self._weekend

    @property
    def week_bounds(self) -> tuple[date, date]:
        return week_bounds(self.weekend)

    # -- state -------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"version": STATE_VERSION, "weekends": {}}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt state file loses progress, not the routine. Starting
            # fresh and saying so beats refusing to open the tab.
            logging.warning("Weekend prep state unreadable (%s); starting fresh", exc)
            return {"version": STATE_VERSION, "weekends": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("weekends"), dict):
            return {"version": STATE_VERSION, "weekends": {}}
        payload.setdefault("version", STATE_VERSION)
        return payload

    def _save(self) -> None:
        """Atomic write, then prune. A half-written state file is worse than none."""
        with self._lock:
            weekends = self._state.setdefault("weekends", {})
            for key in sorted(weekends, reverse=True)[KEEP_WEEKENDS:]:
                weekends.pop(key, None)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._path)

    def weekend_state(self, weekend: str | None = None) -> dict[str, Any]:
        key = weekend or self.weekend
        weekends = self._state.setdefault("weekends", {})
        if key not in weekends:
            weekends[key] = _empty_weekend()
        entry = weekends[key]
        # An older file may predate a step id; fill it rather than KeyError.
        steps = entry.setdefault("steps", {})
        for step in STEP_IDS:
            steps.setdefault(step, {"status": "pending", "at": ""})
        return entry

    def step_status(self, step: str) -> str:
        return str(self.weekend_state()["steps"].get(step, {}).get("status") or "pending")

    def set_step_status(self, step: str, status: str) -> None:
        if step not in STEP_IDS:
            raise ValueError(f"unknown weekend prep step: {step!r}")
        if status not in STEP_STATUSES:
            raise ValueError(f"unknown step status: {status!r}")
        self.weekend_state()["steps"][step] = {"status": status, "at": _now_iso()}
        self._save()
        self.stateChanged.emit()

    @property
    def routine_complete(self) -> bool:
        """Every step done **or skipped**. Skipping is a decision, not a gap."""
        steps = self.weekend_state()["steps"]
        return all(str(steps[step]["status"]) in {"done", "skipped"} for step in STEP_IDS)

    def record_adopted(self, symbol: str, side: str, timeframe: str) -> None:
        entry = self.weekend_state()
        entry.setdefault("adopted", []).append(
            {"symbol": str(symbol).upper(), "side": side, "tf": timeframe, "at": _now_iso()}
        )
        self._save()
        self.stateChanged.emit()

    def record_tag_review(self, trade_id: str, *, corrected_to: str | None = None) -> None:
        review = self.weekend_state().setdefault("tag_review", {"confirmed": [], "corrected": {}})
        if corrected_to is None:
            if trade_id not in review["confirmed"]:
                review["confirmed"].append(trade_id)
        else:
            review.setdefault("corrected", {})[trade_id] = corrected_to
        self._save()
        self.stateChanged.emit()

    # -- refreshes (all manual) -------------------------------------------

    def is_running(self, action: str) -> bool:
        return action in self._inflight

    def board(self, timeframe: str, side: str | None = None) -> weekend_strength.WeekendBoard | None:
        if side is not None:
            return self._board_sides.get(timeframe, {}).get(side)
        return self._boards.get(timeframe)

    @property
    def week_ahead_markdown(self) -> str:
        return self._week_ahead_markdown

    def refresh_board(
        self,
        timeframe: str,
        *,
        side: str = "long",
        downloader: Callable[..., Any] | None = None,
        symbols: list[str] | None = None,
        now: datetime | None = None,
        blocking: bool = False,
    ) -> bool:
        """Fetch and score one timeframe. Returns False if it was already running."""
        if timeframe not in weekend_strength.TIMEFRAMES_BY_KEY:
            raise ValueError(f"unknown timeframe: {timeframe!r}")
        action = f"board:{timeframe}"
        return self._start(
            action,
            lambda: (
                side,
                build_weekend_boards(
                    weekend_strength.TIMEFRAMES_BY_KEY[timeframe],
                    downloader=downloader, symbols=symbols, now=now,
                ),
            ),
            blocking=blocking,
        )

    def refresh_week_ahead(self, *, runner: Callable[[], Any] | None = None, blocking: bool = False) -> bool:
        return self._start("week_ahead", runner or _run_weekly_prep, blocking=blocking)

    def _start(self, action: str, fn: Callable[[], Any], *, blocking: bool) -> bool:
        if action in self._inflight:
            self.statusChanged.emit(f"{action} is already running")
            return False
        self._inflight.add(action)
        if blocking:
            # The test path, and the one the panel never uses: run inline so a
            # test needs no event loop. Nothing schedules this by itself.
            try:
                self._on_done(action, fn())
            except Exception as exc:  # noqa: BLE001
                self._on_failed(action, str(exc))
            return True
        worker = _Worker(action, fn, self)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(
            lambda action=action, worker=worker: self._retire_worker(action, worker)
        )
        self._workers[action] = worker
        worker.start()
        return True

    def _retire_worker(self, action: str, worker: _Worker) -> None:
        if self._workers.get(action) is worker:
            self._workers.pop(action, None)

    def _on_done(self, action: str, result: Any) -> None:
        self._inflight.discard(action)
        if action.startswith("board:"):
            _, timeframe = action.split(":", 1)
            selected_side, boards = result
            self._board_sides[timeframe] = boards
            self._boards[timeframe] = boards[selected_side]
            result = boards[selected_side]
            self.boardChanged.emit(timeframe)
            self.statusChanged.emit(f"{timeframe.upper()} board: {result.accounting}")
        elif action == "week_ahead":
            markdown = str(result or "")
            if markdown.strip():
                self._week_ahead_markdown = markdown
                self.weekAheadReady.emit(markdown)
            entry = self.weekend_state()
            entry.setdefault("week_ahead", {})["ran_at"] = _now_iso()
            self._save()
            self.statusChanged.emit("week ahead ready")

    def _on_failed(self, action: str, message: str) -> None:
        self._inflight.discard(action)
        # Deliberately does not clear the board or the report. An empty board
        # after a blip reads as "nothing is strong", which is a different claim.
        self.statusChanged.emit(f"{action} failed: {message} (showing the last good result)")

    def shutdown(self) -> None:
        workers = list(self._workers.values())
        for worker in workers:
            worker.requestInterruption()
        for worker in workers:
            if worker.isRunning():
                # The fetch functions are not safely cancellable mid-request.
                # Never drop the last reference and let Qt destroy a live
                # QThread; close waits for the bounded provider call to finish.
                worker.wait()
        self._workers.clear()


# ---------------------------------------------------------------------------
# The fetch, as a plain function so the whole pipeline is testable without Qt
# ---------------------------------------------------------------------------


def build_weekend_boards(
    timeframe: weekend_strength.StrengthTimeframe,
    *,
    downloader: Callable[..., Any] | None = None,
    symbols: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, weekend_strength.WeekendBoard]:
    """Batched yfinance over the universe, then the pure board. Zero IB traffic.

    Mirrors the R2 strength board's fetch path deliberately, including its chunk
    size, so the two boards put identical load on the same provider and neither
    can drift into a different pacing story.
    """
    import autopilot_core as core

    from project_paths import UNIVERSE_ALL_FILE
    from watchlist_utils import read_watchlist_symbols

    pool = symbols if symbols is not None else read_watchlist_symbols(UNIVERSE_ALL_FILE)
    pool = [str(item or "").strip().upper() for item in pool]
    pool = [item for item in pool if item]
    moment = now or datetime.now()
    if not pool:
        return {
            side: weekend_strength.WeekendBoard(
                timeframe=timeframe.key, side=side, as_of=moment.isoformat(timespec="seconds")
            )
            for side in ("long", "short")
        }

    fetch = downloader or core._default_downloader
    chunk_size = max(1, int(core.AUTOPILOT_OPEN_SCAN_CHUNK_SIZE))
    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    chunks_attempted = chunks_failed = 0
    last_error = ""
    for start in range(0, len(pool), chunk_size):
        chunk = pool[start : start + chunk_size]
        chunks_attempted += 1
        try:
            data = fetch(chunk, period=timeframe.yf_period, interval=timeframe.yf_interval)
        except Exception as exc:  # noqa: BLE001
            # One bad chunk costs one chunk. The board reports fewer measured
            # names against the same offered count, which is visible in the
            # accounting line rather than silent.
            chunks_failed += 1
            last_error = str(exc)
            logging.warning("Weekend %s chunk %s..%s failed: %s", timeframe.key, chunk[0], chunk[-1], exc)
            continue
        for symbol in chunk:
            try:
                frame = data[symbol] if len(chunk) > 1 else data
            except Exception:
                continue
            rows = core._frame_rows(frame)
            if rows:
                bars_by_symbol[symbol] = rows

    if chunks_attempted and chunks_failed == chunks_attempted:
        # Every chunk failed, so there is no board here - only an absence.
        # Returning an empty one would overwrite the last good board and read as
        # "nothing is strong this week", which is a claim about the market
        # rather than about the provider. Raising routes this to the failure
        # path, where the previous board survives with the error beside it.
        raise RuntimeError(
            f"every {timeframe.key} chunk failed ({chunks_attempted} of {chunks_attempted}); "
            f"last error: {last_error}"
        )
    if chunks_attempted and not bars_by_symbol:
        raise RuntimeError(
            f"{timeframe.key} provider returned no measurable bars for {len(pool)} symbol(s)"
        )

    boards = {
        side: weekend_strength.build_board(timeframe, bars_by_symbol, side=side, now=moment)
        for side in ("long", "short")
    }
    for board in boards.values():
        board.offered = len(pool)
    return boards


def build_weekend_board(
    timeframe: weekend_strength.StrengthTimeframe,
    *,
    side: str = "long",
    downloader: Callable[..., Any] | None = None,
    symbols: list[str] | None = None,
    now: datetime | None = None,
) -> weekend_strength.WeekendBoard:
    """Compatibility wrapper for callers that need one side."""
    return build_weekend_boards(
        timeframe, downloader=downloader, symbols=symbols, now=now
    )[side]


def _run_weekly_prep() -> str:
    """The forward-looking weekly prep, imported lazily inside the worker.

    Lazy on purpose: `market_prep.orchestrator` pulls in the whole pre-session
    stack, and importing that at desk startup for a tab the trader opens on
    Saturdays would cost every weekday launch.
    """
    from market_prep.orchestrator import MarketPrepOrchestrator

    report = MarketPrepOrchestrator().run_weekly_prep()
    markdown = getattr(report, "markdown", None)
    if markdown is None and isinstance(report, dict):
        markdown = report.get("report") or report.get("markdown")
    return str(markdown or "")
