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
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from ui.read_worker import join_worker

from diagnostics.artifact_io import atomic_write_json
import market_calendar
import weekend_strength
from project_paths import WEEKEND_PREP_STATE_FILE
from swallowed import note_swallowed

#: V2 item 2e appended `tag_week`. APPENDED, and the position is argued: the
#: nightly tagger (V2 item 1) leaves provisional tags and needs_review markers
#: all week, and the trader's own answer is what every per-setup statistic on the
#: desk counts. Reviewing them belongs in the weekend routine, after the week has
#: been read and before the week ahead is planned.
#:
#: A weekend already in the state file has no `tag_week` key; `step_status`
#: returns "pending" for a step it has never seen, so an old weekend simply gains
#: an unstarted step rather than failing to load.
STEP_IDS: tuple[str, ...] = (
    "week_review",
    "focus_review",
    "walkaway",
    "tag_week",
    "discovery",
    "week_ahead",
)
STEP_LABELS = {
    "week_review": "Week in review",
    "focus_review": "Focus pick review",
    "walkaway": "Walk-away",
    "tag_week": "Tag this week",
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
    boardFailed = Signal(str, str)      # timeframe key, message
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
        self._restore_boards()

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
            atomic_write_json(self._path, self._state, indent=2)

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

    def _restore_boards(self) -> None:
        stored = self.weekend_state().get("boards") or {}
        for timeframe, payload in stored.items():
            if not isinstance(payload, dict):
                continue
            sides: dict[str, weekend_strength.WeekendBoard] = {}
            for side in ("long", "short"):
                raw = payload.get(side)
                if not isinstance(raw, dict):
                    continue
                try:
                    sides[side] = weekend_strength.WeekendBoard(**raw)
                except TypeError:
                    continue
            if sides:
                self._board_sides[str(timeframe)] = sides
                active = str(payload.get("active_side") or "long")
                self._boards[str(timeframe)] = sides.get(active) or next(iter(sides.values()))

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
            stored = {side: asdict(board) for side, board in boards.items()}
            stored["active_side"] = selected_side
            self.weekend_state().setdefault("boards", {})[timeframe] = stored
            self._save()
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
        if action.startswith("board:"):
            self.boardFailed.emit(action.split(":", 1)[1], message)
        self.statusChanged.emit(f"{action} failed: {message} (showing the last good result)")

    def shutdown(self) -> None:
        workers = list(self._workers.values())
        for worker in workers:
            worker.requestInterruption()
        for worker in workers:
            # The fetch functions are not safely cancellable mid-request, and
            # dropping the last reference to a live QThread lets Qt destroy it
            # mid-run. `join_worker` handles both: it waits, and on timeout it
            # disowns and PARKS the worker rather than dropping it. The wait is
            # bounded because "the provider call is itself bounded" is an
            # assumption about someone else's timeout, and a wrong one costs
            # the trader a desk that will not close (2026-08-26).
            join_worker(worker)
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


# ---------------------------------------------------------------------------
# TJ-5 - the Week Review payload, and the deterministic week / month strip
# ---------------------------------------------------------------------------
#: *"TJ-12's report-card lines re-cut by exchange week for the LAST FOUR WEEKS
#: and for the calendar month to date"* (`plan.md` §12.4 TJ-5 change 3). Four
#: entries, not "however many had facts": a week that vanished because nothing
#: was packed would read as a week that never happened.
WEEK_STRIP_WEEKS = 4

#: The ONE payload the Week Review page renders. Every key is present on a
#: first paint, a failed read and a quiet week alike - the page has one render
#: and no special cases (TJ-1's `empty_payload` rule).
WEEK_PAYLOAD_KEYS: tuple[str, ...] = (
    "week_id",
    "sessions",
    "sessions_with_facts",
    "sessions_missing",
    "cards",
    "week_story",
    "walkaway",
    "strip",
    "tendencies",
    "misses",
    "callouts",
    "learning",
    #: Reserved for TJ-6's kept ideas. The key exists so the page has one shape
    #: before that packet lands; nothing writes it here.
    "ideas",
)

#: The index the small day chart draws. One symbol, from TJ-2's durable session
#: bars - never a fetch, and never the lake.
WEEK_CHART_SYMBOL = "SPY"

#: How many points one small chart carries. A session is ~78 completed M5 bars;
#: this is the ceiling, and a longer tape is thinned evenly rather than cut, so
#: the shape of the day survives.
WEEK_CHART_POINTS = 120


def empty_week_payload(week_id: str = "") -> dict[str, Any]:
    """Every :data:`WEEK_PAYLOAD_KEYS` key, and nothing in any of them."""
    return {
        "week_id": str(week_id or ""),
        "sessions": [],
        "sessions_with_facts": [],
        "sessions_missing": [],
        "cards": [],
        "week_story": {},
        "walkaway": {},
        "strip": {"weeks": (), "month_to_date": {}},
        "tendencies": [],
        "misses": {},
        "callouts": [],
        "learning": {
            "schema": "session_learning_window_v1", "window": {"requested": 5, "sessions": [], "start": "", "end": ""},
            "reads": {"horizons": {}}, "by_hour": [], "by_environment": [],
            "trade_groups": [], "trade_rows": [], "coverage": {}, "error": "",
        },
        "ideas": [],
    }


def _default_ledger_path():
    """The AI job ledger, or ``None``. It never CREATES the store.

    `ai_jobs.ledger.ledger_path()` defaults to ``create=True``; a reader whose
    honest answer may be "night status unknown" may not make a store in order
    to say so (`day_review_service._ledger_path` is the precedent).
    """
    try:
        from ai_jobs import ledger

        path = ledger.ledger_path(create=False)
    except Exception:  # noqa: BLE001 - no store is "unknown", never a failure
        return None
    return path if path is not None and Path(path).exists() else None


def _stored_card(session: str, pack: Any, ledger_path):
    """One session's report card, from the lines the PACK stored (TJ-12/TJ-4).

    A pack cannot be turned back into a `day_report_card.build` input - that
    needs TJ-11's `WalkawayDay` object - and it does not need to be: the pack
    carries the BUILT lines with all of their integers.

    `how_fresh` is asked HERE, per session, because it is deliberately not in
    the pack: it describes the machine's night, and its text moves every time
    the job ledger gains a row. It is always handed a SESSION - `_slot_verdicts`
    guards with ``if session and ...``, so an empty one pools every night the
    ledger tail holds and a five-day window would multiply its own `n` by five.
    """
    import day_report_card

    lines = [
        dict(line)
        for line in ((pack or {}).get("report_card") or {}).get("lines") or ()
        if isinstance(line, dict) and line.get("key") in day_report_card.PACK_LINE_KEYS
    ]
    lines.append(
        day_report_card.how_fresh({"session": session, "ledger_path": ledger_path})
    )
    return day_report_card.ReportCard(session=session, lines=tuple(lines))


def _window_block(
    sessions: Sequence[str], *, root, ledger_path, label_key: str, label: str
) -> dict[str, Any]:
    """One strip cell: the window's sessions, which of them have facts, its lines."""
    import day_report_card
    import day_review_pack

    packs: dict[str, Any] = {}
    for session in sessions:
        pack = day_review_pack.read_pack(session, root=root)
        if isinstance(pack, dict) and pack:
            packs[session] = pack
    cards = [
        _stored_card(session, packs[session], ledger_path)
        for session in sessions
        if session in packs
    ]
    pooled = day_report_card.week_from_cards(cards)
    return {
        label_key: label,
        "sessions": tuple(sessions),
        "sessions_with_facts": tuple(session for session in sessions if session in packs),
        "lines": [dict(line) for line in pooled.lines],
    }


def week_strip(*, friday: str, root=None, ledger_path=None) -> dict[str, Any]:
    """TJ-12's lines re-cut by week and by month. Deterministic, no model.

    `plan.md` §12.4 TJ-5 change 3: *"same functions, longer window, `n` on every
    cell, a week under its floor named and not ranked. No new page, no model."*
    The pooling is `day_report_card.week_from_cards` and nothing here computes a
    statistic of its own - ``rate_lb`` is the ONE Wilson, on the POOLED pair.

    Weeks are NEWEST FIRST and there are always :data:`WEEK_STRIP_WEEKS` of
    them, each named with its own week id even when it holds nothing at all.
    """
    from ai_jobs import week_review_narration as week_module

    anchor = date.fromisoformat(str(friday)[:10])
    if ledger_path is None:
        ledger_path = _default_ledger_path()

    weeks: list[dict[str, Any]] = []
    for index in range(WEEK_STRIP_WEEKS):
        end = anchor - timedelta(days=7 * index)
        sessions = [
            day for day in week_module.week_sessions(end.isoformat())
            if day <= anchor.isoformat()
        ]
        weeks.append(
            _window_block(
                sessions,
                root=root,
                ledger_path=ledger_path,
                label_key="week_id",
                label=week_module.week_id(end.isoformat()),
            )
        )

    month_sessions = _month_to_date_sessions(anchor)
    month = _window_block(
        month_sessions,
        root=root,
        ledger_path=ledger_path,
        label_key="month_id",
        label=f"{anchor.year:04d}-{anchor.month:02d}",
    )
    return {"weeks": tuple(weeks), "month_to_date": month}


def _month_to_date_sessions(anchor: date) -> list[str]:
    """The exchange sessions of ``anchor``'s month, up to and including it.

    Nothing before the first of the month, nothing after the anchor: a month to
    date that reached into the previous month would be a different window
    wearing this one's name.
    """
    out: list[str] = []
    cursor = anchor.replace(day=1)
    while cursor <= anchor:
        try:
            keep = market_calendar.is_session(cursor)
        except Exception:  # noqa: BLE001 - an unanswerable calendar keeps weekdays
            keep = cursor.weekday() < 5
        if keep:
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _spy_points(session: str, root) -> list[dict[str, Any]]:
    """The session's SPY closes, thinned to :data:`WEEK_CHART_POINTS`.

    TJ-2's durable parquet only. Nothing is fetched: a page that reached a
    provider on open would be five network calls behind a tab click.
    """
    try:
        import day_review_bars

        bars = day_review_bars.read_session_bars(session, root=root)
    except Exception:  # noqa: BLE001 - no tape is an empty chart, never a failure
        logging.debug("Day Review bars for %s were unreadable.", session, exc_info=True)
        return []
    rows = list((bars or {}).get(WEEK_CHART_SYMBOL) or ())
    if not rows:
        return []
    step = max(1, len(rows) // WEEK_CHART_POINTS)
    return [
        {"t": str(row.get("dt") or ""), "c": float(row.get("close") or 0.0)}
        for row in rows[::step]
        if row.get("close") is not None
    ][:WEEK_CHART_POINTS]


def _said_vs_did(day: Mapping[str, Any]) -> str:
    """What the trader CALLED beside what they TRADED. Counting, never a verdict."""
    called = int((day.get("said_counts") or {}).get("predictions") or 0)
    traded = int((day.get("trades") or {}).get("n") or 0)
    said = f"You called {called} thing(s)" if called else "You called nothing"
    did = f"took {traded} trade(s)" if traded else "took nothing"
    return f"{said} and {did}."


def _week_callouts() -> list[str]:
    """The review-learning callouts this page has printed since R8.

    Kept, and kept on the WORKER: `build_review_learning_state` was the 8.45 s
    freeze this page's reader exists for (fluidity capture, 2026-08-25). A store
    it cannot read says so in one line rather than costing the whole week page -
    the five day cards and the week story are a different question.
    """
    from ui.panels import weekend_prep_panel

    try:
        from evidence_stats import WEEK_SESSIONS
        from review_learning import build_review_learning_state

        state = build_review_learning_state(window_sessions=WEEK_SESSIONS)
    except Exception as exc:  # noqa: BLE001
        logging.debug("The review-learning state was unreadable.", exc_info=True)
        return [f"Week callouts unavailable: {exc}"]
    return list(weekend_prep_panel.callout_lines(state))


def read_week_review(
    *, friday: str = "", root=None, ledger_path=None, now: datetime | None = None,
    window_sessions: int = 5,
) -> dict[str, Any]:
    """ONE payload for the Week Review page. Every store this page opens.

    Called on the page's worker and nowhere else. It builds no pack, no report
    card and no narration - it READS what the night wrote, and the numbers are
    the night's own: `week_review_narration.build_week_inputs` is the same
    function the Saturday slot narrates from, so the page and the story cannot
    disagree about how many days had facts.
    """
    from ai_jobs import week_review_narration as week_module

    anchor = str(friday or "")[:10] or weekend_id(now)
    if ledger_path is None:
        ledger_path = _default_ledger_path()

    inputs = week_module.build_week_inputs(anchor, root=root, ledger_path=ledger_path)
    stored = week_module.read_week_narration(inputs["week_id"], root=root) or {}

    cards: list[dict[str, Any]] = []
    for day in inputs.get("days") or ():
        session = str(day.get("session") or "")
        has_facts = bool(day.get("has_facts"))
        story = day.get("story") or {}
        chased = str((story.get("chased_against_news") or {}).get("verdict") or "")
        read_horizons: dict[str, Any] = {}
        if has_facts:
            # Read the same saved card the week narration already used. The
            # day card needs its named horizons; its legacy tally pools them.
            try:
                import day_review_pack

                saved = day_review_pack.read_pack(session, root=root) or {}
                lines = (saved.get("report_card") or {}).get("lines") or ()
                read_line = next(
                    (line for line in lines if isinstance(line, dict) and line.get("key") == "your_reads"),
                    {},
                )
                read_horizons = dict(read_line.get("horizons") or {})
            except Exception as swallowed_exc:  # one old/unreadable card remains explicitly unseparated
                note_swallowed("old week card unreadable; read horizons left unseparated", swallowed_exc, quiet=True)
        cards.append(
            {
                "session": session,
                "has_facts": has_facts,
                "headline": str(story.get("headline") or ""),
                "tally": dict(day.get("tally") or {}) if has_facts else {},
                "read_horizons": read_horizons,
                "chased": chased or ("unknown" if has_facts else "unmeasured"),
                "said_vs_did": _said_vs_did(day) if has_facts else "",
                "spy_bars": _spy_points(session, root) if has_facts else [],
            }
        )

    payload = empty_week_payload(inputs["week_id"])
    payload.update(
        {
            "sessions": list(inputs.get("sessions") or ()),
            "sessions_with_facts": list(inputs.get("sessions_with_facts") or ()),
            "sessions_missing": list(inputs.get("sessions_missing") or ()),
            "cards": cards,
            "week_story": dict(stored),
            "walkaway": dict(inputs.get("walkaway_totals") or {}),
            "strip": week_strip(friday=anchor, root=root, ledger_path=ledger_path),
            "tendencies": list(inputs.get("tendencies") or ()),
            "misses": dict(inputs.get("misses") or {}),
            "callouts": _week_callouts(),
        }
    )
    try:
        from session_review import read_learning_window

        payload["learning"] = read_learning_window(
            end_session=anchor, sessions=window_sessions, root=root, now=now
        )
    except Exception as exc:  # the Week Review keeps its verified calendar-week content
        payload["learning"]["error"] = f"learning window unavailable: {exc}"
    # TJ-6, in its OWN guard and read exactly ONCE: the ideas the trader kept,
    # each with the baseline frozen at the keep beside the same measurable now.
    # A store that will not open costs the ideas card and leaves the five day
    # cards, the story and the strip exactly where they are.
    try:
        from ai_jobs import improvement_ideas

        payload["ideas"] = [
            dict(row) for row in improvement_ideas.checked_ideas(end_session=anchor)
        ]
    except Exception:  # noqa: BLE001 - one unreadable store costs one section
        logging.debug("The kept ideas could not be read.", exc_info=True)
    return payload


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
