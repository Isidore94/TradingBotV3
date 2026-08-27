"""Owner of the group RS/RW tape's data (plan.md Phase 0.5 item 11, packet T-2).

One single-flight owner, one timer, one last-good payload - the Strength Board
pattern (`strength_board_service.py`), for the same reason: **zero IB traffic**,
so the locked pacing budget in `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` sec 5.2-5.3
is untouched and does not need re-litigating.

What this replaces
------------------
The tape used to render whatever `compute_group_strengths` had last left on
BounceBot, which meant it moved only when a scan cycle's RRS pass finished - on
2026-08-27 that was 10 to 30 minutes apart, and once 31 minutes late on a flip.
Its one intraday number was a 60-minute window taken from a 5-day fetch, so for
the first hour of the session it straddled the overnight gap.

Here the tape owns its own clock: one batched `yfinance` download of SPY plus
the sector and industry proxy ETFs every 5 minutes, today's completed bars
only, three windows. The download and the maths run on a worker thread; the Qt
side receives a finished payload.

The RS Window tab is deliberately NOT touched. It keeps reading BounceBot's
`rrsSnapshotChanged` payload, which answers a different question - who led over
the selected window at scan time.

Decision support only: no alerts, no watchlist writes, no influence on any
champion path.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal

import autopilot_core as core
import group_rrs
from completed_bars import bar_time
from ui.timer_utils import start_staggered, stop_staggered

#: Refresh cadence, in minutes. Settings-tunable. Five minutes is the bar size:
#: a faster tick could not produce a new answer, because the input only changes
#: when a 5-minute bar completes.
GROUP_TAPE_REFRESH_SETTING = "group_tape_refresh_minutes"
GROUP_TAPE_REFRESH_MINUTES = 5
#: The timer ticks faster than the cadence and `_due` decides - the same shape
#: the Strength Board uses, so a manual refresh or a first run lands promptly
#: instead of waiting out a full period.
_TICK_INTERVAL_MS = 30_000
#: Yahoo rate-limits bursts (a diagnostic run hit `YFRateLimitError` on the
#: 12th single-ticker call). ONE batched request per tick, and never a retry
#: inside the tick - the next tick is the retry.
_FETCH_PERIOD = "1d"
_FETCH_INTERVAL = "5m"
#: How long `shutdown` will wait for an in-flight fetch. Bounded deliberately:
#: an unbounded join here is a hang waiting for a slow Yahoo day, which is the
#: lesson `_GuiGcController` and the 2026-08-26 shutdown freeze both paid for.
#: The worker is a daemon doing a pure read, so abandoning it is safe.
SHUTDOWN_JOIN_SECONDS = 2.0


class GroupTapeService(QObject):
    """Fetches, scores and publishes the sector/industry tape."""

    tapeChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self, parent=None, *, downloader: Callable[..., Any] | None = None) -> None:
        super().__init__(parent)
        self._downloader = downloader
        self._running = False
        self._payload: dict[str, Any] = empty_payload()
        self._last_success: datetime | None = None
        self._last_error = ""
        self._last_attempt: datetime | None = None
        self._worker_thread: threading.Thread | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        start_staggered(self._timer, 17_000)

    # ------------------------------------------------------------------ reads
    @property
    def running(self) -> bool:
        return self._running

    def payload(self) -> dict[str, Any]:
        return dict(self._payload)

    def status_text(self) -> str:
        """One line the trader can trust about the tape's freshness.

        A failed refresh keeps the last good payload (plan.md sec 5: a failed
        publish never destroys the last verified one), so the age and the
        failure both have to be visible - a stale tape that looks current is
        worse than no tape.
        """
        if self._running:
            return "Group tape: refreshing..."
        if self._last_success is None:
            if self._last_error:
                return f"Group tape: never refreshed (last attempt failed: {self._last_error})"
            return "Group tape: never refreshed"
        stamp = self._last_success.strftime("%H:%M:%S")
        measured = int(self._payload.get("measured") or 0)
        offered = int(self._payload.get("offered") or 0)
        text = f"Group tape {stamp}: {measured} of {offered} groups measured"
        note = str(self._payload.get("note") or "")
        if note:
            text = f"{text} · {note}"
        if self._last_error:
            text = f"{text} · last refresh FAILED: {self._last_error}"
        return text

    # ----------------------------------------------------------------- control
    def refresh_now(self) -> bool:
        """Manual refresh. Never gated on quiet hours - hard rule 6."""
        return self._start(manual=True)

    def shutdown(self) -> None:
        stop_staggered(self._timer)
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            # Bounded: see SHUTDOWN_JOIN_SECONDS. If Yahoo is slow the process
            # leaves without it rather than holding the window open.
            worker.join(SHUTDOWN_JOIN_SECONDS)

    # ------------------------------------------------------------------ timer
    def _tick(self) -> None:
        try:
            now = datetime.now()
            if not self._due(now):
                return
            self._start(manual=False)
        except Exception:
            logging.exception("Group tape tick failed")

    def _due(self, now: datetime) -> bool:
        if self._running:
            return False
        # Quiet hours (packet R1): automatic work runs on the session window
        # and stops overnight like every other automatic starter. Fail open on
        # a session lookup that cannot answer, as everywhere else.
        try:
            allowed, _reason = core.auto_scanning_due(now)
        except Exception:
            allowed = True
        if not allowed:
            return False
        if self._last_attempt is None:
            return True
        elapsed = (now - self._last_attempt).total_seconds() / 60.0
        return elapsed >= self._refresh_minutes()

    def _refresh_minutes(self) -> float:
        try:
            from project_paths import get_local_setting

            value = float(
                get_local_setting(GROUP_TAPE_REFRESH_SETTING, GROUP_TAPE_REFRESH_MINUTES)
            )
        except Exception:
            return float(GROUP_TAPE_REFRESH_MINUTES)
        return value if value > 0 else float(GROUP_TAPE_REFRESH_MINUTES)

    # ------------------------------------------------------------------- work
    def _start(self, *, manual: bool) -> bool:
        """Single flight. A refresh already in progress wins; the caller is
        told rather than silently ignored."""
        if self._running:
            self.statusChanged.emit("Group tape refresh already running.")
            return False
        self._running = True
        self._last_attempt = datetime.now()
        self.statusChanged.emit(
            f"Group tape refreshing ({'manual' if manual else 'scheduled'})..."
        )
        self._worker_thread = threading.Thread(
            target=self._worker, name="group-tape", daemon=True
        )
        self._worker_thread.start()
        return True

    def _worker(self) -> None:
        try:
            payload = build_group_tape(downloader=self._downloader)
            self._payload = payload
            self._last_success = datetime.now()
            self._last_error = ""
            self.tapeChanged.emit(dict(payload))
        except Exception as exc:
            self._last_error = str(exc) or exc.__class__.__name__
            logging.exception("Group tape refresh failed")
        finally:
            self._running = False
            self.statusChanged.emit(self.status_text())


# --------------------------------------------------------------------- build


def empty_payload() -> dict[str, Any]:
    """What the tape publishes before its first successful read."""
    return {
        "group_strength": {label: {"sectors": [], "industries": []} for label in group_rrs.RRS_WINDOWS},
        "as_of": None,
        "as_of_text": "",
        "source": "yfinance",
        "status": "",
        "note": "",
        "measured": 0,
        "offered": 0,
    }


def load_industry_etfs(path=None) -> tuple[list[str], str]:
    """The distinct industry proxy ETFs, and a note when they are unavailable.

    136 industries map to 49 ETFs (70 of the industries have no proxy at all
    and are simply absent). A missing or unreadable map means SECTORS ONLY,
    said out loud in the status line - a tape that silently lost two thirds of
    its chips would read as "nothing is moving".
    """
    if path is None:
        from project_paths import INDUSTRY_ETF_MAP_FILE

        path = INDUSTRY_ETF_MAP_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"industry map unreadable ({exc.__class__.__name__}) - sectors only"
    refs = data.get("yahoo_industryKey_to_ref")
    if not isinstance(refs, dict):
        return [], "industry map has no industry->ETF table - sectors only"
    etfs = set()
    for ref in refs.values():
        if not isinstance(ref, dict):
            continue
        etf = str(ref.get("etf") or "").strip().upper()
        if etf:
            etfs.add(etf)
    if not etfs:
        return [], "industry map names no ETFs - sectors only"
    return sorted(etfs), ""


def build_group_tape(
    *,
    downloader: Callable[..., Any] | None = None,
    now: datetime | None = None,
    industry_etfs: list[str] | None = None,
    industry_note: str | None = None,
) -> dict[str, Any]:
    """Fetch today's 5m bars for SPY and every group ETF, and score them.

    A plain function rather than a method so the whole pipeline is testable
    without Qt, a timer or a thread.

    Exactly ONE download call, for every symbol at once, with no retry: Yahoo
    rate-limits bursts, and the next tick is five minutes away and is the
    retry. Symbols the response does not carry are counted, not guessed at.
    """
    moment = now or datetime.now()
    note = industry_note or ""
    if industry_etfs is None:
        industry_etfs, note = load_industry_etfs()

    sector_etfs = dict(group_rrs.SECTOR_ETFS)
    wanted: list[str] = [group_rrs.BENCHMARK]
    for etf in list(sector_etfs.values()) + list(industry_etfs):
        if etf not in wanted:
            wanted.append(etf)

    downloader = downloader or core._default_downloader
    data = downloader(wanted, period=_FETCH_PERIOD, interval=_FETCH_INTERVAL)

    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol in wanted:
        try:
            frame = data[symbol] if len(wanted) > 1 else data
        except Exception:
            continue
        rows = core._frame_rows(frame)
        if rows:
            bars_by_symbol[symbol] = rows

    spy_bars = bars_by_symbol.get(group_rrs.BENCHMARK) or []
    spy_today = group_rrs.session_bars(spy_bars, now=moment)

    payload = empty_payload()
    payload["note"] = note
    payload["offered"] = len(sector_etfs) + len(industry_etfs)

    if not spy_today:
        # Everything is measured AGAINST SPY, so no SPY means no tape - and
        # saying so is the honest answer, not an empty strip.
        payload["status"] = (
            "No completed SPY bars for today yet - the tape fills in once the "
            "session has printed its first bars."
        )
        return payload

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        label: {"sectors": [], "industries": []} for label in group_rrs.RRS_WINDOWS
    }
    measured = 0
    for kind, entries in (
        ("sectors", [(key, etf) for key, etf in sorted(sector_etfs.items())]),
        ("industries", [(etf, etf) for etf in industry_etfs]),
    ):
        for group_key, etf in entries:
            windows = group_rrs.rrs_windows(
                bars_by_symbol.get(etf) or [], spy_bars, now=moment
            )
            if any(value is not None for value in windows.values()):
                measured += 1
            for label, value in windows.items():
                if value is None:
                    # UNKNOWN is left OUT of the payload rather than sent as a
                    # zero: the strip blanks a missing window, and a zero would
                    # claim "exactly in line with SPY".
                    continue
                groups[label][kind].append(
                    {"group_key": group_key, "etf": etf, "rrs": value}
                )

    for label in groups:
        for kind in ("sectors", "industries"):
            groups[label][kind].sort(key=lambda row: -row["rrs"])

    last_stamp = bar_time(spy_today[-1])
    payload["group_strength"] = groups
    payload["as_of"] = last_stamp.isoformat() if last_stamp is not None else None
    payload["as_of_text"] = last_stamp.strftime("%H:%M") if last_stamp is not None else ""
    payload["measured"] = measured
    payload["status"] = _status_line(groups, measured, payload["offered"], note)
    return payload


def _status_line(
    groups: dict[str, Any], measured: int, offered: int, note: str
) -> str:
    ready = [
        label
        for label in group_rrs.WINDOW_ORDER
        if groups.get(label, {}).get("sectors") or groups.get(label, {}).get("industries")
    ]
    if not ready:
        need = min(group_rrs.minimum_bars_for(label) for label in group_rrs.RRS_WINDOWS)
        text = (
            f"Not enough completed bars yet - the 30-minute read needs {need} "
            "of them, and none is invented before that."
        )
    else:
        missing = [label for label in group_rrs.WINDOW_ORDER if label not in ready]
        text = f"{measured} of {offered} groups measured on {'/'.join(ready)} min"
        if missing:
            text += f" ({'/'.join(missing)} min still filling)"
    return f"{text} · {note}" if note else text
