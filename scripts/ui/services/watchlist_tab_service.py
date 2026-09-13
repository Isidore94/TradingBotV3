"""The Watchlist tab's ONE reader (WS-WL item 2).

The `StrengthBoardService` precedent: one owner, one timer, one worker thread,
one last-good payload. The tab renders what this publishes and reads no store
of its own except the cheap ones it can answer instantly.

**The split, and why it is where it is.** Four of the six inputs are small text
or JSONL files in the shared home and cost microseconds. The fifth - the
Journal - is sqlite over a year of fills, and the Journal is the whole point of
the Positions view. So:

* :func:`read_journal` runs on the worker named ``watchlist-tab`` and NEVER on
  the Qt thread; its answer is cached here and handed to every build.
* :func:`gather` reads the cheap stores and is safe from either thread.
* the tab's own ``refresh_now`` rebuilds synchronously from the cheap stores
  plus this cache, so a click answers at once, and asks this service for a
  fresh journal read in the background.

**The Focus store is never touched from the worker.** `FocusPickStore.reload()`
expires the m5 lists and repairs the fade clocks - it is a WRITER - so the Qt
thread freezes it into a `watchlist_views.FocusSnapshot` (three in-memory list
copies) and the snapshot is what crosses the thread boundary.

Nothing here writes anything, arms anything or reaches a broker.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal

import watchlist_views
from ui.timer_utils import start_staggered, stop_staggered

logger = logging.getLogger(__name__)

#: How often the stores are re-checked for a change. This is a `stat()` on four
#: paths, not a read: a tick that finds nothing new costs nothing.
_TICK_INTERVAL_MS = 15_000

#: How far back a CLOSED trade is still worth carrying. A position closed
#: inside this window may still be shown - stale - because the sync that closed
#: it has not been verified since; one closed last spring is simply history.
RECENT_TRADE_DAYS = 14

#: `import_runs.source` -> the broker the trades carry. `RECONCILE` is
#: deliberately absent: it is a repair pass over what is already imported, not
#: a fresh look at the broker, and reading it as a sync would make a stale
#: account look current.
_RUN_BROKERS = (("QUESTRADE", "questrade"), ("IBKR", "ibkr"))

#: The one import-run status that means "we actually saw the broker".
_RUN_VERIFIED = "OK"


class JournalSnapshot:
    """What one journal read answers: the positions, and when each broker was
    last verified. Immutable in practice; replaced wholesale, never patched."""

    __slots__ = ("trades", "last_sync", "read_at", "error")

    def __init__(
        self,
        trades: tuple[dict[str, Any], ...] = (),
        last_sync: dict[str, datetime] | None = None,
        read_at: datetime | None = None,
        error: str = "",
    ) -> None:
        self.trades = tuple(trades)
        self.last_sync = dict(last_sync or {})
        self.read_at = read_at
        self.error = error


def read_journal(*, today: date | None = None) -> JournalSnapshot:
    """Open positions and the last VERIFIED sync per broker. Worker only.

    Bounded on purpose: every OPEN and partly-closed trade, plus everything
    touched in the last :data:`RECENT_TRADE_DAYS` days so a just-closed
    position can still be shown while its sync is unverified. A journal that
    cannot be opened is an EMPTY snapshot with its error named - never an
    exception into the tab, and never a silent zero that reads as "flat".
    """
    try:
        from ui.services import journal_feed
    except Exception as exc:  # noqa: BLE001 - the tab outlives a missing journal
        return JournalSnapshot(error=str(exc))

    reference = today or date.today()
    rows: dict[str, dict[str, Any]] = {}
    error = ""
    try:
        for status in ("OPEN", "CLOSED_PARTIAL"):
            for trade in journal_feed.load_trades(status=status):
                raw = dict(getattr(trade, "raw", None) or {})
                if raw.get("trade_id"):
                    rows[str(raw["trade_id"])] = raw
        for trade in journal_feed.load_trades(
            date_from=reference - timedelta(days=RECENT_TRADE_DAYS)
        ):
            raw = dict(getattr(trade, "raw", None) or {})
            if raw.get("trade_id"):
                rows.setdefault(str(raw["trade_id"]), raw)
    except Exception as exc:  # noqa: BLE001
        error = str(exc) or exc.__class__.__name__
        logger.debug("watchlist tab: journal read failed", exc_info=True)

    last_sync: dict[str, datetime] = {}
    try:
        for run in journal_feed.list_import_runs(limit=60):
            if str(run.get("status") or "").strip().upper() != _RUN_VERIFIED:
                continue
            source = str(run.get("source") or "").strip().upper()
            broker = next(
                (name for prefix, name in _RUN_BROKERS if source.startswith(prefix)), ""
            )
            if not broker:
                continue
            stamp = _parse_stamp(run.get("finished_at") or run.get("started_at"))
            if stamp is None:
                continue
            current = last_sync.get(broker)
            if current is None or stamp > current:
                last_sync[broker] = stamp
    except Exception:  # noqa: BLE001
        logger.debug("watchlist tab: import-run read failed", exc_info=True)

    return JournalSnapshot(
        trades=tuple(rows.values()),
        last_sync=last_sync,
        read_at=datetime.now().astimezone(),
        error=error,
    )


def gather(
    *,
    focus: Any = None,
    armed_alerts: Any = (),
    board_rows: Any = None,
    journal: JournalSnapshot | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Every argument `build_watchlist_rows` takes, read from the live stores.

    The four plain lists, the WS-5D intent stream, today's swing favorites and
    today's decisions. Each one degrades to empty on its own - a store that
    cannot be read costs its column, never the tab.
    """
    import project_paths
    import swing_favorites
    import watchlist_intent_events as intent

    shared: dict[str, list[str]] = {}
    try:
        from watchlist_utils import read_watchlist_symbols

        for name, path in (
            ("longs", project_paths.LONGS_FILE),
            ("shorts", project_paths.SHORTS_FILE),
            ("swinglongs", project_paths.SWING_LONGS_FILE),
            ("shortswings", project_paths.SWING_SHORTS_FILE),
        ):
            shared[name] = read_watchlist_symbols(path)
    except Exception:  # noqa: BLE001
        logger.debug("watchlist tab: shared lists unreadable", exc_info=True)

    try:
        events = intent.read_events(path=project_paths.WATCHLIST_INTENT_EVENTS_FILE)
    except Exception:  # noqa: BLE001
        events = []

    try:
        favorites = swing_favorites.favorites_for_session(
            path=project_paths.SWING_FAVORITES_FILE
        )
    except Exception:  # noqa: BLE001
        favorites = []

    try:
        import pick_feedback

        decisions = pick_feedback.decisions_today()
    except Exception:  # noqa: BLE001
        decisions = None

    snapshot = journal or JournalSnapshot()
    return {
        "shared_lists": shared,
        "focus_store": focus,
        "swing_favorites": favorites,
        "journal_exposures": snapshot.trades,
        "intent_events": events,
        "board_rows": board_rows or {},
        "decisions_today": decisions,
        "armed_alerts": tuple(armed_alerts or ()),
        "last_sync": snapshot.last_sync,
        "now": now,
    }


class WatchlistTabService(QObject):
    """One owner of the Watchlist tab's reading. Publishes rows; writes nothing."""

    rowsChanged = Signal(tuple)
    statusChanged = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        focus_service: Any = None,
        price_alert_service: Any = None,
        board_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._focus_service = focus_service
        self._price_alert_service = price_alert_service
        self._board_provider = board_provider
        self._rows: tuple[watchlist_views.WatchRow, ...] = ()
        self._journal = JournalSnapshot()
        self._running = False
        self._last_error = ""
        self._stamps: dict[str, float] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        start_staggered(self._timer, 7_000)

    # ------------------------------------------------------------------ reads
    def rows(self) -> tuple[watchlist_views.WatchRow, ...]:
        return self._rows

    def journal_snapshot(self) -> JournalSnapshot:
        """The last journal read. Empty until the first worker pass lands."""
        return self._journal

    def status_text(self) -> str:
        if self._running:
            return "Watchlist: reading..."
        if self._last_error:
            return f"Watchlist: last read failed ({self._last_error})"
        if self._journal.read_at is None:
            return "Watchlist: positions not read yet"
        return (
            f"Watchlist: {len(self._rows)} name(s); positions as of "
            f"{self._journal.read_at.strftime('%H:%M:%S')}"
        )

    def board_rows(self) -> Any:
        if self._board_provider is None:
            return {}
        try:
            return self._board_provider() or {}
        except Exception:  # noqa: BLE001
            return {}

    def focus_snapshot(self) -> Any:
        """Frozen on the QT thread - see the module docstring."""
        store = getattr(self._focus_service, "store", None)
        if store is None:
            return None
        try:
            return watchlist_views.focus_snapshot(store)
        except Exception:  # noqa: BLE001
            return None

    def armed_alerts(self) -> tuple:
        service = self._price_alert_service
        if service is not None:
            try:
                return tuple(service.entries())
            except Exception:  # noqa: BLE001
                return ()
        try:
            import price_alerts

            return tuple(price_alerts.load_price_alerts())
        except Exception:  # noqa: BLE001
            return ()

    # --------------------------------------------------------------- control
    def refresh_now(self) -> bool:
        """Start a full read on the worker. False when one is already running."""
        return self._start()

    def shutdown(self) -> None:
        stop_staggered(self._timer)

    # ------------------------------------------------------------------ timer
    def _tick(self) -> None:
        """A `stat()` on the four stores, and a read only when one moved."""
        try:
            if self._stores_changed():
                self._start()
        except Exception:  # noqa: BLE001
            logger.debug("watchlist tab tick failed", exc_info=True)

    def _stores_changed(self) -> bool:
        import project_paths

        paths = {
            "intent": project_paths.WATCHLIST_INTENT_EVENTS_FILE,
            "favorites": project_paths.SWING_FAVORITES_FILE,
            "alerts": project_paths.PRICE_ALERTS_FILE,
            "longs": project_paths.LONGS_FILE,
            "shorts": project_paths.SHORTS_FILE,
        }
        try:
            from ui.services import journal_feed

            paths["journal"] = journal_feed.journal_db_path()
        except Exception:  # noqa: BLE001
            pass
        changed = False
        for name, path in paths.items():
            try:
                stamp = float(path.stat().st_mtime)
            except OSError:
                stamp = 0.0
            if self._stamps.get(name) != stamp:
                self._stamps[name] = stamp
                changed = True
        return changed

    # ------------------------------------------------------------------- work
    def _start(self) -> bool:
        if self._running:
            return False
        self._running = True
        # Frozen HERE, on the Qt thread, before the worker exists.
        inputs = {
            "focus": self.focus_snapshot(),
            "armed_alerts": self.armed_alerts(),
            "board_rows": self.board_rows(),
        }
        threading.Thread(
            target=self._worker, args=(inputs,), name="watchlist-tab", daemon=True
        ).start()
        return True

    def _worker(self, inputs: dict[str, Any]) -> None:
        try:
            self._journal = read_journal()
            payload = gather(journal=self._journal, **inputs)
            rows = watchlist_views.build_watchlist_rows(**payload)
            self._rows = rows
            self._last_error = self._journal.error
            self.rowsChanged.emit(rows)
        except Exception as exc:  # noqa: BLE001
            # The last good rows survive a failed read (plan.md sec 5: a failed
            # publish never destroys the last verified report).
            self._last_error = str(exc) or exc.__class__.__name__
            logger.exception("watchlist tab read failed")
        finally:
            self._running = False
            self.statusChanged.emit(self.status_text())


def _parse_stamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.astimezone()
