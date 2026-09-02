"""Qt adapter over the trader's hand-vetted swing picks.

Two writes per add, in a deliberate order:

1. **The Focus write-through** - the name enters the swing Focus category
   through the existing store, so every swing scan covers it and the desk
   treats it the way it treats any Focus swing pick. It goes in as the
   trader's own: no auto-adoption marker is written, because absence of a
   marker is precisely what makes an entry theirs and keeps every automatic
   removal path off it (plan.md sec 5, "Focus provenance").
2. **The evidence row** - append-only, in `swing_favorites.jsonl`. It is
   written second and its failure is swallowed, because an evidence store is
   never allowed to cost the thing it records. A lost row costs the record of
   the act; it must never cost the pick.

The "taken" mark is a read-only join against the TRADE journal (what the
trader traded), not the Market Journal (what they thought) - two stores,
deliberately not merged. It runs on a worker thread: the journal is sqlite on
a year of fills and `entries_for`-style reads never belong on a paint path
(ground rule 9). It reports nothing when the journal would have to be created
or migrated to answer - a display badge must never be the thing that triggers
a schema migration, and unmeasurable shows nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from PySide6.QtCore import QObject, QThread, Signal

import swing_favorites

#: The pick-feedback like origin these picks carry. It becomes the human-focus
#: tracker's sub-cohort suffix, so a hand-vetted swing pick grades separately
#: from every other manually added swing name.
FOCUS_LIKE_ORIGIN = "vetted"

#: The origin a retraction written by the Focus fade carries (A3). It is not
#: `ORIGIN_TRADER` because the trader did not do it, and a store whose rows all
#: claimed to be theirs could not answer "did I drop this, or did it time out?"
ORIGIN_FADE = "focus_fade"


def default_trades_provider(session_date: str, days: int) -> list[dict[str, Any]]:
    """Journal trades opened in the bounded window, or [] when unanswerable.

    Never initializes or migrates the journal database: the Journal panel owns
    that, in its own worker, behind the trader's own click.
    """
    try:
        from ui.services import journal_feed

        if not journal_feed.store_is_initialized() and journal_feed.store_needs_preparation():
            return []
        start = swing_favorites.taken_lookback_start(session_date, days=days)
        trades = journal_feed.load_trades(date_from=start, date_to=None)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Swing favorites could not read the journal: %s", exc)
        return []
    return [dict(getattr(trade, "raw", None) or {}) for trade in trades]


class _TakenWorker(QThread):
    """One journal read, off the GUI thread. It never raises into Qt."""

    done = Signal(object)

    def __init__(
        self,
        favorites: list[dict[str, Any]],
        session_date: str,
        days: int,
        provider: Callable[[str, int], Iterable[Mapping[str, Any]]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._favorites = favorites
        self._session_date = session_date
        self._days = days
        self._provider = provider

    def run(self) -> None:  # pragma: no cover - exercised through its seam
        try:
            trades = self._provider(self._session_date, self._days)
            marks = swing_favorites.taken_keys(self._favorites, trades)
        except Exception as exc:  # noqa: BLE001
            logging.debug("Swing favorites taken-join failed: %s", exc)
            marks = set()
        self.done.emit(marks)


class SwingFavoritesService(QObject):
    """Owns `swing_favorites.jsonl` and the swing Focus write-through."""

    #: The session's list changed (an add, a removal, or a day roll).
    favoritesChanged = Signal()
    #: The set of (symbol, side) with a matching journal trade. Display only.
    takenChanged = Signal(object)
    statusChanged = Signal(str)

    def __init__(
        self,
        focus_service=None,
        *,
        path: Path | None = None,
        trades_provider: Callable[[str, int], Iterable[Mapping[str, Any]]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._focus_service = focus_service
        self._path = Path(path) if path is not None else swing_favorites.SWING_FAVORITES_FILE
        self._trades_provider = trades_provider or default_trades_provider
        self._taken: set[tuple[str, str]] = set()
        self._worker: _TakenWorker | None = None

    # -- reads -------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    def session_date(self) -> str:
        return swing_favorites.current_session_date()

    def favorites(self, session_date: str = "") -> list[dict[str, Any]]:
        return swing_favorites.favorites_for_session(
            session_date or self.session_date(), path=self._path
        )

    def taken(self) -> set[tuple[str, str]]:
        return set(self._taken)

    # -- writes ------------------------------------------------------------
    def add(self, text: object, side: object) -> list[str]:
        """Place every symbol in `text` on `side`. Returns what was added.

        A name already on the session's list is not re-added and writes no
        second row - the strip is a list of today's picks, not a click log.
        """
        side_text = swing_favorites.normalize_side(side)
        if not side_text:
            return []
        session_date = self.session_date()
        already = {row["symbol"] for row in self.favorites(session_date) if row["side"] == side_text}
        added: list[str] = []
        for symbol in swing_favorites.parse_symbols(text):
            if symbol in already:
                continue
            self._place_in_focus(symbol, side_text)
            self._record(symbol, side_text, swing_favorites.ACTION_ADD, session_date)
            already.add(symbol)
            added.append(symbol)
        if added:
            self.favoritesChanged.emit()
            self.refresh_taken()
        return added

    def remove(self, symbol: object, side: object) -> bool:
        """Retract one pick: the Focus entry goes, a retraction row is appended.

        Nothing in the store is rewritten. The add row stays exactly where it
        was - "added and then thought better of it" is what happened.
        """
        sym = swing_favorites.normalize_symbol(symbol)
        side_text = swing_favorites.normalize_side(side)
        if not sym or not side_text:
            return False
        session_date = self.session_date()
        present = any(
            row["symbol"] == sym and row["side"] == side_text
            for row in self.favorites(session_date)
        )
        if not present:
            return False
        self._drop_from_focus(sym, side_text)
        self._record(sym, side_text, swing_favorites.ACTION_REMOVE, session_date)
        self.favoritesChanged.emit()
        return True

    def retract_faded_picks(self, faded: Iterable[Mapping[str, Any]]) -> int:
        """Append a RETRACTION row for every faded pick this store ever held.

        The Focus entry is already gone - the fade owns that - so this writes
        evidence only, and never an edit: "added on the 3rd, faded on the 17th"
        stays two rows in the order they happened, exactly as a hand removal
        does. A symbol this store never recorded is skipped; the fade covers
        every Focus pick and only some of them were ever hand-vetted here.
        """
        known = {
            (row.get("symbol"), row.get("side"))
            for row in swing_favorites.load_rows(self._path)
            if row.get("action") == swing_favorites.ACTION_ADD
        }
        written = 0
        for row in faded or ():
            if str(row.get("category") or "") != "swing":
                continue
            symbol = swing_favorites.normalize_symbol(row.get("symbol"))
            side = swing_favorites.normalize_side(row.get("side"))
            if not symbol or not side or (symbol, side) not in known:
                continue
            # TODAY's session, never the session it was added in: the
            # retraction is a thing that happened now, and an entry is never
            # backdated.
            recorded = swing_favorites.record_favorite(
                symbol=symbol,
                side=side,
                action=swing_favorites.ACTION_REMOVE,
                session_date=self.session_date(),
                origin=ORIGIN_FADE,
                path=self._path,
            )
            if recorded is not None:
                written += 1
        if written:
            self.favoritesChanged.emit()
        return written

    def _record(self, symbol: str, side: str, action: str, session_date: str) -> None:
        row = swing_favorites.record_favorite(
            symbol=symbol,
            side=side,
            action=action,
            session_date=session_date,
            path=self._path,
        )
        if row is None:
            # Said in the status line, never raised: the pick is already
            # placed and the trader's action stands.
            self.statusChanged.emit(f"{symbol} placed, but the favorites log was not written.")

    # -- Focus write-through ----------------------------------------------
    def _place_in_focus(self, symbol: str, side: str) -> None:
        if self._focus_service is None:
            return
        try:
            # `add` never writes an auto-adoption marker; only
            # `mark_auto_adopted` does, and nothing here calls it.
            #
            # The origin is "vetted" rather than "manual" so these grade as
            # their OWN sub-cohort in the human-focus tracker
            # (`human_focus_swing_vetted`) instead of disappearing into every
            # other hand-typed swing name. That is what makes "how do my
            # hand-picked swings do against the bot's?" a question the
            # existing 1/3/5/10-session grader can already answer.
            self._focus_service.add(symbol, side, "swing", origin=FOCUS_LIKE_ORIGIN)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Swing favorite %s not added to Focus: %s", symbol, exc)
            self.statusChanged.emit(f"{symbol} not added to swing Focus: {exc}")

    def _drop_from_focus(self, symbol: str, side: str) -> None:
        if self._focus_service is None:
            return
        try:
            self._focus_service.remove(symbol, side, "swing")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Swing favorite %s not removed from Focus: %s", symbol, exc)

    # -- taken join --------------------------------------------------------
    def refresh_taken(self, *, blocking: bool = False) -> None:
        """Ask the journal which of today's favorites were actually traded."""
        favorites = self.favorites()
        if not favorites:
            if self._taken:
                self._taken = set()
                self.takenChanged.emit(set())
            return
        session_date = self.session_date()
        days = swing_favorites.TAKEN_LOOKBACK_DAYS
        if blocking:
            try:
                marks = swing_favorites.taken_keys(
                    favorites, self._trades_provider(session_date, days)
                )
            except Exception as exc:  # noqa: BLE001
                logging.debug("Swing favorites taken-join failed: %s", exc)
                marks = set()
            self._on_taken(marks)
            return
        if self._worker is not None and self._worker.isRunning():
            return
        worker = _TakenWorker(favorites, session_date, days, self._trades_provider, self)
        worker.done.connect(self._on_taken)
        worker.finished.connect(self._release_worker)
        self._worker = worker
        worker.start()

    def _on_taken(self, marks) -> None:
        marks = set(marks or ())
        if marks == self._taken:
            return
        self._taken = marks
        self.takenChanged.emit(set(marks))

    def _release_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def shutdown(self, msecs: int = 2000) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(msecs)
