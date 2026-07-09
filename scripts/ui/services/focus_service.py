from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from focus_picks import FocusPickStore
from pick_feedback import record_pick_feedback


class FocusService(QObject):
    """Qt adapter over the plain-Python `FocusPickStore`.

    Focus picks live in two categories: "swing" (multi-day names the bot
    should learn from; synced into the swing watchlists) and "m5" (day-trade
    names; synced into the intraday longs/shorts watchlists).

    Every GUI verdict is also logged to the pick-feedback JSONL (likes with
    their origin, dislikes with the trader's reason, unfavorites) so an AI can
    review it later. Recording is skipped for custom-path stores (tests).

    Emits `focusChanged` on any mutation (the GUI rebuilds chips / markers off it).
    The store persists immediately on each edit, so this also provides autosave.
    """

    focusChanged = Signal()

    def __init__(self, store: FocusPickStore | None = None, parent=None) -> None:
        super().__init__(parent)
        self._store = store or FocusPickStore()
        self._store.add_listener(self.focusChanged.emit)

    @property
    def store(self) -> FocusPickStore:
        return self._store

    # ---- mutations (each triggers focusChanged via the store listener) ----
    def add(self, symbol, side, category="m5", *, origin="", context="") -> bool:
        added = self._store.add(symbol, side, category)
        if added:
            self.record_feedback(symbol, side, "like", category=category, origin=origin, context=context)
        return added

    def add_many(self, symbols, side, category="m5", *, origin="", context="") -> list[str]:
        added = self._store.add_many(symbols, side, category)
        for symbol in added:
            self.record_feedback(symbol, side, "like", category=category, origin=origin, context=context)
        return added

    def remove(self, symbol, side, category="m5") -> bool:
        return self._store.remove(symbol, side, category)

    def remove_everywhere(self, symbol, *, origin="", context="") -> int:
        removed = self._store.remove_everywhere(symbol)
        if removed:
            self.record_feedback(symbol, "", "unfavorite", origin=origin, context=context)
        return removed

    def clear(self, side, category="m5") -> int:
        return self._store.clear(side, category)

    def reload(self) -> None:
        self._store.reload()
        self.focusChanged.emit()

    # ---- feedback log (AI-reviewable pick_feedback.jsonl) ----
    def record_feedback(self, symbol, side, verdict, *, category="", origin="", reason="", context="") -> None:
        if not getattr(self._store, "uses_default_paths", lambda: False)():
            return
        try:
            record_pick_feedback(
                symbol=symbol,
                side=side,
                verdict=verdict,
                category=str(category or ""),
                origin=origin,
                reason=reason,
                context=context,
            )
        except Exception:
            pass  # the log is best-effort; never block a GUI action on it

    # ---- reads ----
    def focus_symbols(self, side, category=None) -> list[str]:
        return self._store.focus_symbols(side, category)

    def all_focus(self, category=None) -> dict[str, list[str]]:
        return self._store.all_focus(category)

    def all_focus_by_category(self) -> dict[str, dict[str, list[str]]]:
        return self._store.all_focus_by_category()

    def is_focus(self, symbol, side=None, category=None) -> bool:
        return self._store.is_focus(symbol, side, category)

    def focus_side(self, symbol, category=None):
        return self._store.focus_side(symbol, category)

    def focus_category(self, symbol):
        return self._store.focus_category(symbol)
