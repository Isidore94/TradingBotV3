from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from focus_picks import FocusPickStore


class FocusService(QObject):
    """Qt adapter over the plain-Python `FocusPickStore`.

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
    def add(self, symbol, side) -> bool:
        return self._store.add(symbol, side)

    def add_many(self, symbols, side) -> list[str]:
        return self._store.add_many(symbols, side)

    def remove(self, symbol, side) -> bool:
        return self._store.remove(symbol, side)

    def clear(self, side) -> int:
        return self._store.clear(side)

    def reload(self) -> None:
        self._store.reload()
        self.focusChanged.emit()

    # ---- reads ----
    def focus_symbols(self, side) -> list[str]:
        return self._store.focus_symbols(side)

    def all_focus(self) -> dict[str, list[str]]:
        return self._store.all_focus()

    def is_focus(self, symbol, side=None) -> bool:
        return self._store.is_focus(symbol, side)

    def focus_side(self, symbol):
        return self._store.focus_side(symbol)
