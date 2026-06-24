"""Trader-curated daily Focus Picks store.

Single source of truth for the user's handpicked daily longs/shorts. Plain Python
(no Qt) so both the headless engine (`run_master`) and the Qt GUI can use it.

Responsibilities:
- Read/write `focus_longs.txt` / `focus_shorts.txt` (shared home).
- Add / paste / remove / clear with order-preserving de-dupe.
- Sync additions into the broad shared watchlists (`longs.txt` / `shorts.txt`) and
  remember *which* shared entries Focus Picks injected, so a later removal never
  deletes a symbol the user maintains independently in the broad watchlist.

See GUI_REDESIGN_PLAN.md (Focus Picks + Human Setup Tracker), Step 1.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from project_paths import (
    FOCUS_LONGS_FILE,
    FOCUS_PICK_MEMBERSHIP_FILE,
    FOCUS_SHORTS_FILE,
    LONGS_FILE,
    SHORTS_FILE,
)
from watchlist_utils import extract_watchlist_symbols, read_watchlist_symbols


def normalize_focus_side(side: object) -> str:
    """Map any side spelling to 'long' or 'short'. Raises ValueError otherwise."""
    text = str(side or "").strip().lower()
    if text in {"long", "l", "buy", "longs"} or text.startswith("long"):
        return "long"
    if text in {"short", "s", "sell", "shorts"} or text.startswith("short"):
        return "short"
    raise ValueError(f"Unrecognized focus side: {side!r}")


def normalize_symbol(symbol: object) -> str:
    """Return the single normalized ticker from arbitrary input, or ''."""
    symbols = extract_watchlist_symbols(str(symbol or ""))
    return symbols[0] if symbols else ""


def _membership_key(symbol: str, side: str) -> str:
    return f"{symbol}|{side}"


class FocusPickStore:
    def __init__(
        self,
        *,
        focus_longs_path: Path = FOCUS_LONGS_FILE,
        focus_shorts_path: Path = FOCUS_SHORTS_FILE,
        longs_path: Path = LONGS_FILE,
        shorts_path: Path = SHORTS_FILE,
        membership_path: Path = FOCUS_PICK_MEMBERSHIP_FILE,
    ) -> None:
        self._focus_paths = {"long": Path(focus_longs_path), "short": Path(focus_shorts_path)}
        self._shared_paths = {"long": Path(longs_path), "short": Path(shorts_path)}
        self._membership_path = Path(membership_path)
        self._lists: dict[str, list[str]] = {"long": [], "short": []}
        self._membership: dict[str, dict] = {}
        self._listeners: list[Callable[[], None]] = []
        self.reload()

    # ------------------------------------------------------------------ reads
    def reload(self) -> None:
        for side, path in self._focus_paths.items():
            self._lists[side] = read_watchlist_symbols(path)
        self._membership = self._load_membership()

    def focus_symbols(self, side: object) -> list[str]:
        return list(self._lists[normalize_focus_side(side)])

    def focus_longs(self) -> list[str]:
        return list(self._lists["long"])

    def focus_shorts(self) -> list[str]:
        return list(self._lists["short"])

    def all_focus(self) -> dict[str, list[str]]:
        return {"long": list(self._lists["long"]), "short": list(self._lists["short"])}

    def is_focus(self, symbol: object, side: object | None = None) -> bool:
        sym = normalize_symbol(symbol)
        if not sym:
            return False
        if side is None:
            return sym in self._lists["long"] or sym in self._lists["short"]
        return sym in self._lists[normalize_focus_side(side)]

    def focus_side(self, symbol: object) -> str | None:
        """Return 'long', 'short', 'both', or None for a symbol."""
        sym = normalize_symbol(symbol)
        is_long = sym in self._lists["long"]
        is_short = sym in self._lists["short"]
        if is_long and is_short:
            return "both"
        if is_long:
            return "long"
        if is_short:
            return "short"
        return None

    # ----------------------------------------------------------------- writes
    def add(self, symbol: object, side: object) -> bool:
        """Add one symbol to a focus side (+ inject into the shared watchlist).

        Returns True if the symbol was newly added, False if it was already there.
        """
        side = normalize_focus_side(side)
        sym = normalize_symbol(symbol)
        if not sym or sym in self._lists[side]:
            return False
        self._lists[side].append(sym)
        self._write_focus(side)
        self._inject_into_shared(sym, side)
        self._notify()
        return True

    def add_many(self, symbols: object, side: object) -> list[str]:
        """Add multiple symbols (e.g. a paste). Returns the newly added symbols."""
        side = normalize_focus_side(side)
        incoming = extract_watchlist_symbols(symbols) if isinstance(symbols, str) else [
            normalize_symbol(item) for item in (symbols or [])
        ]
        added: list[str] = []
        for sym in incoming:
            if sym and sym not in self._lists[side]:
                self._lists[side].append(sym)
                self._inject_into_shared(sym, side, defer_membership_save=True)
                added.append(sym)
        if added:
            self._write_focus(side)
            self._save_membership()
            self._notify()
        return added

    def remove(self, symbol: object, side: object) -> bool:
        """Remove a focus symbol; only un-inject the shared watchlist entry if we
        injected it (never delete an independently maintained broad-list symbol)."""
        side = normalize_focus_side(side)
        sym = normalize_symbol(symbol)
        if sym not in self._lists[side]:
            return False
        self._lists[side] = [item for item in self._lists[side] if item != sym]
        self._write_focus(side)
        self._uninject_from_shared(sym, side)
        self._notify()
        return True

    def clear(self, side: object) -> int:
        """Clear an entire focus side. Returns the number of symbols removed."""
        side = normalize_focus_side(side)
        symbols = list(self._lists[side])
        if not symbols:
            return 0
        for sym in symbols:
            self._uninject_from_shared(sym, side, defer_membership_save=True)
        self._save_membership()
        self._lists[side] = []
        self._write_focus(side)
        self._notify()
        return len(symbols)

    # -------------------------------------------------------------- observers
    def add_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                pass

    # ------------------------------------------------------- shared watchlist
    def _inject_into_shared(self, symbol: str, side: str, *, defer_membership_save: bool = False) -> None:
        shared_path = self._shared_paths[side]
        if symbol not in read_watchlist_symbols(shared_path):
            _append_symbol_to_file(shared_path, symbol)
            self._membership[_membership_key(symbol, side)] = {
                "symbol": symbol,
                "side": side,
                "shared_file": shared_path.name,
                "injected_at": datetime.now().isoformat(timespec="seconds"),
            }
            if not defer_membership_save:
                self._save_membership()

    def _uninject_from_shared(self, symbol: str, side: str, *, defer_membership_save: bool = False) -> None:
        key = _membership_key(symbol, side)
        if key not in self._membership:
            return  # we did not inject it; leave the broad watchlist untouched
        _remove_symbol_from_file(self._shared_paths[side], symbol)
        del self._membership[key]
        if not defer_membership_save:
            self._save_membership()

    # ------------------------------------------------------------- internals
    def _write_focus(self, side: str) -> None:
        _write_symbols(self._focus_paths[side], self._lists[side])

    def _load_membership(self) -> dict[str, dict]:
        if not self._membership_path.exists():
            return {}
        try:
            data = json.loads(self._membership_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_membership(self) -> None:
        try:
            self._membership_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._membership_path.with_name(self._membership_path.name + ".tmp")
            tmp.write_text(json.dumps(self._membership, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._membership_path)
        except OSError:
            # Cloud-synced folders can briefly lock files; membership is best-effort.
            pass

    def membership(self) -> dict[str, dict]:
        return dict(self._membership)


def load_focus_map(
    *,
    focus_longs_path: Path = FOCUS_LONGS_FILE,
    focus_shorts_path: Path = FOCUS_SHORTS_FILE,
) -> dict[str, set[str]]:
    """Lightweight read-only accessor for the engine: {'long': {...}, 'short': {...}}."""
    return {
        "long": set(read_watchlist_symbols(Path(focus_longs_path))),
        "short": set(read_watchlist_symbols(Path(focus_shorts_path))),
    }


def _write_symbols(path: Path, symbols: Iterable[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(symbols), encoding="utf-8")
    except OSError:
        pass


def _append_symbol_to_file(path: Path, symbol: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        separator = "" if (not existing or existing.endswith("\n")) else "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{separator}{symbol}\n")
    except OSError:
        pass


def _remove_symbol_from_file(path: Path, symbol: str) -> None:
    symbols = read_watchlist_symbols(path)
    if symbol not in symbols:
        return
    _write_symbols(path, [item for item in symbols if item != symbol])
