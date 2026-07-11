"""Trader-curated Focus Picks store, split into Swing and M5 categories.

Single source of truth for the user's handpicked names. Plain Python (no Qt) so
both the headless engine (`run_master`) and the Qt GUI can use it.

Categories:
- "swing" — multi-day picks (from D1/H1 bot output or anything that looks good).
  Synced into the Master AVWAP swing watchlists (`swinglongs.txt` /
  `shortswings.txt`) so every master scan covers them and the human-focus
  tracker can grade them over 1/3/5/10 sessions.
- "m5"    — day-trade picks. Synced into the broad intraday watchlists
  (`longs.txt` / `shorts.txt`) that BounceBot sweeps on M5. This is the
  original Focus Picks behavior, so pre-category files/membership just ARE
  the m5 category — no migration needed.

Responsibilities:
- Read/write the per-category focus files.
- Add / paste / remove / clear with order-preserving de-dupe.
- Sync additions into the matching shared watchlist and remember *which*
  shared entries Focus Picks injected, so a later removal never deletes a
  symbol the user maintains independently in the broad list.

See plan.md, Milestone 8 (Human focus lists).
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


FOCUS_CATEGORIES = ("swing", "m5")


def normalize_focus_side(side: object) -> str:
    """Map any side spelling to 'long' or 'short'. Raises ValueError otherwise."""
    text = str(side or "").strip().lower()
    if text in {"long", "l", "buy", "longs"} or text.startswith("long"):
        return "long"
    if text in {"short", "s", "sell", "shorts"} or text.startswith("short"):
        return "short"
    raise ValueError(f"Unrecognized focus side: {side!r}")


def normalize_focus_category(category: object, *, default: str = "m5") -> str:
    """Map any category spelling to 'swing' or 'm5'. Raises ValueError otherwise."""
    text = str(category or "").strip().lower()
    if not text:
        return default
    if text in {"swing", "swings", "d1", "h1", "daily", "multiday", "multi-day"}:
        return "swing"
    if text in {"m5", "5m", "dt", "day", "daytrade", "day-trade", "intraday"}:
        return "m5"
    raise ValueError(f"Unrecognized focus category: {category!r}")


def normalize_symbol(symbol: object) -> str:
    """Return the single normalized ticker from arbitrary input, or ''."""
    symbols = extract_watchlist_symbols(str(symbol or ""))
    return symbols[0] if symbols else ""


def _membership_key(symbol: str, side: str, category: str) -> str:
    # m5 keeps the pre-category "SYM|side" format so existing membership
    # files remain valid; swing entries carry an explicit category suffix.
    if category == "m5":
        return f"{symbol}|{side}"
    return f"{symbol}|{side}|{category}"


class FocusPickStore:
    def __init__(
        self,
        *,
        focus_longs_path: Path = FOCUS_LONGS_FILE,
        focus_shorts_path: Path = FOCUS_SHORTS_FILE,
        longs_path: Path = LONGS_FILE,
        shorts_path: Path = SHORTS_FILE,
        membership_path: Path = FOCUS_PICK_MEMBERSHIP_FILE,
        focus_swing_longs_path: Path | None = None,
        focus_swing_shorts_path: Path | None = None,
        swing_longs_path: Path | None = None,
        swing_shorts_path: Path | None = None,
    ) -> None:
        focus_longs_path = Path(focus_longs_path)
        focus_shorts_path = Path(focus_shorts_path)
        longs_path = Path(longs_path)
        shorts_path = Path(shorts_path)
        # Swing paths default to siblings of the m5 paths: in production that
        # resolves to the real shared-home files; with custom (test) paths it
        # keeps everything inside the same sandbox directory.
        self._focus_paths: dict[str, dict[str, Path]] = {
            "m5": {"long": focus_longs_path, "short": focus_shorts_path},
            "swing": {
                "long": Path(focus_swing_longs_path) if focus_swing_longs_path else focus_longs_path.with_name("focus_swing_longs.txt"),
                "short": Path(focus_swing_shorts_path) if focus_swing_shorts_path else focus_shorts_path.with_name("focus_swing_shorts.txt"),
            },
        }
        self._shared_paths: dict[str, dict[str, Path]] = {
            "m5": {"long": longs_path, "short": shorts_path},
            "swing": {
                "long": Path(swing_longs_path) if swing_longs_path else longs_path.with_name("swinglongs.txt"),
                "short": Path(swing_shorts_path) if swing_shorts_path else shorts_path.with_name("shortswings.txt"),
            },
        }
        self._membership_path = Path(membership_path)
        self._lists: dict[str, dict[str, list[str]]] = {
            category: {"long": [], "short": []} for category in FOCUS_CATEGORIES
        }
        self._membership: dict[str, dict] = {}
        self._listeners: list[Callable[[], None]] = []
        self.reload()

    # ------------------------------------------------------------------ reads
    def reload(self) -> None:
        for category in FOCUS_CATEGORIES:
            for side, path in self._focus_paths[category].items():
                self._lists[category][side] = read_watchlist_symbols(path)
        self._membership = self._load_membership()

    def focus_symbols(self, side: object, category: object = None) -> list[str]:
        """Symbols for a side; one category, or the swing-first union of both."""
        side = normalize_focus_side(side)
        if category is not None:
            return list(self._lists[normalize_focus_category(category)][side])
        combined: list[str] = []
        for cat in FOCUS_CATEGORIES:
            for sym in self._lists[cat][side]:
                if sym not in combined:
                    combined.append(sym)
        return combined

    def focus_longs(self) -> list[str]:
        return self.focus_symbols("long")

    def focus_shorts(self) -> list[str]:
        return self.focus_symbols("short")

    def all_focus(self, category: object = None) -> dict[str, list[str]]:
        return {
            "long": self.focus_symbols("long", category),
            "short": self.focus_symbols("short", category),
        }

    def all_focus_by_category(self) -> dict[str, dict[str, list[str]]]:
        return {category: self.all_focus(category) for category in FOCUS_CATEGORIES}

    def is_focus(self, symbol: object, side: object | None = None, category: object = None) -> bool:
        sym = normalize_symbol(symbol)
        if not sym:
            return False
        categories = (normalize_focus_category(category),) if category is not None else FOCUS_CATEGORIES
        sides = (normalize_focus_side(side),) if side is not None else ("long", "short")
        return any(sym in self._lists[cat][s] for cat in categories for s in sides)

    def focus_side(self, symbol: object, category: object = None) -> str | None:
        """Return 'long', 'short', 'both', or None for a symbol."""
        is_long = self.is_focus(symbol, "long", category)
        is_short = self.is_focus(symbol, "short", category)
        if is_long and is_short:
            return "both"
        if is_long:
            return "long"
        if is_short:
            return "short"
        return None

    def focus_category(self, symbol: object) -> str | None:
        """Return 'swing', 'm5', 'both', or None for a symbol."""
        in_swing = self.is_focus(symbol, category="swing")
        in_m5 = self.is_focus(symbol, category="m5")
        if in_swing and in_m5:
            return "both"
        if in_swing:
            return "swing"
        if in_m5:
            return "m5"
        return None

    # ----------------------------------------------------------------- writes
    def add(self, symbol: object, side: object, category: object = "m5") -> bool:
        """Add one symbol to a focus side (+ inject into the matching shared watchlist).

        Returns True if the symbol was newly added, False if it was already there.
        """
        side = normalize_focus_side(side)
        category = normalize_focus_category(category)
        sym = normalize_symbol(symbol)
        if not sym or sym in self._lists[category][side]:
            return False
        self._lists[category][side].append(sym)
        self._write_focus(side, category)
        self._inject_into_shared(sym, side, category)
        self._notify()
        return True

    def add_many(self, symbols: object, side: object, category: object = "m5") -> list[str]:
        """Add multiple symbols (e.g. a paste). Returns the newly added symbols."""
        side = normalize_focus_side(side)
        category = normalize_focus_category(category)
        incoming = extract_watchlist_symbols(symbols) if isinstance(symbols, str) else [
            normalize_symbol(item) for item in (symbols or [])
        ]
        added: list[str] = []
        for sym in incoming:
            if sym and sym not in self._lists[category][side]:
                self._lists[category][side].append(sym)
                self._inject_into_shared(sym, side, category, defer_membership_save=True)
                added.append(sym)
        if added:
            self._write_focus(side, category)
            self._save_membership()
            self._notify()
        return added

    def remove(self, symbol: object, side: object, category: object = "m5") -> bool:
        """Remove a focus symbol; only un-inject the shared watchlist entry if we
        injected it (never delete an independently maintained broad-list symbol)."""
        side = normalize_focus_side(side)
        category = normalize_focus_category(category)
        sym = normalize_symbol(symbol)
        if sym not in self._lists[category][side]:
            return False
        self._lists[category][side] = [item for item in self._lists[category][side] if item != sym]
        self._write_focus(side, category)
        self._uninject_from_shared(sym, side, category)
        self._notify()
        return True

    def remove_everywhere(self, symbol: object) -> int:
        """Unfavorite: drop a symbol from every category/side it appears in.

        Returns the number of list entries removed. Notifies once.
        """
        sym = normalize_symbol(symbol)
        if not sym:
            return 0
        removed = 0
        for category in FOCUS_CATEGORIES:
            for side in ("long", "short"):
                if sym not in self._lists[category][side]:
                    continue
                self._lists[category][side] = [item for item in self._lists[category][side] if item != sym]
                self._write_focus(side, category)
                self._uninject_from_shared(sym, side, category, defer_membership_save=True)
                removed += 1
        if removed:
            self._save_membership()
            self._notify()
        return removed

    def clear(self, side: object, category: object = "m5") -> int:
        """Clear one focus side of one category. Returns the number of symbols removed."""
        side = normalize_focus_side(side)
        category = normalize_focus_category(category)
        symbols = list(self._lists[category][side])
        if not symbols:
            return 0
        for sym in symbols:
            self._uninject_from_shared(sym, side, category, defer_membership_save=True)
        self._save_membership()
        self._lists[category][side] = []
        self._write_focus(side, category)
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
    def _inject_into_shared(self, symbol: str, side: str, category: str, *, defer_membership_save: bool = False) -> None:
        shared_path = self._shared_paths[category][side]
        if symbol not in read_watchlist_symbols(shared_path):
            _append_symbol_to_file(shared_path, symbol)
            self._membership[_membership_key(symbol, side, category)] = {
                "symbol": symbol,
                "side": side,
                "category": category,
                "shared_file": shared_path.name,
                "injected_at": datetime.now().isoformat(timespec="seconds"),
            }
            if not defer_membership_save:
                self._save_membership()

    def _uninject_from_shared(self, symbol: str, side: str, category: str, *, defer_membership_save: bool = False) -> None:
        key = _membership_key(symbol, side, category)
        if key not in self._membership:
            return  # we did not inject it; leave the broad watchlist untouched
        _remove_symbol_from_file(self._shared_paths[category][side], symbol)
        del self._membership[key]
        if not defer_membership_save:
            self._save_membership()

    # ------------------------------------------------------------- internals
    def _write_focus(self, side: str, category: str) -> None:
        _write_symbols(self._focus_paths[category][side], self._lists[category][side])

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

    def uses_default_paths(self) -> bool:
        return (
            self._focus_paths["m5"]["long"] == Path(FOCUS_LONGS_FILE)
            and self._focus_paths["m5"]["short"] == Path(FOCUS_SHORTS_FILE)
            and self._membership_path == Path(FOCUS_PICK_MEMBERSHIP_FILE)
        )


def load_focus_map(
    *,
    focus_longs_path: Path | None = None,
    focus_shorts_path: Path | None = None,
) -> dict[str, set[str]]:
    """Read-only union accessor for the engine: {'long': {...}, 'short': {...}}.

    With explicit paths it reads exactly those two files (legacy callers/tests).
    Without arguments it unions the swing and m5 focus lists so intraday code
    (BounceBot flagging, alert gold-framing) treats every liked name as focus.
    """
    if focus_longs_path is not None or focus_shorts_path is not None:
        return {
            "long": set(read_watchlist_symbols(Path(focus_longs_path or FOCUS_LONGS_FILE))),
            "short": set(read_watchlist_symbols(Path(focus_shorts_path or FOCUS_SHORTS_FILE))),
        }
    by_category = load_focus_maps_by_category()
    return {
        "long": set().union(*(by_category[cat]["long"] for cat in FOCUS_CATEGORIES)),
        "short": set().union(*(by_category[cat]["short"] for cat in FOCUS_CATEGORIES)),
    }


def load_focus_maps_by_category(
    *,
    focus_longs_path: Path = FOCUS_LONGS_FILE,
    focus_shorts_path: Path = FOCUS_SHORTS_FILE,
    focus_swing_longs_path: Path | None = None,
    focus_swing_shorts_path: Path | None = None,
) -> dict[str, dict[str, set[str]]]:
    """{'swing': {'long': {...}, 'short': {...}}, 'm5': {...}} straight from disk."""
    focus_longs_path = Path(focus_longs_path)
    focus_shorts_path = Path(focus_shorts_path)
    swing_longs = Path(focus_swing_longs_path) if focus_swing_longs_path else focus_longs_path.with_name("focus_swing_longs.txt")
    swing_shorts = Path(focus_swing_shorts_path) if focus_swing_shorts_path else focus_shorts_path.with_name("focus_swing_shorts.txt")
    return {
        "swing": {
            "long": set(read_watchlist_symbols(swing_longs)),
            "short": set(read_watchlist_symbols(swing_shorts)),
        },
        "m5": {
            "long": set(read_watchlist_symbols(focus_longs_path)),
            "short": set(read_watchlist_symbols(focus_shorts_path)),
        },
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
