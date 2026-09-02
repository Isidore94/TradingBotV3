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
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping

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


# R10.E: thin module-level shims. Imported lazily inside each one so
# `focus_picks` stays importable on a machine where the evidence modules are
# absent - the Focus store is the product and must not need them to run.
def _joined_event(**kwargs):
    from focus_membership_events import joined_event

    return joined_event(**kwargs)


def _left_event(**kwargs):
    from focus_membership_events import left_event

    return left_event(**kwargs)


def _expired_event(**kwargs):
    from focus_membership_events import expired_event

    return expired_event(**kwargs)


def _sessions_between(joined_at, left_at):
    from focus_membership_events import sessions_between

    return sessions_between(joined_at, left_at)


def _membership_key(symbol: str, side: str, category: str) -> str:
    # m5 keeps the pre-category "SYM|side" format so existing membership
    # files remain valid; swing entries carry an explicit category suffix.
    if category == "m5":
        return f"{symbol}|{side}"
    return f"{symbol}|{side}|{category}"


# --------------------------------------------------------------- m5 day stamp
# The m5 category is a DAY-TRADE list: picks belong to one calendar day and
# reset on the next (user rule 2026-07-29). The stamp sidecar rides next to
# focus_longs.txt so both the store (which physically clears + un-injects on
# a stale day) and the read-only engine accessors (which just exclude stale
# m5 names) agree on which day the current list belongs to.
def _m5_state_path_for(focus_longs_path: Path) -> Path:
    return Path(focus_longs_path).with_name("focus_m5_state.json")


# ------------------------------------------------------- auto-pick provenance
# The focus files are plain text, one ticker per line, and that format is not
# changing - the trader edits them by hand and every reader in the repo parses
# them that way. So provenance rides beside them in a sidecar (packet R2).
#
# It exists to make ONE invariant structural instead of aspirational: a name
# the trader typed is never removed by an automatic path (plan.md sec 5). With
# no per-entry origin, an auto-adopted pick and a hand-typed name are the same
# line in the same file, so no removal verb could be written safely at all.
# Absence of a marker means user-entered - the safe default, and the one every
# pre-R2 file gets for free.
#
# Day-scoped like the m5 list itself: markers clear with the picks they
# describe, so yesterday's marker can never authorize removing a name the
# trader typed this morning.
def _auto_pick_state_path_for(focus_longs_path: Path) -> Path:
    return Path(focus_longs_path).with_name("focus_auto_picks.json")


# ------------------------------------------------------------- the fade clock
# Phase 0.12 A3 (trader, 2026-09-01). A Focus list only means "the names I am
# watching" while something takes names off it. A pick that has fired no alert
# and printed no pullback event for ten trading days is not being watched; it
# is furniture, and it is what makes the list too long to read.
#
# So every entry carries a CLOCK. It starts at add time and is reset by
# activity - a fired Focus D1 flag, an armed-watch hit, or the trader's own
# "keep in Focus" on the review chart. Ten trading days without a reset and the
# pick moves to a FADED list rather than vanishing: the trader can restore it
# (fresh clock) or discard it, and either way an append-only row says what
# happened.
#
# Three deliberate choices:
#
# * **The clock is SESSIONS** (`market_calendar.trading_days_between`). A
#   long weekend is not two days of silence.
# * **Uncertainty never fades.** A stamp the calendar cannot reason about
#   keeps the pick. Every fade removes something the trader may have typed, so
#   this fails closed in the only direction that is safe.
# * **The removal is the store's own.** It goes through `_uninject_from_shared`
#   like every other removal, so a name the trader maintains in the broad
#   watchlist independently is untouched (plan.md sec 5).
#
# Fading a name the TRADER typed is normally forbidden. It is allowed here by
# an explicit trader authorization on 2026-09-01, and only here: no other
# automatic path gains the right, and `remove_if_auto_adopted` still refuses
# anything without a marker.
FADE_TRADING_DAYS = 10

#: Schema by NAME, never by number.
SCHEMA_FOCUS_FADE_EVENT = "focus_fade_event_v1"

EVENT_FADED = "focus_pick_faded"
EVENT_RESTORED = "focus_pick_restored"
EVENT_DISCARDED = "focus_pick_discarded"


def _atomic_json_write(path: Path, payload: object) -> None:
    """Best-effort atomic JSON write. A lost write costs a sidecar, never a pick."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".tmp")
        staged.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staged, target)
    except OSError:
        pass


def _pick_clock_path_for(focus_longs_path: Path) -> Path:
    return Path(focus_longs_path).with_name("focus_pick_clocks.json")


def _faded_path_for(focus_longs_path: Path) -> Path:
    return Path(focus_longs_path).with_name("focus_faded.json")


def _fade_events_path_for(focus_longs_path: Path) -> Path:
    return Path(focus_longs_path).with_name("focus_fade_events.jsonl")


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _sessions_since(start: date, today: date) -> int | None:
    """Trading days between two dates, or None when the calendar refuses."""
    try:
        from market_calendar import trading_days_between

        return trading_days_between(start, today)
    except Exception as exc:  # SessionCalendarError, or the module unavailable
        logging.debug("Focus fade could not date %s: %s", start, exc)
        return None


def _read_m5_market_date(path: Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    stamp = str(payload.get("market_date") or "").strip()
    return stamp or None


def _write_m5_market_date(path: Path, date_text: str) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".tmp")
        staged.write_text(
            json.dumps({"market_date": date_text}, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staged, target)
    except OSError:
        # Another process or AV scan can briefly lock files; the stamp is best-effort
        # (a missed write just delays the reset to the next successful one).
        pass


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
        self._m5_state_path = _m5_state_path_for(focus_longs_path)
        self._auto_pick_path = _auto_pick_state_path_for(focus_longs_path)
        self._auto_picks: dict[str, dict] = {}
        # A3: the fade clock and the faded list. This store is their single
        # writer - the panel asks, it decides and writes.
        self._pick_clock_path = _pick_clock_path_for(focus_longs_path)
        self._faded_path = _faded_path_for(focus_longs_path)
        self._fade_events_path = _fade_events_path_for(focus_longs_path)
        self._pick_clocks: dict[str, dict] = {}
        self._faded: list[dict] = []
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
        self._auto_picks = self._load_auto_picks()
        self._pick_clocks = self._load_pick_clocks()
        self._faded = self._load_faded()
        # m5 is a day-trade list: a new calendar day starts it empty. This
        # runs on every store construction (GUI start each morning), so the
        # clear also physically un-injects yesterday's picks from
        # longs/shorts.txt before the first scan reads them.
        self.expire_m5_if_new_day()
        self._sync_pick_clocks()

    def expire_m5_if_new_day(self, today: date | None = None) -> int:
        """Reset the m5 (day-trade) lists when the calendar day has rolled.

        Swing picks are untouched - they are multi-day by definition. A
        missing stamp (first run after the upgrade) is grandfathered: the
        current list is stamped as today's rather than guessed stale.
        Returns the number of picks cleared.
        """
        today_text = (today or date.today()).isoformat()
        stamp = _read_m5_market_date(self._m5_state_path)
        if stamp == today_text:
            return 0
        if stamp is None:
            _write_m5_market_date(self._m5_state_path, today_text)
            return 0
        removed = 0
        # R10.E / F5: ONE ROW PER NAME, never a single "cleared N". A survivor
        # is 49% of (symbol, side) pairs in the snapshot store and nobody can
        # tell a survivor from a re-add - which is exactly what a per-name
        # expiry row makes answerable, and what a count would keep invisible.
        expired_at = datetime.now().astimezone().isoformat(timespec="seconds")
        for side in ("long", "short"):
            for sym in list(self._lists["m5"][side]):
                owner = self._membership_owner(sym, side, "m5")
                joined_at = self._episode_started_at(sym, side, "m5")
                self._uninject_from_shared(sym, side, "m5", defer_membership_save=True)
                self._forget_pick_clock(sym, side, "m5", defer_save=True)
                self._emit_membership(
                    _expired_event(
                        symbol=sym,
                        side=side,
                        category="m5",
                        owner=owner,
                        episode="",
                        joined_at=joined_at,
                        at=expired_at,
                    )
                )
                removed += 1
            self._lists["m5"][side] = []
            self._write_focus(side, "m5")
        if removed:
            self._save_membership()
            self._save_pick_clocks()
        # The markers describe picks that no longer exist. Keeping them would
        # let yesterday's provenance authorize removing a name the trader types
        # this morning, which is the one thing the sidecar exists to prevent.
        if self._auto_picks:
            self._auto_picks = {}
            self._save_auto_picks()
        _write_m5_market_date(self._m5_state_path, today_text)
        if removed:
            self._notify()
        return removed

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
    def add(
        self,
        symbol: object,
        side: object,
        category: object = "m5",
        *,
        today: date | None = None,
    ) -> bool:
        """Add one symbol to a focus side (+ inject into the matching shared watchlist).

        Returns True if the symbol was newly added, False if it was already there.
        """
        side = normalize_focus_side(side)
        category = normalize_focus_category(category)
        sym = normalize_symbol(symbol)
        if not sym or sym in self._lists[category][side]:
            return False
        self._lists[category][side].append(sym)
        self._start_pick_clock(sym, side, category, today=today, reason="added")
        self._write_focus(side, category)
        self._inject_into_shared(sym, side, category)
        self._emit_membership(
            _joined_event(
                symbol=sym,
                side=side,
                category=category,
                owner=self._membership_owner(sym, side, category),
                joined_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                origin="focus_store.add",
            )
        )
        if category == "m5":
            # The pick belongs to today; keeps the day stamp honest even in a
            # session that crossed midnight since the last reload.
            _write_m5_market_date(self._m5_state_path, date.today().isoformat())
        self._notify()
        return True

    def add_many(
        self,
        symbols: object,
        side: object,
        category: object = "m5",
        *,
        today: date | None = None,
    ) -> list[str]:
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
                self._start_pick_clock(
                    sym, side, category, today=today, reason="added", defer_save=True
                )
                self._inject_into_shared(sym, side, category, defer_membership_save=True)
                added.append(sym)
        if added:
            self._write_focus(side, category)
            self._save_membership()
            self._save_pick_clocks()
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            for sym in added:
                self._emit_membership(
                    _joined_event(
                        symbol=sym,
                        side=side,
                        category=category,
                        owner=self._membership_owner(sym, side, category),
                        joined_at=stamp,
                        origin="focus_store.add_many",
                    )
                )
            if category == "m5":
                _write_m5_market_date(self._m5_state_path, date.today().isoformat())
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
        owner = self._membership_owner(sym, side, category)
        joined_at = self._episode_started_at(sym, side, category)
        self._lists[category][side] = [item for item in self._lists[category][side] if item != sym]
        self._write_focus(side, category)
        self._uninject_from_shared(sym, side, category)
        self._forget_auto_marker(sym, side, category)
        self._forget_pick_clock(sym, side, category)
        left_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._emit_membership(
            _left_event(
                symbol=sym,
                side=side,
                category=category,
                owner=owner,
                episode="",
                joined_at=joined_at,
                left_at=left_at,
                reason="focus_store.remove",
                sessions_on_list=_sessions_between(joined_at, left_at),
            )
        )
        self._notify()
        return True

    # -------------------------------------------------- auto-pick provenance
    def mark_auto_adopted(
        self,
        symbol: object,
        side: object,
        category: object = "m5",
        *,
        staged_at: str = "",
        reason: str = "",
    ) -> None:
        """Record that THIS store adopted a pick automatically.

        Called only by the auto-adoption path. Nothing else may write a marker:
        the marker is what makes an entry removable by an automatic verb, so
        handing it out freely would defeat the invariant it protects.
        """
        side = normalize_focus_side(side)
        category = normalize_focus_category(category)
        sym = normalize_symbol(symbol)
        if not sym:
            return
        self._auto_picks[_membership_key(sym, side, category)] = {
            "symbol": sym,
            "side": side,
            "category": category,
            "session_date": date.today().isoformat(),
            "staged_at": str(staged_at or ""),
            "reason": str(reason or ""),
            "adopted_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_auto_picks()

    def auto_pick_marker(
        self, symbol: object, side: object, category: object = "m5"
    ) -> dict | None:
        """The marker for one entry, or None when the trader owns it."""
        sym = normalize_symbol(symbol)
        if not sym:
            return None
        try:
            side = normalize_focus_side(side)
            category = normalize_focus_category(category)
        except ValueError:
            return None
        marker = self._auto_picks.get(_membership_key(sym, side, category))
        return dict(marker) if isinstance(marker, dict) else None

    def is_auto_adopted(self, symbol: object, side: object, category: object = "m5") -> bool:
        """True only for an entry this store adopted automatically.

        A missing marker reads as user-entered. That is the safe default and the
        one every focus file written before packet R2 gets for free.
        """
        return self.auto_pick_marker(symbol, side, category) is not None

    def remove_if_auto_adopted(
        self, symbol: object, side: object, category: object = "m5"
    ) -> bool:
        """Scoped removal of ONE auto-adopted entry. True when it was removed.

        Refuses anything without a marker, so no automatic path and no
        "Not today" verb can reach a name the trader typed. Deliberately not
        `remove_everywhere`: the trader's rule is that this touches exactly the
        one M5 entry on that one side, never the swing entry and never the
        other side.
        """
        if not self.is_auto_adopted(symbol, side, category):
            return False
        return self.remove(symbol, side, category)

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
                self._forget_auto_marker(sym, side, category, defer_save=True)
                self._forget_pick_clock(sym, side, category, defer_save=True)
                removed += 1
        if removed:
            self._save_membership()
            self._save_auto_picks()
            self._save_pick_clocks()
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
            self._forget_auto_marker(sym, side, category, defer_save=True)
            self._forget_pick_clock(sym, side, category, defer_save=True)
        self._save_membership()
        self._save_auto_picks()
        self._save_pick_clocks()
        self._lists[category][side] = []
        self._write_focus(side, category)
        self._notify()
        return len(symbols)

    # ------------------------------------------------------ the fade clock A3
    def pick_clock(self, symbol: object, side: object, category: object = "m5") -> date | None:
        """The date this pick's ten sessions are counted from, or None.

        It is the later of "when it was added" and "when it last said
        something" - the same field, moved forward by activity.
        """
        sym = normalize_symbol(symbol)
        if not sym:
            return None
        try:
            key = _membership_key(
                sym, normalize_focus_side(side), normalize_focus_category(category)
            )
        except ValueError:
            return None
        record = self._pick_clocks.get(key)
        return _as_date(record.get("clock_from")) if isinstance(record, dict) else None

    def note_focus_activity(
        self,
        symbol: object,
        side: object | None = None,
        category: object | None = None,
        *,
        reason: str = "",
        today: date | None = None,
    ) -> int:
        """This pick said something. Restart its clock. Returns entries moved.

        Side and category are optional because the callers that have news -
        a fired Focus D1 flag, an armed-watch hit - know the SYMBOL and often
        nothing more. With neither given, every entry for that symbol resets,
        which is the honest reading: the name spoke.
        """
        sym = normalize_symbol(symbol)
        if not sym:
            return 0
        sides = ("long", "short") if side is None else (normalize_focus_side(side),)
        categories = (
            FOCUS_CATEGORIES if category is None else (normalize_focus_category(category),)
        )
        stamp = (today or date.today()).isoformat()
        reason_text = str(reason or "activity")
        moved = 0
        for cat in categories:
            for side_key in sides:
                if sym not in self._lists[cat][side_key]:
                    continue
                record = self._pick_clocks.setdefault(
                    _membership_key(sym, side_key, cat),
                    {"symbol": sym, "side": side_key, "category": cat},
                )
                if record.get("clock_from") == stamp and record.get("reason") == reason_text:
                    continue
                record["clock_from"] = stamp
                record["reason"] = reason_text
                moved += 1
        if moved:
            self._save_pick_clocks()
        return moved

    def fade_stale_picks(
        self,
        *,
        today: date | None = None,
        trading_days: int = FADE_TRADING_DAYS,
    ) -> list[dict]:
        """Move every pick that has been silent too long to the faded list.

        Returns one row per faded entry (already written to the evidence
        stream). A pick whose clock the calendar cannot read is KEPT - see the
        module comment on the fade clock.
        """
        reference = today or date.today()
        faded: list[dict] = []
        for category in FOCUS_CATEGORIES:
            for side in ("long", "short"):
                for sym in list(self._lists[category][side]):
                    key = _membership_key(sym, side, category)
                    record = self._pick_clocks.get(key) or {}
                    start = _as_date(record.get("clock_from"))
                    if start is None:
                        # No readable clock: stamp it today rather than fade a
                        # pick whose age nobody knows.
                        self._start_pick_clock(
                            sym,
                            side,
                            category,
                            today=reference,
                            reason="repaired",
                            defer_save=True,
                        )
                        continue
                    elapsed = _sessions_since(start, reference)
                    if elapsed is None or elapsed < int(trading_days):
                        continue
                    row = {
                        "schema": SCHEMA_FOCUS_FADE_EVENT,
                        "event": EVENT_FADED,
                        "symbol": sym,
                        "side": side,
                        "category": category,
                        "owner": self._membership_owner(sym, side, category),
                        "clock_from": start.isoformat(),
                        "clock_reason": str(record.get("reason") or "added"),
                        "faded_on": reference.isoformat(),
                        "trading_days": int(trading_days),
                        "sessions_silent": int(elapsed),
                    }
                    self._fade_one(sym, side, category, row)
                    faded.append(row)
        self._save_pick_clocks()
        if faded:
            self._save_faded()
            self._notify()
        return faded

    def faded_picks(self) -> list[dict]:
        """The faded list, newest last. Display and the faded review read this."""
        return [dict(row) for row in self._faded]

    def restore_faded(
        self,
        symbol: object,
        side: object,
        category: object = "m5",
        *,
        today: date | None = None,
    ) -> bool:
        """Put a faded pick back in Focus with a FRESH clock.

        A restore is not a fade-proof: it buys another full window, no more.
        """
        row = self._take_faded(symbol, side, category)
        if row is None:
            return False
        self.add(row["symbol"], row["side"], row["category"], today=today)
        self._append_fade_event(
            {
                **row,
                "event": EVENT_RESTORED,
                "restored_on": (today or date.today()).isoformat(),
            }
        )
        self._save_faded()
        self._notify()
        return True

    def discard_faded(self, symbol: object, side: object, category: object = "m5") -> bool:
        """Clear one entry off the faded list. The evidence row stays."""
        row = self._take_faded(symbol, side, category)
        if row is None:
            return False
        self._append_fade_event(
            {**row, "event": EVENT_DISCARDED, "discarded_on": date.today().isoformat()}
        )
        self._save_faded()
        self._notify()
        return True

    # -- fade internals ---------------------------------------------------
    def _fade_one(self, sym: str, side: str, category: str, row: dict) -> None:
        owner = row.get("owner", "")
        joined_at = self._episode_started_at(sym, side, category)
        self._lists[category][side] = [
            item for item in self._lists[category][side] if item != sym
        ]
        self._write_focus(side, category)
        self._uninject_from_shared(sym, side, category)
        self._forget_auto_marker(sym, side, category)
        self._forget_pick_clock(sym, side, category, defer_save=True)
        left_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._emit_membership(
            _left_event(
                symbol=sym,
                side=side,
                category=category,
                owner=owner,
                episode="",
                joined_at=joined_at,
                left_at=left_at,
                reason="focus_fade",
                sessions_on_list=_sessions_between(joined_at, left_at),
            )
        )
        self._faded = [
            item
            for item in self._faded
            if not (
                item.get("symbol") == sym
                and item.get("side") == side
                and item.get("category") == category
            )
        ]
        self._faded.append(dict(row))
        self._append_fade_event(row)

    def _take_faded(self, symbol: object, side: object, category: object) -> dict | None:
        sym = normalize_symbol(symbol)
        if not sym:
            return None
        try:
            side_key = normalize_focus_side(side)
            cat = normalize_focus_category(category)
        except ValueError:
            return None
        kept: list[dict] = []
        found: dict | None = None
        for item in self._faded:
            if (
                found is None
                and item.get("symbol") == sym
                and item.get("side") == side_key
                and item.get("category") == cat
            ):
                found = dict(item)
                continue
            kept.append(item)
        if found is None:
            return None
        self._faded = kept
        return found

    def _start_pick_clock(
        self,
        sym: str,
        side: str,
        category: str,
        *,
        today: date | None = None,
        reason: str = "added",
        defer_save: bool = False,
    ) -> None:
        self._pick_clocks[_membership_key(sym, side, category)] = {
            "symbol": sym,
            "side": side,
            "category": category,
            "clock_from": (today or date.today()).isoformat(),
            "reason": str(reason or "added"),
        }
        if not defer_save:
            self._save_pick_clocks()

    def _forget_pick_clock(
        self, sym: str, side: str, category: str, *, defer_save: bool = False
    ) -> None:
        if self._pick_clocks.pop(_membership_key(sym, side, category), None) is None:
            return
        if not defer_save:
            self._save_pick_clocks()

    def _sync_pick_clocks(self, today: date | None = None) -> None:
        """Every current pick has a clock; no clock outlives its pick.

        A pick already on the list when this shipped gets TODAY. Guessing
        backwards would fade the trader's whole list on the first slow tick,
        which is exactly the failure the "never guess older" rule exists for.
        """
        wanted: set[str] = set()
        changed = False
        for category in FOCUS_CATEGORIES:
            for side in ("long", "short"):
                for sym in self._lists[category][side]:
                    key = _membership_key(sym, side, category)
                    wanted.add(key)
                    if key not in self._pick_clocks:
                        self._start_pick_clock(
                            sym,
                            side,
                            category,
                            today=today,
                            reason="present_at_upgrade",
                            defer_save=True,
                        )
                        changed = True
        for key in list(self._pick_clocks):
            if key not in wanted:
                self._pick_clocks.pop(key, None)
                changed = True
        if changed:
            self._save_pick_clocks()

    def _load_pick_clocks(self) -> dict[str, dict]:
        try:
            payload = json.loads(self._pick_clock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        picks = payload.get("picks") if isinstance(payload, dict) else None
        if not isinstance(picks, dict):
            return {}
        return {
            str(key): dict(value) for key, value in picks.items() if isinstance(value, dict)
        }

    def _save_pick_clocks(self) -> None:
        _atomic_json_write(self._pick_clock_path, {"picks": self._pick_clocks})

    def _load_faded(self) -> list[dict]:
        try:
            payload = json.loads(self._faded_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        rows = payload.get("picks") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict) and row.get("symbol")]

    def _save_faded(self) -> None:
        _atomic_json_write(self._faded_path, {"picks": self._faded})

    def _append_fade_event(self, row: Mapping[str, object]) -> None:
        """One append-only row. Never blocks and never raises into the fade.

        Same rule as every other evidence store here: a failed append costs the
        record, never the thing it records.
        """
        payload = dict(row)
        payload.setdefault("schema", SCHEMA_FOCUS_FADE_EVENT)
        payload["event_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            self._fade_events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._fade_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            logging.info("Focus fade event not written for %s", payload.get("symbol"))

    # -------------------------------------------------------------- observers
    def add_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    # ------------------------------------------------- membership provenance
    def _emit_membership(self, event: dict) -> None:
        """Append one membership event. R10.E.

        **Never blocks and never raises into the Focus write.** The trader's
        list is the product; this is evidence about it, and a store that could
        cost a pick would be worse than no store. A failed append loses the
        event and says so in the log.
        """
        try:
            from evidence_ledger import EvidenceLedger
            from focus_membership_events import (
                SCHEMA_FOCUS_MEMBERSHIP_EVENT,
                STREAM,
            )

            EvidenceLedger(
                stream=STREAM, schema=SCHEMA_FOCUS_MEMBERSHIP_EVENT
            ).append(event)
        except Exception:
            logging.debug("Focus membership event not recorded.", exc_info=True)

    def _membership_owner(self, symbol: str, side: str, category: str) -> str:
        """Who owns this pick, from the auto-pick markers (R10.E / F4).

        Absence of a marker in a store that HAS markers means the trader owns
        it. Absence in a store with NO markers at all is `unknown_legacy`,
        because F4 measured that `focus_auto_picks.json` exists for no
        historical date - calling those picks the trader's would invent
        provenance the system never recorded.
        """
        from focus_membership_events import owner_for

        marker = (self._auto_picks or {}).get(_membership_key(symbol, side, category))
        return owner_for(marker, markers_present=bool(self._auto_picks))

    def _episode_started_at(self, symbol: str, side: str, category: str) -> str:
        """When this episode began, from the membership record if it has one."""
        record = (self._membership or {}).get(_membership_key(symbol, side, category))
        if isinstance(record, dict):
            stamp = str(record.get("joined_at") or record.get("added_at") or "")
            if stamp:
                return stamp
        return date.today().isoformat()

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
            # Another process or AV scan can briefly lock files; membership is best-effort.
            pass

    def _forget_auto_marker(
        self, symbol: str, side: str, category: str, *, defer_save: bool = False
    ) -> None:
        """A marker outlives nothing. Once the entry is gone the marker goes
        with it, so re-adding the name by hand starts it as the trader's."""
        if self._auto_picks.pop(_membership_key(symbol, side, category), None) is None:
            return
        if not defer_save:
            self._save_auto_picks()

    def _load_auto_picks(self, today: date | None = None) -> dict[str, dict]:
        """Markers for TODAY only, validated per entry (R2.1).

        The day-roll clears markers when it runs, but it only runs when the
        store notices the date changed - and the sidecar is a plain file that
        can outlive that: restored from a backup, edited by hand, or written by
        a process that died before `expire_m5_if_new_day` fired. A marker from
        another session must never be trusted, because a stale marker is
        precisely a licence to delete a name the trader typed this morning.

        Each entry carries its own `session_date`, so validation happens per
        marker rather than on the file's header alone: a half-rolled file
        cannot smuggle yesterday's entries in under today's stamp.
        """
        try:
            payload = json.loads(self._auto_pick_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        picks = payload.get("picks")
        if not isinstance(picks, dict):
            return {}
        today_text = (today or date.today()).isoformat()
        kept: dict[str, dict] = {}
        dropped = 0
        for key, marker in picks.items():
            if not isinstance(marker, dict):
                dropped += 1
                continue
            if str(marker.get("session_date") or "") != today_text:
                dropped += 1
                continue
            kept[str(key)] = marker
        if dropped:
            # Loud, because a marker disappearing changes who owns an entry.
            # Silently dropping them would look identical to the trader having
            # typed those names, which is the safe direction but not one to
            # take quietly.
            logging.info(
                "Focus provenance: ignored %d marker(s) not from today's session (%s).",
                dropped,
                today_text,
            )
        return kept

    def _save_auto_picks(self) -> None:
        try:
            self._auto_pick_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._auto_pick_path.with_name(self._auto_pick_path.name + ".tmp")
            tmp.write_text(
                json.dumps(
                    {"market_date": date.today().isoformat(), "picks": self._auto_picks},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._auto_pick_path)
        except OSError:
            # Best-effort, like the membership file: a lost write costs a
            # marker, and a lost marker means the entry reads as the trader's -
            # which is the safe direction to fail.
            pass

    def auto_pick_markers(self) -> dict[str, dict]:
        return dict(self._auto_picks)

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
    today: date | None = None,
) -> dict[str, dict[str, set[str]]]:
    """{'swing': {'long': {...}, 'short': {...}}, 'm5': {...}} straight from disk.

    m5 is a day-trade list: when its stamp sidecar says the files belong to an
    earlier day, the stale names are excluded here (read-only - the physical
    clear happens when a FocusPickStore next loads). A missing stamp includes
    the files unchanged, matching the store's grandfathering.
    """
    focus_longs_path = Path(focus_longs_path)
    focus_shorts_path = Path(focus_shorts_path)
    swing_longs = Path(focus_swing_longs_path) if focus_swing_longs_path else focus_longs_path.with_name("focus_swing_longs.txt")
    swing_shorts = Path(focus_swing_shorts_path) if focus_swing_shorts_path else focus_shorts_path.with_name("focus_swing_shorts.txt")
    stamp = _read_m5_market_date(_m5_state_path_for(focus_longs_path))
    m5_stale = stamp is not None and stamp != (today or date.today()).isoformat()
    return {
        "swing": {
            "long": set(read_watchlist_symbols(swing_longs)),
            "short": set(read_watchlist_symbols(swing_shorts)),
        },
        "m5": {
            "long": set() if m5_stale else set(read_watchlist_symbols(focus_longs_path)),
            "short": set() if m5_stale else set(read_watchlist_symbols(focus_shorts_path)),
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
