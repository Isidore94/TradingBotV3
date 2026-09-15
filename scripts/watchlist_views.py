"""ONE watchlist row set for the Trading Desk's Watchlist tab (WS-WL, WISHLIST 10G).

The desk keeps the same name in five places and every one of them is a
different owner: the four plain watchlist files (the trader's own typing, and
what the scanners read), ``FocusPickStore`` (what the desk is watching today),
the M5 strength board (what the machine offered), ``swing_favorites`` (today's
swing picks) and the Journal (what the broker says is actually ON). WISHLIST
10G asks for those to be readable as ONE list with views, and this module is
the reading.

**Pure.** No store is opened here, nothing is written, no network is reached
and no clock is read unless the caller hands one in. Every input is passed in
by the caller - which is what lets the Qt service build the whole thing on a
worker thread and lets the tests build it without a desk.

Three rules this file exists to keep:

``presence is not authorship``
    ``FocusPickStore.add`` INJECTS its pick into ``longs.txt`` / ``shorts.txt``
    (``_inject_into_shared``), so "the name is on the shared list" cannot mean
    "the trader typed it". Authorship is decided by the WS-5D intent stream:
    a name is ``manual`` unless the newest ``add`` the stream holds for that
    (list, symbol) was written by a machine. A name the stream has never seen
    predates the stream and counts as the trader's, with a BLANK ``first_seen``
    rather than an invented one.

``a source is a badge, never a priority``
    Rows sort by symbol (then side) inside every view, never by where they came
    from - decision 0016 puts names before entries. The views are FILTERS over
    one row set; ``all`` is the union and loses nothing.

``a position is a read-only projection, and uncertainty never deletes``
    A journal position never arms an alert, never adds a name to a list and
    never removes one. A sync older than the previous session's close is STALE:
    the row is kept and labelled. A CLOSED position leaves the list only when
    the sync that says so is verified-fresh; a stale close is still shown.

Row identity is ``(symbol, side)`` - "one symbol may appear once per side" -
which is why ``horizons`` is a frozenset rather than one value: the same name
can sit on ``longs.txt`` and ``swinglongs.txt`` at once, and neither reading is
wrong. ``positions`` is a tuple for the same reason: the trader has four
accounts and folding two of them into one row would either hide an account or
invent a total nobody holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import journal_exposure
import market_calendar
import watchlist_intent_events as intent

# --------------------------------------------------------------------------
# Sources. Six of these are the packet's; `alert` is the seventh and it is a
# DEPARTURE, recorded here: the price-alert board retired with the Focus Picks
# page, and an armed alert on a name that is on no list at all would otherwise
# have no row and no badge - the entry would still be in the store, still
# polling, and invisible. It is a badge like every other one and it ranks
# nothing.
# --------------------------------------------------------------------------
SOURCE_MANUAL = "manual"
SOURCE_FOCUS_TRADER = "focus_trader"
SOURCE_FOCUS_AUTO = "focus_auto"
SOURCE_BOARD = "board"
SOURCE_SWING_FAVORITE = "swing_favorite"
SOURCE_POSITION = "position"
SOURCE_ALERT = "alert"

SOURCES = (
    SOURCE_MANUAL,
    SOURCE_FOCUS_TRADER,
    SOURCE_FOCUS_AUTO,
    SOURCE_BOARD,
    SOURCE_SWING_FAVORITE,
    SOURCE_POSITION,
    SOURCE_ALERT,
)

#: How a badge reads on the tab. Display only.
SOURCE_LABELS = {
    SOURCE_MANUAL: "mine",
    SOURCE_FOCUS_TRADER: "focus",
    SOURCE_FOCUS_AUTO: "auto",
    SOURCE_BOARD: "board",
    SOURCE_SWING_FAVORITE: "swing",
    SOURCE_POSITION: "position",
    SOURCE_ALERT: "alert",
}

HORIZON_DAY = "day"
HORIZON_SWING = "swing"

VIEW_MY_WATCHLIST = "my_watchlist"
VIEW_M5_BOARD = "m5_board"
VIEW_SWING_FAVORITES = "swing_favorites"
VIEW_POSITIONS = "positions"
VIEW_ALL = "all"

#: The selector's order, left to right. `all` is last because it is the
#: fallback, not the default reading.
VIEWS = (
    VIEW_MY_WATCHLIST,
    VIEW_M5_BOARD,
    VIEW_SWING_FAVORITES,
    VIEW_POSITIONS,
    VIEW_ALL,
)

VIEW_LABELS = {
    VIEW_MY_WATCHLIST: "My watchlist",
    VIEW_M5_BOARD: "M5 / TC2000",
    VIEW_SWING_FAVORITES: "Swing favorites",
    VIEW_POSITIONS: "Open positions",
    VIEW_ALL: "All",
}

#: `trades.status` -> what the row says. A status the journal has not got a
#: word for is passed through lowercased rather than guessed at.
STATUS_OPEN = "open"
STATUS_PARTLY_CLOSED = "partly_closed"
STATUS_CLOSED = "closed"

#: The adds the intent stream can vouch for as an ADD TIME. `observed_external`
#: is when the DESK noticed a file had changed, never when the trader typed it
#: (see `watchlist_intent_events`' module docstring), so it is deliberately not
#: here: it leaves `first_seen` blank rather than back-dating a moment nobody
#: measured.
_VOUCHED_ADD_SOURCES = (
    intent.SOURCE_TRADER_EDIT,
    intent.SOURCE_TRADER_PASTE,
    intent.SOURCE_MACHINE_INJECT,
)

#: `FocusPickStore` category -> the horizon a Focus pick is an interest in.
_CATEGORY_HORIZONS = {"m5": HORIZON_DAY, "swing": HORIZON_SWING}

#: `FocusPickStore` category and side -> the shared list its picks are injected
#: into, derived from `LIST_SPECS` so the two cannot drift apart.
_CATEGORY_LISTS = {
    category: {
        side: name
        for name, (list_side, list_horizon) in intent.LIST_SPECS.items()
        if list_horizon == horizon
        for side in (list_side,)
    }
    for category, horizon in _CATEGORY_HORIZONS.items()
}


def _symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _side(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("long"):
        return "long"
    if text.startswith("short"):
        return "short"
    return ""


@dataclass(frozen=True)
class WatchPosition:
    """One broker position on one name, in one account. Read-only projection."""

    broker: str = ""
    account: str = ""
    quantity: float = 0.0
    #: Dollars at the average entry, for a share position. ``None`` means NOT
    #: MEASURED - an option's average price is per contract and this module
    #: will not invent a multiplier to turn it into an exposure.
    exposure: float | None = None
    status: str = STATUS_OPEN
    instrument: str = ""
    bias: str = ""
    last_sync: datetime | None = None
    #: The sync that produced this row is older than the previous session's
    #: close (or there is none at all). The row is SHOWN, greyed; uncertainty
    #: never deletes a position.
    stale: bool = False

    @property
    def label(self) -> str:
        """"questrade 51234567 100 @ open" - display only."""
        parts = [part for part in (self.broker, self.account) if part]
        head = " ".join(parts) or "position"
        return f"{head} {self.quantity:g} ({self.status})"


@dataclass(frozen=True)
class WatchRow:
    """One name, one side, and every owner that has an opinion about it."""

    symbol: str
    side: str = ""
    horizons: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    positions: tuple[WatchPosition, ...] = ()
    #: WS-10B's verdict, carried verbatim. It never touches the order.
    adoption: str = ""
    liked_today: tuple[tuple[str, str], ...] = ()
    rejected_today: tuple[tuple[str, str], ...] = ()
    faded: bool = False
    armed_alerts: int = 0
    #: The earliest add the intent stream can vouch for, or "".
    first_seen: str = ""

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.symbol, self.side)

    @property
    def badges(self) -> tuple[str, ...]:
        """The badge labels, in the module's declared order (never by count)."""
        return tuple(
            SOURCE_LABELS[name] for name in SOURCES if name in self.sources
        )

    @property
    def horizon_text(self) -> str:
        return " / ".join(
            name for name in (HORIZON_DAY, HORIZON_SWING) if name in self.horizons
        )

    def has_source(self, name: str) -> bool:
        return name in self.sources


@dataclass
class _Draft:
    """Mutable accumulator; frozen into a :class:`WatchRow` at the end."""

    symbol: str
    side: str
    horizons: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    positions: list[WatchPosition] = field(default_factory=list)
    adoption: str = ""
    faded: bool = False
    first_seen: str = ""

    def note_first_seen(self, stamp: str) -> None:
        """Keep the EARLIEST vouched add. A blank never overwrites a time."""
        stamp = str(stamp or "")
        if not stamp:
            return
        if not self.first_seen or stamp < self.first_seen:
            self.first_seen = stamp


def build_watchlist_rows(
    *,
    shared_lists: Mapping[str, Sequence[str]] | None = None,
    focus_store: Any = None,
    swing_favorites: Iterable[Mapping[str, Any]] = (),
    journal_exposures: Iterable[Mapping[str, Any]] = (),
    intent_events: Iterable[Mapping[str, Any]] = (),
    board_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    decisions_today: Any = None,
    armed_alerts: Iterable[Mapping[str, Any]] = (),
    last_sync: Mapping[str, datetime] | None = None,
    now: datetime | None = None,
) -> tuple[WatchRow, ...]:
    """Every name the desk is holding an opinion about, once per side.

    Pure: opens nothing, writes nothing, reaches nothing. See the module
    docstring for the three rules the shape follows.
    """
    drafts: dict[tuple[str, str], _Draft] = {}

    def draft(symbol: object, side: object) -> _Draft | None:
        sym = _symbol(symbol)
        if not sym:
            return None
        key = (sym, _side(side))
        existing = drafts.get(key)
        if existing is None:
            existing = _Draft(symbol=sym, side=key[1])
            drafts[key] = existing
        return existing

    authorship, first_seen = _read_intent_stream(intent_events)
    focus_entries = _focus_entries(focus_store)
    injected = _injected_keys(focus_entries)

    # 1. The four plain files. Side and horizon are LIST_SPECS' business, so no
    #    caller can label a swing list as a day-trade interest.
    for list_name, symbols in (shared_lists or {}).items():
        name = str(list_name or "").strip().lower()
        spec = intent.LIST_SPECS.get(name)
        if spec is None:
            continue
        side, horizon = spec
        for symbol in symbols or ():
            row = draft(symbol, side)
            if row is None:
                continue
            row.horizons.add(horizon)
            key = (name, row.symbol)
            if _is_the_traders_own(key, authorship, injected):
                row.sources.add(SOURCE_MANUAL)
            row.note_first_seen(first_seen.get(key, ""))

    # 2. Focus. An auto-adopted pick wears ONE badge - the machine's - because
    #    "auto" is the fact that matters when the trader decides what to keep.
    for symbol, side, category, auto in focus_entries:
        row = draft(symbol, side)
        if row is None:
            continue
        row.horizons.add(_CATEGORY_HORIZONS.get(category, HORIZON_DAY))
        row.sources.add(SOURCE_FOCUS_AUTO if auto else SOURCE_FOCUS_TRADER)

    # 3. A faded pick is not deleted, so it is not dropped here either.
    for entry in _faded_entries(focus_store):
        symbol, side, category, owner = entry
        row = draft(symbol, side)
        if row is None:
            continue
        row.faded = True
        row.horizons.add(_CATEGORY_HORIZONS.get(category, HORIZON_DAY))
        row.sources.add(
            SOURCE_FOCUS_AUTO if str(owner or "") == "machine" else SOURCE_FOCUS_TRADER
        )

    # 4. Today's swing picks (`favorites_for_session` has already replayed the
    #    retractions - a removed favorite is simply not in the list).
    for favorite in swing_favorites or ():
        if not isinstance(favorite, Mapping):
            continue
        row = draft(favorite.get("symbol"), favorite.get("side"))
        if row is None:
            continue
        row.sources.add(SOURCE_SWING_FAVORITE)
        row.horizons.add(HORIZON_SWING)

    # 5. The M5 strength board, with WS-10B's scan verdict carried verbatim.
    for side, rows in (board_rows or {}).items():
        side_text = _side(side)
        for entry in rows or ():
            if not isinstance(entry, Mapping):
                continue
            row = draft(entry.get("symbol"), side_text)
            if row is None:
                continue
            row.sources.add(SOURCE_BOARD)
            row.horizons.add(HORIZON_DAY)
            adoption = str(entry.get("adoption") or "").strip()
            if adoption:
                row.adoption = adoption

    # 6. The broker's own answer. Read-only, and it arms nothing.
    for position, symbol, side in _positions(journal_exposures, last_sync, now):
        row = draft(symbol, side)
        if row is None:
            continue
        row.sources.add(SOURCE_POSITION)
        row.positions.append(position)

    # 7. Armed price-alert legs, per symbol.
    armed_by_symbol = _armed_counts(armed_alerts)
    for symbol, count in armed_by_symbol.items():
        if count <= 0:
            continue
        if not any(key[0] == symbol for key in drafts):
            # Nothing else knows this name. The entry is still armed and still
            # polling, so it gets a row rather than disappearing with the page
            # its board used to live on. No side is claimed: an alert is a
            # price, not a thesis.
            row = draft(symbol, "")
            if row is not None:
                row.sources.add(SOURCE_ALERT)

    liked, rejected = _decision_marks(decisions_today)

    out = [
        WatchRow(
            symbol=item.symbol,
            side=item.side,
            horizons=frozenset(item.horizons),
            sources=frozenset(item.sources),
            positions=tuple(item.positions),
            adoption=item.adoption,
            liked_today=liked.get(item.symbol, ()),
            rejected_today=rejected.get(item.symbol, ()),
            faded=item.faded,
            armed_alerts=int(armed_by_symbol.get(item.symbol, 0)),
            first_seen=item.first_seen,
        )
        for item in drafts.values()
    ]
    # By SYMBOL, then side. Never by source, never by strength, never by
    # result: decision 0016 puts names before entries.
    out.sort(key=lambda row: row.sort_key)
    return tuple(out)


def filter_rows(rows: Iterable[WatchRow], view: str) -> tuple[WatchRow, ...]:
    """One view. A filter over the same rows - nothing is recomputed here."""
    name = str(view or "").strip().lower() or VIEW_ALL
    everything = tuple(rows or ())
    if name == VIEW_ALL:
        return everything
    if name == VIEW_MY_WATCHLIST:
        keep = lambda row: SOURCE_MANUAL in row.sources  # noqa: E731
    elif name == VIEW_M5_BOARD:
        # What the intraday side is watching: the board's own rows and the M5
        # Focus list they feed. A swing-only Focus pick is not on it.
        keep = lambda row: (  # noqa: E731
            SOURCE_BOARD in row.sources
            or (
                HORIZON_DAY in row.horizons
                and bool(row.sources & {SOURCE_FOCUS_TRADER, SOURCE_FOCUS_AUTO})
            )
        )
    elif name == VIEW_SWING_FAVORITES:
        keep = lambda row: SOURCE_SWING_FAVORITE in row.sources  # noqa: E731
    elif name == VIEW_POSITIONS:
        keep = lambda row: bool(row.positions)  # noqa: E731
    else:
        return everything
    return tuple(row for row in everything if keep(row))


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _read_intent_stream(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """``(list, symbol) -> newest add source`` and ``-> earliest vouched ts``.

    The stream is read in FILE ORDER, which is the order the events happened.
    A `remove` resets authorship: a name the machine injected, the trader took
    off and then typed back is theirs.
    """
    authorship: dict[tuple[str, str], str] = {}
    first_seen: dict[tuple[str, str], str] = {}
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        list_name = str(row.get("list") or "").strip().lower()
        symbol = _symbol(row.get("symbol"))
        action = str(row.get("action") or "").strip().lower()
        source = str(row.get("source") or "").strip().lower()
        if not list_name or not symbol:
            continue
        key = (list_name, symbol)
        if action == intent.ACTION_REMOVE:
            authorship.pop(key, None)
            first_seen.pop(key, None)
            continue
        if action != intent.ACTION_ADD:
            continue
        authorship[key] = source
        if source in _VOUCHED_ADD_SOURCES:
            stamp = str(row.get("ts") or "")
            existing = first_seen.get(key, "")
            if stamp and (not existing or stamp < existing):
                first_seen[key] = stamp
    return authorship, first_seen


def _injected_keys(
    focus_entries: Iterable[tuple[str, str, str, bool]],
) -> set[tuple[str, str]]:
    """``(list, symbol)`` pairs a LIVE Focus pick is currently injecting.

    `FocusPickStore._inject_into_shared` puts every pick on the shared list its
    category and side name, so a Focus pick's presence there explains itself -
    it is not a second, independent statement by the trader. The stream is
    still the authority when it has an answer (`_is_the_traders_own`); this is
    what the reading falls back on for the picks it never saw, which is every
    pick older than WS-5D.
    """
    out: set[tuple[str, str]] = set()
    for symbol, side, category, _auto in focus_entries:
        list_name = _CATEGORY_LISTS.get(str(category), {}).get(side)
        if list_name:
            out.add((list_name, symbol))
    return out


def _is_the_traders_own(
    key: tuple[str, str],
    authorship: Mapping[tuple[str, str], str],
    injected: set[tuple[str, str]],
) -> bool:
    """Did the TRADER put this name on this list?

    The intent stream answers first and exactly: the newest add for the pair
    names its writer. Only when the stream has nothing to say does presence
    fall back on the injection above - and a name nothing explains is the
    trader's, because the four files predate every writer that labels itself.
    """
    source = authorship.get(key)
    if source is not None:
        return source != intent.SOURCE_MACHINE_INJECT
    return key not in injected


def _focus_entries(store: Any) -> list[tuple[str, str, str, bool]]:
    """``(symbol, side, category, auto_adopted)`` for every live Focus pick."""
    if store is None:
        return []
    try:
        by_category = store.all_focus_by_category()
    except Exception:  # noqa: BLE001 - a broken store never breaks the reading
        return []
    out: list[tuple[str, str, str, bool]] = []
    for category, sides in (by_category or {}).items():
        for side, symbols in (sides or {}).items():
            for symbol in symbols or ():
                sym = _symbol(symbol)
                if not sym:
                    continue
                try:
                    auto = bool(store.is_auto_adopted(sym, side, category))
                except Exception:  # noqa: BLE001
                    auto = False
                out.append((sym, _side(side), str(category), auto))
    return out


def _faded_entries(store: Any) -> list[tuple[str, str, str, str]]:
    if store is None:
        return []
    try:
        rows = store.faded_picks()
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[str, str, str, str]] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        sym = _symbol(row.get("symbol"))
        if not sym:
            continue
        out.append(
            (sym, _side(row.get("side")), str(row.get("category") or "m5"), str(row.get("owner") or ""))
        )
    return out


#: `journal_exposure` bias -> the side a watchlist row is on. A LONG option is
#: never a bullish setup; `unknown` claims no side at all rather than guessing.
_BIAS_SIDES = {
    journal_exposure.BIAS_BULLISH: "long",
    journal_exposure.BIAS_BULLISH_OR_NEUTRAL: "long",
    journal_exposure.BIAS_BEARISH: "short",
    journal_exposure.BIAS_BEARISH_OR_NEUTRAL: "short",
}


def _positions(
    trades: Iterable[Mapping[str, Any]],
    last_sync: Mapping[str, datetime] | None,
    now: datetime | None,
) -> list[tuple[WatchPosition, str, str]]:
    rows = [trade for trade in (trades or ()) if isinstance(trade, Mapping)]
    if not rows:
        return []
    try:
        exposures = journal_exposure.classify_all(rows)
    except Exception:  # noqa: BLE001
        exposures = {}
    floor = _stale_floor(now)
    syncs = {
        str(name or "").strip().lower(): stamp
        for name, stamp in (last_sync or {}).items()
    }

    out: list[tuple[WatchPosition, str, str]] = []
    for trade in rows:
        status = _position_status(trade.get("status"))
        broker = str(trade.get("broker") or "").strip().lower()
        sync = syncs.get(broker)
        stale = _is_stale(sync, floor)
        if status == STATUS_CLOSED and not stale:
            # A verified refresh says it is off. THAT is what takes it out of
            # the automatic view - never a guess, and never a failed sync.
            continue
        exposure = exposures.get(str(trade.get("trade_id") or ""))
        if exposure is None:
            try:
                exposure = journal_exposure.classify_exposure(trade)
            except Exception:  # noqa: BLE001
                continue
        symbol = _underlying(trade, exposure)
        if not symbol:
            continue
        quantity = _float(trade.get("quantity_opened")) - _float(trade.get("quantity_closed"))
        price = _float(trade.get("average_entry_price"))
        # Shares only. An option's average price is per CONTRACT and this
        # module will not invent a multiplier - NULL is "not measured".
        dollars = quantity * price if exposure.instrument == "STK" else None
        out.append(
            (
                WatchPosition(
                    broker=broker,
                    account=str(trade.get("account_number") or "").strip(),
                    quantity=quantity,
                    exposure=dollars,
                    status=status,
                    instrument=exposure.instrument,
                    bias=exposure.market_bias,
                    last_sync=sync,
                    stale=stale,
                ),
                symbol,
                _BIAS_SIDES.get(exposure.market_bias, ""),
            )
        )
    return out


def _underlying(trade: Mapping[str, Any], exposure: journal_exposure.Exposure) -> str:
    """The TICKER, never the OCC symbol: a put on NVDA is filed under NVDA."""
    for contract in exposure.contracts:
        if contract.underlying:
            return _symbol(contract.underlying)
    return _symbol(trade.get("symbol"))


def _position_status(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "CLOSED_PARTIAL":
        return STATUS_PARTLY_CLOSED
    if text == "CLOSED":
        return STATUS_CLOSED
    if text == "OPEN":
        return STATUS_OPEN
    return text.lower() or STATUS_OPEN


def _stale_floor(now: datetime | None) -> datetime | None:
    """The previous session's close. A sync older than that is stale."""
    moment = now or datetime.now(tz=market_calendar.MARKET_TZ)
    try:
        return market_calendar.session_close(
            market_calendar.previous_session(moment.date())
        )
    except Exception:  # noqa: BLE001 - a calendar that cannot answer stales nothing
        return None


def _is_stale(sync: datetime | None, floor: datetime | None) -> bool:
    if sync is None:
        # Nobody has told us when this was last checked. That is uncertainty,
        # which is shown and labelled - never silently treated as current.
        return True
    if floor is None:
        return False
    stamp = sync if sync.tzinfo is not None else sync.replace(tzinfo=market_calendar.MARKET_TZ)
    return stamp < floor


def _armed_counts(entries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        symbol = _symbol(entry.get("symbol"))
        if not symbol:
            continue
        legs = int(bool(entry.get("armed_above"))) + int(bool(entry.get("armed_below")))
        counts[symbol] = counts.get(symbol, 0) + legs
    return counts


def _decision_marks(
    decisions: Any,
) -> tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, tuple[tuple[str, str], ...]]]:
    """WS-SX's two marks. A like and a veto on one day are independent facts."""
    if decisions is None:
        return {}, {}
    liked = {
        _symbol(symbol): tuple(tuple(item) for item in entries)
        for symbol, entries in (getattr(decisions, "liked", None) or {}).items()
    }
    rejected = {
        _symbol(symbol): tuple(tuple(item) for item in entries)
        for symbol, entries in (getattr(decisions, "rejected", None) or {}).items()
    }
    return liked, rejected


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def focus_snapshot(store: Any) -> "FocusSnapshot":
    """Freeze a live ``FocusPickStore`` into the three reads this module makes.

    The Qt thread owns the store (it is a WRITER: `reload()` expires the m5
    list and repairs the fade clocks), so the worker never touches it. These
    three reads are in-memory list copies - microseconds - and the snapshot
    they produce is what crosses the thread boundary.
    """
    entries = tuple(_focus_entries(store))
    faded = tuple(_faded_entries(store))
    return FocusSnapshot(entries=entries, faded=faded)


@dataclass(frozen=True)
class FocusSnapshot:
    """What ``build_watchlist_rows`` needs from Focus, and nothing else."""

    entries: tuple[tuple[str, str, str, bool], ...] = ()
    faded: tuple[tuple[str, str, str, str], ...] = ()

    def all_focus_by_category(self) -> dict[str, dict[str, list[str]]]:
        out: dict[str, dict[str, list[str]]] = {}
        for symbol, side, category, _auto in self.entries:
            out.setdefault(category, {}).setdefault(side, []).append(symbol)
        return out

    def is_auto_adopted(self, symbol: object, side: object, category: object = "m5") -> bool:
        key = (_symbol(symbol), _side(side), str(category))
        return any(
            (item[0], item[1], item[2]) == key and item[3] for item in self.entries
        )

    def faded_picks(self) -> list[dict[str, Any]]:
        return [
            {"symbol": symbol, "side": side, "category": category, "owner": owner}
            for symbol, side, category, owner in self.faded
        ]
