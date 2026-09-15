"""Packet WS-WL - ONE Trading Desk Watchlist for Focus, lists and positions.

Trader's brief, WISHLIST 10G: the main **Watchlist tab belongs on Trading
Desk**; the Journal links to its Positions view. One list with five views (My
watchlist | M5/TC2000 | Swing favorites | Open positions | All), source badges
and side/horizon shown *without turning source into priority*, single-symbol add
and paste-many with duplicate handling. Journal-derived positions are read-only
projections; an unknown or stale sync never silently removes one; a closed
position leaves the auto view only after a VERIFIED refresh; hand-added names
survive. No alert is armed just because a broker position exists. The standalone
**Chart Review** and **Focus Picks** nav pages retire only after every action
they had is reachable from the tab.

These tests are RED on purpose. The builder may only ADD to them.

--------------------------------------------------------------------------
WHAT THESE TESTS PIN (the seam names the builder has to match)
--------------------------------------------------------------------------

``scripts/watchlist_views.py`` - PURE, opens no store, writes nothing::

    build_watchlist_rows(
        *,
        shared_lists,        # {"longs": [...], "shorts": [...],
                             #  "swinglongs": [...], "shortswings": [...]}
                             #  - the four plain files, keyed the way
                             #    `watchlist_intent_events.LIST_SPECS` keys them,
                             #    so side and horizon are DERIVED here and no
                             #    caller can label a swing list as a day interest
        focus_store,         # a real `focus_picks.FocusPickStore`
        swing_favorites,     # rows from `swing_favorites.favorites_for_session`
        journal_exposures,   # journal `trades` rows (the real column names)
        intent_events,       # rows from `watchlist_intent_events.read_events`
        board_rows,          # {"long": [...], "short": [...]} - `build_board`'s
                             #   own shape, each row carrying WS-10B `adoption`
        decisions_today,     # a `pick_feedback.DayDecisions`
        armed_alerts=(),     # normalized `price_alerts` entries
        last_sync=None,      # {broker: datetime} of the last VERIFIED import run
        now=None,
    ) -> tuple[WatchRow, ...]

    WatchRow(symbol, side, horizons, sources, positions, adoption,
             liked_today, rejected_today, faded, armed_alerts, first_seen)
    WatchPosition(broker, account, quantity, exposure, status, instrument,
                  bias, last_sync, stale)

Two deliberate departures from the packet's wording, both because the real data
forced them, both called out in the handoff:

* ``positions`` is a TUPLE, not one record. One name is legitimately held in two
  accounts (the trader has four), and folding two accounts into one row would
  either hide an account or invent a sum nobody holds. Empty tuple = no position.
* ``horizons`` is a frozenset of ``day`` / ``swing``, not one string. A name can
  be on ``longs.txt`` AND ``swinglongs.txt``; row identity is ``(symbol, side)``
  ("one symbol may appear once per side"), so the horizon cannot be part of the
  key and cannot be a single value either.

Source badges - the trap this packet has to survive: **`FocusPickStore.add`
INJECTS into `longs.txt`/`shorts.txt`**, so mere presence on a shared list
cannot mean "the trader typed it". `manual` is decided by the WS-5D intent
stream: presence on a shared list is `manual` unless the latest `add` the stream
holds for that (list, symbol) has source `machine_inject`. A name the stream has
never seen predates the stream and counts as manual with a BLANK `first_seen`.

``first_seen`` is the earliest add the stream can vouch for (`trader_edit`,
`trader_paste`, `machine_inject`). An ``observed_external`` add is an OBSERVATION
time, not an add time, so it leaves `first_seen` blank rather than back-dating a
moment nobody measured (`watchlist_intent_events` module docstring).

``scripts/ui/panels/watchlist_tab.py``::

    WatchlistTabPanel(QWidget)
        .set_rows(rows) / .rows()
        .set_view(view) / .view()
        .visible_rows()                 -> the rows the table is showing, in order
        .selected_symbol()              -> "SYM" or ""
        .select_symbol(symbol, side="") -> bool
        .add_symbol(text, side=, horizon=) -> AddResult(added, duplicates, rejected)
        .paste_many(text, side=, horizon=) -> AddResult
        .chart_selected()               # -> the shared chart, a board look
        .remove_selected()              # -> the OWNING store, per source
        .restore_selected()             # -> FocusPickStore.restore_faded
        .arm_selected(above=, below=)   # -> PriceAlertService.save_entries
        .set_chart_sink(sink)           # the `watchlists_panel.py` pattern
        .add_input                      # the QLineEdit `Ctrl+L` focuses

``scripts/ui/services/watchlist_tab_service.py``::

    WatchlistTabService(QObject)  - one, owned by MainWindow, the
    StrengthBoardService precedent: `rowsChanged = Signal(tuple)`,
    `.rows()`, `.refresh_now()`, `.shutdown()`, and the BUILD runs on a
    worker thread named `watchlist-tab`, never on the Qt thread.

Nav, after the retirement::

    MainWindow.show_watchlist_positions()   # by page TITLE, never by index
    TradingDeskPanel.show_watchlist()       # raises the tab
    JournalPanel.positionsOnWatchlistRequested = Signal()
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import market_calendar  # noqa: E402
import pick_feedback  # noqa: E402
import swing_favorites  # noqa: E402
import watchlist_intent_events as intent  # noqa: E402
from focus_picks import FocusPickStore  # noqa: E402

class _NotBuiltYet:
    """Stands in for a module WS-WL has not built, so each test FAILS on its own.

    Deliberately not `pytest.importorskip`: a skipped test is not a red test, and
    the tester's contract is that every one of these fails on the current code
    with a line that names what is missing. Deliberately not a bare module-level
    import either - that would be ONE collection error covering thirty tests,
    and the Qt and nav tests below have to fail on their own seams.
    """

    def __init__(self, module: str) -> None:
        self._module = module

    def __getattr__(self, name: str):
        raise AssertionError(
            f"WS-WL: scripts/{self._module}.py does not exist yet "
            f"(this test needs {self._module}.{name})"
        )


try:  # RED until the builder writes item 1.
    import watchlist_views  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - green once the packet lands
    watchlist_views = _NotBuiltYet("watchlist_views")  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# A pinned clock. Friday 2026-09-11; the previous session is Thursday 09-10 and
# its close is 16:00 ET, so "stale" has one unambiguous boundary.
# ---------------------------------------------------------------------------
TODAY = date(2026, 9, 11)
NOW = datetime(2026, 9, 11, 10, 0, tzinfo=market_calendar.MARKET_TZ)
PREVIOUS_CLOSE = market_calendar.session_close(market_calendar.previous_session(TODAY))
FRESH_SYNC = datetime(2026, 9, 11, 9, 45, tzinfo=market_calendar.MARKET_TZ)
STALE_SYNC = datetime(2026, 9, 9, 15, 0, tzinfo=market_calendar.MARKET_TZ)


def test_the_pinned_clock_really_straddles_the_previous_session_close():
    """Guard the fixtures themselves: fresh is after 16:00 Thursday, stale before."""
    assert PREVIOUS_CLOSE.date() == date(2026, 9, 10)
    assert PREVIOUS_CLOSE.hour == 16
    assert FRESH_SYNC > PREVIOUS_CLOSE
    assert STALE_SYNC < PREVIOUS_CLOSE


# ---------------------------------------------------------------------------
# Real-shaped fixtures
# ---------------------------------------------------------------------------
def _write_list(path: Path, symbols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")


def _open_store(home: Path) -> FocusPickStore:
    store = FocusPickStore(
        focus_longs_path=home / "focus_longs.txt",
        focus_shorts_path=home / "focus_shorts.txt",
        longs_path=home / "longs.txt",
        shorts_path=home / "shorts.txt",
        membership_path=home / "focus_pick_membership.json",
    )
    store.home = home  # type: ignore[attr-defined]
    return store


@pytest.fixture
def focus_store(tmp_path):
    """A real store on temp paths. It injects into the shared lists, as it does live."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    for name in ("longs", "shorts", "swinglongs", "shortswings"):
        _write_list(home / f"{name}.txt", [])
    return _open_store(home)


def _shared_lists(store) -> dict[str, list[str]]:
    """The four files as they stand on disk - injections included."""
    from watchlist_utils import read_watchlist_symbols

    home = store.home
    return {
        name: read_watchlist_symbols(home / f"{name}.txt")
        for name in ("longs", "shorts", "swinglongs", "shortswings")
    }


def _trade(
    *,
    trade_id: str,
    symbol: str,
    broker: str = "questrade",
    account: str = "51234567",
    security_type: str = "STK",
    direction: str = "LONG",
    status: str = "OPEN",
    quantity_opened: float = 100.0,
    quantity_closed: float = 0.0,
    average_entry_price: float = 50.0,
) -> dict:
    """One `trades` row with the REAL column names (`journal_store` schema).

    Every column the schema declares is PRESENT - an old row has its keys
    present and empty, never absent.
    """
    return {
        "trade_id": trade_id,
        "broker": broker,
        "account_number": account,
        "account_label": "",
        "symbol": symbol,
        "security_type": security_type,
        "currency": "USD",
        "direction": direction,
        "status": status,
        "opened_at": "2026-09-08T09:41:00-04:00",
        "closed_at": "" if status == "OPEN" else "2026-09-10T15:12:00-04:00",
        "trade_date": "2026-09-08",
        "quantity_opened": quantity_opened,
        "quantity_closed": quantity_closed,
        "average_entry_price": average_entry_price,
        "average_exit_price": 0.0,
        "gross_pnl": 0.0,
        "commission": 0.0,
        "fees": 0.0,
        "net_pnl": 0.0,
        "pnl_usd": None,
        "auto_tag_summary": "",
        "tag_confidence": None,
        "updated_at": "2026-09-10T16:05:00-04:00",
    }


def _intent_rows(tmp_path, rows) -> list[dict]:
    """Write real stream rows through the real writer, then read them back."""
    path = tmp_path / "watchlist_intent_events.jsonl"
    for list_name, symbol, source, when in rows:
        intent.record_changes(
            list_name=list_name,
            added=[symbol],
            source=source,
            writer="tests.ws_wl",
            now=when,
            path=path,
        )
    return intent.read_events(path=path)


def _build(**kwargs):
    base = {
        "shared_lists": {},
        "focus_store": None,
        "swing_favorites": (),
        "journal_exposures": (),
        "intent_events": (),
        "board_rows": {},
        "decisions_today": pick_feedback.DayDecisions(trade_date=TODAY.isoformat()),
        "armed_alerts": (),
        "last_sync": {},
        "now": NOW,
    }
    base.update(kwargs)
    return watchlist_views.build_watchlist_rows(**base)


def _row(rows, symbol, side="long"):
    for row in rows:
        if row.symbol == symbol and row.side == side:
            return row
    raise AssertionError(
        f"no {side} row for {symbol}; rows={[(r.symbol, r.side) for r in rows]}"
    )


# ---------------------------------------------------------------------------
# Item 1 - the pure row builder
# ---------------------------------------------------------------------------
def test_one_name_manual_in_auto_focus_and_in_the_journal_is_one_row_with_three_badges(
    tmp_path, focus_store
):
    """The trader typed AMD, the machine adopted it, and the broker holds it.

    Three owners, one row, three badges - and `manual` survives the machine's
    injection because the stream says who typed it.
    """
    _write_list(focus_store.home / "longs.txt", ["AMD"])
    events = _intent_rows(
        tmp_path,
        [("longs", "AMD", intent.SOURCE_TRADER_EDIT, datetime(2026, 9, 8, 6, 31))],
    )
    focus_store.add("AMD", "long", "m5", today=TODAY)
    focus_store.mark_auto_adopted("AMD", "long", "m5", reason="strength board parity")

    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=[_trade(trade_id="T1", symbol="AMD")],
        intent_events=events,
        last_sync={"questrade": FRESH_SYNC},
    )

    assert len([r for r in rows if r.symbol == "AMD"]) == 1
    row = _row(rows, "AMD")
    assert row.sources == frozenset(
        {
            watchlist_views.SOURCE_MANUAL,
            watchlist_views.SOURCE_FOCUS_AUTO,
            watchlist_views.SOURCE_POSITION,
        }
    )
    assert row.horizons == frozenset({"day"})
    assert len(row.positions) == 1


def test_a_focus_injected_name_the_trader_never_typed_is_not_badged_manual(focus_store):
    """`FocusPickStore.add` writes into `longs.txt`. Presence is not authorship.

    Without this, every machine adoption would read as the trader's own list and
    the badges would say nothing at all.
    """
    focus_store.add("NVDA", "long", "m5", today=TODAY)
    injected = _shared_lists(focus_store)
    assert "NVDA" in injected["longs"], "the store still injects; the premise holds"

    events = [
        intent.build_row(
            list_name="longs",
            symbol="NVDA",
            action=intent.ACTION_ADD,
            source=intent.SOURCE_MACHINE_INJECT,
            writer="focus_picks",
            now=datetime(2026, 9, 11, 6, 40),
        )
    ]
    rows = _build(shared_lists=injected, focus_store=focus_store, intent_events=events)
    row = _row(rows, "NVDA")
    assert watchlist_views.SOURCE_MANUAL not in row.sources
    assert watchlist_views.SOURCE_FOCUS_TRADER in row.sources


def test_the_same_name_long_and_short_is_two_rows_never_one(focus_store):
    """One symbol may appear once PER SIDE."""
    _write_list(focus_store.home / "longs.txt", ["TSLA"])
    _write_list(focus_store.home / "shorts.txt", ["TSLA"])
    rows = _build(shared_lists=_shared_lists(focus_store), focus_store=focus_store)
    sides = sorted(r.side for r in rows if r.symbol == "TSLA")
    assert sides == ["long", "short"]


def test_a_day_list_and_a_swing_list_are_one_row_carrying_both_horizons(focus_store):
    """Same symbol, same side, two horizons - one row, both horizons named."""
    _write_list(focus_store.home / "longs.txt", ["MSFT"])
    _write_list(focus_store.home / "swinglongs.txt", ["MSFT"])
    rows = _build(shared_lists=_shared_lists(focus_store), focus_store=focus_store)
    assert len([r for r in rows if r.symbol == "MSFT"]) == 1
    assert _row(rows, "MSFT").horizons == frozenset({"day", "swing"})


def test_two_accounts_holding_one_name_are_one_row_that_names_both(focus_store):
    """Four accounts exist. Folding two into one hides an account or invents a sum."""
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=[
            _trade(
                trade_id="T1", symbol="AAPL", account="51234567", quantity_opened=100.0
            ),
            _trade(
                trade_id="T2",
                symbol="AAPL",
                account="U9876543",
                broker="ibkr",
                quantity_opened=40.0,
            ),
        ],
        last_sync={"questrade": FRESH_SYNC, "ibkr": FRESH_SYNC},
    )
    row = _row(rows, "AAPL")
    assert {p.account for p in row.positions} == {"51234567", "U9876543"}
    assert sorted(p.quantity for p in row.positions) == [40.0, 100.0]


def test_a_bought_put_is_a_bearish_position_filed_under_its_underlying(focus_store):
    """A LONG option is never a bullish setup (`journal_exposure`).

    The OCC symbol is not a ticker, either: the row is NVDA, not
    NVDA260918P00150000.
    """
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=[
            _trade(
                trade_id="T9",
                symbol="NVDA260918P00150000",
                security_type="OPT",
                direction="LONG",
                quantity_opened=3.0,
            )
        ],
        last_sync={"questrade": FRESH_SYNC},
    )
    assert [r.symbol for r in rows] == ["NVDA"]
    row = _row(rows, "NVDA", side="short")
    assert row.positions[0].bias == "bearish"
    assert row.positions[0].instrument == "OPT"


def test_a_partly_closed_position_keeps_the_quantity_that_is_still_on(focus_store):
    """300 opened, 200 closed. What is still on is 100 - not 300, not 200."""
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=[
            _trade(
                trade_id="T3",
                symbol="SOFI",
                status="CLOSED_PARTIAL",
                quantity_opened=300.0,
                quantity_closed=200.0,
                average_entry_price=12.5,
            )
        ],
        last_sync={"questrade": FRESH_SYNC},
    )
    position = _row(rows, "SOFI").positions[0]
    assert position.status == "partly_closed"
    assert position.quantity == 100.0
    assert position.exposure == pytest.approx(1250.0)


def test_a_stale_sync_leaves_the_position_stale_and_never_removes_it(focus_store):
    """A sync that failed is uncertainty, and uncertainty never deletes."""
    trades = [_trade(trade_id="T4", symbol="PLTR")]
    stale = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=trades,
        last_sync={"questrade": STALE_SYNC},
    )
    fresh = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=trades,
        last_sync={"questrade": FRESH_SYNC},
    )
    assert _row(stale, "PLTR").positions[0].stale is True
    assert _row(stale, "PLTR").positions[0].last_sync == STALE_SYNC
    assert _row(fresh, "PLTR").positions[0].stale is False


def test_a_closed_position_leaves_the_auto_view_only_after_a_verified_refresh(focus_store):
    """Same closed trade, two sync states. Only the verified one may drop it."""
    trades = [
        _trade(
            trade_id="T5",
            symbol="COIN",
            status="CLOSED",
            quantity_opened=50.0,
            quantity_closed=50.0,
        )
    ]
    stale = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=trades,
        last_sync={"questrade": STALE_SYNC},
    )
    fresh = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        journal_exposures=trades,
        last_sync={"questrade": FRESH_SYNC},
    )
    assert _row(stale, "COIN").positions[0].stale is True
    assert [r.symbol for r in fresh] == []


def test_a_hand_added_name_survives_the_position_going_away(tmp_path, focus_store):
    """The trader's own name is theirs. A verified close takes the badge, not the row."""
    _write_list(focus_store.home / "longs.txt", ["COIN"])
    events = _intent_rows(
        tmp_path,
        [("longs", "COIN", intent.SOURCE_TRADER_PASTE, datetime(2026, 9, 9, 7, 2))],
    )
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        intent_events=events,
        journal_exposures=[
            _trade(trade_id="T6", symbol="COIN", status="CLOSED", quantity_closed=100.0)
        ],
        last_sync={"questrade": FRESH_SYNC},
    )
    row = _row(rows, "COIN")
    assert row.sources == frozenset({watchlist_views.SOURCE_MANUAL})
    assert row.positions == ()


def test_an_externally_observed_add_has_a_blank_first_seen(tmp_path, focus_store):
    """`observed_external` is when the DESK saw it, never when the trader typed it.

    A trader edit in the same stream keeps its time, so this is not "first_seen
    is always blank".
    """
    _write_list(focus_store.home / "longs.txt", ["RIVN", "UBER"])
    events = _intent_rows(
        tmp_path,
        [
            (
                "longs",
                "RIVN",
                intent.SOURCE_OBSERVED_EXTERNAL,
                datetime(2026, 9, 11, 6, 30),
            ),
            ("longs", "UBER", intent.SOURCE_TRADER_EDIT, datetime(2026, 9, 10, 7, 15)),
        ],
    )
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        intent_events=events,
    )
    assert _row(rows, "RIVN").first_seen == ""
    assert _row(rows, "RIVN").sources == frozenset({watchlist_views.SOURCE_MANUAL})
    assert _row(rows, "UBER").first_seen.startswith("2026-09-10T")


def test_a_name_the_stream_never_saw_is_manual_with_a_blank_first_seen(focus_store):
    """The four lists predate the stream. Silence is not an invented add."""
    _write_list(focus_store.home / "shortswings.txt", ["XOM"])
    rows = _build(shared_lists=_shared_lists(focus_store), focus_store=focus_store)
    row = _row(rows, "XOM", side="short")
    assert row.first_seen == ""
    assert row.horizons == frozenset({"swing"})
    assert watchlist_views.SOURCE_MANUAL in row.sources


def test_a_faded_focus_pick_is_marked_faded_and_not_dropped(focus_store):
    """Faded is reversible and is not deleted (`focus_faded.json`)."""
    focus_store.add("ROKU", "long", "swing", today=date(2026, 7, 1))
    faded = focus_store.fade_stale_picks(today=TODAY)
    assert [row["symbol"] for row in faded] == ["ROKU"], "the fade premise holds"

    rows = _build(shared_lists=_shared_lists(focus_store), focus_store=focus_store)
    row = _row(rows, "ROKU")
    assert row.faded is True
    assert row.horizons == frozenset({"swing"})


def test_a_swing_favorite_is_its_own_badge_and_a_retraction_takes_it_off(
    tmp_path, focus_store
):
    """`favorites_for_session` replays adds minus retractions; the row follows it."""
    path = tmp_path / "swing_favorites.jsonl"
    swing_favorites.record_favorite(
        symbol="SNOW", side="long", session_date=TODAY.isoformat(), path=path
    )
    swing_favorites.record_favorite(
        symbol="DDOG", side="long", session_date=TODAY.isoformat(), path=path
    )
    swing_favorites.record_favorite(
        symbol="DDOG",
        side="long",
        action=swing_favorites.ACTION_REMOVE,
        session_date=TODAY.isoformat(),
        path=path,
    )
    favorites = swing_favorites.favorites_for_session(TODAY.isoformat(), path=path)

    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        swing_favorites=favorites,
    )
    assert [r.symbol for r in rows] == ["SNOW"]
    assert _row(rows, "SNOW").sources == frozenset({watchlist_views.SOURCE_SWING_FAVORITE})


def test_the_boards_scan_verdict_travels_onto_the_row_without_becoming_a_rank(focus_store):
    """WS-10B's `adoption` is carried, and it never touches the order."""
    board = {
        "long": [
            {"symbol": "AVGO", "strength": 91.0, "adoption": "already_in_focus"},
            {
                "symbol": "ABNB",
                "strength": 12.0,
                "adoption": "not adopted: below session VWAP",
            },
        ],
        "short": [],
    }
    rows = _build(
        shared_lists=_shared_lists(focus_store), focus_store=focus_store, board_rows=board
    )
    assert [r.symbol for r in rows] == ["ABNB", "AVGO"], "sorted by symbol, not by strength"
    assert _row(rows, "AVGO").adoption == "already_in_focus"
    assert _row(rows, "ABNB").adoption == "not adopted: below session VWAP"
    assert _row(rows, "AVGO").sources == frozenset({watchlist_views.SOURCE_BOARD})


def test_todays_like_and_todays_veto_are_two_independent_marks_on_one_row(focus_store):
    """WS-SX: a name can be liked AND vetoed the same day; neither cancels the other."""
    decisions = pick_feedback.DayDecisions(
        trade_date=TODAY.isoformat(),
        liked={"AMD": (("quick", "2026-09-11T09:50:00-04:00"),)},
        rejected={"AMD": (("veto", "2026-09-11T11:20:00-04:00"),)},
    )
    _write_list(focus_store.home / "longs.txt", ["AMD"])
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        decisions_today=decisions,
    )
    row = _row(rows, "AMD")
    assert [kind for kind, _ in row.liked_today] == ["quick"]
    assert [kind for kind, _ in row.rejected_today] == ["veto"]


def test_armed_legs_are_counted_per_symbol_and_a_position_arms_nothing(focus_store):
    """Two armed legs count 2; a disarmed leg counts 0; a broker position counts 0."""
    _write_list(focus_store.home / "longs.txt", ["AMD", "INTC"])
    entries = [
        {
            "symbol": "AMD",
            "above": 150.0,
            "below": 140.0,
            "armed_at": "2026-09-10",
            "armed_above": True,
            "armed_below": True,
            "note": "",
            "history": [],
        },
        {
            "symbol": "INTC",
            "above": 30.0,
            "below": None,
            "armed_at": "2026-09-10",
            "armed_above": False,
            "armed_below": False,
            "note": "expired",
            "history": [],
        },
    ]
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        armed_alerts=entries,
        journal_exposures=[_trade(trade_id="T7", symbol="GOOG")],
        last_sync={"questrade": FRESH_SYNC},
    )
    assert _row(rows, "AMD").armed_alerts == 2
    assert _row(rows, "INTC").armed_alerts == 0
    assert _row(rows, "GOOG").armed_alerts == 0


def test_rows_sort_by_symbol_within_a_view_and_never_by_source(focus_store):
    """Source is a badge, not a priority (decision 0016: names before entries)."""
    _write_list(focus_store.home / "longs.txt", ["ZM", "AAL"])
    focus_store.add("MU", "long", "m5", today=TODAY)
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        swing_favorites=[{"symbol": "BA", "side": "long"}],
        journal_exposures=[_trade(trade_id="T8", symbol="NKE")],
        last_sync={"questrade": FRESH_SYNC},
    )
    assert [r.symbol for r in rows] == ["AAL", "BA", "MU", "NKE", "ZM"]


def test_every_view_is_a_filter_over_the_same_rows(focus_store):
    """Five views, one row set. `all` is the union and loses nothing."""
    _write_list(focus_store.home / "longs.txt", ["AAL"])
    focus_store.add("MU", "long", "m5", today=TODAY)
    rows = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        swing_favorites=[{"symbol": "BA", "side": "long"}],
        board_rows={"long": [{"symbol": "MU", "adoption": "adopted"}], "short": []},
        journal_exposures=[_trade(trade_id="T8", symbol="NKE")],
        last_sync={"questrade": FRESH_SYNC},
    )
    filter_rows = watchlist_views.filter_rows
    assert [r.symbol for r in filter_rows(rows, watchlist_views.VIEW_ALL)] == [
        "AAL",
        "BA",
        "MU",
        "NKE",
    ]
    assert [r.symbol for r in filter_rows(rows, watchlist_views.VIEW_MY_WATCHLIST)] == ["AAL"]
    assert [r.symbol for r in filter_rows(rows, watchlist_views.VIEW_M5_BOARD)] == ["MU"]
    assert [r.symbol for r in filter_rows(rows, watchlist_views.VIEW_SWING_FAVORITES)] == ["BA"]
    assert [r.symbol for r in filter_rows(rows, watchlist_views.VIEW_POSITIONS)] == ["NKE"]
    assert watchlist_views.VIEWS == (
        watchlist_views.VIEW_MY_WATCHLIST,
        watchlist_views.VIEW_M5_BOARD,
        watchlist_views.VIEW_SWING_FAVORITES,
        watchlist_views.VIEW_POSITIONS,
        watchlist_views.VIEW_ALL,
    )
    for row in rows:
        assert row in filter_rows(rows, watchlist_views.VIEW_ALL)


def test_a_restart_re_reads_the_same_rows(tmp_path, focus_store):
    """A second store on the same files answers identically - no in-memory state."""
    _write_list(focus_store.home / "longs.txt", ["AMD"])
    focus_store.add("MU", "long", "swing", today=TODAY)
    events = _intent_rows(
        tmp_path,
        [("longs", "AMD", intent.SOURCE_TRADER_EDIT, datetime(2026, 9, 8, 6, 31))],
    )
    first = _build(
        shared_lists=_shared_lists(focus_store),
        focus_store=focus_store,
        intent_events=events,
    )
    reopened = _open_store(focus_store.home)
    second = _build(
        shared_lists=_shared_lists(reopened),
        focus_store=reopened,
        intent_events=intent.read_events(path=tmp_path / "watchlist_intent_events.jsonl"),
    )
    assert first == second


def test_the_builder_writes_nothing_and_reaches_no_broker(focus_store):
    """A read-only projection. Every file under the home folder is byte-identical."""
    _write_list(focus_store.home / "longs.txt", ["AMD"])
    focus_store.add("MU", "long", "swing", today=TODAY)
    before = {
        path: path.read_bytes()
        for path in sorted(focus_store.home.rglob("*"))
        if path.is_file()
    }

    def _no_network(*args, **kwargs):  # pragma: no cover - the point is it is never hit
        raise AssertionError("the watchlist builder reached the network")

    import socket as _socket

    saved = _socket.socket.connect
    _socket.socket.connect = _no_network  # type: ignore[assignment]
    try:
        _build(
            shared_lists=_shared_lists(focus_store),
            focus_store=focus_store,
            journal_exposures=[_trade(trade_id="T1", symbol="AMD")],
            last_sync={"questrade": FRESH_SYNC},
        )
    finally:
        _socket.socket.connect = saved  # type: ignore[assignment]

    after = {
        path: path.read_bytes()
        for path in sorted(focus_store.home.rglob("*"))
        if path.is_file()
    }
    assert after == before


# ===========================================================================
# Items 2-5 - the tab itself, on the real Trading Desk.
#
# Every test below drives the DESK, not a hand-built panel: the packet's whole
# claim is that these actions are reachable where the trader will look for them.
# ===========================================================================
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QApplication,
    QTabWidget,
)

import project_paths  # noqa: E402

_app = QApplication.instance() or QApplication([])

BIG_DESK = (3456, 2160)  # the trader's screen
WINDOWED_DESK = (1640, 980)  # the windowed profile `test_qt_desk_layout` uses


def _spin(times: int = 6) -> None:
    for _ in range(times):
        _app.processEvents()


def _lay_out(widget, width: int, height: int) -> None:
    """Reparent BEFORE resize/show, or a second resize in one test is ignored."""
    widget.setParent(None)
    widget.resize(width, height)
    widget.show()
    _spin()


#: The shared-home stores a desk test writes through. `conftest.py` has already
#: pointed `TRADINGBOTV3_DATA_DIR` at a temp folder, so none of these can reach
#: `C:\TradingBotData` - but they ARE shared with the rest of the suite, so the
#: fixture SAVES and RESTORES rather than blanking and walking away.
#:
#: Builder note (2026-09-13): the two SWING Focus files were missing from this
#: tuple and a swing pick leaked ACROSS tests - `test_restore_puts_a_faded_
#: pick_back_through_restore_faded` restores ROKU into `focus_swing_longs.txt`
#: (which is where a `category="swing"` add lands), and the next desk then
#: counted 25 rows where its own fixture had written 24. Added here rather than
#: worked around: no assertion is changed, and the blanked slice below still
#: means "the lists that are READ by name".
_DESK_HOME_FILES = (
    "LONGS_FILE",
    "SHORTS_FILE",
    "SWING_LONGS_FILE",
    "SWING_SHORTS_FILE",
    "FOCUS_LONGS_FILE",
    "FOCUS_SHORTS_FILE",
    "FOCUS_SWING_LONGS_FILE",
    "FOCUS_SWING_SHORTS_FILE",
    "SWING_FAVORITES_FILE",
    "WATCHLIST_INTENT_EVENTS_FILE",
    "PRICE_ALERTS_FILE",
    "FOCUS_PICK_MEMBERSHIP_FILE",
    "TRADER_ANNOTATIONS_FILE",
)

#: Blanked to "" rather than deleted: the four plain watchlists and the two
#: Focus files are READ by name and a missing one logs a warning on every load.
_DESK_HOME_BLANKED = _DESK_HOME_FILES[:8]


@pytest.fixture
def desk_home():
    """Start each desk test from an empty shared home, and put it back after."""
    paths = [getattr(project_paths, name) for name in _DESK_HOME_FILES]
    saved = {path: (path.read_bytes() if path.is_file() else None) for path in paths}
    blanked = {getattr(project_paths, name) for name in _DESK_HOME_BLANKED}

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path in blanked:
            path.write_text("", encoding="utf-8")
        elif path.exists():
            path.unlink()

    pick_feedback.clear_reviewed_today_cache()
    try:
        yield project_paths
    finally:
        for path, payload in saved.items():
            if payload is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(payload)
        pick_feedback.clear_reviewed_today_cache()


def _shutdown(widget) -> None:
    """Close a desk AND stop every timer under it before letting it go.

    `close()` + `deleteLater()` is not enough: `deleteLater` only runs on the
    NEXT event-loop pass, and a `QTimer` child keeps firing until then - or
    forever, if nothing pumps. Measured here 2026-09-13: a leaked
    `MasterAvwapPanel` refresh timer fired inside another test's
    `processEvents()` and made
    `test_ws_10a_scan_freshness.py::test_the_manifest_is_read_on_the_refresh_path_and_never_on_paint`
    report "the manifest was re-read 4 time(s) while painting" - a failure in a
    file this packet never touches, in one full-suite run out of five.
    """
    from PySide6.QtCore import QTimer

    for service in ("strength_board_service", "watchlist_tab_service"):
        stop = getattr(getattr(widget, service, None), "shutdown", None)
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001 - teardown never fails a test
                pass
    for timer in widget.findChildren(QTimer):
        timer.stop()
    widget.close()
    widget.deleteLater()
    _spin(8)


@pytest.fixture
def desk(desk_home):
    from ui.panels.trading_desk import TradingDeskPanel

    panel = TradingDeskPanel(workspace_mode="workspace")
    _lay_out(panel, *WINDOWED_DESK)
    yield panel
    _shutdown(panel)


def _tab(desk):
    """The Watchlist tab widget, found by TITLE the way the desk's own nav does."""
    for tabs in desk.findChildren(QTabWidget):
        for index in range(tabs.count()):
            if tabs.tabText(index).strip().lower() == "watchlist":
                return tabs.widget(index)
    raise AssertionError(
        "WS-WL item 2: no tab titled 'Watchlist' anywhere on the Trading Desk; "
        f"tabs found = {[[t.tabText(i) for i in range(t.count())] for t in desk.findChildren(QTabWidget)]}"
    )


def _raise_tab(desk):
    """Make the Watchlist tab current - a QShortcut in a hidden tab never fires."""
    tab = _tab(desk)
    for tabs in desk.findChildren(QTabWidget):
        if tabs.indexOf(tab) >= 0:
            tabs.setCurrentWidget(tab)
    _spin()
    return tab


def _table(tab):
    views = tab.findChildren(QAbstractItemView)
    assert views, "the Watchlist tab has no table"
    return views[0]


def _seed_rows(desk, tab):
    """One row per source, written through the REAL stores the desk owns."""
    project_paths.LONGS_FILE.write_text("AAL\n", encoding="utf-8")
    desk.focus_service.add_many(["MU"], "long", "m5")
    swing_favorites.record_favorite(symbol="BA", side="long")
    tab.refresh_now()
    _spin()
    return tab


# ---------------------------------------------------------------------------
# Item 5 - the retirement, and the inventory behind it
# ---------------------------------------------------------------------------
@pytest.mark.qt
def test_the_desk_has_a_watchlist_tab_and_the_two_nav_pages_are_gone(desk):
    from ui.app import PAGE_SPECS

    titles = [spec.title for spec in PAGE_SPECS]
    assert "Chart Review" not in titles
    assert "Focus Picks" not in titles
    attributes = [spec.attribute for spec in PAGE_SPECS]
    assert "chart_review_panel" not in attributes
    assert "trading_panel.focus_picks_panel" not in attributes
    assert len(titles) == len(set(titles)), "a page title is the nav's key"

    tab = _tab(desk)
    assert type(tab).__name__ == "WatchlistTabPanel"


#: The COMPLETE inventory of what the two retired pages could do, mapped onto
#: the method that does it on the new tab. Recon 2026-09-12 read every button,
#: context-menu item and slot on `focus_picks_panel.py`, `price_alert_board.py`
#: and `chart_review_panel.py`; the Focus Picks page has ZERO keyboard bindings,
#: so every one of these is a mouse path that has to survive.
OLD_PAGE_ACTIONS = (
    ("Focus Picks: Add", "focus_picks_panel.FocusSideEditor.add_from_input", "add_symbol"),
    ("Focus Picks: Paste", "focus_picks_panel.FocusSideEditor.paste", "paste_many"),
    ("Focus Picks: Copy", "focus_picks_panel.FocusSideEditor.copy", "copy_visible"),
    ("Focus Picks: Clear All", "focus_picks_panel.FocusSideEditor.clear_all", "clear_view"),
    ("Focus Picks: chip x", "focus_picks_panel.FocusSideEditor._remove", "remove_selected"),
    ("Focus Picks: Like this pick", "focus_picks_panel.FocusSideEditor._like", "like_selected"),
    ("Focus Picks: Not today", "focus_picks_panel.FocusSideEditor._not_today", "not_today_selected"),
    ("Focus Picks: Refresh", "focus_picks_panel.FocusSideEditor.refresh", "refresh_now"),
    ("Focus Picks: Snapshot Today", "focus_picks_panel.FocusPicksPanel.snapshot_today", "snapshot_today"),
    ("Focus Picks: alert Save", "price_alert_board.PriceAlertBoard._save_input", "arm_selected"),
    ("Focus Picks: alert Remove", "price_alert_board.PriceAlertBoard._remove_selected", "disarm_selected"),
    ("Focus Picks: alert Re-arm", "price_alert_board.PriceAlertBoard._rearm_selected", "rearm_selected"),
    ("Chart Review: Open", "chart_review_panel.ChartReviewPanel.open_symbol", "chart_selected"),
    ("Chart Review: Ctrl+L", "chart_review_panel.ChartReviewPanel.focus_lookup", "focus_lookup"),
    # A3's fade/restore never had a page of its own; the packet puts it here.
    ("Faded pick: Restore", "focus_picks.FocusPickStore.restore_faded", "restore_selected"),
)


@pytest.mark.qt
@pytest.mark.parametrize("label,old_seam,new_method", OLD_PAGE_ACTIONS, ids=[a[0] for a in OLD_PAGE_ACTIONS])
def test_every_action_the_retired_pages_had_is_reachable_from_the_watchlist_tab(
    desk, label, old_seam, new_method
):
    """The packet retires the pages ONLY after their actions land here."""
    tab = _tab(desk)
    assert callable(getattr(tab, new_method, None)), (
        f"{label} ({old_seam}) has no home on the Watchlist tab: "
        f"WatchlistTabPanel.{new_method} is missing"
    )


@pytest.mark.qt
def test_the_chart_review_lookup_shortcut_fires_from_the_watchlist_tab(desk):
    """`Ctrl+L` focused the Chart Review lookup box. It focuses the add box now.

    Bound at the PANEL's scope, not the window's: a `QShortcut` in a hidden tab
    never fires and two window-scope bindings for one sequence fire neither.
    """
    from PySide6.QtGui import QShortcut

    tab = _raise_tab(desk)
    bindings = [
        shortcut
        for shortcut in tab.findChildren(QShortcut)
        if any(sequence.toString() == "Ctrl+L" for sequence in [shortcut.key()])
    ]
    assert len(bindings) == 1, "exactly one Ctrl+L, owned by the tab"
    assert bindings[0].context() in (
        Qt.ShortcutContext.WidgetWithChildrenShortcut,
        Qt.ShortcutContext.WidgetShortcut,
    )

    tab.add_input.clearFocus()
    _spin()
    QTest.keyClick(tab, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier)
    _spin()
    assert tab.add_input.hasFocus()


@pytest.mark.qt
def test_no_price_alert_entry_is_lost_across_the_retirement(desk_home):
    """The board lived on the Focus Picks page. The entries outlive the page."""
    import price_alerts
    from ui.panels.trading_desk import TradingDeskPanel

    saved = price_alerts.save_price_alerts(
        [
            {"symbol": "AMD", "above": 150.0, "below": None, "armed_above": True,
             "armed_below": False, "armed_at": "2026-09-10", "note": "gap fill"},
            {"symbol": "INTC", "above": None, "below": 28.0, "armed_above": False,
             "armed_below": True, "armed_at": "2026-09-09", "note": ""},
        ]
    )
    assert saved, "the price-alert store premise holds"

    panel = TradingDeskPanel(workspace_mode="workspace")
    try:
        _lay_out(panel, *WINDOWED_DESK)
        entries = panel.price_alert_service.entries()
        assert sorted(entry["symbol"] for entry in entries) == ["AMD", "INTC"]
        tab = _tab(panel)
        tab.refresh_now()
        _spin()
        counts = {row.symbol: row.armed_alerts for row in tab.rows()}
        assert counts.get("AMD") == 1
        assert counts.get("INTC") == 1
    finally:
        _shutdown(panel)


# ---------------------------------------------------------------------------
# Item 2 - adding, pasting, charting, removing, restoring, arming
# ---------------------------------------------------------------------------
@pytest.mark.qt
def test_pasting_duplicates_reports_them_and_never_doubles_a_name(desk):
    """Four pasted names, one already on the list, one repeated twice.

    The file ends with TWO lines and the result says so: 1 added, 3 duplicates.
    """
    project_paths.LONGS_FILE.write_text("AMD\n", encoding="utf-8")
    tab = _tab(desk)
    tab.refresh_now()
    _spin()

    result = tab.paste_many("amd\nMSFT\nMSFT\nAMD", side="long", horizon="day")
    _spin()

    from watchlist_utils import read_watchlist_symbols

    assert read_watchlist_symbols(project_paths.LONGS_FILE) == ["AMD", "MSFT"]
    assert tuple(result.added) == ("MSFT",)
    assert len(result.duplicates) == 3


@pytest.mark.qt
def test_a_single_add_writes_the_list_the_horizon_names_and_nothing_else(desk):
    """A swing add goes to `swinglongs.txt`; `longs.txt` is not touched."""
    tab = _tab(desk)
    result = tab.add_symbol("nke", side="long", horizon="swing")
    _spin()

    from watchlist_utils import read_watchlist_symbols

    assert tuple(result.added) == ("NKE",)
    assert read_watchlist_symbols(project_paths.SWING_LONGS_FILE) == ["NKE"]
    assert read_watchlist_symbols(project_paths.LONGS_FILE) == []
    rows = intent.read_events("swinglongs", path=project_paths.WATCHLIST_INTENT_EVENTS_FILE)
    adds = [row for row in rows if row.get("action") == intent.ACTION_ADD]
    assert [(row["symbol"], row["source"]) for row in adds] == [
        ("NKE", intent.SOURCE_TRADER_EDIT)
    ]


@pytest.mark.qt
def test_a_row_click_charts_on_the_centre_chart_and_takes_no_place_in_the_queue(desk):
    """Every ticker click on the desk lands on the centre Visual Alert Review.

    A board chart is a MANUAL look: `MANUAL_CHART_TAG`, never a re-queue and
    never a skip count.
    """
    from ui.models.bounce import MANUAL_CHART_TAG

    centre = desk.alert_center
    tab = _seed_rows(desk, _tab(desk))
    waiting_before = len(getattr(centre, "_review_queue", []) or [])

    assert tab.select_symbol("AAL", side="long")
    tab.chart_selected()
    _spin()

    current = centre._current_review_alert
    assert current is not None
    assert current.symbol == "AAL"
    assert current.tag == MANUAL_CHART_TAG
    assert len(getattr(centre, "_review_queue", []) or []) == waiting_before


@pytest.mark.qt
def test_remove_asks_the_owning_store_and_a_position_row_has_no_remove(desk):
    """Four sources, four owners. Nothing here removes a name it does not own."""
    from watchlist_utils import read_watchlist_symbols

    tab = _seed_rows(desk, _tab(desk))

    assert tab.select_symbol("AAL", side="long")
    assert tab.remove_selected() is True
    _spin()
    # Lead fix 2026-09-13: this test's own _seed_rows adds MU to m5 Focus, and
    # FocusPickStore._inject_into_shared puts a Focus pick on longs.txt; removing
    # the trader's AAL through its owner must leave that injection to ITS owner
    # (measured: longs.txt == ['AAL', 'MU'] before the click). The builder's test
    # directly below removes MU through the Focus verb and reaches [].
    assert read_watchlist_symbols(project_paths.LONGS_FILE) == ["MU"]

    assert tab.select_symbol("MU", side="long")
    assert tab.remove_selected() is True
    _spin()
    assert desk.focus_service.is_focus("MU") is False

    assert tab.select_symbol("BA", side="long")
    assert tab.remove_selected() is True
    _spin()
    rows = swing_favorites.load_rows(project_paths.SWING_FAVORITES_FILE)
    assert rows[-1]["action"] == swing_favorites.ACTION_REMOVE
    assert rows[-1]["symbol"] == "BA"
    assert swing_favorites.favorites_for_session() == []


@pytest.mark.qt
def test_a_position_only_row_offers_no_remove_and_arms_no_alert(desk, monkeypatch):
    """A broker position is a read-only projection. It is not the trader's list."""
    import price_alerts

    trade = {
        "trade_id": "T1", "broker": "questrade", "account_number": "51234567",
        "account_label": "", "symbol": "NKE", "security_type": "STK",
        "currency": "USD", "direction": "LONG", "status": "OPEN",
        "opened_at": "2026-09-08T09:41:00-04:00", "closed_at": "",
        "trade_date": "2026-09-08", "quantity_opened": 100.0, "quantity_closed": 0.0,
        "average_entry_price": 50.0, "average_exit_price": 0.0, "gross_pnl": 0.0,
        "commission": 0.0, "fees": 0.0, "net_pnl": 0.0, "pnl_usd": None,
        "auto_tag_summary": "", "tag_confidence": None,
        "updated_at": "2026-09-10T16:05:00-04:00",
    }
    tab = _tab(desk)
    tab.set_rows(
        watchlist_views.build_watchlist_rows(
            shared_lists={}, focus_store=desk.focus_service.store,
            swing_favorites=(), journal_exposures=[trade], intent_events=(),
            board_rows={}, decisions_today=pick_feedback.DayDecisions(),
            armed_alerts=(), last_sync={"questrade": datetime.now().astimezone()},
        )
    )
    _spin()

    assert tab.select_symbol("NKE", side="long")
    assert tab.can_remove() is False
    assert tab.remove_selected() is False
    assert price_alerts.load_price_alerts() == []


@pytest.mark.qt
def test_restore_puts_a_faded_pick_back_through_restore_faded(desk):
    """`FocusPickStore.restore_faded` is the inverse the packet asked us to name.

    `discard_faded` clears the entry WITHOUT putting the pick back, so a Restore
    button wired to it would quietly lose the name.
    """
    store = desk.focus_service.store
    store.add("ROKU", "long", "swing", today=date(2026, 7, 1))
    faded = store.fade_stale_picks(today=date.today())
    assert [row["symbol"] for row in faded] == ["ROKU"], "the fade premise holds"

    calls = []
    original = type(store).restore_faded

    def _spy(self, symbol, side, category="m5", **kwargs):
        calls.append((symbol, side, category))
        return original(self, symbol, side, category, **kwargs)

    type(store).restore_faded = _spy
    try:
        tab = _tab(desk)
        tab.refresh_now()
        _spin()
        assert tab.select_symbol("ROKU", side="long")
        assert tab.restore_selected() is True
        _spin()
    finally:
        type(store).restore_faded = original

    assert calls and calls[0][0] == "ROKU"
    assert "ROKU" in store.focus_symbols("long", "swing")
    assert [row["symbol"] for row in store.faded_picks()] == []


@pytest.mark.qt
def test_arming_from_the_tab_writes_one_entry_through_save_entries(desk):
    """One entry, one armed leg, and the identity the store already uses."""
    import price_alerts

    tab = _seed_rows(desk, _tab(desk))
    assert tab.select_symbol("AAL", side="long")
    assert tab.arm_selected(above=14.5) is True
    _spin()

    entries = price_alerts.load_price_alerts()
    assert len(entries) == 1
    assert entries[0]["symbol"] == "AAL"
    assert entries[0]["above"] == 14.5
    assert entries[0]["armed_above"] is True
    assert entries[0]["armed_below"] is False

    tab.refresh_now()
    _spin()
    assert {row.symbol: row.armed_alerts for row in tab.rows()}["AAL"] == 1

    # Disarming keeps the entry (A2: a price alert is DISARMED, never deleted).
    assert tab.disarm_selected() is True
    _spin()
    after = price_alerts.load_price_alerts()
    assert len(after) == 1
    assert after[0]["armed_above"] is False


def _annotation_rows():
    path = project_paths.TRADER_ANNOTATIONS_FILE
    if not path.exists():
        return []
    import json

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@pytest.mark.qt
def test_a_like_from_the_tab_records_a_like_and_moves_nothing(desk):
    """A like carries zero privileges (P9). The row stays exactly where it was."""
    from ui.annotations import verdicts

    tab = _seed_rows(desk, _tab(desk))
    before = [row.symbol for row in tab.visible_rows()]

    assert tab.select_symbol("MU", side="long")
    assert tab.like_selected() is True
    _spin()

    rows = [row for row in _annotation_rows() if row.get("symbol") == "MU"]
    assert len(rows) == 1
    assert rows[0]["surface"] == verdicts.SURFACE_FOCUS_PANEL
    tab.refresh_now()
    _spin()
    assert [row.symbol for row in tab.visible_rows()] == before
    assert desk.focus_service.is_focus("MU") is True


@pytest.mark.qt
def test_not_today_drops_an_auto_pick_and_never_a_name_the_trader_typed(desk):
    """The hard invariant: a user-entered name is never auto-removed.

    Same verb, same click, two picks - and only the one the MACHINE placed may
    leave. `remove_if_auto_adopted` is the seam that decides, and the marker's
    absence means the trader owns it.
    """
    store = desk.focus_service.store
    desk.focus_service.add_many(["MINE", "AUTO"], "long", "m5")
    store.mark_auto_adopted("AUTO", "long", "m5", reason="strength board parity")
    tab = _tab(desk)
    tab.refresh_now()
    _spin()

    assert tab.select_symbol("AUTO", side="long")
    assert tab.not_today_selected() is True
    _spin()
    assert tab.select_symbol("MINE", side="long")
    assert tab.not_today_selected() is True
    _spin()

    assert desk.focus_service.is_focus("AUTO") is False
    assert desk.focus_service.is_focus("MINE") is True
    symbols = [row.get("symbol") for row in _annotation_rows()]
    assert sorted(symbols) == ["AUTO", "MINE"], "both decisions are recorded"


@pytest.mark.qt
def test_selection_survives_a_refresh(desk):
    """A refresh that inserts a name ABOVE the selection must not move it."""
    tab = _seed_rows(desk, _tab(desk))
    assert tab.select_symbol("MU", side="long")
    assert tab.selected_symbol() == "MU"

    project_paths.LONGS_FILE.write_text("AAL\nAAA\n", encoding="utf-8")
    tab.refresh_now()
    _spin()

    assert [row.symbol for row in tab.visible_rows()][0] == "AAA"
    assert tab.selected_symbol() == "MU"


@pytest.mark.qt
def test_the_view_selector_offers_the_five_views_and_filters_only(desk):
    tab = _seed_rows(desk, _tab(desk))

    tab.set_view(watchlist_views.VIEW_MY_WATCHLIST)
    _spin()
    assert [row.symbol for row in tab.visible_rows()] == ["AAL"]
    assert len(tab.rows()) == 3, "the backing list is written before any filter"

    tab.set_view(watchlist_views.VIEW_ALL)
    _spin()
    assert [row.symbol for row in tab.visible_rows()] == ["AAL", "BA", "MU"]


# ---------------------------------------------------------------------------
# Item 2 - the service, off the Qt thread
# ---------------------------------------------------------------------------
@pytest.mark.qt
def test_the_rows_are_built_on_a_worker_never_on_the_qt_thread(desk_home, monkeypatch):
    """Nothing expensive belongs on the Qt thread - and this read opens a journal."""
    import threading

    try:
        from ui.services.watchlist_tab_service import WatchlistTabService
    except ModuleNotFoundError as exc:  # RED until item 2 lands
        raise AssertionError(
            "WS-WL item 2: scripts/ui/services/watchlist_tab_service.py does not exist yet"
        ) from exc

    import watchlist_views as views

    seen: list[tuple[int, str]] = []
    original = views.build_watchlist_rows

    def _record(**kwargs):
        current = threading.current_thread()
        seen.append((threading.get_ident(), current.name))
        return original(**kwargs)

    monkeypatch.setattr(views, "build_watchlist_rows", _record)

    service = WatchlistTabService()
    try:
        assert service.refresh_now() is True
        deadline = 50
        while deadline and not seen:
            _spin(4)
            deadline -= 1
        assert seen, "the build never ran"
        assert seen[0][0] != threading.get_ident()
        assert seen[0][1] == "watchlist-tab"
    finally:
        service.shutdown()
        _spin()


# ---------------------------------------------------------------------------
# Item 4 - the Journal's link
# ---------------------------------------------------------------------------
@pytest.mark.qt
def test_the_journal_button_opens_the_watchlist_on_the_positions_view(desk_home):
    """A nav call, not a second list."""
    from ui.app import PAGE_SPECS, MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    try:
        _lay_out(window, *WINDOWED_DESK)
        assert hasattr(window.journal_panel, "positionsOnWatchlistRequested"), (
            "WS-WL item 4: the Journal has no 'Positions on the Watchlist' signal"
        )
        window.journal_panel.positionsOnWatchlistRequested.emit()
        _spin()
        current = PAGE_SPECS[window.pages.currentIndex()].title
        assert current == "Trading Desk"
        tab = _tab(window.trading_panel)
        for tabs in window.trading_panel.findChildren(QTabWidget):
            if tabs.indexOf(tab) >= 0:
                assert tabs.currentWidget() is tab
        assert tab.view() == watchlist_views.VIEW_POSITIONS
    finally:
        _shutdown(window)


# ---------------------------------------------------------------------------
# Item 2 - it has to FIT, on the trader's screen and in a window
# ---------------------------------------------------------------------------
@pytest.mark.qt
@pytest.mark.parametrize("width,height", [BIG_DESK, WINDOWED_DESK], ids=["3456x2160", "1640x980"])
def test_the_watchlist_tab_fits_without_a_horizontal_scrollbar(desk_home, width, height):
    """Half a table behind a scrollbar is the 2026-08 setups defect, again."""
    from ui.panels.trading_desk import TradingDeskPanel

    project_paths.LONGS_FILE.write_text(
        "\n".join(f"SYM{index}" for index in range(24)) + "\n", encoding="utf-8"
    )
    panel = TradingDeskPanel(workspace_mode="workspace")
    try:
        _lay_out(panel, width, height)
        tab = _raise_tab(panel)
        tab.refresh_now()
        _spin()
        table = _table(tab)
        header = table.horizontalHeader()
        assert header.count() >= 4
        for section in range(header.count()):
            assert header.sectionSize(section) > 0, f"column {section} has no width"
        assert header.length() <= table.viewport().width(), (
            f"{header.length()}px of columns in a {table.viewport().width()}px viewport"
        )
        assert len(tab.visible_rows()) == 24
    finally:
        _shutdown(panel)


# ---------------------------------------------------------------------------
# ADDED BY THE BUILDER (2026-09-13), not a rewrite of anything above.
#
# `test_remove_asks_the_owning_store_and_a_position_row_has_no_remove` asserts
# that `longs.txt` is EMPTY after the tab removes AAL. It is not, and cannot
# be: `_seed_rows` also calls `focus_service.add_many(["MU"], "long", "m5")`,
# and `FocusPickStore._inject_into_shared` puts every m5 Focus pick straight
# into `longs.txt` (CLAUDE.md, "every Focus add is injected into longs.txt /
# shorts.txt"). Measured on this branch against a scratch home:
#
#     longs.txt   = ['AAL', 'MU']
#     focus longs = ['MU']
#
# So removing AAL leaves `['MU']`. Writing `[]` instead would delete the
# injection behind the Focus store's back, and BounceBot would stop watching a
# name that is still a Focus pick - a desync. Taking the injection out is the
# Focus store's own job (`remove_everywhere`), which the same test exercises
# one line later. That assertion is left RED rather than weakened; this test
# pins what the stores actually do, so the lead can decide the line with the
# numbers in front of them.
# ---------------------------------------------------------------------------
@pytest.mark.qt
def test_removing_a_manual_name_leaves_a_focus_injection_for_its_own_owner(desk):
    """Remove AAL: `longs.txt` keeps MU, because MU is Focus's line, not a list entry."""
    from watchlist_utils import read_watchlist_symbols

    tab = _seed_rows(desk, _tab(desk))
    assert read_watchlist_symbols(project_paths.LONGS_FILE) == ["AAL", "MU"], (
        "premise: the Focus add injected MU into the shared long list"
    )

    assert tab.select_symbol("AAL", side="long")
    assert tab.remove_selected() is True
    _spin()

    assert read_watchlist_symbols(project_paths.LONGS_FILE) == ["MU"]
    assert desk.focus_service.is_focus("MU") is True

    # ...and the Focus verb is what takes the injection back out.
    assert tab.select_symbol("MU", side="long")
    assert tab.remove_selected() is True
    _spin()
    assert desk.focus_service.is_focus("MU") is False
    assert read_watchlist_symbols(project_paths.LONGS_FILE) == []


@pytest.mark.qt
def test_charting_a_typed_name_adds_it_to_nothing(desk):
    """Chart Review's "Open": look at a name that is on NO list, and keep it off.

    Added by the builder. The tester's inventory maps `ChartReviewPanel.
    open_symbol` onto `chart_selected`, which charts the SELECTED ROW - so on
    its own it cannot reach a name the desk has never heard of, which is the
    one thing the retired page could do that nothing else on the tab could.
    `chart_lookup` is that door, and like the page it was taken from it is
    read-only: the name goes in the machine-local recents and onto the chart,
    and into no watchlist, no Focus list and no CandidateRegistry.
    """
    from ui.models.bounce import MANUAL_CHART_TAG
    from watchlist_utils import read_watchlist_symbols

    tab = _tab(desk)
    centre = desk.alert_center
    waiting_before = len(getattr(centre, "_review_queue", []) or [])

    assert tab.chart_lookup("wmt") == "WMT"
    _spin()

    current = centre._current_review_alert
    assert current is not None
    assert current.symbol == "WMT"
    assert current.tag == MANUAL_CHART_TAG
    assert len(getattr(centre, "_review_queue", []) or []) == waiting_before

    for path in (
        project_paths.LONGS_FILE,
        project_paths.SHORTS_FILE,
        project_paths.SWING_LONGS_FILE,
        project_paths.SWING_SHORTS_FILE,
    ):
        assert "WMT" not in read_watchlist_symbols(path)
    assert desk.focus_service.is_focus("WMT") is False
    tab.refresh_now()
    _spin()
    assert "WMT" not in [row.symbol for row in tab.rows()]

    # Not a ticker: the status says so and nothing is charted.
    assert tab.chart_lookup("not a ticker at all") == ""
