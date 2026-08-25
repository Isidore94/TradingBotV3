"""R7 true USD conversion - the 2026-08-18 deferral, reversed 2026-08-24.

Recorded in `docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md` §2. The
deferral's own reasoning was right and is preserved: *"the FX table books CAD
only, and inventing a rate is exactly the dishonesty the currency refusal was
built to prevent."* What changed is that the FX table no longer books CAD only -
R7's I5 chain fetches a BoC observation per (date, currency) nightly, so a real
USD/CAD rate for each trade's own session is available to book from.

So the three rules that made the CAD path trustworthy carry over unchanged:

* **booked once, at import, never fetched at render** - a tax figure that moves
  when you look at it is not a tax figure, and a display figure that moves is
  just as confusing;
* **a missing rate renders "unconverted"** - never 0, never the native number
  quietly relabelled;
* **a weekend or holiday carries the prior business day's observation**, and the
  booking says which one.

USD is a DISPLAY currency here, not the tax currency: CAD stays the booked
tax-grade value (I5), and the blended-tax badge (I6) is untouched by any of this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_fx as fx  # noqa: E402
from journal_store import JournalStore  # noqa: E402
from ui.services import journal_feed  # noqa: E402


@pytest.fixture
def feed(tmp_path, monkeypatch):
    store = JournalStore(tmp_path / "trade_journal.sqlite3")
    monkeypatch.setattr(journal_feed, "_STORE", store)
    monkeypatch.setattr(journal_feed, "_store", lambda: store)
    return store


def _execution(uid, side, quantity, price, day, **overrides):
    row = {
        "execution_uid": uid, "broker": "QUESTRADE", "account_number": "51830546",
        "account_label": "TFSA", "account_type": "TFSA", "symbol": "AAPL",
        "security_type": "STK", "currency": "USD", "side": side, "quantity": quantity,
        "price": price, "timestamp": f"{day}T09:31:00-07:00", "trade_date": day,
        "commission": 4.95, "fees": 0.05, "gross_amount": None, "net_amount": None,
        "order_id": "", "exchange_exec_id": "", "raw_json": "{}",
    }
    row.update(overrides)
    return row


def _round_trip(store, uid, day, entry, exit_price, **overrides):
    store.upsert_executions(
        [
            _execution(f"QT:{uid}a", "BUY", 100, entry, day, **overrides),
            _execution(f"QT:{uid}b", "SELL", 100, exit_price, day, **overrides),
        ]
    )


def _trade(symbol: str):
    return [t for t in journal_feed.load_trades() if t.symbol == symbol][0]


# ---------------------------------------------------------------------------
# The conversion itself, against known rates
# ---------------------------------------------------------------------------


def test_a_cad_trade_converts_to_usd_at_its_own_sessions_booked_rate(feed):
    """Three sessions, three different rates, each trade using its own.

    One rate applied across a year is the estimate the manual field already
    offers and labels as such. The point of booking is that a trade taken on a
    1.28 day is not worth what a 1.42 day says it is.
    """
    for index, (day, rate, entry, exit_price) in enumerate(
        (
            ("2026-08-05", 1.2800, 80.0, 90.0),
            ("2026-08-06", 1.3700, 80.0, 85.0),
            ("2026-08-07", 1.4200, 80.0, 70.0),
        )
    ):
        _round_trip(
            feed, f"cad{index}", day, entry, exit_price,
            currency="CAD", symbol=f"SHOP{index}.TO",
        )
        fx.seed_rate(feed, day=day, currency="USD", rate_to_cad=rate)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    for index, rate in enumerate((1.2800, 1.3700, 1.4200)):
        trade = _trade(f"SHOP{index}.TO")
        native = trade.raw["net_pnl"]
        value, label = journal_feed.convert_amount(trade, "USD")
        assert label == "USD"
        assert value == pytest.approx(native / rate), f"row {index} used the wrong session"


def test_a_usd_trade_is_its_own_native_number_and_needs_no_rate(feed):
    """Converting USD to USD through a rate would introduce rounding for nothing."""
    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    trade = _trade("AAPL")
    value, label = journal_feed.convert_amount(trade, "USD")
    assert label == "USD"
    assert value == pytest.approx(trade.raw["net_pnl"])


def test_a_missing_rate_renders_unconverted_never_zero_and_never_native(feed):
    """I5's second rule, which is the whole reason the deferral existed."""
    _round_trip(feed, "c", "2026-08-05", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    trade = _trade("SHOP.TO")
    value, label = journal_feed.convert_amount(trade, "USD")
    assert value is None, "a CAD number relabelled USD is the exact defect this prevents"
    assert label == "unconverted"


def test_the_booked_usd_value_is_stored_not_computed_at_render(feed):
    """Booked once at import. The render seam reads a column, it does not divide.

    A rate resolved while a report is open makes the same trade worth different
    amounts on different days.
    """
    _round_trip(feed, "c", "2026-08-05", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    trade = _trade("SHOP.TO")
    assert trade.raw["net_pnl_usd"] == pytest.approx(trade.raw["net_pnl"] / 1.37)
    assert trade.raw["fx_usd_rate"] == pytest.approx(1.37)
    assert trade.raw["fx_usd_rate_date"] == "2026-08-05"


def test_a_carried_back_weekend_rate_records_the_day_it_came_from(feed):
    """The BoC publishes nothing on a day it is closed; saying which observation
    was used is what makes the number auditable."""
    _round_trip(feed, "c", "2026-08-08", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    fx.seed_rate(
        feed, day="2026-08-08", currency="USD", rate_to_cad=1.31,
        effective_date="2026-08-07",
    )
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    trade = _trade("SHOP.TO")
    assert trade.raw["fx_usd_rate_date"] == "2026-08-07"


def test_rebooking_is_idempotent(feed):
    _round_trip(feed, "c", "2026-08-05", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()
    first = _trade("SHOP.TO").raw["net_pnl_usd"]
    feed.book_currency_values()
    assert _trade("SHOP.TO").raw["net_pnl_usd"] == pytest.approx(first)


def test_a_rate_that_disappears_clears_the_booking_rather_than_leaving_it_stale(feed):
    """The CAD pass already does this; a stale USD number would be worse than none."""
    _round_trip(feed, "c", "2026-08-05", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()
    assert _trade("SHOP.TO").raw["net_pnl_usd"] is not None

    with feed.connection() as conn:
        conn.execute("DELETE FROM fx_rates WHERE currency = 'USD'")
    feed.book_currency_values()
    assert _trade("SHOP.TO").raw["net_pnl_usd"] is None
    assert _trade("SHOP.TO").raw["fx_usd_rate"] is None


# ---------------------------------------------------------------------------
# What the nightly pass has to ASK for, or none of the above can happen
# ---------------------------------------------------------------------------


def test_the_nightly_pass_asks_for_a_usd_rate_on_every_non_usd_session(feed):
    """The gap that made this impossible before.

    `rates_needed_for_trades` asked only for each trade's OWN currency, so a
    CAD-only session never had a USD rate booked and could never be shown in
    USD however honest the render seam was.
    """
    _round_trip(feed, "c", "2026-08-05", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    _round_trip(feed, "u", "2026-08-06", 100.0, 110.0)
    feed.rebuild_trades(refresh_tags=False)

    needed = set(fx.rates_needed_for_trades(feed))
    from datetime import date

    assert (date(2026, 8, 5), "USD") in needed, "a CAD session needs a USD rate to display in USD"
    assert (date(2026, 8, 6), "USD") in needed


# ---------------------------------------------------------------------------
# Analytics: totals, and the behaviour that must NOT change
# ---------------------------------------------------------------------------


def test_a_mixed_selection_now_totals_in_usd_at_the_booked_rates(feed):
    import journal_analytics

    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0, symbol="AAPL")
    _round_trip(feed, "c", "2026-08-06", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.40)
    fx.seed_rate(feed, day="2026-08-06", currency="USD", rate_to_cad=1.25)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    rows = [trade.raw for trade in journal_feed.load_trades()]
    key, note = journal_analytics.resolve_pnl_key(rows, "USD")
    assert key == "net_pnl_usd"
    assert "booked" in note.lower() and "estimate" not in note.lower()


def test_an_unconvertible_row_still_refuses_the_usd_total(feed):
    """A total that silently omits the rows it could not convert is worse than
    no total. Unchanged from before this packet."""
    import journal_analytics

    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0, symbol="AAPL")
    _round_trip(feed, "c", "2026-08-06", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.40)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    rows = [trade.raw for trade in journal_feed.load_trades()]
    key, note = journal_analytics.resolve_pnl_key(rows, "USD")
    assert key == ""
    assert "not shown" in note


def test_the_manual_estimate_still_covers_what_booking_cannot(feed):
    """The manual display rate keeps its job and keeps its ESTIMATE label.

    It is the fallback for a session with no booked USD observation at all, and
    it must never be mistaken for the booked figure - so the note still says so,
    and the booked path is preferred whenever it can answer.
    """
    import journal_analytics

    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0, symbol="AAPL")
    _round_trip(feed, "c", "2026-08-06", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()  # no USD rate booked for 08-06: nothing to book from
    fx.set_manual_usd_rate(1.35)
    try:
        rows = [trade.raw for trade in journal_feed.load_trades()]
        key, note = journal_analytics.resolve_pnl_key(rows, "USD")
        assert key == journal_analytics.USD_ESTIMATE_KEY
        assert "ESTIMATE" in note
    finally:
        # The manual rate lives in machine-local settings, which the suite
        # sandboxes per SESSION, not per test. Leaving it set would hand the
        # next test an estimate it never asked for.
        from project_paths import save_local_setting

        save_local_setting(fx.MANUAL_USD_RATE_SETTING, None)


def test_the_none_bucket_is_still_excluded_rather_than_zeroed(feed):
    """R7's 2026-08-18 analytics rule, unchanged by this packet: a bucket whose
    total is None means 'mixed currencies with unconverted rows', and a zero bar
    would claim the setup broke even."""
    import journal_analytics

    rows = [
        {"status": "CLOSED", "currency": "USD", "net_pnl": 10.0, "net_pnl_cad": 14.0,
         "net_pnl_usd": 10.0, "setup_tags": "a"},
        {"status": "CLOSED", "currency": "CAD", "net_pnl": 10.0, "net_pnl_cad": 10.0,
         "net_pnl_usd": None, "setup_tags": "b"},
    ]
    key, note = journal_analytics.resolve_pnl_key(rows, "USD")
    assert key == "" and "not shown" in note


def test_cad_remains_the_tax_grade_value_and_is_untouched(feed):
    """USD is a DISPLAY currency. Nothing here may move the booked CAD number."""
    _round_trip(feed, "c", "2026-08-05", 80.0, 90.0, currency="CAD", symbol="SHOP.TO")
    _round_trip(feed, "u", "2026-08-05", 100.0, 110.0, symbol="AAPL")
    fx.seed_rate(feed, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    feed.rebuild_trades(refresh_tags=False)
    feed.book_currency_values()

    cad_trade = _trade("SHOP.TO")
    usd_trade = _trade("AAPL")
    assert cad_trade.raw["net_pnl_cad"] == pytest.approx(cad_trade.raw["net_pnl"])
    assert usd_trade.raw["net_pnl_cad"] == pytest.approx(usd_trade.raw["net_pnl"] * 1.37)
