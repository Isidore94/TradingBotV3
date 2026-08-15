"""R7 §9 step 8 - FX booking, and the totals that stop lying (B8, I5).

A USD win and a CAD loss were being added together as if the numbers meant the
same thing. For a Canadian trader filing Canadian tax that is a wrong number,
not a rounding error.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_fx as fx  # noqa: E402
from journal_analytics import build_analytics_summary, resolve_pnl_key  # noqa: E402
from journal_store import JournalStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return JournalStore(tmp_path / "trade_journal.sqlite3")


def _round_trip(store, uid: str, currency: str, day: str, entry: float, exit_price: float, qty: float = 100):
    common = dict(
        broker="QUESTRADE", account_number="5", account_label="M", account_type="",
        symbol=f"SYM{uid}", security_type="STK", currency=currency, commission=0.0, fees=0.0,
        gross_amount=None, net_amount=None, order_id="", exchange_exec_id="", raw_json="{}",
    )
    store.upsert_executions(
        [
            dict(common, execution_uid=f"QT:5:{uid}a", side="BUY", quantity=qty, price=entry,
                 timestamp=f"{day}T09:31:00-07:00", trade_date=day),
            dict(common, execution_uid=f"QT:5:{uid}b", side="SELL", quantity=qty, price=exit_price,
                 timestamp=f"{day}T11:31:00-07:00", trade_date=day),
        ]
    )


class _Valet:
    """A stand-in BoC Valet endpoint. No test in this file touches the network."""

    def __init__(self, observations: dict[str, float], *, fail: bool = False):
        self.observations = observations
        self.fail = fail
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("valet unavailable")
        currency = url.split("FX", 1)[1][:3]
        return _Response({
            "observations": [
                {"d": day, f"FX{currency}CAD": {"v": str(value)}}
                for day, value in sorted(self.observations.items())
            ]
        })


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# Booking
# ---------------------------------------------------------------------------


def test_a_rate_is_booked_once_and_read_from_the_table_afterwards(store):
    valet = _Valet({"2026-08-05": 1.37})
    summary = fx.ensure_rates(store, [(date(2026, 8, 5), "USD")], session=valet)
    assert summary["booked"] == 1
    assert valet.calls == 1

    # Second call fetches nothing: the pair is booked.
    fx.ensure_rates(store, [(date(2026, 8, 5), "USD")], session=valet)
    assert valet.calls == 1
    assert fx.stored_rate(store, date(2026, 8, 5), "USD")["rate_to_cad"] == pytest.approx(1.37)


def test_cad_needs_no_observation_and_is_never_fetched(store):
    valet = _Valet({})
    fx.ensure_rates(store, [(date(2026, 8, 5), "CAD")], session=valet)
    assert valet.calls == 0
    assert fx.stored_rate(store, date(2026, 8, 5), "CAD")["rate_to_cad"] == 1.0


def test_a_weekend_carries_the_prior_business_day_and_says_which_one(store):
    """CRA-acceptable, and auditable: effective_date names the observation used."""
    valet = _Valet({"2026-08-07": 1.36})  # Friday only
    summary = fx.ensure_rates(store, [(date(2026, 8, 9), "USD")], session=valet)  # Sunday
    assert summary["carried_back"] == 1
    booked = fx.stored_rate(store, date(2026, 8, 9), "USD")
    assert booked["rate_to_cad"] == pytest.approx(1.36)
    assert booked["effective_date"] == "2026-08-07"
    assert booked["rate_date"] == "2026-08-09"
    assert [row["rate_date"] for row in fx.carried_back_rates(store)] == ["2026-08-09"]


def test_a_failed_fetch_leaves_the_rate_missing_and_never_raises(store):
    summary = fx.ensure_rates(store, [(date(2026, 8, 5), "USD")], session=_Valet({}, fail=True))
    assert summary["booked"] == 0
    assert summary["errors"] and "valet unavailable" in summary["errors"][0]["message"]
    assert fx.stored_rate(store, date(2026, 8, 5), "USD") is None


def test_a_rate_with_no_observation_anywhere_in_the_window_is_reported_not_invented(store):
    summary = fx.ensure_rates(store, [(date(2026, 8, 5), "XYZ")], session=_Valet({"2020-01-01": 1.0}))
    assert summary["booked"] == 0
    assert summary["unavailable"][0]["currency"] == "XYZ"


# ---------------------------------------------------------------------------
# What the trades get
# ---------------------------------------------------------------------------


def test_an_unconverted_trade_is_null_and_never_zero(store):
    """I5. Zero is a number someone will add up; NULL is a state the UI renders."""
    _round_trip(store, "u1", "USD", "2026-08-05", 100.0, 110.0)
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT * FROM trades").fetchone())
    assert row["net_pnl"] == pytest.approx(1000.0)
    assert row["net_pnl_cad"] is None and row["fx_rate"] is None


def test_booking_a_rate_then_rebuilding_converts_the_trade(store):
    _round_trip(store, "u1", "USD", "2026-08-05", 100.0, 110.0)
    store.rebuild_trades(refresh_tags=False)
    fx.seed_rate(store, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    store.book_cad_values()
    with store.connection() as conn:
        row = dict(conn.execute("SELECT * FROM trades").fetchone())
    assert row["net_pnl_cad"] == pytest.approx(1370.0)
    assert row["fx_rate"] == pytest.approx(1.37)
    assert row["fx_rate_date"] == "2026-08-05"


def test_a_cad_trade_converts_by_identity(store):
    _round_trip(store, "c1", "CAD", "2026-08-05", 80.0, 85.0)
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        row = dict(conn.execute("SELECT * FROM trades").fetchone())
    assert row["net_pnl_cad"] == pytest.approx(row["net_pnl"])
    assert row["fx_rate"] == 1.0


def test_removing_a_rate_returns_the_trade_to_unconverted(store):
    """Booking is a projection of the rate table, not a one-way stamp."""
    _round_trip(store, "u1", "USD", "2026-08-05", 100.0, 110.0)
    fx.seed_rate(store, day="2026-08-05", currency="USD", rate_to_cad=1.37)
    store.rebuild_trades(refresh_tags=False)
    with store.connection() as conn:
        assert conn.execute("SELECT net_pnl_cad FROM trades").fetchone()[0] == pytest.approx(1370.0)
        conn.execute("DELETE FROM fx_rates")
    store.book_cad_values()
    with store.connection() as conn:
        assert conn.execute("SELECT net_pnl_cad FROM trades").fetchone()[0] is None


def test_the_pairs_a_journal_needs_are_derivable_without_a_network(store):
    _round_trip(store, "u1", "USD", "2026-08-05", 100.0, 110.0)
    _round_trip(store, "c1", "CAD", "2026-08-05", 80.0, 85.0)
    store.rebuild_trades(refresh_tags=False)
    assert fx.rates_needed_for_trades(store) == [(date(2026, 8, 5), "USD")]


def test_coverage_is_reportable_for_the_health_tab(store):
    _round_trip(store, "u1", "USD", "2026-08-05", 100.0, 110.0)
    _round_trip(store, "c1", "CAD", "2026-08-05", 80.0, 85.0)
    store.rebuild_trades(refresh_tags=False)
    coverage = fx.describe_coverage(store)
    assert coverage["trades"] == 2 and coverage["converted"] == 1
    assert coverage["unconverted"] == [{"currency": "USD", "trades": 1}]


# ---------------------------------------------------------------------------
# B8: the totals
# ---------------------------------------------------------------------------


def test_one_currency_sums_its_own_numbers():
    trades = [
        {"currency": "USD", "status": "CLOSED", "net_pnl": 100.0, "net_pnl_cad": None},
        {"currency": "USD", "status": "CLOSED", "net_pnl": -40.0, "net_pnl_cad": None},
    ]
    assert resolve_pnl_key(trades) == ("net_pnl", "")
    assert build_analytics_summary(trades)["overall"]["net_pnl"] == pytest.approx(60.0)


def test_mixed_currencies_all_converted_sum_in_cad():
    trades = [
        {"currency": "USD", "status": "CLOSED", "net_pnl": 100.0, "net_pnl_cad": 137.0},
        {"currency": "CAD", "status": "CLOSED", "net_pnl": -40.0, "net_pnl_cad": -40.0},
    ]
    key, note = resolve_pnl_key(trades)
    assert key == "net_pnl_cad" and "CAD" in note
    assert build_analytics_summary(trades)["overall"]["net_pnl"] == pytest.approx(97.0)


def test_mixed_currencies_with_anything_unconverted_refuses_to_total():
    """The defect, and the only honest answer to it.

    Adding 100 USD to -40 CAD gives 60 of nothing. Quietly dropping the
    unconverted row instead would give a total that looks right and is not.
    """
    trades = [
        {"currency": "USD", "status": "CLOSED", "net_pnl": 100.0, "net_pnl_cad": None},
        {"currency": "CAD", "status": "CLOSED", "net_pnl": -40.0, "net_pnl_cad": -40.0},
    ]
    key, note = resolve_pnl_key(trades)
    assert key == ""
    assert "no booked FX rate" in note and "USD" in note

    summary = build_analytics_summary(trades)
    assert summary["pnl_key"] == ""
    assert summary["overall"]["net_pnl"] is None, "a refused total is None, never a wrong number"
    assert summary["pnl_note"] == note
    assert summary["currencies"] == ["CAD", "USD"]


def test_an_empty_selection_does_not_claim_a_currency_problem():
    key, note = resolve_pnl_key([])
    assert key == "net_pnl" and note == ""
