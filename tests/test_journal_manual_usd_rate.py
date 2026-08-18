"""The manual USD display rate: an estimate that can never pass for a booking.

Trader request, 2026-08-17. `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` deferred
"true booked USD conversion" - a mixed-currency selection REFUSES a USD total
rather than relabelling native money. That refusal is correct and stays. This
adds the one thing the trader asked for: somewhere to type today's rate so a
mixed selection can show an approximate USD total.

Everything here is built to keep that approximation from ever being mistaken
for the booked, point-in-time CAD figures the tax path depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_analytics as analytics  # noqa: E402
import journal_fx as fx  # noqa: E402


@pytest.fixture
def settings(monkeypatch):
    """An in-memory local-settings store, so no test writes the real one."""
    store: dict[str, object] = {}
    monkeypatch.setattr(
        "project_paths.get_local_setting", lambda key, default=None: store.get(key, default)
    )
    monkeypatch.setattr(
        "project_paths.save_local_setting", lambda key, value: store.__setitem__(key, value)
    )
    return store


def trade(currency, net_pnl, net_pnl_cad=None, status="CLOSED"):
    return {
        "status": status,
        "currency": currency,
        "net_pnl": net_pnl,
        "net_pnl_cad": net_pnl_cad,
    }


class TestStoringTheRate:
    def test_a_stored_rate_round_trips_with_its_stamp(self, settings):
        stored = fx.set_manual_usd_rate("1.3750")

        assert stored["rate_cad_per_usd"] == pytest.approx(1.375)
        assert stored["source"] == "MANUAL_DISPLAY"
        assert stored["entered_at"]
        assert fx.manual_usd_rate()["rate_cad_per_usd"] == pytest.approx(1.375)

    def test_a_blank_clears_it(self, settings):
        fx.set_manual_usd_rate("1.375")
        assert fx.set_manual_usd_rate("") is None
        assert fx.manual_usd_rate() is None

    @pytest.mark.parametrize("bad", ["13.75", "0.1375", "0", "-1.375"])
    def test_an_out_of_range_rate_is_refused_not_stored(self, settings, bad):
        """A fat-fingered decimal would rescale every total silently."""
        with pytest.raises(ValueError):
            fx.set_manual_usd_rate(bad)
        assert fx.manual_usd_rate() is None

    def test_garbage_reads_back_as_unset_rather_than_crashing(self, settings):
        settings[fx.MANUAL_USD_RATE_SETTING] = "not a number"
        assert fx.manual_usd_rate() is None

    def test_it_never_lands_in_the_booked_fx_table(self, settings):
        """The whole point: this is a setting, not an observation."""
        fx.set_manual_usd_rate("1.375")

        assert fx.MANUAL_USD_RATE_SETTING in settings
        assert fx.BOC_SOURCE not in str(settings)


class TestTheEstimate:
    def test_no_rate_leaves_the_existing_refusal_untouched(self, settings):
        trades = [trade("USD", 100.0), trade("CAD", 50.0, net_pnl_cad=50.0)]

        key, note = analytics.resolve_pnl_key(trades, "USD")

        assert key == ""
        assert "not shown" in note
        assert "Enter a USD/CAD rate" in note

    def test_a_rate_converts_the_cad_side_from_its_booked_value(self, settings):
        fx.set_manual_usd_rate("1.25")
        trades = [trade("USD", 100.0), trade("CAD", 50.0, net_pnl_cad=50.0)]

        key, note = analytics.resolve_pnl_key(trades, "USD")

        assert key == analytics.USD_ESTIMATE_KEY
        assert trades[0][analytics.USD_ESTIMATE_KEY] == pytest.approx(100.0)
        assert trades[1][analytics.USD_ESTIMATE_KEY] == pytest.approx(40.0)

    def test_the_note_says_estimate_and_says_it_is_not_tax(self, settings):
        fx.set_manual_usd_rate("1.25")
        trades = [trade("USD", 100.0), trade("CAD", 50.0, net_pnl_cad=50.0)]

        _key, note = analytics.resolve_pnl_key(trades, "USD")

        assert "ESTIMATE" in note
        assert "1.2500" in note
        assert "Not a tax figure" in note

    def test_a_rate_cannot_manufacture_a_missing_observation(self, settings):
        """An unconverted row stays unconverted; the estimate rides the booking."""
        fx.set_manual_usd_rate("1.25")
        trades = [trade("USD", 100.0), trade("CAD", 50.0, net_pnl_cad=None)]

        key, note = analytics.resolve_pnl_key(trades, "USD")

        assert key == ""
        assert "no booked FX rate" in note

    def test_a_usd_only_selection_never_uses_the_estimate(self, settings):
        fx.set_manual_usd_rate("1.25")
        trades = [trade("USD", 100.0), trade("USD", -20.0)]

        key, note = analytics.resolve_pnl_key(trades, "USD")

        assert key == "net_pnl"
        assert note == ""

    def test_cad_mode_is_completely_unaffected_by_the_manual_rate(self, settings):
        """The booked path must not notice this feature exists."""
        fx.set_manual_usd_rate("1.25")
        trades = [trade("USD", 100.0, net_pnl_cad=137.5), trade("CAD", 50.0, net_pnl_cad=50.0)]

        key, note = analytics.resolve_pnl_key(trades, "CAD")

        assert key == "net_pnl_cad"
        assert "booked rate" in note
        assert "ESTIMATE" not in note

    def test_the_estimate_column_is_named_so_a_csv_cannot_mislead(self):
        assert "estimated" in analytics.USD_ESTIMATE_KEY
