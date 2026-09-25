"""P2-8 8b - a second, display-only axis on the regime line from the internals.

Fixtures are `market_internals.build_internals_snapshot` outputs. The rule is
`market_axes` (`market_axes_v1`); missing or stale internals omit the axis.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_axes  # noqa: E402
from market_internals import build_internals_snapshot  # noqa: E402

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 9, 24, 11, 0, tzinfo=ET)


def _series(prior_close: float, last_close: float) -> list[dict]:
    return [
        {"dt": datetime(2026, 9, 23, 15, 55, tzinfo=ET), "close": prior_close},
        {"dt": datetime(2026, 9, 24, 10, 55, tzinfo=ET), "close": last_close},
    ]


def _snapshot(**moves) -> dict:
    """moves: symbol -> percent move on the day. SPY defaults to flat."""
    bars = {symbol: _series(100.0, 100.0 * (1 + pct / 100.0)) for symbol, pct in moves.items()}
    bars.setdefault("SPY", _series(100.0, 100.0))
    return build_internals_snapshot(bars, as_of=NOW.isoformat(timespec="minutes"))


def test_fixture_vol_bid_and_narrow_tape():
    snap = _snapshot(SPY=-0.5, VXX=4.0, RSP=-1.0)
    assert market_axes.internals_phrases(snap) == ("vol bid", "narrow tape")
    assert market_axes.regime_line("bearish_weak", snap, now=NOW) == "bearish_weak + vol bid + narrow tape"


def test_fixture_every_axis_in_the_fixed_order():
    snap = _snapshot(SPY=0.5, VXX=-3.0, RSP=1.0, HYG=0.5, MAGS=-0.5, TLT=-0.5)
    assert market_axes.internals_phrases(snap) == (
        "vol offered", "broad tape", "credit firm", "mega-caps lag", "bonds offered",
    )
    axis = market_axes.internals_axis(snap, now=NOW)
    # MAGS lagging SPY is risk_off in `market_internals`, so the tape is a tilt,
    # and a tilt carries no lean (it is never graded).
    assert axis["state"] == "risk_on_tilt" and axis["lean"] == ""
    clean = market_axes.internals_axis(_snapshot(SPY=0.5, VXX=-3.0, RSP=1.0), now=NOW)
    assert clean["state"] == "risk_on" and clean["lean"] == "up"
    # The chip carries the first two phrases; the tooltip has them all.
    assert market_axes.regime_line("bullish_weak", snap, now=NOW) == "bullish_weak + vol offered + broad tape"


def test_flat_readings_are_left_out():
    snap = _snapshot(SPY=0.0, VXX=0.05, RSP=0.02)
    assert market_axes.internals_phrases(snap) == ()
    assert market_axes.regime_line("neutral_chop", snap, now=NOW) == "neutral_chop"


def test_missing_internals_omit_the_axis_rather_than_guess():
    assert market_axes.internals_axis(None) is None
    assert market_axes.internals_axis(build_internals_snapshot({}, as_of=NOW.isoformat())) is None
    assert market_axes.regime_line("bearish_weak", None) == "bearish_weak"


def test_a_stale_snapshot_is_missing():
    snap = _snapshot(SPY=-0.5, VXX=4.0)
    later = NOW + timedelta(minutes=market_axes.INTERNALS_MAX_AGE_MINUTES + 1)
    assert market_axes.internals_axis(snap, now=later) is None
    assert market_axes.internals_axis(snap, now=NOW + timedelta(minutes=5)) is not None
    assert market_axes.regime_line("bearish_weak", {"readings": {}}, now=NOW) == "bearish_weak"


def test_the_rule_is_versioned():
    assert market_axes.AXES_RULE_VERSION == "market_axes_v1"
    read = market_axes.morning_read("2026-09-23", d1_label="mixed", breadth_row=None, internals_snapshot=None)
    assert read["rule_version"] == market_axes.AXES_RULE_VERSION


def test_the_regime_chip_shows_the_second_axis():
    from ui.panels.bounce_panel import format_auto_regime_reading

    snap = _snapshot(SPY=-0.5, VXX=4.0, RSP=-1.0)
    reading = {"label": "bearish_weak", "env_key": "bearish_weak",
               "internals_axis": market_axes.internals_axis(snap, now=NOW)}
    chip, tip = format_auto_regime_reading(reading)
    assert chip == "Auto: bearish_weak + vol bid + narrow tape"
    assert "Internals (display only): vol bid + narrow tape" in tip


def test_the_chip_is_unchanged_without_internals():
    from ui.panels.bounce_panel import format_auto_regime_reading

    chip, _tip = format_auto_regime_reading({"label": "bearish_weak", "env_key": "bearish_weak"})
    assert chip == "Auto: bearish_weak"


def test_the_service_attaches_the_axis_from_the_bots_recorded_internals(monkeypatch):
    from ui.services import bounce_service as module

    snap = _snapshot(SPY=-0.5, VXX=4.0, RSP=-1.0)
    snap["as_of"] = datetime.now(ET).isoformat(timespec="minutes")
    base = {"label": "bearish_weak", "env_key": "bearish_weak"}

    class Bot:
        latest_market_internals = snap

        def get_auto_regime_reading(self):
            return base

        def entry_assist_state(self):
            return {}

    got = module.with_internals_axis(base, Bot())
    assert got["internals_axis"]["phrases"][:2] == ("vol bid", "narrow tape")
    assert "internals_axis" not in base  # the bot's dict is not mutated

    class NoInternals(Bot):
        latest_market_internals = None

    assert module.with_internals_axis(base, NoInternals()) == base
