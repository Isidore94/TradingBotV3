"""Golden replays of the two real 2026-09-23 Pullback alerts.

Real tapes (see the fixture's ``provider_assumptions``): QDEL SHORT was a correct alert and
must keep firing; ARM LONG ``reclaim_then_lrsi`` fired with no pullback at
all - ARM sat ~50 points above its M30 75-SMA - and is the case the dip gate
exists for.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE = ROOT_DIR / "tests" / "fixtures" / "pullback_dip_gate_real_tapes_v1.json"


def _load():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(payload["now"])
    symbols = {}
    for symbol, entry in payload["symbols"].items():
        parsed = dict(entry)
        for key in ("M15", "M30", "D1"):
            parsed[key] = [
                dict(bar, dt=datetime.fromisoformat(bar["dt"])) for bar in entry[key]
            ]
        parsed["armed_at"] = datetime.fromisoformat(entry["armed_at"])
        symbols[symbol] = parsed
    return now, symbols


def _evaluate_m30(symbol: str):
    from indicators.pullback_sma_reclaim import evaluate

    now, symbols = _load()
    entry = symbols[symbol]
    return evaluate(
        entry["M30"],
        side=entry["side"],
        sma_length=75,
        bar_minutes=30,
        armed_at=entry["armed_at"],
        now=now,
        companion_bars=entry["M15"],
        companion_minutes=15,
        daily_bars=entry["D1"],
    )


def test_qdel_short_m30_loss_with_lrsi_cross_still_fires_on_the_real_tape():
    result = _evaluate_m30("QDEL")
    assert result is not None
    fires = {fire.trigger: fire for fire in result.fired}
    assert set(fires) == {"sma_reclaim_lrsi"}
    fire = fires["sma_reclaim_lrsi"]
    assert fire.bar_dt == datetime(2026, 9, 23, 6, 30)
    assert fire.cross_bar_dt == datetime(2026, 9, 22, 12, 30)
    assert fire.cross_timeframe == "M30"
    assert fire.lrsi_from_below_50 is True
    assert round(fire.sma, 4) == round(10.923817354838054, 4)


def test_arm_long_reclaim_then_lrsi_no_longer_fires_without_a_pullback():
    # v1 fired reclaim_then_lrsi on the M15 06:45 cross. ARM's last touch of
    # its M30 75-SMA is weeks behind the cross: the dip gate calls it stale.
    result = _evaluate_m30("ARM")
    assert result is not None
    assert result.fired == ()
    assert result.details.get("gate_blocked") == {"reclaim_then_lrsi": "dip_stale"}


def test_qdel_fire_carries_the_dip_it_came_from():
    # QDEL rallied up to the SMA on 09-21 08:00 (high 0.1813 under a 0.1830
    # tolerance = 0.2 x D1 ATR 0.9149) and its short LRSI flipped under 20 on
    # that same bar. The margin is thin; this pins it.
    fire = _evaluate_m30("QDEL").fired[0]
    assert fire.dip_path == "touch"
    assert fire.dip_bar_dt == datetime(2026, 9, 21, 8, 0)
    assert fire.bear_flip_bar_dt == datetime(2026, 9, 21, 8, 0)
    assert round(fire.d1_atr, 4) == 0.9149


def test_the_fixture_contract_holds_and_matches_what_the_rule_says():
    from conftest import load_fixture_contract

    contract = load_fixture_contract("pullback_dip_gate_real_tapes_v1")
    assert contract.raw_input_digest() == contract["raw_input_sha256"]
    expected = contract["expected"]["v2"]
    qdel = _evaluate_m30("QDEL").fired[0]
    assert qdel.trigger == expected["QDEL"]["trigger"]
    assert qdel.dip_path == expected["QDEL"]["dip_path"]
    assert qdel.dip_bar_dt.isoformat() == expected["QDEL"]["dip_bar_dt"]
    assert qdel.bear_flip_bar_dt.isoformat() == expected["QDEL"]["bear_flip_bar_dt"]
    assert dict(_evaluate_m30("ARM").details) == expected["ARM"]
