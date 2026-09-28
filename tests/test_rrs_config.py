"""The desk-wide RRS settings (scripts/rrs_config.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rrs_config  # noqa: E402


def test_rolling_is_the_default_with_the_trader_cutoff(monkeypatch):
    monkeypatch.delenv("TRADINGBOTV3_RRS_ENGINE", raising=False)
    assert rrs_config.use_rolling()
    assert rrs_config.cutoff() == 1.0
    assert rrs_config.DESK_TO_ROLLING == 0.5


def test_the_environment_switches_everything_back(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", "desk")
    assert not rrs_config.use_rolling()
    assert rrs_config.cutoff() == 2.0


def test_candle_sizes_sample_the_m5_reads():
    assert rrs_config.intraday_config(5) is rrs_config.INTRADAY
    assert rrs_config.intraday_config(15).sample_every == 3
    assert rrs_config.intraday_config(60).sample_every == 12
    assert rrs_config.intraday_config(60).atr_mode == "hourly"
