"""Trader, 2026-09-23: hide Oil & Gas and Real Estate names from the views.

One switch (default ON) shared by the setups table, the Alert Center and the
phone report. Display only: nothing is deleted, the scan and the stores still
record every name, and an unknown classification is always shown.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CSV_TEXT = (
    "symbol,sectorKey,industryKey,sector,industry,updated_utc\n"
    "APA,energy,oil-gas-e-p,Energy,Oil & Gas E&P,2026-03-13T14:00:47Z\n"
    "AM,energy,oil-gas-midstream,Energy,Oil & Gas Midstream,2026-03-19T18:51:32Z\n"
    "ADC,real-estate,reit-retail,Real Estate,REIT - Retail,2026-03-31T14:57:57Z\n"
    "CCJ,energy,uranium,Energy,Uranium,2026-03-13T14:25:45Z\n"
    "BTU,energy,thermal-coal,Energy,Thermal Coal,2026-03-18T15:05:24Z\n"
    "NVDA,technology,semiconductors,Technology,Semiconductors,2026-03-13T14:00:47Z\n"
)


@pytest.fixture
def classified(tmp_path, monkeypatch):
    """A tmp classification CSV and an in-memory settings store."""
    import project_paths
    import sector_exclusion

    path = tmp_path / "symbol_classification.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    monkeypatch.setattr(project_paths, "SYMBOL_CLASSIFICATION_CACHE_FILE", path)
    settings: dict[str, object] = {}
    monkeypatch.setattr(
        project_paths, "get_local_setting", lambda key, default=None: settings.get(key, default)
    )
    monkeypatch.setattr(
        project_paths, "save_local_setting", lambda key, value: settings.__setitem__(key, value)
    )
    sector_exclusion.clear_cache()
    yield settings
    sector_exclusion.clear_cache()


# --------------------------------------------------------------------------- the rule
@pytest.mark.parametrize(
    ("sector", "industry", "sector_key", "industry_key", "expected"),
    [
        ("Energy", "Oil & Gas Midstream", "", "", True),
        ("Energy", "Oil & Gas E&P", "", "", True),
        ("Energy", "oil & gas integrated", "", "", True),
        ("Energy", "Oil & Gas Equipment & Services", "", "", True),
        ("Energy", "Oil & Gas Refining & Marketing", "", "", True),
        ("Energy", "Oil & Gas Drilling", "", "", True),
        ("", "", "", "oil-gas-midstream", True),
        ("Real Estate", "REIT - Residential", "", "", True),
        ("real estate", "", "", "", True),
        ("", "REIT - Residential", "", "", True),
        ("", "", "real-estate", "", True),
        ("Energy", "Uranium", "energy", "uranium", False),
        ("Energy", "Thermal Coal", "", "", False),
        ("Technology", "Solar", "", "", False),
        ("", "", "", "", False),
        ("Technology", "Semiconductors", "", "", False),
    ],
)
def test_is_excluded(sector, industry, sector_key, industry_key, expected):
    from sector_exclusion import is_excluded

    assert is_excluded(sector, industry, sector_key, industry_key) is expected


def test_the_setting_defaults_to_on_and_round_trips(classified):
    import sector_exclusion

    assert sector_exclusion.hide_enabled() is True
    sector_exclusion.set_hide_enabled(False)
    assert classified[sector_exclusion.SETTING_HIDE_OIL_GAS_REAL_ESTATE] is False
    assert sector_exclusion.hide_enabled() is False


def test_symbol_lookup_reads_the_classification_csv(classified):
    import sector_exclusion

    assert sector_exclusion.symbol_is_excluded("apa")
    assert sector_exclusion.symbol_is_excluded("ADC")
    assert not sector_exclusion.symbol_is_excluded("CCJ")
    assert not sector_exclusion.symbol_is_excluded("NVDA")
    assert not sector_exclusion.symbol_is_excluded("ZZZZ"), "unknown is shown"
    assert sector_exclusion.symbol_is_hidden("APA")
    sector_exclusion.set_hide_enabled(False)
    assert not sector_exclusion.symbol_is_hidden("APA")


def test_a_missing_classification_file_hides_nothing(tmp_path, monkeypatch):
    import project_paths
    import sector_exclusion

    monkeypatch.setattr(project_paths, "SYMBOL_CLASSIFICATION_CACHE_FILE", tmp_path / "none.csv")
    sector_exclusion.clear_cache()
    try:
        assert not sector_exclusion.symbol_is_excluded("APA")
    finally:
        sector_exclusion.clear_cache()


def test_hidden_line():
    from sector_exclusion import hidden_line

    assert hidden_line(0) == ""
    assert hidden_line(2) == "Hidden: 2 oil & gas / real estate"
