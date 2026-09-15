"""The WS-SN3 golden recording states its own provenance, and states it truly.

``tests/test_fixture_contract.py`` proves every shipped fixture carries the
plan.md Milestone 3 contract.  That contract is only worth the bytes it costs
if the ``raw_input_sha256`` it freezes covers the inputs the recording was
actually made from - and the WS-SN3 recording's inputs are a SYNTHETIC universe
built by ``tests/test_ws_sn3_one_rrs_pass.py``, not a stored bar file.  So the
fixture declares that universe in ``recording_inputs``, and these tests pin it
against the module constants the recorder used.  Edit one without the other and
this file fails, rather than the golden quietly comparing against a universe
nobody builds any more.

The recording itself (``passes``, ``scan_extremes``) is never regenerated: it
was made on the PRE-SN3 code at the commit
``test_ws_sn3_one_rrs_pass.PINNED_SOURCE_COMMIT`` names.
"""

from __future__ import annotations

from conftest import load_fixture_contract

import test_ws_sn3_one_rrs_pass as sn3

FIXTURE_NAME = "ws_sn3_rrs_four_pass"


def _contract():
    return load_fixture_contract(FIXTURE_NAME)


def test_the_recording_carries_the_milestone_3_contract():
    contract = _contract()
    assert contract.schema == "ws_sn3_rrs_four_pass_v1"
    assert contract.raw_input_keys == ("recording_inputs",)
    assert contract.expected_keys == ("passes", "scan_extremes")
    # Byte-for-byte is the whole point of this golden: no numeric slack.
    assert contract.tolerance == 0.0
    assert contract.intentional_difference == ""
    # The loader recomputes the hash on every load; say so out loud too.
    assert contract.raw_input_digest() == contract["raw_input_sha256"]


def test_the_declared_universe_is_the_universe_the_tests_build():
    """A frozen input hash over inputs nobody uses proves nothing."""
    declared = _contract()["recording_inputs"]

    assert declared["session_dates"] == [day.isoformat() for day in sn3.SESSION_DATES]
    assert declared["session_open_local"] == "%02d:%02d" % (
        sn3.SESSION_OPEN_HOUR,
        sn3.SESSION_OPEN_MINUTE,
    )
    assert declared["bars_per_session"] == sn3.BARS_PER_SESSION
    assert declared["spy_symbol"] == sn3.SPY_SYMBOL
    assert declared["longs"] == list(sn3.LONGS)
    assert declared["shorts"] == list(sn3.SHORTS)
    assert declared["etf_symbols"] == list(sn3.ETF_SYMBOLS)
    assert declared["sector_etf_map"] == dict(sn3.SECTOR_ETF_MAP)
    assert declared["industry_etf_map_file"] == sn3.INDUSTRY_ETF_MAP_FILE
    assert declared["classifications"] == {
        symbol: list(values) for symbol, values in sn3.CLASSIFICATIONS.items()
    }
    assert declared["series_recipe"] == {
        symbol: list(values) for symbol, values in sn3.SERIES_RECIPE.items()
    }
    assert declared["cycle_timeframes"] == list(sn3.CYCLE_TIMEFRAMES)
    assert declared["gui_timeframe_key"] == sn3.GUI_TIMEFRAME_KEY


def test_the_recorded_passes_survived_the_contract_edit():
    """Adding provenance must never touch what the golden compares against."""
    contract = _contract()
    passes = contract["passes"]
    assert sorted(passes) == sorted([*sn3.CYCLE_TIMEFRAMES, "gui"])
    # The fourth pass of the pre-SN3 code reproduced one of the three; that
    # duplication is the waste SN3 removed, and it is still visible here.
    assert passes["gui"] == passes[sn3.GUI_TIMEFRAME_KEY]
    assert contract["generated_from_commit"] == sn3.PINNED_SOURCE_COMMIT
