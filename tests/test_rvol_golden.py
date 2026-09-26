"""A4b golden: every RVOL reading on 2026-09-25's real volumes, compared exactly.

The fixture freezes real M5 volumes (SPY, QQQ, XLE; 24 traded sessions ending
2026-09-25) and SPY's D1 volumes, with the outputs of ``rvol.py``,
``intraday_rvol_service`` and ``movers_scan`` at 898c7900. Any change to a
number - a Movers pop weight, the bounce bot's rvol gate input, the chart
header - fails here. The mutation tests prove the golden can fail.
"""

from __future__ import annotations

import math

import pytest

from conftest import load_fixture_contract
from rvol_golden_cases import compute_all

CONTRACT = load_fixture_contract("rvol_golden_v1")


def _diffs(actual, expected, path="") -> list[str]:
    """Exact structural diff; floats must be bit-equal (NaN never expected)."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return [f"{path}: keys differ"]
        out: list[str] = []
        for key in expected:
            out += _diffs(actual[key], expected[key], f"{path}.{key}")
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: length differs"]
        out = []
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            out += _diffs(a, e, f"{path}[{index}]")
        return out
    if isinstance(expected, float) and isinstance(actual, float):
        return [] if actual == expected and not math.isnan(actual) else [f"{path}: {actual!r} != {expected!r}"]
    if type(actual) is not type(expected) or actual != expected:
        return [f"{path}: {actual!r} != {expected!r}"]
    return []


def test_golden_contract_is_exact():
    assert CONTRACT.tolerance == 0.0
    assert CONTRACT["intentional_difference"] == ""


def test_every_rvol_reading_matches_the_golden():
    diffs = _diffs(compute_all(CONTRACT["inputs"]), CONTRACT["expected"])
    assert not diffs, f"{len(diffs)} RVOL numbers moved, first: {diffs[:5]}"


def test_golden_records_where_the_implementations_agree():
    """The facts A4b's unify step rests on, pinned so a refactor cannot blur them."""
    for symbol, data in CONTRACT["expected"]["m5"].items():
        slots = data["rvol_slot_baselines"]
        movers15 = data["movers_baseline_15"]
        movers20 = data["movers_baseline"]
        # Same 15-session mean on full sessions; the Movers default (20) differs.
        assert all(slots[i] == movers15[str(i)] for i in range(78)), symbol
        assert any(slots[i] != movers20[str(i)] for i in range(78)), symbol
        for depth, row in data["depths"].items():
            assert (
                row["rvol_session_rvol"]
                == row["rvol_session_rvol_from_baseline"]
                == row["service_session_rvol"]
            ), (symbol, depth)
            assert row["rvol_bar_rvol"] == row["service_last_bar_rvol"], (symbol, depth)
        # Movers' recent rvol (mean of per-bar ratios, 20 sessions) is its own variant.
        assert any(
            row["movers_recent_rvol_span"] != row["rvol_session_rvol"]
            for row in data["depths"].values()
        ), symbol


def _mutated_fails(monkeypatch, module, name, wrap) -> list[str]:
    monkeypatch.setattr(module, name, wrap(getattr(module, name)))
    return _diffs(compute_all(CONTRACT["inputs"]), CONTRACT["expected"])


def test_golden_breaks_when_rvol_slot_baseline_is_mutated(monkeypatch):
    import rvol

    def wrap(original):
        def mutated(*args, **kwargs):
            value = original(*args, **kwargs)
            return None if value is None else value * 1.001
        return mutated

    assert _mutated_fails(monkeypatch, rvol, "same_slot_baseline", wrap)


def test_golden_breaks_when_movers_slot_offset_is_mutated(monkeypatch):
    import movers_scan

    def wrap(original):
        def mutated(moment):
            value = original(moment)
            return None if value is None else value + 1
        return mutated

    assert _mutated_fails(monkeypatch, movers_scan, "session_offset", wrap)


@pytest.mark.parametrize("scale", [1.0 + 1e-12])
def test_golden_breaks_on_a_last_bit_change_in_daily_rvol(monkeypatch, scale):
    import rvol

    def wrap(original):
        def mutated(*args, **kwargs):
            return [None if v is None else v * scale for v in original(*args, **kwargs)]
        return mutated

    assert _mutated_fails(monkeypatch, rvol, "daily_rvol_series", wrap)
