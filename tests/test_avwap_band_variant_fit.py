"""Phase 0.10 B-1 - the hover-comparison table, held to the same golden rows.

The script's job is to put the champion and the challenger side by side for one
symbol and one anchor, so the trader can hover the same bar in OneOption and
read three numbers off a terminal. Its only contract is that the numbers are the
frozen ones, so that is what is asserted - on the same fixture B-0 uses, never
on the live store, so this test stays offline and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from avwap_band_variant_fit import build_rows, render_table  # noqa: E402

FIXTURE = "avwap_band_variant_oneoption_v1"


@pytest.fixture(scope="module")
def contract():
    return load_fixture_contract(FIXTURE)


@pytest.fixture(scope="module")
def frame(contract):
    bars = pd.DataFrame(list(contract["bars"]))
    bars["datetime"] = pd.to_datetime(bars["date"])
    return bars


@pytest.fixture(scope="module")
def anchor(contract, frame):
    stamp = pd.Timestamp(contract["rules_under_test"]["anchor_date"])
    return int(frame.index[frame["datetime"] == stamp][0])


def test_the_table_reproduces_both_golden_rows(contract, frame, anchor):
    rules = contract["rules_under_test"]
    rows = build_rows(frame, anchor, lookback=rules["lookback"])
    by_date = {row["date"]: row for row in rows}

    for reading in contract["expected"]["readings"]:
        row = by_date[reading["date"]]
        vendor_sigma = (reading["upper_1"] - reading["lower_1"]) / 2.0
        assert row["variant_sigma"] == pytest.approx(
            vendor_sigma, abs=rules["sigma_absolute_tolerance"]
        ), reading["date"]
        assert (
            abs(row["variant_centre"] - reading["centre"]) / reading["centre"]
            <= rules["centre_relative_tolerance"]
        ), reading["date"]
        assert row["variant_upper_1"] == pytest.approx(
            row["variant_centre"] + row["variant_sigma"], abs=1e-9
        )


def test_the_table_starts_at_the_anchor_and_runs_to_the_end(frame, anchor):
    rows = build_rows(frame, anchor)
    assert len(rows) == len(frame) - anchor
    assert rows[0]["date"] == frame["datetime"].iloc[anchor].strftime("%Y-%m-%d")
    assert rows[-1]["date"] == frame["datetime"].iloc[-1].strftime("%Y-%m-%d")


def test_the_champion_column_is_the_frozen_champion(frame, anchor):
    """The anchor bar is the one place the two formulas cannot be confused."""
    from master_avwap_lib.legacy import calc_anchored_vwap_bands

    rows = build_rows(frame, anchor)
    vwap, sigma, bands = calc_anchored_vwap_bands(frame.iloc[: anchor + 1], anchor)
    assert rows[0]["champion_centre"] == pytest.approx(vwap, abs=1e-9)
    assert rows[0]["champion_sigma"] == pytest.approx(sigma, abs=1e-9)
    assert sigma == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["champion_upper_1"] == pytest.approx(bands["UPPER_1"], abs=1e-9)
    # ...and the challenger is wide on that same bar. That is the whole finding.
    assert rows[0]["variant_sigma"] > 10.0


def test_an_unmeasurable_variant_prints_a_blank_never_a_zero(frame):
    """Anchored at bar 0 the sigma window is not full, so the cell is empty."""
    rows = build_rows(frame, 0, lookback=20)
    assert rows[0]["variant_sigma"] is None
    assert rows[0]["variant_upper_1"] is None
    text = render_table(rows[:1], symbol="OKTA", anchor_date=rows[0]["date"], lookback=20)
    body = text.splitlines()[-1]
    assert "0.00" not in body.split("|")[-1]


def test_the_rendered_table_names_both_formulas(frame, anchor):
    text = render_table(
        build_rows(frame, anchor), symbol="OKTA", anchor_date="2026-05-29", lookback=20
    )
    assert "champion" in text.lower()
    assert "variant" in text.lower()
    assert "OKTA" in text
    assert "2026-06-02" in text


def test_the_script_writes_nothing_and_dials_nothing_by_default():
    """Offline by construction: no network client and no unconditional write."""
    import ast

    source = (SCRIPTS_DIR / "avwap_band_variant_fit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    banned = {"requests", "urllib", "yfinance", "ibapi", "http", "socket"}
    assert not [name for name in imported if name.split(".")[0] in banned]
    # The only writer is the --csv path.
    assert source.count("write_text") + source.count("to_csv") <= 1
