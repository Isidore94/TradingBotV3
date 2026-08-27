"""Phase 0.10 B-2 item 4 - the band-variant stats export.

One row per (setup family, side, priority bucket) putting the champion's
primary-stop scenario beside the challenger's, on the SAME exit template so the
two are compared at one variable rather than two.

The rules this file pins:

* counts come first and a rate is never printed without its n;
* a cell with no evidence behind it is BLANK, never 0.0 - "no setups stopped
  out" and "no setups" are different claims and a zero cannot say which;
* a setup whose challenger sigma was unmeasurable is COUNTED
  (`n_variant_unmeasured`), never silently dropped, because a formula that
  cannot answer is a property of the formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402


def _scenario(label, source, template, *, status, total_r, level, tradeable=True):
    return {
        "scenario_id": f"{label.lower()}__{template}",
        "stop_reference_label": label,
        "stop_reference_level": level,
        "stop_source_type": source,
        "exit_template_id": template,
        "experimental": False,
        "tradeable": tradeable,
        "status": status,
        "total_r": total_r,
    }


def _setup(
    symbol,
    *,
    side="LONG",
    family="top_pattern",
    bucket="favorite_setup",
    champion=("TARGET_HIT", 1.5, 43.0),
    variant=("STOPPED", -0.8, 42.0),
    variant_sigma=1.25,
):
    protective = "LOWER_1" if side == "LONG" else "UPPER_1"
    scenarios = {}
    for template in ("full_band2", "full_band3"):
        status, total_r, level = champion
        scenarios[f"{protective.lower()}__{template}"] = _scenario(
            protective, "current_anchor", template, status=status, total_r=total_r, level=level
        )
    if variant is not None:
        for template in ("full_band2", "full_band3"):
            status, total_r, level = variant
            scenarios[f"variant_{protective.lower()}__{template}"] = _scenario(
                f"VARIANT_{protective}",
                legacy.BAND_VARIANT_STOP_SOURCE,
                template,
                status=status,
                total_r=total_r,
                level=level,
            )
    return {
        "setup_id": f"2026-08-11:{symbol}:{side}",
        "symbol": symbol,
        "side": side,
        "scan_date": "2026-08-11",
        "entry_price": 45.0,
        "setup_family": family,
        "tracker_setup_family": family,
        "priority_bucket": bucket,
        "tracker_priority_bucket": bucket,
        "entry_feature_snapshot": {"atr20": 1.0},
        "current_anchor_variant": {
            "formula_version": "avwap_bands_oneoption_bb20_v1",
            "vwap": 44.0,
            "stdev": variant_sigma,
            "bands": {} if variant_sigma is None else {"LOWER_1": 42.0, "UPPER_1": 46.0},
            "reason": "" if variant_sigma is not None else "fewer than the lookback's closes",
        },
        "scenarios": scenarios,
    }


def _rows(setups):
    return legacy.build_band_variant_stats_rows({s["setup_id"]: s for s in setups})


def test_a_three_setup_tracker_gives_the_expected_numbers():
    rows = _rows(
        [
            _setup("AAA", champion=("TARGET_HIT", 1.5, 43.0), variant=("TARGET_HIT", 1.0, 42.0)),
            _setup("BBB", champion=("STOPPED", -1.0, 43.0), variant=("TARGET_HIT", 0.5, 42.0)),
            _setup("CCC", champion=("STOPPED", -1.0, 43.0), variant=("STOPPED", -0.5, 42.0)),
        ]
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["setup_family"] == "top_pattern"
    assert row["side"] == "LONG"
    assert row["priority_bucket"] == "favorite_setup"
    assert row["n"] == 3
    assert row["n_variant"] == 3
    assert row["n_variant_unmeasured"] == 0
    assert row["avg_total_r_champion"] == pytest.approx((1.5 - 1.0 - 1.0) / 3)
    assert row["avg_total_r_variant"] == pytest.approx((1.0 + 0.5 - 0.5) / 3)
    assert row["stop_out_rate_champion"] == pytest.approx(2 / 3)
    assert row["stop_out_rate_variant"] == pytest.approx(1 / 3)
    assert row["target_hit_rate_champion"] == pytest.approx(1 / 3)
    assert row["target_hit_rate_variant"] == pytest.approx(2 / 3)
    # entry 45.0, atr 1.0, champion stop 43.0, variant stop 42.0
    assert row["mean_stop_distance_atr_champion"] == pytest.approx(2.0)
    assert row["mean_stop_distance_atr_variant"] == pytest.approx(3.0)


def test_the_two_formulas_are_compared_on_the_same_exit_template():
    """Otherwise the comparison carries two variables and measures neither."""
    rows = _rows([_setup("AAA")])
    assert rows[0]["exit_template_id"] == "full_band2"


def test_a_cell_with_no_evidence_is_blank_never_zero():
    """No variant scenario at all: the variant columns are empty strings."""
    rows = _rows([_setup("AAA", variant=None, variant_sigma=None)])
    row = rows[0]
    assert row["n"] == 1
    assert row["n_variant"] == 0
    assert row["n_variant_unmeasured"] == 1
    for column in (
        "avg_total_r_variant",
        "stop_out_rate_variant",
        "target_hit_rate_variant",
        "mean_stop_distance_atr_variant",
    ):
        assert row[column] == "", column
    # ...and the champion's own cells are still populated.
    assert row["avg_total_r_champion"] == pytest.approx(1.5)


def test_an_unmeasurable_sigma_is_counted_not_dropped():
    rows = _rows(
        [
            _setup("AAA", champion=("TARGET_HIT", 1.0, 43.0), variant=("TARGET_HIT", 1.0, 42.0)),
            _setup("BBB", champion=("TARGET_HIT", 1.0, 43.0), variant=None, variant_sigma=None),
        ]
    )
    row = rows[0]
    assert row["n"] == 2
    assert row["n_variant"] == 1
    assert row["n_variant_unmeasured"] == 1
    # The variant average is over the ONE setup it could measure, and its n says so.
    assert row["avg_total_r_variant"] == pytest.approx(1.0)


def test_longs_and_shorts_and_families_are_separate_rows():
    rows = _rows(
        [
            _setup("AAA", side="LONG"),
            _setup("BBB", side="SHORT"),
            _setup("CCC", side="LONG", family="post_earnings_candle_break"),
            _setup("DDD", side="LONG", bucket="near_favorite_zone"),
        ]
    )
    keys = {(row["setup_family"], row["side"], row["priority_bucket"]) for row in rows}
    assert keys == {
        ("top_pattern", "LONG", "favorite_setup"),
        ("top_pattern", "SHORT", "favorite_setup"),
        ("post_earnings_candle_break", "LONG", "favorite_setup"),
        ("top_pattern", "LONG", "near_favorite_zone"),
    }
    assert all(row["n"] == 1 for row in rows)


def test_an_empty_tracker_exports_no_rows():
    assert legacy.build_band_variant_stats_rows({}) == []


def test_the_export_writes_the_csv_in_the_same_pass(tmp_path, monkeypatch):
    """It rides the existing `export_setup_tracker_views` pass, not a new one."""
    for name in (
        "SETUP_SCENARIOS_FILE",
        "SETUP_DAILY_FILE",
        "SETUP_STATS_FILE",
        "SETUP_TYPE_STATS_FILE",
        "SETUP_TYPE_RECENT_STATS_FILE",
        "SETUP_PLAYBOOKS_FILE",
        "SETUP_SHORT_HORIZON_FILE",
        "SETUP_ATTRIBUTES_FILE",
        "SETUP_ATTRIBUTE_LEADERBOARD_FILE",
        "SETUP_BAND_VARIANT_STATS_FILE",
    ):
        monkeypatch.setattr(legacy, name, tmp_path / f"{name.lower()}.csv")

    setup = _setup("AAA")
    legacy.export_setup_tracker_views({"setups": {setup["setup_id"]: setup}})

    path = tmp_path / "setup_band_variant_stats_file.csv"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "avg_total_r_variant" in text
    assert "n_variant_unmeasured" in text
    # The champion's own scenario CSV never sees the shadow.
    scenarios = (tmp_path / "setup_scenarios_file.csv").read_text(encoding="utf-8")
    assert "VARIANT_LOWER_1" not in scenarios
