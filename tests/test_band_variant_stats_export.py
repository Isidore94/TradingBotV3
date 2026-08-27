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

import logging
import sys
from pathlib import Path
from unittest import mock

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


#: Every export path `export_setup_tracker_views` writes, champion first.
EXPORT_FILES = (
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
)
CHAMPION_FILES = tuple(name for name in EXPORT_FILES if name != "SETUP_BAND_VARIANT_STATS_FILE")


def _redirect_exports(monkeypatch, tmp_path) -> None:
    for name in EXPORT_FILES:
        monkeypatch.setattr(legacy, name, tmp_path / f"{name.lower()}.csv")


def _exploding_builder(*_args, **_kwargs):
    """What one malformed setup dict looks like from inside the shadow builder."""
    raise ValueError("malformed setup dict reached the shadow builder")


def test_the_export_writes_the_csv_in_the_same_pass(tmp_path, monkeypatch):
    """It rides the existing `export_setup_tracker_views` pass, not a new one."""
    _redirect_exports(monkeypatch, tmp_path)

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


# ---------------------------------------------------------------------------
# Review fix 1 (2026-08-26 night): the shadow export may never cost the save.
#
# `export_setup_tracker_views` writes the band-variant CSV as its LAST
# statement, and its caller runs `save_setup_tracker_payload` after it. An
# unguarded raise there would abort the day's tracker save - the evidence store
# costing the thing it records, which R10 forbids everywhere else in this
# codebase.
# ---------------------------------------------------------------------------


def test_a_raising_shadow_builder_never_costs_the_champion_exports(
    tmp_path, monkeypatch, caplog
):
    _redirect_exports(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy, "build_band_variant_stats_rows", _exploding_builder)

    setup = _setup("AAA")
    with caplog.at_level(logging.WARNING):
        # No raise: the caller must get control back.
        legacy.export_setup_tracker_views({"setups": {setup["setup_id"]: setup}})

    for name in CHAMPION_FILES:
        assert getattr(legacy, name).exists(), f"{name} was not written"
    # The shadow's own file is absent rather than half-written, and the failure
    # is SAID rather than swallowed - a quiet miss would read as "no rows".
    assert not legacy.SETUP_BAND_VARIANT_STATS_FILE.exists()
    assert any(
        "band variant" in record.getMessage().lower() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_the_scan_still_saves_its_tracker_when_the_shadow_export_raises(
    tmp_path, monkeypatch
):
    """The reach that actually matters: `save_setup_tracker_payload` is called.

    Exercised through the real `export_setup_tracker_views`, because the point
    is the seam between the two - a test that mocked the export away would
    prove nothing about it.
    """
    _redirect_exports(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy, "build_band_variant_stats_rows", _exploding_builder)

    payload = {"setups": {}, "control_setups": {}, "study_setups": {}, "daily_watchlists": {}}
    with mock.patch.object(legacy, "write_control_discovery_report"), mock.patch.object(
        legacy, "write_master_avwap_study_report"
    ), mock.patch.object(legacy, "save_setup_tracker_payload") as save_mock:
        legacy.update_setup_tracker_from_scan(
            [],
            {"symbols": {}},
            {},
            {},
            None,
            auto_tune=False,
            tracker_payload=payload,
        )

    save_mock.assert_called_once()
    assert save_mock.call_args.args[0] is payload


def test_only_the_shadow_write_is_guarded(tmp_path, monkeypatch):
    """A champion export that fails must still fail LOUDLY.

    The guard is scoped to the shadow deliberately: swallowing a champion
    export failure would hand the trader a stale CSV with no way to know.
    """
    _redirect_exports(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy, "build_tracker_setup_type_rows", _exploding_builder)

    setup = _setup("AAA")
    with pytest.raises(ValueError):
        legacy.export_setup_tracker_views({"setups": {setup["setup_id"]: setup}})


# ---------------------------------------------------------------------------
# Review fix 3 (trader decision, 2026-08-26): the shadow crosses the four
# BASELINE exit templates only.
# ---------------------------------------------------------------------------


def test_the_shadow_stop_skips_the_experimental_exit_templates():
    baseline = [t for t in legacy.SETUP_EXIT_TEMPLATES if not t.get("experimental")]
    experimental = [t for t in legacy.SETUP_EXIT_TEMPLATES if t.get("experimental")]
    assert len(baseline) == 4 and experimental, "the template set changed shape"

    champion_stop = {
        "label": "LOWER_1",
        "level": 43.0,
        "source_type": "current_anchor",
        "close_failure_limit": 2,
    }
    variant_stop = {
        "label": "VARIANT_LOWER_1",
        "level": 42.0,
        "source_type": legacy.BAND_VARIANT_STOP_SOURCE,
        "close_failure_limit": 2,
    }
    scenarios = legacy._build_tracker_scenarios(45.0, [champion_stop, variant_stop], "LONG")

    champion = [s for s in scenarios.values() if not legacy._is_band_variant_scenario(s)]
    shadow = [s for s in scenarios.values() if legacy._is_band_variant_scenario(s)]
    # The champion is untouched - it still crosses every template.
    assert len(champion) == len(legacy.SETUP_EXIT_TEMPLATES)
    assert len(shadow) == len(baseline)
    assert all(not s["experimental"] for s in shadow)
    # ...and the shadow still covers EVERY baseline template, so the stats
    # table's per-template pairing is still possible.
    assert {s["exit_template_id"] for s in shadow} == {t["id"] for t in baseline}


def test_the_candidate_side_and_scenario_side_agree_on_what_the_shadow_is():
    """Two spellings of one question; they must never drift apart."""
    candidate = {"source_type": legacy.BAND_VARIANT_STOP_SOURCE}
    scenario = {"stop_source_type": legacy.BAND_VARIANT_STOP_SOURCE}
    assert legacy._is_band_variant_stop(candidate)
    assert legacy._is_band_variant_scenario(scenario)
    # Neither answers yes to the other's shape by accident.
    assert not legacy._is_band_variant_stop(scenario)
    assert not legacy._is_band_variant_scenario(candidate)
    assert not legacy._is_band_variant_stop({"source_type": "current_anchor"})
