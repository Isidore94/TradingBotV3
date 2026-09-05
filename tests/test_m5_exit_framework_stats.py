"""Packet M5.3 - the April comparison framework finally gets a reader.

`comparison_apr2026` has written 91,674 of the 275,022 rows in
`master_avwap_setup_scenarios.csv` since April: two experimental exit templates
(`exp_full_band2_hard_stop_125r` and `..._no_sma50_short_nearfav`) simulated on
the SAME setups as the baseline templates, so the two can be compared at one
variable. Every champion aggregate skips an `experimental` scenario by design,
and until this packet **no reader of `framework_family == "comparison_apr2026"`
existed anywhere under `scripts/`.** Written, never read.

This file pins the reader:

* one row per `(framework_family, exit_template_id, side, priority_bucket)`, so
  a baseline template and a comparison template on the same setups sit side by
  side with the SAME `n`;
* `experimental` is a COLUMN, so a comparison row can never be read as the
  champion's own record;
* the aggregates the champion scores from do not move - the new export is
  additive, and the fence that keeps the band-variant challenger out of them
  stays exactly where it is;
* the export is guarded: it may never cost the tracker save.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402
from swing_headline import wilson_lower_bound  # noqa: E402

from test_m5_discovery_exports import EXPORT_FILES, _redirect_exports  # noqa: E402


BASELINE_TEMPLATE = "full_band2"
COMPARISON_TEMPLATE = "exp_full_band2_hard_stop_125r"


def _scenario(template, *, status, total_r, experimental, family, version):
    return {
        "scenario_id": f"lower_1__{template}",
        "stop_reference_label": "LOWER_1",
        "stop_reference_level": 42.0,
        "stop_source_type": "band",
        "exit_template_id": template,
        "exit_template_label": template.replace("_", " "),
        "framework_family": family,
        "framework_version": version,
        "experimental": experimental,
        "tradeable": True,
        "status": status,
        "total_r": total_r,
    }


def _paired_setup(symbol, *, baseline, comparison, side="LONG", bucket="favorite_setup"):
    """One setup carrying BOTH templates - the shape the comparison relies on."""
    return {
        "setup_id": f"{symbol}:2026-01-03",
        "symbol": symbol,
        "side": side,
        "scan_date": "2026-01-03",
        "anchor_date": "2026-01-02",
        "priority_bucket": bucket,
        "setup_family": "post_earnings_52w_break",
        "scenarios": {
            "baseline": _scenario(
                BASELINE_TEMPLATE,
                status=baseline[0],
                total_r=baseline[1],
                experimental=False,
                family="baseline",
                version="baseline",
            ),
            "comparison": _scenario(
                COMPARISON_TEMPLATE,
                status=comparison[0],
                total_r=comparison[1],
                experimental=True,
                family="comparison_apr2026",
                version=legacy.TRACKER_EXPERIMENTAL_FRAMEWORK_VERSION,
            ),
        },
    }


def _setups(*records):
    return {record["setup_id"]: record for record in records}


def _read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ---------------------------------------------------------------------------
# The rows themselves
# ---------------------------------------------------------------------------


def test_one_baseline_and_one_comparison_on_the_same_setup_give_two_rows():
    setups = _setups(
        _paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("STOPPED", -1.25))
    )
    rows = legacy.build_exit_framework_stats_rows(setups)

    assert len(rows) == 2
    by_template = {row["exit_template_id"]: row for row in rows}
    assert set(by_template) == {BASELINE_TEMPLATE, COMPARISON_TEMPLATE}
    # SAME setups, so the same n - that equality is what makes it a comparison
    # rather than two unrelated records.
    assert by_template[BASELINE_TEMPLATE]["n"] == by_template[COMPARISON_TEMPLATE]["n"] == 1
    assert by_template[BASELINE_TEMPLATE]["framework_family"] == "baseline"
    assert by_template[COMPARISON_TEMPLATE]["framework_family"] == "comparison_apr2026"


def test_the_comparison_row_is_labelled_experimental_and_the_baseline_is_not():
    rows = legacy.build_exit_framework_stats_rows(
        _setups(_paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("STOPPED", -1.25)))
    )
    by_template = {row["exit_template_id"]: row for row in rows}
    assert by_template[COMPARISON_TEMPLATE]["experimental"] is True
    assert by_template[BASELINE_TEMPLATE]["experimental"] is False
    assert (
        by_template[COMPARISON_TEMPLATE]["framework_version"]
        == legacy.TRACKER_EXPERIMENTAL_FRAMEWORK_VERSION
    )


def test_the_rates_and_the_wilson_bound_are_computed_over_the_closed_rows():
    setups = _setups(
        _paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("TARGET_HIT", 1.1)),
        _paired_setup("BBB", baseline=("TARGET_HIT", 1.4), comparison=("STOPPED", -1.25)),
        _paired_setup("CCC", baseline=("STOPPED", -1.0), comparison=("STOPPED", -1.25)),
    )
    by_template = {
        row["exit_template_id"]: row for row in legacy.build_exit_framework_stats_rows(setups)
    }

    baseline = by_template[BASELINE_TEMPLATE]
    assert baseline["n"] == 3
    assert baseline["n_closed"] == 3
    assert baseline["wins"] == 2
    assert baseline["losses"] == 1
    assert baseline["win_rate"] == pytest.approx(2 / 3)
    assert baseline["win_rate_lb"] == pytest.approx(wilson_lower_bound(2, 3))
    assert baseline["stop_out_rate"] == pytest.approx(1 / 3)
    assert baseline["target_hit_rate"] == pytest.approx(2 / 3)
    assert baseline["avg_closed_r"] == pytest.approx((1.6 + 1.4 - 1.0) / 3)

    comparison = by_template[COMPARISON_TEMPLATE]
    assert comparison["n"] == 3
    assert comparison["wins"] == 1
    assert comparison["stop_out_rate"] == pytest.approx(2 / 3)


def test_a_group_with_nothing_closed_is_blank_and_never_zero():
    setups = _setups(
        _paired_setup("AAA", baseline=("OPEN", 0.4), comparison=("OPEN", 0.2)),
    )
    for row in legacy.build_exit_framework_stats_rows(setups):
        assert row["n"] == 1
        assert row["n_closed"] == 0
        # "nothing stopped out" and "nothing measured" are different claims and
        # a 0.0 cannot say which.
        assert row["win_rate"] is None
        assert row["win_rate_lb"] is None
        assert row["stop_out_rate"] is None
        assert row["target_hit_rate"] is None
        assert row["avg_closed_r"] is None


def test_sides_and_buckets_are_separate_rows():
    setups = _setups(
        _paired_setup("AAA", baseline=("TARGET_HIT", 1.0), comparison=("STOPPED", -1.25)),
        _paired_setup(
            "BBB", baseline=("TARGET_HIT", 1.0), comparison=("STOPPED", -1.25), side="SHORT"
        ),
        _paired_setup(
            "CCC",
            baseline=("TARGET_HIT", 1.0),
            comparison=("STOPPED", -1.25),
            bucket="near_favorite_zone",
        ),
    )
    keys = {
        (row["framework_family"], row["exit_template_id"], row["side"], row["priority_bucket"])
        for row in legacy.build_exit_framework_stats_rows(setups)
    }
    assert len(keys) == 6
    assert ("comparison_apr2026", COMPARISON_TEMPLATE, "SHORT", "favorite_setup") in keys


def test_an_expired_unmeasured_record_leaves_both_sides_of_the_fraction():
    """M3.3's rule, in an EXPORT: uncertainty is excluded and COUNTED."""
    expired = _paired_setup("EXP", baseline=("OPEN", 0.0), comparison=("OPEN", 0.0))
    expired["setup_status"] = legacy.SETUP_STATUS_EXPIRED_UNMEASURED
    setups = _setups(
        _paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("STOPPED", -1.25)),
        expired,
    )
    for row in legacy.build_exit_framework_stats_rows(setups):
        assert row["n"] == 1
        assert row["n_expired_unmeasured"] == 1


def test_an_empty_tracker_exports_no_framework_rows():
    assert legacy.build_exit_framework_stats_rows({}) == []


# ---------------------------------------------------------------------------
# The export, and the champion aggregates that must not move
# ---------------------------------------------------------------------------


def test_the_save_pass_writes_the_framework_csv(tmp_path, monkeypatch):
    _redirect_exports(monkeypatch, tmp_path)
    setups = _setups(
        _paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("STOPPED", -1.25))
    )
    legacy.export_setup_tracker_views({"setups": setups})

    path = legacy.EXIT_FRAMEWORK_STATS_FILE
    assert path.exists()
    families = {row["framework_family"] for row in _read_rows(path)}
    assert families == {"baseline", "comparison_apr2026"}


def _champion_bytes(tmp_path) -> dict[str, bytes]:
    return {
        name: getattr(legacy, name).read_bytes()
        for name in EXPORT_FILES
        if name
        not in {
            "CONTROL_DISCOVERY_STATS_FILE",
            "STUDY_DISCOVERY_STATS_FILE",
            "EXIT_FRAMEWORK_STATS_FILE",
        }
        and getattr(legacy, name).exists()
    }


def test_the_champion_aggregates_are_byte_identical_with_and_without_the_new_export(
    tmp_path, monkeypatch
):
    """The invariant that binds this packet, checked by reproduction.

    The same payload is exported twice into two directories: once normally, and
    once with the three new builders raising so no new file is written at all.
    Every champion export must come out byte for byte the same. If the new
    aggregation ever reached back into the champion's rows - a shared mutable
    row dict, a sort in place, a scenario the fence lets through - this is what
    catches it.
    """
    setups = _setups(
        _paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("STOPPED", -1.25)),
        _paired_setup("BBB", baseline=("STOPPED", -1.0), comparison=("STOPPED", -1.25), side="SHORT"),
    )

    def _export_into(directory: Path, *, disabled: bool) -> dict[str, bytes]:
        directory.mkdir(parents=True, exist_ok=True)
        with monkeypatch.context() as patch:
            for name in EXPORT_FILES:
                patch.setattr(legacy, name, directory / f"{name.lower()}.csv", raising=False)
            if disabled:
                for name in (
                    "build_control_discovery_stats_rows",
                    "build_study_discovery_stats_rows",
                    "build_exit_framework_stats_rows",
                ):
                    patch.setattr(legacy, name, _boom)
            payload = {"setups": setups, "control_setups": {}, "study_setups": {}}
            legacy.export_setup_tracker_views(payload)
            return _champion_bytes(directory)

    with_new = _export_into(tmp_path / "with", disabled=False)
    without_new = _export_into(tmp_path / "without", disabled=True)

    assert set(with_new) == set(without_new)
    for name, blob in with_new.items():
        assert blob == without_new[name], f"{name} moved when the new export ran"


def _boom(*_args, **_kwargs):
    raise ValueError("disabled for the parity half of this test")


def test_a_raising_framework_export_never_costs_the_tracker_save(tmp_path, monkeypatch, caplog):
    import logging

    _redirect_exports(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy, "build_exit_framework_stats_rows", _boom)

    with caplog.at_level(logging.WARNING):
        legacy.export_setup_tracker_views(
            {
                "setups": _setups(
                    _paired_setup("AAA", baseline=("TARGET_HIT", 1.6), comparison=("STOPPED", -1.25))
                )
            }
        )

    assert not legacy.EXIT_FRAMEWORK_STATS_FILE.exists()
    assert legacy.SETUP_SCENARIOS_FILE.exists()
    assert any(
        "exit framework" in record.getMessage().lower() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]
