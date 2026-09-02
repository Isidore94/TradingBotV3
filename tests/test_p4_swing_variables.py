"""P4 Half A: see the evidence, add the variables — and move no number.

Half A does two things and neither may touch the ranking:

* the attribute leaderboard the scanner has written every scan since it was
  built gets a tab, with the sample floor visible;
* twelve variables that were already on the record or the row get attribute
  keys, so the leaderboard can grade them.

The contract this file exists to hold is the second half of that sentence.
`plan.md` sec 5 forbids a scoring or ranking behaviour change without a golden
fixture first, so the golden here is the RANKING ITSELF: the priority score,
the bucket and the expected R of a scan must be byte-identical with and without
the new attributes. If a later edit gives one of these variables a weight, this
file fails first.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402

#: The new keys, named once. A key that stops being produced fails a test
#: rather than quietly leaving a column blank on the tab.
P4_ATTRIBUTE_KEYS = (
    "setup.human_focus_pick",
    "setup.human_focus_side",
    "setup.tracker_setup_family",
    "market.regime_label",
    "setup.sector",
    "setup.industry",
    "volatility.atr20_pct_of_price",
    "structure.dist_sma200_atr",
    "structure.dist_sma50_atr",
    "structure.above_sma200",
    "structure.below_sma50",
    "filters.relvol_20d",
)


def _row(**overrides) -> dict:
    row = {
        "symbol": "AAA",
        "side": "LONG",
        "score": 120,
        "priority_bucket": "favorite_setup",
        "setup_family": "avwape_to_first_dev",
        "tracker_setup_family": "avwape_to_first_dev",
        "human_focus_pick": True,
        "human_focus_side": "LONG",
        "market_regime_label": "bullish_weak",
        "sector": "Energy",
        "industry": "Oil & Gas Exploration",
        "atr20": 2.0,
        "last_close": 50.0,
        "sma200": 40.0,
        "sma50": 55.0,
        "relvol": 1.8,
    }
    row.update(overrides)
    return row


# ==========================================================================
# A2 - the variables are registered
# ==========================================================================
def test_every_new_variable_reaches_the_record():
    """Fail-before-fix: none of these keys exists."""
    attributes, registry = legacy.build_tracker_entry_attributes(_row(), {}, {}, {})

    for key in P4_ATTRIBUTE_KEYS:
        assert key in attributes, f"{key} was not registered"
        assert key in registry, f"{key} has no registry entry (no label for the tab)"
        assert registry[key]["label"], key
        assert registry[key]["description"], key


def test_atr_is_recorded_as_a_percent_BESIDE_the_dollar_bucket():
    """A $2 ATR is a quiet day on a $400 stock and a violent one on a $12 one.
    The dollar bucket pools them; this does not. It is added, never swapped -
    the same unit error the trader ruled on for theta premium."""
    attributes, _ = legacy.build_tracker_entry_attributes(
        _row(atr20=2.0, last_close=50.0), {}, {}, {}
    )
    assert attributes["volatility.atr20_pct_of_price"] == pytest.approx(4.0)
    # The dollar-bucketed field is still registered for the scan factors.
    assert "atr20" in legacy.SCAN_FACTOR_NUMERIC_FIELDS
    assert "atr20_pct_of_price" in legacy.SCAN_FACTOR_NUMERIC_FIELDS


def test_the_sma_geometry_is_signed_and_in_ATR():
    attributes, _ = legacy.build_tracker_entry_attributes(
        _row(last_close=50.0, sma200=40.0, sma50=55.0, atr20=2.0), {}, {}, {}
    )
    assert attributes["structure.dist_sma200_atr"] == pytest.approx(5.0)
    assert attributes["structure.dist_sma50_atr"] == pytest.approx(-2.5)
    assert attributes["structure.above_sma200"] is True
    assert attributes["structure.below_sma50"] is True


def test_a_missing_input_records_NOTHING_rather_than_a_zero():
    """A setup with no SMA200 has an unknown distance, not a distance of zero -
    and a zero would sit in the 0-1 ATR bucket and be graded as if measured."""
    attributes, _ = legacy.build_tracker_entry_attributes(
        _row(sma200=None, sma50=None, atr20=None), {}, {}, {}
    )
    for key in (
        "structure.dist_sma200_atr",
        "structure.dist_sma50_atr",
        "structure.above_sma200",
        "structure.below_sma50",
        "volatility.atr20_pct_of_price",
    ):
        assert key not in attributes, f"{key} was recorded from absent inputs"


def test_a_zero_atr_never_divides():
    attributes, _ = legacy.build_tracker_entry_attributes(
        _row(atr20=0.0, last_close=50.0), {}, {}, {}
    )
    assert "volatility.atr20_pct_of_price" not in attributes
    assert "structure.dist_sma200_atr" not in attributes


def test_the_scan_factor_registers_carry_the_new_fields():
    assert legacy.SCAN_FACTOR_CATEGORICAL_FIELDS["sector"] == ("setup", "Sector")
    assert legacy.SCAN_FACTOR_CATEGORICAL_FIELDS["industry"] == ("setup", "Industry")
    assert "relvol" in legacy.SCAN_FACTOR_NUMERIC_FIELDS
    assert "dist_sma200_atr" in legacy.SCAN_FACTOR_NUMERIC_FIELDS
    assert "dist_sma50_atr" in legacy.SCAN_FACTOR_NUMERIC_FIELDS
    assert "above_sma200" in legacy.SCAN_FACTOR_BOOL_FIELDS
    assert "below_sma50" in legacy.SCAN_FACTOR_BOOL_FIELDS


# ==========================================================================
# THE GOLDEN: Half A moved nothing the ranking reads
# ==========================================================================
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "p4_ranking_unchanged_v1.json"

RANKING_KEYS = (
    "score",
    "static_score",
    "proven_quality_score",
    "priority_bucket",
    "expected_r",
    "expected_r_rank_score",
    "expected_r_note",
    "tracker_win_rate",
    "tracker_profit_factor",
)


#: The Expected-R config is PINNED here rather than loaded.
#: `load_expected_r_config` reads `expected_r_config.json` from the home
#: folder, whose `prior_anchors` are re-fitted by the calibration pass - so a
#: golden that let it load would fail whenever the desk recalibrated, which is
#: a real event and not a regression. Pinning the anchors makes this fixture
#: measure the RANKING CODE, which is what it exists to protect. The pinned
#: values are the shipped `DEFAULT_EXPECTED_R_CONFIG` anchors.
PINNED_EXPECTED_R_CONFIG = None


def _pinned_config() -> dict:
    import copy

    global PINNED_EXPECTED_R_CONFIG
    if PINNED_EXPECTED_R_CONFIG is None:
        PINNED_EXPECTED_R_CONFIG = copy.deepcopy(legacy.DEFAULT_EXPECTED_R_CONFIG)
    return copy.deepcopy(PINNED_EXPECTED_R_CONFIG)


def _ranked_rows() -> list[dict]:
    """One deterministic scan through the real ranking function."""
    rows = [
        {
            **_row(symbol="AAA", score=120, setup_family="avwape_to_first_dev"),
            "quality_points": 120,
        },
        {
            **_row(
                symbol="BBB",
                score=95,
                side="SHORT",
                priority_bucket="high_conviction",
                setup_family="post_earnings_candle_break",
                human_focus_pick=False,
                sector="Technology",
                industry="Semiconductors",
                last_close=200.0,
                atr20=6.0,
                sma200=210.0,
                sma50=190.0,
                relvol=0.7,
            ),
            "quality_points": 95,
        },
        {
            **_row(symbol="CCC", score=80, priority_bucket="near_favorite", sector="", industry=""),
            "quality_points": 80,
        },
    ]
    legacy.apply_expected_r_ranking(
        rows, {}, {}, config=_pinned_config(), reference_date=date(2026, 9, 1)
    )
    return rows


def _ranking_signature(rows) -> list[dict]:
    return [
        {"symbol": row.get("symbol"), **{key: row.get(key) for key in RANKING_KEYS}}
        for row in rows
    ]


def test_the_ranking_is_byte_identical_to_the_frozen_golden():
    """The whole point of Half A.

    The golden was frozen from the ranking BEFORE the twelve attributes were
    added, and it carries its own inputs, so this REPLAYS it rather than
    comparing against rows the test built itself - a test that built its own
    rows could drift away from what was frozen without anything failing.

    Regenerated ONLY by a deliberate scoring packet with a sec-7 promotion
    behind it. `plan.md` sec 5 / decision 0009: no scoring or ranking behaviour
    change without a golden fixture first.

    If this fails, something in Half A gave a captured variable a weight.
    """
    fixture = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    rows = copy.deepcopy(fixture["priority_rows"])
    legacy.apply_expected_r_ranking(
        rows,
        {},
        {},
        config=copy.deepcopy(fixture["expected_r_config"]),
        reference_date=date.fromisoformat(fixture["reference_date"]),
    )
    signature = json.loads(json.dumps(_ranking_signature(rows), sort_keys=True, default=str))

    assert signature == fixture["ranking_signature"]


def test_the_golden_replays_the_inputs_it_was_frozen_with():
    """The fixture's raw inputs are hashed, so an edited input is a changed
    hash rather than a silently different golden."""
    import hashlib

    fixture = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    raw = {key: fixture[key] for key in fixture["raw_input_keys"]}
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == fixture["raw_input_sha256"]


def test_the_new_variables_are_absent_from_every_scoring_input():
    """A structural guard, not a behavioural one: the attribute keys must not
    appear anywhere in the scoring or ranking functions. The golden above
    catches a weight that CHANGES a number; this catches one wired in with no
    measurable effect yet on the fixture."""
    import inspect

    scoring_sources = "\n".join(
        inspect.getsource(func)
        for func in (
            legacy.apply_expected_r_ranking,
            legacy._expected_r_quality_points,
        )
    )
    for key in P4_ATTRIBUTE_KEYS:
        leaf = key.split(".", 1)[1]
        assert key not in scoring_sources, f"{key} reached the ranking"
        if leaf in {"sector", "industry", "regime_label"}:
            continue  # ordinary words that legitimately appear in other names
        assert f'"{leaf}"' not in scoring_sources, f"{leaf} reached the ranking"


# ==========================================================================
# A1 - the tab
# ==========================================================================
def test_the_attribute_tab_greys_and_sinks_a_sub_floor_row():
    pytest.importorskip("PySide6")
    from ui.panels import setup_tracker_panel as panel_module

    rows = [
        {
            "attribute_label": "Lucky",
            "value_label": "yes",
            "closed_tradeable_setup_count": "1",
            "avg_closed_r_edge": "2.9",
        },
        {
            "attribute_label": "Real",
            "value_label": "yes",
            "closed_tradeable_setup_count": "48",
            "avg_closed_r_edge": "0.31",
        },
    ]
    ranked = panel_module._rank_attribute_leaderboard(rows)

    assert [row["attribute_label"] for row in ranked] == ["Real", "Lucky"]
    assert ranked[0]["_meets_floor"] is True
    assert ranked[0]["_muted_row"] is False
    assert ranked[1]["_meets_floor"] is False
    assert ranked[1]["_muted_row"] is True
    assert "below floor" in ranked[1]["meets_n_floor_label"]


def test_the_floor_comes_from_the_one_place_that_owns_it():
    pytest.importorskip("PySide6")
    from evidence_stats import MIN_REPORTABLE_N
    from ui.panels import setup_tracker_panel as panel_module

    assert panel_module._attribute_floor() == MIN_REPORTABLE_N


def test_a_sub_floor_row_is_KEPT_not_dropped():
    """Visibility, not suppression: the export is the tuner's input and this
    tab is a reading of it."""
    pytest.importorskip("PySide6")
    from ui.panels import setup_tracker_panel as panel_module

    rows = [{"attribute_label": "A", "closed_tradeable_setup_count": "1", "avg_closed_r_edge": "9"}]
    assert len(panel_module._rank_attribute_leaderboard(rows)) == 1


def test_a_blank_edge_sorts_last_within_its_group_and_is_never_read_as_zero():
    pytest.importorskip("PySide6")
    from ui.panels import setup_tracker_panel as panel_module

    rows = [
        {"attribute_label": "Blank", "closed_tradeable_setup_count": "40", "avg_closed_r_edge": ""},
        {"attribute_label": "Negative", "closed_tradeable_setup_count": "40", "avg_closed_r_edge": "-0.5"},
    ]
    ranked = panel_module._rank_attribute_leaderboard(rows)
    assert [row["attribute_label"] for row in ranked] == ["Negative", "Blank"]


def test_the_tab_exists_and_the_big_export_is_read_off_the_qt_thread():
    """19.7 MB and 38,617 rows on the live desk, against 5.5 MB for the next
    largest sibling. Parsing it on the render path would freeze the desk on
    every refresh, spinbox step and tab visit."""
    pytest.importorskip("PySide6")
    import inspect

    from ui.panels import setup_tracker_panel as panel_module

    refresh = inspect.getsource(panel_module.SetupTrackerPanel.refresh)
    assert "MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE" not in refresh
    assert "start_attribute_refresh" in refresh

    worker = inspect.getsource(panel_module.SetupTrackerPanel._attributes_worker)
    assert "MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE" in worker
