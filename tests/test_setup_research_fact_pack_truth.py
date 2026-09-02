"""P3: the nightly fact pack tells the truth about its own evidence.

The 2026-08-31 pack is the case. It had 9 eligible cells - every one
AVWAPE_TO_FIRST_DEV/LONG against an ATR stop control, every one NEGATIVE - and
it printed them in a single table sorted by trimmed mean, so rows 10 onward were
n=1 cells reading +2.9R. The 80-row cap then dropped 508 more without saying
which kind. It pooled GENERAL (735 occurrences) and FAVORITE_ZONE_WATCH (486) as
if they were trade setups, which Appendix C explicitly forbids. And it reported
`n` as though outcome rows were samples, when the ERD says
`setup_occurrence` -> `outcome_path` is 1:N and those rows are correlated
diagnostics of one episode.

Five changes, all shadow-only: nothing here reaches a detector, score or alert.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import setup_research  # noqa: E402

NOW = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)


def _outcome(occurrence_id, recipe_id, net_r, *, first_hit="TARGET"):
    return {
        "occurrence_id": occurrence_id,
        "recipe_id": recipe_id,
        "net_r": net_r,
        "first_hit": first_hit,
        "result_state": "RESOLVED",
        "entry_at": datetime(2026, 8, 20, 14, 0, tzinfo=timezone.utc),
    }


def _occurrence(symbol, family, side="LONG", cluster="c1"):
    return {
        "symbol": symbol,
        "canonical_setup_id": family,
        "side": side,
        "dependency_cluster_id": cluster,
    }


def _thick(family, recipe, *, side="LONG", count=40, net_r=0.5, cluster_per_row=True):
    """A cell that clears the floor: n >= 30, >= 5 symbols, >= 5 sessions."""
    outcomes, occurrences = [], {}
    for index in range(count):
        identity = f"{family}-{side}-{index}"
        outcomes.append(
            {
                **_outcome(identity, recipe, net_r),
                "entry_at": datetime(2026, 8, 3 + (index % 10), 14, 0, tzinfo=timezone.utc),
            }
        )
        occurrences[identity] = _occurrence(
            f"S{index % 9}",
            family,
            side,
            cluster=f"cl{index}" if cluster_per_row else "one-move",
        )
    return outcomes, occurrences


# ==========================================================================
# item 1 - episodes beside rows
# ==========================================================================
def test_every_cell_reports_episodes_beside_n():
    """Fail-before-fix: `n_episodes` is not in the stats at all."""
    outcomes, occurrences = _thick("AVWAPE_TO_FIRST_DEV", "r1")
    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)

    stats = pack["eligible_policies"][0]["stats"]
    assert stats["n"] == 40
    assert stats["n_episodes"] == 40


def test_rows_sharing_one_episode_count_once():
    """`dependency_cluster_id` is the episode unit for evidence floors, and it
    deliberately excludes the setup family so simultaneous variants on one move
    share a cluster."""
    outcomes, occurrences = _thick("AVWAPE_TO_FIRST_DEV", "r1", cluster_per_row=False)
    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)

    stats = pack["eligible_policies"][0]["stats"]
    assert stats["n"] == 40
    assert stats["n_episodes"] == 1


def test_the_floor_still_counts_ROWS_in_this_packet():
    """Moving it is a change to what the model may narrate and belongs in its
    own packet (BD-81). Publishing both is what makes that decidable."""
    outcomes, occurrences = _thick("AVWAPE_TO_FIRST_DEV", "r1", cluster_per_row=False)
    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)

    stats = pack["eligible_policies"][0]["stats"]
    assert stats["n_episodes"] == 1
    assert stats["eligible"] is True, "the floor has NOT moved onto episodes yet"
    assert "does NOT yet gate" in stats["eligibility_rule"]


def test_the_pack_publishes_the_shape_of_its_whole_evidence_base():
    """Where the correlation actually is: cells are alternative recipes over
    the SAME occurrences. Measured on the live lake, 9,372 rows rested on 599
    occurrences and 287 clusters."""
    outcomes, occurrences = _thick("AVWAPE_TO_FIRST_DEV", "r1")
    extra, _ = _thick("AVWAPE_TO_FIRST_DEV", "r2")
    pack = setup_research.build_fact_pack(outcomes + extra, occurrences, {}, now=NOW)

    shape = pack["evidence_shape"]
    assert shape["outcome_rows"] == 80
    assert shape["distinct_occurrences"] == 40
    assert shape["distinct_episodes"] == 40
    assert shape["rows_per_occurrence"] == 2.0
    assert "double-counts" in shape["note"]


# ==========================================================================
# item 2 - the eligible block leads
# ==========================================================================
def test_the_eligible_block_leads_and_a_lucky_single_trade_cannot_outrank_it():
    """The 2026-08-31 failure exactly: nine real negative cells buried under 71
    n=1 cells at +2.9R because one list was sorted by trimmed mean."""
    outcomes, occurrences = _thick("AVWAPE_TO_FIRST_DEV", "atr1_1r", net_r=-0.4)
    lucky = _outcome("lucky-1", "sma3_3r", 2.983)
    outcomes.append(lucky)
    occurrences["lucky-1"] = _occurrence("ZZZ", "AVWAPE_TO_FIRST_DEV", "SHORT", "lucky")

    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)

    assert [row["recipe_id"] for row in pack["eligible_policies"]] == ["atr1_1r"]
    assert pack["policies"][0]["recipe_id"] == "atr1_1r", "the answer is first"
    assert pack["ineligible_policies"][0]["recipe_id"] == "sma3_3r"

    markdown = setup_research.render_markdown(pack)
    assert markdown.index("## Eligible policy cells") < markdown.index("Below the evidence floor")
    assert markdown.index("atr1_1r") < markdown.index("sma3_3r")


def test_the_ineligible_block_is_thickest_first_not_best_first():
    outcomes: list[dict] = []
    occurrences: dict[str, dict] = {}
    for index in range(12):  # thick but under the n floor
        identity = f"thick-{index}"
        outcomes.append(_outcome(identity, "thick_recipe", -0.2))
        occurrences[identity] = _occurrence(f"S{index}", "AVWAP_RETEST", "LONG", f"c{index}")
    outcomes.append(_outcome("thin-1", "thin_recipe", 2.9))
    occurrences["thin-1"] = _occurrence("QQQ", "AVWAP_RETEST", "LONG", "cthin")

    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)
    assert [row["recipe_id"] for row in pack["ineligible_policies"]] == [
        "thick_recipe",
        "thin_recipe",
    ]


def test_the_ineligible_block_is_bounded_and_says_what_it_dropped():
    outcomes: list[dict] = []
    occurrences: dict[str, dict] = {}
    for index in range(setup_research.MAX_INELIGIBLE_POLICY_ROWS + 7):
        identity = f"cell-{index}"
        outcomes.append(_outcome(identity, f"recipe_{index}", 0.1))
        occurrences[identity] = _occurrence("AAA", "AVWAP_RETEST", "LONG", f"c{index}")

    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)

    assert len(pack["ineligible_policies"]) == setup_research.MAX_INELIGIBLE_POLICY_ROWS
    assert pack["ineligible_policy_cells_dropped"] == 7
    assert pack["eligible_policy_cells_dropped"] == 0
    assert pack["policy_cells_dropped_from_pack"] == 7
    assert "7 further cell(s) below the floor" in setup_research.render_markdown(pack)


def test_an_empty_eligible_block_says_absent_not_flat():
    outcomes = [_outcome("a", "r", 0.4)]
    occurrences = {"a": _occurrence("AAA", "AVWAP_RETEST")}
    markdown = setup_research.render_markdown(
        setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)
    )
    assert "not a flat result" in markdown
    assert "absent one" in markdown


def test_an_older_pack_without_the_split_still_renders():
    """A pack is never edited; a new reading is a superseding sibling. The
    renderer must read a pack published before this split as its author
    published it, not invent a division they never made."""
    legacy = {
        "generated_at": "2026-08-31T03:00:00+00:00",
        "entry_contract": "first completed M5 close",
        "gate": {"eligible_policy_cells": 1, "note": "n"},
        "policies": [
            {"family": "F", "side": "LONG", "recipe_id": "r1",
             "stats": {"n": 33, "eligible": True, "clipped": {"trimmed_mean": -0.4}, "win_rate": 0.3}},
            {"family": "F", "side": "SHORT", "recipe_id": "r2",
             "stats": {"n": 1, "eligible": False, "clipped": {"trimmed_mean": 2.9}, "win_rate": 1.0}},
        ],
        "not_a_control_signal": "no control",
    }
    markdown = setup_research.render_markdown(legacy)
    assert markdown.index("r1") < markdown.index("r2")
    assert "## Eligible policy cells" in markdown


# ==========================================================================
# item 3 - non-trade families are excluded and named
# ==========================================================================
def test_non_trade_families_are_excluded_from_every_cell():
    """Appendix C: GENERAL "must not become a pooled 'setup' edge";
    FAVORITE_ZONE_WATCH is "never counted as a triggered trade setup"."""
    outcomes, occurrences = _thick("GENERAL", "r1")
    watch, watch_occ = _thick("FAVORITE_ZONE_WATCH", "r1")
    real, real_occ = _thick("AVWAPE_TO_FIRST_DEV", "r1")
    occurrences.update(watch_occ)
    occurrences.update(real_occ)

    pack = setup_research.build_fact_pack(
        outcomes + watch + real, occurrences, {}, now=NOW
    )

    families = {row["family"] for row in pack["policies"]}
    assert families == {"AVWAPE_TO_FIRST_DEV"}


def test_an_excluded_family_is_REPORTED_with_its_counts():
    """Absence is a first-class fact: a family that simply is not in the table
    reads as one with nothing to say."""
    outcomes, occurrences = _thick("GENERAL", "r1", count=35)
    pack = setup_research.build_fact_pack(outcomes, occurrences, {}, now=NOW)

    entry = next(row for row in pack["non_trade_families"] if row["family"] == "GENERAL")
    assert entry["role"] == "FALLBACK"
    assert entry["outcome_rows"] == 35
    assert entry["episodes"] == 35
    assert "pooled" in entry["reason"]
    assert "GENERAL" in setup_research.render_markdown(pack)
    assert "Excluded families" in setup_research.render_markdown(pack)


def test_an_unnamed_family_is_a_trade_setup():
    """A family added tomorrow is measured, not silently excluded. Excluding a
    real setup takes someone typing its name."""
    assert setup_research.family_role("AVWAPE_TO_FIRST_DEV") == setup_research.ROLE_TRADE
    assert setup_research.family_role("SOMETHING_NEW") == setup_research.ROLE_TRADE
    assert setup_research.family_role("GENERAL") == "FALLBACK"
    assert setup_research.family_role("FAVORITE_ZONE_WATCH") == "WATCH_STATE"


# ==========================================================================
# item 4 - coverage state
# ==========================================================================
def test_no_firing_history_reads_UNKNOWN_not_zero(tmp_path):
    """"0 of 32 covered" is a measured claim, and nobody measured it."""
    from research_warehouse import outcome_coverage

    state = outcome_coverage.coverage_state(tmp_path)
    assert state["outcome_buckets_covered"] is None
    assert state["outcome_bucket_count"] is None
    assert "UNKNOWN, not zero" in state["outcome_bucket_coverage_note"]


def test_firings_accumulate_and_the_ring_is_counted(tmp_path):
    from research_warehouse import outcome_coverage

    for bucket in (3, 7, 3, 11):
        assert outcome_coverage.record_firing(
            tmp_path, {"bucket": bucket, "bucket_count": 32, "status": "OK", "symbols": 4}
        )
    state = outcome_coverage.coverage_state(tmp_path)

    assert state["outcome_buckets_covered"] == 3, "a repeat is one bucket, not two"
    assert state["outcome_bucket_count"] == 32
    assert state["outcome_firings_considered"] == 4
    assert state["outcome_buckets_seen"] == [3, 7, 11]


def test_a_step_that_never_reached_a_bucket_is_not_recorded(tmp_path):
    """NO_OCCURRENCES has no bucket. Recording it as bucket 0 would claim
    coverage that never happened."""
    from research_warehouse import outcome_coverage

    assert not outcome_coverage.record_firing(tmp_path, {"status": "NO_OCCURRENCES"})
    assert not outcome_coverage.record_firing(tmp_path, None)
    assert not outcome_coverage.record_firing(None, {"bucket": 1, "bucket_count": 32})
    assert outcome_coverage.read_firings(tmp_path) == []


def test_a_truncated_last_line_costs_one_record_not_the_file(tmp_path):
    from research_warehouse import outcome_coverage

    outcome_coverage.record_firing(tmp_path, {"bucket": 5, "bucket_count": 32})
    path = outcome_coverage.coverage_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "outcome_bucket_cov')

    rows = outcome_coverage.read_firings(tmp_path)
    assert len(rows) == 1
    assert rows[0]["bucket"] == 5


def test_the_window_is_the_last_N_firings(tmp_path):
    from research_warehouse import outcome_coverage

    for bucket in range(40):
        outcome_coverage.record_firing(tmp_path, {"bucket": bucket, "bucket_count": 32})
    assert len(outcome_coverage.read_firings(tmp_path, limit=32)) == 32
    assert outcome_coverage.coverage_state(tmp_path, limit=32)["outcome_firings_considered"] == 32


def test_the_pack_prints_coverage_and_absence_reads_as_absence():
    pack = setup_research.build_fact_pack(
        [], {}, {},
        coverage={
            "outcome_buckets_covered": 6,
            "outcome_bucket_count": 32,
            "outcome_firings_considered": 9,
            "families_without_outcomes": ["POST_EARNINGS_52W_BREAK"],
            "first_m5_session": "2026-08",
        },
        now=NOW,
    )
    markdown = setup_research.render_markdown(pack)

    assert "6 of 32 symbol bucket(s)" in markdown
    assert "POST_EARNINGS_52W_BREAK" in markdown
    assert "not the same as measured and flat" in markdown
    assert "First M5 session in the lake: 2026-08" in markdown


def test_a_pack_with_no_coverage_block_claims_nothing():
    """A pack written before this block existed must not read as zero
    coverage."""
    markdown = setup_research.render_markdown(
        setup_research.build_fact_pack([], {}, {}, coverage={}, now=NOW)
    )
    assert "Bucket coverage" not in markdown
    assert "First M5 session" not in markdown


# ==========================================================================
# item 5 - the readout is not hard-filtered
# ==========================================================================
def test_slice_readout_defaults_to_the_pinned_slice_unchanged():
    """Existing callers get byte-identical rows to before the argument
    existed."""
    import inspect

    from research_warehouse import queries

    signature = inspect.signature(queries.slice_readout)
    assert signature.parameters["setups"].default is queries._UNSET


def test_slice_setups_itself_is_not_widened():
    """It is the pinned Phase-6 slice, and `cli._run_outcomes` uses it to
    choose which occurrences get the legacy slice recipe - widening it would
    change what the warehouse SIMULATES, not just what a reader sees."""
    from research_warehouse import occurrences as occ

    assert set(occ.SLICE_SETUPS) == {"AVWAPE_TO_FIRST_DEV", "POST_EARNINGS_CANDLE_BREAK"}


def test_the_panel_asks_for_every_family_when_the_filter_says_so(monkeypatch):
    pytest.importorskip("PySide6")
    from research_warehouse import queries
    from ui.panels import warehouse_readout_panel as panel_module

    asked: list[object] = []

    class _Store:
        pass

    monkeypatch.setattr(
        "research_warehouse.store.ResearchStore.open", staticmethod(lambda *a, **k: _Store())
    )
    monkeypatch.setattr(
        queries,
        "slice_readout",
        lambda store, **kwargs: asked.append(kwargs.get("setups", "DEFAULT")) or "snapshot",
    )

    assert panel_module.WarehouseReadoutPanel._read_lake(panel_module.SLICE_ONLY) == "snapshot"
    assert panel_module.WarehouseReadoutPanel._read_lake(panel_module.ALL_FAMILIES) == "snapshot"
    assert asked == ["DEFAULT", None]


def test_the_panel_shows_the_columns_the_query_already_computed():
    pytest.importorskip("PySide6")
    from ui.panels import warehouse_readout_panel as panel_module

    keys = {key for key, _label in panel_module.COLUMNS}
    for owed in ("n_symbols", "n_sessions", "n_truncated", "as_observed_only"):
        assert owed in keys, f"{owed} is computed by slice_readout and was dropped"


def test_choosing_a_family_reads_nothing_by_itself():
    """Nothing reads the lake on the render path (sec 20). The combo changes
    what the NEXT Refresh asks for."""
    pytest.importorskip("PySide6")
    import inspect

    from ui.panels import warehouse_readout_panel as panel_module

    source = inspect.getsource(panel_module.WarehouseReadoutPanel.__init__)
    assert "currentIndexChanged" not in source
    assert "_read_lake" not in source


# ==========================================================================
# the pack is never edited
# ==========================================================================
def test_a_rebuilt_pack_is_a_superseding_sibling(tmp_path):
    first = tmp_path / "2026-09-01.json"
    first.write_text("{}", encoding="utf-8")
    assert setup_research._superseding(first).name == "2026-09-01.1.json"
    (tmp_path / "2026-09-01.1.json").write_text("{}", encoding="utf-8")
    assert setup_research._superseding(first).name == "2026-09-01.2.json"


def test_the_pack_still_says_it_controls_nothing():
    pack = setup_research.build_fact_pack([], {}, {}, now=NOW)
    assert "cannot change scanners" in pack["not_a_control_signal"]
    assert "cannot change scanners" in setup_research.render_markdown(pack)
    assert json.dumps(pack, default=str)
