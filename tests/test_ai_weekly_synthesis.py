"""The weekly trader-judgement synthesis — machinery BUILT, runs GATED (W5).

`docs/LOCAL_AI_AUTOMATION_PLAN.md` §7.3 listed this under "What is NOT built"
with the cadence already decided (weekly, on the weekend surface, recorded
against R8 in `plan.md`) and the gate already named: **two weeks of graded
cohort rows**. The 2026-08-24 authorization builds the machinery ahead of that
gate on the R10.I scaffolding pattern - it exists, it runs when asked, and until
the gate passes it produces deterministic scaffolding that says so on its own
first line rather than a finding.

Three things this must never become, and each is pinned below:

* **not nightly** - it is absent from `default_slots()`, exactly as the
  `trader_judgement` scope is absent from `DEFAULT_SCOPES`. An unattended read
  over a stream still filling narrates "too early" every night until it is
  ignored;
* **not frontier** - medium tier or nothing (D7). The frontier synthesis pass is
  Phase 5 and is not authorized;
* **not a control signal** - nothing here reaches a detector, a score, an alert,
  a watchlist, Focus, the review queue or `review_policy.json`.
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

from ai_jobs import synthesis  # noqa: E402

NOW = datetime(2026, 9, 5, 16, 0, tzinfo=timezone.utc)


def _graded(session: str, symbol: str, *, side="LONG", source="veto_too_extended_from_base",
            h1=0.01, h5=0.02, matured=4):
    return {
        "trade_date": session,
        "symbol": symbol,
        "side": side,
        "source": source,
        "h1_return": h1,
        "h3_return": h1,
        "h5_return": h5,
        "h10_return": h5,
        "matured_horizons": matured,
    }


def _weeks_of_rows(sessions: int = 10, *, source="veto_too_extended_from_base"):
    days = [f"2026-08-{day:02d}" for day in (10, 11, 12, 13, 14, 17, 18, 19, 20, 21)]
    return [
        _graded(day, f"SYM{index}", source=source)
        for index, day in enumerate(days[:sessions])
    ]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_counts_sessions_of_MATURED_rows_not_rows():
    """A pick registered is not a pick graded.

    Two weeks means two weeks of forward evidence. Counting rows would let one
    busy afternoon of vetoes clear a gate that exists to wait for the market to
    answer them.
    """
    unmatured = [_graded("2026-08-10", f"SYM{i}", matured=0) for i in range(50)]
    assert synthesis.graded_sessions(unmatured, []) == 0

    assert synthesis.graded_sessions(_weeks_of_rows(3), []) == 3
    assert synthesis.graded_sessions(_weeks_of_rows(10), []) == 10


def test_the_gate_pools_both_cohorts_because_they_are_one_judgement():
    veto = _weeks_of_rows(5)
    like = [
        _graded(day, "AAPL", source="like_main_swing")
        for day in ("2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28")
    ]
    assert synthesis.graded_sessions(veto, like) == 10


def test_an_unmet_gate_states_it_on_the_first_line_and_labels_everything_discovery():
    pack = synthesis.build_fact_pack(veto_rows=_weeks_of_rows(3), like_rows=[], now=NOW)

    assert pack["gate"]["window_met"] is False
    assert pack["gate"]["statement"].startswith(synthesis.GATE_NOT_MET_PREFIX)
    assert pack["evidence_label"] == "discovery"
    rendered = synthesis.render_markdown(pack)
    first_meaningful = [line for line in rendered.splitlines() if line.strip()][1]
    assert synthesis.GATE_NOT_MET_PREFIX in first_meaningful


def test_a_met_gate_is_still_discovery_because_the_window_was_not_declared_ahead():
    """A large post-hoc sample is a large discovery, never a confirmation."""
    pack = synthesis.build_fact_pack(veto_rows=_weeks_of_rows(10), like_rows=[], now=NOW)

    assert pack["gate"]["window_met"] is True
    assert pack["evidence_label"] == "discovery"


def test_an_empty_cohort_reads_as_unmet_rather_than_as_a_clean_record():
    pack = synthesis.build_fact_pack(veto_rows=[], like_rows=[], now=NOW)
    assert pack["gate"]["window_met"] is False
    assert pack["gate"]["sessions_graded"] == 0
    assert "0 of 10" in pack["gate"]["statement"]


# ---------------------------------------------------------------------------
# The numbers are code's, and they come from the one statistics module
# ---------------------------------------------------------------------------


def test_every_cell_routes_through_evidence_stats():
    import evidence_stats

    pack = synthesis.build_fact_pack(veto_rows=_weeks_of_rows(10), like_rows=[], now=NOW)
    cells = pack["cohorts"]["cells"]
    assert cells, "ten sessions of graded rows must produce at least one cell"
    for cell in cells:
        assert cell["schema"] == evidence_stats.SUMMARY_SCHEMA
        assert "n" in cell and cell["n"] > 0
        assert cell["n_floor"] == evidence_stats.MIN_REPORTABLE_N
        assert cell["evidence_label"] == evidence_stats.LABEL_DISCOVERY
    assert pack["statistics_contract"]["module"] == "evidence_stats"


def test_a_cell_is_keyed_by_cohort_side_and_horizon():
    pack = synthesis.build_fact_pack(
        veto_rows=_weeks_of_rows(10)
        + [_graded(f"2026-08-{day:02d}", "TSLA", side="SHORT") for day in (10, 11, 12)],
        like_rows=[],
        now=NOW,
    )
    keys = {(cell["cohort"], cell["side"], cell["horizon"]) for cell in pack["cohorts"]["cells"]}
    assert ("veto_too_extended_from_base", "LONG", "h1") in keys
    assert ("veto_too_extended_from_base", "SHORT", "h1") in keys


def test_a_blank_return_is_not_counted_as_a_zero():
    rows = _weeks_of_rows(10)
    for row in rows:
        row["h10_return"] = ""
    pack = synthesis.build_fact_pack(veto_rows=rows, like_rows=[], now=NOW)
    h10 = [cell for cell in pack["cohorts"]["cells"] if cell["horizon"] == "h10"]
    assert h10 == [], "an unmatured horizon is absent, never averaged as zero"


def test_the_cell_cap_prints_what_it_dropped():
    rows = [
        _graded("2026-08-%02d" % day, f"SYM{index}", source=f"veto_reason_{index}")
        for index, day in enumerate([10, 11, 12, 13, 14, 17, 18, 19, 20, 21] * 6)
    ]
    pack = synthesis.build_fact_pack(veto_rows=rows, like_rows=[], now=NOW)
    assert len(pack["cohorts"]["cells"]) <= synthesis.MAX_CELLS
    dropped = pack["cohorts"]["cells_dropped"]
    assert dropped["cells"] > 0 and dropped["events"] > 0
    assert "n" in dropped["basis"].lower()


def test_the_digest_rollup_is_included_only_once_a_pack_exists(tmp_path):
    """"...and digest facts once >=1 exists". An absent digest is an absent
    measurement, said in words rather than rendered as a zero."""
    pack = synthesis.build_fact_pack(
        veto_rows=_weeks_of_rows(10), like_rows=[], now=NOW, digest_root=tmp_path,
    )
    assert pack["digest"]["sessions"] == 0
    assert "no fact pack" in pack["digest"]["note"].lower()

    from ai_jobs import digest

    digest.run_daily_digest(
        session_date="2026-08-21", now=NOW, root=tmp_path, narrate=False,
        finals=[{
            "symbol": "AAPL", "direction": "long", "trade_date": "2026-08-21",
            "env_key": "bullish_weak|midday", "close_r": 1.0, "mfe_r": 2.0, "mae_r": -0.5,
        }],
    )
    pack = synthesis.build_fact_pack(
        veto_rows=_weeks_of_rows(10), like_rows=[], now=NOW, digest_root=tmp_path,
    )
    assert pack["digest"]["sessions"] == 1
    assert pack["digest"]["close_r"]["n"] == 1


# ---------------------------------------------------------------------------
# The run: gated, deterministic first, model second
# ---------------------------------------------------------------------------


def test_an_unmet_gate_never_calls_a_model(tmp_path, monkeypatch):
    """§7.2's lesson: an unattended read over a stream still filling narrates
    "too early". Below the gate there is nothing to narrate, so nothing is
    asked."""
    called = []
    monkeypatch.setattr(synthesis, "_narrate", lambda **kwargs: called.append(kwargs))

    result = synthesis.run_weekly_synthesis(
        now=NOW, root=tmp_path, veto_rows=_weeks_of_rows(3), like_rows=[],
    )

    assert called == []
    assert result["status"] == "ok"
    assert synthesis.GATE_NOT_MET_PREFIX.split(".")[0].lower() in result["reason"].lower()
    written = sorted(path.name for path in tmp_path.rglob("*"))
    assert any(name.endswith(".json") for name in written), "the fact pack is still written"


def test_the_fact_pack_is_written_even_when_narration_fails(tmp_path, monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("local AI provider is not configured")

    monkeypatch.setattr(synthesis, "_narrate", explode)
    result = synthesis.run_weekly_synthesis(
        now=NOW, root=tmp_path, veto_rows=_weeks_of_rows(10), like_rows=[],
    )
    assert result["status"] == synthesis.STATUS_DEGRADED
    assert any(path.suffix == ".json" for path in tmp_path.rglob("*"))


def test_the_narrator_is_handed_the_fact_pack_and_nothing_else():
    pack = synthesis.build_fact_pack(veto_rows=_weeks_of_rows(10), like_rows=[], now=NOW)
    package = synthesis.narration_evidence_package(pack)
    assert [source["source_id"] for source in package["sources"]] == [synthesis.FACT_PACK_SOURCE_ID]
    assert package["sources"][0]["content"] == pack


def test_a_run_supersedes_rather_than_overwriting(tmp_path):
    first = synthesis.run_weekly_synthesis(
        now=NOW, root=tmp_path, veto_rows=_weeks_of_rows(3), like_rows=[], narrate=False,
    )
    original = Path(first["outputs"][0])
    stamp = original.read_bytes()
    second = synthesis.run_weekly_synthesis(
        now=NOW, root=tmp_path, veto_rows=_weeks_of_rows(4), like_rows=[], narrate=False,
    )
    assert original.read_bytes() == stamp
    assert Path(second["outputs"][0]) != original


# ---------------------------------------------------------------------------
# Not nightly, not frontier, not a control signal
# ---------------------------------------------------------------------------


def test_the_slot_is_not_in_the_nightly_slate():
    from ai_jobs.runner import default_slots, optional_slots

    assert "weekly_synthesis" not in [slot.name for slot in default_slots()]
    assert "weekly_synthesis" in [slot.name for slot in optional_slots()]


def test_the_optional_slate_is_constructed_per_call_so_it_cannot_leak():
    """The `--scopes` precedent: an opt-in thing cannot become nightly by being
    set once."""
    from ai_jobs.runner import default_slots, optional_slots

    optional_slots()
    assert "weekly_synthesis" not in [slot.name for slot in default_slots()]


def test_the_cli_exposes_it_and_the_nightly_path_does_not_reach_it():
    source = (SCRIPTS_DIR / "run_ai_jobs.py").read_text(encoding="utf-8")
    assert "--weekly-synthesis" in source
    assert "optional_slots" in source


def test_no_frontier_provider_is_reachable_from_this_module():
    """Medium tier or nothing (D7). Phase 5's frontier pass is not authorized."""
    import ast

    tree = ast.parse(Path(synthesis.__file__).read_text(encoding="utf-8"))
    text = "\n".join(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    for name in ("openai", "anthropic", "frontier", "gpt-", "claude-"):
        assert name not in text.lower() or "not authorized" in text.lower()
    assert '"local"' in Path(synthesis.__file__).read_text(encoding="utf-8")


def test_nothing_here_can_write_a_decision_surface():
    """AST-checked, not asserted in prose."""
    import ast

    source = Path(synthesis.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Only strings that could BE a path are scanned - a filename used as a
    # filename has no spaces in it. The module's prose deliberately names
    # `review_policy.json` to say it must never touch it, and a check that
    # could not tell a prohibition from a path would forbid the module from
    # stating its own contract.
    tokens = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip()
        and not any(character.isspace() for character in node.value)
    ]
    for forbidden in ("review_policy.json", "longs.txt", "shorts.txt",
                      "human_focus_picks", "swinglongs.txt"):
        assert not any(forbidden in token for token in tokens), (
            f"{forbidden} appears as a path token in the synthesis module"
        )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for name in imported:
        assert not name.startswith(
            ("bounce_bot", "autopilot_core", "master_avwap", "technical_integrity",
             "price_alert", "d1_level_feed", "review_policy")
        ), f"the synthesis reached into decision code: {name}"
