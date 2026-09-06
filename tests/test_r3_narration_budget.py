"""R3 items 1-3 - the narration outgrew the model, and the retry made it worse.

Evidence from the 2026-09-01 night: `setup_research` published its pack at
03:55, 04:30 and 05:00 - three superseding siblings, 29 minutes of lake reads -
and every attempt logged *"narration absent: the local server truncated the
prompt: sent ~176827 tokens (442068 chars), server reported seeing 32771"*.

Two independent faults. The package sent the WHOLE pack, which P3 and P8 had
grown past the model's window; and a truncated prompt returned
`degraded_no_narrative` under `max_attempts=3`, so the runner re-ran a
ten-minute lake pass twice more to fail identically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# The four prose constants a real cell repeats, quoted from the 2026-09-01 pack.
# They are module constants interpolated into every `stats` block, so they are
# identical by construction - which is exactly why they compress.
ELIGIBILITY_RULE = (
    "n >= 30 OUTCOME ROWS, at least 5 symbols, and at least 5 entry sessions; "
    "still discovery, never confirmation."
)
N_FLOOR_NOTE = (
    "n >= 30 is NECESSARY, not sufficient: it clears the floor and says nothing "
    "about concentration, session coverage, or whether the window was declared "
    "in advance"
)
PF_CONVENTION = (
    "profit factor = sum(gains) / abs(sum(losses)). With no losing rows the "
    "denominator is zero and PF is reported as null with all_wins=true."
)


def _cell(index: int) -> dict:
    """A cell shaped like the real thing: measurements nested under `stats`,
    each one carrying the same paragraphs of prose as all the others."""
    return {
        "family": f"FAMILY_{index}",
        "side": "LONG",
        "recipe_id": f"m5close_current_anchor{index % 3 + 1}_2r_v1",
        "stats": {
            "schema": "evidence_summary_v1",
            "eligible": True,
            "evidence_label": "discovery",
            "n": 40 + index,
            "n_episodes": 40 + index,
            "n_floor": 30,
            "meets_n_floor": True,
            "win_rate": 0.41,
            "stop_rate": 0.26,
            "counts": {"events": 40 + index, "sessions": 8, "symbols": 34},
            "raw": {"mean": 0.17, "median": 0.73, "p10": -1.07, "p90": 0.99},
            "profit_factor": {"value": 1.5, "convention": PF_CONVENTION},
            "bootstrap": {
                "low": -0.09,
                "high": 0.42,
                "measured": True,
                "interval": "5-95 percentile of a session-block bootstrap",
            },
            "eligibility_rule": ELIGIBILITY_RULE,
            "n_floor_note": N_FLOOR_NOTE,
        },
    }


def _pack(*, eligible: int = 60, ineligible: int = 40, contexts: int = 80) -> dict:
    from ai_jobs import setup_research

    pack = setup_research.build_fact_pack(
        [], {}, {},
        coverage={"outcomes": 12_439, "first_m5_session": "2026-07-01"},
        recipe_ids=["m5close_current_anchor1_2r_v1", "htf_lrsi_h1_up50_2r_v1"],
    )
    pack["eligible_policies"] = [_cell(index) for index in range(eligible)]
    pack["ineligible_policies"] = [_cell(index) for index in range(ineligible)]
    pack["market_context_cells"] = [_cell(index) for index in range(contexts)]
    pack["gate"] = {"eligible_policy_cells": eligible, "met": eligible > 0, "note": ""}
    return pack


# ---------------------------------------------------------------------------
# 1. a bounded view, and a refusal instead of a sheared prompt
# ---------------------------------------------------------------------------


def test_the_view_carries_the_finding_and_omits_the_input():
    from ai_jobs import setup_research

    # N3 (2026-09-05) made the view BOUNDED: it now cuts itself to the local
    # model's budget and says so in `narrated`. That is a size rule and a
    # different question from this one, which is about WHAT KIND of thing may be
    # in the view at all - so the budget is opened wide here and the cut is
    # tested on its own in `tests/test_n3_narration_bounded.py`. Without this
    # the test would silently be measuring whatever budget the machine running
    # it happens to resolve (11,066 chars under the test harness' isolated
    # settings, 78,119 on the desk) instead of the rule it was written for.
    view = setup_research.narration_view(_pack(), budget=10_000_000)

    # Every eligible cell is a CANDIDATE: those ARE the finding.
    assert len(view["eligible_policies"]) == 60
    # And none of the input.
    assert "ineligible_policies" not in view
    assert "market_context_cells" not in view
    assert "policies" not in view
    # But their COUNTS, so the model can say what it was not shown.
    assert view["omitted"]["ineligible_policies"] == 40
    assert view["omitted"]["market_context_cells"] == 80
    assert view["omitted"]["outcome_rows"] == 12_439
    # The things a reader needs first.
    assert view["gate"]["eligible_policy_cells"] == 60
    assert view["coverage"]["outcomes"] == 12_439
    assert "evidence_shape" in view and "non_trade_families" in view


# The desk's own budget, measured 2026-09-02: `ai_local_evidence_budget_chars`
# is 0 ("derive it"), so it resolves to the ceiling for the configured local
# context window. Pinned here rather than read live, because a test that asks
# the machine what it can afford passes for a reason that is not the code.
DESK_BUDGET_CHARS = 78_119


def test_a_sixty_cell_pack_narrates_under_the_budget(monkeypatch):
    import ai_summary
    from ai_jobs import setup_research

    monkeypatch.setattr(
        ai_summary, "local_evidence_budget_chars", lambda: DESK_BUDGET_CHARS
    )
    pack = _pack()
    package = setup_research._evidence_package(pack)
    sent = package["sources"][0]["content"]
    encoded = json.dumps(sent, sort_keys=True, default=str).encode("utf-8")
    assert len(encoded) <= DESK_BUDGET_CHARS

    # And by a wide margin, not by a hair: the whole pack is what sheared, and
    # the view has to leave room for the grid to keep growing.
    whole = len(json.dumps(pack, sort_keys=True, default=str).encode("utf-8"))
    assert len(encoded) < whole / 3


def test_the_prose_every_cell_repeats_is_stated_once():
    """~900 of each 1,900-char cell was four constants written again."""
    from ai_jobs import setup_research

    view = setup_research.narration_view(_pack())
    conventions = view["conventions"]
    assert "stats.eligibility_rule" in conventions
    assert "stats.profit_factor.convention" in conventions
    for cell in view["eligible_policies"]:
        assert "eligibility_rule" not in cell["stats"]
        assert "convention" not in cell["stats"]["profit_factor"]
        # and the MEASUREMENTS beside them are untouched
        assert "value" in cell["stats"]["profit_factor"]
        assert cell["stats"]["n"]


def test_a_convention_two_cells_disagree_on_is_never_hoisted():
    """Stating it once would silently restate one of them."""
    from ai_jobs import setup_research

    pack = _pack(eligible=3)
    pack["eligible_policies"][1]["stats"]["n_floor_note"] = "a different rule"
    view = setup_research.narration_view(pack)

    assert "stats.n_floor_note" not in view["conventions"]
    for cell in view["eligible_policies"]:
        assert "n_floor_note" in cell["stats"], "it stays inline on every cell"
    # the ones that DO agree are still hoisted
    assert "stats.eligibility_rule" in view["conventions"]


def test_the_hash_is_over_what_was_actually_sent():
    """Hashing the pack while sending a view would make traceability a lie."""
    import hashlib

    from ai_jobs import setup_research

    package = setup_research._evidence_package(_pack(eligible=3))
    sent = package["sources"][0]["content"]
    expected = hashlib.sha256(
        json.dumps(sent, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert package["sources"][0]["sha256"] == expected


def test_an_oversized_view_is_refused_before_any_provider_call(monkeypatch):
    """The 2026-09-01 failure, prevented rather than reported."""
    import ai_summary
    from ai_jobs import setup_research

    calls = []
    monkeypatch.setattr(
        ai_summary, "request_ai_summary", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_evidence_budget_chars", lambda: 500)

    with pytest.raises(setup_research.NarrationTooLarge) as caught:
        setup_research._narrate(_pack())

    assert calls == [], "the provider must never see a prompt we know is too big"
    message = str(caught.value)
    assert "chars against a budget of 500" in message
    assert "60 eligible cell(s)" in message, "it names the size and what still exists"


# ---------------------------------------------------------------------------
# 2. a missing narration is not a failed job
# ---------------------------------------------------------------------------


def test_a_truncating_provider_yields_one_pack_and_one_ok_row(tmp_path, monkeypatch):
    """`degraded_no_narrative` under max_attempts=3 re-ran a ten-minute lake
    pass twice more and published two more packs, to fail identically."""
    import ai_summary
    from ai_jobs import setup_research

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier: "stub")

    def _truncated(**_kwargs):
        raise RuntimeError(
            "the local server truncated the prompt: sent ~176827 token(s)"
        )

    monkeypatch.setattr(ai_summary, "request_ai_summary", _truncated)

    # A pack that MEETS the gate, so the job actually reaches the model. Built
    # BEFORE the patch: `_pack` calls `build_fact_pack` itself.
    ready = _pack(eligible=3)
    monkeypatch.setattr(setup_research, "build_fact_pack", lambda *a, **k: ready)

    result = setup_research.run_setup_research(
        root=tmp_path,
        session_date="2026-09-01",
        inputs=([], {}, {}, {"outcomes": 10, "recipe_ids": ["r1"]}),
    )

    assert result["status"] == "ok", "the pack is the product; the narration is words"
    assert "narration absent" in result["reason"]
    packs = sorted((tmp_path / "2026").glob("2026-09-01*.json"))
    assert len(packs) == 1, packs


def test_the_job_never_returns_a_status_the_runner_would_retry():
    """Any retry of THIS job re-enters the lake, which cannot fix a long prompt."""
    source = (ROOT / "scripts" / "ai_jobs" / "setup_research.py").read_text(
        encoding="utf-8"
    )
    assert '"status": "degraded_no_narrative"' not in source
    assert "narration absent" in source


# ---------------------------------------------------------------------------
# 3. provenance
# ---------------------------------------------------------------------------


def test_the_pack_says_which_code_and_which_grid_built_it():
    """Two packs from one night differed by 3,067 rows and neither said why."""
    from ai_jobs import setup_research

    pack = _pack()
    assert pack["built_by_commit"], "a commit or 'unknown', never absent"
    assert pack["recipe_ids"] == [
        "m5close_current_anchor1_2r_v1",
        "htf_lrsi_h1_up50_2r_v1",
    ]
    # And it travels into the narration, so the words are traceable too.
    view = setup_research.narration_view(pack)
    assert view["built_by_commit"] == pack["built_by_commit"]
    assert view["recipe_ids"] == pack["recipe_ids"]


def test_the_grid_is_never_re_derived_from_the_module():
    """Re-deriving would state the grid this CODE knows, not the one these ROWS
    came from - which is the one thing the field exists to distinguish."""
    from ai_jobs import setup_research

    pack = setup_research.build_fact_pack([], {}, {}, coverage={"outcomes": 0})
    assert pack["recipe_ids"] == [], "empty means the caller did not say"


def test_provenance_never_costs_the_pack(monkeypatch):
    """A missing commit is a less traceable pack; a raise would be no pack."""
    from ai_jobs import setup_research

    monkeypatch.setattr(setup_research, "_BUILT_BY_COMMIT", None)
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))
    )
    assert setup_research._built_by_commit() == "unknown"
    monkeypatch.setattr(setup_research, "_BUILT_BY_COMMIT", None)
