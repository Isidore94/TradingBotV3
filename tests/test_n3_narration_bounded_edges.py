"""N3 edges the packet's seven tests do not reach (builder-added, 2026-09-05).

Three of them:

* the refusal has TWO ways to fire - the head is over the budget, or the head
  fits and the first cell does not - and the tester's case only exercises the
  first. The second is the one that will actually happen as the grid grows;
* a pack that is NOT narrated (the gate is not met) must not print a coverage
  line claiming 0 of 0, because "no narration was attempted" and "the narration
  covered nothing" are different facts;
* the cut has to be deterministic and inside the budget for EVERY budget, not
  just the desk's, or the first night with a different `num_ctx` re-opens this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

sys.path.insert(0, str(ROOT / "tests"))

from test_n3_narration_bounded import (  # noqa: E402
    _live_shaped_pack,
    _selection,
    _synthetic_cell,
    _synthetic_pack,
)

import ai_summary  # noqa: E402
from ai_jobs import setup_research  # noqa: E402


def _encoded(view) -> bytes:
    return json.dumps(view, sort_keys=True, default=str).encode("utf-8")


def test_the_refusal_also_fires_when_the_head_fits_and_the_first_cell_does_not(
    monkeypatch,
):
    """The head under the budget is not enough - a view with no cell says nothing.

    Without this branch the job would send a head-only prompt and the model
    would narrate the gate line with no evidence under it, which reads as a
    finding of nothing rather than as a refusal.
    """
    pack = _synthetic_pack([_synthetic_cell(0, n=621, mean_r=0.5)])
    view, head_chars = setup_research._bounded_narration_view(pack, 10_000_000)
    cell_chars = len(_encoded(view)) - head_chars
    assert cell_chars > 0

    # A budget that fits the head with room to spare, but not the first cell.
    budget = head_chars + cell_chars // 2
    monkeypatch.setattr(ai_summary, "local_evidence_budget_chars", lambda: budget)

    bounded, _head = setup_research._bounded_narration_view(pack, budget)
    assert len(_encoded(bounded)) <= budget, "the head-only view is inside the budget"
    assert bounded["eligible_policies"] == []

    with pytest.raises(setup_research.NarrationTooLarge) as excinfo:
        setup_research._evidence_package(pack)
    assert str(head_chars) in str(excinfo.value)


def test_a_pack_that_is_not_narrated_prints_no_coverage_line(tmp_path, monkeypatch):
    """Below the evidence floor no model is called, so nothing was covered."""
    pack = _synthetic_pack([])
    assert pack["gate"]["met"] is False
    monkeypatch.setattr(setup_research, "build_fact_pack", lambda *a, **k: pack)

    result = setup_research.run_setup_research(
        root=tmp_path, inputs=([], {}, {}, {"outcomes": 0})
    )

    assert result["status"] == "ok"
    assert "no model called below the evidence floor" in result["reason"]
    markdown = next(tmp_path.rglob("*.md")).read_text(encoding="utf-8")
    assert "Narration covers" not in markdown
    assert "## Narration" not in markdown


@pytest.mark.parametrize("budget", [12_000, 25_000, 78_119, 200_000])
def test_the_cut_is_inside_every_budget_and_is_a_prefix_of_the_same_order(
    budget, monkeypatch
):
    """A wider budget only ever ADDS cells to the end of the same list.

    The order is fixed by the pack, so the 12,000-char selection has to be the
    first cells of the 200,000-char one. If it were not, the cut would be
    deciding the order as well as the length and "selected by evidence count"
    would stop being true of the smaller nights.
    """
    monkeypatch.setattr(ai_summary, "local_evidence_budget_chars", lambda: budget)
    pack = _live_shaped_pack()

    view = setup_research.narration_view(pack)
    assert len(_encoded(view)) <= budget
    assert view["narrated"]["of"] == 619

    widest = setup_research.narration_view(pack, budget=10_000_000)
    assert widest["narrated"]["eligible_policy_cells"] == 619
    selection = _selection(view)
    assert selection == _selection(widest)[: len(selection)]
