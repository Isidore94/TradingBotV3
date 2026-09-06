"""N3 - the setup_research narration view is BOUNDED, and says what it left out.

The nightly `setup_research` slot has published no narration for four nights.
The deterministic pack keeps growing (gate #59's lake recompute landed 141,299
recipe outcomes overnight on 2026-09-05) and the whole eligible block is handed
to the model: on the 2026-09-04 pack the encoded view is **658,292 chars against
a 78,119-char budget**, so `_evidence_package` refuses and the night ends with
`narration absent`. No budget a 64k-context model can read will ever fit 658k
chars; the view has to SELECT.

Every fixture in this file is pinned from that live pack
(`tests/fixtures/setup_research_narration_v1.json`, written by the pre-N3
nightly at commit 7f2273d3), so nothing here is a self-portrait of the fix.
The pack head, six eligible cells and the after-like block are verbatim; the
remaining cells are synthesized in that exact shape.

The selection rule under test is a SIZE rule and never a ranking by result
(gate #43): a cell that looks good early is exactly what the frozen research
window protects, so no R statistic may enter the ordering.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ai_summary  # noqa: E402
from ai_jobs import setup_research  # noqa: E402

FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "setup_research_narration_v1.json").read_text(
        encoding="utf-8"
    )
)

#: The desk's measured budget on 2026-09-05: 64k context minus the generation
#: cap at the safe chars-per-token. Pinned so the tests do not read the desk's
#: `local_settings.json`.
BUDGET = 78_119

#: The live 2026-09-04 pack: 619 eligible policy cells, encoding to 658,292
#: chars through the unbounded view. Both numbers are from the pack itself.
LIVE_ELIGIBLE_CELLS = 619

#: The evidence floor every eligible cell cleared: n >= 30 outcome rows.
N_FLOOR = 30

# The families and recipe stems the live pack actually carries, used to spread
# the synthetic cells over the same value space as the real ones.
_SIDES = ("LONG", "SHORT")


def _encoded(view) -> bytes:
    """The bytes `_evidence_package` hashes and sends, encoded its way."""
    return json.dumps(view, sort_keys=True, default=str).encode("utf-8")


def _synthetic_cell(index: int, *, n: int, mean_r: float) -> dict:
    """One eligible policy cell in the live pack's exact shape.

    Cloned from a pinned real cell, then given a distinct `recipe_id` and the
    requested evidence count and R statistics. The five prose constants the
    view hoists (`eligibility_rule`, `n_floor_note`, the profit-factor
    convention, the bootstrap interval and the cell schema) are left BYTE
    IDENTICAL across every cell, because that is what they are on the live pack
    and `_hoist_shared_conventions` refuses to hoist a path one cell disagrees
    on - a synthetic pack whose cells disagreed would have a per-cell size the
    real one does not have.
    """
    template = FIXTURE["eligible_cells"][index % len(FIXTURE["eligible_cells"])]
    cell = copy.deepcopy(template)
    cell["recipe_id"] = f"{template['recipe_id'].rsplit('_v1', 1)[0]}_syn{index:04d}_v1"
    cell["side"] = _SIDES[index % len(_SIDES)]
    stats = cell["stats"]
    stats["n"] = n
    stats["n_episodes"] = n
    stats["counts"] = {"events": n, "sessions": 14, "symbols": min(n, 33)}
    stats["eligible"] = True
    stats["meets_n_floor"] = True
    for block in ("clipped", "raw"):
        stats[block] = dict(stats[block])
        stats[block]["mean"] = round(mean_r, 4)
        stats[block]["trimmed_mean"] = round(mean_r, 4)
        stats[block]["median"] = round(mean_r, 4)
    stats["win_rate"] = round(min(0.99, max(0.01, 0.5 + mean_r / 10)), 4)
    stats["profit_factor"] = dict(stats["profit_factor"])
    stats["profit_factor"]["value"] = round(1.0 + mean_r, 4)
    stats["stop_rate"] = round(min(0.99, max(0.01, 0.5 - mean_r / 10)), 4)
    return cell


def _synthetic_pack(
    cells: list[dict],
    *,
    after_like_cells: list[dict] | None = None,
) -> dict:
    """A fact pack: the live pack's verbatim head with the given cells in it."""
    pack = copy.deepcopy(FIXTURE["pack_head"])
    pack["eligible_policies"] = cells
    pack["ineligible_policies"] = []
    pack["market_context_cells"] = []
    pack["policies"] = list(cells)
    pack["gate"] = dict(pack.get("gate") or {})
    pack["gate"]["eligible_policy_cells"] = len(cells)
    pack["gate"]["met"] = bool(cells)
    after_like = copy.deepcopy(FIXTURE["after_like_head"])
    after_like["cells"] = list(after_like_cells or ())
    pack["after_like"] = after_like
    return pack


def _after_like_cell(index: int, *, n_episodes: int, eligible: bool) -> dict:
    """One after-like grid cell, in the live pack's flat shape.

    The live grid's twenty cells are all BELOW the floor, so `eligible` is
    flipped here to model a night where the grid cleared it - `eligible` is the
    exact flag `narration_view` filters `after_like_eligible` on (P10 C3).
    """
    cell = copy.deepcopy(FIXTURE["after_like_cell"])
    cell["day_offset"] = index
    cell["n"] = n_episodes
    cell["n_episodes"] = n_episodes
    cell["eligible"] = eligible
    cell["meets_n_floor"] = eligible
    return cell


def _live_shaped_pack(count: int = LIVE_ELIGIBLE_CELLS) -> dict:
    """`count` eligible cells with distinct evidence counts and varied R.

    Every count is distinct and every one is at or above the live n floor of 30
    (the live pack runs 30..621 with ties), so the intended order is total on
    the count alone. Mean R is deliberately ANTI-correlated with it: a selection
    that ranked by result would come back in the opposite order.
    """
    cells = [
        _synthetic_cell(
            index,
            n=N_FLOOR + count - 1 - index,
            mean_r=-3.0 + index * (6.0 / count),
        )
        for index in range(count)
    ]
    return _synthetic_pack(cells)


def _selection(view) -> list[tuple[str, str, str]]:
    return [
        (cell["family"], cell["recipe_id"], cell["side"])
        for cell in view["eligible_policies"]
    ]


@pytest.fixture
def budget(monkeypatch):
    """Pin the evidence budget so no desk setting can change a result."""

    def _set(value: int = BUDGET) -> int:
        monkeypatch.setattr(
            ai_summary, "local_evidence_budget_chars", lambda: int(value)
        )
        return int(value)

    _set()
    return _set


@pytest.fixture
def stub_model(monkeypatch):
    """A local provider that answers without a model, capturing what was sent."""
    sent: list[dict] = []

    def _request(**kwargs):
        sent.append(kwargs["evidence"])
        return {"model": "stub-local", "summary": {"headline": "stub"}}

    monkeypatch.setattr(ai_summary, "local_provider_enabled", lambda: True)
    monkeypatch.setattr(ai_summary, "local_model", lambda tier="medium": "stub-local")
    monkeypatch.setattr(ai_summary, "request_ai_summary", _request)
    return sent


def test_narration_view_fits_the_budget(budget):
    """619 real-shaped cells against the real budget: select, do not refuse.

    On e7b12ebe the view carries all 619 cells, encodes to ~658k chars and
    `_evidence_package` raises `NarrationTooLarge`, which is why four nights
    running have no narration.
    """
    pack = _live_shaped_pack()
    assert len(pack["eligible_policies"]) == LIVE_ELIGIBLE_CELLS

    view = setup_research.narration_view(pack)
    encoded = _encoded(view)
    assert len(encoded) <= BUDGET, (
        f"the narration view is {len(encoded)} chars against a {BUDGET}-char budget"
    )

    narrated = view["narrated"]
    assert narrated["of"] == LIVE_ELIGIBLE_CELLS
    assert narrated["eligible_policy_cells"] == len(view["eligible_policies"])
    assert 0 < narrated["eligible_policy_cells"] < LIVE_ELIGIBLE_CELLS

    # The cells kept are the most-measured ones, in evidence-count order.
    kept = [cell["stats"]["n"] for cell in view["eligible_policies"]]
    assert kept == sorted(kept, reverse=True)
    assert kept[0] == N_FLOOR + LIVE_ELIGIBLE_CELLS - 1
    assert min(kept) >= N_FLOOR

    # Item 4: the hash is over the bounded view - what was actually sent.
    package = setup_research._evidence_package(pack)
    sent = package["sources"][0]["content"]
    sent_bytes = _encoded(sent)
    assert len(sent_bytes) <= BUDGET
    assert package["sources"][0]["sha256"] == hashlib.sha256(sent_bytes).hexdigest()


def test_selection_is_by_evidence_count_never_by_result(budget):
    """Gate #43: the ordering is a SIZE rule, blind to every R statistic.

    Two halves. First a TIE: two cells with the same evidence count and
    opposite mean R must come back in `recipe_id` order, not best-first. Then a
    SHUFFLE: every result statistic is permuted across all cells while the
    evidence counts stay put, and the selection must not move - twenty times,
    from a pinned seed.
    """
    budget(40_000)

    tied_high = _synthetic_cell(0, n=400, mean_r=-2.5)
    tied_high["recipe_id"] = "zzz_tied_recipe_v1"
    tied_low = _synthetic_cell(1, n=400, mean_r=+2.5)
    tied_low["recipe_id"] = "aaa_tied_recipe_v1"
    others = [
        _synthetic_cell(index + 2, n=300 - index, mean_r=(index % 7) - 3.0)
        for index in range(60)
    ]
    pack = _synthetic_pack([tied_high, tied_low, *others])

    view = setup_research.narration_view(pack)
    selection = _selection(view)

    # The tie breaks on recipe_id ascending, so the +2.5R cell does NOT jump it.
    ordered_recipes = [recipe for _family, recipe, _side in selection]
    assert ordered_recipes[:2] == ["aaa_tied_recipe_v1", "zzz_tied_recipe_v1"], (
        "a tie on evidence count breaks on recipe_id, never on the better R"
    )
    assert 0 < len(selection) < len(pack["eligible_policies"]), (
        "the budget must actually bind, or the ordering is never exercised"
    )

    # `selected_by` is what the reader is told the basis was; it must name the
    # evidence count and no result statistic.
    basis = view["narrated"]["selected_by"].lower()
    assert "evidence count" in basis
    for forbidden in ("mean_r", "win_rate", "profit_factor", "expectancy"):
        assert forbidden not in basis

    # Shuffle every result statistic across the cells. The evidence counts, the
    # recipe ids, families and sides are untouched, so a selection that used any
    # of these numbers would move and this one may not.
    result_paths = (
        ("clipped", "mean"),
        ("clipped", "trimmed_mean"),
        ("clipped", "median"),
        ("raw", "mean"),
        ("raw", "trimmed_mean"),
        ("raw", "median"),
        ("win_rate",),
        ("stop_rate",),
        ("profit_factor", "value"),
    )
    rng = random.Random(20260905)
    for _round in range(20):
        shuffled = copy.deepcopy(pack)
        for path in result_paths:
            values = []
            for cell in shuffled["eligible_policies"]:
                node = cell["stats"]
                for part in path[:-1]:
                    node = node[part]
                values.append(node[path[-1]])
            rng.shuffle(values)
            for cell, value in zip(shuffled["eligible_policies"], values):
                node = cell["stats"]
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] = value
        rng.shuffle(shuffled["eligible_policies"])
        assert _selection(setup_research.narration_view(shuffled)) == selection


def test_the_pack_markdown_states_the_omission(tmp_path, budget, stub_model, monkeypatch):
    """The `.md` beside the pack says how much of it was narrated."""
    pack = _live_shaped_pack()
    monkeypatch.setattr(setup_research, "build_fact_pack", lambda *a, **k: pack)

    result = setup_research.run_setup_research(
        root=tmp_path, inputs=([], {}, {}, {"outcomes": 141_299})
    )
    assert result["status"] == "ok"

    markdown = next(tmp_path.rglob("*.md")).read_text(encoding="utf-8")
    match = re.search(r"Narration covers (\d+) of (\d+) eligible cells", markdown)
    assert match is not None, (
        "the pack markdown never says how much of it was narrated:\n"
        + markdown[-1500:]
    )
    kept, total = int(match.group(1)), int(match.group(2))
    assert total == LIVE_ELIGIBLE_CELLS
    assert f"{total - kept} omitted for size" in markdown

    # The number in the markdown is the number of cells that were actually sent.
    view = setup_research.narration_view(pack)
    assert kept == view["narrated"]["eligible_policy_cells"] == len(
        view["eligible_policies"]
    )


def test_narration_json_carries_narrated_block(tmp_path, budget, stub_model, monkeypatch):
    """A reader of the narration alone can see what it was written over."""
    pack = _live_shaped_pack()
    monkeypatch.setattr(setup_research, "build_fact_pack", lambda *a, **k: pack)

    setup_research.run_setup_research(
        root=tmp_path, inputs=([], {}, {}, {"outcomes": 141_299})
    )

    narration_files = list(tmp_path.rglob("*.narration.json"))
    assert len(narration_files) == 1, "one narration beside one pack (gate #40)"
    narration = json.loads(narration_files[0].read_text(encoding="utf-8"))
    assert narration["narrated"] == setup_research.narration_view(pack)["narrated"]
    assert narration["narrated"]["of"] == LIVE_ELIGIBLE_CELLS
    assert narration["narrated"]["eligible_policy_cells"] < LIVE_ELIGIBLE_CELLS


def test_refusal_when_even_one_cell_does_not_fit(budget):
    """The refusal survives, narrowed - and it names which part is too big.

    One cell and a 2,000-char budget: the head alone is already over it, so
    nothing can be narrated and the refusal is right. The message must name the
    HEAD's size, which is strictly between the budget and the whole view's size,
    so a message that only restated the total fails here.
    """
    pack = _synthetic_pack([_synthetic_cell(0, n=621, mean_r=0.5)])

    budget(10_000_000)
    whole = len(_encoded(setup_research.narration_view(pack)))

    budget(2_000)
    with pytest.raises(setup_research.NarrationTooLarge) as excinfo:
        setup_research._evidence_package(pack)
    message = str(excinfo.value)
    numbers = {int(token) for token in re.findall(r"\d+", message)}
    assert 2_000 in numbers, "the message still names the budget"
    assert 1 in numbers, "the message still names the cell count"
    assert "head" in message.lower()
    head_sizes = {value for value in numbers if 2_000 < value < whole}
    assert head_sizes, (
        f"no number in the refusal names the head's size (budget 2000, whole view "
        f"{whole}): {message}"
    )


def test_after_like_eligible_cells_still_present_when_they_fit(budget):
    """P10 C3 stands: the after-like ELIGIBLE cells stay in the view."""
    budget(60_000)
    after_like = [
        _after_like_cell(0, n_episodes=90, eligible=True),
        _after_like_cell(1, n_episodes=70, eligible=True),
        _after_like_cell(2, n_episodes=50, eligible=True),
        _after_like_cell(3, n_episodes=40, eligible=False),
    ]
    cells = [_synthetic_cell(index, n=200 - index, mean_r=0.4) for index in range(6)]
    pack = _synthetic_pack(cells, after_like_cells=after_like)

    view = setup_research.narration_view(pack)
    assert len(view["after_like_eligible"]) == 3
    assert [cell["n_episodes"] for cell in view["after_like_eligible"]] == [90, 70, 50]
    assert all(cell["eligible"] for cell in view["after_like_eligible"])

    narrated = view["narrated"]
    assert narrated["after_like_cells"] == 3
    assert narrated["of_after_like"] == 3
    assert narrated["eligible_policy_cells"] == 6
    assert narrated["of"] == 6
    assert len(_encoded(view)) <= 60_000


def test_ledger_reason_names_k_of_n(tmp_path, budget, stub_model, monkeypatch):
    """The nightly's reason line says how much of the pack was narrated."""
    pack = _live_shaped_pack()
    monkeypatch.setattr(setup_research, "build_fact_pack", lambda *a, **k: pack)

    result = setup_research.run_setup_research(
        root=tmp_path, inputs=([], {}, {}, {"outcomes": 141_299})
    )

    kept = setup_research.narration_view(pack)["narrated"]["eligible_policy_cells"]
    assert result["status"] == "ok"
    assert result["model"] == "stub-local"
    assert (
        f"narrated {kept} of {LIVE_ELIGIBLE_CELLS} eligible cell(s)" in result["reason"]
    ), result["reason"]
    assert "narration absent" not in result["reason"]
