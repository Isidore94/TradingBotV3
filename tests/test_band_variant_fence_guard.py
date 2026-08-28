"""Phase 0.10 review fix 2 - the shadow fence stops being a hand-maintained list.

`_is_band_variant_scenario` keeps the band-variant challenger out of every
champion aggregate. Seven readers filter on it, and **three of those seven were
found by the parity fixture rather than by reading the code** - which is the
whole argument for this file: an eighth reader will not be found by reading
either.

So the rule is checked structurally, in the shape of
`test_shutdown_waits_are_bounded.py`: every place in `legacy.py` that iterates a
setup's scenarios either mentions `_is_band_variant_scenario` somewhere in its
enclosing function, or is named in `ALLOWED_UNFENCED` below with the reason it
MUST see the shadow.

The detector is deliberately wider than the `(setup.get("scenarios") or
{}).values()` spelling the fence was written against: `setup["scenarios"]
.values()`, `setup.get("scenarios", {}).values()` and a local
`working_scenarios.values()` all count. A guard that only recognizes today's
spelling would be passed by tomorrow's.

What this does NOT claim: that mentioning the helper means it was used
correctly. It cannot - a name in a function is not a proof about its logic. It
claims the narrower and still useful thing, that no scenario reader was written
without the author meeting the fence. The parity fixture
(`test_tracker_band_variant_parity.py`) is what proves the values did not move.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

LEGACY = SCRIPTS_DIR / "master_avwap_lib" / "legacy.py"
FENCE = "_is_band_variant_scenario"

#: Readers that MUST see the challenger's scenarios, with the reason. A new
#: entry here is a deliberate statement that the shadow belongs in that path -
#: never a way to quiet the test.
ALLOWED_UNFENCED: dict[str, str] = {
    "_extract_tracker_stop_candidates_from_setup": (
        "Rebuilds the stop-candidate list from an existing record on replay. It "
        "must carry the VARIANT_* candidate forward, or a rebuilt record would "
        "silently lose its shadow and stop accruing. It sorts by (label, "
        "source_type), and 'VARIANT_...' sorts after every champion label, so "
        "the append-last ordering survives the rebuild."
    ),
    "_compact_tracker_setup_record": (
        "Sealed-record compaction strips each scenario's per-bar `events` log. "
        "The shadow's log is exactly as disposable as the champion's, and "
        "skipping it would leave the challenger as the only thing keeping a "
        "sealed record large."
    ),
}


def _mentions_scenarios(node: ast.AST) -> bool:
    """True when this expression plausibly resolves a setup's scenarios dict."""
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value == "scenarios"
        ):
            return True
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Constant)
            and child.slice.value == "scenarios"
        ):
            return True
        if isinstance(child, ast.Name) and "scenario" in child.id.lower():
            return True
        if isinstance(child, ast.Attribute) and "scenario" in child.attr.lower():
            return True
    return False


def _is_scenario_iteration(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "values"
        and _mentions_scenarios(node.func.value)
    )


def _functions_with_scenario_reads() -> dict[str, list[int]]:
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"))
    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sites = [child.lineno for child in ast.walk(node) if _is_scenario_iteration(child)]
        if sites:
            found[node.name] = sorted(sites)
    return found


def _module_level_scenario_reads() -> list[int]:
    """A read outside any function would be invisible to the check above."""
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"))
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside.update(
                child.lineno for child in ast.walk(node) if _is_scenario_iteration(child)
            )
    return sorted(
        child.lineno
        for child in ast.walk(tree)
        if _is_scenario_iteration(child) and child.lineno not in inside
    )


def _fenced_functions() -> set[str]:
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Name) and child.id == FENCE for child in ast.walk(node)
        ):
            out.add(node.name)
    return out


def test_every_scenario_reader_is_fenced_or_documented():
    readers = _functions_with_scenario_reads()
    fenced = _fenced_functions()
    unfenced = {
        name: lines
        for name, lines in readers.items()
        if name not in fenced and name not in ALLOWED_UNFENCED
    }
    assert not unfenced, (
        "legacy.py gained a scenario reader that neither filters on "
        f"{FENCE} nor is documented in ALLOWED_UNFENCED: "
        + ", ".join(f"{name} (line{'s' if len(lines) > 1 else ''} {lines})" for name, lines in sorted(unfenced.items()))
        + ". If it must see the challenger's scenarios, add it to "
        "ALLOWED_UNFENCED with the reason; otherwise fence it, or the shadow "
        "will move a champion number the way it moved eight of them before the "
        "fence existed."
    )


def test_the_guard_actually_sees_the_known_readers():
    """A guard that finds nothing passes forever. Pin what it currently finds."""
    readers = _functions_with_scenario_reads()
    # The nine readers as of the Phase 0.10 review. This list may grow or
    # shrink with the code; it exists so a detector that stops matching (a
    # refactor to a helper, a renamed local) fails loudly instead of quietly
    # approving everything.
    assert len(readers) >= 8, f"the detector matched only {sorted(readers)}"
    for expected in (
        "_summarize_tracker_setup_outcome",
        "_flatten_tracker_scenarios",
        "_extract_tracker_stop_candidates_from_setup",
        "_compact_tracker_setup_record",
    ):
        assert expected in readers, f"{expected} is no longer detected as a scenario reader"


def test_the_allowlist_names_only_real_functions():
    """A stale allowlist entry is an invisible exemption for a name that moved."""
    readers = _functions_with_scenario_reads()
    stale = [name for name in ALLOWED_UNFENCED if name not in readers]
    assert not stale, f"ALLOWED_UNFENCED names functions that no longer read scenarios: {stale}"


def test_the_allowlist_carries_a_reason_for_every_entry():
    for name, reason in ALLOWED_UNFENCED.items():
        assert len(reason.split()) >= 12, f"{name}'s allowlist reason is too thin to review"


def test_no_scenario_read_happens_outside_a_function():
    assert _module_level_scenario_reads() == []


def test_the_fence_helper_still_exists_under_that_name():
    """The guard is keyed on a name; if the name moves, the guard is a no-op."""
    from master_avwap_lib import legacy

    assert callable(getattr(legacy, FENCE))
    assert legacy._is_band_variant_scenario(
        {"stop_source_type": legacy.BAND_VARIANT_STOP_SOURCE}
    )
    assert not legacy._is_band_variant_scenario({"stop_source_type": "current_anchor"})
