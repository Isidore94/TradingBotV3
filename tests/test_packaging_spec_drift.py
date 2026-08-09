"""The PyInstaller spec must not fall behind the tree it packages.

CLAUDE.md's frozen-exe policy lists five change kinds that can break the bundle.
Three of them are drift between the repo and ``packaging/tradingbotv3.spec``:

  2. a new non-``.py`` runtime asset (the spec has to mirror it explicitly),
  3. a new top-level package under ``scripts/`` that is imported lazily (the
     ``collect_submodules`` list is hardcoded),
  4. a new dynamic import by name inside a package the spec never collects.

All three share a failure signature that a build cannot catch: the bundle
assembles, launches, looks healthy, and dies at the first lazy import or the
first missing asset -- possibly weeks later, on the desk, mid-session. The only
cheap guard is to check the spec against the tree, which is what this does.

It *executes* the spec rather than parsing it, so it asserts what the spec does
rather than what it looks like, but stubs PyInstaller's collectors so the check
costs milliseconds instead of walking sklearn.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "tradingbotv3.spec"
SCRIPTS = ROOT / "scripts"


# Packages under scripts/ that are deliberately NOT in the frozen desk. Each
# needs a reason, so leaving one out stays a decision instead of an oversight.
PACKAGES_NOT_IN_THE_BUNDLE = {
    "ai_jobs": (
        "local-AI batch layer (plan.md 13b). Its only entry point is "
        "scripts/run_ai_jobs.py, a scheduled CLI run from the repo checkout; "
        "launch_gui.py never reaches it."
    ),
    "gui_app": "legacy Tk UI (scripts/gui.py --ui tk), not the frozen Qt desk.",
    "market_prep_gui": "legacy Tk market-prep UI, not the frozen Qt desk.",
    "indicators": "no importer anywhere in the tree; nothing can reach it, frozen or not.",
}

# Non-.py files under scripts/ that are not runtime assets of the exe.
ASSETS_NOT_IN_THE_BUNDLE = {
    "scripts/launch_gui_auto.ps1": "operator script, run from the repo by Task Scheduler.",
    "scripts/register_0700_autostart.ps1": "operator script, run by hand to register the task.",
    "scripts/register_ai_jobs_task.ps1": "operator script, run by hand to register the AI jobs task.",
}


def _stub_hooks_module():
    """Stand in for PyInstaller.utils.hooks.

    The real collectors import every package they walk, which is what makes a
    genuine spec run slow. The spec only ever uses their results as opaque
    lists, so returning a marker per package preserves its control flow -
    including the ``if not found: raise SystemExit`` guard - at no cost.
    """
    module = types.ModuleType("PyInstaller.utils.hooks")
    module.collect_submodules = lambda package, **kw: [package, f"{package}.__stub__"]
    module.collect_data_files = lambda package, **kw: []
    module.collect_all = lambda package, **kw: ([], [], [package])
    return module


@pytest.fixture(scope="module")
def spec_result():
    """Execute the spec with the PyInstaller build API stubbed out."""
    captured: dict = {}

    class _Analysis:
        def __init__(self, scripts, **kwargs):
            captured["scripts"] = scripts
            captured["kwargs"] = kwargs
            self.pure, self.scripts, self.binaries, self.datas = [], [], [], []

    class _Passthrough:
        def __init__(self, *args, **kwargs):
            pass

    real_hooks = sys.modules.get("PyInstaller.utils.hooks")
    sys.modules["PyInstaller.utils.hooks"] = _stub_hooks_module()
    try:
        namespace = {
            "__file__": str(SPEC),
            "SPECPATH": str(SPEC.parent),
            "DISTPATH": str(ROOT / "dist"),
            "workpath": str(ROOT / "build"),
            "Analysis": _Analysis,
            "PYZ": _Passthrough,
            "EXE": _Passthrough,
            "COLLECT": _Passthrough,
        }
        exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), namespace)
    finally:
        if real_hooks is None:
            sys.modules.pop("PyInstaller.utils.hooks", None)
        else:
            sys.modules["PyInstaller.utils.hooks"] = real_hooks

    captured["namespace"] = namespace
    return captured


def _discovered_packages() -> set[str]:
    return {path.parent.name for path in SCRIPTS.glob("*/__init__.py")}


def _bundled_sources(spec_result) -> set[Path]:
    return {Path(source).resolve() for source, _dest in spec_result["kwargs"]["datas"]}


def test_every_package_under_scripts_is_collected_or_explicitly_excluded(spec_result):
    collected = set(spec_result["namespace"]["FIRST_PARTY_PACKAGES"])
    unaccounted = _discovered_packages() - collected - set(PACKAGES_NOT_IN_THE_BUNDLE)
    assert not unaccounted, (
        f"packages under scripts/ that the spec neither collects nor excludes: {sorted(unaccounted)}. "
        "Add them to FIRST_PARTY_PACKAGES in packaging/tradingbotv3.spec, or to "
        "PACKAGES_NOT_IN_THE_BUNDLE here with the reason they stay out. A package that is in "
        "neither list ships as a bundle that dies at its first lazy import."
    )


def test_the_exclusion_list_does_not_outlive_the_packages_it_names():
    """A stale reason is worse than none: it implies a decision nobody made."""
    gone = set(PACKAGES_NOT_IN_THE_BUNDLE) - _discovered_packages()
    assert not gone, f"PACKAGES_NOT_IN_THE_BUNDLE names packages that no longer exist: {sorted(gone)}"


def test_every_runtime_asset_under_scripts_is_bundled(spec_result):
    """Trigger 2: PyInstaller ships .py only; everything else needs a datas rule."""
    bundled = _bundled_sources(spec_result)
    excluded = {(ROOT / rel).resolve() for rel in ASSETS_NOT_IN_THE_BUNDLE}
    missing = sorted(
        str(path.relative_to(ROOT))
        for path in SCRIPTS.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in (".py", ".pyc")
        and "__pycache__" not in path.parts
        and path.resolve() not in bundled
        and path.resolve() not in excluded
    )
    assert not missing, (
        f"non-.py files under scripts/ that the spec does not bundle: {missing}. "
        "The spec mirrors assets for the packages in FIRST_PARTY_PACKAGES; a file outside those "
        "needs its own datas entry, or an ASSETS_NOT_IN_THE_BUNDLE entry saying why the exe never "
        "reads it."
    )


def test_the_assets_that_are_read_through_file_relative_paths_are_bundled(spec_result):
    """The three the desk reads by __file__-relative path, named individually.

    The sweep above would catch these too, but only while they sit under a
    collected package. Naming them pins the actual requirement: these files
    have to be in the bundle, wherever they end up living.
    """
    bundled = _bundled_sources(spec_result)
    for relative in (
        "scripts/ui/theme.qss",
        "scripts/ui/annotations/vocabularies/veto_reasons_v1.json",
        "scripts/research_warehouse/exploration_cohort.txt",
    ):
        path = (ROOT / relative).resolve()
        assert path.exists(), f"{relative} has moved; update this test and the spec together"
        assert path in bundled, f"{relative} is read at runtime but is not in the spec's datas"


def test_config_dir_is_bundled_where_market_prep_looks_for_it(spec_result):
    """market_prep/config_loader.py resolves Path(__file__).parents[1] / 'config',
    which is the bundle root when frozen."""
    destinations = {dest for _source, dest in spec_result["kwargs"]["datas"]}
    assert "config" in destinations


def test_pyqt5_is_excluded_so_one_qt_binding_wins(spec_result):
    """Two bindings in one process is a crash, and qtpy picks whichever it finds."""
    excludes = set(spec_result["kwargs"]["excludes"])
    assert {"PyQt5", "PyQt6", "PySide2"} <= excludes
    hook = Path(spec_result["kwargs"]["runtime_hooks"][0])
    assert hook.name == "rthook_qt_api.py" and hook.exists()


def test_the_entry_point_is_launch_gui(spec_result):
    assert [Path(p).name for p in spec_result["scripts"]] == ["launch_gui.py"]
