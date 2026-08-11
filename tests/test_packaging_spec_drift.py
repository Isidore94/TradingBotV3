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

The spec is *executed* rather than parsed, with PyInstaller's collectors stubbed
out. Parsing would assert what the file looks like; executing asserts what it
actually does, survives any refactor that keeps the behaviour, and still costs
milliseconds instead of walking sklearn.

**Fix the spec, never this test.** A package that genuinely must not ship, or a
file that is a development helper rather than a runtime asset, belongs in the
explicit allowlist below with the reason written down - which is a decision, not
a weakening.

Merge note (2026-08-09): this file is the reconciliation of two independently
written guards - ``testing-week-2026-08-10``'s and the A4 stream's. Both suites'
assertions are kept, so a few properties are checked twice by design, from
different angles: the package census is checked against the spec's
``FIRST_PARTY_PACKAGES`` tuple *and* against the packages ``collect_submodules``
was actually called for, and the asset sweep is checked with exact-path matching
*and* with tree-covering matching. Each pair of allowlists turned out to name
the same files, so they are one mapping under both suites' names.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "tradingbotv3.spec"
SPEC_PATH = SPEC
SCRIPTS = ROOT / "scripts"


# Packages under scripts/ that are deliberately NOT in the frozen desk. Each
# needs a reason, so leaving one out stays a decision instead of an oversight.
#
# The bar for an entry here is not "we think nothing imports it" - that belief is
# exactly what ships a broken bundle. It is that nothing reachable from
# launch_gui.py, the frozen entry point, can import it. All four below were
# re-verified against the tree at the 2026-08-09 merge, and the frozen
# --selftest exercises the lazy engines that would expose a wrong call here.
PACKAGES_NOT_IN_THE_BUNDLE = {
    "ai_jobs": (
        "local-AI batch layer (plan.md 13b). Its only entry point is "
        "scripts/run_ai_jobs.py, a scheduled CLI run from the repo checkout; "
        "launch_gui.py never reaches it."
    ),
    "gui_app": (
        "legacy Tk UI, imported only by scripts/gui.py --ui tk. launch_gui.py "
        "enters the Qt desk directly and never hops through it."
    ),
    "market_prep_gui": (
        "legacy Tk market-prep UI, imported only by scripts/market_prep_tab.py, "
        "which in turn is imported only by gui_app - so it is reachable from the "
        "Tk entry point alone, not the frozen Qt desk."
    ),
    "indicators": "no importer anywhere in the tree; nothing can reach it, frozen or not.",
}
#: The A4 suite's name for the same allowlist.
UNCOLLECTED_PACKAGES = PACKAGES_NOT_IN_THE_BUNDLE

# Non-.py files under scripts/ that are not runtime assets of the exe. These are
# run on the desk to register Windows scheduled tasks; the frozen app never reads
# them, and shipping them would put task-registration scripts inside a
# distributed bundle.
ASSETS_NOT_IN_THE_BUNDLE = {
    "scripts/launch_gui_auto.ps1": "operator script, run from the repo by Task Scheduler.",
    "scripts/register_0700_autostart.ps1": "operator script, run by hand to register the task.",
    "scripts/register_ai_jobs_task.ps1": "operator script, run by hand to register the AI jobs task.",
    "scripts/run_ai_jobs.ps1": (
        "scheduled-task wrapper: Task Scheduler runs it from the repo checkout, and it "
        "invokes the repo venv's python.exe. The frozen exe never reads it - it is not "
        "reachable from launch_gui.py at all."
    ),
}
#: The A4 suite's name for the same allowlist.
UNBUNDLED_ASSETS = ASSETS_NOT_IN_THE_BUNDLE


def _stub_hooks_module(collected: list[str]):
    """Stand in for PyInstaller.utils.hooks.

    The real collectors import every package they walk, which is what makes a
    genuine spec run slow. The spec only ever uses their results as opaque
    lists, so returning a marker per package preserves its control flow -
    including the ``if not found: raise SystemExit`` guard - at no cost. The
    package names are recorded so the census can assert what was really asked
    for rather than what a tuple in the spec claims.
    """

    def collect_submodules(package, **kwargs):
        collected.append(str(package))
        return [str(package), f"{package}.__stub__"]

    module = types.ModuleType("PyInstaller.utils.hooks")
    module.collect_submodules = collect_submodules
    module.collect_data_files = lambda package, **kw: []
    # duckdb is collected through collect_all when it is installed (LD-04), so
    # the stub has to offer it too or the spec's import line fails outright.
    module.collect_all = lambda package, **kw: ([], [], [package])
    return module


def _execute_spec() -> dict:
    """Execute the spec with the PyInstaller build API stubbed out."""
    captured: dict = {}
    collected: list[str] = []

    class _Analysis:
        """Stands in for Analysis, keeping its kwargs for inspection."""

        def __init__(self, scripts, **kwargs):
            self.args = (scripts,)
            self.kwargs = kwargs
            captured["scripts"] = scripts
            captured["kwargs"] = kwargs
            self.pure, self.scripts, self.binaries, self.datas = [], [], [], []

    class _Passthrough:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    hooks = _stub_hooks_module(collected)
    utils = types.ModuleType("PyInstaller.utils")
    utils.hooks = hooks
    pyinstaller = types.ModuleType("PyInstaller")
    pyinstaller.utils = utils

    saved_modules = {
        name: sys.modules.get(name)
        for name in ("PyInstaller", "PyInstaller.utils", "PyInstaller.utils.hooks")
    }
    sys.modules["PyInstaller"] = pyinstaller
    sys.modules["PyInstaller.utils"] = utils
    sys.modules["PyInstaller.utils.hooks"] = hooks

    namespace: dict = {
        "__file__": str(SPEC),
        "SPECPATH": str(SPEC.parent),
        "DISTPATH": str(ROOT / "dist"),
        "workpath": str(ROOT / "build"),
        "Analysis": _Analysis,
        "PYZ": _Passthrough,
        "EXE": _Passthrough,
        "COLLECT": _Passthrough,
    }
    # The spec prepends the repo roots to sys.path so collect_submodules can
    # import them; leaving that behind would leak into the rest of the suite.
    saved_path = list(sys.path)
    try:
        exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), namespace)
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    namespace["_collected_packages"] = collected
    captured["namespace"] = namespace
    captured["collected"] = collected
    return captured


@pytest.fixture(scope="module")
def spec_result() -> dict:
    """The executed spec, as ``{scripts, kwargs, namespace, collected}``."""
    return _execute_spec()


@pytest.fixture(scope="module")
def spec(spec_result) -> dict:
    """The executed spec's globals - the A4 suite's view of the same run."""
    return spec_result["namespace"]


def _discovered_packages() -> set[str]:
    return {path.parent.name for path in SCRIPTS.glob("*/__init__.py")}


_script_packages = _discovered_packages


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


def test_every_scripts_package_is_collected(spec):
    """The same census, against the calls the spec actually made.

    ``FIRST_PARTY_PACKAGES`` is a tuple; this asserts ``collect_submodules`` was
    really invoked for each of its members, so a spec that declares a package and
    then forgets to iterate it still fails.

    If this fails: add the package to ``FIRST_PARTY_PACKAGES`` in
    ``packaging/tradingbotv3.spec``. Do not add it to the allowlist unless you
    can state why nothing in a frozen run can reach it.
    """
    collected = set(spec["_collected_packages"])
    missing = _script_packages() - collected - set(UNCOLLECTED_PACKAGES)
    assert not missing, (
        "packages under scripts/ missing from the spec's collect_submodules "
        f"list: {sorted(missing)}"
    )


def test_the_uncollected_allowlist_only_names_real_packages():
    """A stale allowlist entry is an exemption nobody is watching."""
    unknown = set(UNCOLLECTED_PACKAGES) - _script_packages()
    assert not unknown, f"allowlisted packages that no longer exist: {sorted(unknown)}"


def test_the_selftest_never_demands_a_package_the_bundle_excludes():
    """The two guards must not contradict each other.

    ``scripts/selftest.py`` asserts that the frozen desk CAN import each module
    in ``LAZY_ENGINE_MODULES``; ``PACKAGES_NOT_IN_THE_BUNDLE`` asserts that the
    frozen desk does NOT ship certain packages. A name in both is unsatisfiable
    by construction, and - this is the part that hurts - it is invisible to the
    whole unfrozen suite, because a repo checkout can always import anything
    under ``scripts/``. Only a real frozen build can collide them, and a frozen
    build is the one thing nobody runs per commit.

    That is exactly what happened on 2026-08-09: ``ai_jobs`` sat in both lists,
    the unfrozen selftest passed 30/30 all week, and the desk's frozen run was
    the first execution anywhere to fail. This test moves that discovery from a
    four-minute rebuild to the normal suite.

    If this fails, decide which side is right and fix THAT side - do not delete
    the assertion. The question to answer is whether a frozen run can reach the
    package. If it cannot, drop it from ``LAZY_ENGINE_MODULES``; if it can, move
    it out of the exclusion list and into ``FIRST_PARTY_PACKAGES``.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import selftest

    # LAZY_ENGINE_MODULES holds dotted module paths ("ui.services.bar_cache");
    # the exclusion list holds top-level package names ("ai_jobs"). Compare on
    # the root so "ai_jobs.briefs" is caught as readily as "ai_jobs".
    demanded_roots = {name.split(".")[0] for name in selftest.LAZY_ENGINE_MODULES}
    contradictions = demanded_roots & set(PACKAGES_NOT_IN_THE_BUNDLE)
    assert not contradictions, (
        "scripts/selftest.py requires the frozen exe to import packages that "
        f"PACKAGES_NOT_IN_THE_BUNDLE deliberately keeps out of it: {sorted(contradictions)}. "
        "These two lists cannot both be satisfied - the frozen --selftest will fail. "
        "Reasons the packages are excluded: "
        + "; ".join(f"{p}: {PACKAGES_NOT_IN_THE_BUNDLE[p]}" for p in sorted(contradictions))
    )


def test_the_spec_still_executes(spec):
    """A spec that cannot be evaluated cannot be checked - or built."""
    assert spec["_collected_packages"], "no collect_submodules calls found"
    assert spec["datas"], "the spec bundles no data files at all"


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


def _covered_by(datas: list, path: Path) -> bool:
    """Does any ``(src, dest)`` rule bundle ``path``, directly or as a tree?"""
    for source, _dest in datas:
        candidate = Path(source)
        if candidate == path:
            return True
        if candidate.is_dir() and candidate in path.parents:
            return True
    return False


def _runtime_assets() -> list[Path]:
    return sorted(
        entry
        for entry in SCRIPTS.rglob("*")
        if entry.is_file()
        and entry.suffix.lower() not in (".py", ".pyc")
        and "__pycache__" not in entry.parts
    )


def test_every_runtime_asset_under_scripts_is_bundled_by_a_datas_rule(spec):
    """The same sweep, but honouring whole-directory ``datas`` rules.

    The exact-path version above cannot see a file bundled because an ancestor
    directory was added as a tree; this one can, so between them a rule of either
    shape counts and neither shape can hide a gap.

    If this fails: extend the spec's ``datas`` rules to cover the new file. Add
    it to UNBUNDLED_ASSETS only when the frozen app provably never reads it.
    """
    datas = list(spec["datas"])
    missing = [
        str(asset.relative_to(ROOT)).replace("\\", "/")
        for asset in _runtime_assets()
        if not _covered_by(datas, asset)
    ]
    assert not [name for name in missing if name not in UNBUNDLED_ASSETS], (
        "non-.py files under scripts/ that the spec does not bundle: "
        f"{[name for name in missing if name not in UNBUNDLED_ASSETS]}"
    )


def test_the_unbundled_allowlist_only_names_real_files():
    for relative in UNBUNDLED_ASSETS:
        assert (ROOT / relative).is_file(), (
            f"UNBUNDLED_ASSETS names {relative}, which no longer exists"
        )


def test_the_assets_that_are_read_through_file_relative_paths_are_bundled(spec_result):
    """The three the desk reads by __file__-relative path, named individually.

    The sweeps above would catch these too, but only while they sit under a
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


def test_the_known_asset_loads_are_covered(spec):
    """The two oldest ``__file__``-relative loads, via tree-aware matching."""
    datas = list(spec["datas"])
    for relative in (
        "scripts/ui/theme.qss",
        "scripts/ui/annotations/vocabularies/veto_reasons_v1.json",
    ):
        assert _covered_by(datas, ROOT / relative), f"{relative} is not bundled"


def test_config_dir_is_bundled_where_market_prep_looks_for_it(spec_result):
    """market_prep/config_loader.py resolves Path(__file__).parents[1] / 'config',
    which is the bundle root when frozen."""
    destinations = {dest for _source, dest in spec_result["kwargs"]["datas"]}
    assert "config" in destinations


def test_the_root_config_tree_is_bundled(spec):
    """The same requirement, pinning the source side as well as the destination."""
    destinations = {str(dest) for _source, dest in spec["datas"]}
    sources = {str(source) for source, _dest in spec["datas"]}
    assert str(ROOT / "config") in sources
    assert "config" in destinations


def test_pyqt5_is_excluded_so_one_qt_binding_wins(spec_result):
    """Two bindings in one process is a crash, and qtpy picks whichever it finds."""
    excludes = set(spec_result["kwargs"]["excludes"])
    assert {"PyQt5", "PyQt6", "PySide2"} <= excludes
    hook = Path(spec_result["kwargs"]["runtime_hooks"][0])
    assert hook.name == "rthook_qt_api.py" and hook.exists()


def test_pyqt5_stays_out_of_the_bundle(spec):
    """The same exclusion, read off the Analysis object the spec bound to ``a``."""
    excludes = spec["a"].kwargs["excludes"]
    assert "PyQt5" in excludes


def test_the_entry_point_is_launch_gui(spec_result):
    assert [Path(p).name for p in spec_result["scripts"]] == ["launch_gui.py"]


def test_the_selftest_entrypoint_is_reachable():
    """--selftest is what replaces the trader's post-build click-through.

    It only does that if the frozen entry script actually routes to it, so
    pin the wiring rather than trusting the flag's presence in a docstring.
    """
    source = (ROOT / "launch_gui.py").read_text(encoding="utf-8")
    assert "--selftest" in source
    assert "run_selftest" in source
