"""The PyInstaller spec must keep up with the tree it is bundling.

The spec is a hand-maintained list of packages and asset rules, and the repo
grows packages faster than anyone remembers to edit it. The failure that
causes is the nastiest one in the project: a bundle that starts, looks
healthy, and dies at the first lazy import weeks later - because PyInstaller
only ships what the import graph reaches, and this app imports its engines
inside functions on purpose.

So this test asserts the two properties a human reviewer cannot hold in their
head:

1. every top-level package under ``scripts/`` is named in the spec's
   ``collect_submodules`` calls;
2. every non-``.py`` file the app reads at runtime is covered by one of the
   spec's ``datas`` rules.

**Fix the spec, never this test.** A package that genuinely must not ship, or
a file that is a development helper rather than a runtime asset, belongs in
the explicit allowlist below with the reason written down - which is a
decision, not a weakening.

The spec is EXECUTED rather than parsed. Parsing would assert what the file
looks like; executing it with the PyInstaller API stubbed out asserts what it
actually does, and survives any refactor that keeps the behaviour.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "packaging" / "tradingbotv3.spec"
SCRIPTS = ROOT / "scripts"

#: Non-``.py`` files under ``scripts/`` that are deliberately NOT bundled,
#: with the reason. These are run by hand on the desk to register Windows
#: scheduled tasks; the frozen app never reads them, and shipping them would
#: put task-registration scripts inside a distributed bundle.
UNBUNDLED_ASSETS: dict[str, str] = {
    "scripts/launch_gui_auto.ps1": "operator script, registers the 06:00 task",
    "scripts/register_0700_autostart.ps1": "operator script, run manually",
    "scripts/register_ai_jobs_task.ps1": "operator script, run manually",
}

#: Top-level packages under ``scripts/`` that are deliberately NOT collected.
#: Empty on purpose: a package in the tree is a package something can import,
#: and "we think nothing imports it" is exactly the belief that ships a broken
#: bundle. Retired-but-present code (desk_link) still has live importers.
UNCOLLECTED_PACKAGES: dict[str, str] = {}


def _load_spec() -> dict:
    """Execute the spec with the PyInstaller API stubbed; return its globals.

    The stubs record rather than build: ``collect_submodules`` returns a
    non-empty list (the spec fails the build on an empty one, correctly) and
    remembers which package it was asked for.
    """
    collected: list[str] = []

    def collect_submodules(package, *args, **kwargs):
        collected.append(str(package))
        return [f"{package}._stub"]

    def collect_data_files(package, *args, **kwargs):
        return []

    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_submodules = collect_submodules
    hooks.collect_data_files = collect_data_files
    utils = types.ModuleType("PyInstaller.utils")
    utils.hooks = hooks
    pyinstaller = types.ModuleType("PyInstaller")
    pyinstaller.utils = utils

    saved = {
        name: sys.modules.get(name)
        for name in ("PyInstaller", "PyInstaller.utils", "PyInstaller.utils.hooks")
    }
    sys.modules["PyInstaller"] = pyinstaller
    sys.modules["PyInstaller.utils"] = utils
    sys.modules["PyInstaller.utils.hooks"] = hooks

    class _Recorder:
        """Stands in for Analysis/PYZ/EXE/COLLECT, keeping their kwargs."""

        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        # Analysis exposes these to the EXE/COLLECT calls below it.
        pure = ()
        scripts = ()
        binaries = ()
        datas = ()

    namespace: dict = {
        "SPECPATH": str(SPEC_PATH.parent),
        "Analysis": _Recorder,
        "PYZ": _Recorder,
        "EXE": _Recorder,
        "COLLECT": _Recorder,
        "__file__": str(SPEC_PATH),
    }
    saved_path = list(sys.path)
    try:
        exec(compile(SPEC_PATH.read_text(encoding="utf-8"), str(SPEC_PATH), "exec"), namespace)
    finally:
        sys.path[:] = saved_path
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    namespace["_collected_packages"] = collected
    return namespace


@pytest.fixture(scope="module")
def spec() -> dict:
    return _load_spec()


def _script_packages() -> set[str]:
    return {
        entry.name
        for entry in SCRIPTS.iterdir()
        if entry.is_dir() and (entry / "__init__.py").is_file()
    }


def test_the_spec_still_executes(spec):
    """A spec that cannot be evaluated cannot be checked - or built."""
    assert spec["_collected_packages"], "no collect_submodules calls found"
    assert spec["datas"], "the spec bundles no data files at all"


def test_every_scripts_package_is_collected(spec):
    """A package the spec does not name ships absent and fails lazily.

    If this fails: add the package to the ``for package in (...)`` tuple in
    ``packaging/tradingbotv3.spec``. Do not add it to UNCOLLECTED_PACKAGES
    unless you can state why nothing in a frozen run can reach it.
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


def test_every_runtime_asset_under_scripts_is_bundled(spec):
    """A missing asset is a bundle that starts and then cannot read its own data.

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


def test_the_known_asset_loads_are_covered(spec):
    """The two assets the app reads through ``__file__`` at runtime."""
    datas = list(spec["datas"])
    for relative in (
        "scripts/ui/theme.qss",
        "scripts/ui/annotations/vocabularies/veto_reasons_v1.json",
    ):
        assert _covered_by(datas, ROOT / relative), f"{relative} is not bundled"


def test_the_root_config_tree_is_bundled(spec):
    """market_prep.config_loader resolves CONFIG_DIR at the bundle root."""
    destinations = {str(dest) for _source, dest in spec["datas"]}
    sources = {str(source) for source, _dest in spec["datas"]}
    assert str(ROOT / "config") in sources
    assert "config" in destinations


def test_pyqt5_stays_out_of_the_bundle(spec):
    """Two Qt bindings in one process is a crash, and qtpy picks either."""
    excludes = spec["a"].kwargs["excludes"]
    assert "PyQt5" in excludes


def test_the_selftest_entrypoint_is_reachable():
    """--selftest is what replaces the trader's post-build click-through.

    It only does that if the frozen entry script actually routes to it, so
    pin the wiring rather than trusting the flag's presence in a docstring.
    """
    source = (ROOT / "launch_gui.py").read_text(encoding="utf-8")
    assert "--selftest" in source
    assert "run_selftest" in source
