"""``launch_gui.py --selftest``: prove a build can actually reach its engines.

The signature packaging failure in this repo is not a bundle that fails to
start. It is a bundle that starts fine, looks healthy, and then dies at the
first lazy import - because PyInstaller only sees what the import graph
reaches, and this app deliberately imports its engines inside functions so the
desk opens fast. "It launched" has therefore never been evidence of anything,
and the cost of finding out has been a five-to-ten minute click-through by the
trader on every packaging change.

This module is the automated replacement. It performs, in one pass and with no
window and no network:

* an import of every engine the desk loads lazily - the scan entry, BounceBot,
  market prep, the chart/snapshot path, the AI jobs runner, the annotation and
  capture layer, the research warehouse;
* a load of every non-``.py`` runtime asset the app reads through
  ``__file__``-relative paths, which is the class of failure a spec's data
  mirroring silently causes (the Qt stylesheet, the veto vocabulary);
* a check that the frozen path assumptions still hold.

Every check is import-and-read only. Nothing here opens a socket, contacts a
broker, writes to the home folder, or constructs a QApplication - a selftest
that needed a display would be no better than the click-through it replaces.

Exit code 0 means every check passed. Non-zero names what did not.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, Iterable


#: Modules the desk imports lazily, inside functions, long after startup.
#: Each name here is one that a spec omission would turn into a mid-session
#: ModuleNotFoundError rather than a failed launch.
LAZY_ENGINE_MODULES: tuple[str, ...] = (
    # the scan subprocess entry and its library
    "master_avwap",
    "master_avwap_lib.legacy",
    "master_avwap_lib.levels",
    # the child process the frozen desk spawns for every scan. Absent from the
    # bundle, --run-scan dies on import and the desk cannot scan at all.
    "scan_worker",
    # the intraday detector
    "bounce_bot",
    "bounce_bot_lib.legacy",
    # pre-session services
    "market_prep",
    "market_prep.config_loader",
    # the chart path (built on worker threads, imported there)
    "chart_snapshot",
    "chart_levels",
    "chart_watch",
    "ui.services.chart_data_service",
    "ui.services.bar_cache",
    "ui.services.safe_import",
    # capture + annotations
    "ui.annotations.store",
    "ui.annotations.vocabulary",
    "ui.annotations.setup_claims",
    "ui.annotations.veto_cohort",
    # NOT ai_jobs: the local AI batch layer is deliberately out of the bundle
    # (PACKAGES_NOT_IN_THE_BUNDLE in tests/test_packaging_spec_drift.py). Its
    # only entry point is scripts/run_ai_jobs.py, a scheduled CLI run from the
    # repo checkout, so the frozen exe cannot import it and must not be asked
    # to. test_selftest_modules_are_actually_bundled keeps the two in step.
    # shadow/evidence engines
    "market_state",
    "greatness_monitor",
    "research_warehouse.config",
    "diagnostics",
    # the desk itself
    "ui.app",
    "ui.theme",
    "project_paths",
)


def _check_stylesheet() -> None:
    """ui/theme.qss is read through ``__file__``; a spec miss loses it."""
    from ui import theme

    rendered = theme.build_stylesheet("dark")
    if "@" in rendered.split("\n")[0] or not rendered.strip():
        raise RuntimeError("theme.qss rendered empty or with unreplaced tokens")


def _check_veto_vocabulary() -> None:
    """The veto picklist is a bundled JSON asset, not a Python constant."""
    from ui.annotations.vocabulary import load_veto_vocabulary

    vocabulary = load_veto_vocabulary()
    if not vocabulary.reasons:
        raise RuntimeError("the veto vocabulary loaded with no reasons")


def _check_setup_claims() -> None:
    """Setup claims come from the setup-doc registry the rail reads."""
    from ui.annotations.setup_claims import all_setup_claims

    if not all_setup_claims():
        raise RuntimeError("no setup claims resolved")


def _check_frozen_path_assumptions() -> None:
    """A frozen run must never have ``<_MEIPASS>/scripts`` on sys.path.

    PyInstaller's importer claims any path under ``_MEIPASS`` even when the
    directory does not exist, so that entry makes every first-party package
    resolve to a phantom location and every submodule import fail. The bundle
    exposes ``scripts/`` contents as top-level modules, so the path is only
    ever correct for a source checkout.
    """
    if not getattr(sys, "frozen", False):
        return
    meipass = getattr(sys, "_MEIPASS", "")
    if not meipass:
        return
    from pathlib import Path

    phantom = str(Path(meipass) / "scripts")
    if any(str(entry) == phantom for entry in sys.path):
        raise RuntimeError(f"sys.path contains the frozen phantom root {phantom}")


def _check_chart_level_payload() -> None:
    """The A4 paint-line builder runs with no store present.

    Exercised rather than merely imported: the level path reaches into
    ``master_avwap_lib.levels`` lazily, which is precisely the shape of import
    a bundle drops.
    """
    from datetime import datetime, timedelta

    import chart_levels

    bars = [
        {
            "dt": datetime(2026, 6, 1) + timedelta(days=index),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
        for index in range(3)
    ]
    levels = chart_levels.build_d1_levels(
        "SELFTEST",
        bars,
        levels_dir="/nonexistent-selftest",
        ai_state_path="/nonexistent-selftest/ai_state.json",
    )
    if not any(level["group"] == chart_levels.GROUP_PREV_DAY for level in levels):
        raise RuntimeError("prev-day levels did not build from bars alone")


def _check_testing_plan() -> None:
    """Settings > Testing Plan renders a markdown file from OUTSIDE scripts/.

    That places it beyond the spec's package-asset sweep and beyond the drift
    test, which only walks scripts/ - so an explicit `datas` rule is the only
    thing bundling it, and this is the only check that would notice the rule
    being lost. A frozen desk missing it shows "plan file not found" on the one
    page the trader opens when nothing else is behaving.
    """
    from ui.widgets.testing_plan_view import resolve_testing_plan_path

    path = resolve_testing_plan_path()
    if not path.is_file():
        raise RuntimeError(f"the testing plan is not in this build ({path})")
    if len(path.read_text(encoding="utf-8").strip()) < 500:
        raise RuntimeError(f"the testing plan bundled empty or truncated ({path})")


#: (name, callable) - asset loads and behavioural probes, run after imports.
ASSET_CHECKS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("ui/theme.qss", _check_stylesheet),
    ("ui/annotations/vocabularies/veto_reasons_v*.json", _check_veto_vocabulary),
    ("docs/DESK_TESTING_PLAN.md", _check_testing_plan),
    ("setup claim registry", _check_setup_claims),
    ("frozen sys.path assumptions", _check_frozen_path_assumptions),
    ("chart level payload", _check_chart_level_payload),
)


def run_selftest(
    *,
    modules: Iterable[str] = LAZY_ENGINE_MODULES,
    checks: Iterable[tuple[str, Callable[[], None]]] = ASSET_CHECKS,
    stream=None,
    verbose: bool = False,
) -> int:
    """Run every check. Returns 0 when all pass, 1 otherwise.

    Every failure is collected and reported: stopping at the first one would
    mean a second four-minute rebuild to discover the second problem.
    """
    out = stream if stream is not None else sys.stdout
    failures: list[tuple[str, str]] = []
    passed = 0

    from importlib import import_module

    for name in modules:
        try:
            import_module(name)
        except Exception:
            failures.append((f"import {name}", traceback.format_exc()))
        else:
            passed += 1
            if verbose:
                print(f"  ok  import {name}", file=out)

    for name, check in checks:
        try:
            check()
        except Exception:
            failures.append((f"check {name}", traceback.format_exc()))
        else:
            passed += 1
            if verbose:
                print(f"  ok  check {name}", file=out)

    total = passed + len(failures)
    if failures:
        print(f"selftest FAILED: {len(failures)} of {total} checks", file=out)
        for name, detail in failures:
            print(f"\n--- {name} ---\n{detail}", file=out)
        return 1
    frozen = " (frozen)" if getattr(sys, "frozen", False) else ""
    print(f"selftest OK: {total}/{total} checks passed{frozen}", file=out)
    return 0
