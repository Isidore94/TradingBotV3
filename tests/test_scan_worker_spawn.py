"""The scan child must actually start, in both builds.

Written after the frozen desk spent two sessions unable to scan (2026-08-12 and
2026-08-13). ``_run_master_scan_subprocess`` spawned ``sys.executable -c
"<code>"``; under PyInstaller ``sys.executable`` is ``TradingBotV3.exe``, which
parsed ``-c`` as its own CLI and exited 2 one second after every slot fired.

The packaging guards could not see it: ``test_packaging_spec_drift`` checks the
bundle's *contents* and ``--selftest`` checks that lazy imports resolve. Neither
spawns anything. So the load-bearing test here is the one that really launches a
child process and waits for the marker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _payload(**kwargs) -> str:
    spec = {"update_setup_tracker": None}
    spec.update(kwargs)
    return json.dumps(spec, sort_keys=True)


# ---------------------------------------------------------------------------
# The payload contract
# ---------------------------------------------------------------------------
def test_payload_keeps_none_distinct_from_false():
    """None selects the entry point; False is 'scan, but do not write'."""
    import scan_worker

    assert scan_worker.parse_payload(_payload())["update_setup_tracker"] is None
    assert scan_worker.parse_payload(_payload(update_setup_tracker=False))[
        "update_setup_tracker"
    ] is False
    assert scan_worker.parse_payload(_payload(update_setup_tracker=True))[
        "update_setup_tracker"
    ] is True


@pytest.mark.parametrize("bad", [None, "", "not json", "[1,2]"])
def test_a_malformed_payload_refuses_rather_than_guessing(bad):
    """A default here would run a different scan than the one requested."""
    import scan_worker

    with pytest.raises(ValueError):
        scan_worker.parse_payload(bad)


def test_the_worker_dispatches_each_branch(monkeypatch):
    """Two branches now, not three.

    There used to be a shared/local pair in front of the tracker-writing call,
    and both ran the identical scan over the identical files (packet R1). What
    is left is the distinction that was always real: let the scanner decide
    whether to write the setup tracker, or tell it explicitly.
    """
    import scan_worker

    calls: list[tuple[str, dict]] = []
    module = type(sys)("master_avwap_lib.runner")
    module.run_master = lambda **kw: calls.append(("run_master", kw))
    monkeypatch.setitem(sys.modules, "master_avwap_lib.runner", module)

    scan_worker.run(_payload())
    scan_worker.run(_payload(update_setup_tracker=True))

    assert calls[0] == ("run_master", {})
    assert calls[1] == (
        "run_master",
        {
            "update_setup_tracker": True,
            "require_ib_for_setup_tracker": True,
        },
    )


# ---------------------------------------------------------------------------
# The transport, which is what actually broke
# ---------------------------------------------------------------------------
def test_a_frozen_build_never_passes_dash_c_to_its_own_executable(monkeypatch):
    from ui.services import scan_service

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\dist\TradingBotV3\TradingBotV3.exe")
    command = scan_service.scan_worker_command(_payload())

    assert "-c" not in command, "this is the defect: the app parses -c as its own CLI"
    assert command[1] == scan_service.SCAN_WORKER_FLAG
    assert json.loads(command[2])["update_setup_tracker"] is None


def test_a_source_build_still_uses_the_interpreter_form(monkeypatch):
    from ui.services import scan_service

    monkeypatch.delattr(sys, "frozen", raising=False)
    command = scan_service.scan_worker_command(_payload())
    assert command[1] == "-c"
    assert "scan_worker" in command[2]


def test_the_frozen_flag_is_answered_before_the_desk_argument_parser(monkeypatch):
    """--run-scan must be intercepted the way --selftest is, or it exits 2."""
    sys.path.insert(0, str(ROOT_DIR))
    import launch_gui

    seen: list[str] = []
    worker = type(sys)("scan_worker")
    worker.run = lambda payload: (seen.append(payload), 0)[1]
    monkeypatch.setitem(sys.modules, "scan_worker", worker)

    def explode(*args, **kwargs):  # the desk must never be constructed here
        raise AssertionError("--run-scan reached the GUI entry point")

    app_module = type(sys)("ui.app")
    app_module.main = explode
    monkeypatch.setitem(sys.modules, "ui.app", app_module)
    monkeypatch.setattr(sys, "argv", ["TradingBotV3.exe", "--run-scan", _payload()])

    assert launch_gui.main() == 0
    assert json.loads(seen[0])["update_setup_tracker"] is None


def test_the_marker_has_one_definition():
    """A drifting copy would hang every scan waiting for a string never printed."""
    import scan_worker
    from ui.services import scan_service

    assert scan_service._SCAN_OK_MARKER is scan_worker.SCAN_OK_MARKER


@pytest.mark.slow
def test_the_source_spawn_really_starts_a_child_and_prints_the_marker(tmp_path):
    """The check the packaging guards never made: actually spawn it.

    A stub ``master_avwap_lib`` stands in for the real scanner so this stays
    offline and fast; everything else -- argv shape, interpreter, PYTHONPATH,
    import of ``scan_worker``, the marker on stdout -- is the real path.
    """
    from ui.services import scan_service

    stub = tmp_path / "master_avwap_lib"
    stub.mkdir()
    (stub / "__init__.py").write_text("", encoding="utf-8")
    (stub / "runner.py").write_text(
        textwrap.dedent(
            """
            def run_master(**kwargs):
                print("ran run_master", kwargs)
            """
        ),
        encoding="utf-8",
    )

    command = scan_worker_command_for_source(scan_service, _payload())
    env = {
        **_clean_env(),
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(SCRIPTS_DIR)]),
    }
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env=env, cwd=str(ROOT_DIR)
    )
    assert result.returncode == 0, result.stderr
    assert "ran run_master" in result.stdout
    assert scan_service._SCAN_OK_MARKER in result.stdout


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def scan_worker_command_for_source(scan_service, payload: str) -> list[str]:
    frozen = getattr(sys, "frozen", False)
    try:
        if frozen:
            del sys.frozen  # type: ignore[attr-defined]
        return scan_service.scan_worker_command(payload)
    finally:
        if frozen:
            sys.frozen = True  # type: ignore[attr-defined]
