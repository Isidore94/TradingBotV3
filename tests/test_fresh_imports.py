"""Fresh-interpreter import checks (P2-11 review, blocker 1).

`master_avwap_lib.legacy` is no longer loaded by the package, so the order in
which a process first touches legacy and runner changed. Each check here runs
in a NEW interpreter with scratch stores, because this test process has long
since imported everything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _run(code: str, tmp_path: Path, *, timeout: float = 240) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        {
            "TRADINGBOTV3_DATA_DIR": str(tmp_path / "data"),
            "LOCALAPPDATA": str(tmp_path / "local"),
            "TRADINGBOT_DIAGNOSTICS_DIR": str(tmp_path / "diag"),
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "PYTHONPATH": f"{SCRIPTS}{os.pathsep}{ROOT}",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=timeout, cwd=str(tmp_path)
    )


def test_runner_imports_first_in_a_fresh_interpreter(tmp_path):
    """The scan child's first import. legacy's bottom used to import runner back."""
    result = _run(
        "from master_avwap_lib.runner import run_master\n"
        "import master_avwap_lib, master_avwap_lib.legacy as legacy\n"
        "assert legacy.run_master is run_master\n"
        "assert master_avwap_lib.run_master is run_master\n"
        "import master_avwap\n"
        "assert master_avwap.run_master is run_master and callable(master_avwap.main)\n"
        "print('ok')\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "ok" in result.stdout


def test_legacy_imports_first_in_a_fresh_interpreter(tmp_path):
    result = _run(
        "import master_avwap_lib.legacy as legacy\n"
        "from master_avwap_lib.runner import run_master\n"
        "assert legacy.run_master is run_master\n"
        "assert 'run_master' in dir(legacy)\n"
        "print('ok')\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "ok" in result.stdout


def test_the_scan_worker_entry_reaches_run_master(tmp_path):
    """scan_worker.run imports runner itself; only the scan body is stubbed.

    A meta-path hook swaps `run_master` the moment runner finishes loading, so
    scan_worker's own import is still the first import of runner.
    """
    code = r'''
import importlib.abc, importlib.machinery, sys

calls = []


class _StubAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "master_avwap_lib.runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            module.run_master = lambda **kwargs: calls.append(kwargs)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _StubAfterLoad())
from scan_worker import SCAN_OK_MARKER, run

assert run('{"update_setup_tracker": false}') == 0
assert calls and calls[0]["update_setup_tracker"] is False, calls
print("reached", SCAN_OK_MARKER)
'''
    result = _run(code, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "reached" in result.stdout


@pytest.mark.timeout(900)
def test_every_scripts_module_imports_as_the_first_import(tmp_path):
    result = _run(
        f"import runpy, sys\nsys.argv = ['sweep']\nrunpy.run_path({str(ROOT / 'tests' / 'fresh_import_sweep.py')!r}, run_name='__main__')\n",
        tmp_path,
        timeout=880,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["count"] > 300
    assert report["failed"] == {}, json.dumps(report["failed"], indent=1)
