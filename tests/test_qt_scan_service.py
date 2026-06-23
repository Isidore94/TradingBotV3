import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_master_scan_subprocess_uses_child_python(monkeypatch):
    from ui.services import scan_service

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="SCAN_SUBPROCESS_OK\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = scan_service._run_master_scan_subprocess(use_shared_watchlists=True)

    assert captured["args"][:2] == [sys.executable, "-c"]
    assert captured["kwargs"]["cwd"] == str(ROOT_DIR)
    assert str(SCRIPTS_DIR) in captured["kwargs"]["env"]["PYTHONPATH"]
    assert result["subprocess_stdout"] == "SCAN_SUBPROCESS_OK"


def test_master_scan_subprocess_reports_child_failure(monkeypatch):
    from ui.services import scan_service

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=3221225477, stdout="", stderr="Fatal Python error"),
    )

    try:
        scan_service._run_master_scan_subprocess(use_shared_watchlists=False)
    except RuntimeError as exc:
        assert "exited with code 3221225477" in str(exc)
        assert "Fatal Python error" in str(exc)
    else:
        raise AssertionError("expected child process failure")
