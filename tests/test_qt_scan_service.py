import io
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_master_scan_subprocess_uses_child_python(monkeypatch):
    from ui.services import scan_service

    captured = {}

    class FakeProc:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.stdout = io.StringIO("SCAN_SUBPROCESS_OK\n")
            self.stderr = io.StringIO("")

        def poll(self):
            return None  # still alive: theta enrichment finishing in the background

        def wait(self):
            return 0

    monkeypatch.setattr(scan_service.subprocess, "Popen", FakeProc)

    result = scan_service._run_master_scan_subprocess(use_shared_watchlists=True)

    assert captured["args"][:2] == [sys.executable, "-c"]
    assert captured["kwargs"]["cwd"] == str(ROOT_DIR)
    assert str(SCRIPTS_DIR) in captured["kwargs"]["env"]["PYTHONPATH"]
    # The marker alone completes the scan; process exit is not required.
    assert result["subprocess_stdout"] == "SCAN_SUBPROCESS_OK"


def test_master_scan_subprocess_reports_child_failure(monkeypatch):
    from ui.services import scan_service

    class FakeProc:
        def __init__(self, _args, **_kwargs):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("Fatal Python error")

        def poll(self):
            return 3221225477

        def wait(self):
            return 3221225477

    monkeypatch.setattr(scan_service.subprocess, "Popen", FakeProc)

    try:
        scan_service._run_master_scan_subprocess(use_shared_watchlists=False)
    except RuntimeError as exc:
        assert "exited with code 3221225477" in str(exc)
        assert "Fatal Python error" in str(exc)
    else:
        raise AssertionError("expected child process failure")
