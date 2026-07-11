"""Phase 0.5 (plan.md): the deterministic smoke command stays green."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_all_smoke_checks_pass():
    import smoke_check

    results = smoke_check.run_smoke()
    failures = [(name, detail) for name, ok, detail in results if not ok]
    assert not failures, failures
    assert len(results) == len(smoke_check.CHECKS)
