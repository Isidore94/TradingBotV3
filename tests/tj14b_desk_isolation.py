"""One desk-day per test, for the tests that drive the real Mentor slot.

NOT a test module. Import `fresh_mentor_pull_tally` into any test module that
calls `MainWindow._show_trade_mentor_prompt` for real:

    from tj14b_desk_isolation import fresh_mentor_pull_tally  # noqa: F401

WHY IT EXISTS. TJ-14B item 4 gives the desk's day-time Questrade attempts ONE
owner and persists their tally BESIDE the Mentor slot state, so the per-day cap
survives a desk restart (lead decision 3, which also closes TJ-9's advisory that
`_journal_retry_date` lived only in memory). On a desk that is exactly right:
one machine, one day, one budget.

In the suite it is not. Every test that builds a `MainWindow` gets a fresh
window and a fresh `TradeMentorService`, but they all resolve the SAME
machine-local `TRADE_MENTOR_SLOTS_FILE` (`project_paths` reads the environment
once, at import), and they all drive the same session date. Without this the
cards of one test spend the cap of the next, and the ORDER of the file decides
which assertions hold - which is how a green suite hides a real rule.

It lives here rather than in `tests/conftest.py` because only three modules need
it and `conftest.py` is loaded by every test in the repository; and rather than
in `tj14b_support.py` because that module is the TJ-14B fixtures' data, and TJ-9's
desk-seam tests need this without needing any of that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def fresh_mentor_pull_tally():
    """Drop the persisted day's journal-pull tally before each test."""
    try:
        from project_paths import TRADE_MENTOR_SLOTS_FILE

        path = Path(TRADE_MENTOR_SLOTS_FILE)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.pop("pull_tally", None) is not None:
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
                )
    except Exception:  # noqa: BLE001 - isolation never fails a test by itself
        pass
    yield
