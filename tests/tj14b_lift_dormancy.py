"""The ONE authorised amendment to the TJ-14B red tests. NOT a test module.

Lead decision 1, 2026-09-19 night: a question is ASKED only when its answer has
a reader (decision 0021 answer 28). Three kinds are registered with their
trigger, options, `writes`, `answer_key` and the consumer they WILL have, and
carry `dormant_until`:

* `trade_origin` and `open_position_check` -> TJ-12 (the Process line and the
  long-hold rows);
* `grader_gap` -> TJ-10 (no deterministic reader emits `needs_trader_input`
  yet).

`mentor_questions.pending()` never returns a dormant kind on a live card, so a
tester test that needs one to FIRE lifts the dormancy for that test only, with
`dataclasses.replace(kind, dormant_until="")` - exactly the amendment the lead
authorised. Nothing else in those files moves.

**Only `trade_origin` and `open_position_check` are lifted**, and deliberately
not `grader_gap`: `tests/test_tj14b_budget.py::_seven_budgeted_subjects` counts
"two unplanned trades, two long-held open positions, one traded quick like, one
grader gap and one AI question = 7 budgeted subjects", and its three trade rows
are byte-identical apart from their ids, so a rule-based trigger fires
`trade_origin` on all THREE. Lifting these two and leaving the (still unemitted)
grader gap dormant is what makes the fixture's own arithmetic - seven owed,
three asked, four carried - hold.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: The kinds whose dormancy is lifted for these tests, and nothing else.
LIFTED = ("trade_origin", "open_position_check")


@pytest.fixture(autouse=True)
def lift_dormancy(monkeypatch):
    import mentor_questions

    monkeypatch.setattr(
        mentor_questions,
        "REGISTRY",
        tuple(
            dataclasses.replace(kind, dormant_until="")
            if kind.kind in LIFTED
            else kind
            for kind in mentor_questions.REGISTRY
        ),
    )
