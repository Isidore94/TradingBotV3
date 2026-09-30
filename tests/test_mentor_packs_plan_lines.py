"""Trade Mentor plan pack: plan ids, empty plan, never creates or snapshots the plan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import plan_lines  # noqa: E402


def test_fixture_lines_carry_plan_ids():
    pack = plan_lines.fixture()
    assert pack.ids == ("plan:risk:1", "plan:risk:2")
    assert "[plan:risk:2] No new entries after 12:30 PT." in pack.as_text()


def test_a_missing_plan_is_no_plan_lines_and_is_not_created(tmp_path):
    target = tmp_path / "trading_plan.md"
    pack = plan_lines.build(path=target)
    assert pack.ids == ()
    assert plan_lines.EMPTY_TEXT in pack.as_text()
    assert not target.exists()


def test_an_empty_plan_says_no_plan_lines_and_nothing_else(tmp_path):
    target = tmp_path / "trading_plan.md"
    target.write_text("# Trading plan\n\n## Risk\n\n", encoding="utf-8")
    pack = plan_lines.build(path=target)
    assert pack.as_text().splitlines()[1:] == [plan_lines.EMPTY_TEXT]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trading_plan.md"], "no snapshot may be written"
