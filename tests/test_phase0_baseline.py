"""Phase 0 baseline guards from plan.md: dormant defects and structural rules."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PRODUCTION_ROOTS = (ROOT_DIR / "scripts", ROOT_DIR / "market_prep")


class DuplicateTopLevelDefinitionTests(unittest.TestCase):
    """A later duplicate silently shadows the earlier definition (plan 10.4).

    The Master legacy module shipped two `_priority_expected_r_text`
    definitions for months; this guard keeps that class of defect out.
    """

    def test_no_duplicate_top_level_definitions_in_production_modules(self):
        offenders: list[str] = []
        for root in PRODUCTION_ROOTS:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                seen: dict[str, int] = {}
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name in seen:
                            offenders.append(
                                f"{path.relative_to(ROOT_DIR)}: {node.name} "
                                f"(lines {seen[node.name]} and {node.lineno})"
                            )
                        seen[node.name] = node.lineno
        self.assertEqual(offenders, [], "duplicate top-level definitions found:\n" + "\n".join(offenders))


class PreviousDayExtremesTests(unittest.TestCase):
    """Regression for the dormant `datetimde` typo (plan 17 / 21.2)."""

    def test_previous_day_extremes_returns_prior_session_high_low(self):
        from bounce_bot_lib.legacy import BounceBot

        rows = []
        for hour, high, low in ((10, 101.0, 99.0), (11, 103.5, 98.5), (12, 102.0, 99.5)):
            rows.append({"time": f"20260709  {hour:02d}:00:00", "high": high, "low": low})
        for hour, high, low in ((10, 110.0, 105.0), (11, 111.0, 106.0)):
            rows.append({"time": f"20260710  {hour:02d}:00:00", "high": high, "low": low})
        df = pd.DataFrame(rows)

        prev_high, prev_low = BounceBot.get_previous_day_extremes(SimpleNamespace(), df)

        self.assertEqual(prev_high, 103.5)
        self.assertEqual(prev_low, 98.5)

    def test_previous_day_extremes_without_prior_session_is_none(self):
        from bounce_bot_lib.legacy import BounceBot

        df = pd.DataFrame(
            [{"time": "20260710  10:00:00", "high": 110.0, "low": 105.0}]
        )
        prev_high, prev_low = BounceBot.get_previous_day_extremes(SimpleNamespace(), df)
        self.assertIsNone(prev_high)
        self.assertIsNone(prev_low)


if __name__ == "__main__":
    unittest.main()
