"""bounce_bot's script entry point must read the module-level ``sys``.

A local ``import sys`` inside ``main()`` made ``sys`` local to the whole
function, so the ``sys.argv`` check at its top raised UnboundLocalError.
"""

from __future__ import annotations

import symtable
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / "scripts" / "bounce_bot_lib" / "legacy.py"


def _function_table(name: str) -> symtable.SymbolTable:
    top = symtable.symtable(LEGACY.read_text(encoding="utf-8"), str(LEGACY), "exec")
    for child in top.get_children():
        if child.get_name() == name and child.get_type() == "function":
            return child
    raise AssertionError(f"{name}() not found in {LEGACY}")


def test_main_reads_the_module_level_sys():
    symbol = _function_table("main").lookup("sys")
    assert not symbol.is_local(), "main() binds sys locally; sys.argv at its top would raise"
    assert symbol.is_global()
