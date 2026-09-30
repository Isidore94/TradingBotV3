"""Trade Mentor packs: the registry contract every pack module must meet."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_packs import registry  # noqa: E402


def test_phase_zero_registers_context_plan_and_recall():
    assert set(registry.names()) >= {"context_pack", "plan_lines", "recall"}


@pytest.mark.parametrize("name", sorted(registry.names()))
def test_every_pack_meets_the_contract(name):
    module = registry.modules()[name]
    assert module.SCHEMA["type"] == "function"
    assert module.SCHEMA["function"]["name"] == name
    assert module.SCHEMA["function"]["parameters"]["type"] == "object"
    assert callable(module.build) and callable(module.fixture)
    pack = module.fixture()
    assert pack.name == name
    assert pack.ids, f"{name}'s fixture carries no citable ids"
    assert len(pack.ids) == len(set(pack.ids)), f"{name} repeats an id: {pack.ids}"
    for row_id in pack.ids:
        assert f"[{row_id}]" in pack.as_text()
    assert (ROOT_DIR / "tests" / f"test_mentor_packs_{name}.py").exists(), f"{name} has no fixture test file"


def test_tool_schemas_follow_the_registry():
    names = [schema["function"]["name"] for schema in registry.tool_schemas()]
    assert names == list(registry.names())
    assert len(names) == len(set(names))


def test_an_unknown_or_broken_pack_is_empty_not_a_raise(monkeypatch):
    assert registry.build("no_such_pack").ids == ()
    module = registry.modules()["plan_lines"]
    monkeypatch.setattr(module, "build", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    pack = registry.build("plan_lines")
    assert pack.ids == () and "unknown" in pack.as_text()


def test_packs_are_qt_free():
    for dotted in registry.PACK_MODULES + ("mentor_packs.registry", "mentor_packs.citations"):
        source = (SCRIPTS_DIR / Path(*dotted.split("."))).with_suffix(".py").read_text(encoding="utf-8")
        assert "PySide6" not in source and "from ui" not in source and "import ui" not in source, dotted


def test_pack_json_round_trip():
    pack = registry.modules()["plan_lines"].fixture()
    again = registry.pack_from_json(pack.as_json())
    assert again.ids == pack.ids and again.as_text() == pack.as_text()
