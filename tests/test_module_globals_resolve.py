"""Every global name a module reads must actually exist in that module.

This exists because of a real defect. Packet R1 deleted
`get_shared_watchlist_paths` from `master_avwap_lib.legacy`'s import block.
`master_avwap_lib.gui` copies legacy's globals wholesale
(``globals().update(vars(_legacy))``) and called that name inside
`refresh_tracker_storage_summary`, so the legacy Tk GUI began raising
NameError at construction. The whole suite stayed green: these modules are
imported but never constructed, and the frozen self-test only imports too.

The check is static and conservative - it only reports a name that is loaded,
is not bound anywhere in its own scope chain, is not a module global, and is
not a builtin - so a hit is a genuine NameError waiting for whoever opens
that window.
"""

import ast
import builtins
import importlib
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

#: Modules the suite imports but never constructs, so nothing else would
#: notice a name vanishing out from under them.
UNCONSTRUCTED_MODULES = (
    "master_avwap_lib.gui",
    "gui_app.app",
    "gui_app.master_panel",
)


def _bound_names(node: ast.AST) -> set[str]:
    """Names bound anywhere inside one scope, not descending into nested ones."""
    bound: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
            continue  # its body is a different scope
        if isinstance(child, ast.Lambda):
            continue
        for sub in ast.walk(child):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                bound.add(sub.id)
            elif isinstance(sub, ast.arg):
                bound.add(sub.arg)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                bound.add(sub.name)
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                bound.update(sub.names)
    return bound


def _scope_bound(node: ast.AST) -> set[str]:
    """Everything bound in a function/lambda scope, including its parameters."""
    bound = set(_bound_names(node))
    args = getattr(node, "args", None)
    if args is not None:
        for group in ("posonlyargs", "args", "kwonlyargs"):
            bound.update(arg.arg for arg in getattr(args, group, []) or [])
        for single in (args.vararg, args.kwarg):
            if single is not None:
                bound.add(single.arg)
    return bound


def _unresolved(tree: ast.AST, available: set[str]) -> set[str]:
    missing: set[str] = set()

    def visit(node: ast.AST, enclosing: set[str]) -> None:
        scope = enclosing | _scope_bound(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                visit(child, scope)
                continue
            for sub in ast.walk(child):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    visit(sub, scope)
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id not in scope:
                        missing.add(sub.id)

    visit(tree, set())
    return {name for name in missing if name not in available}


@pytest.mark.parametrize("module_name", UNCONSTRUCTED_MODULES)
def test_every_global_a_module_reads_actually_exists(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional GUI deps
        pytest.skip(f"{module_name} needs {exc.name}")
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    available = set(dir(module)) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    missing = _unresolved(tree, available)
    assert not missing, (
        f"{module_name} reads name(s) that do not exist in it: {sorted(missing)}. "
        "A wholesale `globals().update(vars(legacy))` copy makes this silent until "
        "the window is actually opened."
    )
