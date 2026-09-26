"""Patch a module global the Alert Center reads, in every module that now holds its code.

`AlertCenterPanel` is split across `ui/panels/alert_center_panel.py` and mixin
modules under `ui/panels/alert_center/`. A method looks names up in its OWN
module, so a test that freezes `datetime` (or stubs a `chart_watch` function)
must patch each module that holds a copy of the name.
"""

from __future__ import annotations

import importlib

PANEL_MODULE = "ui.panels.alert_center_panel"
#: The mixin modules holding moved `AlertCenterPanel` methods that read module globals.
MIXIN_MODULES: tuple[str, ...] = (
    "ui.panels.alert_center.wall",
    "ui.panels.alert_center.any_bounce",
    "ui.panels.alert_center.h1",
)


def patch_alert_center_global(monkeypatch, name: str, value) -> None:
    """Set `name` to `value` in the panel module and every mixin module that has it."""
    patched = 0
    for module_name in (PANEL_MODULE, *MIXIN_MODULES):
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    assert patched, f"no Alert Center module holds {name!r}"
