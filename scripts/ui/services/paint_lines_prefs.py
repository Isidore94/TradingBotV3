"""Which paint-line groups the D1 chart is currently showing.

Machine-local, exactly like ``chart_review_recent_lookups.json``: what lines
one desk likes on its chart is a display preference, not shared state, and it
has no business in the Drive-synced home folder where a second machine would
inherit it.

Stored as the HIDDEN groups rather than the visible ones, which is what makes
the default honest. Every group defaults ON, so an empty (or missing, or
corrupt) file means "show everything" - and a group added by a later version
appears switched on for a trader who has never heard of it, instead of
silently missing because their saved list predates it.

The one exception is ``chart_levels.GROUPS_HIDDEN_BY_DEFAULT`` (Phase 0.10):
a formula under TEST must not appear on a chart nobody asked for it on, so
those groups are off until the trader switches them on. They are recorded in a
second list, ``shown_groups``, so the two defaults can coexist in one file: a
preference file written before such a group existed keeps it off, and the
trader's own hidden groups survive being rewritten beside it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from chart_levels import GROUP_NAMES, GROUPS_HIDDEN_BY_DEFAULT
from project_paths import LOCAL_SETTINGS_DIR

PAINT_LINES_FILE = LOCAL_SETTINGS_DIR / "chart_paint_lines.json"


class PaintLinesPrefs:
    """Per-machine show/hide state for the chart's level groups."""

    def __init__(self, path: Path = PAINT_LINES_FILE) -> None:
        self._path = Path(path)
        self._hidden: set[str] = set()
        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def reload(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        hidden: set[str] = set()
        shown: set[str] = set()
        if isinstance(payload, dict):
            raw = payload.get("hidden_groups")
            if isinstance(raw, list):
                # Unknown names are dropped rather than kept: a stale entry
                # would be an invisible switch nothing in the UI can turn back.
                hidden = {str(name) for name in raw if str(name) in GROUP_NAMES}
            raw_shown = payload.get("shown_groups")
            if isinstance(raw_shown, list):
                shown = {str(name) for name in raw_shown if str(name) in GROUP_NAMES}
        # A default-hidden group is hidden unless this file explicitly says the
        # trader switched it on. Absence is the default, in both directions.
        self._hidden = hidden | (set(GROUPS_HIDDEN_BY_DEFAULT) - shown)

    def hidden_groups(self) -> list[str]:
        return sorted(self._hidden)

    def is_visible(self, group: str) -> bool:
        return str(group) not in self._hidden

    def set_visible(self, group: str, visible: bool) -> None:
        group = str(group)
        if group not in GROUP_NAMES:
            return
        if visible:
            self._hidden.discard(group)
        else:
            self._hidden.add(group)
        self._save()

    def set_hidden_groups(self, groups) -> None:
        """Replace the hidden set wholesale.

        ``groups`` is the COMPLETE hidden set, so a caller that wants a
        default-hidden group to stay hidden must include it. This is the bulk
        setter for a control that already knows every group's state; a caller
        holding only a partial list wants ``set_visible`` instead.
        """
        self._hidden = {str(name) for name in groups or () if str(name) in GROUP_NAMES}
        self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "hidden_groups": sorted(self._hidden - GROUPS_HIDDEN_BY_DEFAULT),
                        "shown_groups": sorted(GROUPS_HIDDEN_BY_DEFAULT - self._hidden),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError:
            pass  # a display preference never blocks a chart
