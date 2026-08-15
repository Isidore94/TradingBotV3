"""Read-only viewer for `docs/DESK_TESTING_PLAN.md` (Settings ▸ Testing Plan).

The markdown file is the single source of truth; this is a window onto it and
nothing else. It owns no timer, writes no state, and touches no engine - the
trader can open it mid-session at any time and the desk cannot notice.

A missing or unreadable file says so plainly rather than showing whatever was
loaded last. A stale runbook read as current is worse than no runbook: it would
have the trader checking for log lines the build no longer prints.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

#: Where the plan lives, relative to whichever root this build has.
PLAN_RELATIVE_PATH = Path("docs") / "DESK_TESTING_PLAN.md"


def resolve_testing_plan_path() -> Path:
    """The plan's location in a source checkout or inside the frozen bundle.

    The two roots differ: a source run finds it three levels up from this file,
    while a frozen run has no `scripts/` tree at all and unpacks the bundled
    copy under ``sys._MEIPASS``. The spec has an explicit `datas` rule putting
    it at `docs/` there - this asset sits OUTSIDE `scripts/`, so the spec's
    package-asset sweep (and the drift test that guards it) would never have
    noticed it going missing.
    """
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        return Path(meipass) / PLAN_RELATIVE_PATH
    return Path(__file__).resolve().parents[3] / PLAN_RELATIVE_PATH


TESTING_PLAN_PATH = resolve_testing_plan_path()

MISSING_MESSAGE = (
    "# Plan file not found\n\n"
    "`docs/DESK_TESTING_PLAN.md` could not be read, so there is nothing to show.\n\n"
    "This panel deliberately shows no cached copy - an out-of-date testing plan "
    "would have you checking for things the current build does not do.\n\n"
    "Expected location:\n\n"
    "`{path}`\n"
)


class TestingPlanView(QFrame):
    """Renders the testing-plan markdown, with a manual refresh."""

    def __init__(self, path: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        # Resolved per instance, not captured at import: a frozen run and a
        # source run disagree about the root, and tests inject their own.
        self._path = Path(path) if path is not None else resolve_testing_plan_path()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, 1)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip(
            "Re-read the plan from disk. It is a file in the repo, so it changes "
            "whenever the testing sequence changes."
        )
        self.refresh_button.clicked.connect(self.reload)
        controls.addWidget(self.refresh_button)
        layout.addLayout(controls)

        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(False)
        # Nothing here is clickable-through: the links are repo-relative and
        # would resolve to nothing useful inside the desk.
        self.viewer.setOpenLinks(False)
        layout.addWidget(self.viewer, 1)

        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def reload(self) -> bool:
        """Re-read the file. True when it rendered, False when it could not."""
        text, modified = self._read()
        if text is None:
            self.viewer.setMarkdown(MISSING_MESSAGE.format(path=self._path))
            self.status_label.setText(
                f"⚠ Plan file not found: {self._path}"
            )
            return False
        self.viewer.setMarkdown(text)
        stamp = modified.strftime("%Y-%m-%d %H:%M") if modified else "unknown"
        self.status_label.setText(f"{self._path.name} · last changed {stamp}")
        return True

    def _read(self) -> tuple[str | None, datetime | None]:
        try:
            text = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logging.warning("Testing plan could not be read: %s", self._path, exc_info=True)
            return None, None
        if not text.strip():
            # An empty file is as unhelpful as a missing one, and silently
            # rendering blank would look like a rendering bug rather than a
            # content problem.
            return None, None
        try:
            modified = datetime.fromtimestamp(self._path.stat().st_mtime)
        except OSError:
            modified = None
        return text, modified
