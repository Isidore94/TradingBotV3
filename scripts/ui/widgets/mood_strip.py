r"""The two-click mood strip - five faces and at most two chips (TJ-7 change 2).

`plan.md` §12.4 "TJ-7" change 2: *"A two-click strip (mood 1-5 + up to two
chips) on the Trade Mentor popup and the desk's journal tab; optional, never
required, never asked twice for one row."*

ONE widget, used by BOTH surfaces, so the cap, the codes and the "nothing
pre-selected" rule cannot drift into two versions. It is a pure input: it reads
no store, writes nothing, and hands its answer back as a mapping the host passes
to `market_journal`'s one writer.

Three rules it holds, and why each one exists:

* **Nothing is ever pre-selected.** No remembered face, no default 3, no "same
  as yesterday". A mood is the trader's own click or nothing at all - a machine
  that filled one in would be putting words in their mouth in an append-only
  ledger.
* **The cap is the VOCABULARY's** (`trader_state_tags.MAX_STATE_TAGS`, read at
  click time). A third chip un-checks itself here rather than producing a row
  the writer would refuse at the end of the trader's typing.
* **The codes come from the vocabulary file**, never a literal list, so a v2
  ships beside v1 and this strip follows it without an edit.

Style is keyed in `ui/theme.qss` on the object names below; nothing here sets a
per-widget stylesheet, because a stylesheet is a parse on the Qt thread.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme

_log = logging.getLogger(__name__)

#: The object names `theme.qss` keys on.
FACE_OBJECT_NAME = "MoodFaceButton"
CHIP_OBJECT_NAME = "StateTagChipButton"

#: What the strip says about itself. Short, and honest about being optional.
PROMPT = "How were you? (optional)"


class MoodStrip(QWidget):
    """Five faces, up to two chips, and an answer nobody filled in for you."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MoodStrip")
        import market_journal

        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)

        self.prompt = QLabel(PROMPT, self)
        self.prompt.setObjectName("SectionSubtitle")
        body.addWidget(self.prompt)

        faces = QHBoxLayout()
        faces.setContentsMargins(0, 0, 0, 0)
        faces.setSpacing(3)
        self._faces: dict[int, QPushButton] = {}
        self._face_group = QButtonGroup(self)
        self._face_group.setExclusive(True)
        for score in market_journal.MOOD_SCALE:
            button = QPushButton(str(score), self)
            button.setObjectName(FACE_OBJECT_NAME)
            button.setCheckable(True)
            button.setChecked(False)
            button.setToolTip(f"{score} of {len(market_journal.MOOD_SCALE)}")
            button.setFixedWidth(theme.px(28))
            self._face_group.addButton(button, int(score))
            self._faces[int(score)] = button
            faces.addWidget(button)
        faces.addStretch(1)
        body.addLayout(faces)

        chips = QHBoxLayout()
        chips.setContentsMargins(0, 0, 0, 0)
        chips.setSpacing(3)
        self._chips: dict[str, QPushButton] = {}
        self._codes: tuple[str, ...] = ()
        for code, label in self._vocabulary():
            chip = QPushButton(label, self)
            chip.setObjectName(CHIP_OBJECT_NAME)
            chip.setCheckable(True)
            chip.setChecked(False)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.toggled.connect(lambda checked, name=code: self._on_chip(name, checked))
            self._chips[code] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        body.addLayout(chips)

        self._face_group.idToggled.connect(lambda _id, _on: self.changed.emit())

    # -- the vocabulary ----------------------------------------------------
    def _vocabulary(self) -> tuple[tuple[str, str], ...]:
        """`(code, label)` per chip, from the versioned file. Never a literal.

        A missing vocabulary is a packaging defect. It costs the CHIPS and
        nothing else: the faces still work, the host's Save is untouched, and
        the reason is logged rather than raised in front of the trader.
        """
        try:
            import trader_state_tags

            book = trader_state_tags.load_vocabulary()
        except Exception:  # noqa: BLE001 - a missing picklist never breaks a card
            _log.warning("The state-tag vocabulary could not be read.", exc_info=True)
            return ()
        rows = tuple(
            (str(entry["code"]), str(entry.get("label") or entry["code"]))
            for entry in book.get("entries") or ()
        )
        self._codes = tuple(code for code, _label in rows)
        return rows

    def _cap(self) -> int:
        """The cap, read from its ONE owner AT CLICK TIME."""
        try:
            import trader_state_tags

            return int(trader_state_tags.MAX_STATE_TAGS)
        except Exception:  # noqa: BLE001
            return 2

    # -- the answer --------------------------------------------------------
    def mood_button(self, score: Any) -> QAbstractButton | None:
        """The face for `score`, or ``None``."""
        try:
            return self._faces.get(int(score))
        except (TypeError, ValueError):
            return None

    def state_tag_button(self, code: str) -> QAbstractButton | None:
        """The chip for `code`, or ``None``."""
        return self._chips.get(str(code or ""))

    def answer(self) -> dict[str, Any]:
        """`{"mood": int|None, "state_tags": tuple}` - what the trader clicked."""
        score: int | None = None
        for value, button in self._faces.items():
            if button.isChecked():
                score = int(value)
                break
        chosen = tuple(
            code for code in self._codes if self._chips[code].isChecked()
        )
        return {"mood": score, "state_tags": chosen}

    def is_touched(self) -> bool:
        """Did the trader click anything at all on this strip?"""
        answer = self.answer()
        return answer["mood"] is not None or bool(answer["state_tags"])

    def reset(self) -> None:
        """Back to nothing selected. Yesterday's face is not tomorrow's."""
        self._face_group.setExclusive(False)
        for button in self._faces.values():
            button.setChecked(False)
        self._face_group.setExclusive(True)
        for chip in self._chips.values():
            chip.setChecked(False)

    # -- the cap -----------------------------------------------------------
    def _on_chip(self, code: str, checked: bool) -> None:
        if checked:
            chosen = [name for name in self._codes if self._chips[name].isChecked()]
            if len(chosen) > self._cap():
                # The third click un-checks ITSELF: the two the trader already
                # chose stay, and the writer never sees a row it would refuse.
                self._chips[code].setChecked(False)
                return
        self.changed.emit()


__all__ = ["CHIP_OBJECT_NAME", "FACE_OBJECT_NAME", "PROMPT", "MoodStrip"]
