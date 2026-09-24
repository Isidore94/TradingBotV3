"""Tag chips for the Trades tab: your tags, one-click common tags, and
machine suggestions (dashed) that one click accepts.

The tags line edit stays the one source of truth; chips only edit its text.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtWidgets import QCompleter, QHBoxLayout, QLabel, QPushButton, QWidget

from journal_analytics import is_link_candidate, is_rejection_tag, split_tags
from journal_store import TAG_STATUS_PROVISIONAL

TAG_SEPARATOR = "; "

#: How many one-click common tags the Trades tab offers.
COMMON_TAG_LIMIT = 8

#: Chip text longer than this is shortened; the tooltip keeps the full tag.
CHIP_TEXT_LIMIT = 28

#: A row shows at most this many chips, then says how many more there are.
CHIPS_PER_ROW = 6


def short_tag(tag: str, limit: int = CHIP_TEXT_LIMIT) -> str:
    """``tag`` cut to ``limit`` characters with an ellipsis when longer."""
    return tag if len(tag) <= limit else tag[: max(1, limit - 3)].rstrip() + "..."


def parse_tags(text: str) -> list[str]:
    """The tags in one tags field, in order, without duplicates."""
    seen: set[str] = set()
    tags = []
    for tag in split_tags(text):
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            tags.append(tag)
    return tags


def add_tag(text: str, tag: str) -> str:
    """``text`` with ``tag`` appended, unless it is already there (any case)."""
    tags = parse_tags(text)
    tag = str(tag or "").strip()
    if tag and tag.casefold() not in {existing.casefold() for existing in tags}:
        tags.append(tag)
    return TAG_SEPARATOR.join(tags)


def remove_tag(text: str, tag: str) -> str:
    """``text`` without ``tag`` (any case)."""
    wanted = str(tag or "").strip().casefold()
    return TAG_SEPARATOR.join(existing for existing in parse_tags(text) if existing.casefold() != wanted)


def common_tags(rows: Iterable[dict], limit: int = COMMON_TAG_LIMIT) -> list[str]:
    """Your most-used setup tags on these trades: confirmed only, never links,
    rejections or machine guesses."""
    counts: Counter[str] = Counter()
    spelling: dict[str, str] = {}
    for row in rows:
        if str(row.get("tag_status") or "confirmed") == TAG_STATUS_PROVISIONAL:
            continue
        for tag in parse_tags(str(row.get("setup_tags") or "")):
            if is_link_candidate(tag) or is_rejection_tag(tag):
                continue
            key = tag.casefold()
            spelling.setdefault(key, tag)
            counts[key] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [spelling[key] for key, _count in ordered[: max(0, int(limit))]]


class MultiTagCompleter(QCompleter):
    """Completes the tag being typed after the last separator, keeping the rest."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = QStringListModel(self)
        self.setModel(self._model)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterMode(Qt.MatchContains)
        self.setCompletionMode(QCompleter.PopupCompletion)

    def set_tags(self, tags: Iterable[str]) -> None:
        names = sorted({str(tag).strip() for tag in tags if str(tag or "").strip()}, key=str.casefold)
        if names != self._model.stringList():
            self._model.setStringList(names)

    def tags(self) -> list[str]:
        return list(self._model.stringList())

    def splitPath(self, path: str) -> list[str]:  # noqa: N802 - Qt override
        return [path.split(";")[-1].strip()]

    def pathFromIndex(self, index) -> str:  # noqa: N802 - Qt override
        completion = str(self.model().data(index, Qt.DisplayRole) or "")
        widget = self.widget()
        current = widget.text() if widget is not None else ""
        head = [part.strip() for part in current.split(";")[:-1] if part.strip()]
        return TAG_SEPARATOR.join([*head, completion])


class TagChipBar(QWidget):
    """A row of clickable tag chips. ``kind`` picks the chip style:

    ``mine`` (click removes), ``quick`` (click adds) or ``suggested`` (click accepts).
    """

    tagClicked = Signal(str)

    OBJECT_NAMES = {
        "mine": "JournalTagChip",
        "quick": "JournalQuickTagChip",
        "suggested": "JournalSuggestedTagChip",
    }

    def __init__(
        self, kind: str, caption: str = "", parent: QWidget | None = None, *, empty_text: str = ""
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self._caption_text = caption
        self._empty_text = empty_text
        self._tags: list[str] = []
        self._provisional = False
        self.chips: list[QPushButton] = []
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.caption = QLabel(caption)
        self.caption.setObjectName("MutedLabel")
        self.caption.setVisible(bool(caption or empty_text))
        if not caption:
            self.caption.setText(empty_text)
        self._layout.addWidget(self.caption)
        self.more_label = QLabel("")
        self.more_label.setObjectName("MutedLabel")
        self._layout.addWidget(self.more_label)
        self._layout.addStretch(1)

    def tags(self) -> list[str]:
        return list(self._tags)

    def set_provisional(self, provisional: bool) -> None:
        if provisional != self._provisional:
            self._provisional = provisional
            self._rebuild()

    def set_tags(self, tags: Iterable[str]) -> None:
        tags = list(tags)
        if tags == self._tags:
            return
        self._tags = tags
        self._rebuild()

    def _chip_text(self, tag: str) -> str:
        if self.kind == "mine":
            return f"{short_tag(tag)}  x"
        if self.kind == "suggested":
            return f"+ {short_tag(tag)}"
        return short_tag(tag)

    def _chip_tip(self, tag: str) -> str:
        if self.kind == "mine":
            return f"Remove '{tag}' (then Save tags and notes)"
        if self.kind == "suggested":
            return f"Accept '{tag}' as your tag now"
        return f"Add '{tag}' (then Save tags and notes)"

    def _rebuild(self) -> None:
        for chip in self.chips:
            self._layout.removeWidget(chip)
            chip.deleteLater()
        self.chips = []
        shown = self._tags[:CHIPS_PER_ROW]
        for index, tag in enumerate(shown):
            chip = QPushButton(self._chip_text(tag))
            chip.setObjectName(self.OBJECT_NAMES.get(self.kind, "JournalQuickTagChip"))
            if self.kind == "mine" and self._provisional:
                chip.setProperty("provisional", True)
            chip.setToolTip(self._chip_tip(tag))
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _checked=False, value=tag: self.tagClicked.emit(value))
            self._layout.insertWidget(1 + index, chip)
            self.chips.append(chip)
        hidden = len(self._tags) - len(shown)
        self.more_label.setText(f"+{hidden} more in the field below" if hidden and self.kind == "mine"
                                else (f"+{hidden} more" if hidden else ""))
        self.more_label.setVisible(bool(hidden))
        if not self._caption_text:
            self.caption.setText("" if self._tags else self._empty_text)
            self.caption.setVisible(not self._tags and bool(self._empty_text))
        self.setVisible(bool(self._tags) or self.kind == "mine")
