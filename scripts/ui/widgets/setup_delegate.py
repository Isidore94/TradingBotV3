from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QToolTip

import avwape_side
import compression_chip
import earnings_warning
from ui import theme
from ui.models.setup import SetupRow
from ui.models.setup_table_model import ROW_ROLE, SetupTableModel
from swallowed import note_swallowed


_COLUMN_KEYS = [key for key, _label in SetupTableModel.COLUMNS]
_ROW_HEIGHT = 40
_CHIP_HEIGHT = 22
_PAD = 10
#: Gap between two chips sharing one cell (WS-WS).
_CHIP_GAP = 6
#: Narrower than this and the second chip is not drawn at all: a 12px sliver of
#: colour is not a badge, and the tooltip still says the whole thing.
_MIN_CHIP_WIDTH = 22

#: WS-SX tooltip wording. The MARK says that a decision exists; the tooltip says
#: WHICH and WHEN. The kinds are `pick_feedback.LIKE_KINDS` / `REJECT_KINDS`; a
#: kind with no entry here is printed as itself rather than swallowed, so a new
#: verdict shows up as text instead of disappearing.
_LIKE_LABELS = {"quick": "quick", "claimed": "claimed", "like": "star"}
_REJECT_LABELS = {
    "veto": "Vetoed today",
    "dislike": "Disliked today",
    "not_today": "Not today",
    "pass": "Passed today",
    "m5_click_away": "Passed today",
    "remove_today": "Removed from today",
}
_REJECT_SUFFIXES = {"m5_click_away": "(M5 click-away)"}


def _clock_text(stamp: object) -> str:
    """`HH:MM` as the row was written - the stamp's own wall clock, unconverted.

    The three ledgers spell time differently (`ts` with seconds, `created_at`
    with microseconds and an explicit offset). Nothing here re-zones a stamp:
    the trader wants to know when THEY clicked, which is what the row already
    says.
    """
    text = str(stamp or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).strftime("%H:%M")
    except ValueError as exc:
        note_swallowed("setup stamp not ISO; trying the time part", exc, quiet=True)
    marker = text.find("T")
    return text[marker + 1 : marker + 6] if marker != -1 else ""


def _resized(font: QFont, delta: float, *, minimum: float = 1.0) -> QFont:
    """A copy of ``font`` ``delta`` units larger, in whatever unit it uses.

    The theme sizes every styled widget in **pixels** on purpose (`theme.py`
    emits ``"{size}px"`` so type is device-independent across the desk and the
    MacBook). For a pixel-sized font ``QFont.pointSizeF()`` returns **-1**, and
    arithmetic on that is what produced the console flood the trader reported
    on 2026-08-21::

        font.setPointSizeF(font.pointSizeF() + 1.0)   # -1 + 1 = 0.0

    ``QFont::setPointSizeF: Point size <= 0`` - once per visible row per
    repaint, from inside ``paint()``. Qt rejected the call, so the mark kept the
    row size; the neighbouring star, at ``-1 + 2.0``, was accepted and drew at
    **one point**.

    So: ask the font which unit it is in and stay in that unit.
    """
    resized = QFont(font)
    points = font.pointSizeF()
    if points > 0:
        resized.setPointSizeF(max(minimum, points + delta))
        return resized
    pixels = font.pixelSize()
    if pixels > 0:
        # A point is about 4/3 of a pixel at 96 DPI, which is the ratio the
        # theme's own px sizes were chosen against.
        resized.setPixelSize(max(int(minimum), int(round(pixels + delta * 4.0 / 3.0))))
        return resized
    # Neither unit is readable - leave the font exactly as it came rather than
    # inventing a size for it.
    return resized


class SetupTableDelegate(QStyledItemDelegate):
    """Paints the setups table as a scannable surface: side/bucket chips, a
    score bar, a favorite accent stripe, and de-emphasized study rows.

    All colors come from the active theme tokens so light/dark both look right.
    """

    _focus_lookup = None
    _decision_lookup = None

    def set_focus_lookup(self, lookup) -> None:
        """`lookup(symbol) -> bool` flags Focus Picks with a star in the Symbol cell."""
        self._focus_lookup = lookup

    def set_decision_lookup(self, lookup) -> None:
        """`lookup(symbol) -> SymbolDecisions` - today's likes and rejects (WS-SX).

        Presentation only, and NEVER a file read: the panel hands in a lookup
        over one already-parsed, mtime-keyed snapshot, because `paint` runs once
        per visible cell per repaint.
        """
        self._decision_lookup = lookup

    def _is_focus(self, row) -> bool:
        if self._focus_lookup is None or not isinstance(row, SetupRow) or not row.symbol:
            return False
        try:
            return bool(self._focus_lookup(row.symbol))
        except Exception:
            return False

    def _decisions(self, row):
        """Today's decisions for this row's symbol, or None when unknown.

        A lookup that raises is the same as no lookup: the table is never worth
        an exception, and an unanswered column is simply today's plain mark.
        """
        if self._decision_lookup is None or not isinstance(row, SetupRow) or not row.symbol:
            return None
        try:
            return self._decision_lookup(row.symbol)
        except Exception:
            return None

    @staticmethod
    def _liked_today(decisions) -> tuple:
        return tuple(getattr(decisions, "liked", ()) or ())

    @staticmethod
    def _rejected_today(decisions) -> tuple:
        return tuple(getattr(decisions, "rejected", ()) or ())

    def helpEvent(self, event, view, option, index):  # noqa: N802 (Qt override)
        """The ★/✕ tooltips say which decision and when (WS-SX item 3).

        The delegate owns them rather than the model's `ToolTipRole`, because
        the answer is the same already-parsed decision snapshot `paint` reads -
        putting it in the model would mean the lookup living in two places and
        the model re-emitting `dataChanged` for a hover.
        """
        if event is not None and event.type() == QEvent.Type.ToolTip:
            text = self._decision_tooltip(index) or self._bucket_tooltip(index)
            if text:
                QToolTip.showText(event.globalPos(), text, view)
                return True
        return super().helpEvent(event, view, option, index)

    @staticmethod
    def _wrong_side_read(row):
        """This row's side-of-the-anchor reading, or None. Never raises.

        `paint` and `sizeHint` both ask, so it stays what `avwape_side` is: a
        dict lookup and one short string split, no I/O and no clock.
        """
        if not isinstance(row, SetupRow):
            return None
        try:
            return avwape_side.read_row(row.raw)
        except Exception:
            return None

    @staticmethod
    def _compression_read(row):
        """This row's compression reading, or None. Never raises (PCT-3).

        `paint` and `sizeHint` both ask, so it stays what `compression_chip` is:
        a handful of dict lookups, no I/O and no clock. The numbers arrive on
        `row.raw` from the `ai_state` merge in `data_feed`, off the Qt thread.
        """
        if not isinstance(row, SetupRow):
            return None
        try:
            return compression_chip.read_row(row.raw)
        except Exception:
            return None

    @staticmethod
    def _earnings_badge(row) -> str:
        """S10b: "ER 5d" for a SHORT 0-14 days before earnings, else "". Never raises."""
        if not isinstance(row, SetupRow):
            return ""
        try:
            return earnings_warning.badge_text(row.days_to_earnings, row.side)
        except Exception:
            return ""

    def _earnings_tooltip(self, index) -> str:
        """The bucket cell's earnings warning line for a SHORT (S10b). Memory only."""
        key = _COLUMN_KEYS[index.column()] if index.column() < len(_COLUMN_KEYS) else ""
        if key != "bucket":
            return ""
        row = index.data(ROW_ROLE)
        if not isinstance(row, SetupRow):
            return ""
        try:
            return earnings_warning.short_into_earnings(
                row.days_to_earnings, row.side, earnings_warning.cached_stat()
            )
        except Exception:
            return ""

    def _wrong_side_tooltip(self, index) -> str:
        """The bucket cell's extra line when the row is on the wrong side.

        ADDED to the tooltip the cell already has (its bucket label), never
        instead of it - WS-WS hides nothing, and that includes text.
        """
        key = _COLUMN_KEYS[index.column()] if index.column() < len(_COLUMN_KEYS) else ""
        if key != "bucket":
            return ""
        return avwape_side.tooltip_text(self._wrong_side_read(index.data(ROW_ROLE)))

    def _compression_tooltip(self, index) -> str:
        """The bucket cell's extra line when the scan flagged the row compressed.

        The same shape as the wrong-side line: an ADDITION, never a replacement.
        """
        key = _COLUMN_KEYS[index.column()] if index.column() < len(_COLUMN_KEYS) else ""
        if key != "bucket":
            return ""
        return compression_chip.tooltip_text(self._compression_read(index.data(ROW_ROLE)))

    def _bucket_tooltip(self, index) -> str:
        """Everything the bucket cell has to say, its own label first.

        Its bucket label, then WS-WS's wrong-side line, then PCT-3's compression
        line - each one only when it has something to say. `""` when neither
        badge applies, so a plain cell still falls through to Qt's own tooltip
        handling exactly as it did before either packet existed.
        """
        extra = [
            text
            for text in (
                self._wrong_side_tooltip(index),
                self._compression_tooltip(index),
                self._earnings_tooltip(index),
            )
            if text
        ]
        if not extra:
            return ""
        existing = str(index.data(Qt.ItemDataRole.ToolTipRole) or "").strip()
        return "\n".join([existing, *extra] if existing else extra)

    def _decision_tooltip(self, index) -> str:
        key = _COLUMN_KEYS[index.column()] if index.column() < len(_COLUMN_KEYS) else ""
        if key not in {"favorite", "dislike"}:
            return ""
        row = index.data(ROW_ROLE)
        if not isinstance(row, SetupRow):
            return ""
        decisions = self._decisions(row)
        lines: list[str] = []
        if key == "favorite":
            if self._is_focus(row):
                lines.append("In Focus")
            for kind, stamp in self._liked_today(decisions):
                label = _LIKE_LABELS.get(str(kind), str(kind))
                when = _clock_text(stamp)
                lines.append(f"Liked today ({label}, {when})" if when else f"Liked today ({label})")
        else:
            for kind, stamp in self._rejected_today(decisions):
                label = _REJECT_LABELS.get(str(kind), str(kind))
                when = _clock_text(stamp)
                suffix = _REJECT_SUFFIXES.get(str(kind), "")
                lines.append(" ".join(part for part in (label, when, suffix) if part))
        return "\n".join(lines)

    def sizeHint(self, option, index):  # noqa: N802 (Qt override)
        size = super().sizeHint(option, index)
        width = size.width()
        key = _COLUMN_KEYS[index.column()] if index.column() < len(_COLUMN_KEYS) else ""
        if key == "bucket":
            read = self._wrong_side_read(index.data(ROW_ROLE))
            if read is not None and read.wrong:
                # `fit_columns` sizes a column by asking this, so a column that
                # never asks for the second chip never gets the room to paint
                # it. Width only: the row height is the setups height either way
                # (G2b pins that).
                width += _chip_width(option.font, avwape_side.WRONG_SIDE_LABEL) + _CHIP_GAP
            compression = self._compression_read(index.data(ROW_ROLE))
            if compression is not None and compression.flag:
                # PCT-3: the same reasoning for the third chip.
                width += _chip_width(option.font, compression_chip.COMPRESSED_LABEL) + _CHIP_GAP
            earnings_badge = self._earnings_badge(index.data(ROW_ROLE))
            if earnings_badge:
                width += _chip_width(option.font, earnings_badge) + _CHIP_GAP
        return QSize(width, max(size.height(), _ROW_HEIGHT))

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        row = index.data(ROW_ROLE)
        key = _COLUMN_KEYS[index.column()] if index.column() < len(_COLUMN_KEYS) else ""
        rect = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_setup = isinstance(row, SetupRow)
        bucket = row.bucket.strip().lower() if is_setup else ""
        is_favorite = bucket in {"favorite_setup", "high_conviction"}
        is_study = "study" in bucket

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Background: alternating base, favorite tint, then selection on top.
        base = theme.color("bg_elevated") if index.row() % 2 else theme.color("bg_panel")
        painter.fillRect(rect, QColor(base))
        if is_favorite and not selected:
            painter.fillRect(rect, _alpha("favorite", 24))
        if selected:
            painter.fillRect(rect, QColor(theme.color("selection")))

        # Hairline row separator (calmer than a full grid).
        painter.setPen(QPen(_alpha("border", 90), 1))
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())

        # Favorite accent stripe on the leading column.
        if is_favorite and index.column() == 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.color("favorite")))
            painter.drawRoundedRect(QRect(rect.left() + 2, rect.top() + 6, 3, rect.height() - 12), 1.5, 1.5)

        if key == "favorite" and is_setup:
            # WS-SX: in Focus OR liked today - one boolean, one filled star.
            liked = bool(self._liked_today(self._decisions(row)))
            self._favorite_star(painter, option, rect, self._is_focus(row) or liked)
        elif key == "dislike" and is_setup:
            rejected = bool(self._rejected_today(self._decisions(row)))
            self._dislike_mark(painter, option, rect, rejected)
        elif key == "side" and is_setup and row.side in {"LONG", "SHORT"}:
            self._chip(painter, option, rect, row.side, "long" if row.side == "LONG" else "short")
        elif key == "bucket" and is_setup and row.bucket:
            bucket_chip = self._chip(
                painter, option, rect, row.bucket_label, _bucket_token(bucket), study=is_study
            )
            # WS-WS (WISHLIST 9): a LONG whose close sits under its AVWAPE, or a
            # SHORT whose close sits over it, is BADGED - after the bucket chip,
            # never over it. Display only: the row is still here, still in the
            # same place, still with the same score.
            last_chip = bucket_chip
            read = self._wrong_side_read(row)
            if read is not None and read.wrong:
                last_chip = (
                    self._chip(
                        painter,
                        option,
                        rect,
                        avwape_side.WRONG_SIDE_LABEL,
                        "caution",
                        study=is_study,
                        after=bucket_chip,
                    )
                    or last_chip
                )
            # PCT-3 (trader 2026-09-15): the scan has always docked a compressed
            # row's score in silence. The chip says so - AFTER the bucket and
            # wrong-side chips, never over them. Display only: nothing is
            # hidden, nothing is re-ordered, no score moves.
            compression = self._compression_read(row)
            if compression is not None and compression.flag:
                last_chip = (
                    self._chip(
                        painter,
                        option,
                        rect,
                        compression_chip.COMPRESSED_LABEL,
                        compression_chip.COMPRESSED_TOKEN,
                        study=is_study,
                        after=last_chip,
                    )
                    or last_chip
                )
            # S10b: a SHORT 0-14 days before earnings is badged, last. Display only.
            earnings_badge = self._earnings_badge(row)
            if earnings_badge:
                self._chip(
                    painter, option, rect, earnings_badge, "caution",
                    study=is_study, after=last_chip,
                )
        elif key == "score" and is_setup and row.score is not None:
            self._score(painter, option, rect, row.score, selected)
        else:
            self._text(painter, option, rect, index, key, is_study, selected)

        painter.restore()

    def _favorite_star(self, painter, option, rect, focused: bool) -> None:
        """Clickable favorite column: filled gold ★ for focus picks, hollow ☆ otherwise."""
        painter.setFont(_resized(option.font, 2.0))
        painter.setPen(QColor(theme.color("favorite")) if focused else _alpha("text_secondary", 150))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "★" if focused else "☆")

    def _dislike_mark(self, painter, option, rect, rejected: bool = False) -> None:
        """Clickable dislike column: ✕ prompts for a why and logs it for AI review.

        WS-SX: BRIGHT RED (`reject_today`, solid) once the trader has rejected
        this name today - vetoed, disliked, passed, or clicked away from its M5
        alert. Otherwise the same dimmed mark it has always been.
        """
        painter.setFont(_resized(option.font, 1.0))
        painter.setPen(QColor(theme.color("reject_today")) if rejected else _alpha("short", 140))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "✕")

    def _text(self, painter, option, rect, index, key, is_study, selected) -> None:
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if not text:
            return
        if selected:
            color = QColor(theme.color("text_primary"))
        elif is_study:
            color = QColor(theme.color("text_secondary"))
        else:
            fg = index.data(Qt.ItemDataRole.ForegroundRole)
            color = QColor(fg) if isinstance(fg, QColor) else QColor(theme.color("text_primary"))

        font = QFont(option.font)
        if key == "symbol":
            font.setBold(True)
        painter.setFont(font)

        align = index.data(Qt.ItemDataRole.TextAlignmentRole)
        align = int(align) if align else int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        text_rect = rect.adjusted(_PAD, 0, -_PAD, 0)
        elided = QFontMetrics(font).elidedText(str(text), Qt.TextElideMode.ElideRight, text_rect.width())
        painter.setPen(color)
        painter.drawText(text_rect, align, elided)

    def _chip(self, painter, option, rect, text, token, study=False, after=None):
        """One pill. Returns the rect it took, or None when it did not fit.

        `after` is another chip's rect in the same cell: the pill starts a gap
        past its right edge instead of at the cell's padding (WS-WS). A second
        chip with no room left is not drawn - a coloured sliver says nothing,
        and the tooltip still carries the whole sentence.
        """
        color = QColor(theme.color(token))
        font = _resized(option.font, -1.0, minimum=7.5)
        font.setBold(True)
        metrics = QFontMetrics(font)
        chip_h = min(_CHIP_HEIGHT, rect.height() - 8)
        left = rect.left() + _PAD
        if after is not None:
            left = after.right() + _CHIP_GAP
        # Unchanged for the first chip: with `left == rect.left() + _PAD` this is
        # exactly the `rect.width() - _PAD - 4` it has always been.
        available = rect.width() - 4 - (left - rect.left())
        chip_w = min(metrics.horizontalAdvance(text) + 20, available)
        if after is not None and chip_w < _MIN_CHIP_WIDTH:
            return None
        chip_rect = QRect(left, rect.top() + (rect.height() - chip_h) // 2, chip_w, chip_h)

        painter.setBrush(_alpha(token, 36))
        painter.setPen(QPen(_alpha(token, 130), 1))
        painter.drawRoundedRect(chip_rect, chip_h / 2, chip_h / 2)

        painter.setFont(font)
        painter.setPen(color)
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, chip_rect.width() - 12)
        painter.setOpacity(0.85 if study else 1.0)
        painter.drawText(chip_rect, int(Qt.AlignmentFlag.AlignCenter), elided)
        painter.setOpacity(1.0)
        return chip_rect

    def _score(self, painter, option, rect, score, selected) -> None:
        token = _score_token(score)
        # Number, sitting above the bar.
        font = QFont(option.font)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(theme.color("text_primary") if selected else theme.color(token)))
        number_rect = rect.adjusted(_PAD, 0, -_PAD, -8)
        painter.drawText(number_rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), f"{score:.1f}")

        # Track + proportional fill.
        track = QRect(rect.left() + _PAD, rect.bottom() - 9, rect.width() - 2 * _PAD, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha("border", 130))
        painter.drawRoundedRect(track, 2, 2)
        fraction = max(0.0, min(1.0, score / 100.0))
        fill_w = int(track.width() * fraction)
        if fill_w > 0:
            painter.setBrush(_alpha(token, 210))
            painter.drawRoundedRect(QRect(track.left(), track.top(), fill_w, track.height()), 2, 2)


def _chip_width(font: QFont, text: str) -> int:
    """What one pill of this text costs, in the chip's own font (WS-WS)."""
    chip_font = _resized(font, -1.0, minimum=7.5)
    chip_font.setBold(True)
    return QFontMetrics(chip_font).horizontalAdvance(text) + 20


def _bucket_token(bucket: str) -> str:
    normalized = bucket.strip().lower()
    if normalized in {"favorite_setup", "high_conviction"}:
        return "favorite"
    if normalized == "near_favorite_zone":
        return "near"
    if "study" in normalized:
        return "study"
    return "neutral"


def _score_token(score: float) -> str:
    if score >= 80:
        return "long"
    if score < 45:
        return "caution"
    return "accent"


def _alpha(token: str, alpha: int) -> QColor:
    color = QColor(theme.color(token))
    color.setAlpha(alpha)
    return color
