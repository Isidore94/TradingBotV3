from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from ui import theme
from ui.models.setup import SetupRow
from ui.models.setup_table_model import ROW_ROLE, SetupTableModel


_COLUMN_KEYS = [key for key, _label in SetupTableModel.COLUMNS]
_ROW_HEIGHT = 40
_CHIP_HEIGHT = 22
_PAD = 10


class SetupTableDelegate(QStyledItemDelegate):
    """Paints the setups table as a scannable surface: side/bucket chips, a
    score bar, a favorite accent stripe, and de-emphasized study rows.

    All colors come from the active theme tokens so light/dark both look right.
    """

    def sizeHint(self, option, index):  # noqa: N802 (Qt override)
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(size.height(), _ROW_HEIGHT))

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

        if key == "side" and is_setup and row.side in {"LONG", "SHORT"}:
            self._chip(painter, option, rect, row.side, "long" if row.side == "LONG" else "short")
        elif key == "bucket" and is_setup and row.bucket:
            self._chip(painter, option, rect, row.bucket_label, _bucket_token(bucket), study=is_study)
        elif key == "score" and is_setup and row.score is not None:
            self._score(painter, option, rect, row.score, selected)
        else:
            self._text(painter, option, rect, index, key, is_study, selected)

        painter.restore()

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

    def _chip(self, painter, option, rect, text, token, study=False) -> None:
        color = QColor(theme.color(token))
        font = QFont(option.font)
        font.setBold(True)
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1.0))
        metrics = QFontMetrics(font)
        chip_h = min(_CHIP_HEIGHT, rect.height() - 8)
        chip_w = min(metrics.horizontalAdvance(text) + 20, rect.width() - _PAD - 4)
        chip_rect = QRect(rect.left() + _PAD, rect.top() + (rect.height() - chip_h) // 2, chip_w, chip_h)

        painter.setBrush(_alpha(token, 36))
        painter.setPen(QPen(_alpha(token, 130), 1))
        painter.drawRoundedRect(chip_rect, chip_h / 2, chip_h / 2)

        painter.setFont(font)
        painter.setPen(color)
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, chip_rect.width() - 12)
        painter.setOpacity(0.85 if study else 1.0)
        painter.drawText(chip_rect, int(Qt.AlignmentFlag.AlignCenter), elided)
        painter.setOpacity(1.0)

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
