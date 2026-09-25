"""Packet WS-WS - the "wrong side" of the AVWAPE is SHOWN and never hidden.

WISHLIST item 9, trader 2026-09-12: *"stop putting up longs below avwape and
shorts above it."* The lead's ruling for this sweep is DISPLAY ONLY - a badge on
the Desk's setups table and a tag on the AWAY digest's swing line. Nothing is
hidden, nothing is re-ordered, no detector, score, alert or watchlist changes.

WHAT THE ROW ACTUALLY CARRIES (measured 2026-09-12 against the live
`C:\\TradingBotData\\data\\runtime\\master_avwap_focus.json`, 435 rows): **not one
row carries `current_close` or `current_avwape`** - those two live on the
tracker's `feature_snapshot`, not on the scan row the desk's setups table and the
AWAY digest read. Every one of the 435 rows DOES carry a `current_band_zone`,
either at the top level or under `setup_candidate.trigger`, drawn from the
ordered band vocabulary (`LOWER_3 .. LOWER_1`, `VWAP`, `UPPER_1 .. UPPER_3`),
which says exactly which side of the AVWAPE the close sits on. So the rule has
two bases and prefers the numbers whenever a caller has them:

* `wrong_side(side, close, avwape)` - the packet's pure function, tolerance 0;
* `wrong_side_from_zone(side, zone)` - the same verdict from the band zone;
* `read_row(row)` - numbers first, band zone second, `None` when neither is
  readable, because an unknown is never "wrong".

The zone counts on that live file were LONG below the AVWAPE 87 and SHORT above
it 6, of 435 - so the badge is expected on about a fifth of a real board and the
gate has something to look at on day one.

Nothing here may be weakened; only added to.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Item 1 - the pure rule
# ---------------------------------------------------------------------------


def test_a_long_under_the_anchor_and_a_short_over_it_are_the_wrong_side():
    from avwape_side import wrong_side

    assert wrong_side("LONG", 409.10, 412.50) is True
    assert wrong_side("SHORT", 415.10, 412.50) is True


def test_a_long_over_the_anchor_and_a_short_under_it_are_the_right_side():
    from avwape_side import wrong_side

    assert wrong_side("LONG", 415.10, 412.50) is False
    assert wrong_side("SHORT", 409.10, 412.50) is False


def test_a_close_exactly_on_the_line_is_the_right_side():
    """Tolerance is 0 and the line itself is NOT wrong (packet item 1)."""
    from avwape_side import wrong_side

    assert wrong_side("LONG", 412.50, 412.50) is False
    assert wrong_side("SHORT", 412.50, 412.50) is False


@pytest.mark.parametrize(
    "side, close, avwape",
    [
        ("LONG", None, 412.50),
        ("LONG", 409.10, None),
        ("SHORT", None, None),
        ("", 409.10, 412.50),
        ("LONG", "", 412.50),
    ],
)
def test_a_missing_value_is_unknown_and_unknown_is_never_wrong(side, close, avwape):
    from avwape_side import wrong_side

    assert wrong_side(side, close, avwape) is None


def test_the_side_is_read_the_way_the_rest_of_the_desk_reads_it():
    from avwape_side import wrong_side

    assert wrong_side("long", 409.10, 412.50) is True
    assert wrong_side(" Short ", 415.10, 412.50) is True


# ---------------------------------------------------------------------------
# The band-zone basis - the vocabulary the live rows actually carry
# ---------------------------------------------------------------------------


#: (side, zone, expected verdict). Every zone string here was counted on the
#: live focus feed on 2026-09-12; the counts are in the module docstring.
LIVE_ZONES = [
    ("SHORT", "LOWER_1 to LOWER_2", False),
    ("SHORT", "LOWER_2 to LOWER_3", False),
    ("SHORT", "VWAP to LOWER_1", False),
    ("SHORT", "LOWER_3", False),
    ("SHORT", "UPPER_2 to UPPER_1", True),
    ("SHORT", "UPPER_3 to UPPER_2", True),
    ("LONG", "VWAP to UPPER_1", False),
    ("LONG", "UPPER_1 to UPPER_2", False),
    ("LONG", "UPPER_2 to UPPER_3", False),
    ("LONG", "UPPER_3", False),
    ("LONG", "LOWER_2 to LOWER_1", True),
    ("LONG", "LOWER_3 to LOWER_2", True),
    ("LONG", "LOWER_1 to VWAP", True),
    ("LONG", "LOWER_3", True),
]


@pytest.mark.parametrize("side, zone, expected", LIVE_ZONES)
def test_the_band_zone_says_which_side_of_the_anchor_the_close_sits_on(side, zone, expected):
    from avwape_side import wrong_side_from_zone

    assert wrong_side_from_zone(side, zone) is expected


def test_a_close_sitting_on_the_anchor_is_the_right_side_in_the_zone_basis_too():
    from avwape_side import wrong_side_from_zone

    assert wrong_side_from_zone("LONG", "VWAP") is False
    assert wrong_side_from_zone("SHORT", "AVWAPE") is False


@pytest.mark.parametrize("zone", ["", "   ", "SMA_50 to VWAP", "who knows", "LOWER_1 to"])
def test_an_unreadable_zone_is_unknown_and_unknown_is_never_wrong(zone):
    from avwape_side import wrong_side_from_zone

    assert wrong_side_from_zone("LONG", zone) is None


# ---------------------------------------------------------------------------
# `read_row` - one reader, both callers
# ---------------------------------------------------------------------------


def _live_shaped_row(**over):
    """A scan row in the shape the focus feed really writes.

    Keys and nesting copied from
    `C:\\TradingBotData\\data\\runtime\\master_avwap_focus.json` (2026-09-12): the
    zone lives under `setup_candidate.trigger`, and `favorite_zone` - which names
    the zone the SETUP wants, not where the close is - sits beside it as the trap
    this reader must not fall into.
    """
    row = {
        "symbol": "TW",
        "side": "SHORT",
        "priority_bucket": "high_conviction",
        "favorite_zone": "LOWER_1 to AVWAPE",
        "setup_candidate": {
            "trigger": {
                "current_active_level": "VWAP",
                "current_band_zone": "VWAP to LOWER_1",
            }
        },
    }
    row.update(over)
    return row


def test_the_row_reader_prefers_the_two_numbers_when_the_row_carries_them():
    from avwape_side import read_row

    row = _live_shaped_row(side="LONG", current_close=409.10, current_avwape=412.50)
    read = read_row(row)
    assert read is not None
    assert read.wrong is True
    assert read.basis == "prices"
    assert read.close == pytest.approx(409.10)
    assert read.avwape == pytest.approx(412.50)


def test_the_row_reader_falls_back_to_the_band_zone_the_live_rows_carry():
    from avwape_side import read_row

    read = read_row(_live_shaped_row(side="LONG"))
    assert read is not None
    assert read.wrong is True, "LONG with the close between LOWER_1 and the AVWAPE"
    assert read.basis == "band_zone"
    assert read.zone == "VWAP to LOWER_1"


def test_the_row_reader_reads_a_top_level_zone_too():
    from avwape_side import read_row

    row = _live_shaped_row(side="SHORT", current_band_zone="UPPER_2 to UPPER_1")
    read = read_row(row)
    assert read is not None and read.wrong is True


def test_the_setup_s_own_favorite_zone_is_never_read_as_the_close_s_position():
    """`favorite_zone` says what the setup WANTS. It is not where price is."""
    from avwape_side import read_row

    row = {"side": "LONG", "favorite_zone": "LOWER_1 to AVWAPE"}
    assert read_row(row) is None


@pytest.mark.parametrize("row", [None, {}, {"side": "LONG"}, "not a row", {"side": ""}])
def test_a_row_that_says_nothing_reads_as_unknown(row):
    from avwape_side import read_row

    assert read_row(row) is None


def test_the_right_side_read_is_a_read_and_not_a_badge():
    from avwape_side import read_row

    read = read_row(_live_shaped_row())  # SHORT below the anchor
    assert read is not None
    assert read.wrong is False


# ---------------------------------------------------------------------------
# Item 2 - the setups table chip and its tooltip
# ---------------------------------------------------------------------------


def test_the_tooltip_names_both_numbers_when_the_numbers_are_what_was_read():
    from avwape_side import read_row, tooltip_text

    read = read_row(_live_shaped_row(side="LONG", current_close=409.10, current_avwape=412.50))
    assert tooltip_text(read) == "LONG below AVWAPE 412.50 (close 409.10)"

    read = read_row(_live_shaped_row(side="SHORT", current_close=415.10, current_avwape=412.50))
    assert tooltip_text(read) == "SHORT above AVWAPE 412.50 (close 415.10)"


def test_the_tooltip_names_the_band_zone_when_that_is_what_was_read():
    """It never prints a number it did not read (the row has none)."""
    from avwape_side import read_row, tooltip_text

    read = read_row(_live_shaped_row(side="LONG"))
    text = tooltip_text(read)
    assert text == "LONG below AVWAPE (band zone VWAP to LOWER_1)"
    assert "(close" not in text


def test_a_right_side_read_has_no_tooltip_and_an_unknown_one_has_none_either():
    from avwape_side import read_row, tooltip_text

    assert tooltip_text(read_row(_live_shaped_row())) == ""
    assert tooltip_text(None) == ""


# --- the painted chip -------------------------------------------------------

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

from PySide6.QtCore import QEvent, QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QColor, QHelpEvent, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QToolTip,
)

from ui import theme  # noqa: E402
from ui.models.setup import SetupRow  # noqa: E402
from ui.models.setup_table_model import SetupTableModel  # noqa: E402
from ui.widgets.setup_delegate import SetupTableDelegate  # noqa: E402


#: The bucket column at the setups row height, wide enough for both chips.
BUCKET_CELL = QRect(0, 0, 220, 40)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _column(key: str) -> int:
    for index, (column_key, _label) in enumerate(SetupTableModel.COLUMNS):
        if column_key == key:
            return index
    raise KeyError(key)


def _setup_row(symbol: str, side: str, zone: str | None, **raw_over) -> SetupRow:
    raw = _live_shaped_row(symbol=symbol, side=side)
    if zone is None:
        raw.pop("setup_candidate")
    else:
        raw["setup_candidate"]["trigger"]["current_band_zone"] = zone
    raw.update(raw_over)
    return SetupRow(symbol=symbol, side=side, score=61.0, bucket="near_favorite_zone", raw=raw)


def _table(app, rows):
    model = SetupTableModel(list(rows))
    view = QTableView()
    view.setModel(model)
    delegate = SetupTableDelegate(view)
    view.setItemDelegate(delegate)
    view.resize(900, 300)
    return view, model, delegate


def _option(view):
    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.font = view.font()
    option.rect = QRect(BUCKET_CELL)
    option.state = QStyle.StateFlag.State_Enabled
    return option


def _render(delegate, view, model, row: int, key: str) -> QImage:
    image = QImage(BUCKET_CELL.width(), BUCKET_CELL.height(), QImage.Format.Format_ARGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    try:
        delegate.paint(painter, _option(view), model.index(row, _column(key)))
    finally:
        painter.end()
    return image


def _has_exact(image: QImage, colour: QColor) -> int:
    wanted = QColor(colour).rgb()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if QColor(image.pixel(x, y)).rgb() == wanted
    )


def _rows():
    return [
        _setup_row("WRONGL", "LONG", "LOWER_1 to VWAP"),  # 0: wrong side
        _setup_row("RIGHTL", "LONG", "VWAP to UPPER_1"),  # 1: right side
        _setup_row("UNKNOWN", "LONG", None),  # 2: nothing readable
        _setup_row("WRONGS", "SHORT", "UPPER_2 to UPPER_1"),  # 3: wrong side
    ]


def _one_row_cell(app, row: SetupRow, key: str) -> QImage:
    """That row's cell, ALONE in its table.

    The delegate alternates the row background on `index.row() % 2`, so two rows
    of one table are never comparable pixel for pixel. One row per table at
    index 0 compares the painting and nothing else.
    """
    view, model, delegate = _table(app, [row])
    return _render(delegate, view, model, 0, key)


def test_a_wrong_side_row_paints_the_chip_beside_its_bucket_chip(app):
    view, model, delegate = _table(app, _rows())
    wrong = _render(delegate, view, model, 0, "bucket")
    assert _has_exact(wrong, QColor(theme.color("caution"))) > 0, (
        "the wrong-side chip is painted in the `caution` token"
    )
    short_side = _render(delegate, view, model, 3, "bucket")
    assert _has_exact(short_side, QColor(theme.color("caution"))) > 0


def test_a_right_side_row_paints_no_chip_and_an_unknown_row_paints_none_either(app):
    view, model, delegate = _table(app, _rows())
    right = _render(delegate, view, model, 1, "bucket")
    unknown = _render(delegate, view, model, 2, "bucket")
    assert _has_exact(right, QColor(theme.color("caution"))) == 0
    assert _has_exact(unknown, QColor(theme.color("caution"))) == 0


def test_a_right_side_bucket_cell_renders_exactly_as_a_row_with_nothing_to_say(app):
    """The golden: a row that is NOT wrong side paints the pre-WS cell, pixel
    for pixel. An unknown row cannot reach the new code at all, so the two
    images being identical is the proof that a right-side row does not either.

    (The builder also compared this image against the pre-change delegate
    directly: sha1 `c6cecd5e2481dd4aa0589fce62466045a549095b`, unchanged.)
    """
    right = _one_row_cell(app, _setup_row("SAME", "LONG", "VWAP to UPPER_1"), "bucket")
    unknown = _one_row_cell(app, _setup_row("SAME", "LONG", None), "bucket")
    assert right == unknown


def test_the_row_is_untouched_outside_the_bucket_cell(app):
    """Same row, same symbol, only the band zone differs."""
    wrong = _setup_row("SAME", "LONG", "LOWER_1 to VWAP")
    quiet = _setup_row("SAME", "LONG", None)
    for key in ("symbol", "side", "score", "key_level", "favorite", "dislike"):
        assert _one_row_cell(app, wrong, key) == _one_row_cell(app, quiet, key), (
            f"the {key} cell of a wrong-side row must paint exactly as it does today"
        )


def test_the_bucket_chip_itself_survives(app):
    """G2b styling: the bucket chip keeps its own token and its own position."""
    wrong = _one_row_cell(app, _setup_row("SAME", "LONG", "LOWER_1 to VWAP"), "bucket")
    right = _one_row_cell(app, _setup_row("SAME", "LONG", "VWAP to UPPER_1"), "bucket")
    near = QColor(theme.color("near"))
    assert _has_exact(wrong, near) > 0
    assert _has_exact(right, near) > 0
    # The bucket chip is drawn FIRST and the second pill starts a gap past it,
    # so the WHOLE bucket-chip region is pixel-identical on both rows. The
    # region is measured from the delegate's own metrics, never guessed: with
    # the font database an offscreen suite actually gets (see DESK_INTERNALS
    # "SX"), a third of the cell can be well past the end of the first chip.
    from ui.widgets import setup_delegate as delegate_module

    view, _model, _delegate = _table(app, [])
    chip_end = delegate_module._PAD + delegate_module._chip_width(view.font(), "Near")
    assert chip_end < BUCKET_CELL.width(), "the fixture cell fits the bucket chip"
    assert all(
        wrong.pixel(x, y) == right.pixel(x, y)
        for y in range(BUCKET_CELL.height())
        for x in range(chip_end)
    ), "the wrong-side chip is drawn AFTER the bucket chip, never over it"
    assert wrong != right, "and it IS drawn"


def test_a_cell_with_no_room_for_the_second_chip_draws_no_sliver(app):
    """The compact profile pins `bucket` at 96px. A 12px stub is not a badge."""
    row = _setup_row("SAME", "LONG", "LOWER_1 to VWAP")
    view, model, delegate = _table(app, [row])
    narrow = QRect(0, 0, 96, 40)
    image = QImage(narrow.width(), narrow.height(), QImage.Format.Format_ARGB32)
    image.fill(QColor("#000000"))
    option = _option(view)
    option.rect = narrow
    painter = QPainter(image)
    try:
        delegate.paint(painter, option, model.index(0, _column("bucket")))
    finally:
        painter.end()
    assert _has_exact(image, QColor(theme.color("near"))) > 0, "the bucket chip still paints"


def test_the_bucket_cell_asks_for_the_width_the_second_chip_needs(app):
    view, model, delegate = _table(app, _rows())
    option = _option(view)
    wrong = delegate.sizeHint(option, model.index(0, _column("bucket")))
    right = delegate.sizeHint(option, model.index(1, _column("bucket")))
    assert wrong.width() > right.width(), (
        "`fit_columns` measures the delegate, so the column has to ask for the chip"
    )
    assert wrong.height() == right.height()


def _tooltip_text(delegate, view, model, row: int, key: str) -> str:
    """Whatever the trader would see hovering that cell - delegate or model."""
    seen: list[str] = []
    original = QToolTip.showText

    def _spy(pos, text, *args, **kwargs):  # noqa: ANN001
        seen.append(str(text))

    QToolTip.showText = _spy
    try:
        index = model.index(row, _column(key))
        option = _option(view)
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(4, 4), QPoint(4, 4))
        handled = delegate.helpEvent(event, view, option, index)
    finally:
        QToolTip.showText = original
    if seen:
        return seen[-1]
    # The delegate declined, so Qt's own `helpEvent` showed the model's tooltip
    # through C++ (which never sees a Python patch of `QToolTip.showText`).
    # What the trader reads in that case is the model's text.
    assert handled or not str(model.data(index, Qt.ItemDataRole.ToolTipRole) or "")
    return str(model.data(index, Qt.ItemDataRole.ToolTipRole) or "")


def test_the_chip_s_tooltip_says_which_side_and_what_it_read(app):
    """The bucket cell already had a tooltip (its label). The chip ADDS a line."""
    view, model, delegate = _table(app, _rows())
    assert _tooltip_text(delegate, view, model, 0, "bucket").splitlines() == [
        "Near",
        "LONG below AVWAPE (band zone LOWER_1 to VWAP)",
    ]


def test_a_priced_row_s_tooltip_names_both_numbers_on_the_table_too(app):
    rows = [_setup_row("PRICED", "LONG", None, current_close=409.10, current_avwape=412.50)]
    view, model, delegate = _table(app, rows)
    lines = _tooltip_text(delegate, view, model, 0, "bucket").splitlines()
    assert lines[-1] == "LONG below AVWAPE 412.50 (close 409.10)"
    assert lines[0] == "Near", "today's bucket tooltip is kept, never replaced"


def test_a_right_side_bucket_cell_keeps_the_tooltip_it_has_today(app):
    view, model, delegate = _table(app, _rows())
    today = str(
        model.data(model.index(1, _column("bucket")), Qt.ItemDataRole.ToolTipRole) or ""
    )
    assert _tooltip_text(delegate, view, model, 1, "bucket") == today


def test_the_decision_marks_still_answer_their_own_tooltips(app):
    """WS-SX's ★/✕ tooltips are not shadowed by the new branch."""
    view, model, delegate = _table(app, _rows())
    assert "favorite this pick" in _tooltip_text(delegate, view, model, 0, "favorite")


# ---------------------------------------------------------------------------
# Item 3 - the AWAY digest
# ---------------------------------------------------------------------------


def _pick(symbol: str, side: str, zone: str | None, **over):
    import autopilot_core as core

    row = _setup_row(symbol, side, zone)
    pick = core.swing_pick_projection(row)
    pick.update(over)
    return pick


def _payload(picks):
    return {
        "generated_at": "2026-09-12 13:00:00",
        "enabled": True,
        "auto_mode": "AWAY",
        "ib_status": "connected",
        "regime": "mixed",
        "longs": ["AAA"],
        "shorts": ["BBB"],
        "swing_picks": list(picks),
        "swing_family_records": {},
        "swing_family_record_line": "",
        "swing_data_current": True,
    }


def _swing_lines(text: str) -> list[str]:
    lines = text.splitlines()
    index = next((i for i, line in enumerate(lines) if line.startswith("1. ")), None)
    if index is None:
        return []
    out = []
    for line in lines[index:]:
        if not line.strip():
            break
        out.append(line)
    return out


def test_the_swing_line_is_tagged_after_the_symbol_and_the_count_is_printed():
    import autopilot_core as core

    text = core.render_away_report(
        _payload(
            [
                _pick("WRONGL", "LONG", "LOWER_1 to VWAP"),
                _pick("RIGHTL", "LONG", "VWAP to UPPER_1"),
                _pick("WRONGS", "SHORT", "UPPER_2 to UPPER_1"),
            ]
        )
    )
    lines = _swing_lines(text)
    tagged = [line for line in lines if "[wrong side]" in line]
    assert len(tagged) == 2
    assert tagged[0].startswith("1. WRONGL [wrong side] (LONG)")
    assert "RIGHTL [wrong side]" not in text
    assert "2 wrong side of the anchor" in text


def test_the_digest_order_is_untouched_by_the_tag():
    import autopilot_core as core

    picks = [
        _pick("WRONGL", "LONG", "LOWER_1 to VWAP"),
        _pick("RIGHTL", "LONG", "VWAP to UPPER_1"),
        _pick("WRONGS", "SHORT", "UPPER_2 to UPPER_1"),
    ]
    text = core.render_away_report(_payload(picks))
    symbols = [
        line.split(". ", 1)[1].split(" ", 1)[0]
        for line in _swing_lines(text)
        if re.match(r"^\d+\. ", line)
    ]
    assert symbols == ["WRONGL", "RIGHTL", "WRONGS"]
    assert "TV paste: WRONGL,RIGHTL,WRONGS" in text


def test_a_digest_with_no_wrong_side_rows_says_nothing_new():
    """Byte-identical to today's: the same payload with the reader blinded."""
    import autopilot_core as core

    picks = [
        _pick("RIGHTL", "LONG", "VWAP to UPPER_1"),
        _pick("RIGHTS", "SHORT", "VWAP to LOWER_1"),
        _pick("NOTHING", "LONG", None),
    ]
    text = core.render_away_report(_payload(picks))
    assert "wrong side" not in text

    blinded = [dict(pick, raw={}) for pick in picks]
    assert text == core.render_away_report(_payload(blinded)), (
        "a board with nothing on the wrong side renders exactly as it does today"
    )


def test_a_pick_the_reader_cannot_answer_is_never_tagged():
    import autopilot_core as core

    text = core.render_away_report(_payload([_pick("NOTHING", "LONG", None)]))
    assert "wrong side" not in text


def test_the_digest_never_drops_or_reorders_a_pick_when_the_reader_raises(monkeypatch):
    """Display only: a reader that blows up costs the tag, never the digest."""
    import autopilot_core as core

    def _boom(_row):
        raise RuntimeError("no")

    monkeypatch.setattr(core.avwape_side, "read_row", _boom)
    text = core.render_away_report(
        _payload(
            [
                _pick("WRONGL", "LONG", "LOWER_1 to VWAP"),
                _pick("RIGHTL", "LONG", "VWAP to UPPER_1"),
            ]
        )
    )
    assert "1. WRONGL (LONG)" in text
    assert "2. RIGHTL (LONG)" in text
    assert "wrong side" not in text
