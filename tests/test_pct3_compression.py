r"""Packet PCT-3 - compression is MEASURED before it is tuned.

`docs/PULLBACK_COMPRESSION_TRENDLINE_PLAN.md` section 6. Trader, 2026-09-15:
*"we need a way to measure for compression as the most COMMON veto I have is a
compression veto. we need to fine tune this so we stop getting so many compressed
picks."* The lead's answer for this packet is **measure + chip + a read-only
calibration report** - no threshold, penalty or detector change. So every test
here either shows a number the scan already computed and threw away, or reads a
number back; none of them moves a score.

The five items and where they are pinned below:

1. **Copy-through.** `summarize_anchor_compression` computes `compression_score`
   and three ATR ratios on every priority row and they die inside the function
   (`legacy.py:4927-5005`; only `compression_flag / penalty / note` reach the row
   at :26989-26999). They must join the row, the `ai_state` symbol entry and
   `build_tracker_setup_record`'s `compression_summary`, with
   `compression_rule_version = "anchor_compression_v1"`. The VALUES here are not
   typed in - the test calls `summarize_anchor_compression` on the same slice the
   scan used, so a wrong number fails as a wrong number and a missing field fails
   as a `KeyError`.
2. **Chip reader** `scripts/compression_chip.py` - a pure `read_row`, string-safe
   on `compression_flag` (the trap is `bool("False") is True`).
3. **Delegate** - a `compressed` chip in the `caution` token after the bucket
   chip, and an unflagged row painting exactly what it paints today.
4. **Calibration CLI** `python -m compression_calibration` - read-only, joins the
   coded `compressed` / `support_resistance_cluttered` vetoes to that session's
   population, prints one table per measure, writes one CSV, reads no bar dated
   after the session, and refuses a live home folder without `--live`.
5. **`compression_break` v1** - previous completed session compressed, this
   close out of that box in the setup's direction, bar range >= 1.0 ATR-20 ->
   `compression_break_recent` + `compression_break_rule_version` + the
   `COMPRESSION_BREAK` tag. Scores byte-identical in every case.

Nothing here may be weakened, skipped or deleted; only added to.

House rules honoured: live stores are READ-ONLY (every scratch home is a
`tmp_path`, the CLI subprocess gets `TRADINGBOTV3_DATA_DIR` and `LOCALAPPDATA`
pointed at it before any `scripts/` import, and the live-folder refusal is proven
by monkeypatching `project_paths`, never by pointing anything at
`C:\TradingBotData`); no literal `vocab_version` is typed in - the fixture asks
the vocabulary which version defines each code.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# The synthetic daily frames. One builder, four shapes, all driven through the
# REAL scan path (`legacy._evaluate_priority_snapshot_for_date`, which is what
# `build_completed_bar_priority_rows` and the live scan both call).
# ---------------------------------------------------------------------------

#: 120 business days; the earnings anchor sits at index 90 so the anchored slice
#: is the 30-bar box every compression rule below is measured over.
FRAME_DATES = pd.bdate_range("2026-03-02", periods=120)
ANCHOR_INDEX = 90
ANCHOR_ISO = FRAME_DATES[ANCHOR_INDEX].date().isoformat()
EVALUATION_DATE = FRAME_DATES[-1].date()


def _base_series() -> tuple[list[float], list[float], list[float], list[float]]:
    """A clean uptrend into the anchor, then a tight box: 103.00 +/- 0.30."""
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    for index in range(len(FRAME_DATES)):
        if index < ANCHOR_INDEX:
            base = 80.0 + index * 0.25
            opens.append(base)
            highs.append(base + 1.2)
            lows.append(base - 1.2)
            closes.append(base + 0.3)
        else:
            base = 103.0
            wiggle = 0.15 * ((index % 3) - 1)
            opens.append(base + wiggle)
            highs.append(base + 0.30)
            lows.append(base - 0.30)
            closes.append(base + wiggle)
    return opens, highs, lows, closes


def _frame(*, last_bar: tuple[float, float, float, float] | None = None, expanding: bool = False) -> pd.DataFrame:
    opens, highs, lows, closes = _base_series()
    if expanding:
        # No box at all: the range WIDENS every session after the anchor, so the
        # previous completed session cannot read as compressed.
        for index in range(ANCHOR_INDEX, len(FRAME_DATES) - 1):
            base = 103.0 + (index - ANCHOR_INDEX) * 0.30
            opens[index] = base - 0.6
            highs[index] = base + 1.8
            lows[index] = base - 1.8
            closes[index] = base + 0.5
    if last_bar is not None:
        opens[-1], highs[-1], lows[-1], closes[-1] = last_bar
    frame = pd.DataFrame(
        {
            "datetime": FRAME_DATES,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000] * len(FRAME_DATES),
        }
    )
    for column in ("open", "high", "low", "close"):
        assert (frame["low"] <= frame[column]).all() and (frame[column] <= frame["high"]).all(), (
            "plan.md sec 5: a candle's four prices carry low <= open, close <= high"
        )
    return frame


#: Still inside the box on the last bar - the plain "this pick is compressed" row.
COMPRESSED_FRAME = _frame()
#: Compressed yesterday, and today's bar leaves the box UP by 3.0 ATR with a
#: range of 3.7 ATR-20. This is the `compression_break` case.
BREAK_FRAME = _frame(last_bar=(103.10, 105.60, 103.00, 105.40))
#: Compressed yesterday, today's close clears the box by 0.69 ATR (past any
#: buffer) but the bar's own range is 0.66 ATR-20 - under the 1.0 ATR rule.
SMALL_RANGE_FRAME = _frame(last_bar=(103.40, 103.75, 103.35, 103.72))
#: Never compressed, and today's bar still leaves the range by 0.71 ATR with a
#: range of 1.41 ATR-20 - so ONLY the missing prior compression can refuse it.
EXPANDING_FRAME = _frame(expanding=True, last_bar=(111.0, 116.0, 110.8, 115.8))

#: Measured on this branch (`claude/pct-3-compression` off the plan branch) with
#: the current `legacy.py`. PCT-3 is additive: these four must not move by one
#: decimal place when the copy-through and the break tag land.
SCORE_TODAY = {
    "COMPRESSED": -70.0,
    "BREAK": -12.0,
    "SMALL_RANGE": -10.0,
    "EXPANDING": 33.0,
}


def _snapshot(frame: pd.DataFrame, side: str = "LONG") -> dict:
    """The real scan evaluation for the last completed bar of `frame`."""
    from master_avwap_lib import legacy

    snapshot = legacy._evaluate_priority_snapshot_for_date(
        symbol="TEST",
        side=side,
        df_full=frame,
        evaluation_date=EVALUATION_DATE,
        current_anchor_iso=ANCHOR_ISO,
        previous_anchor_iso=None,
        recent_earnings_dates=[ANCHOR_ISO],
        latest_release_info=None,
        history_state={},
    )
    assert snapshot, "the synthetic frame must produce a priority snapshot"
    return snapshot


def _anchor_slice(frame: pd.DataFrame) -> pd.DataFrame:
    anchor_date = FRAME_DATES[ANCHOR_INDEX].date()
    return frame[
        (frame["datetime"].dt.date >= anchor_date) & (frame["datetime"].dt.date <= EVALUATION_DATE)
    ].copy()


def _expected_compression(frame: pd.DataFrame, symbol_entry: dict) -> dict:
    """What `summarize_anchor_compression` says about the slice the scan used.

    The anchor stdev and the ATR-20 are read off the symbol entry the scan just
    wrote, so this is the same function on the same inputs - not a re-derivation
    that could agree by accident.
    """
    from master_avwap_lib import legacy

    stdev = symbol_entry["current_anchor"]["stdev"]
    atr20 = symbol_entry["atr20"]
    return legacy.summarize_anchor_compression(_anchor_slice(frame), stdev, atr20)


# ---------------------------------------------------------------------------
# Item 1 - the copy-through
# ---------------------------------------------------------------------------

#: The five fields item 1 adds beside `compression_flag / penalty / note`.
COPY_THROUGH_FIELDS = (
    "compression_score",
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "compression_rule_version",
)

ANCHOR_COMPRESSION_RULE_VERSION = "anchor_compression_v1"


def _assert_copy_through(carrier: dict, expected: dict, where: str) -> None:
    missing = [field for field in COPY_THROUGH_FIELDS if field not in carrier]
    assert not missing, f"{where} is missing {missing}"
    assert carrier["compression_score"] == expected["compression_score"], where
    assert carrier["compression_stdev_atr_ratio"] == pytest.approx(
        expected["compression_stdev_atr_ratio"]
    ), where
    assert carrier["compression_range_atr_ratio"] == pytest.approx(
        expected["compression_range_atr_ratio"]
    ), where
    assert carrier["compression_close_range_atr_ratio"] == pytest.approx(
        expected["compression_close_range_atr_ratio"]
    ), where
    assert carrier["compression_rule_version"] == ANCHOR_COMPRESSION_RULE_VERSION, where


def test_a_compressed_priority_row_carries_the_numbers_the_measure_computed():
    """The scan already knows the score and the three ratios. Show them."""
    snapshot = _snapshot(COMPRESSED_FRAME)
    row = snapshot["priority_row"]
    expected = _expected_compression(COMPRESSED_FRAME, snapshot["symbol_entry"])

    assert expected["is_compressed"] is True, "the fixture must actually be compressed"
    assert expected["compression_score"] == 3
    assert row["compression_flag"] is True

    _assert_copy_through(row, expected, "the priority row")


def test_the_ai_state_symbol_entry_carries_them_too():
    """`master_avwap_ai_state.json` is what the desk's setups table merges from,
    so a field that stops at the row never reaches a chip."""
    snapshot = _snapshot(COMPRESSED_FRAME)
    symbol_entry = snapshot["symbol_entry"]
    expected = _expected_compression(COMPRESSED_FRAME, symbol_entry)

    _assert_copy_through(symbol_entry, expected, "the ai_state symbol entry")


def test_the_tracker_setup_record_carries_a_compression_summary_with_the_numbers():
    from master_avwap_lib import legacy

    snapshot = _snapshot(COMPRESSED_FRAME)
    expected = _expected_compression(COMPRESSED_FRAME, snapshot["symbol_entry"])
    record = legacy.build_tracker_setup_record(
        snapshot["priority_row"],
        snapshot["symbol_entry"],
        snapshot["feature_row"],
        "2026-08-15T00:00:00",
        None,
        scan_date=EVALUATION_DATE.isoformat(),
    )
    assert record is not None

    assert "compression_summary" in record, "build_tracker_setup_record must publish it"
    summary = record["compression_summary"]
    assert summary["is_compressed"] is True
    assert summary["compression_penalty"] == int(snapshot["priority_row"]["compression_penalty"])
    assert summary["compression_note"] == snapshot["priority_row"]["compression_note"]
    _assert_copy_through(summary, expected, "the tracker record's compression_summary")


def test_showing_the_numbers_does_not_move_one_score():
    """The characterization pin for item 1 and item 4 together.

    Each number was read off the CURRENT `legacy.py` on this branch before a line
    of PCT-3 existed. PCT-3 is additive by its own terms: a copy-through and a
    label may not change a score, and the existing exact-score assertions in
    `tests/test_master_avwap_setups.py` must keep passing beside these.
    """
    for name, frame in (
        ("COMPRESSED", COMPRESSED_FRAME),
        ("BREAK", BREAK_FRAME),
        ("SMALL_RANGE", SMALL_RANGE_FRAME),
        ("EXPANDING", EXPANDING_FRAME),
    ):
        row = _snapshot(frame)["priority_row"]
        assert row["score"] == SCORE_TODAY[name], f"{name}: the score moved"


# ---------------------------------------------------------------------------
# Item 2 - the chip reader
# ---------------------------------------------------------------------------


def _chip_row(**over) -> dict:
    """A setups-table row after the `ai_state` merge, in the live field names."""
    row = {
        "symbol": "TEST",
        "side": "LONG",
        "priority_bucket": "near_favorite_zone",
        "current_band_zone": "VWAP to UPPER_1",
        "compression_flag": True,
        "compression_penalty": 10,
        "compression_note": "Compressed post-earnings structure (stdev=0.52 ATR, range=2.10 ATR, close_range=1.40 ATR)",
        "compression_score": 3,
        "compression_stdev_atr_ratio": 0.52,
        "compression_range_atr_ratio": 2.1,
        "compression_close_range_atr_ratio": 1.4,
        "compression_rule_version": ANCHOR_COMPRESSION_RULE_VERSION,
    }
    row.update(over)
    return row


def _ratio_values(ratios) -> list[float]:
    """The three ATR ratios out of whatever container `CompressionRead` uses.

    The packet names the field (`ratios`) and not its shape, so this test pins
    the VALUES and lets the builder choose a NamedTuple, a tuple or a mapping.
    """
    if hasattr(ratios, "_asdict"):
        return [float(value) for value in ratios._asdict().values()]
    if isinstance(ratios, dict):
        return [float(value) for value in ratios.values()]
    return [float(value) for value in ratios]


def test_the_chip_reader_reads_a_row_that_carries_the_fields():
    import compression_chip

    read = compression_chip.read_row(_chip_row())
    assert read is not None
    assert read.flag is True
    assert read.score == 3
    assert read.penalty == 10
    assert "Compressed post-earnings structure" in read.note
    assert sorted(_ratio_values(read.ratios)) == [
        pytest.approx(0.52),
        pytest.approx(1.4),
        pytest.approx(2.1),
    ]


def test_a_row_without_the_fields_is_not_a_compressed_row():
    """Every row on the desk looks like this until the `ai_state` merge lands.

    An unmerged row must read as "nothing to say" - never as flagged, and never
    as an exception on the Qt thread.
    """
    import compression_chip

    bare = {"symbol": "TEST", "side": "LONG", "priority_bucket": "near_favorite_zone"}
    read = compression_chip.read_row(bare)
    assert read is None or read.flag is False


@pytest.mark.parametrize("raw_flag", [True, "True", "true", "TRUE", 1, "1", "yes"])
def test_the_flag_is_read_the_same_however_the_row_spells_it(raw_flag):
    import compression_chip

    read = compression_chip.read_row(_chip_row(compression_flag=raw_flag))
    assert read is not None and read.flag is True, f"{raw_flag!r} means compressed"


@pytest.mark.parametrize("raw_flag", [False, "False", "false", "FALSE", 0, "0", "", None, "no"])
def test_a_false_flag_is_never_read_as_true_however_the_row_spells_it(raw_flag):
    """`bool("False")` is `True`. That is the whole defect class this covers, and
    the empty string is what a CSV round-trip leaves behind."""
    import compression_chip

    read = compression_chip.read_row(_chip_row(compression_flag=raw_flag))
    assert read is None or read.flag is False, f"{raw_flag!r} does not mean compressed"


@pytest.mark.parametrize("row", [None, {}, "not a row", 7, []])
def test_the_chip_reader_never_raises_on_a_row_it_cannot_read(row):
    """`paint` calls this once per visible cell per repaint, so an unreadable
    row is an answer ("nothing to say"), never an exception."""
    import compression_chip

    read = compression_chip.read_row(row)
    assert read is None or read.flag is False


def test_the_chip_tooltip_names_the_score_the_ratios_and_the_penalty():
    import compression_chip

    text = compression_chip.tooltip_text(compression_chip.read_row(_chip_row()))
    assert "compressed" in text.lower()
    assert "3/3" in text, "the score is shown out of its maximum"
    assert "0.52" in text and "2.1" in text and "1.4" in text
    assert "10" in text, "the penalty the scan actually applied"


def test_an_unflagged_row_has_nothing_to_say():
    import compression_chip

    read = compression_chip.read_row(_chip_row(compression_flag=False))
    assert compression_chip.tooltip_text(read) == ""
    assert compression_chip.tooltip_text(None) == ""


# ---------------------------------------------------------------------------
# Item 3 - the painted chip
# ---------------------------------------------------------------------------

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


#: The bucket column at the setups row height, wide enough for two chips. The
#: same cell WS-WS renders, so the two packets' goldens are comparable.
BUCKET_CELL = QRect(0, 0, 260, 40)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _column(key: str) -> int:
    for index, (column_key, _label) in enumerate(SetupTableModel.COLUMNS):
        if column_key == key:
            return index
    raise KeyError(key)


def _delegate_row(**raw_over) -> SetupRow:
    """A RIGHT-side row, so the only `caution` pixels a cell can hold are this
    packet's chip and never WS-WS's `wrong side` one."""
    raw = _chip_row(**raw_over)
    return SetupRow(symbol="TEST", side="LONG", score=61.0, bucket="near_favorite_zone", raw=raw)


def _render(app, row: SetupRow, key: str = "bucket") -> QImage:
    """That row's cell, ALONE in its table (the delegate alternates the row
    background on `index.row() % 2`, so two rows of one table never compare)."""
    model = SetupTableModel([row])
    view = QTableView()
    view.setModel(model)
    delegate = SetupTableDelegate(view)
    view.setItemDelegate(delegate)
    view.resize(900, 300)

    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.font = view.font()
    option.rect = QRect(BUCKET_CELL)
    option.state = QStyle.StateFlag.State_Enabled

    image = QImage(BUCKET_CELL.width(), BUCKET_CELL.height(), QImage.Format.Format_ARGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    try:
        delegate.paint(painter, option, model.index(0, _column(key)))
    finally:
        painter.end()
    return image


def _count_exact(image: QImage, colour: QColor) -> int:
    wanted = QColor(colour).rgb()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if QColor(image.pixel(x, y)).rgb() == wanted
    )


def test_a_flagged_row_paints_a_compressed_chip_and_an_unflagged_one_paints_none(app):
    flagged = _render(app, _delegate_row(compression_flag=True))
    unflagged = _render(app, _delegate_row(compression_flag=False))

    assert _count_exact(flagged, QColor(theme.color("caution"))) > 0, (
        "the compressed chip is painted in the `caution` token"
    )
    assert _count_exact(unflagged, QColor(theme.color("caution"))) == 0
    assert flagged != unflagged


def test_a_row_that_never_heard_of_compression_paints_what_it_paints_today(app):
    """The golden. An unflagged row and a row with no compression keys at all
    must be the same image - the second cannot reach the new code, so equality
    is the proof that the first does not paint either.

    This one is GREEN before the fix by design: it is the "nothing else moved"
    half of item 3's contract, and it goes red the moment a chip, a gap or a
    changed bucket rect leaks onto a row the scan did not flag.
    """
    unflagged = _render(app, _delegate_row(compression_flag=False))
    absent = _render(
        app,
        SetupRow(
            symbol="TEST",
            side="LONG",
            score=61.0,
            bucket="near_favorite_zone",
            raw={
                "symbol": "TEST",
                "side": "LONG",
                "priority_bucket": "near_favorite_zone",
                "current_band_zone": "VWAP to UPPER_1",
            },
        ),
    )
    assert unflagged == absent
    assert _count_exact(unflagged, QColor(theme.color("caution"))) == 0


def test_the_chip_never_reaches_a_cell_that_is_not_the_bucket_cell(app):
    """Hides nothing, moves nothing: only the bucket cell gains a chip."""
    flagged = _delegate_row(compression_flag=True)
    quiet = _delegate_row(compression_flag=False)
    for key in ("symbol", "side", "score", "favorite", "dislike"):
        assert _render(app, flagged, key) == _render(app, quiet, key), key


def _hover_text(row: SetupRow, key: str = "bucket") -> str:
    """Whatever the trader would read hovering that cell - delegate or model.

    Lifted from `tests/test_ws_ws_wrong_side.py` so both packets read the same
    tooltip the same way.
    """
    model = SetupTableModel([row])
    view = QTableView()
    view.setModel(model)
    delegate = SetupTableDelegate(view)
    view.setItemDelegate(delegate)
    view.resize(900, 300)

    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.font = view.font()
    option.rect = QRect(BUCKET_CELL)
    option.state = QStyle.StateFlag.State_Enabled
    index = model.index(0, _column(key))

    seen: list[str] = []
    original = QToolTip.showText

    def _spy(pos, text, *args, **kwargs):  # noqa: ANN001
        seen.append(str(text))

    QToolTip.showText = _spy
    try:
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(4, 4), QPoint(4, 4))
        delegate.helpEvent(event, view, option, index)
    finally:
        QToolTip.showText = original
    if seen:
        return seen[-1]
    return str(model.data(index, Qt.ItemDataRole.ToolTipRole) or "")


def test_the_bucket_tooltip_says_compressed_on_a_flagged_row(app):
    """The tooltip is ADDED to the bucket cell's own label, never instead of it."""
    text = _hover_text(_delegate_row(compression_flag=True))
    assert "Near" in text, "the bucket cell keeps the tooltip it already had"
    assert "compressed" in text.lower()
    assert "3/3" in text
    assert "0.52" in text and "2.1" in text and "1.4" in text

    quiet = _hover_text(_delegate_row(compression_flag=False))
    assert "compressed" not in quiet.lower()


# ---------------------------------------------------------------------------
# Item 4 - the calibration CLI
# ---------------------------------------------------------------------------

#: The session every fixture row belongs to.
SESSION_DATE = "2026-08-20"
SINCE = "2026-08-01"

#: The population the CLI must join against: six rows shown that session, four of
#: them vetoed (three `compressed`, one v1 `support_resistance_cluttered`), two
#: not. `EEE` carries an UNCODED veto - a `reason_code` key that is PRESENT and
#: EMPTY, which is what 136 of the live rows look like - and must land in "rest".
#:
#: The three anchor ratios separate the two groups perfectly and their medians
#: are exact: vetoed stdev 0.55 / rest 1.30, vetoed range 1.75 / rest 5.50,
#: vetoed close-range 0.80 / rest 3.50. Only AAA carries the current
#: `compression_flag`, so the flag's hit rate on the vetoed set is 1/4 = 0.25 -
#: which is the packet's whole point: the measure disagrees with the eye.
POPULATION = (
    # symbol, vetoed, stdev, range, close_range, compression_flag, tight_bars
    ("AAA", True, 0.40, 1.0, 0.5, True, True),
    ("BBB", True, 0.50, 1.5, 0.7, False, True),
    ("CCC", True, 0.60, 2.0, 0.9, False, True),
    ("DDD", True, 0.70, 2.5, 1.1, False, True),
    ("EEE", False, 1.20, 5.0, 3.0, False, False),
    ("FFF", False, 1.40, 6.0, 4.0, False, False),
)

VETOED_SYMBOLS = {row[0] for row in POPULATION if row[1]}
REST_SYMBOLS = {row[0] for row in POPULATION if not row[1]}

EXPECTED_MEDIANS = {
    "compression_stdev_atr_ratio": (0.55, 1.30),
    "compression_range_atr_ratio": (1.75, 5.50),
    "compression_close_range_atr_ratio": (0.80, 3.50),
}

#: One printed table per measure. The three anchor ratios and `range10_atr14` /
#: `range20_atr14` are the packet's own names; the last two name the packet's
#: "Bollinger(20, 2) width percentile over 120 sessions" and "ATR-14 / ATR-50".
MEASURE_KEYS = (
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "range10_atr14",
    "range20_atr14",
    "bollinger20_width_pct",
    "atr14_atr50",
)

_NUMBER = r"-?(?:\d+(?:\.\d+)?|nan)"


def _veto_versions() -> dict[str, int]:
    """The version that DEFINES each code, asked of the vocabulary itself.

    Never a literal: `CLAUDE.md` - *"Never assert a literal `vocab_version` in a
    test."*
    """
    from ui.annotations.vocabulary import available_veto_versions, load_veto_vocabulary

    wanted = {"compressed": None, "support_resistance_cluttered": None}
    for version in sorted(available_veto_versions()):
        vocabulary = load_veto_vocabulary(version=version)
        for code in wanted:
            if wanted[code] is None and code in vocabulary.codes:
                wanted[code] = int(vocabulary.vocab_version)
    missing = [code for code, version in wanted.items() if version is None]
    assert not missing, f"the build no longer carries {missing}"
    return wanted


def _annotation_rows() -> list[dict]:
    versions = _veto_versions()
    rows: list[dict] = []
    for index, symbol in enumerate(("AAA", "BBB", "CCC")):
        rows.append(
            {
                "schema_version": 1,
                "event_id": f"evt-compressed-{index}",
                "event_type": "veto",
                "symbol": symbol,
                "session_date": SESSION_DATE,
                "created_at": f"2026-08-20T09:4{index}:00-07:00",
                "source": "chart_review",
                "reason_code": "compressed",
                "vocab_version": versions["compressed"],
                "timeframe": "D1",
                "side": "LONG",
                "surface": "chart_review",
                "scan_date": SESSION_DATE,
                "tracker_setup_id": f"{symbol}-LONG-{SESSION_DATE}",
                "canonical_setup_id": f"{symbol}-LONG",
                "priority_bucket": "near_favorite_zone",
                "score": 61.0,
                "expected_r": 0.8,
            }
        )
    rows.append(
        {
            "schema_version": 1,
            "event_id": "evt-cluttered-v1",
            "event_type": "veto",
            "symbol": "DDD",
            "session_date": SESSION_DATE,
            "created_at": "2026-08-20T09:50:00-07:00",
            "source": "chart_review",
            "reason_code": "support_resistance_cluttered",
            "vocab_version": versions["support_resistance_cluttered"],
            "timeframe": "D1",
            "side": "LONG",
            "surface": "chart_review",
            "scan_date": SESSION_DATE,
            "tracker_setup_id": f"DDD-LONG-{SESSION_DATE}",
            "canonical_setup_id": "DDD-LONG",
            "priority_bucket": "near_favorite_zone",
            "score": 55.0,
            "expected_r": 0.6,
        }
    )
    rows.append(
        {
            # An UNCODED veto. The key is PRESENT and EMPTY and there is no
            # `vocab_version`, exactly as `ui/annotations/store.py` writes one.
            # It is a veto and it is not a COMPRESSED veto.
            "schema_version": 1,
            "event_id": "evt-uncoded",
            "event_type": "veto",
            "symbol": "EEE",
            "session_date": SESSION_DATE,
            "created_at": "2026-08-20T09:55:00-07:00",
            "source": "chart_review",
            "reason_code": "",
            "timeframe": "D1",
            "side": "LONG",
            "surface": "chart_review",
        }
    )
    return rows


def _tracker_payload() -> dict:
    """The scan population, in `build_tracker_setup_record`'s own field names.

    Hand-written on purpose - a fixture generated by the code under test would
    pass whatever that code happens to emit. The five copy-through fields are
    here because item 1 puts them here; the CLI reads the anchor measures off
    this row rather than recomputing an anchor it cannot know.
    """
    setups = {}
    for symbol, vetoed, stdev, band, close_range, flag, _tight in POPULATION:
        setup_id = f"{symbol}-LONG-{SESSION_DATE}"
        setups[setup_id] = {
            "setup_id": setup_id,
            "symbol": symbol,
            "side": "LONG",
            "scan_date": SESSION_DATE,
            "entry_trade_date": SESSION_DATE,
            "anchor_date": "2026-07-06",
            "entry_price": 103.0,
            "priority_bucket": "near_favorite_zone",
            "priority_score": 61.0 if vetoed else 58.0,
            "setup_family": "general",
            "setup_status": "open",
            "bar_status": "completed",
            "view_mode": "PREVIEW",
            "compression_flag": flag,
            "compression_penalty": 10 if flag else 0,
            "compression_note": "Compressed post-earnings structure" if flag else "",
            "compression_score": 3 if flag else 1,
            "compression_stdev_atr_ratio": stdev,
            "compression_range_atr_ratio": band,
            "compression_close_range_atr_ratio": close_range,
            "compression_rule_version": ANCHOR_COMPRESSION_RULE_VERSION,
            "entry_feature_snapshot": {
                "atr20": 2.0,
                "current_close": 103.0,
                "compression_flag": flag,
                "compression_penalty": 10 if flag else 0,
                "compression_note": "Compressed post-earnings structure" if flag else "",
            },
        }
    return {
        "saved_at": "2026-08-20T13:05:00-07:00",
        "saved_by": "test_pct3_compression",
        "data_session": SESSION_DATE,
        "daily_watchlists": {},
        "setups": setups,
        "control_setups": {},
        "study_setups": {},
        "stats": [],
        "setup_type_stats": [],
        "attribute_registry": {},
    }


def _bar_frame(*, tight: bool, extra_sessions: int = 0) -> pd.DataFrame:
    """200 sessions ending on the vetoed session, plus optional LATER sessions.

    The later sessions are a wild gap up: if any measure reads them, the answer
    changes, which is what `..._never_reads_a_bar_after_the_session` proves.
    """
    sessions = pd.bdate_range(end=pd.Timestamp(SESSION_DATE), periods=200)
    rows = []
    for index, stamp in enumerate(sessions):
        base = 100.0 + (index % 7) * 0.2
        half = 0.25 if (tight and index >= 170) else 2.5
        rows.append((stamp, base, base + half, base - half, base + half / 4.0))
    if extra_sessions:
        later = pd.bdate_range(start=pd.Timestamp(SESSION_DATE) + pd.Timedelta(days=1), periods=extra_sessions)
        for index, stamp in enumerate(later):
            base = 300.0 + index * 12.0
            rows.append((stamp, base, base + 25.0, base - 25.0, base + 20.0))
    frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    frame["volume"] = 1_000_000
    return frame


def _write_bar_cache(cache_dir: Path, *, extra_sessions: int = 0) -> None:
    """Through `daily_bar_cache`'s own rule, then the writer's own `to_csv`."""
    from master_avwap_lib import daily_bar_cache

    cache_dir.mkdir(parents=True, exist_ok=True)
    for symbol, _vetoed, _stdev, _band, _close_range, _flag, tight in POPULATION:
        frame = _bar_frame(tight=tight, extra_sessions=extra_sessions)
        writable, _drops = daily_bar_cache.filter_writable_rows(frame, symbol=symbol)
        assert len(writable) == len(frame), (
            "every fixture bar is a completed, possible candle; the cache writer keeps them all"
        )
        writable.to_csv(cache_dir / f"{symbol}.csv", index=False)


def _scratch_env(home: Path, local_appdata: Path, diagnostics: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["TRADINGBOTV3_DATA_DIR"] = str(home)
    env["LOCALAPPDATA"] = str(local_appdata)
    env["TRADINGBOT_DIAGNOSTICS_DIR"] = str(diagnostics)
    env["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("PYTHONPATH", None)
    return env


def _resolved_paths(env: dict[str, str]) -> dict[str, str]:
    """Ask the CHILD's own `project_paths` where the scratch stores live.

    Re-deriving them here would be a second copy of `project_paths`'s layout; a
    one-line child that prints them is the real answer, and it proves the scratch
    home is nowhere near `C:\\TradingBotData` before anything is written.
    """
    code = (
        "import json, project_paths as p;"
        "print(json.dumps({"
        "'annotations': str(p.TRADER_ANNOTATIONS_FILE),"
        "'tracker': str(p.MASTER_AVWAP_SETUP_TRACKER_FILE),"
        "'bars': str(p.DAILY_BARS_CACHE_DIR),"
        "'data': str(p.DATA_DIR)}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(SCRIPTS_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout.strip().splitlines()[-1])
    assert "TradingBotData" not in paths["data"], (
        f"the scratch home must never resolve under the live folder: {paths['data']}"
    )
    return paths


def _build_scratch_home(tmp_path: Path, *, extra_sessions: int = 0) -> tuple[dict[str, str], Path]:
    home = tmp_path / "home"
    local_appdata = tmp_path / "localappdata"
    diagnostics = tmp_path / "diagnostics"
    for directory in (home, local_appdata, diagnostics):
        directory.mkdir(parents=True, exist_ok=True)

    env = _scratch_env(home, local_appdata, diagnostics)
    paths = _resolved_paths(env)

    annotations = Path(paths["annotations"])
    annotations.parent.mkdir(parents=True, exist_ok=True)
    with annotations.open("w", encoding="utf-8") as handle:
        for row in _annotation_rows():
            handle.write(json.dumps(row) + "\n")

    tracker = Path(paths["tracker"])
    tracker.parent.mkdir(parents=True, exist_ok=True)
    tracker.write_text(json.dumps(_tracker_payload(), indent=1), encoding="utf-8")

    _write_bar_cache(Path(paths["bars"]), extra_sessions=extra_sessions)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return env, out_dir


def _run_cli(env: dict[str, str], out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "compression_calibration", "--since", SINCE, "--out", str(out_dir), *extra],
        cwd=str(SCRIPTS_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _measure_block(stdout: str, key: str) -> str:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if key in line:
            return "\n".join(lines[index : index + 4])
    raise AssertionError(f"no table for {key!r} in:\n{stdout}")


def test_the_calibration_cli_prints_one_table_per_measure_with_n_medians_and_an_auc(tmp_path):
    """Four vetoed rows against two, the v1 code pooled in, the uncoded veto out.

    `n = 4 / 2` is the number that proves both joins: a CLI that ignored the v1
    `support_resistance_cluttered` row prints 3/3, and one that counted the
    uncoded veto prints 5/1.
    """
    env, out_dir = _build_scratch_home(tmp_path)
    result = _run_cli(env, out_dir)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    for key in MEASURE_KEYS:
        block = _measure_block(result.stdout, key)
        counts = re.search(r"n\s*=\s*(\d+)\s*/\s*(\d+)", block)
        assert counts, f"{key}: no `n = vetoed / rest` in:\n{block}"
        assert (int(counts.group(1)), int(counts.group(2))) == (4, 2), key

        medians = re.search(rf"median\s*=\s*({_NUMBER})\s*/\s*({_NUMBER})", block)
        assert medians, f"{key}: no `median = vetoed / rest` in:\n{block}"

        auc = re.search(rf"auc\s*=\s*({_NUMBER})", block, re.IGNORECASE)
        assert auc, f"{key}: no `auc =` in:\n{block}"
        assert 0.0 <= float(auc.group(1)) <= 1.0, f"{key}: an AUC outside [0, 1]"

    for key, (vetoed_median, rest_median) in EXPECTED_MEDIANS.items():
        block = _measure_block(result.stdout, key)
        medians = re.search(rf"median\s*=\s*({_NUMBER})\s*/\s*({_NUMBER})", block)
        assert float(medians.group(1)) == pytest.approx(vetoed_median), key
        assert float(medians.group(2)) == pytest.approx(rest_median), key
        auc = float(re.search(rf"auc\s*=\s*({_NUMBER})", block, re.IGNORECASE).group(1))
        assert auc in (0.0, 1.0), (
            f"{key} separates the fixture perfectly, so its AUC is 1.0 or 0.0 - got {auc}"
        )


def test_the_calibration_cli_reports_the_current_flag_s_hit_rate_on_the_vetoed_set(tmp_path):
    """One of the four vetoed rows carries `compression_flag`. 1/4 = 0.25 - the
    packet's finding, printed rather than argued."""
    env, out_dir = _build_scratch_home(tmp_path)
    result = _run_cli(env, out_dir)
    assert result.returncode == 0, result.stderr

    match = re.search(rf"compression_flag.*?hit[ _]rate\s*=\s*({_NUMBER})", result.stdout, re.IGNORECASE)
    assert match, f"no `compression_flag hit rate =` line in:\n{result.stdout}"
    assert float(match.group(1)) == pytest.approx(0.25)


def test_the_calibration_cli_writes_one_csv_row_per_shown_row(tmp_path):
    env, out_dir = _build_scratch_home(tmp_path)
    result = _run_cli(env, out_dir)
    assert result.returncode == 0, result.stderr

    written = sorted(out_dir.glob("compression_calibration_*.csv"))
    assert len(written) == 1, f"expected exactly one CSV, found {[p.name for p in written]}"
    assert re.fullmatch(r"compression_calibration_\d{4}-\d{2}-\d{2}\.csv", written[0].name), written[0].name

    rows = list(csv.DictReader(written[0].open("r", encoding="utf-8", newline="")))
    assert len(rows) == len(POPULATION), "every row shown that session is a row in the CSV"
    assert {row["symbol"] for row in rows} == {entry[0] for entry in POPULATION}
    assert {row["session_date"] for row in rows} == {SESSION_DATE}

    veto_column = next((name for name in rows[0] if "veto" in name.lower()), "")
    assert veto_column, f"the CSV must say which rows were vetoed: {list(rows[0])}"
    flagged = {
        row["symbol"]
        for row in rows
        if str(row[veto_column]).strip().lower() in {"1", "true", "yes", "y"}
    }
    assert flagged == VETOED_SYMBOLS, (
        "the uncoded veto on EEE is a veto and is not a COMPRESSED veto"
    )


def test_the_calibration_cli_never_reads_a_bar_dated_after_the_session(tmp_path):
    """Point-in-time (plan.md sec 5).

    The same fixture twice: once with the cache ending on the session, once with
    ten wild sessions written AFTER it. Every measure must come out identical -
    a 300-and-climbing tail would move every range, every ATR and every Bollinger
    percentile if it were read.
    """
    clean_env, clean_out = _build_scratch_home(tmp_path / "clean")
    poisoned_env, poisoned_out = _build_scratch_home(tmp_path / "poisoned", extra_sessions=10)

    clean_result = _run_cli(clean_env, clean_out)
    poisoned_result = _run_cli(poisoned_env, poisoned_out)
    assert clean_result.returncode == 0, clean_result.stderr
    assert poisoned_result.returncode == 0, poisoned_result.stderr

    def _measures(out_dir: Path) -> list[dict[str, str]]:
        path = next(iter(sorted(out_dir.glob("compression_calibration_*.csv"))))
        rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        volatile = ("generated", "as_of", "run_at", "timestamp", "written")
        return [
            {
                name: value
                for name, value in sorted(row.items())
                if not any(token in name.lower() for token in volatile)
            }
            for row in sorted(rows, key=lambda row: row["symbol"])
        ]

    assert _measures(clean_out) == _measures(poisoned_out)


def test_the_calibration_cli_refuses_the_live_home_folder_without_live(tmp_path, capsys):
    r"""Read-only on the live stores, and it says so before it reads anything.

    `project_paths.DATA_DIR` is monkeypatched to the live path - nothing is ever
    pointed at `C:\TradingBotData` for real, and the three stores are patched to
    empty scratch paths so a guard that failed to fire still could not touch it.
    The guard therefore has to read `project_paths.DATA_DIR` at CALL time, the
    way `d1_environment_store.py:293` already does.
    """
    import project_paths

    import compression_calibration

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(project_paths, "DATA_DIR", Path(r"C:\TradingBotData\data"), raising=False)
        monkey.setattr(project_paths, "PERSISTENT_DATA_DIR", Path(r"C:\TradingBotData"), raising=False)
        monkey.setattr(
            project_paths, "TRADER_ANNOTATIONS_FILE", tmp_path / "no-such-annotations.jsonl", raising=False
        )
        monkey.setattr(
            project_paths, "MASTER_AVWAP_SETUP_TRACKER_FILE", tmp_path / "no-such-tracker.json", raising=False
        )
        monkey.setattr(project_paths, "DAILY_BARS_CACHE_DIR", tmp_path / "no-such-bars", raising=False)
        try:
            code = compression_calibration.main(["--since", SINCE, "--out", str(tmp_path)])
        except SystemExit as stop:
            code = stop.code
    finally:
        monkey.undo()

    assert code not in (0, None), "a run against the live home folder must exit non-zero"
    printed = capsys.readouterr()
    said = f"{printed.out}\n{printed.err}"
    assert "--live" in said, f"the message must name the way through:\n{said}"
    assert "TradingBotData" in said, f"the message must name the folder it refused:\n{said}"
    assert not list(tmp_path.glob("compression_calibration_*.csv")), "a refusal writes nothing"


# ---------------------------------------------------------------------------
# Item 5 (packet item 4) - the `compression_break` v1 tag
# ---------------------------------------------------------------------------

COMPRESSION_BREAK_RULE_VERSION = "compression_break_v1"


def _prior_session_compression(frame: pd.DataFrame, symbol_entry: dict) -> dict:
    """`summarize_anchor_compression` on `price_slice[:-1]` - the packet's rule
    for "the previous completed session was compressed", stated once here."""
    from master_avwap_lib import legacy

    return legacy.summarize_anchor_compression(
        _anchor_slice(frame).iloc[:-1],
        symbol_entry["current_anchor"]["stdev"],
        symbol_entry["atr20"],
    )


def test_a_close_out_of_yesterday_s_box_on_a_wide_bar_is_a_compression_break():
    from master_avwap_lib import setup_tagging

    snapshot = _snapshot(BREAK_FRAME)
    row = snapshot["priority_row"]
    entry = snapshot["symbol_entry"]

    prior = _prior_session_compression(BREAK_FRAME, entry)
    assert prior["is_compressed"] is True, "the premise: yesterday's slice was compressed"
    box_high = float(_anchor_slice(BREAK_FRAME).iloc[:-1]["high"].max())
    last = BREAK_FRAME.iloc[-1]
    assert float(last["close"]) > box_high, "the close leaves the box upward"
    assert (float(last["high"]) - float(last["low"])) / entry["atr20"] >= 1.0, "a wide bar"

    assert row["compression_break_recent"] is True
    assert row["compression_break_rule_version"] == COMPRESSION_BREAK_RULE_VERSION
    assert str(row.get("compression_break_note") or "").strip(), "a flag with no note says nothing"
    assert entry["compression_break_recent"] is True

    tags = setup_tagging.derive_setup_tag_payload(row)["setup_tags"]
    assert "COMPRESSION_BREAK" in tags, tags


def test_a_narrow_bar_leaving_the_box_is_not_a_compression_break():
    """The close clears yesterday's box by 0.69 ATR, so only the 1.0-ATR range
    rule can refuse it."""
    from master_avwap_lib import setup_tagging

    snapshot = _snapshot(SMALL_RANGE_FRAME)
    row = snapshot["priority_row"]
    entry = snapshot["symbol_entry"]

    assert _prior_session_compression(SMALL_RANGE_FRAME, entry)["is_compressed"] is True
    last = SMALL_RANGE_FRAME.iloc[-1]
    box_high = float(_anchor_slice(SMALL_RANGE_FRAME).iloc[:-1]["high"].max())
    assert float(last["close"]) > box_high
    assert (float(last["high"]) - float(last["low"])) / entry["atr20"] < 1.0

    assert row["compression_break_recent"] is False
    assert "COMPRESSION_BREAK" not in setup_tagging.derive_setup_tag_payload(row)["setup_tags"]


def test_a_wide_bar_with_no_compression_behind_it_is_not_a_compression_break():
    """The range widened every session, so there is no box to leave - even
    though today's bar is 1.41 ATR wide and closes 0.71 ATR past the range."""
    from master_avwap_lib import setup_tagging

    snapshot = _snapshot(EXPANDING_FRAME)
    row = snapshot["priority_row"]
    entry = snapshot["symbol_entry"]

    assert _prior_session_compression(EXPANDING_FRAME, entry)["is_compressed"] is False
    last = EXPANDING_FRAME.iloc[-1]
    box_high = float(_anchor_slice(EXPANDING_FRAME).iloc[:-1]["high"].max())
    assert float(last["close"]) > box_high, "it does leave the range"
    assert (float(last["high"]) - float(last["low"])) / entry["atr20"] >= 1.0, "and it is a wide bar"

    assert row["compression_break_recent"] is False
    assert "COMPRESSION_BREAK" not in setup_tagging.derive_setup_tag_payload(row)["setup_tags"]


def test_a_short_never_reads_an_upward_break_as_its_own():
    """The break is DIRECTIONAL: the same frame, read SHORT, breaks nothing."""
    from master_avwap_lib import setup_tagging

    row = _snapshot(BREAK_FRAME, side="SHORT")["priority_row"]
    assert row["compression_break_recent"] is False
    assert "COMPRESSION_BREAK" not in setup_tagging.derive_setup_tag_payload(row)["setup_tags"]


def test_the_break_tag_is_stamped_with_its_rule_version_on_every_row():
    """A labelled rule (`compression_break_v1`) is re-tunable by name after the
    calibration report; an unlabelled one is not."""
    for frame in (BREAK_FRAME, SMALL_RANGE_FRAME, EXPANDING_FRAME, COMPRESSED_FRAME):
        row = _snapshot(frame)["priority_row"]
        assert row["compression_break_rule_version"] == COMPRESSION_BREAK_RULE_VERSION
