r"""PCT-3: WHERE the compression numbers join the setups table row.

Added by the builder beside the tester's `tests/test_pct3_compression.py`, which
pins the reader, the chip, the CLI and the tag but deliberately leaves the merge
seam to the builder. The packet's requirement for that seam is one sentence:
*"rows lack the fields until the report is merged with the `ai_state` symbol
entry on the ChartDataService / data-feed worker ... and never reads a file on
paint."*

The seam chosen is `data_feed.merge_compression_from_ai_state`, called from
`enrich_setup_rows_for_display` - the LOAD path every setups-table source already
goes through (`load_latest_setup_rows_with_meta`), and the same place the group
context is filled in. So the two halves of the requirement are:

1. it is a plain function over plain rows: no Qt import, no widget, no signal -
   so it runs wherever the loader runs, including a worker thread, which this
   file proves by running it on one with no `QApplication` involved;
2. `SetupTableDelegate.paint` never reaches a file. `paint` runs once per
   visible cell per repaint (`CLAUDE.md`: *nothing expensive belongs on the Qt
   thread*), so the test watches every `open()` the paint pass makes and
   requires that none of them is the 38 MB `ai_state` file - and that a
   compression lookup that RAISES cannot reach the paint pass at all.

`tests/conftest.py` points `project_paths` at a test directory; nothing here
reads or writes a live store.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


COMPRESSION_ENTRY = {
    "compression_flag": True,
    "compression_penalty": 10,
    "compression_note": "Compressed post-earnings structure (stdev=0.52 ATR)",
    "compression_score": 3,
    "compression_stdev_atr_ratio": 0.52,
    "compression_range_atr_ratio": 2.1,
    "compression_close_range_atr_ratio": 1.4,
    "compression_rule_version": "anchor_compression_v1",
}


def _ai_state_file(tmp_path: Path) -> Path:
    path = tmp_path / "master_avwap_ai_state.json"
    path.write_text(
        json.dumps(
            {
                "symbols": {
                    "TEST": {
                        "side": "LONG",
                        "atr20": 2.0,
                        "last_close": 103.0,
                        "current_anchor": {"date": "2026-07-06", "vwap": 102.0, "stdev": 1.04},
                        **COMPRESSION_ENTRY,
                    },
                    "QUIET": {
                        "side": "LONG",
                        "atr20": 2.0,
                        "last_close": 50.0,
                        "current_anchor": {"date": "2026-07-06", "vwap": 49.0, "stdev": 3.0},
                        "compression_flag": False,
                        "compression_penalty": 0,
                        "compression_note": "",
                        "compression_score": 1,
                        "compression_stdev_atr_ratio": 1.5,
                        "compression_range_atr_ratio": 6.0,
                        "compression_close_range_atr_ratio": 4.0,
                        "compression_rule_version": "anchor_compression_v1",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def ai_state(tmp_path, monkeypatch):
    """A scratch `ai_state` file, with the module's mtime cache emptied."""
    from ui.services import ai_state_levels

    path = _ai_state_file(tmp_path)
    monkeypatch.setattr(ai_state_levels, "MASTER_AVWAP_AI_STATE_FILE", path, raising=False)
    monkeypatch.setitem(ai_state_levels._cache, "mtime", None)
    monkeypatch.setitem(ai_state_levels._cache, "levels", {})
    monkeypatch.setitem(ai_state_levels._cache, "compression", {})
    yield path


def _row(symbol: str = "TEST"):
    """A row exactly as `load_setup_rows_from_priority_report` builds one.

    The ranked report line carries the symbol, side, score, family and bucket -
    and nothing at all about compression. That absence is the premise of the
    whole seam.
    """
    from ui.models.setup import SetupRow

    return SetupRow(
        symbol=symbol,
        side="LONG",
        score=61.0,
        bucket="near_favorite_zone",
        raw={
            "symbol": symbol,
            "side": "LONG",
            "priority_bucket": "near_favorite_zone",
            "current_band_zone": "VWAP to UPPER_1",
        },
    )


def test_a_report_row_has_no_compression_fields_until_the_merge_runs(ai_state):
    """The premise, stated as a test so it fails if the report line ever gains
    the fields and this seam becomes dead code."""
    import compression_chip
    from ui.services.data_feed import merge_compression_from_ai_state

    row = _row()
    assert compression_chip.read_row(row.raw) is None, "nothing to say before the merge"

    assert merge_compression_from_ai_state([row]) == 1
    read = compression_chip.read_row(row.raw)
    assert read is not None and read.flag is True
    assert read.score == 3
    assert read.penalty == 10
    assert read.ratios.stdev_atr == pytest.approx(0.52)
    assert read.rule_version == "anchor_compression_v1"


def test_the_merge_never_overwrites_a_reading_the_row_already_carries(ai_state):
    """The focus feed's rows come out of the scan with their own numbers; the
    merge FILLS, so a fresher row can never be overwritten by a staler file."""
    from ui.services.data_feed import merge_compression_from_ai_state

    row = _row()
    row.raw["compression_flag"] = False
    row.raw["compression_score"] = 0

    merge_compression_from_ai_state([row])
    assert row.raw["compression_flag"] is False
    assert row.raw["compression_score"] == 0
    assert row.raw["compression_stdev_atr_ratio"] == pytest.approx(0.52), "the gaps are still filled"


def test_an_unreadable_ai_state_file_costs_the_rows_nothing(tmp_path, monkeypatch):
    """An evidence read never costs the thing it annotates (plan.md sec 5).

    A missing file, an unparseable file and a lookup that raises all leave the
    rows exactly as they came, and none of them raises.
    """
    from ui.services import ai_state_levels
    from ui.services.data_feed import merge_compression_from_ai_state

    monkeypatch.setitem(ai_state_levels._cache, "mtime", None)
    monkeypatch.setitem(ai_state_levels._cache, "levels", {})
    monkeypatch.setitem(ai_state_levels._cache, "compression", {})

    monkeypatch.setattr(
        ai_state_levels, "MASTER_AVWAP_AI_STATE_FILE", tmp_path / "no-such-file.json", raising=False
    )
    row = _row()
    assert merge_compression_from_ai_state([row]) == 0
    assert "compression_flag" not in row.raw

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(ai_state_levels, "MASTER_AVWAP_AI_STATE_FILE", broken, raising=False)
    assert merge_compression_from_ai_state([_row()]) == 0

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the file feed is down")

    monkeypatch.setattr(ai_state_levels, "load_symbol_compression", _explode, raising=False)
    assert merge_compression_from_ai_state([_row()]) == 0


def test_the_load_path_is_what_merges(ai_state, monkeypatch):
    """`enrich_setup_rows_for_display` is the seam every setups-table source
    already goes through, so no caller has to remember to ask."""
    from ui.services import data_feed

    row = _row()
    data_feed.enrich_setup_rows_for_display([row], supplemental_rows=[])
    assert row.raw.get("compression_score") == 3


def test_the_merge_runs_off_the_gui_thread(ai_state):
    """The seam is a plain function over plain rows - it needs no Qt at all.

    Proven by running it on a worker thread with no `QApplication` touched: if
    the merge ever grew a widget, a signal or a `QObject` parent, constructing
    it here would fail or warn about thread affinity.
    """
    from ui.services.data_feed import merge_compression_from_ai_state

    rows = [_row(), _row("QUIET")]
    result: dict[str, object] = {}

    def _work() -> None:
        try:
            result["filled"] = merge_compression_from_ai_state(rows)
            result["thread"] = threading.current_thread().name
        except Exception as exc:  # pragma: no cover - the failure this guards
            result["error"] = exc

    worker = threading.Thread(target=_work, name="pct3-merge-worker")
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert "error" not in result, result.get("error")
    assert result["filled"] == 2
    assert result["thread"] == "pct3-merge-worker"
    assert rows[0].raw["compression_flag"] is True
    assert rows[1].raw["compression_flag"] is False


pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
)

from ui.models.setup_table_model import SetupTableModel  # noqa: E402
from ui.widgets.setup_delegate import SetupTableDelegate  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _paint_every_cell(row) -> list[str]:
    """Paint every column of `row` once, returning every path `open()` saw."""
    model = SetupTableModel([row])
    view = QTableView()
    view.setModel(model)
    delegate = SetupTableDelegate(view)
    view.setItemDelegate(delegate)
    view.resize(900, 300)

    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.font = view.font()
    option.rect = QRect(0, 0, 260, 40)
    option.state = QStyle.StateFlag.State_Enabled

    opened: list[str] = []
    real_open = builtins.open

    def _spy(file, *args, **kwargs):  # noqa: ANN001
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    image = QImage(260, 40, QImage.Format.Format_ARGB32)
    image.fill(QColor("#000000"))
    painter = QPainter(image)
    builtins.open = _spy
    try:
        for column in range(len(SetupTableModel.COLUMNS)):
            delegate.sizeHint(option, model.index(0, column))
            delegate.paint(painter, option, model.index(0, column))
    finally:
        builtins.open = real_open
        painter.end()
    return opened


def test_paint_never_reads_the_ai_state_file(app, ai_state):
    """The chip is painted from `row.raw`, which the LOAD path filled.

    If a future change made `paint` ask the file feed instead, this catches it:
    `paint` and `sizeHint` together may not open the 38 MB `ai_state` JSON - or
    any file at all under the data directory.
    """
    from ui.services.data_feed import merge_compression_from_ai_state

    row = _row()
    merge_compression_from_ai_state([row])
    assert row.raw["compression_flag"] is True, "the chip has something to paint"

    opened = _paint_every_cell(row)
    offenders = [path for path in opened if "master_avwap_ai_state" in path or str(ai_state) == path]
    assert not offenders, f"paint opened the ai_state file: {offenders}"


def test_a_compression_feed_that_raises_can_never_reach_paint(app, ai_state, monkeypatch):
    """A file feed that is down is a chip that is not painted, never a traceback
    on the Qt thread. If `paint` called the feed, this would raise."""
    from ui.services import ai_state_levels
    from ui.services.data_feed import merge_compression_from_ai_state

    row = _row()
    merge_compression_from_ai_state([row])

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the file feed is down")

    monkeypatch.setattr(ai_state_levels, "load_symbol_compression", _explode, raising=False)
    monkeypatch.setattr(ai_state_levels, "load_symbol_levels", _explode, raising=False)
    _paint_every_cell(row)  # must not raise
