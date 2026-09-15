r"""PCT-3: WHERE the compression numbers join a setups-table row, and on which thread.

Added by the BUILDER beside the tester's `tests/test_pct3_compression.py`, which
pins the reader, the chip, the CLI and the tag but leaves the merge seam to the
builder. Rewritten in the fix round after the first review measured what the
first attempt actually cost:

> `merge_compression_from_ai_state` runs inside `enrich_setup_rows_for_display`,
> which `master_avwap_panel.refresh_from_reports` and `_on_scan_finished` call on
> the Qt thread - a measured 281-292 ms parse of the 36 MB ai_state on every
> mtime change.

`CLAUDE.md`: *nothing expensive belongs on the Qt thread, and "expensive"
includes a stylesheet.* 281 ms per watched-file change is not a chip, it is a
stall. So the contract these tests pin is now three-sided:

1. the PARSE happens in `ai_state_levels.warm_cache()`, which only a worker
   calls (`master_avwap_panel._AiStateCompressionWorker`);
2. the Qt thread's merge reads `cached_symbol_compression()` - memory only, no
   `open`, no `stat`, no `json.load` - and a cold cache simply has nothing to
   say, which is a row with no chip rather than a stall;
3. `paint` never reaches a file at all, and a feed that RAISES cannot reach it.

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


# ---------------------------------------------------------------------------
# Which call parses, and which call may not
# ---------------------------------------------------------------------------


def test_a_report_row_has_no_compression_fields_until_the_cache_is_warm(ai_state):
    """The premise, and the cold-cache answer.

    Before any worker has run, the Qt thread's merge fills NOTHING - it does not
    reach for the file to make the chip appear sooner. `warm_cache()` is what
    reads, and the next merge is what fills.
    """
    import compression_chip
    from ui.services import ai_state_levels
    from ui.services.data_feed import merge_compression_from_ai_state

    row = _row()
    assert compression_chip.read_row(row.raw) is None, "nothing to say before the merge"
    assert merge_compression_from_ai_state([row]) == 0, "a cold cache fills nothing"
    assert "compression_flag" not in row.raw

    assert ai_state_levels.warm_cache() is True, "the worker's parse, off the Qt thread"
    assert merge_compression_from_ai_state([row]) == 1

    read = compression_chip.read_row(row.raw)
    assert read is not None and read.flag is True
    assert read.score == 3
    assert read.penalty == 10
    assert read.ratios.stdev_atr == pytest.approx(0.52)
    assert read.rule_version == "anchor_compression_v1"


def test_the_qt_thread_merge_opens_nothing(ai_state):
    """The rule, as a measurement rather than a promise.

    `merge_compression_from_ai_state()` with its default arguments is what
    `enrich_setup_rows_for_display` calls, which is what
    `master_avwap_panel.refresh_from_reports` calls on the Qt thread. It may not
    `open`, `stat` or `json.load` anything - on the desk that parse is 281-292
    ms of frozen table per watched-file change.
    """
    from ui.services import ai_state_levels
    from ui.services.data_feed import merge_compression_from_ai_state

    ai_state_levels.warm_cache()

    opened: list[str] = []
    real_open = builtins.open
    real_load = json.load
    loads = {"count": 0}

    def _spy_open(file, *args, **kwargs):  # noqa: ANN001
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def _spy_load(*args, **kwargs):
        loads["count"] += 1
        return real_load(*args, **kwargs)

    row = _row()
    builtins.open = _spy_open
    json.load = _spy_load
    try:
        filled = merge_compression_from_ai_state([row])
    finally:
        builtins.open = real_open
        json.load = real_load

    assert filled == 1, "the warm cache still fills the row"
    assert loads["count"] == 0, "the Qt-thread merge parsed JSON"
    assert not [path for path in opened if "master_avwap_ai_state" in path], opened


def test_allow_read_is_the_worker_s_door_and_it_does_parse(ai_state):
    """The escape hatch exists, is explicit, and is the only way in."""
    from ui.services.data_feed import merge_compression_from_ai_state

    loads = {"count": 0}
    real_load = json.load

    def _spy_load(*args, **kwargs):
        loads["count"] += 1
        return real_load(*args, **kwargs)

    row = _row()
    json.load = _spy_load
    try:
        filled = merge_compression_from_ai_state([row], allow_read=True)
    finally:
        json.load = real_load

    assert filled == 1
    assert loads["count"] == 1, "allow_read=True is the call that reads"


def test_warming_twice_parses_once(ai_state):
    """`warm_cache` is mtime-keyed, so a burst of watcher signals costs ONE parse."""
    from ui.services import ai_state_levels

    loads = {"count": 0}
    real_load = json.load

    def _spy_load(*args, **kwargs):
        loads["count"] += 1
        return real_load(*args, **kwargs)

    json.load = _spy_load
    try:
        first = ai_state_levels.warm_cache()
        second = ai_state_levels.warm_cache()
    finally:
        json.load = real_load

    assert first is True, "the first warm read the file and the cache moved"
    assert second is False, "already warm - nothing changed, so no refresh is owed"
    assert loads["count"] == 1


def test_an_unstat_able_file_still_answers_nothing_for_the_level_feed(tmp_path, monkeypatch):
    """`load_symbol_levels` is older than this packet and keeps its contract.

    It answered `{}` when the file could not be stat-ed, and it still does: a
    caller that cannot see the file is told nothing, not told yesterday.
    """
    from ui.services import ai_state_levels

    path = _ai_state_file(tmp_path)
    monkeypatch.setattr(ai_state_levels, "MASTER_AVWAP_AI_STATE_FILE", path, raising=False)
    monkeypatch.setitem(ai_state_levels._cache, "mtime", None)
    monkeypatch.setitem(ai_state_levels._cache, "levels", {})
    monkeypatch.setitem(ai_state_levels._cache, "compression", {})
    assert ai_state_levels.load_symbol_levels()

    monkeypatch.setattr(
        ai_state_levels, "MASTER_AVWAP_AI_STATE_FILE", tmp_path / "gone.json", raising=False
    )
    assert ai_state_levels.load_symbol_levels() == {}
    assert ai_state_levels.load_symbol_compression() == {}
    # The cache itself is NOT dropped: a feed that blinked is not a reason to
    # take the chips off every row on the next warm read.
    assert ai_state_levels.cached_symbol_compression()


def test_the_merge_never_overwrites_a_reading_the_row_already_carries(ai_state):
    """The focus feed's rows come out of the scan with their own numbers; the
    merge FILLS, so a fresher row can never be overwritten by a staler file."""
    from ui.services import ai_state_levels
    from ui.services.data_feed import merge_compression_from_ai_state

    ai_state_levels.warm_cache()
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
    assert ai_state_levels.warm_cache() is False
    row = _row()
    assert merge_compression_from_ai_state([row]) == 0
    assert "compression_flag" not in row.raw

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(ai_state_levels, "MASTER_AVWAP_AI_STATE_FILE", broken, raising=False)
    assert ai_state_levels.warm_cache() is False
    assert merge_compression_from_ai_state([_row()]) == 0

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the file feed is down")

    monkeypatch.setattr(ai_state_levels, "cached_symbol_compression", _explode, raising=False)
    assert merge_compression_from_ai_state([_row()]) == 0


def test_the_load_path_is_what_merges(ai_state):
    """`enrich_setup_rows_for_display` is the seam every setups-table source
    already goes through, so no caller has to remember to ask."""
    from ui.services import ai_state_levels, data_feed

    ai_state_levels.warm_cache()
    row = _row()
    data_feed.enrich_setup_rows_for_display([row], supplemental_rows=[])
    assert row.raw.get("compression_score") == 3


def test_the_parse_runs_off_the_gui_thread(ai_state):
    """The expensive half needs no Qt at all.

    Proven by running `warm_cache` on a worker thread with no `QApplication`
    touched - which is exactly what `_AiStateCompressionWorker.run` does.
    """
    from ui.services import ai_state_levels

    result: dict[str, object] = {}

    def _work() -> None:
        try:
            result["changed"] = ai_state_levels.warm_cache()
            result["thread"] = threading.current_thread().name
        except Exception as exc:  # pragma: no cover - the failure this guards
            result["error"] = exc

    worker = threading.Thread(target=_work, name="pct3-warm-worker")
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert "error" not in result, result.get("error")
    assert result["changed"] is True
    assert result["thread"] == "pct3-warm-worker"

    rows = [_row(), _row("QUIET")]
    from ui.services.data_feed import merge_compression_from_ai_state

    assert merge_compression_from_ai_state(rows) == 2
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
    `paint` and `sizeHint` together may not open the 36 MB `ai_state` JSON.
    """
    from ui.services import ai_state_levels
    from ui.services.data_feed import merge_compression_from_ai_state

    ai_state_levels.warm_cache()
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

    ai_state_levels.warm_cache()
    row = _row()
    merge_compression_from_ai_state([row])

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the file feed is down")

    monkeypatch.setattr(ai_state_levels, "cached_symbol_compression", _explode, raising=False)
    monkeypatch.setattr(ai_state_levels, "load_symbol_compression", _explode, raising=False)
    monkeypatch.setattr(ai_state_levels, "load_symbol_levels", _explode, raising=False)
    _paint_every_cell(row)  # must not raise


def test_the_panel_s_own_refresh_parses_nothing_on_the_qt_thread(app, ai_state, tmp_path, monkeypatch):
    """The whole Qt-thread path, end to end, with the file watched.

    `refresh_from_reports` is what the report watcher, the trader's click and
    `_on_scan_finished` all reach. It starts the worker (a `stat`, not a parse)
    and then loads and merges. Nothing on this thread may `json.load` the
    ai_state file - which is the 281-292 ms the review measured.
    """
    import project_paths

    from ui.panels import master_avwap_panel as panel_module
    from ui.services import ai_state_levels

    # The panel asks `project_paths` where the file is; the fixture only moved
    # the service's own reference, so point both at the scratch file.
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_AI_STATE_FILE", ai_state, raising=False)

    started = {"count": 0}

    class _Signal:
        """Just enough of a Qt signal to be connected to and never emitted."""

        @staticmethod
        def connect(_slot):
            return None

    class _NoWorker:
        """The worker, not started - this test is about the Qt thread only.

        It carries `done` AND `finished`, because the panel connects both (the
        second is what frees the thread, advisory 4 of review round 2). A stub
        missing one of them is a test that would pass while the panel crashed.
        """

        def __init__(self, *_args, **_kwargs):
            started["count"] += 1
            self.done = _Signal()
            self.finished = _Signal()

        def start(self):
            return None

        def deleteLater(self):  # noqa: N802 (Qt spelling)
            return None

    monkeypatch.setattr(panel_module, "_AiStateCompressionWorker", _NoWorker)

    panel = panel_module.MasterAvwapPanel()
    try:
        opened: list[str] = []
        real_open = builtins.open

        def _spy_open(file, *args, **kwargs):  # noqa: ANN001
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        builtins.open = _spy_open
        try:
            panel.refresh_from_reports(emit_empty=False)
        finally:
            builtins.open = real_open

        assert started["count"] == 1, "the parse was handed to the worker"
        offenders = [path for path in opened if "master_avwap_ai_state" in path]
        assert not offenders, (
            "refresh_from_reports opened the ai_state file on the Qt thread - "
            f"that is the 281 ms stall: {offenders}"
        )
        assert ai_state_levels.cache_signature() is None, "and the cache is still cold"
    finally:
        panel.deleteLater()
