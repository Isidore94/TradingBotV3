"""p9 swing table: today's work in the Master AVWAP setups table (trader, 2026-09-26).

"I want everything from today that will help me find the best swing trades in the master
avwap table. That is basically my 'swing' table."

Pins: the Long leaders rows / chip / merge (no duplicate, never in place), unknown is blank
(never 0), the regime grade / SP4 / strength / study / source cells, the Best swing order,
the regime line, the column visibility, the worker-only reads.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from ui.models import swing_columns as sc  # noqa: E402
from ui.models.setup import SetupRow  # noqa: E402


def _row(symbol, side="LONG", bucket="high_conviction", family="avwap_band_bounce", score=70.0, **raw):
    return SetupRow(symbol=symbol, side=side, score=score, bucket=bucket,
                    raw={"setup_family": family, **raw})


def _leader(symbol, setup="leader_pullback", promoted=True, rs=0.9, **extra):
    return {"symbol": symbol, "as_of": "2026-09-25", "setup": setup, "setup_family": setup,
            "sector": "Technology", "entry_limit": 101.25, "stop": 97.5,
            "exit": "hold up to 10 sessions, stop 97.50 under the pullback low",
            "status": "ready" if promoted else "listed, not promoted", "promoted": promoted,
            "market_working": "yes", "rs_percentile": rs, "reasons": ["ran 40% inside 40 sessions"],
            "strength_filter": "yes", "strength_sma50_atr": 1.25, **extra}


def _payload(*rows, as_of="2026-09-25"):
    return {"as_of": as_of, "market_working": "yes", "market_rule": "trader regime", "rows": list(rows)}


def _regime_payload(cells=None, regime="uptrend"):
    return {
        "current": {"regime": regime, "label": "Uptrend", "start_date": "2026-09-14", "day_count": 12},
        "regimes": [regime, "chop"],
        "labels": {regime: "Uptrend", "chop": "Chop"},
        "swing": cells or {},
    }


def _cell(side, bucket, family, grade, n, regime="uptrend"):
    import setup_grades

    key = setup_grades.swing_key(side, bucket, family)
    return key, {"side": side, "bucket": bucket, "family": family, "all": {"grade": "B", "n": 200},
                 "by_regime": {regime: {"grade": grade, "n": n, "win_rate": 0.6, "basis": "raw"}}}


# --------------------------------------------------------------------------- merge
def test_a_leader_already_in_the_table_is_merged_not_duplicated():
    scan = [_row("NVDA"), _row("AMD", side="SHORT", bucket="favorite_setup")]
    before = dict(scan[0].raw)
    out = sc.merge_long_leaders(scan, _payload(_leader("NVDA"), _leader("MU", promoted=False)), "2026-09-25")
    assert [r.symbol for r in out] == ["NVDA", "AMD", "MU"]
    nvda = out[0]
    assert nvda.bucket == "high_conviction", "its own bucket stays primary"
    assert sc.LONG_LEADER_BUCKET in nvda.bucket_keys and "high_conviction" in nvda.bucket_keys
    assert sc.leader_info(nvda)["entry_limit"] == 101.25
    assert scan[0].raw == before, "never merged in place"
    mu = out[2]
    assert (mu.side, mu.bucket, mu.score, mu.source) == ("LONG", sc.LONG_LEADER_BUCKET, None, sc.LEADER_SOURCE)
    assert mu.bucket_label == "Long leader"
    assert mu.key_level == "limit 101.25 · stop 97.50"


def test_a_short_only_symbol_gets_its_own_long_leader_row():
    out = sc.merge_long_leaders([_row("AMD", side="SHORT")], _payload(_leader("AMD")), "2026-09-25")
    assert [(r.symbol, r.side) for r in out] == [("AMD", "SHORT"), ("AMD", "LONG")]
    assert sc.leader_info(out[0]) is None


def test_two_setups_on_one_name_are_one_row():
    payload = _payload(_leader("NVDA"), _leader("NVDA", setup="post_earnings_drift", promoted=False))
    out = sc.merge_long_leaders([], payload, "2026-09-25")
    assert len(out) == 1
    assert sc.leader_info(out[0])["setups"] == ["leader_pullback", "post_earnings_drift"]
    assert out[0].setup_tags == ["leader pullback", "post-earnings drift"]


def test_a_stale_long_setups_payload_adds_nothing():
    stale = _payload(_leader("NVDA"), as_of="2026-09-10")
    assert [r.symbol for r in sc.merge_long_leaders([_row("AMD")], stale, "2026-09-25")] == ["AMD"]
    assert sc.merge_long_leaders([_row("AMD")], None, "2026-09-25")[0].symbol == "AMD"


def test_the_leader_age_rule_is_the_focus_age_rule():
    import long_setups

    assert sc.LEADER_MAX_AGE_DAYS == long_setups.FOCUS_MAX_AGE_DAYS


def test_a_merged_leader_row_is_exempt_from_longs_off():
    import longs_market_gate

    out = sc.merge_long_leaders([_row("NVDA")], _payload(_leader("NVDA")), "")
    assert longs_market_gate.row_is_exempt(out[0].raw)


def test_leader_cell_and_tooltip():
    row = sc.merge_long_leaders([], _payload(_leader("NVDA")), "")[0]
    assert sc.leader_text(row) == "ready · limit 101.25 · stop 97.50 · RS 90"
    tip = sc.leader_tooltip(row)
    assert "hold up to 10 sessions" in tip and "90th percentile" in tip and "ran 40%" in tip
    assert sc.leader_text(_row("X")) == ""


# --------------------------------------------------------------------------- cells: unknown is blank
def test_the_regime_grade_cell():
    key, entry = _cell("SHORT", "favorite_setup", "avwap_retest_followthrough", "A", 42)
    payload = _regime_payload({key: entry})
    short = _row("TSLA", side="SHORT", bucket="favorite_setup", family="avwap_retest_followthrough")
    assert sc.regime_grade_text(short, payload) == "A n42"
    assert sc.regime_grade_text(_row("NVDA"), payload) == "untested in this regime"
    assert sc.regime_grade_text(short, {}) == "", "no payload is blank, never a grade"
    assert sc.regime_grade_text(short, {"current": None, "swing": {key: entry}}) == ""


def test_the_regime_grade_tooltip_carries_the_path_and_the_best_exit():
    key, entry = _cell("SHORT", "favorite_setup", "avwap_retest_followthrough", "A", 42)
    row = _row("TSLA", side="SHORT", bucket="favorite_setup", family="avwap_retest_followthrough")
    context = {
        "path": {"SHORT|avwap_retest_followthrough": {"mfe_atr": 1.4, "mae_atr": -0.8, "n": 90}},
        "exits": {"SHORT|avwap_retest_followthrough": {"n": 80, "current_r": -0.04, "stop_only_r": 0.35,
                                                       "trail_r": 0.2}},
    }
    tip = sc.regime_grade_tooltip(row, _regime_payload({key: entry}), context)
    assert "Uptrend (day 12)" in tip
    assert "best +1.40 ATR, worst -0.80 ATR" in tip
    assert "stop 1 ATR, hold 10, +0.35R" in tip


def test_sp4_is_the_live_score_plus_the_family_adjust_and_blank_when_unknown():
    evidence = {"families": {"LONG|avwap_band_bounce": {"n": 120, "sessions": 30, "beat_low_h5": 0.6,
                                                         "mean_move_atr_h10": 0.5}}}
    row = _row("NVDA", score=70.0)
    assert sc.sp4_text(row, evidence) == "86.0 (+16)"  # 60 x 0.10 + 20 x 0.5 = +16
    assert sc.sp4_text(row, {}) == "", "no evidence is unknown, not the live score"
    assert sc.sp4_text(_row("MU", score=None), evidence) == ""
    assert "SHADOW" in sc.sp4_tooltip(row, evidence)


def test_strength_is_longs_only_and_blank_when_unknown():
    assert sc.strength_text(_row("A", perm_strength_filter="yes", perm_dist_sma50_atr=1.8), {}) == "yes +1.8 ATR"
    assert sc.strength_text(_row("B", perm_strength_filter="unknown"), {}) == ""
    assert sc.strength_text(_row("C"), {}) == ""
    assert sc.strength_text(_row("D", side="SHORT", perm_strength_filter="yes"), {}) == ""
    # The horizon index read on the worker fills a row the scan did not stamp.
    assert sc.strength_text(_row("E"), {"study": {"E|LONG": {"strength_filter": "no"}}}) == "no"


def test_study_tags_and_the_retired_favourite_zone_long():
    row = _row("F", bucket="", study_families="leader_pullback_long", favzone_long_retired="favorite_setup")
    assert sc.study_tags(row, {}) == ["leader pullback (study)", sc.RETIRED_TEXT]
    assert "SHORT-only" in sc.study_tooltip(row, {})
    assert sc.study_tags(_row("G"), {}) == []
    assert sc.study_tags(_row("H"), {"study": {"H|LONG": {"study_families": "band_bounce_leader_long"}}}) == [
        "band bounce leader (study)"]


def test_source_badges():
    context = {"sources": {"NVDA": ["momentum_scanner", "journal_traded"], "MU": ["journal_traded"]}}
    assert sc.source_badges(_row("NVDA"), context) == ["momentum", "traded"]
    assert sc.source_badges(_row("MU"), context) == ["traded"]
    assert sc.source_badges(_row("AMD"), context) == []


# --------------------------------------------------------------------------- Best swing
def test_best_swing_ranks_the_gate_then_promoted_leaders_and_proven_shorts_by_grade():
    key_a, entry_a = _cell("SHORT", "favorite_setup", "avwap_breakout", "A", 40)
    key_c, entry_c = _cell("SHORT", "", "avwap_retest_followthrough", "C", 35)
    payload = _regime_payload({key_a: entry_a, key_c: entry_c})
    rows = sc.merge_long_leaders(
        [
            _row("PLAIN", bucket="near_favorite_zone"),
            _row("RETEST", side="SHORT", bucket="", family="avwap_retest_followthrough"),
            _row("OTHER", side="SHORT", bucket="high_conviction", family="sma_breakout"),
            _row("FAVZ", side="SHORT", bucket="favorite_setup", family="avwap_breakout"),
        ],
        _payload(_leader("LEAD"), _leader("LIST", promoted=False)), "",
    )
    ordered = [r.symbol for r in sc.best_swing_order(rows, gate_verdict="yes", regime_payload=payload)]
    assert ordered == ["FAVZ", "RETEST", "LEAD", "PLAIN", "OTHER", "LIST"]
    closed = [r.symbol for r in sc.best_swing_order(rows, gate_verdict="no", regime_payload=payload)]
    assert closed == ["FAVZ", "RETEST", "OTHER", "PLAIN", "LEAD", "LIST"], "longs last while the gate says no"


def test_best_swing_is_stable_and_hides_nothing():
    rows = [_row(f"S{index}") for index in range(5)]
    assert sc.best_swing_order(rows, gate_verdict="", regime_payload={}) == rows


# --------------------------------------------------------------------------- the worker read
def test_path_medians_read_the_ten_session_measured_rows_only():
    rows = [
        {"side": "SHORT", "setup_family": "fam", "horizon_sessions": "10", "measured": "True",
         "mfe_atr": "1.0", "mae_atr": "-0.5"},
        {"side": "SHORT", "setup_family": "fam", "horizon_sessions": "10", "measured": "True",
         "mfe_atr": "2.0", "mae_atr": "-1.5"},
        {"side": "SHORT", "setup_family": "fam", "horizon_sessions": "10", "measured": "True",
         "mfe_atr": "3.0", "mae_atr": "-1.0"},
        {"side": "SHORT", "setup_family": "fam", "horizon_sessions": "5", "measured": "True",
         "mfe_atr": "9.0", "mae_atr": "-9.0"},
        {"side": "SHORT", "setup_family": "fam", "horizon_sessions": "10", "measured": "False",
         "mfe_atr": "", "mae_atr": ""},
    ]
    from ui.services import swing_table_context as swing_context

    assert swing_context.path_medians(rows) == {"SHORT|fam": {"mfe_atr": 2.0, "mae_atr": -1.0, "n": 3}}


def test_read_swing_context_reads_what_the_modules_publish(tmp_path, monkeypatch):
    import project_paths
    from ui.services import swing_table_context as swing_context
    from ui.services import working_lately_service as wl

    long_file = tmp_path / "long_setups.json"
    long_file.write_text(json.dumps(_payload(_leader("NVDA"))), encoding="utf-8")
    sp4_file = tmp_path / "family_side_evidence.json"
    sp4_file.write_text(json.dumps({"schema": "family_side_evidence_v1", "families": {"LONG|x": {"n": 1}}}),
                        encoding="utf-8")
    path_file = tmp_path / "swing_path_facts.csv"
    with path_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["side", "setup_family", "horizon_sessions", "measured", "mfe_atr", "mae_atr"])
        writer.writeheader()
        writer.writerow({"side": "LONG", "setup_family": "x", "horizon_sessions": "10", "measured": "True",
                         "mfe_atr": "1.5", "mae_atr": "-0.5"})
    momentum = tmp_path / "momentum.json"
    momentum.write_text(json.dumps({"schema": 1, "sessions": ["2026-09-25"],
                                    "members": {"MU": {"last_seen": "2026-09-25", "hits": 2}}}), encoding="utf-8")
    monkeypatch.setattr(project_paths, "LONG_SETUPS_FILE", long_file)
    monkeypatch.setattr(project_paths, "FAMILY_SIDE_EVIDENCE_FILE", sp4_file)
    monkeypatch.setattr(project_paths, "SWING_PATH_FACTS_FILE", path_file)
    monkeypatch.setattr(project_paths, "MOMENTUM_UNIVERSE_MEMBERSHIP_FILE", momentum)
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", tmp_path / "missing.sqlite3")
    monkeypatch.setattr(wl, "cached_exit_model_review", lambda: {"cells": [
        {"family": "x", "side": "LONG", "n": 5, "stop_only_r": 0.2}]})
    monkeypatch.setattr(wl, "cached_horizon_index", lambda: {
        ("NVDA", "LONG", "2026-09-25"): {"study_families": "leader_pullback_long", "strength_filter": "yes"},
        ("NVDA", "LONG", "2026-09-24"): {"study_families": "band_bounce_leader_long"},
    })
    swing_context._CACHE.clear()
    out = swing_context.read_swing_context("2026-09-25")
    assert out["long_setups"]["rows"][0]["symbol"] == "NVDA"
    assert out["sp4"]["families"] == {"LONG|x": {"n": 1}}
    assert out["path"] == {"LONG|x": {"mfe_atr": 1.5, "mae_atr": -0.5, "n": 1}}
    assert out["exits"]["LONG|x"]["stop_only_r"] == 0.2
    assert out["study"] == {"NVDA|LONG": {"study_families": "leader_pullback_long", "strength_filter": "yes"}}
    assert out["sources"] == {"MU": ["momentum_scanner"]}


def test_read_swing_context_with_nothing_published_is_empty_not_an_error(tmp_path, monkeypatch):
    import project_paths
    from ui.services import swing_table_context as swing_context
    from ui.services import working_lately_service as wl

    for name in ("LONG_SETUPS_FILE", "FAMILY_SIDE_EVIDENCE_FILE", "SWING_PATH_FACTS_FILE",
                 "MOMENTUM_UNIVERSE_MEMBERSHIP_FILE", "JOURNAL_DB_FILE"):
        monkeypatch.setattr(project_paths, name, tmp_path / f"{name}.missing")
    monkeypatch.setattr(wl, "cached_exit_model_review", lambda: None)
    monkeypatch.setattr(wl, "cached_horizon_index", lambda: None)
    swing_context._CACHE.clear()
    out = swing_context.read_swing_context("2026-09-25")
    assert out["long_setups"] is None
    assert all(out[name] == {} for name in ("sp4", "path", "exits", "study", "sources"))


def test_the_tracker_caches_are_peeked_never_built(monkeypatch):
    from ui.services import working_lately_service as wl

    monkeypatch.setattr(wl, "_LOOKING_BACK_CACHE", {})
    assert wl.cached_exit_model_review() is None and wl.cached_horizon_index() is None
    wl._LOOKING_BACK_CACHE["exit_model_review"] = ("key", {"cells": []})
    assert wl.cached_exit_model_review() == {"cells": []}


# --------------------------------------------------------------------------- the panel
@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _panel(monkeypatch, settings=None, rows=None):
    import project_paths
    from ui.panels import master_avwap_panel as module

    store = {} if settings is None else settings
    getter = lambda key, default=None: store.get(key, default)  # noqa: E731
    setter = lambda key, value: store.__setitem__(key, value)  # noqa: E731
    for target in (module, project_paths):
        monkeypatch.setattr(target, "get_local_setting", getter)
        monkeypatch.setattr(target, "save_local_setting", setter)
    rows = [] if rows is None else rows
    monkeypatch.setattr(module, "load_latest_setup_rows_with_meta", lambda: {
        "rows": list(rows), "data_date": "2026-09-25", "source": "focus", "is_stale": False})
    panel = module.MasterAvwapPanel(None)
    return panel, store


def _scan_rows():
    return [
        _row("NVDA"),
        _row("TSLA", side="SHORT", bucket="favorite_setup", family="avwap_breakout"),
        _row("AMD", bucket="near_favorite_zone"),
    ]


def test_the_long_leaders_chip_is_first_and_on_by_default(qapp, monkeypatch):
    from ui.panels import master_avwap_panel as module

    panel, store = _panel(monkeypatch)
    try:
        assert module.BUCKET_CHIP_KEYS[0] == sc.LONG_LEADER_BUCKET
        assert list(panel.bucket_chips)[0] == sc.LONG_LEADER_BUCKET
        assert panel.bucket_chips[sc.LONG_LEADER_BUCKET].text() == "Long leaders"
        assert sc.LONG_LEADER_BUCKET in panel.active_bucket_chip_keys()
    finally:
        panel.deleteLater()


def test_a_stored_chip_selection_gains_the_leaders_chip_once(qapp, monkeypatch):
    settings = {"qt_setups_bucket_chips": ["high_conviction"]}
    first, _ = _panel(monkeypatch, settings)
    try:
        assert first.active_bucket_chip_keys() == {"high_conviction", sc.LONG_LEADER_BUCKET}
        first.set_bucket_chips({"high_conviction"})  # the trader unchecks it
    finally:
        first.deleteLater()
    second, _ = _panel(monkeypatch, settings)
    try:
        assert second.active_bucket_chip_keys() == {"high_conviction"}, "unchecking sticks"
    finally:
        second.deleteLater()
    everything, _ = _panel(monkeypatch, {"qt_setups_bucket_chips": []})
    try:
        assert everything.active_bucket_chip_keys() == set(), "All stays All"
    finally:
        everything.deleteLater()


def test_the_context_merges_the_leaders_into_the_table_without_a_duplicate(qapp, monkeypatch):
    panel, _ = _panel(monkeypatch, rows=_scan_rows())
    try:
        panel.refresh_from_reports()
        panel.set_swing_context({"long_setups": _payload(_leader("NVDA"), _leader("MU", promoted=False))})
        symbols = [(r.symbol, r.side) for r in panel.model.rows()]
        assert symbols.count(("NVDA", "LONG")) == 1
        assert ("MU", "LONG") in symbols
        panel.set_bucket_chips({sc.LONG_LEADER_BUCKET})
        assert {r.symbol for r in panel.filtered_rows()} == {"NVDA", "MU"}
        # A re-merge from the scan's rows never stacks a second copy.
        panel.set_rows(list(panel._working_lately_source_rows))
        assert [(r.symbol, r.side) for r in panel.model.rows()].count(("NVDA", "LONG")) == 1
    finally:
        panel.deleteLater()


def test_the_best_swing_switch_orders_and_restores(qapp, monkeypatch):
    import longs_market_gate as g

    panel, store = _panel(monkeypatch, rows=_scan_rows())
    try:
        panel.refresh_from_reports()
        panel.set_swing_context({"long_setups": _payload(_leader("MU"))})
        default = [r.symbol for r in panel.model.rows()]
        assert default == ["NVDA", "TSLA", "AMD", "MU"], "the scan's order stays the default"
        panel.best_swing_toggle.setChecked(True)
        assert store["qt_setups_best_swing_sort"] is True
        assert [r.symbol for r in panel.model.rows()] == ["TSLA", "MU", "NVDA", "AMD"]
        panel.set_longs_gate(g.Verdict(day="2026-09-25", verdict="no", reason="SPY under its 20-day"))
        assert [r.symbol for r in panel.model.rows()] == ["TSLA", "NVDA", "AMD", "MU"]
        panel.best_swing_toggle.setChecked(False)
        assert [r.symbol for r in panel.model.rows()] == default
    finally:
        panel.deleteLater()


def test_the_regime_line_says_regime_longs_and_spy(qapp, monkeypatch):
    import longs_market_gate as g

    panel, _ = _panel(monkeypatch)
    try:
        panel.set_regime_grades(_regime_payload())
        panel.set_spy_regime({"as_of": "x", "symbols": {"SPY": {
            "M5": "bullish_weak", "M30": "neutral_chop", "H1": "bullish_strong", "H4": "bearish_weak",
            "D1": "bullish_weak", "W": "bullish_strong"}}})
        panel.set_longs_gate(g.Verdict(day="2026-09-25", verdict="yes"))
        assert panel.swing_regime_label.text() == (
            "Regime: Uptrend, day 12 · Longs: market working · SPY M5 Up M30 Chop H1 Up+ H4 Dn D1 Up W Up+")
        panel.longs_off_toggle.setChecked(True)
        panel.set_longs_gate(g.Verdict(day="2026-09-25", verdict="no", reason="SPY is under its 20-day"))
        assert "Longs:" not in panel.swing_regime_label.text(), "one banner: the longs-off banner says it"
        assert panel.longs_off_banner_text().startswith("Longs off")
    finally:
        panel.deleteLater()


def test_the_regime_line_with_nothing_known(qapp, monkeypatch):
    panel, _ = _panel(monkeypatch)
    try:
        panel.set_longs_gate(None)
        assert panel.swing_regime_label.text() == "Longs: market unknown"
    finally:
        panel.deleteLater()


def _visible(panel):
    return {key for column, (key, _l) in enumerate(panel.model.COLUMNS) if not panel.table.isColumnHidden(column)}


def test_swing_columns_default_set_and_the_switch(qapp, monkeypatch):
    key, entry = _cell("LONG", "high_conviction", "avwap_band_bounce", "B", 40)
    panel, store = _panel(monkeypatch, rows=_scan_rows())
    try:
        panel.resize(1640, 900)  # the desk's setups pane: nothing is squeezed out
        panel.show()
        panel.refresh_from_reports()
        panel.set_column_profile("compact")
        assert not set(sc.SWING_COLUMNS) & _visible(panel), "nothing known = no swing column"
        panel.set_regime_grades(_regime_payload({key: entry}))
        panel.set_swing_context({"long_setups": _payload(_leader("NVDA")), "sources": {"AMD": ["journal_traded"]}})
        assert set(sc.SWING_COLUMNS) & _visible(panel) == {"regime_grade"}, "compact: the regime grade only"
        panel.set_column_profile("full")
        assert set(sc.SWING_COLUMNS) & _visible(panel) == {"regime_grade", "leader", "strength", "universe"}, (
            "full: every swing column that has something to say")
        panel._swing_columns_action.setChecked(False)
        assert store["qt_setups_swing_columns"] is False
        assert not set(sc.SWING_COLUMNS) & _visible(panel)
    finally:
        panel.deleteLater()


def test_the_model_cells_and_blank_unknowns(qapp):
    from PySide6.QtCore import Qt

    from ui.models.setup_table_model import SetupTableModel

    model = SetupTableModel()
    model.set_rows(sc.merge_long_leaders([_row("NVDA"), _row("TSLA", side="SHORT")],
                                         _payload(_leader("NVDA")), ""))
    col = {k: i for i, (k, _l) in enumerate(model.COLUMNS)}

    def cell(row, key, role=Qt.ItemDataRole.DisplayRole):
        return model.data(model.index(row, col[key]), role)

    for key in sc.SWING_COLUMNS:
        assert cell(1, key) == "", f"{key}: unknown is blank"
    assert cell(0, "leader") == "ready · limit 101.25 · stop 97.50 · RS 90"
    assert cell(0, "strength") == "yes +1.2 ATR"
    assert "Long leader (leader pullback)" in cell(0, "bucket", Qt.ItemDataRole.ToolTipRole)
    assert "Long leader" in cell(0, "leader", Qt.ItemDataRole.ToolTipRole)


def test_the_delegate_paints_the_leader_chip_colour(qapp):
    from ui.widgets import setup_delegate

    assert setup_delegate._bucket_token(sc.LONG_LEADER_BUCKET) == "leader"
    merged = sc.merge_long_leaders([_row("NVDA")], _payload(_leader("NVDA")), "")[0]
    assert setup_delegate._merged_leader(merged)
    assert not setup_delegate._merged_leader(sc.merge_long_leaders([], _payload(_leader("MU")), "")[0])
    from ui import theme

    for name in ("dark", "light"):
        assert "leader" in theme.THEMES[name]


def test_nothing_is_read_on_the_qt_thread(qapp, monkeypatch):
    """The whole table path - context, rows, every cell and tooltip, the regime line - opens no file."""
    import builtins
    import io
    import threading

    from PySide6.QtCore import Qt

    key, entry = _cell("LONG", "high_conviction", "avwap_band_bounce", "B", 40)
    panel, _ = _panel(monkeypatch, rows=_scan_rows())
    try:
        panel.refresh_from_reports()
        opened: list[str] = []
        real_open, real_io_open = builtins.open, io.open

        # Only the Qt (main) thread counts: a leftover worker from an earlier test in the
        # same process may legitimately write its own file (seen: alert_chart_watches.json).
        def on_qt_thread() -> bool:
            return threading.current_thread() is threading.main_thread()

        def spy(file, *args, **kwargs):
            if on_qt_thread():
                opened.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy)
        monkeypatch.setattr(io, "open", spy)
        monkeypatch.setattr(
            Path, "read_text", lambda self, *a, **k: (opened.append(str(self)) if on_qt_thread() else None) or ""
        )
        panel.set_regime_grades(_regime_payload({key: entry}))
        panel.set_swing_context({"long_setups": _payload(_leader("NVDA"), _leader("MU")),
                                 "sp4": {"families": {"LONG|avwap_band_bounce": {"n": 99}}}})
        panel.best_swing_toggle.setChecked(True)
        panel.set_column_profile("full")
        model = panel.model
        for row in range(model.rowCount()):
            for column in range(model.columnCount()):
                for role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                    model.data(model.index(row, column), role)
        panel.refresh_swing_regime_line()
        monkeypatch.setattr(builtins, "open", real_open)
        monkeypatch.setattr(io, "open", real_io_open)
        assert opened == []
    finally:
        panel.deleteLater()


def test_the_swing_context_is_read_on_the_worker_thread(qapp, monkeypatch):
    from PySide6.QtCore import QCoreApplication

    from ui.services import swing_table_context as swing_context

    threads: list[bool] = []

    def fake_read(data_date=""):
        threads.append(threading.current_thread() is threading.main_thread())
        return {"long_setups": _payload(_leader("MU"))}

    monkeypatch.setattr(swing_context, "read_swing_context", fake_read)
    panel, _ = _panel(monkeypatch, rows=_scan_rows())
    try:
        panel._reads_swing_context = True
        panel.refresh_from_reports()
        worker = panel._swing_worker
        assert worker is not None, "the report refresh starts the read"
        assert worker.wait(5000)
        QCoreApplication.processEvents()
        assert threads == [False], "read off the Qt thread"
        assert ("MU", "LONG") in {(r.symbol, r.side) for r in panel.model.rows()}
        panel._start_swing_context_read()
        assert panel._swing_worker is None, "nothing moved: no second read"
    finally:
        panel.deleteLater()


def test_a_test_panel_reads_no_swing_context(qapp, monkeypatch):
    panel, _ = _panel(monkeypatch)
    try:
        panel.refresh_from_reports()
        assert panel._swing_worker is None
    finally:
        panel.deleteLater()


def test_the_desk_hands_the_regime_grades_to_the_swing_table():
    from unittest.mock import MagicMock

    from ui.panels.trading_desk import TradingDeskPanel

    desk = SimpleNamespace(
        _working_lately_snapshot={"setup_grades_by_regime": {"current": {"regime": "uptrend"}}},
        m5_alert_bar=MagicMock(), alert_center=MagicMock(), master_panel=MagicMock(),
        live_results_strip=MagicMock(),
    )
    TradingDeskPanel._push_working_lately_order(desk)
    desk.master_panel.set_regime_grades.assert_called_once_with({"current": {"regime": "uptrend"}})
