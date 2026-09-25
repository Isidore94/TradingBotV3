"""P12 (plan to 8/10): permutation report history, weak-variant verdicts and their chips.

Rank and annotate only: a verdict never hides a row, and nothing here feeds a
detector, a score, an alert, Focus, the queue or `review_policy.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT =Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import setup_key_labels  # noqa: E402
import setup_permutation_search as search  # noqa: E402
import setup_permutation_verdicts as verdicts  # noqa: E402


def _report(generated_at="2026-09-12T12:00:00+00:00", **extra):
    return {"schema": search.REPORT_SCHEMA, "generated_at": generated_at, "populations": {}, **extra}


# --- 1. report history ------------------------------------------------------


def test_history_writes_one_file_per_run_date_and_skips_unchanged_content(tmp_path):
    folder = tmp_path / "permutation_report_history"
    first = search.append_history(_report(), folder, today=date(2026, 9, 12))
    assert first == folder / "2026-09-12.json"
    assert json.loads(first.read_text(encoding="utf-8"))["schema"] == search.REPORT_SCHEMA
    # Same content, new run stamp: nothing written.
    assert search.append_history(_report("2026-09-19T12:00:00+00:00"), folder, today=date(2026, 9, 19)) is None
    changed = search.append_history(_report(source="x"), folder, today=date(2026, 9, 19))
    assert changed == folder / "2026-09-19.json"
    assert [day.isoformat() for day, _path in search.history_files(folder)] == ["2026-09-12", "2026-09-19"]


def test_history_prunes_files_older_than_600_days(tmp_path):
    folder = tmp_path / "permutation_report_history"
    folder.mkdir()
    (folder / "2024-01-06.json").write_text(json.dumps(_report(source="old")), encoding="utf-8")
    (folder / "2025-06-07.json").write_text(json.dumps(_report(source="kept")), encoding="utf-8")
    (folder / "notes.txt").write_text("not a report", encoding="utf-8")
    search.append_history(_report(source="new"), folder, today=date(2026, 9, 26))
    names = sorted(path.name for path in folder.iterdir())
    assert names == ["2025-06-07.json", "2026-09-26.json", "notes.txt"]


def test_the_cli_keeps_history_beside_its_out_and_the_live_default_is_the_constant(tmp_path, monkeypatch):
    import project_paths

    live_out = Path(project_paths.SETUP_PERMUTATION_REPORT_FILE)
    assert live_out.parent / search.HISTORY_DIR_NAME == Path(project_paths.SETUP_PERMUTATION_REPORT_HISTORY_DIR)
    monkeypatch.setattr(search, "read_outcomes", lambda _path: [])
    out = tmp_path / "scratch" / "permutation_report.json"
    assert search.main(["--outcomes", "x.parquet", "--ledger-root", str(tmp_path / "lake"), "--out", str(out)]) == 0
    assert len(search.history_files(out.parent / search.HISTORY_DIR_NAME)) == 1


# --- 2. verdicts ------------------------------------------------------------

FAMILY = "avwap_band_bounce LONG"


def _key(label, rate, n=40, passed=False):
    facets = dict(part.split("=", 1) for part in label.split(" + "))
    entry = {"label": label, "facets": facets, "holdout": {"n": n, "win_rate": rate}}
    if passed:
        entry["holdout"]["passed"] = True
    return entry


def _saturday(passes=(), fails=(), baseline=0.5, data_date=None, horizon="5"):
    """One report: swing, one horizon, one family; `passes` are keys, `fails` are top_rejected."""
    family = {
        "family": "avwap_band_bounce", "side": "LONG", "verdict": "key_found" if passes else "no_key_found",
        "holdout_baseline": {"win_rate": baseline},
        "keys": [_key(label, rate, passed=True) for label, rate in passes],
        "top_rejected": [_key(label, rate) for label, rate in fails],
    }
    block = {"families": {FAMILY: family}}
    if data_date:
        block["holdout_window"] = ["2026-08-01", data_date]
    return {"schema": search.REPORT_SCHEMA, "populations": {"swing": {"horizons": {horizon: block}}}}


def _two_horizons(first, second):
    """One report holding both reports' swing horizons."""
    merged = json.loads(json.dumps(first))
    merged["populations"]["swing"]["horizons"].update(second["populations"]["swing"]["horizons"])
    return merged


def _by_label(payload):
    return {v["label"]: v for v in payload["verdicts"]}


def test_three_saturdays_give_weak_candidate_and_none():
    reports = [
        ("2026-09-12", _saturday(passes=[("ma_support=sma50_support", 0.6)], fails=[("band_zone=vwap", 0.3)])),
        ("2026-09-19", _saturday(passes=[("ma_support=sma50_support", 0.62), ("hv_level=near_hv", 0.7)],
                                 fails=[("band_zone=vwap", 0.41), ("compression=tight", 0.35),
                                        ("weekly_ema15_hold=hold", 0.3)])),
        ("2026-09-26", _saturday(passes=[("ma_support=sma50_support", 0.58), ("compression=tight", 0.66)],
                                 fails=[("band_zone=vwap", 0.39), ("hv_level=near_hv", 0.2)])),
    ]
    out = _by_label(verdicts.build_verdicts(reports))
    # Two fails in a row (three, in fact) -> weak variant, citing the newest two.
    weak = out["band_zone=vwap"]
    assert weak["verdict"] == verdicts.WEAK and weak["streak"] == 3
    assert [c["report_date"] for c in weak["citations"]] == ["2026-09-19", "2026-09-26"]
    assert [c["holdout_win_rate"] for c in weak["citations"]] == [0.41, 0.39]
    assert "2026-09-19 41% on n=40 vs 50% baseline and 2026-09-26 39%" in weak["citation"]
    # Two passes in a row -> promotion candidate.
    assert out["ma_support=sma50_support"]["verdict"] == verdicts.CANDIDATE
    # Fail then pass, and pass then fail -> no verdict.
    assert "compression=tight" not in out and "hv_level=near_hv" not in out
    # Absent from the newest report -> no verdict.
    assert "weekly_ema15_hold=hold" not in out


def test_a_key_absent_in_one_of_the_two_reports_has_no_verdict():
    reports = [
        ("2026-09-12", _saturday(fails=[("band_zone=vwap", 0.3)])),
        ("2026-09-19", _saturday(fails=[("compression=tight", 0.3)])),
        ("2026-09-26", _saturday(fails=[("band_zone=vwap", 0.3)])),
    ]
    assert verdicts.build_verdicts(reports)["verdicts"] == []


def test_a_fail_needs_a_hold_out_below_the_baseline_on_enough_episodes():
    at_baseline = [(day, _saturday(fails=[("band_zone=vwap", 0.5)])) for day in ("2026-09-19", "2026-09-26")]
    assert verdicts.build_verdicts(at_baseline)["verdicts"] == []
    thin = [
        ("2026-09-19", {"populations": {"swing": {"horizons": {"5": {"families": {FAMILY: {
            "holdout_baseline": {"win_rate": 0.5},
            "top_rejected": [_key("band_zone=vwap", 0.1, n=verdicts.HOLDOUT_MIN_N - 1)]}}}}}}}),
        ("2026-09-26", _saturday(fails=[("band_zone=vwap", 0.1)])),
    ]
    assert verdicts.build_verdicts(thin)["verdicts"] == []
    assert verdicts.build_verdicts(at_baseline[:1])["verdicts"] == []  # one report is never a verdict


def test_the_search_writes_history_then_verdicts_beside_its_out(tmp_path, monkeypatch):
    import project_paths

    live_out = Path(project_paths.SETUP_PERMUTATION_REPORT_FILE)
    assert live_out.parent / search.VERDICTS_FILE_NAME == Path(project_paths.SETUP_PERMUTATION_VERDICTS_FILE)
    out = tmp_path / "permutation_report.json"
    history = tmp_path / search.HISTORY_DIR_NAME
    history.mkdir()
    (history / "2026-09-18.json").write_text(
        json.dumps(_saturday(fails=[("band_zone=vwap", 0.4)], data_date="2026-09-18")), encoding="utf-8")
    monkeypatch.setattr(search, "read_outcomes", lambda _path: [])
    monkeypatch.setattr(search, "build_report",
                        lambda *_a, **_k: _saturday(fails=[("band_zone=vwap", 0.3)], data_date="2026-09-25"))
    assert search.main(["--outcomes", "x", "--ledger-root", str(tmp_path / "lake"), "--out", str(out)]) == 0
    assert (history / "2026-09-25.json").is_file()  # named by the data date, not the run date
    payload = json.loads((tmp_path / search.VERDICTS_FILE_NAME).read_text(encoding="utf-8"))
    assert [v["verdict"] for v in payload["verdicts"]] == [verdicts.WEAK]
    assert payload["reports_compared"] == ["2026-09-25", "2026-09-18"]


def test_a_rerun_on_the_same_data_date_replaces_the_report_and_gives_no_verdict(tmp_path):
    folder = tmp_path / search.HISTORY_DIR_NAME
    first = _saturday(fails=[("band_zone=vwap", 0.3)], data_date="2026-09-25")
    rerun = _saturday(fails=[("band_zone=vwap", 0.31)], data_date="2026-09-25")
    assert search.append_history(first, folder, today=date(2026, 9, 26)) == folder / "2026-09-25.json"
    assert search.append_history(rerun, folder, today=date(2026, 9, 30)) == folder / "2026-09-25.json"
    assert [path.name for _day, path in search.history_files(folder)] == ["2026-09-25.json"]
    assert search.report_data_date(json.loads((folder / "2026-09-25.json").read_text("utf-8"))) == date(2026, 9, 25)
    assert verdicts.build_verdicts(verdicts.read_history(folder))["verdicts"] == []


def test_two_reports_count_only_when_their_data_dates_are_five_sessions_apart():
    def run(day, rate):
        return (day, _saturday(fails=[("band_zone=vwap", rate)], data_date=day))

    # Thursday then Friday: two data dates, one session apart - not two reports.
    assert verdicts.build_verdicts([run("2026-09-24", 0.3), run("2026-09-25", 0.3)])["verdicts"] == []
    # A report five sessions before the newest is the pair; the one in between is skipped.
    payload = verdicts.build_verdicts([run("2026-09-18", 0.35), run("2026-09-24", 0.3), run("2026-09-25", 0.32)])
    assert payload["reports_compared"] == ["2026-09-25", "2026-09-18"]
    cited = [c["report_date"] for c in payload["verdicts"][0]["citations"]]
    assert cited == ["2026-09-18", "2026-09-25"]


# --- 3. show it -------------------------------------------------------------

THREE_SATURDAYS = [
    ("2026-09-12", _saturday(passes=[("ma_support=sma50_support", 0.6)], fails=[("band_zone=vwap", 0.3)])),
    ("2026-09-19", _saturday(passes=[("ma_support=sma50_support", 0.62)], fails=[("band_zone=vwap", 0.41)])),
    ("2026-09-26", _saturday(passes=[("ma_support=sma50_support", 0.58)], fails=[("band_zone=vwap", 0.39)])),
]
WEAK_KEY = "setup_permutations.v1|avwap_band_bounce|LONG|band_zone=vwap;ma_support=sma20_support"
CANDIDATE_KEY = "setup_permutations.v1|avwap_band_bounce|LONG|band_zone=lower_1;ma_support=sma50_support"
PLAIN_KEY = "setup_permutations.v1|avwap_band_bounce|LONG|band_zone=lower_1;ma_support=sma20_support"


@pytest.fixture(autouse=True)
def _fresh_label_cache():
    setup_key_labels.reset_cache_for_tests()
    yield
    setup_key_labels.reset_cache_for_tests()


def _verdicts_payload():
    return verdicts.build_verdicts(THREE_SATURDAYS)


def _write_features_and_verdicts(folder: Path) -> Path:
    features = folder / "d1_features.csv"
    features.write_text(
        "symbol,side,last_trade_date,permutation_label,permutation_key\n"
        f"WEAK,LONG,2026-09-25,vwap|sma20_support,{WEAK_KEY}\n"
        f"CAND,LONG,2026-09-25,lower_1|sma50_support,{CANDIDATE_KEY}\n"
        f"PLAIN,LONG,2026-09-25,lower_1|sma20_support,{PLAIN_KEY}\n"
        f"SHRT,SHORT,2026-09-25,vwap,{WEAK_KEY.replace('|LONG|', '|SHORT|')}\n",
        encoding="utf-8",
    )
    (folder / setup_key_labels.VERDICTS_FILE_NAME).write_text(json.dumps(_verdicts_payload()), encoding="utf-8")
    return features


def _row(symbol, side="LONG", family="avwap_band_bounce", raw=None):
    from types import SimpleNamespace

    return SimpleNamespace(symbol=symbol, side=side, last_trade_date="2026-09-25",
                           raw={"setup_family": family, **(raw or {})})


def test_rows_carry_the_verdict_of_a_key_their_facets_contain(tmp_path):
    features = _write_features_and_verdicts(tmp_path)
    rows = [_row("WEAK"), _row("CAND"), _row("PLAIN"), _row("SHRT", side="SHORT"),
            _row("OWN", raw={"permutation_key": WEAK_KEY, "permutation_label": "vwap"})]
    setup_key_labels.attach_labels(rows, allow_read=True, path=features)
    assert [setup_key_labels.display_label(row) for row in rows] == [
        "vwap|sma20_support (weak variant h5)",
        "lower_1|sma50_support (candidate h5)",
        "lower_1|sma20_support",
        "vwap",  # a SHORT row never takes a LONG family's verdict
        "vwap (weak variant h5)",  # the row's own stamped key is enough
    ]
    tip = setup_key_labels.verdict_tooltip(rows[0])
    assert "2026-09-19 41%" in tip and "2026-09-26 39%" in tip
    # Nothing hidden, nothing rescored: the same rows, the same label column.
    assert rows[0].raw["permutation_label"] == "vwap|sma20_support"


def test_the_setups_table_shows_the_chip_and_the_citation():
    from PySide6.QtCore import Qt

    from ui.models.setup import SetupRow
    from ui.models.setup_table_model import SetupTableModel

    weak = {"verdict": verdicts.WEAK, "horizon": "5", "primary": True,
            "citation": "failed hold-out 2026-09-19 41% and 2026-09-26 39%"}
    row = SetupRow(symbol="WEAK", side="LONG", raw={"permutation_label": "vwap", "permutation_verdicts": [weak]})
    model = SetupTableModel([row])
    key_col = [key for key, _label in SetupTableModel.COLUMNS].index("setup_key")
    tags_col = [key for key, _label in SetupTableModel.COLUMNS].index("setup_tags")
    assert model.data(model.index(0, key_col)) == "vwap (weak variant h5)"
    tip = model.data(model.index(0, tags_col), Qt.ItemDataRole.ToolTipRole)
    assert "Setup key: vwap (weak variant h5)" in tip and "2026-09-26 39%" in tip


def _weak(raw):
    return {**raw, "permutation_verdicts": [{"verdict": verdicts.WEAK, "horizon": "5", "primary": True,
                                             "citation": "x"}]}


def test_one_horizon_drives_the_chip_and_the_sort_and_the_others_stay_in_the_tooltip(tmp_path):
    weak_h5 = _saturday(fails=[("band_zone=vwap", 0.3)])
    cand_h20 = _saturday(passes=[("band_zone=vwap", 0.7)], horizon="20")
    reports = [("2026-09-18", _two_horizons(weak_h5, cand_h20)), ("2026-09-25", _two_horizons(weak_h5, cand_h20))]
    payload = verdicts.build_verdicts(reports)
    assert {(v["horizon"], v["verdict"], v["primary"]) for v in payload["verdicts"]} == {
        ("5", verdicts.WEAK, True), ("20", verdicts.CANDIDATE, False)}
    features = _write_features_and_verdicts(tmp_path)
    (tmp_path / setup_key_labels.VERDICTS_FILE_NAME).write_text(json.dumps(payload), encoding="utf-8")
    rows = [_row("WEAK"), _row("A1")]
    setup_key_labels.attach_labels(rows, allow_read=True, path=features)
    assert setup_key_labels.row_chips(rows[0]) == ["weak variant h5"]
    tip = setup_key_labels.verdict_tooltip(rows[0])
    assert "weak variant h5" in tip and "candidate h20" in tip
    assert [r.symbol for r in setup_key_labels.weak_variants_last(rows, lambda r: 0)] == ["A1", "WEAK"]
    # The mirror case: candidate at h5, weak at h20 -> the candidate chip, and no demotion.
    flipped = [(day, _two_horizons(_saturday(passes=[("band_zone=vwap", 0.7)]),
                                   _saturday(fails=[("band_zone=vwap", 0.3)], horizon="20")))
               for day in ("2026-09-18", "2026-09-25")]
    (tmp_path / setup_key_labels.VERDICTS_FILE_NAME).write_text(
        json.dumps(verdicts.build_verdicts(flipped)), encoding="utf-8")
    setup_key_labels.reset_cache_for_tests()
    rows = [_row("WEAK"), _row("A1")]
    setup_key_labels.attach_labels(rows, allow_read=True, path=features)
    assert setup_key_labels.row_chips(rows[0]) == ["candidate h5"]
    assert [r.symbol for r in setup_key_labels.weak_variants_last(rows, lambda r: 0)] == ["WEAK", "A1"]


def test_a_weak_variant_sorts_after_its_family_peers_with_the_same_grade_and_is_never_hidden():
    rows = [
        _row("W1", raw=_weak({})),          # weak, family A
        _row("B1", family="fam_b"),
        _row("A1"),
        _row("W2", raw=_weak({})),          # weak, family A, keeps order after W1
        _row("A2"),
        _row("B2", family="fam_b"),
        _row("LONE", family="fam_c", raw=_weak({})),  # no peers: stays put
    ]

    def group(row):
        return (row.side, row.raw["setup_family"])

    out = setup_key_labels.weak_variants_last(rows, group)
    assert [row.symbol for row in out] == ["B1", "A1", "A2", "W1", "W2", "B2", "LONE"]
    assert sorted(id(row) for row in out) == sorted(id(row) for row in rows)
    # A different grade is not a peer: the weak row only passes same-grade rows.
    grades = {"W1": "B", "A1": "A", "A2": "B"}
    graded = [_row("A1"), _row("W1", raw=_weak({})), _row("A2")]
    out = setup_key_labels.weak_variants_last(graded, lambda r: (r.raw["setup_family"], grades[r.symbol]))
    assert [row.symbol for row in out] == ["A1", "A2", "W1"]


def test_the_setups_panel_applies_the_weak_variant_order_under_the_priority_switch(monkeypatch):
    from types import SimpleNamespace

    import working_lately
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    panel = MasterAvwapPanel.__new__(MasterAvwapPanel)
    panel.model = SimpleNamespace(grade_cell_for=lambda row: {"grade": "A"})
    rows = [_row("W1", raw=_weak({})), _row("A1"), _row("B1", family="fam_b")]
    monkeypatch.setattr(working_lately, "prioritise_enabled", lambda: True)
    assert [r.symbol for r in panel._weak_variants_last(rows)] == ["A1", "W1", "B1"]
    monkeypatch.setattr(working_lately, "prioritise_enabled", lambda: False)
    assert [r.symbol for r in panel._weak_variants_last(rows)] == ["W1", "A1", "B1"]
    source = (ROOT / "scripts" / "ui" / "panels" / "master_avwap_panel.py").read_text(encoding="utf-8")
    assert "rows = self._weak_variants_last(self._by_points(" in source


def test_the_away_digest_says_weak_variant_in_the_text():
    import autopilot_core as core

    picks = [
        {"symbol": "WEAK", "side": "LONG", "bucket": "Favorite", "family": "avwap_band_bounce",
         "expected_r": 1.0, "raw": _weak({"permutation_label": "vwap"})},
        {"symbol": "CAND", "side": "LONG", "bucket": "Favorite", "family": "avwap_band_bounce",
         "expected_r": 1.0, "raw": {"permutation_label": "lower_1", "permutation_verdicts": [
             {"verdict": verdicts.CANDIDATE, "horizon": "5", "primary": True}]}},
        {"symbol": "NOKEY", "side": "LONG", "bucket": "Favorite", "family": "avwap_band_bounce",
         "expected_r": 1.0, "raw": _weak({})},
        {"symbol": "BARE", "side": "LONG", "bucket": "Favorite", "family": "avwap_band_bounce",
         "expected_r": 1.0, "raw": {}},
    ]
    text = core.render_away_report({"auto_mode": "AWAY", "swing_picks": picks})
    line = {sym: next(ln for ln in text.splitlines() if f" {sym} " in ln and "(LONG)" in ln)
            for sym in ("WEAK", "CAND", "NOKEY", "BARE")}
    assert line["WEAK"].endswith("| key vwap (weak variant h5)")
    assert line["CAND"].endswith("| key lower_1 (candidate h5)")
    # No key: the chip is its own "|" field, so the line stays parseable.
    assert line["NOKEY"].endswith("| weak variant h5") and "| key" not in line["NOKEY"]
    assert "weak variant" not in line["BARE"] and "candidate" not in line["BARE"]


def test_the_setup_keys_panel_has_a_verdict_column_and_shows_weak_variants():
    from ui.panels import setup_keys_panel as panel_module

    assert ("verdict", "Verdict") in panel_module.COLUMNS
    report = THREE_SATURDAYS[-1][1]
    rows = panel_module.report_rows(report, "swing", "5", _verdicts_payload())
    by_key = {row["key"]: row for row in rows}
    assert by_key["ma_support=sma50_support"]["verdict"] == "candidate"
    assert "2026-09-26 58%" in by_key["ma_support=sma50_support"]["verdict_tip"]
    weak = by_key["band_zone=vwap"]
    assert weak["verdict"] == "weak variant" and weak["holdout"].startswith("failed: 39% on n=40")
    assert "2026-09-19 41%" in weak["verdict_tip"]
    # No verdicts file: the column is blank, the rows are the report's.
    assert [row["verdict"] for row in panel_module.report_rows(report, "swing", "5")] == [""]


def test_the_setup_keys_panel_reads_verdicts_off_the_qt_thread_with_the_citation_on_hover(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ui.panels import setup_keys_panel as panel_module

    report_path = tmp_path / "permutation_report.json"
    report_path.write_text(json.dumps({**THREE_SATURDAYS[-1][1], "generated_at": "x"}), encoding="utf-8")
    (tmp_path / "permutation_verdicts.json").write_text(json.dumps(_verdicts_payload()), encoding="utf-8")
    panel = panel_module.SetupKeysPanel(report_path=report_path)
    panel.refresh()
    panel._worker.wait(15000)
    for _ in range(20):
        app.processEvents()
    column = [key for key, _label in panel_module.COLUMNS].index("verdict")
    cells = {panel.table.item(i, 2).text(): panel.table.item(i, column) for i in range(panel.row_count())}
    assert cells["band_zone=vwap"].text() == "weak variant"
    assert "2026-09-26 39%" in cells["band_zone=vwap"].toolTip()
    panel.shutdown()
