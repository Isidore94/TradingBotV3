"""P12 (plan to 8/10): permutation report history, weak-variant verdicts and their chips.

Rank and annotate only: a verdict never hides a row, and nothing here feeds a
detector, a score, an alert, Focus, the queue or `review_policy.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

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


def _saturday(passes=(), fails=(), baseline=0.5):
    """One report: swing, horizon 5, one family; `passes` are keys, `fails` are top_rejected."""
    family = {
        "family": "avwap_band_bounce", "side": "LONG", "verdict": "key_found" if passes else "no_key_found",
        "holdout_baseline": {"win_rate": baseline},
        "keys": [_key(label, rate, passed=True) for label, rate in passes],
        "top_rejected": [_key(label, rate) for label, rate in fails],
    }
    return {"schema": search.REPORT_SCHEMA, "populations": {
        "swing": {"horizons": {"5": {"families": {FAMILY: family}}}}}}


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
    (history / "2026-09-19.json").write_text(json.dumps(_saturday(fails=[("band_zone=vwap", 0.4)])),
                                             encoding="utf-8")
    monkeypatch.setattr(search, "read_outcomes", lambda _path: [])
    monkeypatch.setattr(search, "build_report", lambda *_a, **_k: _saturday(fails=[("band_zone=vwap", 0.3)]))
    assert search.main(["--outcomes", "x", "--ledger-root", str(tmp_path / "lake"), "--out", str(out)]) == 0
    payload = json.loads((tmp_path / search.VERDICTS_FILE_NAME).read_text(encoding="utf-8"))
    assert [v["verdict"] for v in payload["verdicts"]] == [verdicts.WEAK]
    assert len(payload["reports_read"]) == 2
