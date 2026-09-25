"""P1-4 / 4c - the permutation search on synthetic populations.

One planted key must be found; one planted lucky facet (wins in selection, not
on the last 20 sessions) must be rejected by the hold-out.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_search as search  # noqa: E402
from research_warehouse import trial_ledger  # noqa: E402

SESSIONS = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(60)]
HOLDOUT = set(SESSIONS[-search.HOLDOUT_SESSIONS:])


def _population(*, population="swing", family="avwap_band_bounce", side="LONG", horizon=1, seed=7,
                planted=True, per_session=20):
    rng = random.Random(seed)
    rows = []
    for day in SESSIONS:
        for n in range(per_session):
            key_on = rng.random() < 0.3
            lucky = rng.random() < 0.3
            if not planted:
                p = 0.5
            elif key_on:
                p = 0.85
            elif lucky:
                p = 0.3 if day in HOLDOUT else 0.9
            else:
                p = 0.4
            win = rng.random() < p
            rows.append({
                "population": population, "family": family, "side": side, "horizon": horizon,
                "session": day, "episode_id": f"{day}:{n}", "win": win, "r": 1.0 if win else -1.0,
                "f_ma_support": "sma100_support" if key_on else "no_ma_support",
                "f_lucky": "lucky" if lucky else "plain",
                "f_noise": rng.choice(["a", "b", "c"]),
                "f_weekday": "unknown",
            })
    return rows


def _families(report, population="swing", horizon="1"):
    return report["populations"][population]["horizons"][horizon]["families"]


def test_the_planted_key_is_found_and_the_lucky_facet_is_rejected(tmp_path):
    report = search.build_report(_population(), ledger_root=tmp_path)
    family = _families(report)["avwap_band_bounce LONG"]
    assert family["verdict"] == search.VERDICT_KEY
    top = family["keys"][0]
    assert top["facets"] == {"ma_support": "sma100_support"}
    assert top["depth"] == 1
    assert top["lift_pp"] > 20
    assert top["holdout"]["passed"] is True
    assert top["selection"]["n"] >= search.MIN_N and top["selection"]["sessions"] >= search.MIN_SESSIONS
    # The lucky facet wins big in selection and must not be reported alone.
    reported = [key["facets"] for key in family["keys"]]
    assert {"lucky": "lucky"} not in reported
    lucky = next(cell for cell in family["top_rejected"] if cell["facets"] == {"lucky": "lucky"})
    assert lucky["selection"]["wilson_lb"] > family["baseline"]["win_rate"]  # it WOULD pass selection
    assert lucky["reason"].startswith("hold-out")
    assert all(len(key["facets"]) <= search.MAX_DEPTH for key in family["keys"])
    # unknown is never a key value.
    assert all("unknown" not in key["facets"].values() for key in family["keys"])


def test_a_deeper_key_must_beat_its_best_parent_on_holdout(tmp_path):
    report = search.build_report(_population(), ledger_root=tmp_path)
    keys = _families(report)["avwap_band_bounce LONG"]["keys"]
    by_facets = {tuple(sorted(key["facets"].items())): key for key in keys}
    for key in keys:
        if key["depth"] == 1:
            continue
        for part in key["facets"].items():
            parent = by_facets.get((part,))
            if parent is not None:
                assert key["holdout"]["win_rate"] > parent["holdout"]["win_rate"]


def test_no_key_found_is_an_answer(tmp_path):
    report = search.build_report(_population(planted=False, seed=3), ledger_root=tmp_path)
    family = _families(report)["avwap_band_bounce LONG"]
    assert family["verdict"] == search.VERDICT_NONE
    assert family["keys"] == []
    assert family["cells_tested"] > 0


def test_a_thin_family_says_too_little_data(tmp_path):
    rows = [row for row in _population() if row["session"] in SESSIONS[:5]]
    report = search.build_report(rows, ledger_root=tmp_path)
    assert _families(report)["avwap_band_bounce LONG"]["verdict"] == search.VERDICT_THIN


def test_populations_horizons_and_sides_are_never_pooled(tmp_path):
    rows = (_population() + _population(population="m5", family="ema_15", horizon=0, seed=11)
            + _population(horizon=5, seed=12) + _population(side="SHORT", seed=13))
    report = search.build_report(rows, ledger_root=tmp_path)
    assert set(report["populations"]) == {"swing", "m5"}
    assert set(report["populations"]["swing"]["horizons"]) == {"1", "5"}
    assert set(_families(report)) == {"avwap_band_bounce LONG", "avwap_band_bounce SHORT"}
    m5 = _families(report, "m5", "0")["ema_15 LONG"]
    swing_long = _families(report)["avwap_band_bounce LONG"]
    selection_n = sum(1 for row in _population() if row["session"] not in HOLDOUT)
    assert m5["baseline"]["n"] == selection_n  # horizon 0: nothing embargoed
    # Horizon 1: the last selection session's outcome lands in the hold-out, so it is embargoed.
    embargoed = sum(1 for row in _population() if row["session"] == SESSIONS[39])
    assert swing_long["baseline"]["n"] == selection_n - embargoed
    assert report["populations"]["swing"]["horizons"]["1"]["holdout_window"] == [SESSIONS[40], SESSIONS[-1]]


def test_every_grid_is_in_the_ledger_before_its_outcomes_are_read(tmp_path, monkeypatch):
    log = []
    real_stats, real_register = search.stats_for, search.register_grid
    monkeypatch.setattr(search, "stats_for", lambda rows: (log.append("read"), real_stats(rows))[1])

    def register(root, trial):
        trial_id = real_register(root, trial)
        in_ledger = trial_id in {row["trial_id"] for row in trial_ledger.load(root)}
        log.append(("register", trial["declared_cell_count"], in_ledger))
        return trial_id

    monkeypatch.setattr(search, "register_grid", register)
    report = search.build_report(_population(), ledger_root=tmp_path)
    registrations = [entry for entry in log if isinstance(entry, tuple)]
    assert registrations and all(entry[2] for entry in registrations)
    # Two baseline reads, then each grid: its registration, then exactly 2 reads per cell.
    assert log[:2] == ["read", "read"]
    position = 2
    for entry in registrations:
        assert log[position] == entry
        cells = entry[1]
        assert log[position + 1: position + 1 + 2 * cells] == ["read"] * (2 * cells)
        position += 1 + 2 * cells
    family = _families(report)["avwap_band_bounce LONG"]
    ledger_ids = {row["trial_id"] for row in trial_ledger.load(tmp_path)}
    assert set(family["trial_ids"]) <= ledger_ids
    assert len(family["trial_ids"]) <= search.MAX_DEPTH
    # A rerun on the same data re-declares nothing new.
    before = len(trial_ledger.load(tmp_path))
    monkeypatch.setattr(search, "register_grid", real_register)
    search.build_report(_population(), ledger_root=tmp_path)
    assert len(trial_ledger.load(tmp_path)) == before


def test_the_cli_reads_parquet_and_writes_the_report(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "permutation_outcomes.parquet"
    pq.write_table(pa.Table.from_pylist(_population()), source)
    out = tmp_path / "permutation_report.json"
    assert search.main(["--outcomes", str(source), "--ledger-root", str(tmp_path / "lake"), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == search.REPORT_SCHEMA
    assert _families(report)["avwap_band_bounce LONG"]["keys"][0]["facets"] == {"ma_support": "sma100_support"}


@pytest.mark.parametrize(("wins", "n"), [(0, 10), (10, 10), (7, 30)])
def test_wilson_matches_the_desk_one(wins, n):
    from swing_headline import wilson_lower_bound

    assert search.wilson_lower_bound(wins, n) == pytest.approx(wilson_lower_bound(wins, n))


def test_selection_rows_whose_outcome_reaches_the_holdout_are_embargoed(tmp_path):
    """A 5-session outcome scanned 3 sessions before the hold-out is measured inside it."""
    assert search.embargoed_sessions(_population(horizon=5), HOLDOUT, 5) == set(SESSIONS[35:40])
    assert search.embargoed_sessions(_population(horizon=0), HOLDOUT, 0) == set()
    # Plant a leak: the embargoed sessions win every time, but only the hold-out can see why.
    rows = _population(planted=False, horizon=5, seed=5)
    for row in rows:
        if row["session"] in SESSIONS[35:40]:
            row["win"] = True
            row["f_leak"] = "leaky"
        else:
            row["f_leak"] = "clean"
    report = search.build_report(rows, ledger_root=tmp_path)
    block = report["populations"]["swing"]["horizons"]["5"]
    assert block["embargoed_sessions"] == SESSIONS[35:40]
    family = block["families"]["avwap_band_bounce LONG"]
    selection_n = sum(1 for row in rows if row["session"] not in HOLDOUT and row["session"] not in SESSIONS[35:40])
    assert family["baseline"]["n"] == selection_n
    assert all(key["facets"].get("leak") != "leaky" for key in family["keys"])
    assert all(cell["facets"].get("leak") != "leaky" for cell in family["top_rejected"])
