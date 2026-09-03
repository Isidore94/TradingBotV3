"""R4 fix round 1, blocker 2 - the digest's Wilson bound was computed on a
pooled-horizon n.

`master_avwap_tier_outcomes.csv` carries one row per `(scan_row_id, horizon)` -
the tracker grades every scan row at 1, 3, 5 and 10 sessions. `swing_family_records`
counted them all, so n was inflated by CORRELATED looks at one decision rather
than by independent decisions. Measured on the live file over
2026-08-06..2026-09-02: **11,097 rows over 4,433 distinct `scan_row_id`**, a 2.5x
inflation.

An inflated n makes every Wilson lower bound too TIGHT, and unevenly, so it
changes the ORDER - which is the entire output of this function. Live:
`favorite_zone_watch` pooled (1054/1829) bound 0.5535 against
`mid_earnings_ema21_retest` pooled (12/15) 0.5481, and the two swap once one
horizon is declared. This is the phone surface the trader acts on.

**The A11 fixture could not see it**: it had one row per family. Every fixture
here carries all four horizons for the same picks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import autopilot_core as core  # noqa: E402

HORIZONS = (1, 3, 5, 10)
WINDOW = ("2026-08-01", "2026-09-30")


def _csv(tmp_path, rows) -> Path:
    path = tmp_path / "tier_outcomes.csv"
    header = "scan_date,scan_row_id,setup_family,horizon_sessions,win,stale_horizon\n"
    body = "".join(
        f"{r['scan_date']},{r['scan_row_id']},{r['family']},{r['horizon']},{r['win']},{r.get('stale', 'False')}\n"
        for r in rows
    )
    path.write_text(header + body, encoding="utf-8")
    return path


def _every_horizon(family, *, wins, losses, start=0):
    """One pick graded at all four horizons - four rows, ONE decision."""
    rows = []
    index = start
    for verdict, count in (("True", wins), ("False", losses)):
        for _ in range(count):
            for horizon in HORIZONS:
                rows.append(
                    {
                        "scan_date": "2026-09-01",
                        "scan_row_id": f"{family}-{index}",
                        "family": family,
                        "horizon": horizon,
                        "win": verdict,
                    }
                )
            index += 1
    return rows


def test_only_the_declared_horizon_is_counted(tmp_path):
    """Four rows per pick is four looks at one decision, not four decisions."""
    path = _csv(tmp_path, _every_horizon("avwap_retest", wins=6, losses=4))

    records = core.swing_family_records(path, window=WINDOW)

    assert records["avwap_retest"] == {"wins": 6, "losses": 4}, (
        "pooled would be 24/16 - the same rate on a 4x n, which is the defect"
    )


def test_the_declared_horizon_is_five_and_it_is_a_named_constant(tmp_path):
    assert core.SWING_DIGEST_HORIZON_SESSIONS == 5

    rows = [
        {
            "scan_date": "2026-09-01",
            "scan_row_id": "p1",
            "family": "avwap_retest",
            "horizon": horizon,
            # Only the 5-session row is a win, so the count says which was read.
            "win": "True" if horizon == 5 else "False",
        }
        for horizon in HORIZONS
    ]

    records = core.swing_family_records(_csv(tmp_path, rows), window=WINDOW)

    assert records["avwap_retest"] == {"wins": 1, "losses": 0}


#: A family whose record DRIFTS across horizons - strong at 1/3/10, weak at 5 -
#: beside a thin one that is strong at 5. This is the live shape:
#: `favorite_zone_watch` graded 57.6% pooled and 54.1% at one declared horizon,
#: which is what moved it past, and then behind, `mid_earnings_ema21_retest`.
DRIFTING = {1: (60, 40), 3: (60, 40), 5: (46, 54), 10: (60, 40)}
THIN_BUT_GOOD_AT_FIVE = {1: (4, 4), 3: (4, 4), 5: (7, 1), 10: (4, 4)}


def _by_horizon(family, per_horizon, *, start=0):
    """One pick per row id, graded at all four horizons with drifting verdicts."""
    rows = []
    index = start
    for horizon, (wins, losses) in per_horizon.items():
        for verdict, count in (("True", wins), ("False", losses)):
            for _ in range(count):
                rows.append(
                    {
                        "scan_date": "2026-09-01",
                        "scan_row_id": f"{family}-{index}",
                        "family": family,
                        "horizon": horizon,
                        "win": verdict,
                    }
                )
                index += 1
    return rows


def test_pooling_changes_the_order_and_the_declared_horizon_fixes_it(tmp_path):
    """The live flip, in miniature. This is why the constant exists.

    Pooled, the drifting family's extra horizons carry it above the thin one;
    at the declared horizon the thin one is ahead. The RATE differs by horizon,
    so pooling is not merely a bigger sample of the same thing - it is a
    different answer.
    """
    from swing_headline import wilson_lower_bound

    rows = _by_horizon("drifting", DRIFTING) + _by_horizon(
        "thin", THIN_BUT_GOOD_AT_FIVE, start=10_000
    )
    path = _csv(tmp_path, rows)

    # Counted the way the first version counted: every horizon row.
    pooled_drifting = wilson_lower_bound(226, 400)
    pooled_thin = wilson_lower_bound(19, 32)
    assert pooled_drifting > pooled_thin

    records = core.swing_family_records(path, window=WINDOW)
    assert records["drifting"] == {"wins": 46, "losses": 54}
    assert records["thin"] == {"wins": 7, "losses": 1}

    declared_drifting = wilson_lower_bound(46, 100)
    declared_thin = wilson_lower_bound(7, 8)
    assert declared_thin > declared_drifting, "one horizon reverses the order"


def test_the_digest_order_follows_the_declared_horizon(tmp_path):
    """End to end: the ranking the trader reads on the phone."""
    rows = _by_horizon("drifting", DRIFTING) + _by_horizon(
        "thin", THIN_BUT_GOOD_AT_FIVE, start=10_000
    )
    records = core.swing_family_records(_csv(tmp_path, rows), window=WINDOW)

    payload = {
        "generated_at": "2026-09-02 10:00:00",
        "enabled": True,
        "auto_mode": "AWAY",
        "ib_status": "connected",
        "regime": "trend_up",
        "longs": [],
        "shorts": [],
        "swing_data_current": True,
        "swing_data_line": "Swing data: current session 2026-09-02 (scan)",
        "alerts": [],
        "swing_family_records": records,
        "swing_picks": [
            {"symbol": "DRIFT", "side": "LONG", "bucket": "Favorite", "family": "drifting", "expected_r": 0.1},
            {"symbol": "THIN", "side": "LONG", "bucket": "Near Favorite Zone", "family": "thin", "expected_r": 0.1},
        ],
    }

    text = core.render_away_report(payload)
    assert text.index("THIN (LONG)") < text.index("DRIFT (LONG)")


def test_a_stale_horizon_row_is_dropped_the_way_the_leaderboard_drops_it(tmp_path):
    """"5 sessions later" indexes a symbol's own scan rows, not exchange sessions."""
    rows = [
        {"scan_date": "2026-09-01", "scan_row_id": "p1", "family": "fam", "horizon": 5, "win": "True"},
        {"scan_date": "2026-09-01", "scan_row_id": "p2", "family": "fam", "horizon": 5, "win": "True", "stale": "True"},
        {"scan_date": "2026-09-01", "scan_row_id": "p3", "family": "fam", "horizon": 5, "win": "False", "stale": "True"},
    ]

    records = core.swing_family_records(_csv(tmp_path, rows), window=WINDOW)

    assert records["fam"] == {"wins": 1, "losses": 0}


def test_an_unmeasurable_staleness_is_kept_because_uncertainty_is_not_a_deletion(tmp_path):
    rows = [
        {"scan_date": "2026-09-01", "scan_row_id": "p1", "family": "fam", "horizon": 5, "win": "True", "stale": ""},
        {"scan_date": "2026-09-01", "scan_row_id": "p2", "family": "fam", "horizon": 5, "win": "False", "stale": "None"},
    ]

    records = core.swing_family_records(_csv(tmp_path, rows), window=WINDOW)

    assert records["fam"] == {"wins": 1, "losses": 1}


def test_the_live_file_really_does_carry_one_row_per_pick_per_horizon():
    """The premise, measured rather than recalled. Read-only."""
    import csv as _csv

    from project_paths import MASTER_AVWAP_TIER_OUTCOMES_FILE as path

    if not Path(path).exists():  # pragma: no cover - a fresh install
        pytest.skip("no live tier outcomes on this machine")

    rows = [
        row
        for row in _csv.DictReader(open(path, newline="", encoding="utf-8-sig"))
        if "2026-08-06" <= str(row.get("scan_date") or "")[:10] <= "2026-09-02"
    ]
    picks = {row["scan_row_id"] for row in rows}

    assert len(rows) > len(picks), "the file is one row per (pick, horizon)"
    assert {str(row.get("horizon_sessions")) for row in rows} <= {"1", "3", "5", "10"}
