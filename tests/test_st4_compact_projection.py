"""Packet ST4 fix round - the compact scoring projection IS the record.

Blocker found by the reviewer on `claude/st4-first-actionable` @ `37e63b9c`.

`_build_scoring_projection` (legacy.py) writes a COMPACT projection per setup -
symbol, side, dates, context fields, and `_scoring_outcome_summary`. It carries
**no `scenarios` key at all**, and `master_avwap_tracker_scoring_snapshot.json`
on the desk holds 11,372 of them. For the LIVE scoring path that summary is not
a cache in front of the scenarios; it is the only copy of the answer.

The first ST4 build required `representative_status` in a cached summary before
trusting it and recomputed otherwise. On the snapshot the recompute found no
scenarios, returned ``tradeable_scenario_count == 0``, and every setup was
dropped: `build_recent_tracker_setup_family_rows` went 32 rows -> 0 and
`build_tracker_setup_type_rows` went 74 nonzero `score_delta` -> 0. The first D1
scan after a merge writes those through `apply_recent_tracker_setup_family_adjustments`
and `apply_tracker_setup_type_adjustments` (runner.py), so the live
`recent_tracker_score_delta` / `setup_type_score_delta` would have gone to zero.

The rule this pins: **a missing key is never a reason to recompute.** Under the
default policy with no `as_of_session`, a cached summary is taken exactly as it
was before ST4 existed. The bypass applies only to a non-default policy or a
replay, and on a record with no scenarios those answer
``representative_status == "unknown_compact"`` over the cached numbers rather
than an empty summary.

**Packet ST7 (2026-09-06, decision 0019) flipped the default to
``first_actionable_v2``, and the rule above is unchanged - only which policy is
"the default" moved.** So the bypass now fires for ``closed_first_v1`` NAMED
explicitly, and a default (v2) read of a compact record takes the cache verbatim.
That is what the ST7 tests require and what keeps the live scoring population
alive across the flip; every leg below that used to reach one arm through the
bare signature now NAMES its policy, and a DEFAULT leg was added beside it.

One consequence is pinned here rather than inferred: a pre-ST4 cache carries no
``representative_status`` AT ALL, and v2's "pending stays pending" rule keys on
that column. An ABSENT column is not a pending trade - it is a row from before
the column existed - so the aggregate reads its ``closed_setups`` instead and
counts it as ``no_representative_in_population``. Reading an absent column as
"not closed" would zero the live scoring population all over again.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402
from master_avwap_lib import selection_policy  # noqa: E402

REFERENCE_DATE = date(2026, 2, 2)


def _pre_st4_summary(*, closed: int, avg_closed_r, rep_closed_r, target_hit, stopped):
    """A cached summary shaped exactly as `main` @ `84ee24d6` wrote it.

    Deliberately WITHOUT `representative_status`, `representative_exit_date`,
    `selection_policy` or `as_of_session` - that absence is the whole test.
    """
    return {
        "representative_stop_label": "LOWER_1",
        "representative_exit_template_id": "full_band2",
        "representative_total_r": rep_closed_r,
        "representative_closed_r": rep_closed_r,
        "tradeable_scenario_count": 2,
        "open_tradeable_scenario_count": 2 - closed,
        "closed_tradeable_scenario_count": closed,
        "avg_total_r": avg_closed_r,
        "raw_avg_total_r": avg_closed_r,
        "median_total_r": avg_closed_r,
        "avg_closed_r": avg_closed_r,
        "raw_avg_closed_r": avg_closed_r,
        "open_distortion": 0.0,
        "best_total_r": avg_closed_r,
        "worst_total_r": avg_closed_r,
        "max_abs_total_r": abs(avg_closed_r or 0.0),
        "outlier_clipped": False,
        "avg_days_held": 4.0,
        "max_days_held": 4,
        "any_target_hit": target_hit,
        "any_stopped": stopped,
    }


def _compact_projection(
    *, symbol, side, anchor_date, scan_date, setup_family, closed, avg_closed_r,
    rep_closed_r, target_hit, stopped, setup_status="CLOSED",
):
    """A projection with NO `scenarios` key - exactly what the snapshot holds."""
    return {
        "symbol": symbol,
        "side": side,
        "anchor_date": anchor_date,
        "scan_date": scan_date,
        "setup_status": setup_status,
        "priority_bucket": "tracked",
        "setup_family": setup_family,
        "favorite_signals": [],
        "favorite_zone": "None",
        "compression_label": "N",
        "retest_label": "None",
        "_scoring_outcome_summary": _pre_st4_summary(
            closed=closed,
            avg_closed_r=avg_closed_r,
            rep_closed_r=rep_closed_r,
            target_hit=target_hit,
            stopped=stopped,
        ),
    }


def _snapshot_population():
    return {
        "p1": _compact_projection(
            symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
            setup_family="avwape_bounce", closed=2, avg_closed_r=1.8, rep_closed_r=1.8,
            target_hit=True, stopped=False,
        ),
        "p2": _compact_projection(
            symbol="AMD", side="LONG", anchor_date="2026-01-06", scan_date="2026-01-08",
            setup_family="avwape_bounce", closed=1, avg_closed_r=-1.0, rep_closed_r=-1.0,
            target_hit=False, stopped=True,
        ),
        "p3": _compact_projection(
            symbol="TSLA", side="SHORT", anchor_date="2026-01-09", scan_date="2026-01-12",
            setup_family="post_earnings_52w_break", closed=1, avg_closed_r=3.2,
            rep_closed_r=3.2, target_hit=True, stopped=False, setup_status="OPEN",
        ),
        "p4": _compact_projection(
            symbol="AAPL", side="LONG", anchor_date="2026-01-20", scan_date="2026-01-23",
            setup_family="post_earnings_52w_break", closed=1, avg_closed_r=1.1,
            rep_closed_r=1.1, target_hit=True, stopped=False,
        ),
    }


def test_a_compact_projection_still_produces_recent_family_rows():
    """The blocker, at the seam that feeds live scoring."""
    rows = m.build_recent_tracker_setup_family_rows(
        _snapshot_population(), reference_date=REFERENCE_DATE, lookback_days=45
    )

    assert rows, "a snapshot of compact projections must not vanish"
    assert len(rows) == 3
    by_label = {row["type_label"]: row for row in rows}
    long_bounce = by_label["LONG | tracked | family=avwape_bounce"]
    assert int(long_bounce["tracked_setups"]) == 2
    assert int(long_bounce["closed_setups"]) == 2
    assert int(long_bounce["n_wins"]) == 1
    assert int(long_bounce["n_losses"]) == 1
    # The cached numbers reached the aggregate untouched.
    assert long_bounce["representative_closed_r"] is not None


def test_a_compact_projection_still_produces_setup_type_rows_with_score_deltas():
    rows = m.build_tracker_setup_type_rows(_snapshot_population())

    assert rows, "the setup-type scoring input must not vanish either"
    assert all(int(row.get("tracked_setups", 0) or 0) > 0 for row in rows)
    # The cached R reached the aggregate. On the broken build every one of
    # these was absent because the row itself was never produced; `score_delta`
    # needs more sample than four synthetic setups can give, so the LIVE
    # reproduction on a copy of `master_avwap_tracker_scoring_snapshot.json`
    # (74 nonzero deltas on base, 0 on the broken branch) is what pins that
    # half - it is recorded in the checkpoint, not reproducible in a unit test.
    assert all(row.get("avg_total_r") is not None for row in rows)
    assert sum(int(row.get("tracked_setups", 0) or 0) for row in rows) == 4


def test_the_default_read_takes_a_pre_st4_cache_exactly_as_it_is():
    """No key was added to, removed from or changed in an old cached summary."""
    projection = _compact_projection(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
        setup_family="avwape_bounce", closed=2, avg_closed_r=1.8, rep_closed_r=1.8,
        target_hit=True, stopped=False,
    )
    cached = projection["_scoring_outcome_summary"]

    summary = m._summarize_tracker_setup_outcome(projection)

    assert summary == cached
    assert "representative_status" not in summary
    # And the explicit default is the same read.
    assert (
        m._summarize_tracker_setup_outcome(
            projection, policy=selection_policy.DEFAULT_SELECTION_POLICY
        )
        == cached
    )


def test_a_non_default_read_of_a_compact_record_is_named_not_empty():
    """A non-default policy and a replay cannot be evaluated without scenarios.

    They say so. Since ST7 the non-default policy is `closed_first_v1` by name;
    before it was `first_actionable_v2`. The RULE did not move - only which
    policy the default is - and the default read still takes the cache verbatim.
    """
    projection = _compact_projection(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
        setup_family="avwape_bounce", closed=2, avg_closed_r=1.8, rep_closed_r=1.8,
        target_hit=True, stopped=False,
    )

    for kwargs in (
        {"policy": selection_policy.SELECTION_CLOSED_FIRST_V1},
        {"as_of_session": "2026-01-20"},
        {
            "policy": selection_policy.SELECTION_FIRST_ACTIONABLE_V2,
            "as_of_session": "2026-01-20",
        },
    ):
        summary = m._summarize_tracker_setup_outcome(projection, **kwargs)
        assert int(summary["tradeable_scenario_count"]) == 2, (
            "an empty summary here is what dropped every setup"
        )
        assert summary["representative_status"] == "unknown_compact"
        assert summary["avg_closed_r"] == 1.8

    # ST7: naming the new DEFAULT is the verbatim read, never the bypass.
    named_default = m._summarize_tracker_setup_outcome(
        projection, policy=selection_policy.SELECTION_FIRST_ACTIONABLE_V2
    )
    assert named_default == projection["_scoring_outcome_summary"]
    assert "representative_status" not in named_default

    # And under a replay such an episode is neither graded nor pending; it is
    # named. (Under the default with no replay it is graded from the cache -
    # `test_a_compact_projection_still_produces_recent_family_rows`.)
    rows = m.build_recent_tracker_setup_family_rows(
        _snapshot_population(),
        reference_date=REFERENCE_DATE,
        lookback_days=45,
        as_of_session="2026-01-20",
    )
    assert rows
    for row in rows:
        n_episodes = int(row["n_episodes"])
        assert int(row["n_pending"]) == 0
        assert f"unknown_compact_in_population={n_episodes}" in str(row["excluded_reasons"])
        assert int(row["n_wins"]) == 0 and int(row["n_losses"]) == 0


def test_a_v1_REPLAY_of_a_compact_record_grades_nothing_either():
    """Re-review round.

    v1 answers a compact record straight out of its cached summary, and that
    cache was written WITHOUT any cutoff. So a v1 replay would have graded
    trades the `as_of_session` could not have seen - a confident wrong number,
    and the mirror image of v2 zeroing on the same input. `unknown_compact` is
    therefore unmeasurable under BOTH policies whenever an `as_of_session` is
    given.

    The DEFAULT read (no `as_of`) is untouched: it returns the cache verbatim,
    so `unknown_compact` never appears and every shipped number stands - the
    assertions at the end are that guard.
    """
    population = _snapshot_population()
    kwargs = {"reference_date": REFERENCE_DATE, "lookback_days": 45}

    replayed = m.build_recent_tracker_setup_family_rows(
        population,
        as_of_session="2026-01-20",
        selection_policy=selection_policy.SELECTION_CLOSED_FIRST_V1,
        **kwargs,
    )
    assert replayed, "a replay must still produce rows, just ungraded ones"
    for row in replayed:
        n_episodes = int(row["n_episodes"])
        assert row["selection_policy"] == selection_policy.SELECTION_CLOSED_FIRST_V1
        assert int(row["n_wins"]) == 0, "a v1 replay may not grade from an uncut cache"
        assert int(row["n_losses"]) == 0
        assert int(row["closed_setups"]) == 0
        assert int(row["n_pending"]) == 0, "not pending either - it is unmeasurable"
        assert f"unknown_compact_in_population={n_episodes}" in str(row["excluded_reasons"])

    # ST7: the DEFAULT replay is v2 and grades nothing either, for the same
    # reason - the cutoff cannot be applied to a summary written without one.
    default_replay = m.build_recent_tracker_setup_family_rows(
        population, as_of_session="2026-01-20", **kwargs
    )
    assert default_replay
    for row in default_replay:
        assert row["selection_policy"] == selection_policy.SELECTION_FIRST_ACTIONABLE_V2
        assert int(row["n_wins"]) == 0
        assert int(row["n_losses"]) == 0
        assert int(row["n_pending"]) == 0
        assert (
            f"unknown_compact_in_population={int(row['n_episodes'])}"
            in str(row["excluded_reasons"])
        )

    # ...and the default build over the same population still grades normally.
    default = m.build_recent_tracker_setup_family_rows(population, **kwargs)
    assert sum(int(row["n_wins"]) for row in default) == 3
    assert sum(int(row["n_losses"]) for row in default) == 1
    for row in default:
        # A pre-ST4 cache carries no `representative_status`, so the default
        # read names the gap as `no_representative_in_population` - counted,
        # kept, graded, and NEVER `unknown_compact`, which only the bypass path
        # can write. This is the line that says the default did not move.
        assert "unknown_compact" not in str(row["excluded_reasons"])
        assert (
            f"no_representative_in_population={int(row['n_episodes'])}"
            in str(row["excluded_reasons"])
        )
        assert int(row["n_excluded"]) == 0, "a named population fact is not a drop"


def test_the_compare_cli_refuses_a_scoring_snapshot(tmp_path, capsys):
    """Re-review round: a scoring snapshot is not a tracker.

    Handed one, v1 would answer every setup from a cache the cutoff never
    touched while v2 zeroes, and the report would read "v2 is broken" when the
    input was the wrong file. Refuse it by NAME, before anything is written.
    """
    import json

    import tracker_selection_compare as compare

    snapshot = tmp_path / "master_avwap_tracker_scoring_snapshot.json"
    snapshot.write_text(
        json.dumps({"data_session": "2026-02-02", "setups": _snapshot_population()}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    rc = compare.main(["--tracker", str(snapshot), "--out", str(out_dir)])

    assert rc == 2
    assert not out_dir.exists(), "a refused run writes nothing"
    message = capsys.readouterr().err
    assert "REFUSING" in message
    assert "_scoring_outcome_summary" in message
    assert "master_avwap_tracker_scoring_snapshot.json" in message

    # A real tracker record - scenarios present - is still accepted.
    tracker = tmp_path / "tracker.json"
    real = {
        "s": {
            "symbol": "AAPL", "side": "LONG", "anchor_date": "2026-01-20",
            "scan_date": "2026-01-23", "priority_bucket": "tracked",
            "setup_family": "post_earnings_52w_break", "setup_status": "CLOSED",
            "favorite_signals": [],
            "scenarios": {
                "a": {
                    "scenario_id": "a", "stop_reference_label": "LOWER_1",
                    "exit_template_id": "full_band2", "framework_family": "baseline",
                    "framework_version": "baseline", "experimental": False,
                    "tradeable": True, "status": "TARGET_HIT", "total_r": 1.1,
                    "days_held": 4, "entry_price": 100.0,
                    "initial_risk_per_share": 5.0, "initial_risk_usd": 500.0,
                    "direction": 1.0,
                    "events": [{"trade_date": "2026-01-29", "reason": "FINAL_TARGET",
                                "price": 110.0, "shares": 100}],
                }
            },
        }
    }
    tracker.write_text(
        json.dumps({"data_session": "2026-02-02", "setups": real}), encoding="utf-8"
    )
    assert compare.main(["--tracker", str(tracker), "--out", str(out_dir)]) == 0
    assert len(list(out_dir.iterdir())) == 2
