"""R10.F - the LIKE cohort gets a forward record, like the vetoes already had.

Audit C1: **52 `like_claim` rows over 2 sessions and no `like_cohort_*` file.**
The veto trio has graded the trader's rejections since the cohort packet
shipped; their endorsements were never graded at all. So "were my vetoes any
good?" was answerable and "were my likes any good?" was not - which is the more
interesting of the two.

The tests below mostly assert that this MIRRORS the veto cohort rather than
paralleling it: same key, same first-of-day rule, same sideless refusal, same
delegate. A difference between the two cohorts must come from the data, never
from two implementations that drifted.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import cohorts  # noqa: E402
from ui.annotations import like_cohort  # noqa: E402

NOW = datetime(2026, 8, 24, 22, 0, 0, tzinfo=timezone.utc)


def _claim(symbol, side, *, session="2026-08-20", setup="post_earnings_52w_break"):
    return {
        "event_type": "like_claim",
        "symbol": symbol,
        "side": side,
        "session_date": session,
        "claimed_setup_id": setup,
    }


def _write_annotations(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _bars(symbol, directory, *, days=14, start=100.0):
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    stamps = pd.bdate_range("2026-08-20", periods=days)
    pd.DataFrame(
        {
            "datetime": stamps,
            "open": [start + i for i in range(days)],
            "high": [start + i + 1 for i in range(days)],
            "low": [start + i - 1 for i in range(days)],
            "close": [start + i for i in range(days)],
            "volume": [1_000_000] * days,
        }
    ).to_parquet(directory / f"{symbol}.parquet", index=False)


def _read(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ==========================================================================
# the cohort key
# ==========================================================================
def test_the_cohort_is_the_claimed_setup_not_a_reason_code():
    """A like says "I think this is a post-earnings 52w break", so the cohort
    that means anything is the one per claimed family."""
    assert like_cohort.like_cohort_source("post_earnings_52w_break") == "like_post_earnings_52w_break"


def test_a_claim_with_no_setup_id_is_its_own_cohort_not_a_dropped_row():
    """The trader liked the chart and declined to name it. That is a real
    answer and a cohort worth watching on its own."""
    assert like_cohort.like_cohort_source("") == "like_unclaimed"
    assert like_cohort.like_cohort_source(None) == "like_unclaimed"


def test_the_pick_key_matches_the_focus_tracker_so_the_math_agrees():
    """The identity the outcome math resolves a row by - date, symbol and the
    normalized side - has to be the same on both sides of the delegate.

    Since 2026-09-01 the focus key carries a FOURTH element, the category slot,
    so one name on both the swing and the M5 list gets a row for each. The
    cohort merges deliberately do NOT: their documented rule is one graded row
    per name per day (a chart claimed twice is one judgement), and widening
    their key would quietly repeal it. The head of the key is what must agree,
    and does.
    """
    from human_focus_tracking import _pick_key as focus_key
    from human_focus_tracking import pick_source_family

    row = {"trade_date": "2026-08-20", "symbol": "aapl", "side": "short"}
    assert like_cohort._pick_key(row) == focus_key(row)[:3]
    # And a like source occupies its own slot rather than being read as a
    # focus category, so the delegate keeps like rows apart from focus rows.
    assert pick_source_family("like_post_earnings_52w_break") == "like_post_earnings_52w_break"


# ==========================================================================
# the sideless refusal, kept verbatim from the veto cohort
# ==========================================================================
def test_a_sideless_claim_is_counted_and_never_graded(tmp_path):
    """`human_focus_tracking._side_label` reads a blank side as LONG, so
    grading one would manufacture a directional claim the trader never made."""
    path = tmp_path / "annotations.jsonl"
    _write_annotations(path, [_claim("AAA", "LONG"), _claim("BBB", ""), _claim("CCC", "  ")])

    rows, skipped = like_cohort.like_pick_rows(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()],
        now=NOW,
    )

    assert [row["symbol"] for row in rows] == ["AAA"]
    assert skipped == 2


def test_the_first_claim_of_the_day_wins():
    """Exactly as the veto cohort does. The annotation log keeps every claim in
    full; the cohort grades the name once."""
    rows, _skipped = like_cohort.like_pick_rows(
        [
            _claim("AAA", "LONG", setup="post_earnings_52w_break"),
            _claim("AAA", "LONG", setup="second_dev_breakout"),
        ],
        now=NOW,
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "like_post_earnings_52w_break"


# ==========================================================================
# ground rule 7 - UTC and the session, both
# ==========================================================================
def test_every_row_carries_utc_and_an_explicit_session_date():
    """The veto trio predates this rule and stamps market-local. Carrying both
    here makes the ET/PT question moot rather than answered differently in two
    places."""
    rows, _skipped = like_cohort.like_pick_rows([_claim("AAA", "LONG")], now=NOW)
    row = rows[0]

    assert row["claimed_at_utc"].endswith("+00:00")
    assert row["session_date"] == "2026-08-20"
    assert row["trade_date"] == "2026-08-20"


# ==========================================================================
# the merge
# ==========================================================================
def test_the_merge_is_idempotent_and_never_removes_a_row(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    picks = tmp_path / "like_cohort_picks.csv"
    _write_annotations(annotations, [_claim("AAA", "LONG"), _claim("BBB", "SHORT")])

    first = like_cohort.merge_like_cohort_picks(
        annotations_path=annotations, picks_path=picks, now=NOW
    )
    second = like_cohort.merge_like_cohort_picks(
        annotations_path=annotations, picks_path=picks, now=NOW
    )

    assert first["added"] == 2
    assert second["added"] == 0
    assert len(_read(picks)) == 2


# ==========================================================================
# the slot
# ==========================================================================
def test_the_slot_is_appended_after_the_veto_slot():
    """Later phases append; they never reorder."""
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    # R10.I appended `evidence_report` after this one, so the like slot is no
    # longer last - but it is still after the veto slot, which is what this
    # test exists to hold. Later phases append; they never reorder.
    assert names.index("veto_cohort_grading") < names.index("like_cohort_grading")
    assert names.index("like_cohort_grading") < names.index("evidence_report")


def test_the_slot_grades_the_history_retroactively(tmp_path):
    """Unlike the veto cohort - which writes a pick row at capture time -
    nothing has ever written a like pick, so the first run grades everything."""
    annotations = tmp_path / "annotations.jsonl"
    bars = tmp_path / "bars"
    _write_annotations(annotations, [_claim("AAA", "LONG"), _claim("BBB", "SHORT")])
    _bars("AAA", bars)
    _bars("BBB", bars)

    import ui.annotations.like_cohort as module

    original = module.merge_like_cohort_picks
    picks = tmp_path / "like_cohort_picks.csv"

    def _merge(**kwargs):
        kwargs.setdefault("annotations_path", annotations)
        return original(**kwargs)

    module.merge_like_cohort_picks = _merge
    try:
        result = cohorts.run_like_cohort_grading(
            session_date="2026-08-24",
            picks_path=picks,
            outcomes_path=tmp_path / "like_cohort_outcomes.csv",
            performance_path=tmp_path / "like_cohort_performance.csv",
            daily_bars_dir=bars,
        )
    finally:
        module.merge_like_cohort_picks = original

    assert result["status"] == "ok"
    assert result["merged"] == 2
    assert result["graded"] == 2
    rows = _read(tmp_path / "like_cohort_outcomes.csv")
    assert {row["symbol"] for row in rows} == {"AAA", "BBB"}
    # Both names rise, so the long gains and the short loses.
    by_symbol = {row["symbol"]: row for row in rows}
    assert float(by_symbol["AAA"]["h5_return"]) > 0
    assert float(by_symbol["BBB"]["h5_return"]) < 0


def test_no_claims_is_skipped_not_failed(tmp_path):
    import ui.annotations.like_cohort as module

    original = module.merge_like_cohort_picks
    module.merge_like_cohort_picks = lambda **kwargs: {
        "added": 0, "total_rows": 0, "skipped_no_side": 0, "written": True
    }
    try:
        result = cohorts.run_like_cohort_grading(
            session_date="2026-08-24",
            picks_path=tmp_path / "absent.csv",
            outcomes_path=tmp_path / "out.csv",
            performance_path=tmp_path / "perf.csv",
            daily_bars_dir=tmp_path / "bars",
        )
    finally:
        module.merge_like_cohort_picks = original

    assert result["status"] == "skipped"
    assert not (tmp_path / "out.csv").exists()


def test_the_performance_rollup_carries_the_robust_half(tmp_path):
    """R10.C: the rollup routes through `evidence_stats`, so the like cohort
    gets ground rule 10's discipline for free rather than a second definition
    of a win rate."""
    annotations = tmp_path / "annotations.jsonl"
    bars = tmp_path / "bars"
    _write_annotations(annotations, [_claim("AAA", "LONG")])
    _bars("AAA", bars)

    import ui.annotations.like_cohort as module

    original = module.merge_like_cohort_picks
    module.merge_like_cohort_picks = lambda **kwargs: original(
        annotations_path=annotations,
        picks_path=kwargs.get("picks_path"),
        now=kwargs.get("now"),
    )
    try:
        cohorts.run_like_cohort_grading(
            session_date="2026-08-24",
            picks_path=tmp_path / "picks.csv",
            outcomes_path=tmp_path / "out.csv",
            performance_path=tmp_path / "perf.csv",
            daily_bars_dir=bars,
        )
    finally:
        module.merge_like_cohort_picks = original

    rows = _read(tmp_path / "perf.csv")
    assert rows
    for column in ("median_return", "trimmed_mean_return", "ci_basis", "evidence_label"):
        assert column in rows[0]
    assert rows[0]["evidence_label"] == "discovery"


def test_the_cohort_prefix_is_registered_after_the_veto_one():
    """The order decides which base prefix claims a source, so inserting one
    above would silently re-home existing rows."""
    from human_focus_tracking import COHORT_BASE_BY_SOURCE_PREFIX

    names = [base for base, _prefix in COHORT_BASE_BY_SOURCE_PREFIX]
    assert "human_focus_like" in names
    assert names.index("human_focus_veto") < names.index("human_focus_like")
