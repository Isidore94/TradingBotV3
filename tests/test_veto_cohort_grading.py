"""W1/W2/W3: the veto cohort gets graded, keyed and made readable.

``update_veto_cohort_outcomes`` shipped with the cohort packet and had **zero
callers** for its whole life: picks accumulated on every veto commit and
nothing ever graded them. These tests cover the caller, the cohort key that
now carries its vocabulary version, and the opt-in evidence scope.

Nothing here may influence a detector, a score, an alert, a watchlist, Focus,
the review queue or ``review_policy.json``. Several tests exist only to keep
that true.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import cohorts  # noqa: E402

PICK_COLUMNS = ["trade_date", "symbol", "side", "source", "snapshotted_at", "active_at_snapshot"]


def _write_picks(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PICK_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PICK_COLUMNS})


def _pick(symbol: str, side: str, *, date: str = "2026-08-03", source: str = "veto_v2_volume_dry") -> dict:
    return {
        "trade_date": date,
        "symbol": symbol,
        "side": side,
        "source": source,
        "snapshotted_at": f"{date}T09:31:00",
        "active_at_snapshot": "1",
    }


def _bars(symbol: str, directory: Path, *, days: int = 14, start: float = 100.0) -> None:
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    stamps = pd.bdate_range("2026-08-03", periods=days)
    frame = pd.DataFrame(
        {
            "datetime": stamps,
            "open": [start + i for i in range(days)],
            "high": [start + i + 1 for i in range(days)],
            "low": [start + i - 1 for i in range(days)],
            "close": [start + i for i in range(days)],
            "volume": [1_000_000] * days,
        }
    )
    frame.to_parquet(directory / f"{symbol}.parquet", index=False)


@pytest.fixture
def graded(tmp_path):
    """A picks file, bars for its symbols, and paths for the outputs."""
    picks = tmp_path / "veto_cohort_picks.csv"
    bars = tmp_path / "daily_bars"
    _write_picks(picks, [_pick("AAA", "LONG"), _pick("BBB", "SHORT")])
    _bars("AAA", bars)
    _bars("BBB", bars)
    return {
        "picks_path": picks,
        "outcomes_path": tmp_path / "veto_cohort_outcomes.csv",
        "performance_path": tmp_path / "veto_cohort_performance.csv",
        "daily_bars_dir": bars,
    }


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _without_stamp(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "updated_at"} for row in rows]


# ==========================================================================
# W1 - the nightly grading job
# ==========================================================================
def test_it_grades_the_picks_it_is_given(graded):
    result = cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    assert result["status"] == "ok"
    assert result["graded"] == 2
    assert result["skipped_no_side"] == 0
    rows = _read(graded["outcomes_path"])
    assert {row["symbol"] for row in rows} == {"AAA", "BBB"}
    assert {row["side"] for row in rows} == {"LONG", "SHORT"}


def test_a_short_is_graded_as_a_short(graded):
    """The side is the whole reason a blank one may never be guessed."""
    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    rows = {row["symbol"]: row for row in _read(graded["outcomes_path"])}
    # Both symbols rise over the horizon, so the long gains and the short loses.
    assert float(rows["AAA"]["h5_return"]) > 0
    assert float(rows["BBB"]["h5_return"]) < 0


def test_running_twice_the_same_night_changes_nothing_but_the_stamp(graded):
    """Idempotence, stated precisely.

    Byte-identical is NOT the claim and would be the wrong one: ``updated_at``
    is a provenance column and it is correct for it to say when grading last
    ran. What must not move is anything measured - the row set, the ordering,
    every entry price and every forward return. That is what this asserts, and
    it is the property that makes a re-run safe.
    """
    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    first = _read(graded["outcomes_path"])
    first_perf = _read(graded["performance_path"])

    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    second = _read(graded["outcomes_path"])

    assert _without_stamp(first) == _without_stamp(second)
    assert len(first) == len(second)
    assert _without_stamp(first_perf) == _without_stamp(_read(graded["performance_path"]))
    # And the only column that moved is the one that is allowed to.
    differing = {
        key
        for before, after in zip(first, second)
        for key in before
        if before[key] != after[key]
    }
    assert differing <= {"updated_at"}


def test_a_sideless_row_is_counted_and_never_graded(graded, tmp_path):
    """``human_focus_tracking._side_label`` reads a blank side as LONG, so
    handing it one would manufacture a directional claim the trader never
    made and fold a fabricated return into a cohort average."""
    _write_picks(
        graded["picks_path"],
        [_pick("AAA", "LONG"), _pick("BBB", ""), _pick("CCC", "   ")],
    )
    _bars("CCC", graded["daily_bars_dir"])
    result = cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)

    assert result["graded"] == 1
    assert result["skipped_no_side"] == 2
    assert "2 skipped for no side" in result["reason"]
    symbols = {row["symbol"] for row in _read(graded["outcomes_path"])}
    assert symbols == {"AAA"}, "a sideless veto must not appear as a long"


def test_the_staged_copy_is_cleaned_up(graded):
    _write_picks(graded["picks_path"], [_pick("AAA", "LONG"), _pick("BBB", "")])
    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    leftovers = list(graded["picks_path"].parent.glob("*.gradeable.tmp"))
    assert leftovers == []


def test_a_healthy_file_is_graded_without_staging_a_copy(graded, monkeypatch):
    """The common path touches no extra file."""
    staged: list = []
    real = cohorts._write_pick_subset
    monkeypatch.setattr(
        cohorts,
        "_write_pick_subset",
        lambda *a, **k: (staged.append(a[0]), real(*a, **k))[1],
    )
    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    assert staged == []


def test_no_picks_is_skipped_not_failed(tmp_path):
    result = cohorts.run_veto_cohort_grading(
        session_date="2026-08-20",
        picks_path=tmp_path / "absent.csv",
        outcomes_path=tmp_path / "out.csv",
        performance_path=tmp_path / "perf.csv",
        daily_bars_dir=tmp_path / "bars",
    )
    assert result["status"] == "skipped"
    assert not (tmp_path / "out.csv").exists()


def test_only_sideless_picks_is_skipped_and_says_so(graded):
    _write_picks(graded["picks_path"], [_pick("AAA", ""), _pick("BBB", "")])
    result = cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    assert result["status"] == "skipped"
    assert result["skipped_no_side"] == 2
    assert "never assumed" in result["reason"]


def test_a_grading_failure_leaves_the_previous_outcomes_intact(graded, monkeypatch):
    """A failed publish never destroys the last verified artifact."""
    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    good = graded["outcomes_path"].read_bytes()
    good_perf = graded["performance_path"].read_bytes()

    import ui.annotations.veto_cohort as veto_cohort

    monkeypatch.setattr(
        veto_cohort,
        "update_veto_cohort_outcomes",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("bar store exploded")),
    )
    with pytest.raises(RuntimeError):
        cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    assert graded["outcomes_path"].read_bytes() == good
    assert graded["performance_path"].read_bytes() == good_perf


def test_a_matured_pick_is_never_recomputed(graded):
    """The mechanism behind idempotence, pinned.

    ``update_human_focus_outcomes`` skips a pick whose ten forward sessions
    are all recorded, so a re-run does not re-read bars for settled history.
    Found while writing the failure test above: patching the outcome computer
    to raise did NOT raise, because these fixture picks had already matured.
    """
    import human_focus_tracking

    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    rows = _read(graded["outcomes_path"])
    assert all(row["fully_matured"] in {"1", "true", "True"} for row in rows)

    calls: list = []
    real = human_focus_tracking._compute_pick_outcome
    human_focus_tracking._compute_pick_outcome = lambda *a, **k: (
        calls.append(1),
        real(*a, **k),
    )[1]
    try:
        cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    finally:
        human_focus_tracking._compute_pick_outcome = real
    assert calls == [], "a settled pick must not be recomputed"


def test_an_unreadable_bar_store_grades_nothing_and_keeps_the_file(graded):
    cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    good = graded["outcomes_path"].read_bytes()
    graded["daily_bars_dir"] = graded["picks_path"].parent / "no_such_dir"
    result = cohorts.run_veto_cohort_grading(session_date="2026-08-20", **graded)
    assert result["status"] == "ok"
    assert graded["outcomes_path"].read_bytes() == good


def test_partition_never_guesses(graded):
    rows = [
        {"side": "LONG"}, {"side": "short"}, {"side": " SHORT "},
        {"side": ""}, {"side": None}, {}, {"side": "buy"},
    ]
    gradeable, ungradeable = cohorts.partition_by_gradeable_side(rows)
    assert len(gradeable) == 3
    assert len(ungradeable) == 4, "anything not explicitly LONG/SHORT is ungradeable"


# ---- the slot -------------------------------------------------------------
def test_the_grading_slot_is_registered_last_and_is_cheap():
    from ai_jobs.runner import default_slots

    slots = default_slots()
    names = [slot.name for slot in slots]
    assert names[-1] == "veto_cohort_grading", "appended; the runner never reorders"
    assert names[:3] == ["journal_import", "ai_summary", "ticker_briefs"]
    slot = slots[-1]
    assert slot.reserve_minutes == 5.0
    assert slot.max_attempts == 3
    assert slot.enabled


def test_the_grading_slot_calls_no_model(monkeypatch):
    """Deterministic: it must not need, or touch, the local provider."""
    import ai_summary

    monkeypatch.setattr(
        ai_summary,
        "local_provider_enabled",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("provider consulted")),
    )
    from ai_jobs.runner import default_slots

    slot = next(s for s in default_slots() if s.name == "veto_cohort_grading")
    assert slot.run is cohorts.run_veto_cohort_grading


# ==========================================================================
# W2 - the cohort key carries its vocabulary version
# ==========================================================================
def test_the_same_code_in_two_vocabularies_grades_in_two_cohorts():
    from ui.annotations.veto_cohort import veto_cohort_source

    assert veto_cohort_source("volume_dry", 1) != veto_cohort_source("volume_dry", 2)
    assert veto_cohort_source("volume_dry", 2) == "veto_v2_volume_dry"


def test_an_absent_version_keeps_the_historical_key():
    """Rows already in the picks file are never rewritten, so the unversioned
    form has to keep resolving to the cohort they were filed under."""
    from ui.annotations.veto_cohort import veto_cohort_source

    assert veto_cohort_source("volume_dry") == "veto_volume_dry"
    assert veto_cohort_source("volume_dry", None) == "veto_volume_dry"


def test_the_version_is_read_from_the_annotation():
    from ui.annotations.veto_cohort import veto_pick_rows

    rows, skipped = veto_pick_rows(
        [
            {
                "event_type": "veto",
                "symbol": "NVDA",
                "reason_code": "compressed",
                "vocab_version": 2,
                "side": "LONG",
                "session_date": "2026-08-20",
            }
        ],
        now=datetime(2026, 8, 20, 9, 31),
    )
    assert skipped == 0
    assert rows[0]["source"] == "veto_v2_compressed"


@pytest.mark.parametrize("value,expected", [(2, "v2"), ("2", "v2"), ("v2", "v2"), ("", ""), (None, ""), ("junk", "")])
def test_version_tags_parse_forgivingly(value, expected):
    """Unparseable is absent, not fatal: a cohort row is evidence, and losing
    a veto because a version field was malformed is the worse failure."""
    from ui.annotations.veto_cohort import _vocab_version_tag

    assert _vocab_version_tag(value) == expected


def test_a_blank_reason_code_still_refuses():
    from ui.annotations.veto_cohort import veto_cohort_source

    with pytest.raises(ValueError):
        veto_cohort_source("", 2)


# ==========================================================================
# W3 - the opt-in trader_judgement scope
# ==========================================================================
def test_the_scope_is_registered_but_opt_in():
    import ai_summary
    from ai_jobs import briefs

    assert "trader_judgement" in ai_summary.SCOPE_LABELS
    assert "trader_judgement" in ai_summary.SCOPE_BUDGET_WEIGHTS
    assert "trader_judgement" not in briefs.DEFAULT_SCOPES
    assert "trader_judgement" not in briefs.TICKER_BRIEF_SCOPES


def test_the_nightly_slate_is_unchanged():
    """The whole point of opt-in: the unattended run must not pick this up."""
    from ai_jobs import briefs

    assert briefs.DEFAULT_SCOPES == (
        "daily_report",
        "market_conditions",
        "setup_trackers",
        "journal_review",
    )


def test_the_scope_funds_the_distilled_answers_before_the_raw_log():
    """Same rule as setup_trackers: the raw stream last, or it starves every
    analysis derived from it."""
    import ai_summary

    specs = ai_summary._source_specs()["trader_judgement"]
    assert [source_id for source_id, _label, _path in specs] == [
        "judgement.veto_performance",
        "judgement.veto_outcomes",
        "judgement.annotations",
    ]


def test_the_scope_resolves_all_three_sources():
    import ai_summary

    package = ai_summary.build_evidence_package(["trader_judgement"])
    coverage = package["coverage"]
    seen = set(coverage["usable_source_ids"]) | {
        row["source_id"] for row in coverage["excluded"]
    }
    assert seen == {
        "judgement.veto_performance",
        "judgement.veto_outcomes",
        "judgement.annotations",
    }
    assert coverage["counts"]["requested"] == 3


def test_missing_cohort_files_degrade_rather_than_break(tmp_path, monkeypatch):
    import ai_summary

    monkeypatch.setattr(ai_summary, "VETO_COHORT_PERFORMANCE_FILE", tmp_path / "nope.csv")
    monkeypatch.setattr(ai_summary, "VETO_COHORT_OUTCOMES_FILE", tmp_path / "nope2.csv")
    package = ai_summary.build_evidence_package(["trader_judgement"])
    excluded = {
        row["source_id"]: row["status"] for row in package["coverage"]["excluded"]
    }
    assert excluded["judgement.veto_performance"] == ai_summary.SOURCE_STATUS_MISSING
    assert excluded["judgement.veto_outcomes"] == ai_summary.SOURCE_STATUS_MISSING
    assert "judgement.veto_performance" not in package["coverage"]["usable_source_ids"]


def test_the_two_caveats_travel_with_the_scope_as_data():
    """They are machine-written facts about the capture UI, in the same sense
    ``coverage`` is - not something the model is asked to infer."""
    import ai_summary

    package = ai_summary.build_evidence_package(["trader_judgement"])
    caveats = package.get("scope_caveats") or []
    joined = " ".join(caveats)
    assert len(caveats) == 2
    assert "Main swing" in joined and "user interface" in joined
    # The picklist widened on 2026-08-21; a caveat that still described the
    # old, narrower control would be a machine-written falsehood shipped as
    # data - the one thing this scope's caveats exist to prevent.
    assert "2nd-Dev Breakout" in joined and "Post-Earnings" in joined
    assert "Veto D1 - but M5 today" in joined and "traded the same day" in joined


def test_other_scopes_carry_no_caveats():
    import ai_summary

    package = ai_summary.build_evidence_package(["daily_report"])
    assert "scope_caveats" not in package


def test_the_scope_can_be_selected_on_demand():
    """A weekend exercise must not require editing the nightly default."""
    from ai_jobs.runner import default_slots

    slots = default_slots(summary_scopes=("trader_judgement",))
    assert [slot.name for slot in slots] == [
        "journal_import",
        "ai_summary",
        "ticker_briefs",
        "veto_cohort_grading",
    ]
    # And the override is per-call: building again without it is untouched.
    assert default_slots()[1].run.__name__ == "run_daily_summary"


def test_an_unknown_scope_is_rejected_at_the_cli():
    import run_ai_jobs

    with pytest.raises(SystemExit):
        run_ai_jobs.main(["--scopes", "not_a_scope"])
