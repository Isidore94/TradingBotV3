r"""TJ-15 items 2-4 - the nightly slot, the published pack, and the reader.

plan.md §12.4 TJ-15: *"a deterministic Stage 1 slot `miss_contrast`, after the
cohort graders"*, which *"writes one JSON pack beside the digest"* and which
Week Review later reads *"as a table under the misses"*. plan.md §12.3 and
TJ-13 item 9: a new slot appends INSIDE its decision-0018 stage, sets
`max_attempts` (never 0) and a `reserve_minutes`, and is listed in
`EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`.

**This file never starts local inference and never reads the wall clock for a
slate.** The scheduled task `TradingBotV3 AI Jobs` fires at 22:00 Pacific and
holds `local_writer_lock("ai_jobs_runner")` for hours, so the runner tests here
replace that mutex with a no-op, pass an explicit frozen `now`, name the night
kind instead of asking `night_kind()`, and point the AI store at `tmp_path`.

The contract these tests pin (the builder may add keys, never remove one):

    ai_jobs.miss_contrast.run_miss_contrast(
        *, session_date="", now=None, root=None,
        decisions=None, features=None, daily_bars=None, window_sessions=None,
    ) -> {"status", "model", "reason", "outputs": [path, ...]}

    ai_jobs.miss_contrast.read_latest(session_date, *, root=None) -> dict | None

    pack = {"schema", "session_date", "window_sessions", "real_miss_rule",
            "groups": [...], "leaders": (...), "statement": str}
    group = {"verdict", "reason_code", "label_a", "label_b",
             "n", "measured", "pending", "unmeasured", "no_point_in_time_scan",
             "misses", "correct", "rate", "low", "high", "reportable",
             "compared", "top", "statement", "features", "unmeasured_features"}
"""

from __future__ import annotations

import csv
import json
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ET = ZoneInfo("America/New_York")
#: 02:00 ET on a Wednesday: inside the trader's 01:00-09:00 window, and a fixed
#: instant, so no test here can build a slate from the clock the desk is on.
OVERNIGHT = datetime(2026, 8, 12, 2, 0, tzinfo=ET)

SLOT = "miss_contrast"
#: The slots that CLOSE decision 0018's stage 1 and OPEN stage 2 today
#: (`ai_jobs/runner.py`, measured 2026-09-19).
STAGE_TWO_AND_AFTER = (
    "ai_summary", "ticker_briefs", "market_story_narration",
    "journal_enrichment", "review_policy_draft", "setup_research",
)
COHORT_GRADERS = (
    "veto_cohort_grading", "like_cohort_grading",
    "pass_cohort_grading", "rejection_cohort_grading",
)

SESSION = "2026-09-18"
NOW = datetime(2026, 9, 19, 2, 0)
FEATURE_COLUMNS = ("run_id", "run_timestamp", "run_date", "symbol", "side",
                   "pct_from_current_vwap", "atr20")


# ---------------------------------------------------------------------------
# fixtures that keep this file off the desk's live machinery
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pin_the_desk_zone(monkeypatch):
    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")


@pytest.fixture
def ai_store(tmp_path, monkeypatch):
    """A scratch AI store. The live one is never written by a test."""
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "ai_store"
    root.mkdir()
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(root))
    return root


@pytest.fixture
def unlocked(monkeypatch):
    """No cross-process mutex.

    The nightly task holds `ai_jobs_runner` for hours; a test that waited on it
    would pass by day and hang after 22:00. The lock is not what these tests
    are about, so it is replaced rather than contended for.
    """
    import local_writer_lock as lock_mod

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)


@pytest.fixture
def window_open(monkeypatch):
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (True, "window open"))
    return window


def _registered_slot():
    from ai_jobs import runner

    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert SLOT in by_name, sorted(by_name)
    return by_name[SLOT]


# ---------------------------------------------------------------------------
# fixture data
# ---------------------------------------------------------------------------


def _sessions_ending(session: str, count: int) -> list[str]:
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _sessions_after(session: str, count: int) -> list[str]:
    from market_calendar import is_session

    cursor = date.fromisoformat(session)
    out: list[str] = []
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_session(cursor):
            out.append(cursor.isoformat())
    return out


def _daily(*, run: bool, session: str, forward: int = 5) -> list[dict]:
    """Flat history (Wilder ATR(14) exactly 2.00), then a run or a fade.

    ``forward`` is how many sessions after the decision exist on disk. TJ-11's
    D1 horizon is ``max(walkaway_day.HORIZONS)``: five is a CLOSED horizon and
    three is an open one.
    """
    flat = [
        {"dt": f"{day}T00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for day in _sessions_ending(session, 15)
    ]
    first = (100.0, 103.0, 99.9, 102.5) if run else (100.0, 100.5, 98.5, 98.6)
    rest = (102.5, 103.0, 102.0, 102.5) if run else (98.6, 99.0, 98.0, 98.5)
    after = ([first] + [rest] * 4)[:forward]
    days = _sessions_after(session, len(after))
    return flat + [
        {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}
        for day, (o, h, low, c) in zip(days, after)
    ]


def _decision(symbol: str, *, session: str, reason: str, verdict: str = "veto") -> dict:
    return {
        "session_date": session, "symbol": symbol, "side": "LONG",
        "category": "chart_review", "verdict": verdict, "source": "annotations",
        "timeframe": "D1", "stamp": f"{session}T10:05:00-04:00",
        "capture_id": f"e-{symbol}", "reason": reason, "decision_session": session,
    }


def _history(tmp_path: Path, rows) -> Path:
    path = tmp_path / "d1_features_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _scan(symbol: str, *, session: str, vwap: float, atr: float = 2.0) -> dict:
    return {
        "run_id": f"{session}-064000", "run_timestamp": f"{session}T06:40:00",
        "run_date": session, "symbol": symbol, "side": "LONG",
        "pct_from_current_vwap": vwap, "atr20": atr,
    }


def _group(pack, reason: str) -> dict:
    groups = [g for g in pack["groups"] if g.get("reason_code") == reason]
    assert len(groups) == 1, pack["groups"]
    return groups[0]


# ---------------------------------------------------------------------------
# item 2 - the slot's registration and its stage
# ---------------------------------------------------------------------------


def test_the_contrast_is_a_registered_deterministic_slot_with_a_retry_budget():
    """plan.md §12.3: a slot declares `uses_model` honestly, sets
    `max_attempts` - never 0 - and reserves a window.

    A deterministic contrast calls no model, so `uses_model` is False and there
    is no model-free half to declare: `model_free_kwargs` exists for
    `daily_digest`'s narrated second artifact and nothing here has one.
    """
    slot = _registered_slot()

    assert slot.enabled is True
    assert slot.uses_model is False, "a deterministic contrast starts no inference"
    assert slot.model_free_kwargs is None
    assert isinstance(slot.max_attempts, int) and slot.max_attempts > 0, (
        "plan.md §12.3: set max_attempts, never 0"
    )
    assert slot.reserve_minutes > 0
    assert slot.description.strip()


def test_the_contrast_sits_after_the_cohort_graders_and_ahead_of_every_narration():
    """Decision 0018: a later phase appends INSIDE its stage, never across one.

    After the graders because it reads what a decision turned out to be; ahead
    of every model slot because stage 1 finishes before stage 2 starts.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    here = names.index(SLOT)
    for grader in COHORT_GRADERS:
        assert names.index(grader) < here, f"{grader} must be graded first"
    for later in STAGE_TWO_AND_AFTER:
        assert here < names.index(later), f"{SLOT} belongs ahead of {later}"


def test_every_night_runs_the_contrast_because_the_deterministic_stage_always_runs(tmp_path):
    """Weeknight, Saturday AND Sunday (TJ-13A / decision 0018's amendment).

    Sunday is the one that bites: `slots_for("sunday")` runs stage 1 plus the
    weekend's backlog, and stage 1 is computed by walking the slate up to and
    including `runner._STAGE_ONE_LAST_SLOT` - `day_review_facts` today. A slot
    appended AFTER that name is not in stage 1 by that function's reckoning and
    silently leaves the Sunday slate, however deterministic it is.

    The kind is NAMED here rather than read from `night_kind()`: a slate built
    from the wall clock is a different test every night.
    """
    from ai_jobs import runner

    led = tmp_path / "never_written.jsonl"
    for kind in ("weeknight", "saturday"):
        names = [slot.name for slot in runner.slots_for(kind)]
        assert SLOT in names, (kind, names)
    sunday = [
        slot.name
        for slot in runner.slots_for("sunday", session_date=SESSION, ledger_path=led)
    ]
    assert SLOT in sunday, sunday
    assert "ticker_briefs" not in sunday, "sunday is stage 1 plus the backlog"


def test_a_forced_run_outside_the_window_still_does_the_contrast_and_never_fails_the_night(
    tmp_path, ai_store, unlocked, monkeypatch
):
    """Two rules at once, through the real runner.

    `--force` buys the clock for a DETERMINISTIC slot and not for one that can
    load a 14 GB model (TJ-13A item 1), so a slot that declares `uses_model`
    dishonestly records SKIPPED here instead of `ok`.

    And the home folder is EMPTY - no annotations, no features file, no daily
    store. An evidence job is never allowed to cost the night: a session with
    nothing to contrast is an `ok` row with a reason, not a failure.
    """
    from ai_jobs import runner, window

    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (False, "window closed"))

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots(
        [_registered_slot()], now=OVERNIGHT, force=True, ledger_path=led
    )

    assert len(report.results) == 1, report.results
    row = report.results[0]
    assert row["job"] == SLOT
    # LEAD AMENDMENT 2026-09-19: the tester wrote "ok", but the runner records every
    # successful FORCED run as `manual_test` (`manual = bool(force)`), pinned by
    # tests/test_ai_jobs_runner.py - a forced row never counts as session coverage.
    # What this test is about is that the forced deterministic slot RAN.
    assert row["status"] == "manual_test", row
    assert not str(row.get("error") or ""), row
    assert not str(row.get("model") or ""), "a deterministic slot names no model"


def test_the_slot_writes_its_pack_into_the_ai_store_beside_the_digest(
    tmp_path, ai_store, unlocked, window_open, monkeypatch
):
    """"one JSON pack beside the digest" - `store.digests_dir()`, resolved at
    call time so the store env points it at a scratch folder.
    """
    from ai_jobs import runner, store

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots([_registered_slot()], now=OVERNIGHT, ledger_path=led)

    row = report.results[0]
    assert row["status"] == "ok", row
    outputs = [Path(p) for p in (row.get("outputs") or ())]
    assert outputs, row
    digests = store.digests_dir()
    for path in outputs:
        assert path.parent == digests, (path, digests)
        assert path.exists()
        json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# item 3 - the pack
# ---------------------------------------------------------------------------


def _build(tmp_path, decisions, features, daily_bars, *, session=SESSION, now=NOW):
    from ai_jobs import miss_contrast

    out = miss_contrast.run_miss_contrast(
        session_date=session, now=now, root=tmp_path / "packs",
        decisions=decisions, features=features, daily_bars=daily_bars,
    )
    assert out["status"] == "ok", out
    paths = [Path(p) for p in out["outputs"]]
    assert len(paths) == 1, paths
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _population(tmp_path, *, big: int = 34, thin: int = 4):
    """`big` vetoes of one reason and `thin` of another. Two thirds ran."""
    decisions, scans, bars = [], [], {}
    for index in range(big + thin):
        reason = "extended" if index < big else "compressed"
        symbol = f"S{index:03d}"
        ran = index % 3 == 0
        decisions.append(_decision(symbol, session="2026-09-11", reason=reason))
        scans.append(_scan(symbol, session="2026-09-11", vwap=4.0 if ran else 1.0))
        bars[symbol] = _daily(run=ran, session="2026-09-11")
    return decisions, _history(tmp_path, scans), bars


def test_the_pack_names_its_rule_its_window_and_how_many_features_it_compared(tmp_path):
    """A pack that does not say which rule measured it is not evidence.

    `narrated K of N` style: every group states `observational` and the number
    of features it could compare, so "the top three" is never read as "the only
    three".
    """
    from ai_jobs import miss_contrast
    import evidence_stats
    import real_miss

    decisions, features, bars = _population(tmp_path)
    _path, pack = _build(tmp_path, decisions, features, bars)

    assert pack["session_date"] == SESSION
    assert pack["real_miss_rule"] == real_miss.REAL_MISS_V1
    assert pack["window_sessions"] == evidence_stats.LATELY_SESSIONS
    assert str(pack["schema"]).strip()

    group = _group(pack, "extended")
    assert group["verdict"] == "veto"
    assert (group["label_a"], group["label_b"]) == ("real_miss", "correct_rejection")
    assert group["compared"] >= 1
    assert "observational" in group["statement"], group["statement"]
    assert f"of {group['compared']}" in group["statement"], group["statement"]
    assert len(group["features"]) <= group["top"]
    # The reader can reach it again without re-running the night.
    again = miss_contrast.read_latest(SESSION, root=tmp_path / "packs")
    assert again is not None and again["session_date"] == SESSION


def test_a_second_run_of_the_same_session_never_rewrites_the_first_pack(tmp_path):
    """A pack is superseded, never edited - the desk's rule for every pack."""
    decisions, features, bars = _population(tmp_path)
    first, _pack = _build(tmp_path, decisions, features, bars)
    before = first.read_bytes()

    second, _again = _build(tmp_path, decisions, features, bars)

    assert first.read_bytes() == before, "the first pack was rewritten"
    assert second.exists()


def test_a_reason_under_the_floor_is_named_with_its_counts_and_never_ranked(tmp_path):
    """`MIN_REPORTABLE_N` is the floor under a NAMED finding, not a delete key.

    The thin reason keeps its row and its integer counts - hiding it would tell
    the trader their compression vetoes were never looked at - but it is absent
    from what the pack ranks and from the sentence that names a leader.
    """
    from evidence_stats import MIN_REPORTABLE_N

    decisions, features, bars = _population(tmp_path, big=MIN_REPORTABLE_N + 4, thin=4)
    _path, pack = _build(tmp_path, decisions, features, bars)

    thin = _group(pack, "compressed")
    assert thin["n"] == 4
    assert thin["measured"] == 4
    assert thin["reportable"] is False
    assert isinstance(thin["misses"], int) and isinstance(thin["correct"], int)
    assert thin["misses"] + thin["correct"] == thin["measured"]

    big = _group(pack, "extended")
    assert big["reportable"] is True

    assert tuple(pack["leaders"]) == ("extended",), pack["leaders"]
    assert "compressed" not in pack["statement"], pack["statement"]


def test_a_rate_leaves_an_open_horizon_out_of_both_halves(tmp_path):
    """TJ-11 blocker 2, inherited by every population built on that rule.

    AAA's five-session horizon has closed and it ran. BBB was decided three
    sessions ago and has faded so far, but its clock has not run out: it is
    `pending`, printed, and in NEITHER half. Pool it as a no-run and the rate
    reads 50% where the closed-only truth is 100%.
    """
    import walkaway_day

    horizon = max(walkaway_day.HORIZONS)
    closed_on, open_on = "2026-09-08", "2026-09-11"
    # 02:00 on the day after 2026-09-16, so three sessions have closed since
    # the second decision and five since the first.
    now = datetime(2026, 9, 17, 2, 0)

    decisions = [
        _decision("AAA", session=closed_on, reason="extended"),
        _decision("BBB", session=open_on, reason="extended"),
    ]
    features = _history(tmp_path, [
        _scan("AAA", session=closed_on, vwap=4.0),
        _scan("BBB", session=open_on, vwap=1.0),
    ])
    bars = {
        "AAA": _daily(run=True, session=closed_on, forward=horizon),
        "BBB": _daily(run=False, session=open_on, forward=3),
    }

    _path, pack = _build(
        tmp_path, decisions, features, bars, session="2026-09-16", now=now
    )

    group = _group(pack, "extended")
    assert group["n"] == 2
    assert group["pending"] == 1, group
    assert group["measured"] == 1, group
    assert (group["misses"], group["correct"]) == (1, 0)
    assert group["rate"] == pytest.approx(1.0)


def test_the_slot_asks_the_real_miss_rule_and_never_carries_a_copy_of_it(tmp_path, monkeypatch):
    """`REAL_MISS_V1` is ONE function (TJ-11). TJ-15 calls it.

    Two copies of "did it run" drift the day one of them is tuned, and the page
    and the pack then disagree about the same name. Replace the rule and the
    pack must move with it; a slot with its own excursion arithmetic will not.
    """
    import real_miss

    decisions, features, bars = _population(tmp_path, big=6, thin=0)
    _path, before = _build(tmp_path, decisions, features, bars)
    group = _group(before, "extended")
    assert (group["misses"], group["correct"]) == (2, 4), group

    monkeypatch.setattr(real_miss, "verdict", lambda *_a, **_k: real_miss.RUN)
    _path, after = _build(tmp_path, decisions, features, bars)
    moved = _group(after, "extended")
    assert (moved["misses"], moved["correct"]) == (6, 0), moved


def test_likes_are_contrasted_as_real_runs_against_duds(tmp_path):
    """plan.md TJ-15: *"and real runs against duds among likes"*.

    A like has no reason code - its `reason` is the trader's free-text note -
    so the likes are ONE group and its two labels are not the vetoes'.
    """
    decisions, scans, bars = [], [], {}
    for index in range(6):
        symbol = f"L{index:03d}"
        ran = index % 2 == 0
        decisions.append(
            _decision(symbol, session="2026-09-11", reason="looks clean", verdict="like")
        )
        scans.append(_scan(symbol, session="2026-09-11", vwap=4.0 if ran else 1.0))
        bars[symbol] = _daily(run=ran, session="2026-09-11")

    _path, pack = _build(tmp_path, decisions, _history(tmp_path, scans), bars)

    likes = [g for g in pack["groups"] if g["verdict"] == "like"]
    assert len(likes) == 1, pack["groups"]
    group = likes[0]
    assert group["reason_code"] == ""
    assert (group["label_a"], group["label_b"]) == ("real_run", "dud")
    assert (group["misses"], group["correct"]) == (3, 3)
