r"""TJ-16 item 3 - the deterministic Stage 1 slot `prediction_contrast`.

`plan.md` §12.4 TJ-16 item 3: *"right against wrong over `LATELY_SESSIONS` and
over the whole ledger, per context field - counts, the Wilson interval,
`observational, top 3 of K`, nothing named under the floor, separately for
`Rest of day` and `Next 5 sessions`. No model."*

**The contrast is TJ-15's, called.** `evidence_contrast.contrast` already holds
the rank key, the two floors and the sentence; a second contrast would be a
second answer to the same question.

This file never starts local inference and never reads the wall clock for a
slate: the runner tests replace the machine mutex, pass a frozen `now` and NAME
the night kind.

The contract these tests pin (the builder may add keys, never remove one):

    ai_jobs.prediction_contrast.run_prediction_contrast(
        *, session_date="", now=None, root=None, reads_root=None, rows=None,
        tags=None, window_sessions=None, min_side=None, min_total=None,
    ) -> {"status", "model", "reason", "outputs": [path, ...]}

    ai_jobs.prediction_contrast.build_pack(session_date, *, now=None, rows=None,
        tags=None, window_sessions=..., min_side=None, min_total=None, notes=())
    ai_jobs.prediction_contrast.read_latest(session_date, *, root=None)
    ai_jobs.prediction_contrast.tendencies(pack, *, limit=3) -> list[dict]

    pack = {"schema", "session_date", "source", "window_sessions", "reads",
            "empty", "statement", "excluded_by_source", "horizons"}
    horizon = {"n", "right", "wrong", "flat", "pending", "unmeasured",
               "lately": <evidence_contrast.contrast()>,
               "all": <evidence_contrast.contrast()>,
               "tables": {"by_hour": [cell], "by_environment": [cell]}}
    cell = {"key", "n", "right", "wrong", "flat", "rate", "low", "high",
            "reportable"}

    FEATURE NAMES. A numeric context field keeps its own name (`hour`,
    `gap_pct`). A CATEGORICAL one becomes one feature per value,
    ``f"{field}:{value}"``, worth 1.0 on a row holding that value and 0.0 on a
    row holding a DIFFERENT MEASURED value - and NOTHING AT ALL on a row whose
    field reads `unmeasured`, because a field the desk could not measure is not
    a zero (plan.md sec 5).
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj16_support as fx  # noqa: E402

ET = ZoneInfo("America/New_York")
#: 02:00 ET the night after the fixture's last session - inside the trader's
#: 01:00-09:00 window, and a FIXED instant, so no slate here is built from the
#: clock the desk happens to be on. `runner.session_date_for` reads 2026-09-18
#: from it, which is the session the fixture ledger ends on.
OVERNIGHT = datetime(2026, 9, 19, 2, 0, tzinfo=ET)

SLOT = "prediction_contrast"
#: The slot TJ-15 registered; item 3 says this one goes directly after it.
BEFORE_ME = "miss_contrast"
#: The pair that CLOSES decision 0018's stage 1 today. WS-10D and WS-RP each pin
#: `measured_report` directly after `market_story_rollups`, so a new stage 1
#: slot goes ABOVE the pair, never between them and never after them (a slot
#: after `runner._STAGE_ONE_LAST_SLOT` silently leaves the Sunday slate).
STAGE_ONE_CLOSERS = ("market_story_rollups", "measured_report")
STAGE_TWO_AND_AFTER = (
    "ai_summary", "ticker_briefs", "market_story_narration",
    "journal_enrichment", "review_policy_draft", "setup_research",
)

NOW = fx.morning_after(fx.LAST_SESSION)


# ---------------------------------------------------------------------------
# fixtures that keep this file off the desk's live machinery
# ---------------------------------------------------------------------------
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
    """No cross-process mutex: the nightly task holds it for hours."""
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


def _pack(tmp_path, rows, *, session=fx.LAST_SESSION, now=NOW, **kwargs):
    from ai_jobs import prediction_contrast

    out = prediction_contrast.run_prediction_contrast(
        session_date=session, now=now, root=tmp_path / "packs", rows=list(rows), **kwargs
    )
    assert out["status"] == "ok", out
    paths = [Path(item) for item in out["outputs"]]
    assert len(paths) == 1, paths
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _day(pack: Mapping[str, Any]) -> Mapping[str, Any]:
    return pack["horizons"]["rest_of_day"]


def _feature(contrast: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = [row for row in contrast["features"] if row.get("feature") == name]
    assert len(rows) == 1, [row.get("feature") for row in contrast["features"]]
    return rows[0]


def _table_cell(horizon: Mapping[str, Any], table: str, key: Any) -> Mapping[str, Any]:
    cells = [row for row in horizon["tables"][table] if row.get("key") == key]
    assert len(cells) == 1, horizon["tables"][table]
    return cells[0]


# ---------------------------------------------------------------------------
# the slot's registration and its place in the night
# ---------------------------------------------------------------------------
def test_the_prediction_contrast_is_a_registered_deterministic_slot_with_a_budget():
    """plan.md §12.3: a slot declares `uses_model` honestly, sets
    `max_attempts` - never 0 - and reserves a window. Right against wrong is
    arithmetic; it starts no inference and names no model.
    """
    slot = _registered_slot()

    assert slot.enabled is True
    assert slot.uses_model is False, "a deterministic contrast starts no inference"
    assert slot.model_free_kwargs is None
    assert isinstance(slot.max_attempts, int) and slot.max_attempts > 0
    assert slot.reserve_minutes > 0
    assert slot.description.strip()


def test_the_prediction_contrast_sits_directly_after_the_miss_contrast():
    """Item 3: *"a deterministic Stage 1 slot `prediction_contrast`, after
    `miss_contrast`"*.

    Pinned against its NEIGHBOURS rather than by index: three other packets are
    appending slots to this slate in the same wave. It stays ABOVE the pair
    that closes stage 1 - `runner._STAGE_ONE_LAST_SLOT` is `measured_report`
    and `_deterministic_stage` walks up to and INCLUDING it, so a slot appended
    below that name leaves the Sunday slate however deterministic it is.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    here = names.index(SLOT)
    assert names[here - 1] == BEFORE_ME, names
    for closer in STAGE_ONE_CLOSERS:
        assert here < names.index(closer), f"{SLOT} belongs above {closer}"
    for later in STAGE_TWO_AND_AFTER:
        assert here < names.index(later), f"{SLOT} belongs ahead of {later}"
    # The pair the two report packets pin stays adjacent and in its order.
    assert names[names.index("measured_report") - 1] == "market_story_rollups", names


def test_every_night_runs_the_prediction_contrast_including_sunday(tmp_path):
    """Sunday is the one that bites: its slate is stage 1 plus the weekend's
    backlog, computed by walking the slate up to `_STAGE_ONE_LAST_SLOT`.

    The night kind is NAMED here rather than read from `night_kind()`: a slate
    built from the wall clock is a different test every night.
    """
    from ai_jobs import runner

    led = tmp_path / "never_written.jsonl"
    for kind in ("weeknight", "saturday"):
        assert SLOT in [slot.name for slot in runner.slots_for(kind)], kind
    sunday = [
        slot.name
        for slot in runner.slots_for(
            "sunday", session_date=fx.LAST_SESSION, ledger_path=led
        )
    ]
    assert SLOT in sunday, sunday
    assert "ticker_briefs" not in sunday, "sunday is stage 1 plus the backlog"


def test_a_forced_run_with_an_empty_ledger_records_a_manual_row_and_no_failure(
    tmp_path, ai_store, unlocked, monkeypatch
):
    """Two rules at once, through the REAL runner.

    `--force` buys the clock for a deterministic slot (TJ-13A item 1) and the
    row it writes is `manual_test`, never `ok`: a deliberate run publishes real
    artifacts and never counts as the session's coverage.

    And the ledger is EMPTY - which is the state the desk is in on 2026-09-20,
    the day the click card shipped. An evidence job is never allowed to cost the
    thing it records: a session with nothing to contrast is a recorded reason,
    not a failed night.
    """
    import project_paths
    from ai_jobs import runner, window

    monkeypatch.setattr(project_paths, "DAY_REVIEW_READS_DIR", tmp_path / "dr" / "reads")
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (False, "window closed"))

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots(
        [_registered_slot()], now=OVERNIGHT, force=True, ledger_path=led
    )

    assert len(report.results) == 1, report.results
    row = report.results[0]
    assert row["job"] == SLOT
    assert row["status"] == "manual_test", row
    assert not str(row.get("error") or ""), row
    assert not str(row.get("model") or ""), "a deterministic slot names no model"


def test_the_slot_reads_the_ledger_from_disk_and_writes_its_pack_by_the_digest(
    tmp_path, ai_store, unlocked, window_open, monkeypatch
):
    """The whole path, through the real runner: the store TJ-10 writes, this
    slot's own reader, one JSON pack in `store.digests_dir()`, and `read_latest`
    finding it again without re-running the night.
    """
    import project_paths
    from ai_jobs import prediction_contrast, runner, store

    sessions, rows = fx.two_weeks_of_clicks()
    ledger_root = tmp_path / "day_review"
    fx.store_ledger(ledger_root, rows)
    monkeypatch.setattr(project_paths, "DAY_REVIEW_READS_DIR", ledger_root / "reads")

    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots(
        [_registered_slot()], now=OVERNIGHT, ledger_path=led, only=SLOT
    )
    row = report.results[0]
    assert row["status"] == "ok", row

    outputs = [Path(item) for item in (row.get("outputs") or ())]
    assert outputs, row
    for path in outputs:
        assert path.parent == store.digests_dir(), (path, store.digests_dir())

    session = runner.session_date_for(OVERNIGHT)
    pack = prediction_contrast.read_latest(session)
    assert pack is not None, session
    assert pack["reads"] == 40, pack["reads"]
    assert _day(pack)["right"] == 22
    assert _day(pack)["wrong"] == 18


# ---------------------------------------------------------------------------
# item 3 - the contrast itself
# ---------------------------------------------------------------------------
def test_right_against_wrong_goes_through_the_one_contrast_function(tmp_path, monkeypatch):
    """TJ-15 built `evidence_contrast.contrast` to be reused, and this is the
    reuse. Replace it and the pack must move with it; a slot carrying its own
    AUC, its own rank key or its own floors will not.
    """
    import evidence_contrast
    from ai_jobs import prediction_contrast

    _sessions, rows = fx.two_weeks_of_clicks()
    _path, before = _pack(tmp_path, rows)
    assert before["horizons"]["rest_of_day"]["all"]["features"], before

    marker = {
        "label_a": "right", "label_b": "wrong", "n_a": -1, "n_b": -1,
        "compared": 0, "thin": 0, "min_side": 0, "min_total": 0, "top": 0,
        "statement": "swapped by the test", "unmeasured_features": [],
        "thin_features": [], "features": [], "sentinel": "tj16",
    }
    monkeypatch.setattr(evidence_contrast, "contrast", lambda *_a, **_k: dict(marker))
    _path, after = _pack(tmp_path, rows)

    assert after["horizons"]["rest_of_day"]["all"]["sentinel"] == "tj16", after
    assert after["horizons"]["rest_of_day"]["lately"]["sentinel"] == "tj16", after
    assert prediction_contrast.PACK_SCHEMA, "the pack still names its own schema"


def test_a_feature_under_the_floors_is_named_with_its_counts_and_never_ranked(tmp_path):
    """`evidence_contrast`'s FEATURE floor: ten rows a side AND
    `MIN_REPORTABLE_N` across both.

    `gap_pct` is measured on five of the forty rows - three right, two wrong -
    which is exactly the shape that produced an AUC of 1.0 off four rows
    against one on the live desk. It keeps its row in `thin_features` with both
    counts, carries no AUC, and can never be a tendency.
    """
    import evidence_contrast

    _sessions, rows = fx.two_weeks_of_clicks()
    _path, pack = _pack(tmp_path, rows)
    everything = _day(pack)["all"]

    assert everything["min_side"] == evidence_contrast.MIN_CONTRAST_SIDE_N
    assert everything["min_total"] == evidence_contrast.MIN_REPORTABLE_N

    thin = [row for row in everything["thin_features"] if row["feature"] == "gap_pct"]
    assert len(thin) == 1, everything["thin_features"]
    assert (thin[0]["n_a"], thin[0]["n_b"]) == (3, 2), thin
    assert "auc" not in thin[0], thin[0]
    assert "gap_pct" not in [row["feature"] for row in everything["features"]]
    assert "observational" in everything["statement"], everything["statement"]


def test_a_context_field_the_desk_could_not_measure_is_not_a_zero(tmp_path):
    """Eight of the forty rows have no `spy_vs_prior_range` reading at all.

    They are not "not above": they are unknown, and they leave that feature's
    population. 32 rows measured it - 18 right and 14 wrong, both sides over
    the feature floor - so the feature is ranked on 32. A builder that read the
    eight blanks as zeros ranks it on 40 and moves the AUC with rows nobody
    measured.
    """
    _sessions, rows = fx.two_weeks_of_clicks()
    _path, pack = _pack(tmp_path, rows)

    feature = _feature(_day(pack)["all"], "spy_vs_prior_range:above")
    assert (feature["n_a"], feature["n_b"]) == (18, 14), feature
    assert feature["n_a"] + feature["n_b"] == 32

    # A field measured on every row keeps all forty.
    both = _feature(_day(pack)["all"], "d1_environment:trending_up")
    assert (both["n_a"], both["n_b"]) == (22, 18), both


def test_a_flat_reading_is_in_neither_half_of_right_against_wrong(tmp_path):
    """The market did nothing, so the trader was neither right nor wrong.

    `flat` stays in the ACCURACY denominator (TJ-10's lead decision 3) and is
    counted on the pack - but a contrast is right against wrong, and folding
    the flat rows into either side would be inventing a verdict.
    """
    sessions = fx.sessions_ending(count=2)
    decided = fx.graded_session(
        sessions[0],
        [{"hour": hour, "direction": "up", "confidence": "medium"}
         for hour in fx.CLICK_HOURS],
        rising=True,
    )
    # Half a band-width of movement: a directional call inside the band is flat.
    nothing = fx.graded_session(
        sessions[1],
        [{"hour": hour, "direction": "up", "confidence": "medium"}
         for hour in fx.CLICK_HOURS],
        rising=True,
        band_multiple=0.5,
    )
    _path, pack = _pack(tmp_path, decided + nothing, min_side=1, min_total=1)

    day = _day(pack)
    assert (day["right"], day["wrong"], day["flat"]) == (4, 0, 4), day
    assert day["n"] == 8, "a flat reading is in the accuracy denominator"
    assert (day["all"]["n_a"], day["all"]["n_b"]) == (4, 0), day["all"]


def test_the_lately_window_and_the_whole_ledger_are_different_populations(tmp_path):
    """Item 3 asks for both: *"over `LATELY_SESSIONS` and over the whole
    ledger"*.

    Twenty-four sessions on disk, `LATELY_SESSIONS` (20) in the window. The
    last twenty hold 36 right and 44 wrong; all twenty-four hold 52 and 44. A
    pack that printed one number for both would be hiding the four oldest
    sessions - which is where three of the trader's best days are.
    """
    import evidence_stats

    _sessions, rows = fx.calibration_ledger()
    _path, pack = _pack(tmp_path, rows)
    day = _day(pack)

    assert pack["window_sessions"] == evidence_stats.LATELY_SESSIONS
    assert (day["lately"]["n_a"], day["lately"]["n_b"]) == (36, 44), day["lately"]
    assert (day["all"]["n_a"], day["all"]["n_b"]) == (52, 44), day["all"]
    assert day["n"] == 96


def test_the_two_horizons_get_their_own_contrast_and_are_never_pooled(tmp_path):
    """Forty rest-of-day rows and five five-session rows. 45 is the number a
    pack that pooled the two horizons would print.
    """
    sessions, day_rows = fx.two_weeks_of_clicks()
    rows = day_rows + fx.five_session_clicks(sessions)
    _path, pack = _pack(tmp_path, rows)

    assert _day(pack)["n"] == 40
    week = pack["horizons"]["next_5_sessions"]
    assert week["n"] == 5
    assert (week["right"], week["wrong"]) == (3, 2)
    assert _day(pack)["all"]["n_a"] + _day(pack)["all"]["n_b"] == 40
    assert week["all"]["n_a"] + week["all"]["n_b"] == 5


def test_a_clicked_read_and_an_extracted_one_are_never_pooled_in_the_pack(tmp_path):
    """Decision 0021 answer 29. The pack is the CLICKS' and says so, and the
    extracted rows are COUNTED rather than silently dropped - a population
    nobody can see was skipped is one the reader assumes was included.
    """
    import market_journal
    import market_read_grades as grades

    sessions, rows = fx.two_weeks_of_clicks()
    session = sessions[-1]
    typed = market_journal.build_entry(
        text="SPY is leaking under the open and the sellers keep showing up.",
        session_date=session,
        timeframe="M5",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        now=fx.stamp_at(session, 11),
    )
    read = grades.read_rows([typed], session=session)[0]
    assert read["source"] == grades.SOURCE_EXTRACTED, read
    extracted = grades.grade_read(
        read,
        m5_bars=fx.session_tape(session, rising=False),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.morning_after(session),
        context=fx.context(
            hour=11, direction=read["direction"], confidence="",
            last_hour_spy="down", d1_environment="trending_down",
        ),
    )

    _path, pack = _pack(tmp_path, rows + [extracted])

    assert pack["source"] == grades.SOURCE_CLICK
    assert pack["excluded_by_source"][grades.SOURCE_EXTRACTED] == 1, pack
    assert pack["reads"] == 40
    assert _day(pack)["n"] == 40


# ---------------------------------------------------------------------------
# the tables the live gate reads, and the tendencies TJ-5 may narrate
# ---------------------------------------------------------------------------
def test_two_weeks_of_clicks_make_the_by_hour_and_by_environment_tables(tmp_path):
    """Live gate #161: *"a right-vs-wrong table by hour and by environment,
    every cell with `n`"*.

    Hand-counted in `tj16_support`: the 07:00 and 08:00 clicks went 7-3, the
    09:00 and 10:00 clicks went 4-6 - the trader reads the open better than the
    middle of the day. By environment: 16-8 when the desk called the prior
    session trending up, 6-10 when it called it trending down.
    """
    _sessions, rows = fx.two_weeks_of_clicks()
    _path, pack = _pack(tmp_path, rows)
    day = _day(pack)

    hours = day["tables"]["by_hour"]
    assert [cell["key"] for cell in hours] == [7, 8, 9, 10], hours
    for cell in hours:
        assert cell["n"] == 10, cell
    assert (_table_cell(day, "by_hour", 7)["right"], _table_cell(day, "by_hour", 7)["wrong"]) == (7, 3)
    assert (_table_cell(day, "by_hour", 9)["right"], _table_cell(day, "by_hour", 9)["wrong"]) == (4, 6)
    assert _table_cell(day, "by_hour", 7)["rate"] == pytest.approx(0.7)

    up = _table_cell(day, "by_environment", "trending_up")
    down = _table_cell(day, "by_environment", "trending_down")
    assert (up["n"], up["right"], up["wrong"]) == (24, 16, 8), up
    assert (down["n"], down["right"], down["wrong"]) == (16, 6, 10), down
    for cell in day["tables"]["by_hour"] + day["tables"]["by_environment"]:
        assert "n" in cell and "low" in cell and "high" in cell, cell
        assert cell["reportable"] is False, "every cell here is under the floor"


def test_no_tendency_is_named_from_a_cell_under_the_floor(tmp_path):
    """Two weeks is forty clicks, and every cell of it is under
    `MIN_REPORTABLE_N`. The week story may narrate NOTHING from this pack.

    This is the failure mode the floor exists for: ten clicks at 07:00 going
    7-3 is the most quotable sentence in the pack and the least supported.
    """
    from ai_jobs import prediction_contrast

    _sessions, rows = fx.two_weeks_of_clicks()
    _path, pack = _pack(tmp_path, rows)

    assert prediction_contrast.tendencies(pack) == []


def test_a_tendency_cites_its_own_cell_and_its_n_and_there_are_at_most_three(tmp_path):
    """Item 5: *"may narrate at most three tendencies, each citing its cell and
    its `n`"*. The reader hands TJ-5 the cell; the model never counts.
    """
    import evidence_stats
    from ai_jobs import prediction_contrast

    _sessions, rows = fx.calibration_ledger()
    _path, pack = _pack(tmp_path, rows)

    found = prediction_contrast.tendencies(pack)
    assert 0 < len(found) <= 3, found
    for item in found:
        assert item["n"] >= evidence_stats.MIN_REPORTABLE_N, item
        cell = item["cell"]
        assert cell["table"] in ("by_hour", "by_environment"), item
        assert cell["horizon"] in pack["horizons"], item
        table = pack["horizons"][cell["horizon"]]["tables"][cell["table"]]
        assert any(row["key"] == cell["key"] and row["n"] == item["n"] for row in table), item
        assert str(item.get("text") or "").strip(), item


def test_a_second_run_of_the_same_session_never_rewrites_the_first_pack(tmp_path):
    """A pack is superseded, never edited - the desk's rule for every pack."""
    _sessions, rows = fx.two_weeks_of_clicks()
    first, _pack_one = _pack(tmp_path, rows)
    before = first.read_bytes()

    second, _pack_two = _pack(tmp_path, rows)

    assert first.read_bytes() == before, "the first pack was rewritten"
    assert second.exists() and second != first
