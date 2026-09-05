"""Packet Q4 - the overnight run protects its deterministic work.

Four things, and each one is a rule the night did not have before:

* **Q4.1** the Phase 2 collection window counts CONSECUTIVE clean exchange
  sessions and names where the run broke. It counted DISTINCT session packs, so
  ten packs scattered across a month read as a met window and a pack that
  recorded an unreadable source counted as clean.
* **Q4.2** the other half of that gate - the trader's spot-audit of at least
  three packs - is RECORDED in a file only a trader-run CLI writes, and
  `journal_enrichment` refuses until it exists. This is the behaviour change:
  until the audit is recorded the enrichment slot writes nothing and says
  ``refused: audit not recorded``.
* **Q4.3** decision 0018: every deterministic slot runs before the narration
  pair, so a six-hour narration night can no longer starve the cheap
  deterministic work that nothing narrated feeds.
* **Q4.4** `entry_index.json`: one compact, deterministic, superseding-write
  index of what the packs hold, for a frontier reader.

No model is called anywhere in this file, and no live store is read.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import digest, enrichment  # noqa: E402

NOW = datetime(2026, 8, 22, 3, 0, tzinfo=ZoneInfo("America/New_York"))
ET = ZoneInfo("America/New_York")

#: Two trading weeks, weekends excluded by the exchange calendar rather than by
#: this test. 2026-08-15/16 is the weekend inside the span.
TEN_SESSIONS = [f"2026-08-{day:02d}" for day in (10, 11, 12, 13, 14, 17, 18, 19, 20, 21)]


@pytest.fixture(autouse=True)
def _packs_have_no_environment_noise(monkeypatch):
    """A pack written in a test must be CLEAN unless the test dirties it.

    Without this every pack records "ai job ledger: No AI store configured",
    which is a real failure record - the desk's own packs carry `unavailable:
    {}` - and would make every session in this file unclean for a reason that
    has nothing to do with what is under test.
    """
    monkeypatch.setattr(digest, "_read_job_rows", lambda: [])
    monkeypatch.setattr(digest, "_read_review_events", lambda: [
        {"trade_date": day, "action": action, "dwell_ms": 1000}
        for day in TEN_SESSIONS
        for action in ("like_advance", "remove_today")
    ])


def _final(symbol: str = "AAPL", day: str = "2026-08-10") -> dict:
    return {
        "symbol": symbol,
        "direction": "long",
        "trade_date": day,
        "env_key": "bullish_weak|midday",
        "close_r": 1.0,
        "mfe_r": 2.0,
        "mae_r": -0.5,
    }


def _write_sessions(root: Path, days, *, unavailable_on=()) -> None:
    for day in days:
        digest.run_daily_digest(
            session_date=day,
            now=NOW,
            root=root,
            narrate=False,
            finals=[_final("AAPL", day)],
        )
        if day in unavailable_on:
            # A pack that names a source it could not read is INCOMPLETE by its
            # own summary. Write the superseding sibling the runner would have
            # written - later `generated_at`, which is what makes it the newest
            # pack for the day and therefore the one the gate reads.
            pack = digest.build_fact_pack(
                session_date=day,
                is_session=True,
                finals=[_final("AAPL", day)],
                unavailable={"alert review events": "file is locked"},
                now=NOW + timedelta(hours=1),
            )
            digest._publish(
                digest.superseding_path(digest.facts_path(root, day)),
                digest.render_fact_pack(pack),
            )


# ==========================================================================
# Q4.1 - the run of consecutive clean sessions
# ==========================================================================


def test_ten_consecutive_clean_sessions_with_a_weekend_inside_meet_the_window(tmp_path):
    """A weekend is not a gap - the exchange calendar knows it is not a session.

    Weekday arithmetic gets this wrong twice (`market_calendar` is the one
    clock), so the walk goes through `previous_session`.
    """
    _write_sessions(tmp_path, TEN_SESSIONS)
    # The weekend's empty pack exists too, exactly as the nightly writes it.
    digest.run_daily_digest(session_date="2026-08-15", now=NOW, root=tmp_path, narrate=False)

    assert digest.clean_digest_sessions(tmp_path) == 10
    state = digest.digest_gate_state(tmp_path)
    assert state["sessions_consecutive_clean"] == 10
    assert state["window_met"] is True
    # The old distinct count survives for every existing reader.
    assert state["sessions_collected"] == 10


def test_a_missing_weekday_pack_breaks_the_run_and_the_gap_is_named(tmp_path):
    """Ten distinct packs are not ten consecutive sessions.

    This is the defect: the old count would have said 9 of 10 and kept
    climbing, when what exists is a five-session run with a hole in front of it.
    """
    days = [day for day in TEN_SESSIONS if day != "2026-08-14"]
    _write_sessions(tmp_path, days)

    state = digest.digest_gate_state(tmp_path)
    assert state["sessions_collected"] == 9, "the old count is unchanged"
    assert state["sessions_consecutive_clean"] == 5, "17..21 only"
    assert state["first_gap_session"] == "2026-08-14"
    assert state["window_met"] is False
    assert "2026-08-14" in state["statement"]


def test_a_pack_that_recorded_a_failure_breaks_the_run(tmp_path):
    """Clean means the pack's own failure record is EMPTY.

    `unavailable` is the field the pack carries; its own summary already calls
    such a pack INCOMPLETE. A gate that counts an incomplete pack as clean is
    counting the thing it exists to exclude.
    """
    _write_sessions(tmp_path, TEN_SESSIONS, unavailable_on=("2026-08-14",))

    state = digest.digest_gate_state(tmp_path)
    assert state["sessions_consecutive_clean"] == 5
    assert state["first_gap_session"] == "2026-08-14"
    assert state["window_met"] is False


def test_a_non_session_pack_neither_counts_nor_breaks(tmp_path):
    """A holiday pack is a visible gap in the ledger, not a broken run."""
    _write_sessions(tmp_path, TEN_SESSIONS)
    # A pack the digest wrote as a non-session on a day the calendar calls one.
    digest.run_daily_digest(
        session_date="2026-08-24", now=NOW, root=tmp_path, is_session=False, narrate=False,
    )
    state = digest.digest_gate_state(tmp_path, as_of="2026-08-24")
    assert state["sessions_consecutive_clean"] == 10
    assert state["sessions_collected"] == 10, "a non-session pack still does not count"


def test_no_packs_at_all_is_zero_and_names_no_gap(tmp_path):
    state = digest.digest_gate_state(tmp_path)
    assert state["sessions_consecutive_clean"] == 0
    assert state["sessions_collected"] == 0
    assert state["first_gap_session"] is None
    assert state["window_met"] is False


# ==========================================================================
# Q4.2 - the audit is recorded, and enrichment waits for it
# ==========================================================================


def test_the_window_alone_no_longer_meets_the_gate(tmp_path):
    _write_sessions(tmp_path, TEN_SESSIONS)
    state = digest.digest_gate_state(tmp_path)

    assert state["window_met"] is True
    assert state["audit_recorded"] is False
    assert state["audit_packs"] == []
    assert state["gate_met"] is False
    assert "audit" in state["statement"].lower()


def test_enrichment_refuses_until_the_audit_is_recorded(tmp_path, monkeypatch):
    """The behaviour change, in one test.

    The window is met, the model is up, and the slot still writes nothing -
    because nobody has recorded that a human read three packs against raw
    evidence.
    """
    called: list = []
    monkeypatch.setattr(enrichment, "_enrich_one", lambda **kwargs: called.append(kwargs))
    _write_sessions(tmp_path / "digests", TEN_SESSIONS)

    result = enrichment.run_journal_enrichment(
        session_date="2026-08-21", now=NOW, digest_root=tmp_path / "digests",
    )

    assert called == []
    assert result["outputs"] == []
    assert result["status"] == "ok"
    assert "refused: audit not recorded" in result["reason"]


def test_the_gate_strip_never_says_enrichment_is_met_while_it_refuses(tmp_path, monkeypatch):
    """A counter that reported `window_met` would contradict the slot.

    Out of the packet's file list by one line, and deliberately: Q4.2 makes the
    slot refuse on `gate_met`, and a strip that says "Enrichment met (10/10)"
    on a night the ledger says `refused: audit not recorded` is a surface
    lying about a job.
    """
    from ai_jobs import enrichment as enrichment_mod
    from ai_jobs import gate_counters

    _write_sessions(tmp_path, TEN_SESSIONS)
    monkeypatch.setattr(
        enrichment_mod, "gate_state", lambda *a, **k: digest.digest_gate_state(tmp_path)
    )
    assert gate_counters._enrichment_counter().met is False

    digest.record_audit_approval(
        tmp_path, packs=["2026-08-19", "2026-08-20", "2026-08-21"], now=NOW,
    )
    assert gate_counters._enrichment_counter().met is True


def test_both_gate_counters_show_the_number_the_gate_turns_on(tmp_path, monkeypatch):
    """The RATIO and the verdict must come from the same count.

    Reviewer blocker 1. Both counters passed `have=sessions_collected` - the
    distinct count Q4.1 deliberately kept for old readers - while `met` turned
    on the consecutive run. With ten scattered packs and a two-session run the
    strip read "Digest 10/10" and not met, which is the strip inviting the
    trader to distrust the strip.
    """
    from ai_jobs import enrichment as enrichment_mod
    from ai_jobs import gate_counters

    # Ten distinct session packs; the hole at 2026-08-13 leaves a run of six
    # (2026-08-14 .. 2026-08-21), so the two numbers cannot be confused.
    _write_sessions(tmp_path, [day for day in TEN_SESSIONS if day != "2026-08-13"])
    digest.run_daily_digest(
        session_date="2026-07-31", now=NOW, root=tmp_path, narrate=False,
        finals=[_final("AAPL", "2026-07-31")],
    )
    state = digest.digest_gate_state(tmp_path)
    assert (state["sessions_collected"], state["sessions_consecutive_clean"]) == (10, 6)

    monkeypatch.setattr(
        enrichment_mod, "gate_state", lambda *a, **k: digest.digest_gate_state(tmp_path)
    )
    digest_counter = gate_counters._digest_counter(tmp_path)
    enrichment_counter = gate_counters._enrichment_counter()

    # The TEXT, not just `.met`: the ratio is what the trader reads.
    assert digest_counter.text() == "Digest 6/10"
    assert enrichment_counter.text() == "Enrichment 6/10"
    assert digest_counter.have == 6 and enrichment_counter.have == 6
    assert digest_counter.met is False and enrichment_counter.met is False
    assert gate_counters.strip_text([digest_counter, enrichment_counter]) == (
        "Digest 6/10 · Enrichment 6/10"
    )


def test_the_cli_refuses_fewer_than_three_packs(tmp_path):
    _write_sessions(tmp_path, TEN_SESSIONS)
    with pytest.raises(ValueError, match="at least 3"):
        digest.record_audit_approval(tmp_path, packs=["2026-08-20", "2026-08-21"], now=NOW)
    assert not digest.audit_approval_path(tmp_path).exists()

    code = digest.main([
        "approve-audit", "--root", str(tmp_path),
        "--pack", "2026-08-20", "--pack", "2026-08-21",
    ])
    assert code != 0
    assert not digest.audit_approval_path(tmp_path).exists()


def test_the_cli_refuses_a_date_that_has_no_pack(tmp_path):
    _write_sessions(tmp_path, TEN_SESSIONS)
    with pytest.raises(ValueError, match="2026-07-01"):
        digest.record_audit_approval(
            tmp_path, packs=["2026-08-19", "2026-08-20", "2026-07-01"], now=NOW,
        )
    assert not digest.audit_approval_path(tmp_path).exists()


def test_three_audited_packs_meet_the_gate_and_enrichment_proceeds(tmp_path, monkeypatch):
    root = tmp_path / "digests"
    _write_sessions(root, TEN_SESSIONS)
    written = digest.record_audit_approval(
        root,
        packs=["2026-08-19", "2026-08-20", "2026-08-21"],
        note="read them against the outcome store",
        now=NOW,
    )
    assert written == digest.audit_approval_path(root)

    state = digest.digest_gate_state(root)
    assert state["audit_recorded"] is True
    assert state["audit_packs"] == ["2026-08-19", "2026-08-20", "2026-08-21"]
    assert state["gate_met"] is True

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["approved_by"] == "trader"
    assert payload["approved_at"].startswith("2026-08-22")
    assert payload["note"] == "read them against the outcome store"

    # And the slot gets past the gate: it fails on the journal store, not on
    # the gate, which is the proof the gate let it through.
    calls: list = []

    def _boom():
        calls.append("store")
        raise RuntimeError("no journal in this test")

    monkeypatch.setattr(enrichment, "_journal_store", _boom)
    result = enrichment.run_journal_enrichment(
        session_date="2026-08-21", now=NOW, digest_root=root,
    )
    assert calls == ["store"]
    assert enrichment.GATE_NOT_MET_PREFIX not in result["reason"]


def test_only_the_trader_cli_writes_the_approval_and_the_runner_never_does():
    """The runner must not be able to approve its own evidence."""
    import ast
    import inspect

    from ai_jobs import runner

    runner_source = Path(inspect.getsourcefile(runner)).read_text(encoding="utf-8")
    assert "record_audit_approval" not in runner_source

    digest_tree = ast.parse(Path(digest.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(digest_tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "run_daily_digest":
            continue
        names = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert "record_audit_approval" not in names
        break
    else:  # pragma: no cover - the function exists
        pytest.fail("run_daily_digest not found")


# ==========================================================================
# Q4.3 - decision 0018: the deterministic stage runs before narration
# ==========================================================================


def test_every_deterministic_slot_precedes_the_narration_pair():
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    narration_at = min(names.index("ai_summary"), names.index("ticker_briefs"))
    for deterministic in (
        "journal_import", "journal_auto_tag", "veto_cohort_grading",
        "like_cohort_grading", "sidecar_completion", "pass_cohort_grading",
        "rejection_cohort_grading", "note_vocabulary_audit",
        "preference_trade_outcomes", "evidence_report", "daily_digest",
    ):
        assert names.index(deterministic) < narration_at, deterministic
    # And the model-gated slots still sit after the narration pair.
    for gated in ("journal_enrichment", "review_policy_draft", "setup_research"):
        assert names.index(gated) > names.index("ticker_briefs"), gated


def _stub_slots(names, *, raises=()):
    from ai_jobs.runner import JobSlot

    def _make(name):
        def _run(**kwargs):
            if name in raises:
                raise RuntimeError(f"{name} exploded")
            return {"model": "", "outputs": [], "reason": name}
        return JobSlot(name=name, run=_run, reserve_minutes=0.5)

    return [_make(name) for name in names]


def _run_stubbed(tmp_path, slots):
    from unittest import mock

    from ai_jobs import runner, store, window

    led = tmp_path / "ledger.jsonl"
    with mock.patch.object(store, "store_available", return_value=(True, "ready")), \
            mock.patch.object(window, "launch_allowed", return_value=(True, "open")), \
            mock.patch.object(window, "market_session_block", return_value=""):
        runner.run_slots(
            slots, now=datetime(2026, 8, 12, 2, 0, tzinfo=ET), ledger_path=led,
        )
    return [json.loads(line) for line in led.read_text(encoding="utf-8").splitlines()]


def test_every_deterministic_ledger_row_is_written_before_ai_summary_starts(tmp_path):
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    rows = _run_stubbed(tmp_path, _stub_slots(names))
    order = [row["job"] for row in rows]
    assert order == names
    assert order.index("daily_digest") < order.index("ai_summary")


def test_a_raising_narration_slot_leaves_every_deterministic_row_already_written(tmp_path):
    """Not just `ok` - already WRITTEN, which is what the reorder bought.

    Renamed and strengthened after the reviewer's advisory (a): asserting only
    the statuses passes on the OLD order too, because a failure never took the
    rest of the night down there either. What decision 0018 changed is that
    every deterministic row is on disk BEFORE the narration slot is entered, so
    that is what this asserts.
    """
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    rows = _run_stubbed(tmp_path, _stub_slots(names, raises=("ai_summary",)))
    order = [row["job"] for row in rows]
    by_job = {row["job"]: row for row in rows}
    assert by_job["ai_summary"]["status"] == "failed"
    for deterministic in (
        "journal_import", "journal_auto_tag", "veto_cohort_grading",
        "like_cohort_grading", "sidecar_completion", "pass_cohort_grading",
        "rejection_cohort_grading", "note_vocabulary_audit",
        "preference_trade_outcomes", "evidence_report", "daily_digest",
    ):
        assert by_job[deterministic]["status"] == "ok", deterministic
        assert order.index(deterministic) < order.index("ai_summary"), deterministic


# ==========================================================================
# Q4.4 - entry_index.json
# ==========================================================================


def test_the_digest_run_writes_the_entry_index_beside_the_packs(tmp_path):
    digest.run_daily_digest(
        session_date="2026-08-21", now=NOW, root=tmp_path, narrate=False,
        finals=[_final("AAPL", "2026-08-21")],
    )
    path = digest.entry_index_path(tmp_path)
    assert path.exists()

    index = digest.read_entry_index(tmp_path)
    assert index["schema_version"] == digest.ENTRY_INDEX_SCHEMA
    assert index["latest_complete_session"] == "2026-08-21"
    assert index["generated_at"]
    # A real sha, not "". `definitions_git_commit` reads `.git/HEAD`, and in a
    # git WORKTREE - which is where every agent builds - `.git` is a FILE
    # pointing at the real gitdir, so it returned empty for the whole packet.
    assert len(index["git_commit"]) == 40, index["git_commit"]
    assert all(char in "0123456789abcdef" for char in index["git_commit"])
    # The version identifiers the packs actually carry - nothing invented.
    assert index["versions"]["facts_schema"] == digest.FACTS_SCHEMA
    assert index["versions"]["statistics_schema"]
    assert index["versions"]["n_floor"]


def test_a_superseded_pack_is_marked_and_counted(tmp_path):
    _write_sessions(tmp_path, ["2026-08-20", "2026-08-21"], unavailable_on=("2026-08-20",))
    index = digest.build_entry_index(tmp_path, as_of="2026-08-21")
    by_day = {row["session_date"]: row for row in index["sessions"]}

    assert by_day["2026-08-20"]["superseded"] is True
    assert by_day["2026-08-20"]["versions"] == 2
    assert by_day["2026-08-20"]["clean"] is False
    assert by_day["2026-08-20"]["failures"] == {"alert review events": "file is locked"}
    assert by_day["2026-08-21"]["superseded"] is False
    assert by_day["2026-08-21"]["versions"] == 1
    assert by_day["2026-08-21"]["coverage"] is not None

    # Reviewer blocker 2: the path must be the file the VALUES came from. It
    # cited `facts_path`, which is always version 1, while every number was
    # read from the newest sibling - so a reader following the citation on a
    # superseded session would open the pack that was corrected.
    cited = Path(by_day["2026-08-20"]["pack_path"])
    assert cited.name == "2026-08-20.1.json", by_day["2026-08-20"]["pack_path"]
    payload = json.loads(cited.read_text(encoding="utf-8"))
    assert payload["generated_at"] == (NOW + timedelta(hours=1)).isoformat(timespec="seconds")
    assert payload["unavailable"] == by_day["2026-08-20"]["failures"]
    # An unsuperseded session still cites its only file.
    assert Path(by_day["2026-08-21"]["pack_path"]).name == "2026-08-21.json"


def test_a_same_second_correction_still_wins_over_the_pack_it_corrects(tmp_path):
    """The tiebreak when two siblings share a `generated_at`.

    A re-run inside the same second is normal, and sorting by NAME would put
    `2026-08-20.1.json` before `2026-08-20.json` - '1' sorts before 'j' - and
    hand the correction's place to the pack it corrects.
    """
    digest.run_daily_digest(
        session_date="2026-08-20", now=NOW, root=tmp_path, narrate=False,
        finals=[_final("AAPL", "2026-08-20")],
    )
    correction = digest.build_fact_pack(
        session_date="2026-08-20", is_session=True,
        finals=[_final("AAPL", "2026-08-20")],
        unavailable={"alert review events": "file is locked"},
        now=NOW,  # the SAME second as the pack it supersedes
    )
    digest._publish(
        digest.superseding_path(digest.facts_path(tmp_path, "2026-08-20")),
        digest.render_fact_pack(correction),
    )

    newest = digest.latest_pack_files_by_session(tmp_path)["2026-08-20"]
    assert newest[0].name == "2026-08-20.1.json"
    assert digest.pack_failures(newest[1]) == {"alert review events": "file is locked"}
    assert digest.digest_gate_state(tmp_path)["sessions_consecutive_clean"] == 0


def test_every_cited_pack_path_is_the_pack_the_numbers_came_from(tmp_path):
    """The same rule, everywhere the index prints a `pack_path`."""
    _write_sessions(tmp_path, ["2026-08-20", "2026-08-21"], unavailable_on=("2026-08-20",))
    index = digest.build_entry_index(tmp_path, as_of="2026-08-21")

    cited = {row["pack_path"] for row in index["sessions"]}
    for section in digest.ENTRY_INDEX_SECTIONS:
        cited |= {entry["pack_path"] for entry in index[section]["entries"]}
    for change in index["changes_vs_prior_window"]["cleared_the_floor"]:
        if change["pack_path"]:
            cited.add(change["pack_path"])

    for path in cited:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        day = payload["session_date"]
        newest = digest.latest_packs_by_session(tmp_path)[day]
        assert payload["generated_at"] == newest["generated_at"], path


def test_the_four_sections_are_distinct_keys_and_never_merged(tmp_path):
    _write_sessions(tmp_path, TEN_SESSIONS)
    index = digest.build_entry_index(tmp_path, as_of="2026-08-21")
    for section in digest.ENTRY_INDEX_SECTIONS:
        assert section in index, section
        assert "entries" in index[section]
        assert "note" in index[section]
    assert len(set(digest.ENTRY_INDEX_SECTIONS)) == 4
    assert digest.ENTRY_INDEX_SECTIONS == (
        "intraday_held_run", "swing_win_rates",
        "preference_observations", "journal_execution",
    )
    for entry in index["intraday_held_run"]["entries"]:
        assert entry["pack_path"] and entry["cell_key"]
    assert index["open_questions_for_a_ticker_brief"] == []
    assert "pending_experiments" in index


def test_a_cell_that_clears_the_floor_only_this_window_is_named(tmp_path):
    """By FLOOR STATUS only - never by ranking immature cells."""
    floor = digest.ENTRY_INDEX_FLOOR
    # The prior window: one thin session. This window: enough rows to clear.
    digest.run_daily_digest(
        session_date="2026-07-24", now=NOW, root=tmp_path, narrate=False,
        finals=[_final("AAPL", "2026-07-24")],
    )
    digest.run_daily_digest(
        session_date="2026-08-21", now=NOW, root=tmp_path, narrate=False,
        finals=[_final(f"SYM{i}", "2026-08-21") for i in range(floor + 5)],
    )

    index = digest.build_entry_index(tmp_path, as_of="2026-08-21")
    cleared = index["changes_vs_prior_window"]["cleared_the_floor"]
    keys = {row["cell_key"] for row in cleared}
    assert "outcomes.overall.mfe_r" in keys
    row = next(r for r in cleared if r["cell_key"] == "outcomes.overall.mfe_r")
    assert row["section"] in digest.ENTRY_INDEX_SECTIONS
    assert row["this_window"]["n"] >= floor
    assert row["prior_window"]["n"] < floor
    assert row["pack_path"]
    # "46 cleared, 0 fell" reads as a finding when it is often just "there was
    # no prior window". The count says which.
    assert index["changes_vs_prior_window"]["prior_window_packs"] == 1
    assert index["changes_vs_prior_window"]["this_window_packs"] == 1


def test_an_empty_prior_window_says_so_rather_than_reading_as_a_finding(tmp_path):
    _write_sessions(tmp_path, ["2026-08-20", "2026-08-21"])
    index = digest.build_entry_index(tmp_path, as_of="2026-08-21")
    changes = index["changes_vs_prior_window"]
    assert changes["prior_window_packs"] == 0
    assert changes["this_window_packs"] == 2
    assert "no pack" in changes["note"].lower() or "prior window" in changes["note"].lower()


def test_the_index_write_is_superseding_and_a_failure_leaves_the_last_good_one(
    tmp_path, monkeypatch,
):
    _write_sessions(tmp_path, ["2026-08-20"])
    first = digest.write_entry_index(tmp_path, as_of="2026-08-20")
    good = first.read_text(encoding="utf-8")

    def _explode(path, content):
        raise OSError("the share went away mid-write")

    monkeypatch.setattr(digest, "_publish", _explode)
    with pytest.raises(OSError):
        digest.write_entry_index(tmp_path, as_of="2026-08-20")
    assert first.read_text(encoding="utf-8") == good
    assert not list(first.parent.glob("entry_index.json.tmp"))


def test_a_failed_rename_leaves_no_half_written_temp_behind(tmp_path, monkeypatch):
    """The rename is where a share drops out, and the tmp is what it leaves.

    The previous test patches `_publish` whole, so it never exercised the
    rename. This one lets the temp file be written and then fails `os.replace`,
    which is the real failure and the one that leaves litter beside the packs.
    """
    _write_sessions(tmp_path, ["2026-08-20"])
    first = digest.write_entry_index(tmp_path, as_of="2026-08-20")
    good = first.read_text(encoding="utf-8")

    def _explode(src, dst):
        raise OSError("the share went away between write and rename")

    monkeypatch.setattr(digest.os, "replace", _explode)
    with pytest.raises(OSError):
        digest.write_entry_index(tmp_path, as_of="2026-08-20")

    assert first.read_text(encoding="utf-8") == good
    assert list(first.parent.glob("*.tmp")) == []


def test_a_failed_index_write_never_fails_the_digest(tmp_path, monkeypatch):
    def _explode(root, **kwargs):
        raise OSError("the share went away")

    monkeypatch.setattr(digest, "write_entry_index", _explode)
    result = digest.run_daily_digest(
        session_date="2026-08-21", now=NOW, root=tmp_path, narrate=False,
        finals=[_final("AAPL", "2026-08-21")],
    )
    assert result["status"] == "ok"
    assert digest.facts_path(tmp_path, "2026-08-21").exists()


def test_pending_experiments_are_listed_unranked_or_skipped_with_a_note(tmp_path):
    _write_sessions(tmp_path, ["2026-08-21"])
    index = digest.build_entry_index(tmp_path, as_of="2026-08-21")
    pending = index["pending_experiments"]
    assert "entries" in pending and "note" in pending
    for row in pending["entries"]:
        assert "trial_id" in row and "declared_window" in row
        assert "rank" not in row
