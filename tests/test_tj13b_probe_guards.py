"""TJ-13B, builder's half: the guards around the probe, and the basis it names.

The tester's three files pin WHAT a probe measures and what the week seam
answers. These pin the rules a measurement of a 27B on a 32 GB desk lives
under, none of which can be proven by a number in a ledger row:

* the probe stands down while the nightly runner holds the machine lock - the
  one thing that must never happen on this box is a second model load beside a
  working one, and the scheduled task fires every 30 minutes for eight hours;
* the guard is HELD across the model call, not merely checked before it;
* the typed command is what hands the probe that guard, so the wiring cannot be
  dropped in a later edit without a red test;
* the copy of the week's packs is deleted afterwards;
* one load a night unless the operator asks for another;
* the four numbers say which BASIS they were measured on, because a single call
  cannot split a weight load from the work and this module does not pretend it
  can.

**NO MODEL IS LOADED HERE.** Every endpoint is a fake ``post`` and every lock is
a fake. Nothing in this file depends on the state of this machine's real AI
lock - the nightly runner holds it most of every night, and a test that passes
or fails on that is not a test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from zoneinfo import ZoneInfo  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
#: 23:00 PDT Saturday = 02:00 ET Sunday: inside the live 01:00-09:00 ET window.
SATURDAY_NIGHT = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)
WEEKEND_SESSION = "2026-09-18"

ENDPOINT = "http://127.0.0.1:11434/v1"
LARGE_TAG = "hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M"
MEDIUM_TAG = "gemma3:12b-tbv3ctx-64k"
WEEK_PACKS = ("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")


class _MultiPatch:
    def __init__(self, *patches):
        self._patches = patches

    def __enter__(self):
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        return False


def _live_settings(**extra):
    import ai_summary
    import project_paths

    values = {
        "ai_offhours_start": "01:00",
        "ai_offhours_end": "09:00",
        "ai_local_endpoint_url": ENDPOINT,
        "ai_local_model_large": LARGE_TAG,
        "ai_local_model_medium": MEDIUM_TAG,
        "ai_local_context_tokens": 65536,
    }
    values.update(extra)

    def _get(key, default=None):
        return values.get(key, default)

    return _MultiPatch(
        mock.patch.object(ai_summary, "get_local_setting", _get),
        mock.patch.object(project_paths, "get_local_setting", _get),
    )


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _reply_text() -> str:
    import ai_summary

    sections = {name: [] for name in ai_summary.MODEL_SUMMARY_SECTIONS}
    return json.dumps({"executive_summary": "One measured week.", **sections})


def _fake_post(calls, *, usage=None, extra_body=None):
    def _post(url, **kwargs):
        calls.append((url, kwargs))
        payload = {
            "id": "chatcmpl-probe",
            "choices": [
                {"message": {"role": "assistant", "content": _reply_text()},
                 "finish_reason": "stop"}
            ],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 41_234, "completion_tokens": 812, "total_tokens": 42_046},
        }
        payload.update(extra_body or {})
        return _Response(payload)

    return _post


def _never_post(*args, **kwargs):
    raise AssertionError("a refused probe must not start a model request")


def _write_packs(root: Path) -> Path:
    facts = root / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    for day in WEEK_PACKS:
        (facts / f"{day}.json").write_text(
            json.dumps(
                {
                    "schema": "ai_daily_fact_pack_v1",
                    "session_date": day,
                    "generated_at": f"{day}T23:10:00-04:00",
                    "headline": f"probe marker {day}",
                    "sections": {"decisions": {"n": 4}},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return root


def _probe_rows(path: Path) -> list[dict]:
    from ai_jobs import model_probe

    if not path.exists():
        return []
    return [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("job") == model_probe.PROBE_JOB
    ]


@pytest.fixture(autouse=True)
def _a_free_machine_lock(monkeypatch):
    """A free lock for every test here that does not hand the probe its own.

    Since 2026-09-20 `run_model_probe` takes the machine's AI-jobs lock by
    DEFAULT. The live nightly runner holds that lock for most of every night,
    and nothing in this file may pass or fail on whether it happens to be
    working - the tests that are ABOUT the lock inject their own, which
    overrides this.
    """
    import contextlib

    from ai_jobs import model_probe

    monkeypatch.setattr(model_probe, "runner_lock", lambda: contextlib.nullcontext())


# ---------------------------------------------------------------------------
# the machine lock
# ---------------------------------------------------------------------------


def _held_lock():
    """A lock that is already taken, the way the nightly runner takes it."""
    from local_writer_lock import LocalLockUnavailable

    def _factory():
        raise LocalLockUnavailable(
            "another process has held the writer lock ai_jobs_runner for longer "
            "than the timeout; refusing to publish concurrently"
        )

    return _factory


def test_a_probe_stands_down_while_the_nightly_runner_holds_the_lock(tmp_path):
    """The scheduled task fires every 30 minutes for eight hours and holds this
    lock for the whole slate. A probe that ignored it would load a 27B beside a
    working 12B on a 32 GB box."""
    from ai_jobs import ledger, model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_never_post,
            ledger_path=led,
            lock=_held_lock(),
        )

    assert result["status"] == ledger.STATUS_SKIPPED
    assert result["measurement"] is None
    assert "lock" in result["reason"].lower()
    rows = _probe_rows(led)
    assert len(rows) == 1 and rows[0]["status"] == ledger.STATUS_SKIPPED
    assert not rows[0].get(model_probe.MEASUREMENT_FIELD)
    assert model_probe.PROBE_JOB not in ledger.completed_jobs(WEEKEND_SESSION, path=led)


def test_a_box_with_no_exclusion_primitive_is_refused_rather_than_guessed_at(tmp_path):
    """`ai_jobs.runner` runs UNGUARDED when the box offers no primitive, because
    what it protects is a night of cheap deterministic work. What is at stake
    here is a second model load, so uncertainty refuses."""
    from ai_jobs import ledger, model_probe
    from ai_jobs.runner import NO_PRIMITIVE_MARKER
    from local_writer_lock import LocalLockUnavailable

    def _no_primitive():
        raise LocalLockUnavailable(NO_PRIMITIVE_MARKER)

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_never_post,
            ledger_path=led,
            lock=_no_primitive,
        )

    assert result["status"] == ledger.STATUS_SKIPPED
    assert result["measurement"] is None


def test_the_guard_is_held_across_the_model_call_not_merely_checked_first(tmp_path):
    """A lock released before the call would leave the 30-minute firing free to
    start the night's slate on top of a loading 27B."""
    from ai_jobs import model_probe

    events: list[str] = []

    class _Guard:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *exc):
            events.append("exit")
            return False

    def _post(url, **kwargs):
        events.append("post")
        return _fake_post([])(url, **kwargs)

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    with _live_settings():
        model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_post,
            ledger_path=led,
            lock=_Guard,
        )

    assert events == ["enter", "post", "exit"], events


def test_the_default_guard_is_the_machines_own_lock(tmp_path, monkeypatch):
    """Review 2026-09-20: a bare call must not be able to load a 27B.

    There is no value of ``lock`` that means "no guard at all" - leaving it out
    and passing ``None`` both take the machine's own AI-jobs lock, because the
    most dangerous call in this package must not be the one that names nothing.
    """
    from ai_jobs import model_probe

    entered: list[str] = []

    class _Recorded:
        def __enter__(self):
            entered.append("enter")
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(model_probe, "runner_lock", _Recorded)

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    with _live_settings():
        model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT,
            post=_fake_post([]), ledger_path=led,
        )
        model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT, force=True,
            post=_fake_post([]), ledger_path=led, lock=None,
        )

    assert entered == ["enter", "enter"]


def test_the_typed_command_is_what_hands_the_probe_the_real_lock(monkeypatch):
    """The CLI is the one caller that actually loads a model, so it is the one
    that must pass the guard. Pinned so a later edit cannot quietly drop it."""
    import run_ai_jobs
    from ai_jobs import ledger, model_probe

    seen: list[dict] = []

    def _fake_probe(**kwargs):
        seen.append(dict(kwargs))
        return {"status": ledger.STATUS_MANUAL, "reason": "measured", "measurement": {}}

    monkeypatch.setattr(model_probe, "run_model_probe", _fake_probe)
    monkeypatch.setattr(model_probe, "reserve_minutes_from_probe", lambda **k: 30.0)

    assert run_ai_jobs.main(["--probe-model", "large"]) == 0
    assert seen[0]["lock"] is model_probe.runner_lock


# ---------------------------------------------------------------------------
# the copy, and one load a night
# ---------------------------------------------------------------------------


def test_the_copy_of_the_week_is_deleted_when_the_probe_finishes(tmp_path):
    """It runs on a COPY so the live store is read once and never written. A
    copy that outlived the probe would be a second store nobody maintains."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    temp_root = Path(tempfile.gettempdir())
    before = {path.name for path in temp_root.glob("tbv3-model-probe-*")}

    with _live_settings():
        model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_fake_post([]),
            ledger_path=led,
        )

    assert {path.name for path in temp_root.glob("tbv3-model-probe-*")} == before


def test_one_load_a_night_unless_the_operator_asks_for_another(tmp_path):
    """A 27B load is minutes of the box. A second one on the same session has to
    be asked for, and --force is how it is asked for."""
    from ai_jobs import ledger, model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    calls: list[tuple[str, dict]] = []

    with _live_settings():
        first = model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT,
            post=_fake_post(calls), ledger_path=led,
        )
        second = model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT,
            post=_never_post, ledger_path=led,
        )
        third = model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT, force=True,
            post=_fake_post(calls), ledger_path=led,
        )

    assert first["status"] == ledger.STATUS_MANUAL
    assert second["status"] == ledger.STATUS_SKIPPED and second["measurement"] is None
    assert third["status"] == ledger.STATUS_MANUAL
    assert len(calls) == 2
    # Two measurements on the ledger, and the newest is the one that counts.
    assert len([row for row in _probe_rows(led) if row.get("model_probe")]) == 2


# ---------------------------------------------------------------------------
# what the numbers are, and what they are not
# ---------------------------------------------------------------------------


def test_the_measurement_says_which_basis_its_numbers_came_from(tmp_path):
    """One call cannot split a weight load from the prompt evaluation and the
    generation. The row says which basis it used rather than implying a
    precision the measurement does not have."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT,
            post=_fake_post([]), ledger_path=led,
        )

    measurement = result["measurement"]
    assert measurement["basis"] == model_probe.BASIS_END_TO_END
    assert measurement["peak_memory_basis"] in {
        "system_in_use", "probe_process_rss", "unmeasured"
    }
    # The configured context is recorded BESIDE what the server accepted, never
    # instead of it: the gap between them is the finding.
    assert measurement["context_tokens_configured"] == 65536
    assert measurement["context_tokens_accepted"] == 41_234


def test_a_server_that_reports_its_own_timings_is_believed_over_the_wall_clock(tmp_path):
    """llama.cpp's server returns a `timings` block. When it is there the split
    is exact, and the basis says so."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    timings = {
        "timings": {
            "prompt_ms": 30_000.0,
            "predicted_ms": 90_000.0,
            "predicted_per_second": 7.5,
        }
    }

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT,
            post=_fake_post([], extra_body=timings), ledger_path=led,
        )

    measurement = result["measurement"]
    assert measurement["basis"] == model_probe.BASIS_SERVER_TIMINGS
    assert measurement["tokens_per_second"] == 7.5
    assert measurement["load_seconds"] == 120.0


def test_a_server_that_reports_no_completion_tokens_leaves_the_reserve_unmeasured(tmp_path):
    """Missing data is uncertainty, never zero. A probe that could not measure
    throughput still records what it DID measure, and the reserve stays None."""
    from ai_jobs import ledger, model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large", packs_root=packs, now=SATURDAY_NIGHT,
            post=_fake_post([], usage={"prompt_tokens": 41_234}), ledger_path=led,
        )

    assert result["status"] == ledger.STATUS_MANUAL
    assert result["measurement"]["context_tokens_accepted"] == 41_234
    assert result["measurement"]["tokens_per_second"] == 0.0
    assert model_probe.reserve_minutes_from_probe(tier="large", ledger_path=led) is None


# ---------------------------------------------------------------------------
# --force, driven through the REAL command line (review 2026-09-20, blocker 1)
# ---------------------------------------------------------------------------


def _cli_desk(monkeypatch, tmp_path, *, moment):
    """Wire `run_ai_jobs.main` to a scratch desk: scratch ledger, scratch packs,
    a fixed clock, and a model request that is a fake.

    `main` names no ledger path, no packs root and no `post`, which is the
    whole point of driving the flag through it - so each of those is redirected
    here instead. `requests.post` is replaced by an explosion: this file must
    never be able to reach 127.0.0.1:11434.
    """
    import ai_summary
    import requests
    from ai_jobs import ledger, store, window

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    monkeypatch.setattr(ledger, "ledger_path", lambda *, create=True: led)
    monkeypatch.setattr(store, "digests_dir", lambda *, create=True: packs)
    real_now = window.market_now
    monkeypatch.setattr(
        window, "market_now", lambda now=None: real_now(moment if now is None else now)
    )
    monkeypatch.setattr(requests, "post", _never_post)

    asked: list[str] = []

    def _fake_request(**kwargs):
        asked.append(str(kwargs.get("model") or ""))
        return {
            "model": kwargs.get("model"),
            "summary": {"executive_summary": "measured"},
            "usage": {"prompt_tokens": 41_234, "completion_tokens": 800 + len(asked)},
        }

    monkeypatch.setattr(ai_summary, "request_ai_summary", _fake_request)
    return led, asked


def test_force_is_what_lets_the_trader_measure_the_model_twice(tmp_path, monkeypatch):
    """The probe's own refusal says "pass --force to measure it again". Until
    the flag reached it that sentence was false: `--probe-model large` and
    `--probe-model large --force` behaved identically, so a desk with one
    measurement on the session could never be re-measured from the command
    line."""
    import run_ai_jobs
    from ai_jobs import model_probe

    led, asked = _cli_desk(monkeypatch, tmp_path, moment=SATURDAY_NIGHT)

    with _live_settings():
        first = run_ai_jobs.main(["--probe-model", "large"])
        again = run_ai_jobs.main(["--probe-model", "large"])
        forced = run_ai_jobs.main(["--probe-model", "large", "--force"])

    assert first == 0
    assert again == 1, "a second probe on the same session is refused"
    assert forced == 0, "--force is how the trader asks for another measurement"
    assert asked == [LARGE_TAG, LARGE_TAG], "the refused run must not ask the model"

    measured = [row for row in _probe_rows(led) if row.get("model_probe")]
    assert len(measured) == 2
    # Append-only: two rows, and the newest is the one the reserve reads.
    newest = model_probe.latest_measurement(tier="large", ledger_path=led)
    assert newest["completion_tokens"] == 802
    assert newest == measured[-1]["model_probe"]


def test_force_still_does_not_buy_the_clock_from_the_command_line(tmp_path, monkeypatch):
    """14:00 on a Saturday is exactly when the trader is at the desk. --force
    re-spends the already-measured check and nothing else."""
    import run_ai_jobs
    from ai_jobs import ledger

    daytime = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)
    led, asked = _cli_desk(monkeypatch, tmp_path, moment=daytime)

    with _live_settings():
        assert run_ai_jobs.main(["--probe-model", "large", "--force"]) == 1

    assert asked == []
    rows = _probe_rows(led)
    assert rows and rows[-1]["status"] == ledger.STATUS_SKIPPED
    assert "window" in rows[-1]["reason"].lower()


def test_force_does_not_beat_a_held_lock_from_the_command_line(tmp_path, monkeypatch):
    """The one refusal --force must never touch: another job is on the box."""
    import run_ai_jobs
    from ai_jobs import ledger, model_probe

    led, asked = _cli_desk(monkeypatch, tmp_path, moment=SATURDAY_NIGHT)
    monkeypatch.setattr(model_probe, "runner_lock", _held_lock())

    with _live_settings():
        assert run_ai_jobs.main(["--probe-model", "large", "--force"]) == 1

    assert asked == []
    rows = _probe_rows(led)
    assert rows and rows[-1]["status"] == ledger.STATUS_SKIPPED
    assert "lock" in rows[-1]["reason"].lower()


def test_the_weeknight_slate_is_e8c04f88s_set_moved_only_at_12_to_14():
    """The tester byte-pinned the weeknight slate to `e8c04f88`; the LEAD then
    moved `miss_contrast` above the `market_story_rollups` / `measured_report`
    pair on main (`1f260ffa`). Re-pinning a byte-pin is only safe if something
    proves the re-pin is that move and nothing else: same twenty names, same
    set, same order everywhere outside positions 12-14."""
    from ai_jobs import runner

    pinned_at_e8c04f88 = (
        "journal_import",
        "journal_auto_tag",
        "veto_cohort_grading",
        "like_cohort_grading",
        "sidecar_completion",
        "pass_cohort_grading",
        "rejection_cohort_grading",
        "note_vocabulary_audit",
        "preference_trade_outcomes",
        "evidence_report",
        "daily_digest",
        "theta_pick_grading",
        "market_story_rollups",
        "miss_contrast",
        "measured_report",
        "ticker_briefs",
        "market_story_narration",
        "journal_enrichment",
        "review_policy_draft",
        "setup_research",
    )
    slate = tuple(slot.name for slot in runner.slots_for("weeknight"))
    # LEAD AMENDMENT 2026-09-20 (TJ-10 integration): TJ-10 registered ONE new
    # deterministic slot, `read_grades_mature`, directly after
    # `theta_pick_grading`. This guard is about the e8c04f88 set and the 12-14
    # move, so the new slot is pinned where it sits and then set aside.
    assert slate[slate.index("theta_pick_grading") + 1] == "read_grades_mature"
    # LEAD AMENDMENT 2026-09-20 (TJ-4 + TJ-16 integration): three new slots,
    # each pinned where it sits and then set aside the way `read_grades_mature`
    # is, so this guard stays about the e8c04f88 set and the 12-14 move.
    # TJ-16's deterministic `prediction_contrast` sits directly after
    # `miss_contrast` in stage 1. In stage 2 the day story goes first (gate
    # #158 wants it finished before 23:30 Pacific and it is what the trader
    # reads in the morning), then TJ-16's word tagger, then the briefs with
    # their 120-minute reserve (a weeknight has no `ai_summary`).
    assert slate[slate.index("miss_contrast") + 1] == "prediction_contrast"
    # LEAD AMENDMENT 2026-09-21 (TJ-9E): a FOURTH stage-2 slot, the exit-note
    # reader, registered after `week_review_narration` - which a weeknight does
    # not carry - so on this slate it sits directly before the briefs and the
    # two-step chain becomes a three-step one.
    assert slate[slate.index("ticker_briefs") - 3] == "day_review_narration"
    assert slate[slate.index("ticker_briefs") - 2] == "observation_tags"
    assert slate[slate.index("ticker_briefs") - 1] == "exit_note_fields"
    set_aside = (
        "read_grades_mature",
        "prediction_contrast",
        "day_review_narration",
        "observation_tags",
        "exit_note_fields",
        # TJ-6 (2026-09-20): a new stage-3 slot is pinned where it sits by
        # `EXPECTED_SLOT_ORDER` and then set aside here, because this guard is
        # about the set of slots as it stood at e8c04f88.
        "improvement_ideas",
    )
    today = tuple(name for name in slate if name not in set_aside)

    assert len(today) == len(pinned_at_e8c04f88)
    assert set(today) == set(pinned_at_e8c04f88), "a slot was added or removed"
    assert today[:12] == pinned_at_e8c04f88[:12]
    assert today[15:] == pinned_at_e8c04f88[15:]
    assert sorted(today[12:15]) == sorted(pinned_at_e8c04f88[12:15])
    assert today[12:15] == ("miss_contrast", "market_story_rollups", "measured_report")


# ---------------------------------------------------------------------------
# the provider seam's edges
# ---------------------------------------------------------------------------


def test_openai_is_refused_by_the_request_seam_without_sending_anything(tmp_path):
    """"`openai` remains a setting, off." Typing it into the setting must not
    quietly spend money through a seam built for the local tiers."""
    import ai_summary
    from ai_jobs import provider

    overrides = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\n", encoding="utf-8")
        overrides[source_id] = path

    with _live_settings(ai_week_review_provider="openai"):
        evidence = ai_summary.build_evidence_package(
            ["daily_report"], source_overrides=overrides
        )
        assert provider.week_review_provider() == "openai"
        with pytest.raises(ValueError):
            provider.request_with_fallback(
                provider="openai", evidence=evidence, post=_never_post
            )


def test_an_unknown_week_review_provider_falls_back_to_the_large_local_model():
    """A typo in a settings file must not leave the week story with no provider
    at all - and it must not silently become a cloud one either."""
    from ai_jobs import provider

    with _live_settings(ai_week_review_provider="gpt-42"):
        assert provider.week_review_provider() == provider.LOCAL_LARGE


def test_a_medium_provider_asks_the_medium_model_and_has_nowhere_to_fall_back_to(tmp_path):
    """TJ-5 may pin the week story to the medium tier while the 27B is
    unmeasured. That is one model, and its failure is a failure."""
    import ai_summary
    from ai_jobs import ledger, provider

    overrides = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\n", encoding="utf-8")
        overrides[source_id] = path

    calls: list[str] = []

    def _post(url, **kwargs):
        calls.append(kwargs["json"]["model"])
        raise ConnectionError("[WinError 10061] No connection could be made")

    with _live_settings():
        evidence = ai_summary.build_evidence_package(
            ["daily_report"], source_overrides=overrides
        )
        outcome = provider.request_with_fallback(
            provider=provider.LOCAL_MEDIUM, evidence=evidence, post=_post
        )

    assert calls == [MEDIUM_TAG]
    assert outcome["status"] == ledger.STATUS_FAILED
    assert outcome["result"] is None
    assert outcome["attribution"]["model_asked"] == MEDIUM_TAG
    assert outcome["attribution"]["fallback_reason"]
