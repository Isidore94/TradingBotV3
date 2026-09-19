"""TJ-13A reviewer advisories, fixed before the merge (2026-09-19).

Three things tonight's first Saturday slate would have shown.

1. **`ai_summary` had `max_attempts=0`.** That was written when a failing
   summary was EXPENSIVE - it cost hours, so the night could not repeat it
   anyway. TJ-13A item 3 made a dead endpoint cheap, and cheap plus unbounded
   is a loop: the reviewer drove 16 Saturday-night firings against a degrading
   summary and it RAN TEN TIMES, writing ten `degraded_no_narrative` rows and
   ten four-file exports for one session, and Sunday would have done it again.
   plan.md §12.3 says every slot sets `max_attempts` and never 0.

2. **`--slot <name>` was filtered against the NIGHT'S slate**, so
   `--slot ai_summary` on a weeknight matched nothing, ran nothing and exited
   0 - while `run_ai_jobs.py`'s own docstring advertises that exact line. A
   typed slot is the operator's explicit choice and resolves against every
   registered slot; the night-only clock still refuses a model slot by day, so
   this widens what can be NAMED and not what can RUN.

3. **`record_rejected_reply` wrote to the DAS.** It is called from inside a
   slot, and `\\\\MINI-PC\\...` can be asleep: a ~20 s spin-up to file a
   diagnostic makes the record cost more than the thing it records. It writes
   LOCAL now, and prunes.

NO MODEL IS CALLED HERE.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")
SATURDAY_NIGHT = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)
WEEKEND_SESSION = "2026-09-18"

LIVE_START = "01:00"
LIVE_END = "09:00"


def _settings(**values):
    from ai_jobs import store

    return mock.patch.object(
        store._paths(),
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


def _no_session_block(monkeypatch):
    from ai_jobs import window

    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")


def _frozen_clock(monkeypatch, moment):
    from ai_jobs import window

    real = window.market_now
    monkeypatch.setattr(window, "market_now", lambda now=None: real(moment))


def _rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 1 - a cheap failure must not become a loop
# ---------------------------------------------------------------------------


def test_the_summary_slot_declares_an_attempt_cap():
    """plan.md §12.3: every slot sets `max_attempts`, never 0."""
    from ai_jobs import runner

    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert by_name["ai_summary"].max_attempts == 3


def test_sixteen_saturday_firings_run_a_degrading_summary_three_times(tmp_path, monkeypatch):
    """The reviewer's scenario, driven to the end.

    The scheduled task fires every 30 minutes for eight hours, so a weekend
    night is ~16 firings. Each one used to start the summary again.

    The spy's reason VARIES per attempt, the way the live rows do (they carry
    the slice counts and the elapsed read), so what bites here is the attempt
    CAP and not `attempt_cap_reason`'s identical-error rule - which would stop
    an unvarying failure even sooner, and is a different safeguard.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    _frozen_clock(monkeypatch, SATURDAY_NIGHT)
    led = tmp_path / "ledger.jsonl"
    runs: list[int] = []

    def _degrading(**kwargs):
        runs.append(len(runs) + 1)
        return {
            "status": ledger.STATUS_DEGRADED,
            "reason": (
                f"summary for {WEEKEND_SESSION} from 19 usable source(s) read as "
                f"{50 + len(runs)} of 53 slice(s); completion=unsynthesized_fallback"
            ),
            "outputs": [f"ai_summary_{len(runs)}.json"],
        }

    slot = next(
        replace(s, run=_degrading)
        for s in runner.default_slots()
        if s.name == "ai_summary"
    )

    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            for _firing in range(16):
                runner.run_slots([slot], now=SATURDAY_NIGHT, ledger_path=led)

    assert len(runs) == 3, f"16 firings ran the summary {len(runs)} times"

    rows = _rows(led)
    degraded = [row for row in rows if row["status"] == ledger.STATUS_DEGRADED]
    assert len(degraded) == 3
    # ...and the night says, once, that it is finished with this slot.
    terminal = [row for row in rows if row.get(ledger.TERMINAL_FIELD)]
    assert len(terminal) == 1
    assert "cap is 3" in terminal[0]["reason"]
    # Every later firing costs a lookup and nothing else - no ledger spam.
    assert len(rows) == 4


def test_a_capped_summary_is_not_offered_again_on_the_sunday_slate(tmp_path, monkeypatch):
    """Otherwise Sunday night simply repeats Saturday's loop.

    `slots_for("sunday")` already excludes a capped slot; this pins it for the
    slot the cap was just added to, because that is the pair the reviewer drove.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    _frozen_clock(monkeypatch, SATURDAY_NIGHT)
    led = tmp_path / "ledger.jsonl"

    def _degrading(**kwargs):
        return {
            "status": ledger.STATUS_DEGRADED,
            "reason": f"attempt at {datetime.now().isoformat()}",
        }

    slot = next(
        replace(s, run=_degrading)
        for s in runner.default_slots()
        if s.name == "ai_summary"
    )
    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            for _firing in range(6):
                runner.run_slots([slot], now=SATURDAY_NIGHT, ledger_path=led)

    names = [
        s.name
        for s in runner.slots_for(
            "sunday", session_date=WEEKEND_SESSION, ledger_path=led
        )
    ]
    assert "ai_summary" not in names


def test_a_daytime_forced_skip_does_not_burn_the_nights_attempts(tmp_path, monkeypatch):
    """The cap counts WORK, not refusals.

    A forced daytime run of a model slot records `skipped` (TJ-13A item 1), and
    `ledger.ATTEMPT_STATUSES` holds only `failed` and `degraded_no_narrative` -
    so an operator who tries three times at lunchtime has not spent the night's
    three attempts. Pinned because the cap is new and the two rows look alike
    in the file.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    led = tmp_path / "ledger.jsonl"
    saturday_afternoon = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)

    slot = next(
        replace(s, run=lambda **k: {})
        for s in runner.default_slots()
        if s.name == "ai_summary"
    )
    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            for _try in range(3):
                runner.run_slots(
                    [slot],
                    now=saturday_afternoon,
                    force=True,
                    ledger_path=led,
                )

    assert ledger.attempt_rows("ai_summary", WEEKEND_SESSION, path=led) == []
    assert not ledger.attempt_cap_reason(
        "ai_summary", WEEKEND_SESSION, max_attempts=3, path=led
    )


# ---------------------------------------------------------------------------
# 2 - a typed --slot is the operator's explicit choice
# ---------------------------------------------------------------------------


def _capture(monkeypatch, seen):
    from ai_jobs import runner

    def _fake(slots, **kwargs):
        seen.extend(slot.name for slot in slots)
        seen.append(f"only={kwargs.get('only', '')}")
        return runner.RunReport(session_date=WEEKEND_SESSION, started_at=SATURDAY_NIGHT)

    monkeypatch.setattr(runner, "run_slots", _fake)


def test_a_typed_slot_is_found_even_when_tonights_slate_omits_it(monkeypatch):
    """`--slot ai_summary` on a weeknight used to match nothing and exit 0.

    Silently. While the module docstring advertised that exact command.
    """
    import run_ai_jobs
    from ai_jobs import runner

    seen: list[str] = []
    _capture(monkeypatch, seen)
    monkeypatch.setattr(runner, "night_kind", lambda now=None: "weeknight", raising=False)

    assert run_ai_jobs.main(["--slot", "ai_summary"]) == 0
    assert "ai_summary" in seen, "a typed slot must be reachable"
    assert "only=ai_summary" in seen


def test_a_typed_optional_slot_is_reachable_too(monkeypatch):
    import run_ai_jobs
    from ai_jobs import runner

    seen: list[str] = []
    _capture(monkeypatch, seen)
    monkeypatch.setattr(runner, "night_kind", lambda now=None: "weeknight", raising=False)

    assert run_ai_jobs.main(["--slot", "weekly_synthesis"]) == 0
    assert "weekly_synthesis" in seen


def test_an_unknown_slot_name_is_an_error_and_never_a_silent_zero(monkeypatch, capsys):
    """"I typed it wrong" and "it ran and found nothing" must not look alike."""
    import run_ai_jobs
    from ai_jobs import runner

    seen: list[str] = []
    _capture(monkeypatch, seen)
    monkeypatch.setattr(runner, "night_kind", lambda now=None: "weeknight", raising=False)

    with pytest.raises(SystemExit) as caught:
        run_ai_jobs.main(["--slot", "ai_sumary"])

    assert caught.value.code != 0
    message = capsys.readouterr().err
    assert "ai_sumary" in message
    assert "ai_summary" in message, "the valid names are shown"
    assert seen == [], "nothing ran"


def test_a_typed_model_slot_still_obeys_the_night_clock(tmp_path, monkeypatch):
    """Widening what can be NAMED must not widen what can RUN.

    `--slot ai_summary --force` at 14:00 on a Saturday reaches the slot now,
    and the slot still refuses: local inference is night-only.
    """
    from ai_jobs import ledger, runner, store

    _no_session_block(monkeypatch)
    led = tmp_path / "ledger.jsonl"
    ran: list[str] = []

    slot = next(
        replace(s, run=lambda **k: ran.append("ai_summary"))
        for s in runner.default_slots()
        if s.name == "ai_summary"
    )
    with mock.patch.object(store, "store_available", return_value=(True, "ready")):
        with _settings(ai_offhours_start=LIVE_START, ai_offhours_end=LIVE_END):
            report = runner.run_slots(
                [slot],
                now=datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC),
                force=True,
                only="ai_summary",
                ledger_path=led,
            )

    assert ran == []
    assert [row["status"] for row in report.results] == [ledger.STATUS_SKIPPED]


# ---------------------------------------------------------------------------
# 3 - the rejected-reply log is LOCAL and bounded
# ---------------------------------------------------------------------------


def test_a_rejected_reply_is_written_local_and_never_to_the_das():
    """It is written from inside a slot, and the DAS can be asleep."""
    import ai_summary
    import project_paths

    assert ai_summary.rejected_replies_dir() == project_paths.AI_REJECTED_REPLIES_DIR
    resolved = str(project_paths.AI_REJECTED_REPLIES_DIR).upper()
    assert not resolved.startswith("\\\\"), "no UNC path: that is the DAS"
    assert "MINI-PC" not in resolved


def test_the_rejected_reply_log_keeps_the_newest_and_prunes_the_rest(tmp_path):
    """Bounded by COUNT, because each file is already bounded by size."""
    import ai_summary

    keep = ai_summary.MAX_REJECTED_REPLY_FILES
    for index in range(keep + 25):
        path = ai_summary.record_rejected_reply(
            schema_name="tradingbot_trade_enrichment",
            text=json.dumps({"attempt": index}),
            error=f"rejection {index}",
            logs_dir=tmp_path,
        )
        assert path is not None

    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == keep
    # The newest survived: the last rejection is the one being debugged.
    newest = max(files, key=lambda p: p.stat().st_mtime)
    assert json.loads(newest.read_text(encoding="utf-8"))["error"].startswith("rejection")
    kept_errors = {
        json.loads(path.read_text(encoding="utf-8"))["error"] for path in files
    }
    assert f"rejection {keep + 24}" in kept_errors
    assert "rejection 0" not in kept_errors


def test_pruning_never_raises_into_the_slot(tmp_path, monkeypatch):
    """The rule that outranks the feature, again, on the new code path."""
    import ai_summary

    def _explode(self, *args, **kwargs):
        raise OSError("the directory went away mid-prune")

    monkeypatch.setattr(Path, "unlink", _explode)

    for index in range(ai_summary.MAX_REJECTED_REPLY_FILES + 3):
        ai_summary.record_rejected_reply(
            schema_name="s", text="{}", error=str(index), logs_dir=tmp_path
        )
    # It still wrote what it was asked to write; only the tidying failed.
    assert len(sorted(tmp_path.glob("*.json"))) >= ai_summary.MAX_REJECTED_REPLY_FILES
