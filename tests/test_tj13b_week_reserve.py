"""TJ-13B item 1, second half: the week story's reserve is DERIVED, never guessed.

`plan.md` §12.4 TJ-13 item 7: *"The builder first measures the 27B on this desk -
load time, tokens per second, peak memory beside a running desk - on a copy of
one week's packs, and the slot's `reserve_minutes` comes from that number."*

Two consequences, and this file pins both:

* a reserve read off a recorded measurement moves when the measurement moves -
  a slower model or a longer load reserves more minutes;
* with NO measurement recorded there is no honest number, so the seam says so
  and the large model does not run. `plan.md` §12.3: a slot sets a
  `reserve_minutes` and `max_attempts` (never 0); a slot whose reserve is a
  guess would reserve the window against a model nobody has ever timed.

The week slot itself (`week_review_narration`) arrives with TJ-5. What TJ-13B
owes it is the ANSWER: this is the seam TJ-5 reads.

Contract pinned by these tests, for the builder::

    ai_jobs.model_probe.latest_measurement(*, tier="large", ledger_path=None)
        -> dict | None
    ai_jobs.model_probe.reserve_minutes_from_probe(*, tier="large",
                                                   ledger_path=None)
        -> float | None            # None == never measured, never a default
    ai_jobs.provider.week_review_plan(*, ledger_path=None) -> dict
        {"provider", "model", "reserve_minutes", "may_run_large", "reason"}

NO MODEL IS LOADED HERE: these tests read ledger rows and call no endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

WEEKEND_SESSION = "2026-09-18"
LARGE_TAG = "hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M"
MEDIUM_TAG = "gemma3:12b-tbv3ctx-64k"


def _settings(**values):
    import ai_summary
    import project_paths

    def _get(key, default=None):
        return values.get(key, default)

    class _Both:
        def __enter__(self):
            self._a = mock.patch.object(ai_summary, "get_local_setting", _get)
            self._b = mock.patch.object(project_paths, "get_local_setting", _get)
            self._a.start()
            self._b.start()
            return self

        def __exit__(self, *exc):
            self._b.stop()
            self._a.stop()
            return False

    return _Both()


def _record_measurement(
    path: Path,
    *,
    load_seconds: float,
    tokens_per_second: float,
    context_tokens_accepted: int = 41_234,
    peak_memory_mb: float = 21_800.0,
    model: str = LARGE_TAG,
    tier: str = "large",
) -> dict:
    """One probe row, written through the ledger's own writer.

    Shaped exactly as `run_model_probe` writes it, so this fixture is the
    contract and not a second one.
    """
    from ai_jobs import ledger, model_probe

    return ledger.record(
        job=model_probe.PROBE_JOB,
        status=ledger.STATUS_MANUAL,
        session_date=WEEKEND_SESSION,
        model=model,
        reason=f"measured {model} on a copy of 5 pack(s)",
        path=path,
        extra={
            "model_probe": {
                "model": model,
                "tier": tier,
                "packs": 5,
                "load_seconds": load_seconds,
                "tokens_per_second": tokens_per_second,
                "peak_memory_mb": peak_memory_mb,
                "context_tokens_accepted": context_tokens_accepted,
            }
        },
    )


def test_a_recorded_measurement_is_what_the_seam_reads_back(tmp_path):
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    _record_measurement(led, load_seconds=612.0, tokens_per_second=4.0)

    measured = model_probe.latest_measurement(tier="large", ledger_path=led)
    assert measured is not None
    assert measured["tokens_per_second"] == 4.0
    assert measured["load_seconds"] == 612.0
    assert measured["model"] == LARGE_TAG


def test_the_newest_measurement_wins_and_the_older_one_is_not_averaged(tmp_path):
    """The ledger is append-only, so two probes leave two rows. The desk the
    week story will run on is the one measured LAST."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    _record_measurement(led, load_seconds=900.0, tokens_per_second=2.0)
    _record_measurement(led, load_seconds=300.0, tokens_per_second=8.0)

    measured = model_probe.latest_measurement(tier="large", ledger_path=led)
    assert measured["tokens_per_second"] == 8.0
    assert measured["load_seconds"] == 300.0


def test_a_medium_tier_probe_is_never_read_as_the_large_one(tmp_path):
    """Rows are keyed by the tier they measured. A 12B probe answers nothing
    about the 27B, and 476 live ledger rows name only the 12B."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    _record_measurement(
        led, load_seconds=30.0, tokens_per_second=8.0, model=MEDIUM_TAG, tier="medium"
    )

    assert model_probe.latest_measurement(tier="large", ledger_path=led) is None
    assert model_probe.reserve_minutes_from_probe(tier="large", ledger_path=led) is None


def test_the_reserve_covers_the_measured_load_time(tmp_path):
    """A 10-minute load cannot fit in a reserve under 10 minutes.

    This is the one arithmetic floor that holds whatever formula the builder
    derives: the model has to BE there before it writes a word.
    """
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    _record_measurement(led, load_seconds=600.0, tokens_per_second=6.0)

    reserve = model_probe.reserve_minutes_from_probe(tier="large", ledger_path=led)
    assert reserve is not None
    assert reserve >= 10.0, f"a 600 s load reserved {reserve} min"


def test_a_slower_model_reserves_more_minutes_than_a_faster_one(tmp_path):
    """Derived, not written down: change the measurement and the number moves."""
    from ai_jobs import model_probe

    slow = tmp_path / "slow.jsonl"
    fast = tmp_path / "fast.jsonl"
    _record_measurement(slow, load_seconds=300.0, tokens_per_second=3.0)
    _record_measurement(fast, load_seconds=300.0, tokens_per_second=30.0)

    slow_reserve = model_probe.reserve_minutes_from_probe(tier="large", ledger_path=slow)
    fast_reserve = model_probe.reserve_minutes_from_probe(tier="large", ledger_path=fast)
    assert slow_reserve > fast_reserve, (
        f"3 tok/s reserved {slow_reserve} min and 30 tok/s reserved {fast_reserve}; "
        "the reserve is not reading the measurement"
    )


def test_a_longer_load_reserves_more_minutes_at_the_same_speed(tmp_path):
    from ai_jobs import model_probe

    quick = tmp_path / "quick.jsonl"
    slow_load = tmp_path / "slow_load.jsonl"
    _record_measurement(quick, load_seconds=60.0, tokens_per_second=6.0)
    _record_measurement(slow_load, load_seconds=900.0, tokens_per_second=6.0)

    assert model_probe.reserve_minutes_from_probe(
        tier="large", ledger_path=slow_load
    ) > model_probe.reserve_minutes_from_probe(tier="large", ledger_path=quick)


def test_with_no_measurement_there_is_no_reserve_at_all(tmp_path):
    """Not a default, not 15.0, not the window. `None` is the honest answer to
    "how long does a model nobody has ever run take"."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    led.write_text("", encoding="utf-8")

    assert model_probe.reserve_minutes_from_probe(tier="large", ledger_path=led) is None
    assert model_probe.latest_measurement(tier="large", ledger_path=led) is None


def test_with_no_measurement_the_week_plan_says_so_and_does_not_run_the_large_model(tmp_path):
    """The week slot asks this seam whether it may run the 27B tonight."""
    from ai_jobs import provider

    led = tmp_path / "ledger.jsonl"
    led.write_text("", encoding="utf-8")

    with _settings(ai_local_model_large=LARGE_TAG, ai_local_model_medium=MEDIUM_TAG):
        plan = provider.week_review_plan(ledger_path=led)

    assert plan["may_run_large"] is False
    assert plan["reserve_minutes"] is None
    assert plan["model"] != LARGE_TAG
    reason = plan["reason"].lower()
    assert "probe" in reason or "measure" in reason, plan["reason"]


def test_with_a_measurement_the_week_plan_names_the_large_model_and_its_reserve(tmp_path):
    from ai_jobs import model_probe, provider

    led = tmp_path / "ledger.jsonl"
    _record_measurement(led, load_seconds=600.0, tokens_per_second=6.0)

    with _settings(ai_local_model_large=LARGE_TAG, ai_local_model_medium=MEDIUM_TAG):
        plan = provider.week_review_plan(ledger_path=led)

    assert plan["may_run_large"] is True
    assert plan["model"] == LARGE_TAG
    assert plan["reserve_minutes"] == model_probe.reserve_minutes_from_probe(
        tier="large", ledger_path=led
    )
    assert plan["reserve_minutes"] > 0


def test_the_plan_is_a_report_and_writes_no_ledger_row(tmp_path):
    """Reading what was measured must not itself look like a measurement."""
    from ai_jobs import provider

    led = tmp_path / "ledger.jsonl"
    _record_measurement(led, load_seconds=600.0, tokens_per_second=6.0)
    before = led.read_text(encoding="utf-8")

    with _settings(ai_local_model_large=LARGE_TAG, ai_local_model_medium=MEDIUM_TAG):
        provider.week_review_plan(ledger_path=led)

    assert led.read_text(encoding="utf-8") == before
    assert len(
        [line for line in before.splitlines() if line.strip()]
    ) == 1, "fixture drift: one measurement row"
    assert json.loads(before.splitlines()[0])["model_probe"]["packs"] == 5
