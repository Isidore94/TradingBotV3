"""Forced re-simulation of every outcome bucket (BD-98).

The nightly build is idempotent by knowledge: a terminal outcome row is never
recomputed. After the duplicated M5 bars of 2026-08/09 that idempotency is the
problem - the stored rows were computed over doubled series - so ``force``
re-simulates terminal rows and ``recompute-outcomes`` walks every bucket with
it, one lock per bucket, inside a time budget.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from scripts.research_warehouse import cli, outcomes

UTC = timezone.utc
NOW = datetime(2026, 9, 4, 8, 0, tzinfo=UTC)


class _Store:
    """Just enough store for build_outcomes: a publish that records rows."""

    def __init__(self):
        self.published = []
        self.root = None

    def publish(self, dataset, rows, job_id=""):
        self.published.append((dataset, list(rows), job_id))

        class _R:
            rows_published = len(rows)

        return _R()


def _occurrence():
    return {"occurrence_id": "occ-1", "symbol": "AAPL", "trigger_at": NOW, "canonical_setup_id": "x"}


def _terminal_row():
    return {"result_state": next(iter(outcomes.TERMINAL_RESULT_STATES)), "r_multiple": 1.0}


def test_force_resimulates_a_terminal_row_and_writes_only_when_it_changed(monkeypatch):
    recipe = outcomes.SWING_HOUSE_V1
    stored = _terminal_row()
    monkeypatch.setattr(
        outcomes, "latest_outcomes", lambda store, ids: {(ids[0], recipe.recipe_id, outcomes.OUTCOME_DEFINITION_ID): stored}
    )
    computed = {"result_state": stored["result_state"], "r_multiple": 1.0, "occurrence_id": "occ-1"}
    monkeypatch.setattr(outcomes, "simulate_swing", lambda *a, **k: dict(computed))
    monkeypatch.setattr(outcomes, "_same_outcome", lambda previous, now: previous.get("r_multiple") == now.get("r_multiple"))

    store = _Store()
    plain = outcomes.build_outcomes(store, [_occurrence()], recipes=(recipe,), now=NOW)
    assert plain.skipped.get("ALREADY_SIMULATED") == 1 and not store.published

    same = outcomes.build_outcomes(store, [_occurrence()], recipes=(recipe,), now=NOW, force=True)
    assert same.skipped.get("UNCHANGED") == 1 and not store.published, "a re-simulation that learned nothing writes nothing"

    computed["r_multiple"] = 0.4  # the doubled-bars row was wrong
    changed = outcomes.build_outcomes(store, [_occurrence()], recipes=(recipe,), now=NOW, force=True)
    assert changed.rows == 1 and store.published[0][0] == "outcome_path"


def test_the_bucket_override_reaches_the_symbol_split(monkeypatch):
    """`_run_outcomes(bucket=)` must select that bucket, not the (day, hour) one."""
    seen = {}

    def fake_latest(store, year):
        return {f"occ-{i}": {"occurrence_id": f"occ-{i}", "symbol": f"SYM{i:03d}"} for i in range(200)}

    monkeypatch.setattr(cli.occurrences, "latest_occurrences", fake_latest)

    def fake_window(store, day):
        seen["window"] = True
        raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(cli.features, "daily_history_window", fake_window)
    with pytest.raises(_Stop):
        cli._run_outcomes(object(), date(2026, 9, 4), NOW, "r", bucket=7)
    assert seen["window"]
    # A bucket with no symbols returns NOTHING_IN_BUCKET rather than raising.
    monkeypatch.setattr(cli.occurrences, "latest_occurrences", lambda store, year: {"occ-1": {"occurrence_id": "occ-1", "symbol": "ZZZ"}} if year == 2026 else {})
    monkeypatch.setattr(cli, "OUTCOME_BUCKET_MIN_SYMBOLS", 0)
    import hashlib

    zzz_bucket = int(hashlib.sha256(b"ZZZ").hexdigest()[:8], 16) % cli.OUTCOME_BUCKETS
    other = (zzz_bucket + 1) % cli.OUTCOME_BUCKETS
    result = cli._run_outcomes(object(), date(2026, 9, 4), NOW, "r", bucket=other)
    assert result["status"] == "NOTHING_IN_BUCKET" and result["bucket"] == other


def test_recompute_walks_every_bucket_under_its_own_lock_and_respects_the_budget(monkeypatch, tmp_path):
    calls = []

    def fake_run(store, day, stamp, run_id, *, bucket=None, force=False):
        calls.append((bucket, force, run_id))
        return {"status": "OK", "bucket": bucket, "bucket_count": 32, "symbols": 3, "occurrences": 5, "m5_close": {"rows": 1, "skipped": {}}}

    monkeypatch.setattr(cli, "_run_outcomes", fake_run)
    firings = []
    monkeypatch.setattr(cli.outcome_coverage, "record_firing", lambda root, step, run_id="", now=None: firings.append((step["bucket"], run_id)))

    class _S:
        root = tmp_path

    plan = cli.run_recompute_outcomes(_S(), buckets=[3, 4, 5], now=NOW, lock_path=tmp_path / "lock")
    assert plan["applied"] is False and plan["buckets_planned"] == [3, 4, 5] and calls == []

    report = cli.run_recompute_outcomes(_S(), buckets=[3, 4, 5], apply=True, now=NOW, lock_path=tmp_path / "lock")
    assert report["status"] == "OK" and report["buckets_done"] == [3, 4, 5]
    assert [c[0] for c in calls] == [3, 4, 5] and all(c[1] is True for c in calls)
    assert [f[0] for f in firings] == [3, 4, 5] and firings[0][1] == "outcomes_recompute-b03"
    assert not (tmp_path / "lock").exists(), "the lock is released between buckets"

    # A zero budget starts the first bucket and then stops.
    calls.clear()
    budget = cli.run_recompute_outcomes(_S(), buckets=[1, 2], apply=True, now=NOW, lock_path=tmp_path / "lock", time_budget_minutes=0.0)
    assert budget["buckets_done"] == [1] and budget["buckets_skipped"] == [2] and budget["status"] == "BUDGET_EXHAUSTED"


def test_recompute_is_inert_without_a_store():
    assert cli.run_recompute_outcomes(None)["status"] == "DISABLED"
