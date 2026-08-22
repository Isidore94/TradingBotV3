"""R9.5: `sector_cohort_divergence`, a SHADOW-only cohort watch.

Ladder: PLANNED -> IMPLEMENTED -> **GREEN with the golden fixture frozen first**
-> SHADOW, and it stops there. Nothing in it may reach a detector, score,
ranking, routing, alert, watchlist, Focus, the review queue or
`review_policy.json`.

`tests/fixtures/sector_cohort_v1.json` was frozen by
`scripts/build_sector_cohort_fixture.py` on 2026-08-22, BEFORE this detector
existed. Its inputs are hand-constructed so each case isolates exactly one rule
and no vendor bar revision can move it. SPY is flat throughout, so each ETF's
spread is its own move - which is also what the real 2026-08-21 looked like
(SPY -0.05%, XLU -2.57%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sector_cohort_divergence as scd  # noqa: E402

FIXTURE = json.loads(
    (ROOT_DIR / "tests" / "fixtures" / "sector_cohort_v1.json").read_text(encoding="utf-8")
)


def _observations():
    return scd.detect_cohorts(
        benchmark_bars=FIXTURE["benchmark"]["bars"],
        etf_bars=FIXTURE["etfs"],
        config=scd.resolve_config(),
    )


# ---------------------------------------------------------------------------
# the golden fixture, case by case
# ---------------------------------------------------------------------------
def test_the_fixture_was_frozen_before_the_detector():
    """A golden file that can be regenerated on a whim is not golden."""
    assert FIXTURE["fixture"] == "sector_cohort_v1"
    assert FIXTURE["frozen_at"] == "2026-08-22"
    assert FIXTURE["rules_under_test"] == {
        "threshold_pct": 0.75,
        "persistence_bars": 3,
        "completed_bars_only": True,
        "session_only": True,
    }


def test_the_detector_reproduces_every_frozen_expectation():
    fired = {obs.etf: obs for obs in _observations()}
    for expected in FIXTURE["expected_observations"]:
        etf = expected["etf"]
        if expected["fires"]:
            assert etf in fired, f"{etf} should fire: {expected['why']}"
            assert fired[etf].first_fire_bar_index == expected["first_fire_bar_index"], etf
        else:
            assert etf not in fired, f"{etf} must not fire: {expected['why']}"


def test_two_qualifying_bars_are_not_three():
    """XLK reaches -0.90% and holds it for exactly two bars. The persistence
    rule exists for precisely this shape."""
    assert "XLK" not in {obs.etf for obs in _observations()}


def test_an_opening_gap_that_closes_never_fires():
    """31 of the review's 179 measured fires sat on the 09:30 bar and were gap
    artifacts. XRT is that shape."""
    assert "XRT" not in {obs.etf for obs in _observations()}


def test_the_long_mirror_fires_too():
    """A short-only cohort rule would have found the utilities and missed every
    sector that leads."""
    xlv = {obs.etf: obs for obs in _observations()}.get("XLV")
    assert xlv is not None
    assert xlv.direction == "long"


def test_the_short_side_is_labelled_short():
    xlu = {obs.etf: obs for obs in _observations()}["XLU"]
    assert xlu.direction == "short"
    assert xlu.spread_at_fire_pct < 0
    assert xlu.max_abs_spread_pct >= 0.75


# ---------------------------------------------------------------------------
# the invariants
# ---------------------------------------------------------------------------
def test_a_forming_bar_is_never_consumed():
    """plan.md sec 5: state transitions use completed bars only.

    Truncating the last bar must not change any verdict, because the last bar
    was never allowed to decide one.
    """
    truncated = {etf: bars[:-1] for etf, bars in FIXTURE["etfs"].items()}
    full = {obs.etf: obs.first_fire_bar_index for obs in _observations()}
    partial = {
        obs.etf: obs.first_fire_bar_index
        for obs in scd.detect_cohorts(
            benchmark_bars=FIXTURE["benchmark"]["bars"][:-1],
            etf_bars=truncated,
            config=scd.resolve_config(),
        )
    }
    assert full == partial


def test_a_missing_benchmark_yields_nothing_rather_than_a_guess():
    """Missing data is uncertainty, never confirmation. With no SPY there is no
    spread, and a bare ETF move is not a divergence."""
    assert scd.detect_cohorts(benchmark_bars=[], etf_bars=FIXTURE["etfs"], config=scd.resolve_config()) == []


def test_an_etf_with_too_few_bars_is_skipped_and_counted():
    etfs = dict(FIXTURE["etfs"])
    etfs["XLU"] = etfs["XLU"][:2]
    result = scd.run_detection(
        benchmark_bars=FIXTURE["benchmark"]["bars"],
        etf_bars=etfs,
        config=scd.resolve_config(),
    )
    assert "XLU" not in {obs.etf for obs in result.observations}
    assert result.coverage["etfs_skipped_short_series"] == 1


def test_a_session_is_re_derived_never_carried():
    """Two sessions in one frame must be measured independently, and the rule
    only ever looks at one of them."""
    assert scd.SESSION_ONLY is True
    day_one = FIXTURE["etfs"]["XLU"]
    doubled = day_one + [
        {**bar, "dt": bar["dt"].replace("2026-08-21", "2026-08-22")} for bar in day_one
    ]
    observations = scd.detect_cohorts(
        benchmark_bars=FIXTURE["benchmark"]["bars"],
        etf_bars={"XLU": doubled},
        config=scd.resolve_config(),
    )
    # The benchmark only covers 08-21, so only that session is measurable.
    assert {obs.session for obs in observations} == {"2026-08-21"}


def test_an_unknown_sector_is_excluded_rather_than_admitted():
    etfs = dict(FIXTURE["etfs"])
    etfs[""] = etfs["XLU"]
    observations = scd.detect_cohorts(
        benchmark_bars=FIXTURE["benchmark"]["bars"], etf_bars=etfs, config=scd.resolve_config()
    )
    assert all(obs.etf for obs in observations)


# ---------------------------------------------------------------------------
# gates 1, 3 and 7
# ---------------------------------------------------------------------------
def test_gate_1_the_config_is_versioned_and_hashed():
    config = scd.resolve_config()
    assert config["version"] == scd.CONFIG_VERSION
    first = scd.config_hash(config)
    assert first == scd.config_hash(scd.resolve_config()), "the hash must be stable"
    moved = dict(config, threshold_pct=1.5)
    assert scd.config_hash(moved) != first, "a changed rule must change the identity"


def test_gate_3_every_run_accounts_for_its_coverage():
    result = scd.run_detection(
        benchmark_bars=FIXTURE["benchmark"]["bars"],
        etf_bars=FIXTURE["etfs"],
        config=scd.resolve_config(),
    )
    for key in (
        "etfs_requested",
        "etfs_measured",
        "etfs_skipped_short_series",
        "benchmark_bars",
        "bars_consumed",
        "observations",
    ):
        assert key in result.coverage, key
    assert result.coverage["etfs_measured"] == 5
    assert result.coverage["observations"] == 2


def test_gate_7_one_switch_and_it_is_off():
    """SHADOW means it must be possible to stop it without a code revert, and
    a thing at SHADOW ships off."""
    assert scd.SECTOR_COHORT_DEFAULTS["enabled"] is False
    assert scd.resolve_config()["enabled"] is False
    assert scd.resolve_config({"enabled": True})["enabled"] is True


# ---------------------------------------------------------------------------
# the ladder boundary
# ---------------------------------------------------------------------------
def test_it_reaches_no_live_surface():
    """The whole reason it is allowed to exist at SHADOW.

    Parsed rather than grepped: the module docstring names these surfaces in
    order to promise it does not touch them, and a substring scan cannot tell a
    promise from a call.
    """
    import ast

    tree = ast.parse((SCRIPTS_DIR / "sector_cohort_divergence.py").read_text(encoding="utf-8"))
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            reached.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            reached.add(node.module or "")
            reached.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            reached.add(node.attr)
        elif isinstance(node, ast.Name):
            reached.add(node.id)
    for forbidden in (
        "review_policy",
        "focus_service",
        "FocusService",
        "record_review_event",
        "add_alert",
        "CandidateRegistry",
        "LONGS_FILE",
        "FOCUS_LONGS_FILE",
        "focus_picks",
        "pick_feedback",
    ):
        assert not any(forbidden in name for name in reached), f"{forbidden}: {sorted(reached)}"


def test_it_spends_no_ib_budget():
    """Batched yfinance over the ETF set, the Strength Board template."""
    source = (SCRIPTS_DIR / "sector_cohort_divergence.py").read_text(encoding="utf-8")
    for forbidden in ("ibapi", "IBKR", "ib_client", "reqHistoricalData"):
        assert forbidden not in source, forbidden


def test_the_shadow_log_lands_under_shadow_evidence(tmp_path):
    result = scd.run_detection(
        benchmark_bars=FIXTURE["benchmark"]["bars"],
        etf_bars=FIXTURE["etfs"],
        config=scd.resolve_config(),
    )
    out = tmp_path / "sector_cohort.jsonl"
    scd.write_shadow_rows(result, path=out, config=scd.resolve_config())
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "a run with observations writes them"
    assert all(row["schema"] == scd.SHADOW_SCHEMA for row in rows)
    assert all(row["config_hash"] for row in rows)
    assert {row["kind"] for row in rows} == {"coverage", "observation"}
    assert scd.default_shadow_path().parent.name == "sector_cohort"
    assert scd.default_shadow_path().parent.parent.name == "shadow_evidence"


def test_a_quiet_run_still_writes_its_coverage(tmp_path):
    """A day with no cohort is evidence, not silence - otherwise an outage and
    a calm market look identical in the log."""
    result = scd.run_detection(
        benchmark_bars=FIXTURE["benchmark"]["bars"],
        etf_bars={"XLE": FIXTURE["etfs"]["XLE"]},
        config=scd.resolve_config(),
    )
    assert result.observations == []
    out = tmp_path / "sector_cohort.jsonl"
    scd.write_shadow_rows(result, path=out, config=scd.resolve_config())
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["kind"] for row in rows] == ["coverage"]


# ---------------------------------------------------------------------------
# member entry timing (the archetype, reused rather than re-derived)
# ---------------------------------------------------------------------------
def _member_session(**overrides):
    """A member that satisfies the archetype unless a rule is overridden."""
    return scd.member_entry(
        bars=overrides.get("bars", _archetype_bars()),
        prior_day_low=overrides.get("prior_day_low", 99.0),
        config=scd.resolve_config(),
    )


def _archetype_bars():
    """High in the first three bars, prior-day low broken in the first hour,
    then a close below session VWAP and below the prior bar's low after 10:00."""
    from datetime import datetime, timedelta

    start = datetime(2026, 8, 21, 9, 30)
    path = (
        [100.5, 100.2, 99.4, 98.6, 98.2, 98.4]           # 09:30-09:55, breaks 99.0
        + [98.6, 98.8, 99.0, 99.1, 99.0, 98.9]           # 10:00-10:25 retrace
        + [98.3, 98.0, 97.6, 97.2, 96.9, 96.5]           # 10:30-10:55 rolls over
        + [96.2] * 20
    )
    return [
        {
            "dt": (start + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:00"),
            "open": round(close + 0.05, 4),
            "high": round(close + 0.15, 4),
            "low": round(close - 0.15, 4),
            "close": round(close, 4),
            "volume": 50_000.0,
        }
        for i, close in enumerate(path)
    ]


def test_a_member_entry_is_after_ten_and_below_vwap_and_the_prior_bar_low():
    entry = _member_session()
    assert entry is not None
    assert entry.entry_time_et >= "10:00"
    assert entry.entry_time_et <= "11:30"
    assert entry.stop > entry.fill, "a short's stop is above the fill"


def test_a_member_whose_high_is_not_in_the_first_three_bars_is_not_one():
    bars = _archetype_bars()
    # Index 10 is 10:20, before the 10:30 decision bar - so by then the session
    # high is NOT in the opening drive and the archetype does not hold. A spike
    # at index 20 (11:10) would be after the decision and must NOT disqualify
    # it, which is the point-in-time half of the same rule.
    bars[10]["high"] = 200.0
    assert _member_session(bars=bars) is None


def test_a_high_made_after_the_entry_does_not_retroactively_disqualify_it():
    """Point-in-time: the decision could only read the session so far."""
    bars = _archetype_bars()
    bars[20]["high"] = 200.0
    assert _member_session(bars=bars) is not None


def test_a_member_that_never_broke_the_prior_day_low_is_not_one():
    assert _member_session(prior_day_low=1.0) is None


def test_a_member_with_no_prior_day_low_is_excluded_not_assumed():
    assert _member_session(prior_day_low=None) is None


@pytest.mark.parametrize("field", ["fill", "stop", "risk_pct", "entry_time_et"])
def test_the_member_record_carries_what_a_measurement_needs(field):
    entry = _member_session()
    assert getattr(entry, field) is not None
