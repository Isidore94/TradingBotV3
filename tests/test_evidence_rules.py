"""R10 ground rule 5: known-bad legacy rows are TAGGED, never edited.

`daily_volume_mixed_v1` is the first rule in the registry. It answers one
question about a market session - "did any scan that day write IB-unit daily
volume into the durable store?" - from the run manifests' own
`provider.daily_bars.success.*` counters, so the answer is derived evidence
rather than a hard-coded date list.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_rules  # noqa: E402


def _manifest(tmp_path: Path, run_id: str, started_at: str, **success) -> Path:
    counters = {f"provider.daily_bars.success.{k}": v for k, v in success.items()}
    counters["provider.daily_bars.lookup"] = sum(success.values())
    payload = {
        "schema": "run_manifest_v1",
        "run_id": run_id,
        "job_type": "master_scan",
        "started_at": started_at,
        "status": "ok",
        "counters": counters,
    }
    path = tmp_path / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the registry itself
# ---------------------------------------------------------------------------
def test_the_rule_is_registered_under_its_versioned_name():
    spec = evidence_rules.RULES[evidence_rules.RULE_DAILY_VOLUME_MIXED]
    assert spec.name == "daily_volume_mixed_v1"
    assert spec.name.endswith("_v1"), "rules are versioned by name and never edited in place"
    assert spec.summary


def test_every_registered_rule_is_keyed_by_its_own_name():
    for key, spec in evidence_rules.RULES.items():
        assert key == spec.name


# ---------------------------------------------------------------------------
# reading the manifests
# ---------------------------------------------------------------------------
def test_an_ib_run_marks_its_session_mixed(tmp_path):
    _manifest(tmp_path, "master_scan-a", "2026-08-21T14:00:30+00:00", ibkr=1222, yahoo=22)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 21)] == evidence_rules.VERDICT_MIXED


def test_a_yahoo_only_session_reads_shares(tmp_path):
    _manifest(tmp_path, "master_scan-a", "2026-08-17T14:30:13+00:00", yahoo=1364)
    _manifest(tmp_path, "master_scan-b", "2026-08-17T20:00:19+00:00", yahoo=1176)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 17)] == evidence_rules.VERDICT_SHARES


def test_one_ib_run_contaminates_a_session_of_yahoo_runs(tmp_path):
    """2026-08-20: two desks ran concurrently and only one used IB."""
    _manifest(tmp_path, "master_scan-a", "2026-08-20T16:00:11+00:00", ibkr=444, yahoo=5)
    _manifest(tmp_path, "master_scan-b", "2026-08-20T16:00:15+00:00", yahoo=1138)
    _manifest(tmp_path, "master_scan-c", "2026-08-20T20:30:39+00:00", yahoo=1147)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 20)] == evidence_rules.VERDICT_MIXED


def test_sessions_are_market_local_not_utc(tmp_path):
    """20:30 ET on 08-21 is 2026-08-22 in UTC. The session is 08-21."""
    _manifest(tmp_path, "master_scan-a", "2026-08-22T00:30:00+00:00", ibkr=10)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert list(verdicts) == [date(2026, 8, 21)]


def test_a_run_that_fetched_no_daily_bars_votes_on_nothing(tmp_path):
    _manifest(tmp_path, "master_scan-a", "2026-08-19T14:30:25+00:00")
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts.get(date(2026, 8, 19), evidence_rules.VERDICT_UNKNOWN) == \
        evidence_rules.VERDICT_UNKNOWN


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def test_a_session_with_no_manifest_is_unknown_not_clean(tmp_path):
    """Manifests are pruned to 90. Absence of evidence is not evidence of Yahoo."""
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert evidence_rules.daily_volume_mixed_v1(date(2026, 5, 1), verdicts).verdict == \
        evidence_rules.VERDICT_UNKNOWN


def test_an_unreadable_manifest_makes_its_session_unknown(tmp_path):
    """The session comes off the run id's own stamp, which is how scans name them."""
    _manifest(tmp_path, "master_scan-20260818T160023Z-aaaaaa",
              "2026-08-18T16:00:23+00:00", yahoo=1168)
    (tmp_path / "master_scan-20260818T190023Z-bbbbbb.json").write_text(
        "{ truncated", encoding="utf-8")
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 18)] == evidence_rules.VERDICT_UNKNOWN, \
        "a manifest we cannot read may have been the IB one"


def test_a_proven_ib_run_outranks_an_unreadable_one(tmp_path):
    """Mixed dominates: the contamination is proven whatever the unreadable run did."""
    _manifest(tmp_path, "master_scan-20260818T160023Z-aaaaaa",
              "2026-08-18T16:00:23+00:00", ibkr=1168, yahoo=9)
    (tmp_path / "master_scan-20260818T190023Z-bbbbbb.json").write_text(
        "{ truncated", encoding="utf-8")
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 18)] == evidence_rules.VERDICT_MIXED


def test_an_undatable_unreadable_manifest_is_reported_rather_than_dropped(tmp_path):
    """It taints no particular session, so it must not vanish either."""
    _manifest(tmp_path, "master_scan-20260818T160023Z-aaaaaa",
              "2026-08-18T16:00:23+00:00", yahoo=1168)
    (tmp_path / "junk.json").write_text("{ truncated", encoding="utf-8")
    assert None in evidence_rules.unreadable_sessions(tmp_path)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 18)] == evidence_rules.VERDICT_SHARES


def test_a_missing_manifest_directory_is_unknown_rather_than_an_error(tmp_path):
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path / "nope")
    assert verdicts == {}


# ---------------------------------------------------------------------------
# what a rollup prints
# ---------------------------------------------------------------------------
def test_the_tag_carries_the_rule_name_and_a_reason(tmp_path):
    _manifest(tmp_path, "master_scan-a", "2026-08-21T14:00:30+00:00", ibkr=1222, yahoo=22)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    tag = evidence_rules.daily_volume_mixed_v1(date(2026, 8, 21), verdicts)
    assert tag.rule == "daily_volume_mixed_v1"
    assert tag.tagged is True
    assert "IB" in tag.reason or "ibkr" in tag.reason


def test_a_clean_session_is_not_tagged(tmp_path):
    _manifest(tmp_path, "master_scan-a", "2026-08-17T14:30:13+00:00", yahoo=1364)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    tag = evidence_rules.daily_volume_mixed_v1(date(2026, 8, 17), verdicts)
    assert tag.tagged is False
    assert tag.verdict == evidence_rules.VERDICT_SHARES


def test_a_rollup_reports_the_tagged_count_beside_n(tmp_path):
    """Ground rule 6: counted in every rollup beside n, never silently dropped."""
    _manifest(tmp_path, "master_scan-a", "2026-08-21T14:00:30+00:00", ibkr=1222, yahoo=22)
    _manifest(tmp_path, "master_scan-b", "2026-08-17T14:30:13+00:00", yahoo=1364)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    sessions = [date(2026, 8, 21)] * 3 + [date(2026, 8, 17)] * 5 + [date(2026, 5, 1)] * 2
    counts = evidence_rules.daily_volume_tag_counts(sessions, verdicts)
    assert counts == {
        evidence_rules.VERDICT_MIXED: 3,
        evidence_rules.VERDICT_SHARES: 5,
        evidence_rules.VERDICT_UNKNOWN: 2,
    }
    line = evidence_rules.format_tag_counts(counts)
    assert "n=10" in line
    assert "3" in line and "2" in line
    assert "daily_volume_mixed_v1" in line


def test_the_rollup_line_stays_quiet_when_nothing_is_tagged(tmp_path):
    counts = {evidence_rules.VERDICT_SHARES: 12}
    assert evidence_rules.format_tag_counts(counts) == "n=12"


# ---------------------------------------------------------------------------
# it is a reader, and only a reader
# ---------------------------------------------------------------------------
def test_the_module_never_writes():
    """R10 ground rule 5: the registry tags on the way OUT and edits nothing."""
    import inspect

    source = inspect.getsource(evidence_rules)
    for forbidden in ("write_text(", "write_bytes(", ".to_parquet", "os.replace", '"w"', "'w'"):
        assert forbidden not in source, f"a reader-side rule registry must not {forbidden}"


def test_the_verdicts_are_json_serialisable_so_they_can_be_frozen(tmp_path):
    """Manifest retention is 90 runs; a rollup that must stay reproducible
    snapshots the verdicts rather than re-deriving them later."""
    _manifest(tmp_path, "master_scan-a", "2026-08-21T14:00:30+00:00", ibkr=1222, yahoo=22)
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    frozen = evidence_rules.freeze_verdicts(verdicts)
    assert json.loads(json.dumps(frozen)) == frozen
    assert evidence_rules.thaw_verdicts(frozen) == verdicts


@pytest.mark.parametrize("provider", ["ibkr", "IBKR", "ib"])
def test_any_non_yahoo_provider_counts_as_ib_unit(tmp_path, provider):
    """The store's unit problem is 'not Yahoo', not 'named ibkr'."""
    _manifest(tmp_path, "master_scan-a", "2026-08-21T14:00:30+00:00", **{provider: 5})
    verdicts = evidence_rules.daily_volume_session_verdicts(tmp_path)
    assert verdicts[date(2026, 8, 21)] == evidence_rules.VERDICT_MIXED


def test_the_default_manifest_dir_is_the_diagnostics_tree():
    from diagnostics.run_manifest import default_manifest_dir

    assert evidence_rules.default_manifest_dir() == default_manifest_dir()


def test_scanning_reports_each_run_with_its_counts(tmp_path):
    _manifest(tmp_path, "master_scan-a", "2026-08-21T14:00:30+00:00", ibkr=1222, yahoo=22)
    runs = evidence_rules.scan_daily_bar_runs(tmp_path)
    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "master_scan-a"
    assert run.yahoo == 22
    assert run.non_yahoo == {"ibkr": 1222}
    assert run.session_date == date(2026, 8, 21)
    assert run.started_at == datetime(2026, 8, 21, 14, 0, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# R10.A: the other three rules the audit proved
# ---------------------------------------------------------------------------
def test_every_rule_states_its_measured_precision_not_a_round_number():
    """`h1_bar_start_v1` is 9,623/9,914, and the registry has to say so."""
    spec = evidence_rules.RULES[evidence_rules.RULE_H1_BAR_START]
    assert "9,623" in spec.precision and "9,914" in spec.precision
    assert "100%" not in spec.precision


def test_an_h1_row_stamped_on_the_half_hour_is_a_bar_start():
    tag = evidence_rules.h1_bar_start_v1("h1_ema10_bounce", "2026-08-19 09:30:00")
    assert tag.tagged and tag.rule == "h1_bar_start_v1"
    assert "bar start" in tag.reason


def test_the_h1_rule_is_conjunctive_family_and_minute():
    """291 of 6,054 non-H1 rows also land on minute 30; family alone is not it."""
    assert not evidence_rules.h1_bar_start_v1("regime_pause_rw", "2026-08-19 09:30:00").tagged
    assert not evidence_rules.h1_bar_start_v1("h1_ema10_bounce", "2026-08-19 09:55:00").tagged


@pytest.mark.parametrize("stamp", ["", None, "not-a-time", "2026-08-19"])
def test_an_unreadable_stamp_is_unknown_rather_than_untagged(stamp):
    tag = evidence_rules.h1_bar_start_v1("h1_ema10_bounce", stamp)
    assert tag.verdict == evidence_rules.VERDICT_UNKNOWN
    assert tag.tagged is False


def test_a_zero_close_that_equals_its_entry_is_a_fabricated_final():
    """1,164 of 1,164 zero finals have `eod_close == entry_price`; 0 of 5,743 others."""
    tag = evidence_rules.fabricated_zero_v1(close_r=0.0, eod_close=12.34, entry_price=12.34)
    assert tag.tagged
    assert "never measured" in tag.reason or "fabricat" in tag.reason


def test_a_real_scratch_is_not_tagged():
    """A genuine 0R exit closes somewhere other than exactly its entry."""
    assert not evidence_rules.fabricated_zero_v1(
        close_r=0.0, eod_close=12.35, entry_price=12.34
    ).tagged
    assert not evidence_rules.fabricated_zero_v1(
        close_r=1.5, eod_close=12.34, entry_price=12.34
    ).tagged


def test_a_missing_number_in_the_zero_rule_is_unknown():
    tag = evidence_rules.fabricated_zero_v1(close_r=0.0, eod_close=None, entry_price=12.34)
    assert tag.verdict == evidence_rules.VERDICT_UNKNOWN


def test_a_stop_closer_than_a_tenth_of_a_percent_is_below_the_floor():
    """R9.3's floor. 1,127 all-time finals qualify; max |close_r| all-time is 799."""
    tag = evidence_rules.risk_below_floor_v1(risk_per_share=0.004, entry_price=10.0)
    assert tag.tagged
    tag = evidence_rules.risk_below_floor_v1(risk_per_share=0.05, entry_price=10.0)
    assert not tag.tagged


def test_the_floor_rule_is_unknown_when_it_cannot_measure():
    for risk, entry in ((None, 10.0), (0.5, None), (0.5, 0.0)):
        assert evidence_rules.risk_below_floor_v1(
            risk_per_share=risk, entry_price=entry
        ).verdict == evidence_rules.VERDICT_UNKNOWN


def test_duplicates_are_counted_over_a_stated_window():
    """Every number from this store carries its window (Amendment 2a)."""
    rows = [
        {"event_id": "a", "event_type": "registered"},
        {"event_id": "a", "event_type": "registered"},
        {"event_id": "b", "event_type": "registered"},
        {"event_id": "c", "event_type": "final"},
        {"event_id": "c", "event_type": "final"},
        {"event_id": "c", "event_type": "final"},
    ]
    result = evidence_rules.duplicate_row_v1(rows, window="2026-08-07..2026-08-21")
    assert result["window"] == "2026-08-07..2026-08-21"
    assert result["by_event_type"]["registered"] == {"extra_rows": 1, "ids": 1}
    assert result["by_event_type"]["final"] == {"extra_rows": 2, "ids": 1}
    assert result["duplicate_ids"] == 2
    assert result["rows"] == 6


def test_a_row_with_no_id_is_counted_as_unidentifiable_not_as_unique():
    rows = [{"event_id": "", "event_type": "registered"},
            {"event_type": "registered"}]
    result = evidence_rules.duplicate_row_v1(rows, window="w")
    assert result["rows_without_id"] == 2
    assert result["duplicate_ids"] == 0


def test_the_window_is_required_so_a_number_cannot_travel_without_it():
    with pytest.raises(ValueError):
        evidence_rules.duplicate_row_v1([], window="")


def test_the_family_can_be_derived_from_the_event_id():
    """The outcome CSV carries no `family` column - it is in the id.

    Validating this rule against the live store the first time, I passed a
    `family` that did not exist and it tagged 0 of 9,914 minute-30 rows. With
    the family derived it tags 9,623, which is the audit's number exactly.
    """
    event_id = "AAPL_long_20260724_06_30_00_h1_blue_after_red"
    assert evidence_rules.family_from_event_id(event_id) == "h1_blue_after_red"
    tag = evidence_rules.h1_bar_start_v1(None, "2026-07-24 06:30:00", event_id=event_id)
    assert tag.tagged


def test_an_id_that_does_not_carry_a_family_yields_nothing():
    assert evidence_rules.family_from_event_id("short_id") == ""
    assert evidence_rules.family_from_event_id(None) == ""
    assert not evidence_rules.h1_bar_start_v1(None, "2026-07-24 06:30:00",
                                              event_id="short_id").tagged


def test_an_explicit_family_wins_over_the_id():
    tag = evidence_rules.h1_bar_start_v1(
        "regime_pause_rw", "2026-07-24 06:30:00",
        event_id="AAPL_long_20260724_06_30_00_h1_blue_after_red",
    )
    assert not tag.tagged


# ---------------------------------------------------------------------------
# milestone_stop_erased_v1 (Decision B.1, 2026-08-25)
# ---------------------------------------------------------------------------
def _recovered_final(stop_hit, source="legacy_csv_milestones"):
    import json as _json

    return {
        "event_id": "a",
        "event_type": "final",
        "stop_hit": stop_hit,
        "context_json": _json.dumps(
            {"finalization": {"basis": "last_measured_bar", "measurement_source": source}}
        ),
    }


def test_a_recovered_final_that_dropped_a_milestone_stop_is_tagged():
    tag = evidence_rules.milestone_stop_erased_v1(
        final_row=_recovered_final("False"),
        milestone_rows=[
            {"event_type": "3_bar", "stop_hit": "True"},
            {"event_type": "12_bar", "stop_hit": "False"},
        ],
    )
    assert tag.rule == evidence_rules.RULE_MILESTONE_STOP_ERASED
    assert tag.verdict == evidence_rules.VERDICT_MIXED
    assert "1 milestone row(s)" in tag.reason


def test_a_recovered_final_that_kept_its_stop_is_clean():
    tag = evidence_rules.milestone_stop_erased_v1(
        final_row=_recovered_final("True"),
        milestone_rows=[{"event_type": "3_bar", "stop_hit": "True"}],
    )
    assert tag.verdict == evidence_rules.VERDICT_SHARES


def test_no_surviving_stop_row_reads_clean_and_says_what_that_means():
    """Clean here means "no evidence of erasure", not "no erasure" - the rule
    cannot see a trade whose milestone rows were pruned."""
    tag = evidence_rules.milestone_stop_erased_v1(
        final_row=_recovered_final("False"),
        milestone_rows=[{"event_type": "12_bar", "stop_hit": "False"}],
    )
    assert tag.verdict == evidence_rules.VERDICT_SHARES
    assert "no evidence of erasure" in tag.reason


def test_a_final_measured_in_state_is_unknown_not_clean():
    """Missing data is uncertainty, never confirmation. This rule has nothing
    to say about a final that never went through milestone recovery."""
    tag = evidence_rules.milestone_stop_erased_v1(
        final_row=_recovered_final("False", source=""),
        milestone_rows=[{"event_type": "3_bar", "stop_hit": "True"}],
    )
    assert tag.verdict == evidence_rules.VERDICT_UNKNOWN


def test_an_unparseable_context_does_not_raise():
    tag = evidence_rules.milestone_stop_erased_v1(
        final_row={"stop_hit": "False", "context_json": "{not json"},
        milestone_rows=[{"event_type": "3_bar", "stop_hit": "True"}],
    )
    assert tag.verdict == evidence_rules.VERDICT_UNKNOWN


def test_the_rule_is_described_for_a_report_footer():
    text = evidence_rules.describe(evidence_rules.RULE_MILESTONE_STOP_ERASED)
    assert "milestone_stop_erased_v1" in text
    assert "2026-08-25" in text
