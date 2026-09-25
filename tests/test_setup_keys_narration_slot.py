"""P1-4 / 4d - the Saturday setup-keys narration slot and the research pack copy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for entry in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ai_jobs import ledger, runner  # noqa: E402
from ai_jobs import setup_keys_narration as narration  # noqa: E402

SLOT = "setup_keys_narration"
FAMILY_ID = "swing|h5|avwap_band_bounce SHORT"
NONE_ID = "swing|h5|post_earnings_candle_break SHORT"


def _report(generated_at="2026-09-26T12:00:00+00:00"):
    key = {"rank": 1, "depth": 1, "label": "ma_support=sma100_support", "lift_pp": 21.0,
           "selection": {"n": 240, "sessions": 40, "win_rate": 0.8, "wilson_lb": 0.74},
           "holdout": {"n": 90, "sessions": 20, "win_rate": 0.78, "wilson_lb": 0.68}}
    return {
        "schema": "setup_permutation_report_v1", "generated_at": generated_at,
        "populations": {"swing": {"horizons": {"5": {"families": {
            "avwap_band_bounce SHORT": {"verdict": "key_found", "keys": [key],
                                        "baseline": {"n": 800, "sessions": 40, "win_rate": 0.59},
                                        "holdout_baseline": {"n": 300, "sessions": 20, "win_rate": 0.57}},
            "post_earnings_candle_break SHORT": {"verdict": "no_key_found", "keys": [],
                                                 "baseline": {"n": 120, "sessions": 30, "win_rate": 0.5},
                                                 "holdout_baseline": {"n": 40, "sessions": 20, "win_rate": 0.5}},
        }}}}},
    }


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    import project_paths

    report = tmp_path / "permutation_report.json"
    out = tmp_path / "setup_keys_narration.json"
    monkeypatch.setattr(project_paths, "SETUP_PERMUTATION_REPORT_FILE", report)
    monkeypatch.setattr(project_paths, "SETUP_KEYS_NARRATION_FILE", out)
    return report, out


def _answer(sentences):
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return {"model": "test-model", "summary": {"sentences": sentences}}

    return request, calls


GOOD = [
    {"family_id": FAMILY_ID, "text": "SMA100 support held up on the last 20 sessions.",
     "cites": [f"{FAMILY_ID}|key1", f"{FAMILY_ID}|holdout_baseline"]},
    {"family_id": NONE_ID, "text": "No key was found for this family.", "cites": [f"{NONE_ID}|verdict"]},
]


# --- where it sits


def test_the_slot_is_stage_three_saturday_only_and_before_the_ideas():
    names = [slot.name for slot in runner.default_slots()]
    assert names.index(SLOT) == names.index("setup_research") + 1
    assert names[-1] == "improvement_ideas"
    assert SLOT in runner.WEEKEND_ONLY_SLOTS
    by_name = {slot.name: slot for slot in runner.default_slots()}
    assert by_name[SLOT].uses_model is True
    assert by_name[SLOT].max_attempts == 2
    weeknight = [slot.name for slot in runner.slots_for(runner.NIGHT_WEEKNIGHT)]
    saturday = [slot.name for slot in runner.slots_for(runner.NIGHT_SATURDAY)]
    assert SLOT not in weeknight
    assert SLOT in saturday


def test_the_rules_line_names_the_slot():
    text = (ROOT_DIR / "docs" / "RULES.md").read_text(encoding="utf-8")
    assert "setup_keys_narration" in text


# --- the slot


def test_no_report_skips_before_any_model(paths):
    request, calls = _answer(GOOD)
    result = narration.run_setup_keys_narration(request=request)
    assert result["status"] == ledger.STATUS_SKIPPED
    assert calls == []


def test_facts_only_and_a_checked_answer_is_stored(paths):
    report, out = paths
    report.write_text(json.dumps(_report()), encoding="utf-8")
    request, calls = _answer(GOOD)
    result = narration.run_setup_keys_narration(request=request, session_date="2026-09-25")
    assert result["status"] == ledger.STATUS_OK, result
    evidence = calls[0]["evidence"]
    assert set(evidence) <= set(narration.EVIDENCE_KEYS)
    assert evidence["allowed_family_ids"][0] == FAMILY_ID  # keys found first
    stored = json.loads(out.read_text(encoding="utf-8"))
    by_id = {family["family_id"]: family for family in stored["families"]}
    assert by_id[FAMILY_ID]["sentences"][0]["cites"] == [f"{FAMILY_ID}|key1", f"{FAMILY_ID}|holdout_baseline"]
    assert stored["model"] == "test-model"
    # Unchanged report: no second ask.
    again, again_calls = _answer(GOOD)
    assert narration.run_setup_keys_narration(request=again)["status"] == ledger.STATUS_OK
    assert again_calls == []


@pytest.mark.parametrize(
    "bad",
    [
        [{"family_id": FAMILY_ID, "text": "Made up.", "cites": ["swing|h5|other|key1"]}],
        [{"family_id": FAMILY_ID, "text": "Wrong family fact.", "cites": [f"{NONE_ID}|verdict"]}],
        [{"family_id": "swing|h9|ghost LONG", "text": "Ghost.", "cites": ["x"]}],
        [{"family_id": FAMILY_ID, "text": "No cite.", "cites": []}],
        [{"family_id": FAMILY_ID, "text": f"s{n}", "cites": [f"{FAMILY_ID}|verdict"]} for n in range(4)],
    ],
)
def test_a_bad_answer_is_rejected_whole_and_the_last_good_file_stands(paths, bad):
    report, out = paths
    report.write_text(json.dumps(_report()), encoding="utf-8")
    good, _ = _answer(GOOD)
    assert narration.run_setup_keys_narration(request=good)["status"] == ledger.STATUS_OK
    before = out.read_bytes()
    newer = _report("2026-10-03T12:00:00+00:00")
    newer["populations"]["swing"]["horizons"]["5"]["families"]["avwap_band_bounce SHORT"]["baseline"]["n"] = 900
    report.write_text(json.dumps(newer), encoding="utf-8")  # new facts: the slot must ask again
    request, _calls = _answer(bad)
    result = narration.run_setup_keys_narration(request=request)
    assert result["status"] == ledger.STATUS_FAILED
    assert "rejected" in result["reason"]
    assert out.read_bytes() == before


def test_a_dead_model_is_degraded_and_writes_nothing(paths):
    report, out = paths
    report.write_text(json.dumps(_report()), encoding="utf-8")

    def request(**kwargs):
        raise ConnectionError("ollama down")

    result = narration.run_setup_keys_narration(request=request)
    assert result["status"] == ledger.STATUS_DEGRADED
    assert not out.exists()


# --- the research pack


def test_the_research_pack_carries_the_report(tmp_path):
    from scripts import research_pack as rp

    report = tmp_path / "permutation_report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    sources = rp.Sources(lake_root=None, journal_db=None, bounce_outcomes_csv=None, bounces_csv=None,
                         extra_files={}, ai_store_root=None, protected=(), setup_keys_report=report)
    manifest = rp.export_pack(sources, tmp_path / "pack")
    assert manifest["tables"]["setup_permutation_report"]["status"] == "ok"
    assert json.loads((tmp_path / "pack" / "setup_permutation_report.json").read_text(encoding="utf-8")) == _report()
    empty = rp.Sources(lake_root=None, journal_db=None, bounce_outcomes_csv=None, bounces_csv=None,
                       extra_files={}, ai_store_root=None, protected=())
    assert rp.export_pack(empty, tmp_path / "pack2")["tables"]["setup_permutation_report"]["status"] == "missing"
