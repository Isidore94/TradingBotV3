"""P8-P4 part 1: three night AI slots that failed or read as failure every night.

1. `day_review_narration` timed out at 540 s on 7 of 11 nights with a ~40 KB
   pack: the pack is now trimmed to a time budget and the ledger names its size.
2. `econ_brief` rejected "1:00 pm ET" / "1 p.m." against a 13:00 ET event.
3. `theta_pick_grading`: picks that were never quoted can never be graded.

No model is called; every transport is faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tj4_support as day_fx  # noqa: E402
from test_ai_night_fixes_2026_09_23 import _live_sized_pack  # noqa: E402


# ---------------------------------------------------------------------------
# 1. day story: trimmed to the time budget, deterministically, still verifiable
# ---------------------------------------------------------------------------
def _bulky_pack():
    """The live-sized fixture with its uncited sections grown past the budget."""
    pack = _live_sized_pack()
    pack["report_card"]["lines"] = [
        {"source_id": f"report_card:k{i}", "key": f"k{i}", "text": "a line " * 60}
        for i in range(30)
    ]
    pack["congruence"] = [
        {"source_id": f"congruence:c{i}", "text": "agreed " * 80, "source_ids": []}
        for i in range(20)
    ]
    return pack


def test_a_day_pack_over_the_budget_is_trimmed_deterministically(tmp_path):
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    pack = _bulky_pack()
    whole = narration._model_pack(pack)
    assert len(json.dumps(whole, default=str)) > narration.MAX_DAY_EVIDENCE_CHARS

    first = narration._day_evidence(pack, tmp_path)
    second = narration._day_evidence(pack, tmp_path)
    text = json.dumps(first, sort_keys=True, default=str)

    assert text == json.dumps(second, sort_keys=True, default=str)
    assert len(text) <= narration.MAX_DAY_EVIDENCE_CHARS
    trimmed = first["pack_trimmed"]
    assert trimmed == list(narration.DAY_TRIM_ORDER[: len(trimmed)])
    assert trimmed[0] == "report_card"
    assert first["pack"]["report_card"] != whole["report_card"]
    # Everything the verifier needs is still in front of the model.
    for source_id in day_review_pack.allowed_source_ids(pack):
        assert json.dumps(source_id) in text
    assert first["pack"]["reads"] == pack["reads"]
    assert first["pack"]["trader_said"] == pack["trader_said"]


def _v2_answer(evidence: dict) -> dict:
    return {
        "headline": "A day.",
        "what_happened": "It moved.",
        "what_you_thought": "Up.",
        "read_explanations": {key: "Explained." for key in evidence["read_explanations"]},
        "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
        "process": "Fine.",
        "sources": evidence["allowed_source_ids"][:2],
    }


def test_a_trimmed_day_pack_still_verifies_and_the_ledger_names_what_was_sent(tmp_path):
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    pack = _bulky_pack()
    day_review_pack.write_pack(pack, root=root)
    seen: list[dict] = []

    def request(**kwargs):
        seen.append(kwargs["evidence"])
        return {"summary": _v2_answer(kwargs["evidence"]), "model": "fake-12b"}

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=request, only_this_session=True,
    )

    assert outcome["status"] == "ok", outcome
    assert seen and seen[0]["pack_trimmed"][0] == "report_card"
    assert "grounded day story written" in outcome["reason"]
    assert "bytes (~" in outcome["reason"] and "tokens est.)" in outcome["reason"]
    assert "dropped report_card" in outcome["reason"]
    assert narration.narration_path(day_fx.SESSION, root=root).exists()


def test_a_failed_day_call_still_names_the_size_it_sent(tmp_path):
    import day_review_pack
    from ai_jobs import day_review_narration as narration

    root = tmp_path / "day_review"
    day_review_pack.write_pack(day_fx.build(), root=root)

    def request(**kwargs):
        raise RuntimeError("Read timed out. (read timeout=540)")

    outcome = narration.run_day_review_narration(
        session_date=day_fx.SESSION, now=day_fx.OVERNIGHT, root=root,
        request=request, only_this_session=True,
    )

    assert outcome["status"] == "degraded_no_narrative"
    assert "Read timed out" in outcome["reason"]
    assert "bytes (~" in outcome["reason"]
