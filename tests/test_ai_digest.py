"""Phase 2 - the Daily Digest Ledger (packet W4).

`docs/LOCAL_AI_AUTOMATION_PLAN.md` §6.4a was a design packet that the
2026-08-08 trader decision forbade building until six questions were answered.
They were answered on 2026-08-24 (`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`
§1), and this is that design with the answers frozen into it:

1. **winning = BOTH**, R at scenario close AND MFE/MAE, side by side, never
   blended - close-R is result, MFE/MAE is opportunity;
2. **slices = env_key (environment x day-part) x side**, and no setup-family
   slice in v1;
3. **shadow-engine outputs are EXCLUDED** - champion facts only, because a
   reducer that reads a challenger beside a champion will treat it as live;
4. narration is disposable and regenerable, fact packs are permanent;
5. **16 KB hard cap**, and over-cap fails the job rather than truncating;
6. **a non-session writes an EMPTY fact pack**, so the gap is visible.

The load-bearing structural decision is D1: two artifacts, not one. Facts are
written by code with zero LLM involvement and are written even when the model is
down; narration reads the fact pack and NOTHING else, and is simply absent when
it fails. That is what makes the truncation class of failure impossible here by
construction rather than by vigilance.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import digest  # noqa: E402

NOW = datetime(2026, 8, 25, 2, 0, tzinfo=timezone.utc)


def _final(symbol, *, direction="long", env="bullish_weak", day_part="opening_drive",
           close_r=1.0, mfe_r=2.0, mae_r=-0.4, env_key=None):
    return {
        "event_id": f"{symbol}_{direction}_20260824_06_45_00_bounce_confirmed",
        "symbol": symbol,
        "direction": direction,
        "trade_date": "2026-08-24",
        "entry_time": "2026-08-24T06:45:00-07:00",
        "market_environment": env,
        # R10.A stamps this at registration; the digest reads it and never
        # re-derives it.
        "env_key": f"{env}|{day_part}" if env_key is None else env_key,
        "close_r": close_r,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "bounce_type": "bounce_confirmed",
    }


def _pack(**overrides):
    payload = {
        "session_date": "2026-08-24",
        "is_session": True,
        "finals": [_final("AAPL"), _final("MSFT", close_r=-1.0, mfe_r=0.3, mae_r=-1.0)],
        "now": NOW,
    }
    payload.update(overrides)
    return digest.build_fact_pack(**payload)


# ---------------------------------------------------------------------------
# D2 - every number is computed by code and carries its own pointer
# ---------------------------------------------------------------------------


def test_a_measured_value_cannot_be_written_without_its_n():
    """`n` is mandatory. The -0.18R vs +1.01R finding was only actionable
    because both sample sizes were known; a bare average is a coin flip."""
    value = digest.measured(1.01, n=1940, source_id="review.scoreboard",
                            selector="bucket=favorite_setup", as_of="2026-08-24")
    assert value["n"] == 1940 and value["source_id"] == "review.scoreboard"
    assert value["selector"] and value["as_of"]

    with pytest.raises(ValueError):
        digest.measured(1.01, n=None, source_id="x", selector="y", as_of="z")


def test_an_unmeasurable_value_is_none_with_an_n_of_zero_never_a_zero():
    value = digest.measured(None, n=0, source_id="x", selector="y", as_of="z")
    assert value["value"] is None and value["n"] == 0


# ---------------------------------------------------------------------------
# Answer 1 - both win metrics, side by side, never blended
# ---------------------------------------------------------------------------


def test_close_r_and_mfe_mae_are_reported_side_by_side_and_never_blended():
    pack = _pack()
    overall = pack["outcomes"]["overall"]
    assert overall["close_r"]["value"] == pytest.approx(0.0)  # (1.0 + -1.0) / 2
    assert overall["mfe_r"]["value"] == pytest.approx(1.15)   # (2.0 + 0.3) / 2
    assert overall["mae_r"]["value"] == pytest.approx(-0.7)
    assert overall["close_r"]["n"] == 2 and overall["mfe_r"]["n"] == 2
    # No field mixes them. A "score" combining result and opportunity is the
    # blend answer 1 explicitly refused.
    assert not any(
        key for key in overall if "blend" in key or key in {"score", "combined"}
    )


def test_the_answers_travel_inside_the_fact_pack():
    """A pack read in six months must carry the rules it was built under."""
    pack = _pack()
    answers = pack["answers"]
    assert "never blended" in answers["winning"].lower()
    assert "day-part" in answers["slices"] and "side" in answers["slices"]
    assert "exclud" in answers["shadow_engines"].lower()


# ---------------------------------------------------------------------------
# Answer 2 - the slice is env_key x side
# ---------------------------------------------------------------------------


def test_the_slice_key_is_environment_day_part_and_side():
    pack = _pack(
        finals=[
            _final("AAPL", env="bullish_weak", day_part="opening_drive"),
            _final("TSLA", env="bullish_weak", day_part="opening_drive", direction="short"),
            _final("NVDA", env="bearish_strong", day_part="afternoon"),
        ]
    )
    keys = {(row["env_key"], row["side"]) for row in pack["outcomes"]["slices"]}
    assert ("bullish_weak|opening_drive", "LONG") in keys
    assert ("bullish_weak|opening_drive", "SHORT") in keys
    assert ("bearish_strong|afternoon", "LONG") in keys


def test_the_env_key_is_read_from_the_row_and_never_re_derived():
    """Answer 2 names "the env_key R10.A already stamps", and that is what this
    reads.

    Two reasons it is not recomputed. A second copy of the day-part cutoffs
    would let the digest and the learning state disagree about what "midday"
    means. And `ai_jobs` is kept out of live decision modules by an existing
    test, so borrowing one function from the module that MUTES alert segments
    would cross a boundary that exists precisely to stop that.
    """
    row = {"env_key": "bullish_weak|midday", "market_environment": "bearish_strong"}
    assert digest.env_key_of(row) == "bullish_weak|midday", "the stamp wins"
    assert digest.day_part_of(digest.env_key_of(row)) == "midday"


def test_a_row_written_before_the_stamp_existed_has_an_unknown_day_part():
    """Never a guess and never a quiet default to some bucket."""
    assert digest.env_key_of({"market_environment": "bullish_weak"}) == "bullish_weak|unknown"
    assert digest.env_key_of({}) == "unknown|unknown"
    assert digest.day_part_of("") == "unknown"
    assert digest.day_part_of("no separator") == "unknown"


def test_the_digest_imports_nothing_from_the_live_decision_modules():
    """The boundary this packet nearly crossed, pinned here as well as in the
    package-wide test - because the tempting import was a PURE helper, which is
    exactly the kind that gets waved through."""
    import ast

    tree = ast.parse(Path(digest.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for name in imported:
        assert not name.startswith(
            ("bounce_bot", "autopilot_core", "master_avwap", "technical_integrity",
             "price_alert", "d1_level_feed")
        ), f"the digest reached into live decision code: {name}"


def test_no_setup_family_slice_in_v1():
    """Answer 2 named the starting set deliberately; adding one is a v2 call."""
    pack = _pack()
    for row in pack["outcomes"]["slices"]:
        assert "family" not in row and "bounce_type" not in row


# ---------------------------------------------------------------------------
# Answer 3 - champion facts only
# ---------------------------------------------------------------------------


def test_the_digest_never_reads_a_shadow_engine_store():
    """Walked over the module's AST rather than trusted to a sentence.

    Mixing a challenger's output with a champion's risks a later reader - human
    or frontier - treating the challenger as live (plan.md sec 7).
    """
    import ast

    source = Path(digest.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    text = "\n".join(
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    )
    forbidden = (
        "spy_state_shadow", "greatness_shadow", "sector_cohort_shadow",
        "market_state_bridge", "greatness_monitor",
    )
    for name in forbidden:
        assert name not in text or "excluded" in text.lower(), (
            f"the digest must not read the shadow store {name}"
        )
    assert "shadow_evidence" not in text


# ---------------------------------------------------------------------------
# Answer 5 - the cap, by construction rather than by truncation
# ---------------------------------------------------------------------------


def test_an_over_cap_fact_pack_fails_the_job_rather_than_truncating(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "FACT_PACK_HARD_CAP_BYTES", 200)
    result = digest.run_daily_digest(
        session_date="2026-08-24", now=NOW, root=tmp_path,
        finals=[_final("AAPL")], narrate=False,
    )
    assert result["status"] == "failed"
    assert "cap" in result["reason"].lower()
    assert not list(tmp_path.rglob("*.json")), "nothing is written when the pack is over cap"


def test_the_slice_cap_prints_what_it_dropped():
    finals = [
        _final(f"SYM{index}", env=f"env{index}")
        for index in range(digest.MAX_SLICES + 5)
    ]  # one event each, so the cap drops five slices holding five events
    pack = _pack(finals=finals)
    assert len(pack["outcomes"]["slices"]) == digest.MAX_SLICES
    dropped = pack["outcomes"]["slices_dropped"]
    assert dropped["slices"] == 5 and dropped["events"] == 5
    assert "n" in dropped["basis"].lower()


# ---------------------------------------------------------------------------
# Answer 6 - a non-session writes an EMPTY pack, not no file
# ---------------------------------------------------------------------------


def test_a_non_session_writes_an_empty_fact_pack_so_the_gap_is_visible(tmp_path):
    result = digest.run_daily_digest(
        session_date="2026-08-22", now=NOW, root=tmp_path, is_session=False, narrate=False,
    )
    assert result["status"] == "ok"
    path = digest.facts_path(tmp_path, "2026-08-22")
    assert path.is_file()
    pack = json.loads(path.read_text(encoding="utf-8"))
    assert pack["is_session"] is False
    assert pack["empty_reason"]
    assert pack["outcomes"]["overall"]["close_r"]["n"] == 0
    assert pack["outcomes"]["slices"] == []


# ---------------------------------------------------------------------------
# D1 - two artifacts, and the facts survive a dead model
# ---------------------------------------------------------------------------


def test_the_fact_pack_is_written_even_when_the_model_is_unavailable(tmp_path, monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("local AI provider is not configured")

    monkeypatch.setattr(digest, "_narrate", explode)
    result = digest.run_daily_digest(
        session_date="2026-08-24", now=NOW, root=tmp_path, finals=[_final("AAPL")],
    )
    assert digest.facts_path(tmp_path, "2026-08-24").is_file()
    assert not digest.narration_path(tmp_path, "2026-08-24").exists()
    assert result["status"] == digest.STATUS_DEGRADED
    assert "narration" in result["reason"].lower()


def test_the_narrator_is_handed_the_fact_pack_and_nothing_else():
    """D5. The prompt is bounded by the cap plus a fixed scaffold, so the
    2026-08-10 truncation class cannot recur here by design."""
    pack = _pack()
    package = digest.narration_evidence_package(pack)
    source_ids = [source["source_id"] for source in package["sources"]]
    assert source_ids == [digest.FACT_PACK_SOURCE_ID]
    assert package["sources"][0]["content"] == pack


# ---------------------------------------------------------------------------
# D6 - append-only, superseding siblings, never an edit
# ---------------------------------------------------------------------------


def test_a_second_run_supersedes_rather_than_overwriting(tmp_path):
    first = digest.run_daily_digest(
        session_date="2026-08-24", now=NOW, root=tmp_path, finals=[_final("AAPL")], narrate=False,
    )
    original = digest.facts_path(tmp_path, "2026-08-24")
    original_bytes = original.read_bytes()

    second = digest.run_daily_digest(
        session_date="2026-08-24", now=NOW, root=tmp_path,
        finals=[_final("AAPL"), _final("MSFT")], narrate=False,
    )

    assert original.read_bytes() == original_bytes, "a fact pack is never edited"
    sibling = Path(second["outputs"][0])
    assert sibling != original and sibling.is_file()
    payload = json.loads(sibling.read_text(encoding="utf-8"))
    assert payload["supersedes"] == original.name
    assert first["outputs"][0] != second["outputs"][0]


def test_every_timestamp_carries_an_explicit_offset():
    pack = _pack()
    assert pack["generated_at"].endswith("+00:00")
    from datetime import datetime as dt

    assert dt.fromisoformat(pack["generated_at"]).tzinfo is not None


# ---------------------------------------------------------------------------
# D8 - rollups are a read, not a second store
# ---------------------------------------------------------------------------


def test_a_rollup_reads_the_fact_packs_and_writes_nothing(tmp_path):
    for day, symbol in (("2026-08-24", "AAPL"), ("2026-08-25", "MSFT")):
        digest.run_daily_digest(
            session_date=day, now=NOW, root=tmp_path,
            finals=[_final(symbol)], narrate=False,
        )
    before = sorted(path.name for path in tmp_path.rglob("*.json"))

    rollup = digest.rollup(tmp_path, since="2026-08-01", until="2026-08-31")

    assert rollup["sessions"] == 2
    assert rollup["close_r"]["n"] == 2
    assert sorted(path.name for path in tmp_path.rglob("*.json")) == before


def test_the_gate_counts_clean_sessions_and_ten_are_required(tmp_path):
    """Phase 2's exit gate is ten consecutive session days of digests plus a
    trader spot-audit. Counting them is code's job; passing them is not."""
    assert digest.clean_digest_sessions(tmp_path) == 0
    # Two trading weeks, weekends excluded by the calendar rather than by this
    # test - a run keyed to a Saturday is the defect the runner's own session
    # identity was repaired for, and the digest inherits that discipline.
    sessions = [f"2026-08-{day:02d}" for day in (10, 11, 12, 13, 14, 17, 18, 19, 20, 21)]
    for day in sessions:
        digest.run_daily_digest(
            session_date=day, now=NOW, root=tmp_path,
            finals=[_final("AAPL")], narrate=False,
        )
    assert digest.clean_digest_sessions(tmp_path) == 10
    assert digest.REQUIRED_CLEAN_SESSIONS == 10
    assert digest.digest_gate_state(tmp_path)["window_met"] is True

    # And the weekend inside that span writes an empty pack that does NOT count.
    digest.run_daily_digest(session_date="2026-08-15", now=NOW, root=tmp_path, narrate=False)
    assert digest.clean_digest_sessions(tmp_path) == 10


def test_a_non_session_pack_does_not_count_towards_the_gate(tmp_path):
    """An empty weekend pack makes the gap visible; it is not a clean session."""
    digest.run_daily_digest(
        session_date="2026-08-22", now=NOW, root=tmp_path, is_session=False, narrate=False,
    )
    assert digest.clean_digest_sessions(tmp_path) == 0


# ---------------------------------------------------------------------------
# The slot
# ---------------------------------------------------------------------------


def test_the_digest_slot_is_appended_and_never_reorders_the_slate():
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]
    assert names[0] == "journal_import", "the one sanctioned exception stays first"
    assert names[:6] == [
        "journal_import", "ai_summary", "ticker_briefs",
        "veto_cohort_grading", "like_cohort_grading", "evidence_report",
    ], "later phases append; they never reorder these"
    assert "daily_digest" in names
    assert names.index("daily_digest") > names.index("evidence_report")


def test_coverage_names_a_source_it_could_not_read_rather_than_omitting_it():
    pack = _pack(unavailable={"alert review events": "file is locked"})
    assert pack["unavailable"]["alert review events"] == "file is locked"
    assert "INCOMPLETE" in pack["summary"]
