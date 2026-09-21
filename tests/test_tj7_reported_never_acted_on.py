r"""TJ-7's hard invariant - a mood is REPORTED and never acted on. STRUCTURAL.

`plan.md` sec 5 and `.claude/packets/TJ-LOOP0_COMMON.md`: *"Nothing here may
reach a detector, score, alert, watchlist, Focus, the review queue or
`review_policy.json`. Every number is REPORTED, never acted on."*

A mood is the softest evidence the desk will ever hold and the easiest to
misuse: it is the trader's own word about themselves, it arrives AFTER the
session more often than not, and there is no version of "the desk traded
differently because you said you felt rushed" that is honest. So the fence is
STRUCTURAL rather than behavioural, the way TJ-12's and TJ-6's fences are: the
source of every module that could act is read, and none of them may so much as
mention the field.

Three separate questions, three tests:

1. no module that DECIDES anything reads a mood;
2. only the trader's own two surfaces (and the Mentor answer they click) WRITE
   one - no job, no nightly slot, no importer, no grader;
3. the mood code path cannot reach an R statistic, an outcome or a verdict, so
   nothing can select, rank or pre-fill a mood by result.

A fourth test states what this packet does NOT add: no nightly slot. TJ-7 is
fields, a strip, a pack section and a context field.

RED FOR: `scripts/trader_state_tags.py` does not exist yet (the import-fence
test reads its source). The three scans over shipped modules are STATED GREEN
GUARDS - they pass today and exist so the builder cannot make them stop.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for _extra in (SCRIPTS_DIR, ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

#: Everything that decides something: the two champion detectors, the pure
#: indicators, the Focus gates, the watchlist writers, the review queue and the
#: policy the AI is allowed to rank with.
DECIDERS = (
    "bounce_bot.py",
    "bounce_bot_lib/legacy.py",
    "master_avwap.py",
    "master_avwap_lib/legacy.py",
    "focus_adoption_gate.py",
    "regime_pause_focus.py",
    "armed_alert_expiry.py",
    "candidate_registry.py",
    "daytrade_watchlist_reset.py",
    "market_state.py",
    "greatness_monitor.py",
    "review_learning.py",
    "ui/panels/alert_center_panel.py",
)

#: The words a mood is stored and read under. A decider that names ANY of them
#: has read the trader's mood.
MOOD_WORDS = ("mood_of", "mood_at", "trader_state_tags", "state_tags", '"mood"', "'mood'")

#: Who may hand a mood to the journal's writer. The trader's two surfaces, the
#: click they answer through, and the writer itself.
ALLOWED_MOOD_WRITERS = {
    "market_journal.py",
    "mentor_questions.py",
    "ui/services/market_journal_service.py",
    "ui/widgets/trade_mentor_card.py",
    "ui/panels/day_review_panel.py",
    "ui/panels/market_journal_panel.py",
}

#: Anything that could put a RESULT in front of a mood.
OUTCOME_MODULES = {
    "setup_scoreboard", "outcome_semantics", "walkaway_day", "expected_r",
    "swing_headline", "evidence_stats", "market_read_grades", "real_miss",
    "evidence_contrast", "working_lately", "review_learning", "pick_feedback",
}


def _sources(names) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        path = SCRIPTS_DIR / name
        assert path.is_file(), f"{name} is not where this test thinks it is"
        out[name] = path.read_text(encoding="utf-8", errors="replace")
    return out


def _indicator_files() -> list[str]:
    return [
        str(path.relative_to(SCRIPTS_DIR)).replace("\\", "/")
        for path in sorted((SCRIPTS_DIR / "indicators").glob("*.py"))
    ]


def test_no_detector_score_alert_watchlist_focus_or_review_module_reads_a_mood():
    """GREEN TODAY and it must stay green."""
    offenders: list[str] = []
    for name, source in _sources(list(DECIDERS) + _indicator_files()).items():
        for word in MOOD_WORDS:
            if word in source:
                offenders.append(f"{name}: {word}")
    assert not offenders, offenders


def test_nothing_writes_a_mood_into_review_policy():
    """`review_policy.json` ranks and annotates; it has no suppression field and
    it gains no mood field either."""
    # LEAD AMENDMENT 2026-09-20 (TJ-7 integration, on the reviewer's advice):
    # this used to flag any file whose SOURCE held both `review_policy` and a
    # mood word. Three modules the packet REQUIRED TJ-7 to touch state this very
    # invariant in prose ("...never reaches `review_policy.json`"), so the guard
    # was unsatisfiable by its own packet and the builder had to reword shipped
    # docstrings to pass it. The rule it protects is about CONTACT, not words:
    # a module that imports, loads, drafts or saves the policy may not also
    # handle a mood. The reviewer proved the same thing by tracing importers.
    policy_contact = (
        "import review_policy",
        "from review_policy",
        "REVIEW_POLICY_FILE",
        "save_review_policy",
        "load_review_policy",
        "draft_policy_from_state",
    )
    hits: list[str] = []
    touching: list[str] = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if not any(token in source for token in policy_contact):
            continue
        touching.append(str(path.relative_to(SCRIPTS_DIR)))
        if "mood" in source or "state_tags" in source:
            hits.append(str(path.relative_to(SCRIPTS_DIR)))
    # The narrowed scan must still SEE the policy's own modules, or it guards nothing.
    assert touching, "no module touches the review policy - the contact tokens went stale"
    assert not hits, hits


def test_only_the_traders_own_surfaces_hand_a_mood_to_the_journal():
    """A mood is the trader's own click or nothing. No job, no importer, no
    grader and no nightly slot may pass one to a journal writer."""
    offenders: list[str] = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        name = str(path.relative_to(SCRIPTS_DIR)).replace("\\", "/")
        if name in ALLOWED_MOOD_WRITERS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - not our file to fix
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            called_name = (
                called.attr if isinstance(called, ast.Attribute)
                else called.id if isinstance(called, ast.Name)
                else ""
            )
            if called_name not in ("build_entry", "write_entry", "build_mood"):
                continue
            for keyword in node.keywords:
                if keyword.arg in ("mood", "state_tags", "process"):
                    offenders.append(f"{name}: {called_name}({keyword.arg}=...)")
    assert not offenders, offenders


def test_the_mood_vocabulary_cannot_reach_an_outcome():
    """No R statistic, verdict or outcome may select, rank or pre-fill a mood,
    so the module that owns the vocabulary cannot import one."""
    source = (SCRIPTS_DIR / "trader_state_tags.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & OUTCOME_MODULES, sorted(imported & OUTCOME_MODULES)


def test_this_packet_adds_no_nightly_slot():
    """TJ-7 is fields, a strip, a pack section and a context field. A slot here
    would move NINE order pins and is not what the packet asks for."""
    from ai_jobs import runner

    names = set()
    for kind in runner.NIGHT_KINDS:
        names.update(str(getattr(slot, "name", "")) for slot in runner.slots_for(kind))
    assert names, "no slate could be built"
    assert not {name for name in names if "mood" in name or "state_tag" in name}, names
