r"""TJ-6 - an idea is a SUGGESTION the trader reads, and nothing else. RED.

The hard invariant of this whole program (`plan.md` sec 5, §12.3 ground rules,
decision 0021 answer 23, and the packet's own words): nothing TJ-6 writes may be
read by a detector, a score, an alert, a watchlist, Focus, the review queue or
`review_policy.json`; a kept idea never becomes a rule by itself; and
*"the AI never writes `WISHLIST.md`"* - the trader pastes.

Proved STRUCTURALLY, the way TJ-12's tests prove a reader exists: by walking the
imports of every module under `scripts/` and by hashing the two files an
over-helpful job would edit.

VERIFIED ON THIS BRANCH (1b9d77e0): nothing anywhere imports
`ai_jobs.improvement_ideas` and `AI_IDEAS_FILE` appears in no file, so the first
test fails on the seams that do not exist yet.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj6_support as fx  # noqa: E402

SCRIPTS = ROOT_DIR / "scripts"

#: The names that mean "this module touches the ideas".
IDEA_NAMES = ("improvement_ideas", "AI_IDEAS_FILE", "AI_IDEAS_STATE_FILE")

#: Every module allowed to know an idea exists: the store and slot itself, the
#: path owner, the two page workers, the two pages and the widget between them,
#: and the runner that registers the slot. Everything else on the desk must be
#: unable to see one.
ALLOWED = {
    "ai_jobs/improvement_ideas.py",
    "ai_jobs/runner.py",
    "project_paths.py",
    "ui/services/day_review_service.py",
    "ui/services/weekend_prep_service.py",
    "ui/panels/day_review_panel.py",
    "ui/panels/weekend_prep_panel.py",
    "ui/widgets/ideas_card.py",
}

#: The seams that MUST exist once TJ-6 lands, so this file cannot pass by the
#: feature simply not being there.
REQUIRED = {
    "ai_jobs/improvement_ideas.py",
    "ai_jobs/runner.py",
    "project_paths.py",
    "ui/services/day_review_service.py",
    "ui/services/weekend_prep_service.py",
}

#: What a job may never reach. Detector, scoring and alert files are ask-first
#: for a reason; an ideas module that imported one is one refactor away from an
#: idea changing a signal.
FORBIDDEN_IMPORTS = (
    "master_avwap",
    "bounce_bot",
    "legacy",
    "indicators",
    "focus_pick_store",
    "review_learning",
    "regime_pause_focus",
    "candidate_registry",
)


def _mentions(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(name in text for name in IDEA_NAMES)


def _touching_modules() -> set[str]:
    out: set[str] = set()
    for path in SCRIPTS.rglob("*.py"):
        if _mentions(path):
            out.add(path.relative_to(SCRIPTS).as_posix())
    return out


def test_only_the_two_pages_and_their_workers_can_see_an_idea():
    """The whole desk, scanned. An idea reaches two pages and stops there."""
    touching = _touching_modules()
    assert REQUIRED <= touching, sorted(REQUIRED - touching)
    assert touching <= ALLOWED, sorted(touching - ALLOWED)


def test_the_ideas_module_imports_no_detector_scorer_or_alert():
    """It reads packs, narrations, totals and two measurables. Nothing else."""
    path = SCRIPTS / "ai_jobs" / "improvement_ideas.py"
    assert path.exists(), "there is no ideas module yet"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [
        name
        for name in imported
        for forbidden in FORBIDDEN_IMPORTS
        if forbidden in name
    ]
    assert offenders == [], offenders
    # A nightly job that imported Qt would load the GUI stack in a child
    # process at below-normal priority, at 23:00, for nothing.
    assert [name for name in imported if name.split(".")[0] == "ui"] == []


def test_a_night_of_ideas_writes_only_the_ideas_file(tmp_path, monkeypatch):
    """Its `outputs` name one file, and that file is the ideas store.

    A slot whose outputs reached into `DAY_REVIEW_DIR` would be rewriting the
    packs it was asked to read.
    """
    from ai_jobs import improvement_ideas

    night = fx.install_stores(monkeypatch, tmp_path)
    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    allowed = list(inputs["allowed_source_ids"])
    measurable = str(improvement_ideas.MEASURABLES[0].name)
    before = sorted(path.name for path in night["root"].rglob("*") if path.is_file())

    outcome = improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION,
        now=fx.WEEKNIGHT,
        root=night["root"],
        request=fx.fake_request(
            fx.reply(
                [
                    fx.idea_payload(
                        "Wait for the second test.",
                        measurable=measurable,
                        evidence=allowed[:1],
                    )
                ]
            )
        ),
    )

    outputs = [Path(item) for item in outcome.get("outputs") or ()]
    assert outputs == [night["ideas"]], outcome.get("outputs")
    after = sorted(path.name for path in night["root"].rglob("*") if path.is_file())
    assert after == before, "the ideas slot wrote into the Day Review store"


def test_a_night_of_ideas_never_edits_wishlist_or_the_plan(tmp_path, monkeypatch):
    """*"Kept `program` ideas are listed 'For WISHLIST - copy'"* - listed, for
    the TRADER to paste. Both files are hashed before and after a full run."""
    from ai_jobs import improvement_ideas

    watched = {
        name: hashlib.sha256((ROOT_DIR / name).read_bytes()).hexdigest()
        for name in ("WISHLIST.md", "plan.md")
        if (ROOT_DIR / name).exists()
    }
    assert watched, "neither file is in this checkout"

    night = fx.install_stores(monkeypatch, tmp_path)
    inputs = improvement_ideas.build_ideas_inputs(fx.SESSION, root=night["root"])
    allowed = list(inputs["allowed_source_ids"])
    improvement_ideas.run_improvement_ideas(
        session_date=fx.SESSION,
        now=fx.WEEKNIGHT,
        root=night["root"],
        request=fx.fake_request(
            fx.reply(
                [
                    fx.idea_payload(
                        "Put the walk-away table beside the trades.",
                        kind="program",
                        measurable="",
                        evidence=allowed[:1],
                    )
                ]
            )
        ),
    )

    for name, digest in watched.items():
        assert hashlib.sha256((ROOT_DIR / name).read_bytes()).hexdigest() == digest, name


def test_keeping_an_idea_changes_no_policy_and_no_watchlist(tmp_path, monkeypatch):
    """A kept idea never becomes a rule by itself.

    The one file the night is allowed to draft is `review_policy_draft.json`,
    and that belongs to another slot; a Keep must leave every ranking, policy
    and list on the desk exactly where it was.
    """
    import project_paths
    from ai_jobs import improvement_ideas

    night = fx.install_stores(monkeypatch, tmp_path, write_the_week=False)
    row = fx.stored_idea_row(
        "Wait for the second test.",
        session=fx.SESSION,
        measurable=str(improvement_ideas.MEASURABLES[0].name),
    )
    fx.write_ideas(night["ideas"], [row])

    watched = [
        getattr(project_paths, name, None)
        for name in ("REVIEW_POLICY_FILE", "LONGS_FILE", "SHORTS_FILE", "MASTER_AVWAP_FOCUS_FILE")
    ]
    existing = {
        Path(path): Path(path).read_bytes()
        for path in watched
        if path is not None and Path(path).exists()
    }
    missing = [Path(path) for path in watched if path is not None and not Path(path).exists()]

    improvement_ideas.keep_idea(row["idea_id"], end_session=fx.SESSION)

    for path, body in existing.items():
        assert path.read_bytes() == body, path
    for path in missing:
        assert not path.exists(), f"keeping an idea created {path}"
