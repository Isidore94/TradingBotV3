"""Phase 0.13 packet P7 - one name per setup, and one row per registered grid.

Both are READ-ONLY crosswalks. As of the 2026-09-02 merge the registry has
exactly two readers - the fact pack's role lookup and the selftest's asset check,
both pinned below - and the trial ledger still has none.

What these tests protect is not behaviour; it is the claim the registry makes:
that every name any of the five naming sites uses resolves to exactly one entry,
and that every recipe belongs to exactly one declared trial. A registry that is
wrong about that is worse than no registry, because a later packet will freeze
the identity graph on top of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# The registry resolves every name in every source
# ---------------------------------------------------------------------------


def test_every_scanner_family_resolves_to_exactly_one_entry():
    import setup_registry as registry
    from master_avwap_lib.setup_tagging import _FAMILY_TAGS

    for family_key, canonical in _FAMILY_TAGS.items():
        by_key = registry.resolve(family_key)
        by_canonical = registry.resolve(canonical)
        assert by_key is by_canonical, family_key
        assert by_key["canonical_setup_id"] == canonical


def test_every_documented_setup_resolves_to_exactly_one_entry():
    import setup_registry as registry
    from setup_docs import SETUP_DOCS

    for docs_key in SETUP_DOCS:
        entry = registry.resolve(docs_key)
        assert entry["docs_key"] == docs_key, docs_key


def test_every_playbook_family_resolves_to_exactly_one_entry():
    import setup_registry as registry
    from setup_playbook_study import PLAYBOOK

    for family in PLAYBOOK:
        entry = registry.resolve(family)
        assert entry["playbook_family"] == family, family


def test_every_offered_claim_resolves_to_exactly_one_entry():
    """A claim that resolves to nothing grades under a name no other table uses."""
    import setup_registry as registry
    from ui.annotations.setup_claims import all_setup_claims

    for claim in all_setup_claims():
        entry = registry.resolve(claim.setup_id)
        assert entry["claim_id"] == claim.setup_id, claim.setup_id


def test_the_fifth_naming_site_is_covered_too():
    """`legacy.py`'s `*_STUDY_FAMILY` constants - eight are named nowhere else."""
    import setup_registry as registry
    from build_setup_registry import legacy_study_families

    families = legacy_study_families()
    assert len(families) >= 17, "the constants moved; regenerate the registry"
    for family in families:
        registry.resolve(family)


def test_no_entry_is_orphaned():
    """Every row is reachable from at least one naming site."""
    import setup_registry as registry

    for key, entry in registry.registry().items():
        reachable = (
            entry["family_tag_key"]
            or entry["docs_key"]
            or entry["playbook_family"]
            or entry["claim_id"]
            or entry["legacy_study_constant"]
        )
        assert reachable, f"{key} is named by nothing"
        assert entry["sources"], f"{key} records no source"


def test_one_name_never_names_two_setups():
    """The index is built collision-free or it raises; this asserts it is built."""
    import setup_registry as registry

    assert len(registry.registry()) >= 50
    # Building the index is the assertion: `_index` raises on a collision.
    assert registry.resolve("avwap_band_bounce")["canonical_setup_id"] == "AVWAP_BAND_BOUNCE"


# ---------------------------------------------------------------------------
# The roles Appendix C names
# ---------------------------------------------------------------------------


def test_general_is_a_fallback_and_favorite_zone_watch_is_a_watch_state():
    """Appendix C: "must not become a pooled 'setup' edge" / "never counted as a
    triggered trade setup"."""
    import setup_registry as registry

    assert registry.role("GENERAL") == registry.ROLE_FALLBACK
    assert registry.role("FAVORITE_ZONE_WATCH") == registry.ROLE_WATCH_STATE
    assert registry.role("baseline_every5") == registry.ROLE_CONTROL
    assert registry.role("avwap_band_bounce") == registry.ROLE_TRADE_SETUP


def test_every_role_is_appendix_cs_vocabulary():
    import setup_registry as registry

    for entry in registry.registry().values():
        assert entry["role"] in registry.ROLES, entry["setup_id"]


def test_the_fact_pack_reads_the_registry_and_keeps_its_own_wording():
    """One owner for the role map, two spellings for the same thing.

    The fact pack (P3) prints `TRADE`; Appendix C writes `TRADE_SETUP`. The
    registry keeps the spec's spelling and this translates, so replacing the fact
    pack's own two-entry map changes no output.
    """
    import setup_registry as registry

    assert registry.fact_pack_role("GENERAL") == "FALLBACK"
    assert registry.fact_pack_role("FAVORITE_ZONE_WATCH") == "WATCH_STATE"
    assert registry.fact_pack_role("AVWAP_BREAKOUT") == "TRADE"
    # An unknown family still grades as a trade setup - a registry gap must not
    # silently reclassify live evidence.
    assert registry.fact_pack_role("SOMETHING_NEW") == "TRADE"


# ---------------------------------------------------------------------------
# What the registry refuses to claim
# ---------------------------------------------------------------------------


def test_nothing_is_established_that_the_sources_do_not_establish():
    """A guessed side reads as established in the column an experiment trusts."""
    import setup_registry as registry

    for entry in registry.registry().values():
        assert entry["supported_sides"] == [], entry["setup_id"]
        assert entry["primary_recipe"] == "", entry["setup_id"]
        assert "supported_sides" in entry["unestablished"]
        assert entry["authoritative_when"].startswith("plan.md P4.1")


def test_the_two_alias_tables_agree_or_the_disagreement_is_listed():
    """Where `setup_tagging` and `setup_docs` disagree, it is DATA, not a fix."""
    import setup_registry as registry
    from master_avwap_lib.setup_tagging import _FAMILY_TAGS
    from setup_docs import SETUP_DOC_ALIASES

    listed = {
        (item.get("alias"), item.get("docs_key"))
        for item in registry.known_divergences()
        if item["kind"] == "alias_points_at_another_family"
    }
    for alias_key, docs_key in SETUP_DOC_ALIASES.items():
        tag_canonical = _FAMILY_TAGS.get(alias_key)
        if tag_canonical is None:
            continue
        docs_entry = registry.resolve(docs_key)
        if docs_entry["canonical_setup_id"] == tag_canonical:
            continue
        assert (alias_key, docs_key) in listed, (alias_key, docs_key)

    for item in registry.known_divergences():
        assert item["resolved_by"].startswith("plan.md P4.1"), item


def test_the_frozen_json_is_what_the_generator_produces():
    """Regenerating must be a no-op, or the crosswalk is stale and lying."""
    import build_setup_registry as builder

    fresh = builder.build()
    frozen = json.loads(builder.REGISTRY_PATH.read_text(encoding="utf-8"))
    assert fresh == frozen, "run: python scripts/build_setup_registry.py --write"


def test_the_registry_is_read_only():
    import setup_registry as registry

    entry = registry.resolve("avwap_band_bounce")
    with pytest.raises(TypeError):
        entry["role"] = "CONTROL"  # type: ignore[index]


def test_an_unknown_name_raises_rather_than_defaulting():
    """A silent fallback to GENERAL files a real finding under 'untagged'."""
    import setup_registry as registry

    with pytest.raises(registry.SetupRegistryError):
        registry.resolve("a_setup_nothing_names")
    assert registry.find("a_setup_nothing_names") is None


# ---------------------------------------------------------------------------
# The trial ledger
# ---------------------------------------------------------------------------


def test_every_recipe_belongs_to_exactly_one_ledger_row():
    """The packet's named test. Two owners double-counts one look."""
    from research_warehouse import outcomes, trial_ledger

    recipe_ids = list(outcomes.RECIPES)
    for name in ("M5_CLOSE_RECIPES", "HTF_LRSI_RECIPES"):
        for recipe in getattr(outcomes, name, ()):
            recipe_ids.append(recipe.recipe_id)

    assert len(recipe_ids) >= 59
    for recipe_id in recipe_ids:
        owners = trial_ledger.owners_of(recipe_id)
        assert len(owners) == 1, f"{recipe_id} -> {owners}"


def test_the_htf_lrsi_declaration_matches_the_grid_it_was_written_blind_against():
    """P7 declared this grid from another branch's constants. The merge checked it.

    When P7 was built, `HTF_LRSI_RECIPES` lived only on
    `claude/focus-declutter-lrsi-htf`; the ledger declared the grid anyway,
    because a look-counter that starts when the code merges would record the
    family as never having been examined. Both landed on 2026-09-02, so the
    declaration can now be checked against the real thing instead of trusted -
    and this asserts the CODE's count rather than a literal 16, so widening the
    grid without re-declaring it fails here.
    """
    from research_warehouse import outcomes, trial_ledger

    row = next(
        item for item in trial_ledger.BACKFILL_TRIALS
        if item["trial_id"] == "htf_lrsi_entry_grid_v1"
    )
    assert row["declared_cell_count"] == len(outcomes.HTF_LRSI_RECIPES) == 16
    assert trial_ledger.owner_of("htf_lrsi_h4_up50_2r_v1") == "htf_lrsi_entry_grid_v1"


def test_every_backfilled_trial_names_its_authorization():
    """An experiment nobody authorized is not a registered question."""
    from research_warehouse import trial_ledger

    for row in trial_ledger.BACKFILL_TRIALS:
        assert row["authorization"].strip()
        assert row["question"].strip()
        assert row["failure_mode"].strip()
        assert row["declared_cells"]
        assert row["declared_floors"]
        assert row["declared_window"]
        # `registered` or `collecting` - P8 added the second, which says the
        # declared window's clock is running. What must NEVER be true at
        # declaration time is a status that implies the numbers were seen.
        assert row["status"] in {
            trial_ledger.STATUS_REGISTERED,
            trial_ledger.STATUS_COLLECTING,
        }, row["trial_id"]
        # Nothing may be declared with an outcome already in it.
        assert row["outcome"] == ""


def test_a_declared_cell_count_matches_the_declared_grid():
    """The count and the axes are two statements of one fact and must agree."""
    from research_warehouse import trial_ledger

    for row in trial_ledger.BACKFILL_TRIALS:
        cells = row["declared_cells"]
        if row["trial_id"] == "m5_close_recipe_grid_v1":
            # 5 sources x 3 ranks x 3 targets, plus 3 ATR controls x 3 targets.
            expected = (
                len(cells["stop_source"]) * len(cells["stop_rank"]) * len(cells["target_r"])
                + len(cells["control_atr_multiple"]) * len(cells["target_r"])
            )
        else:
            expected = 1
            for values in cells.values():
                expected *= len(values)
        assert row["declared_cell_count"] == expected, row["trial_id"]


def test_registration_is_append_only_and_idempotent(tmp_path):
    from research_warehouse import trial_ledger

    written = trial_ledger.backfill(tmp_path)
    assert len(written) == len(trial_ledger.BACKFILL_TRIALS)
    assert trial_ledger.backfill(tmp_path) == []

    rows = trial_ledger.load(tmp_path)
    assert len(rows) == len(trial_ledger.BACKFILL_TRIALS)
    assert {row["trial_id"] for row in rows} == {
        row["trial_id"] for row in trial_ledger.BACKFILL_TRIALS
    }


def test_a_declaration_is_never_rewritten(tmp_path):
    """Editing a grid of 54 down to 3 after the fact is the failure this prevents."""
    from research_warehouse import trial_ledger

    trial_ledger.backfill(tmp_path)
    rewritten = dict(trial_ledger.BACKFILL_TRIALS[0])
    rewritten["declared_cell_count"] = 3

    assert trial_ledger.register(tmp_path, rewritten) is False
    stored = [
        row for row in trial_ledger.load(tmp_path)
        if row["trial_id"] == rewritten["trial_id"]
    ]
    assert len(stored) == 1
    assert stored[0]["declared_cell_count"] == 54


def test_a_trial_without_an_authorization_pointer_is_refused(tmp_path):
    from research_warehouse import trial_ledger

    with pytest.raises(ValueError):
        trial_ledger.register(tmp_path, {"trial_id": "x", "authorization": ""})
    with pytest.raises(ValueError):
        trial_ledger.register(tmp_path, {"trial_id": "", "authorization": "plan.md"})
    with pytest.raises(ValueError):
        trial_ledger.register(
            tmp_path, {"trial_id": "x", "authorization": "plan.md", "status": "done"}
        )


def test_the_ledger_never_reads_an_outcome():
    """A pre-declaration that can see the numbers is not a pre-declaration."""
    source = (ROOT / "scripts" / "research_warehouse" / "trial_ledger.py").read_text(
        encoding="utf-8"
    )
    body = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith(("#", '"', "*", "-"))
    )
    for banned in ("read_rows", "latest_outcomes", "ResearchStore", "total_r"):
        assert banned not in body, banned


def test_a_half_written_line_is_skipped_rather_than_crashing(tmp_path):
    from research_warehouse import trial_ledger

    trial_ledger.backfill(tmp_path)
    path = trial_ledger.ledger_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"trial_id": "half')

    rows = trial_ledger.load(tmp_path)
    assert len(rows) == len(trial_ledger.BACKFILL_TRIALS)


def test_the_registry_has_exactly_the_readers_it_was_given():
    """P7 shipped with NO production reader. The 2026-09-02 merge gave it one.

    That reader is packet P7's own owed item: `setup_research.family_role` was a
    two-entry role map of P3's, kept only because the registry did not exist when
    P3 was built, and P7 named the swap as owed to whichever branch landed second.
    Both landed on 2026-09-02, so it is done.

    The list is deliberately explicit rather than a "nothing imports it" rule that
    is now false: a NEW reader should be a decision somebody makes on purpose,
    because the registry is still not authoritative (plan.md P4.1 is where that
    changes) and a caller that treats it as settled would be reading columns
    nothing has established yet.
    """
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-l", "setup_registry", "--", "scripts/*.py", "scripts/**/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    importers = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert importers == {
        "scripts/setup_registry.py",
        "scripts/build_setup_registry.py",
        # The one authorized PRODUCTION reader (P7's owed swap, landed 2026-09-02).
        "scripts/ai_jobs/setup_research.py",
        # Not a reader in the same sense: the selftest loads the frozen JSON to
        # prove the BUNDLE can find it. That check exists because the registry is
        # the first non-.py asset at the scripts/ root, so the packaging spec had
        # to grow a second sweep, and a `datas` rule proves a file was bundled
        # while only a frozen run proves the process can read it.
        "scripts/selftest.py",
    }, importers


def test_the_fact_packs_role_map_has_one_owner_now():
    """The swap must change the ONTOLOGY's owner and not the pack's output."""
    from ai_jobs import setup_research

    assert not hasattr(setup_research, "NON_TRADE_FAMILY_ROLES"), (
        "the second role map is what P7 removed; it must not come back"
    )
    assert setup_research.family_role("GENERAL") == "FALLBACK"
    assert setup_research.family_role("FAVORITE_ZONE_WATCH") == "WATCH_STATE"
    assert setup_research.family_role("AVWAPE_TO_FIRST_DEV") == setup_research.ROLE_TRADE
    # Still TRADE for a family the registry has never heard of: a registry gap
    # must not silently reclassify live evidence.
    assert setup_research.family_role("SOMETHING_NEW") == setup_research.ROLE_TRADE


def test_the_trial_ledger_has_exactly_one_production_writer():
    """R1: it gained one, on purpose - the warehouse build.

    Gate 37 asks for a ledger row after one overnight run and nothing in
    production wrote one, so the declarations that are supposed to predate every
    outcome would have been written after them, by hand, whenever somebody
    remembered. `cli.run_build` registers them beside `record_firing`, in the
    same never-costs-the-build shape, and `register` refuses a trial_id the
    ledger already carries so every firing after the first writes nothing.

    Still an explicit list rather than "no importer": a SECOND writer would be a
    real decision, because a ledger written from two places can disagree about
    when a declaration was made.
    """
    import subprocess

    # IMPORT-shaped, not any mention: P8's authorization block in `outcomes.py`
    # names the trial ledger in prose, which is exactly what a registered grid
    # should do and is not a dependency.
    result = subprocess.run(
        ["git", "grep", "-lE", r"^\s*(from|import)\s+.*trial_ledger", "--", "scripts/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    importers = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    # The module does not import itself, so the one name here is the one writer.
    assert importers == {"scripts/research_warehouse/cli.py"}, importers


def test_every_trial_row_says_when_it_was_registered(tmp_path):
    """A declaration with no date cannot support the claim it exists to make.

    The ledger's whole point is that the question was written down BEFORE the
    numbers arrived. An undated row and a row written afterwards look identical
    six months later, so `registered_at` is not metadata here - it is the
    evidence.
    """
    from datetime import datetime

    from research_warehouse import trial_ledger

    for row in trial_ledger.BACKFILL_TRIALS:
        stamped = datetime.fromisoformat(str(row["registered_at"]))
        assert stamped.tzinfo is not None, row["trial_id"]

    trial_ledger.backfill(tmp_path)
    for row in trial_ledger.load(tmp_path):
        stamped = datetime.fromisoformat(str(row["registered_at"]))
        assert stamped.tzinfo is not None, row["trial_id"]


def test_a_backfilled_row_carries_its_authorization_date_not_todays(tmp_path):
    """A backfilled row stamped "today" would claim the look happened today."""
    from research_warehouse import trial_ledger

    stamps = {row["trial_id"]: str(row["registered_at"]) for row in trial_ledger.BACKFILL_TRIALS}
    assert stamps["avwap_band_challenger_v1"].startswith("2026-08-26")
    assert stamps["htf_lrsi_entry_grid_v1"].startswith("2026-09-01")


def test_a_freshly_registered_trial_is_stamped_by_the_ledger(tmp_path):
    """A caller may not choose the moment its own declaration was made."""
    from datetime import datetime, timezone

    from research_warehouse import trial_ledger

    before = datetime.now(timezone.utc)
    assert trial_ledger.register(
        tmp_path,
        {"trial_id": "fresh_v1", "authorization": "plan.md, this test"},
    )
    row = next(r for r in trial_ledger.load(tmp_path) if r["trial_id"] == "fresh_v1")
    stamped = datetime.fromisoformat(row["registered_at"])
    assert stamped.tzinfo is not None
    assert stamped >= before.replace(microsecond=0)
