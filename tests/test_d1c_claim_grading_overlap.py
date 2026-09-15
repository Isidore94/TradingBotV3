"""Packet D1C-B fix round - the overlap is the SCAN'S FACT, plus seven advisories.

Reviewer NO-GO, 2026-09-14 (reproduced on copies of the live stores): the
overlap sets were built from `EligibleRead.rows` - horizon-5 rows that survived
`POLICY_SCANROW_V1` - but counted over every liked row, so a liked name from the
last three sessions could never be "also FAV that day": its fifth later scan row
does not exist yet, so the tier file carries it only at horizon 1. Live counts
were FAV 18 against a true 22 and Near 17 against a true 31, and all 14 misses
sat in the newest sessions - the ones the trader is actually looking at.

**Lead ruling: ONE definition, and it is the scan's fact.** `also_fav` /
`also_near` / `also_hc` answer *"did the scan ALSO carry this symbol and side in
that bucket on the claim's own session?"* - a question about what the scan SAW,
not about what has since matured. So the index is built from the TIER ROWS
THEMSELVES, at any horizon and whatever their eligibility, keyed
`(symbol, side, scan_date)` -> the buckets seen. Eligibility still governs the
FAV / HC / Near populations, which are a different question.

The advisories folded in here: the panel's cached CSV read, `also_hc` on the
surface, `lately` letting the policy own its length, the display label in the
`Claimed setup` column, and the provenance line.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tests import d1c_claim_grading_fixtures as fx  # noqa: E402


def _build(*, picks=None, outcomes=None, tiers=None, claims=None, window="all"):
    import claimed_pick_evidence

    return claimed_pick_evidence.build_comparison(
        like_picks=fx.like_picks() if picks is None else picks,
        like_outcomes=fx.like_outcomes() if outcomes is None else outcomes,
        tier_rows=fx.tier_rows() if tiers is None else tiers,
        claims=fx.claims() if claims is None else claims,
        as_of=date.fromisoformat(fx.AS_OF),
        window=window,
    )


# ---------------------------------------------------------------------------
# THE BLOCKER - a young claim's overlap is still the scan's fact
# ---------------------------------------------------------------------------


def test_a_liked_row_whose_tier_row_is_only_horizon_one_is_still_also_fav():
    """The defect, in one fixture.

    NEW is claimed on 2026-09-09 and the scan carried it as a `favorite_setup`
    that same day - but only at horizon 1, because its fifth later scan row has
    not happened yet. That is exactly the shape of every claim made this week,
    and it MUST count as `also FAV that day`.
    """
    session = "2026-09-09"
    picks = [fx.like_pick(session, "NEW", "LONG", fx.BREAKOUT)]
    outcomes = [
        fx.like_outcome(
            session, "NEW", "LONG", fx.BREAKOUT,
            h5_return="0.031000", matured_horizons="1,3,5,10",
        )
    ]
    tiers = [
        fx.tier_row(session, "NEW", "LONG", "favorite_setup", win="True", horizon_sessions="1"),
    ]
    comparison = _build(picks=picks, outcomes=outcomes, tiers=tiers, claims=[])
    liked = comparison.populations["liked"]

    assert liked.n == 1
    assert liked.also_fav == 1, (
        "the overlap was read off the ELIGIBLE horizon-5 rows, so a claim whose "
        "fifth scan row has not happened yet can never be 'also FAV that day'"
    )
    assert comparison.populations["fav"].n == 0, (
        "eligibility still governs the FAV POPULATION - that is a different "
        "question and the horizon-1 row is not a graded FAV observation"
    )


def test_a_stale_horizon_tier_row_still_proves_the_scan_carried_the_name():
    """`stale_horizon` is a MEASUREMENT verdict, not a sighting verdict."""
    session = "2026-09-09"
    picks = [fx.like_pick(session, "STL", "SHORT", fx.BREAKOUT)]
    outcomes = [
        fx.like_outcome(
            session, "STL", "SHORT", fx.BREAKOUT,
            h5_return="0.021000", matured_horizons="1,3,5,10",
        )
    ]
    tiers = [
        fx.tier_row(
            session, "STL", "SHORT", "near_favorite_zone", win="True", stale_horizon="True"
        ),
    ]
    comparison = _build(picks=picks, outcomes=outcomes, tiers=tiers, claims=[])

    assert comparison.populations["liked"].also_near == 1
    assert comparison.populations["near"].n == 0


def test_the_overlap_equals_the_plain_join_over_the_raw_tier_rows():
    """The count is reproducible by hand off the raw file, in both windows."""
    tiers = fx.tier_rows() + fx.high_conviction_rows()

    def _expected(bucket: str, comparison) -> int:
        seen = {
            (
                str(row.get("symbol") or "").upper(),
                "SHORT" if str(row.get("side") or "").upper().startswith("SHORT") else "LONG",
                str(row.get("scan_date") or "")[:10],
            )
            for row in tiers
            if str(row.get("priority_bucket") or "") == bucket
        }
        return sum(
            1
            for row in comparison.liked_rows
            if (row.symbol, row.side, row.session_date) in seen
        )

    for window in ("all", "lately"):
        comparison = _build(tiers=tiers, window=window)
        liked = comparison.populations["liked"]
        assert liked.also_fav == _expected("favorite_setup", comparison), window
        assert liked.also_near == _expected("near_favorite_zone", comparison), window
        assert liked.also_hc == _expected("high_conviction", comparison), window


def test_the_canonical_fixture_keeps_its_one_and_one():
    """The fix must not change the answer where the old code was right."""
    liked = _build(window="all").populations["liked"]
    assert liked.also_fav == 1
    assert liked.also_near == 1


# ---------------------------------------------------------------------------
# Advisory 4 - `also_hc` reaches the surface
# ---------------------------------------------------------------------------


def test_also_hc_is_carried_and_rendered_beside_the_other_two():
    import claimed_pick_evidence

    comparison = _build(tiers=fx.tier_rows() + fx.high_conviction_rows())
    assert comparison.populations["liked"].also_hc == 0
    text = claimed_pick_evidence.render_text(comparison)
    assert "also HC that day" in text


def test_the_population_block_has_an_also_hc_column():
    pytest.importorskip("PySide6")
    from ui.panels import setup_tracker_panel

    keys = [key for key, _header in setup_tracker_panel.CLAIM_POPULATION_COLUMNS]
    assert "also_hc" in keys


# ---------------------------------------------------------------------------
# Advisory 7 - the claim record is SHOWN somewhere, so "never graded" is true
# ---------------------------------------------------------------------------


def test_the_report_states_how_many_rows_carry_a_claim_record():
    import claimed_pick_evidence

    text = claimed_pick_evidence.render_text(_build(window="all"))
    # AAA is the one claimed_picks row; AAA, BBB, CCC, DDD, EEE, OLD are the six
    # liked rows in the `all` window, so five are annotation-only.
    assert "claims joined: 1 with a claimed_picks row, 5 annotation-only (pre-D1C)" in text


# ---------------------------------------------------------------------------
# Advisory 6 - the Claimed setup column prints what the CLI prints
# ---------------------------------------------------------------------------


def test_the_claimed_setup_column_shows_the_display_label_not_the_cohort_id():
    pytest.importorskip("PySide6")
    from ui.panels import setup_tracker_panel

    columns = dict(setup_tracker_panel.CLAIM_SETUP_COLUMNS)
    assert columns.get("setup") == "Claimed setup", (
        "the column headed 'Claimed setup' must carry the display label"
    )
    rows = setup_tracker_panel.claim_setup_table_rows(_build(window="all"))
    assert rows[0]["setup"] == "avwap_breakout"
    # The raw cohort id stays on the row - it is the join key back to the store.
    assert rows[0]["source"] == fx.BREAKOUT


# ---------------------------------------------------------------------------
# Advisory 5 - `lately` lets the policy own its own length
# ---------------------------------------------------------------------------


def test_lately_passes_end_and_all_passes_an_explicit_wide_window():
    import claimed_pick_evidence
    from swing_evidence import SwingOutcomePolicy

    seen: list[dict] = []
    original = claimed_pick_evidence.read_eligible_rows

    def _spy(rows, policy, *, end=None, window=None):
        seen.append({"end": end, "window": window})
        return original(rows, policy, end=end, window=window)

    claimed_pick_evidence.read_eligible_rows = _spy
    try:
        _build(window="lately")
    finally:
        claimed_pick_evidence.read_eligible_rows = original

    by_kind = {("window" if call["window"] is not None else "end"): call for call in seen}
    assert set(by_kind) == {"end", "window"}, seen
    assert by_kind["end"]["end"] == fx.AS_OF, "lately did not pass end="
    assert by_kind["window"]["window"] == (
        claimed_pick_evidence.EARLIEST_WINDOW_START,
        fx.AS_OF,
    )
    assert isinstance(SwingOutcomePolicy, type)


# ---------------------------------------------------------------------------
# Advisory 1 - the 11 MB tier export is parsed once per file version
# ---------------------------------------------------------------------------


def test_the_panel_parses_an_unchanged_tier_export_once(tmp_path, monkeypatch):
    """Measured 0.33 s per refresh today, on a page that refreshes on a spinbox
    step and on every tab visit. The panel already owns the answer -
    `_load_csv_rows_cached` with the same `_csv_signature` every other export
    memoizes on - so the claim read is routed through it."""
    pytest.importorskip("PySide6")
    import claimed_pick_evidence
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    paths = fx.write_store(tmp_path)
    for name, key in (
        ("LIKE_COHORT_PICKS_FILE", "picks_path"),
        ("LIKE_COHORT_OUTCOMES_FILE", "outcomes_path"),
        ("MASTER_AVWAP_TIER_OUTCOMES_FILE", "tier_path"),
        ("CLAIMED_PICKS_FILE", "claims_path"),
    ):
        monkeypatch.setattr(claimed_pick_evidence, name, paths[key], raising=False)

    parsed: list[str] = []
    original = setup_tracker_panel._load_csv_rows

    def _spy(path):
        parsed.append(str(path))
        return original(path)

    monkeypatch.setattr(setup_tracker_panel, "_load_csv_rows", _spy)

    first = setup_tracker_panel._read_claim_evidence()
    second = setup_tracker_panel._read_claim_evidence()

    tier = str(paths["tier_path"])
    assert parsed.count(tier) == 1, (
        f"the tier export was parsed {parsed.count(tier)} times on an unchanged "
        "file; it goes through _load_csv_rows_cached"
    )
    assert first["populations"] == second["populations"]
    assert first["populations"], "the read produced nothing to compare"
