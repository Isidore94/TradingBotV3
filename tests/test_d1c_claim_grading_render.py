"""Packet D1C-B - the lead's three rulings on the tester's findings, pinned.

Added by the BUILDER (2026-09-14) beside the tester's 41. Nothing here weakens
or restates one of theirs; each test is a ruling the lead made when the tester
reported that the packet's wording and the code disagreed:

1. **Tracker-side cells.** `swing_evidence.POLICY_SCANROW_V1.immature_value` is
   empty, so `read_eligible_rows` can never yield a pending row for it - a v1
   outcome row does not exist until the symbol's own later scan row does. The
   cell therefore says `0 (mature by construction)` rather than a bare zero,
   `unmeasured` says `-` rather than 0, and the READ-LEVEL exclusions are
   printed ONCE as a footnote instead of being repeated per bucket.
2. **`all` is an explicit wide window.** `read_eligible_rows(end=)` moves only
   the RIGHT edge of the lately window, so a caller that wants every row must
   pass `window=`. Proven here by showing what `end=` alone would have dropped.
3. **The overlap is printed in words**: `of which N also FAV that day`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tests import d1c_claim_grading_fixtures as fx  # noqa: E402


def _comparison(window: str = "all"):
    import claimed_pick_evidence

    return claimed_pick_evidence.build_comparison(
        like_picks=fx.like_picks(),
        like_outcomes=fx.like_outcomes(),
        tier_rows=fx.tier_rows(),
        claims=fx.claims(),
        as_of=date.fromisoformat(fx.AS_OF),
        window=window,
    )


def _rendered(window: str = "all") -> str:
    import claimed_pick_evidence

    return claimed_pick_evidence.render_text(_comparison(window))


# ---------------------------------------------------------------------------
# Ruling 3 - the overlap is named, in words, and never added to anything
# ---------------------------------------------------------------------------


def test_the_report_says_of_which_n_also_fav_that_day():
    text = _rendered("all")
    assert "of which 1 also FAV that day" in text
    assert "1 also Near that day" in text
    # Named, never summed: 4 liked + 5 FAV must not appear as a count anywhere.
    assert " n=9" not in text


# ---------------------------------------------------------------------------
# Ruling 1 - the tracker's cells say what they are
# ---------------------------------------------------------------------------


def test_a_tracker_pending_cell_says_mature_by_construction_and_liked_counts():
    from swing_evidence import POLICY_SCANROW_V1

    assert POLICY_SCANROW_V1.immature_value == "", (
        "the policy gained a maturity value - the 'mature by construction' "
        "sentence is now a lie and must be rewritten, not deleted"
    )
    text = _rendered("all")
    assert "0 (mature by construction)" in text
    # My liked trades keeps its own real pending count - this week's claims
    # genuinely have not reached their fifth session.
    assert "pending 1 " in text


def test_a_tracker_unmeasured_cell_is_a_dash_and_never_a_zero():
    import claimed_pick_evidence

    comparison = _comparison("all")
    assert comparison.populations["fav"].unmeasured is None
    assert comparison.populations["liked"].unmeasured == 1
    assert claimed_pick_evidence._unmeasured_text(comparison.populations["fav"]) == "-"


def test_the_tracker_read_exclusions_are_printed_once_and_never_per_bucket():
    """FV5 is `stale_horizon` and FV6 carries horizon 1. Two rows, one line."""
    text = _rendered("all")
    footnote = "tracker read excluded: stale_horizon 1, wrong_horizon 1"
    assert text.count(footnote) == 1


# ---------------------------------------------------------------------------
# Ruling 2 - `end=` only moves the right edge
# ---------------------------------------------------------------------------


def test_the_all_window_is_explicit_because_end_alone_would_drop_the_old_rows():
    from swing_evidence import POLICY_SCANROW_V1, read_eligible_rows

    end_only = read_eligible_rows(fx.tier_rows(), POLICY_SCANROW_V1, end=fx.AS_OF)
    favourites = [
        row for row in end_only.rows if row.get("priority_bucket") == "favorite_setup"
    ]
    assert len(favourites) == 4, "the lately window is what `end=` alone produces"

    assert _comparison("all").populations["fav"].n == 5, (
        "the `all` window did not pass an explicit wide window= and silently "
        "read the lately one"
    )


# ---------------------------------------------------------------------------
# The same two rulings, where the trader actually reads them
# ---------------------------------------------------------------------------


def test_the_tab_rows_carry_the_two_rulings_verbatim():
    pytest.importorskip("PySide6")
    from ui.panels import setup_tracker_panel

    rows = {
        str(row["population"]): row
        for row in setup_tracker_panel.claim_population_table_rows(_comparison("lately"))
    }
    assert rows["FAV"]["pending"] == "0 (mature by construction)"
    assert rows["FAV"]["unmeasured"] == "-"
    assert rows["FAV"]["also_fav"] == "-", (
        "the overlap question belongs to My liked trades; a zero here would "
        "claim it was asked of FAV and answered none"
    )
    assert rows["My liked trades"]["pending"] == 1
    assert rows["My liked trades"]["also_fav"] == 1
