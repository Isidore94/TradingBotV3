"""ST2 builder tests - the three lead decisions the tester's file does not pin.

Added by the builder on `claude/st2-real-counts-build`, each proven to fail on
the packet's base (`main` at ``84ee24d6``) before the fix existed:
``working_lately`` is a new module, ``win_rate_lb`` is not on a setup-type row,
and ``setup_type_population_sentence`` does not exist.

The three decisions (lead session, 2026-09-06):

1. ``select_leader`` takes ``min_n`` as an ARGUMENT, defaulting to
   ``evidence_stats.MIN_REPORTABLE_N``, so the two-session block can pass
   ``SHORT_TERM_MIN_SAMPLES`` without either block declaring a second
   statistics contract.
2. When nothing reaches the floor the verdict is ``no_evidence`` AND carries a
   ``discovery_leader`` - the best live row below the floor by Wilson bound - so
   the banner prints ``No leader at the n=30 floor - leading on thin evidence:
   <side> <family> (n=12), discovery only``. **Never the word leader for it.**
3. The Setup Types tab sorts by the bound INSIDE each side (row field
   ``win_rate_lb``) and carries a population sentence naming the outcome kind.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

OUTCOME_KIND = "trade_r_representative_exit"


def _qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is a gui extra
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel_module():
    if _qt_app() is None:  # pragma: no cover - PySide6 is a gui extra
        pytest.skip("PySide6 is not installed")
    from ui.panels import setup_tracker_panel

    setup_tracker_panel.clear_setup_tracker_csv_cache()
    yield setup_tracker_panel
    setup_tracker_panel.clear_setup_tracker_csv_cache()


def _last_completed_session() -> date:
    import market_calendar

    return market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))


def _row(
    family: str,
    *,
    wins: int,
    losses: int,
    side: str = "LONG",
    namespace: str = "live",
    session: str | None = None,
    avg_closed_r: float = 0.5,
) -> dict:
    n = wins + losses
    return {
        "namespace": namespace,
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "status": "",
        "closed_setups": str(n),
        "tracked_setups": str(n),
        "avg_closed_r": str(avg_closed_r),
        "target_hit_rate": "0.5",
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "0",
        "n_unmeasured": "0",
        "n_pending": "0",
        "outcome_kind": OUTCOME_KIND,
        "horizon_basis": "30d lookback, representative exit",
        "latest_measured_session": session or _last_completed_session().isoformat(),
    }


# ---------------------------------------------------------------------------
# Decision 1 - the floor is an argument, and it is named in the reason
# ---------------------------------------------------------------------------


def test_the_floor_is_an_argument_so_a_thin_block_can_declare_its_own():
    from evidence_stats import MIN_REPORTABLE_N
    from working_lately import select_leader

    last_session = _last_completed_session()
    rows = [_row("thin_but_real", wins=8, losses=4)]

    default = select_leader(rows, kind="swing", last_completed_session=last_session)
    assert default.state == "no_evidence"
    assert default.coverage["min_n"] == MIN_REPORTABLE_N
    assert f"n={MIN_REPORTABLE_N}" in default.reason

    lowered = select_leader(
        rows, kind="swing_short_term", last_completed_session=last_session, min_n=6
    )
    assert lowered.state == "leader"
    assert lowered.leader["setup_family"] == "thin_but_real"
    assert lowered.coverage["min_n"] == 6
    # The floor a verdict was measured against is never left to the reader.
    assert "n=6" in lowered.reason


# ---------------------------------------------------------------------------
# Decision 2 - a discovery row is shown and is never called a leader
# ---------------------------------------------------------------------------


def test_nothing_at_the_floor_still_names_the_best_live_row_as_discovery():
    from evidence_stats import MIN_REPORTABLE_N
    from working_lately import select_leader

    last_session = _last_completed_session()
    rows = [
        _row("thin_leader", wins=9, losses=3),  # n=12, bound ~0.469
        _row("thinner_still", wins=4, losses=4),  # n=8, bound ~0.190
    ]
    verdict = select_leader(rows, kind="swing", last_completed_session=last_session)

    assert verdict.state == "no_evidence"
    assert verdict.leader is None, "a thin row is never the leader field"
    discovery = verdict.coverage["discovery_leader"]
    assert discovery is not None
    assert discovery["setup_family"] == "thin_leader", "ordered by the bound, not by n"
    assert verdict.coverage["discovery_reason"] == "floor"
    assert verdict.coverage["under_floor"] == 2
    assert f"n={MIN_REPORTABLE_N}" in verdict.reason


def test_a_verdict_with_a_leader_never_carries_a_discovery_row_as_well():
    from working_lately import select_leader

    last_session = _last_completed_session()
    rows = [_row("supported", wins=24, losses=16), _row("thin", wins=3, losses=1)]
    verdict = select_leader(rows, kind="swing", last_completed_session=last_session)

    assert verdict.state == "leader"
    assert verdict.coverage["discovery_leader"] is None, (
        "a banner that can print both a leader and a discovery row will"
    )


def test_the_banner_prints_the_floor_sentence_and_never_the_word_leader_for_it(
    panel_module, tmp_path, monkeypatch
):
    """The sentence the lead declared, rendered by the real banner."""
    from evidence_stats import MIN_REPORTABLE_N

    rows = [
        _row("thin_leader", wins=9, losses=3, avg_closed_r=1.4),
        _row("thinner_still", wins=4, losses=4, side="SHORT", avg_closed_r=0.2),
    ]
    csv_path = tmp_path / "recent.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", csv_path)

    panel = panel_module.SetupTrackerPanel()
    _load_the_tracker(panel)
    try:
        html = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()

    assert f"No leader at the n={MIN_REPORTABLE_N} floor" in html, html
    assert "leading on thin evidence" in html, html
    assert "thin_leader" in html, html
    assert "(n=12)" in html, html
    assert "discovery only" in html, html
    # The word "Leader" is the crown. A thin row never wears it.
    assert "<b>Leader" not in html, html
    assert "Leader:" not in html, html


# ---------------------------------------------------------------------------
# Decision 3 - the Setup Types tab sorts by the bound, inside each side
# ---------------------------------------------------------------------------


def _type_row(
    family: str,
    *,
    side: str,
    wins: int,
    losses: int,
    score_delta: int = 0,
    zone: str = "Z1",
) -> dict:
    n = wins + losses
    return {
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "favorite_zone": zone,
        "retest_label": "AVWAPE",
        "compression_label": "N",
        "tracked_setups": str(n),
        "tradeable_setups": str(n),
        "closed_setups": str(n),
        "open_setups": "0",
        "avg_closed_r": "0.4",
        "avg_closed_r_edge": "0.1",
        "target_hit_rate": "0.5",
        "stop_rate": "0.4",
        "score_delta": str(score_delta),
        "ranking_score": "0.0",
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "1",
        "n_unmeasured": "2",
        "n_pending": "3",
        "win_rate": str(wins / n) if n else "",
        "outcome_kind": OUTCOME_KIND,
        "n_expired_unmeasured": "4",
    }


def test_setup_types_sort_by_the_bound_inside_each_side_not_by_the_score(panel_module):
    """`tight` (24-6, bound 0.627) beats `fat` (54-36, bound 0.497) even though
    `fat` carries the bigger `score_delta` - the sort IS the bound. Sides stay
    apart: LONG and SHORT are two books, not two ends of one list."""
    rows = [
        _type_row("fat", side="LONG", wins=54, losses=36, score_delta=9),
        _type_row("tight", side="LONG", wins=24, losses=6, score_delta=-3),
        _type_row("short_book", side="SHORT", wins=40, losses=10, score_delta=99),
        _type_row("ungraded", side="LONG", wins=0, losses=0, score_delta=50, zone="Z9"),
    ]
    enriched = panel_module._setup_type_headline_rows(rows)
    by_family = {row["setup_family"]: row for row in enriched}

    # The bound is a real field on the row, under the name the lead named.
    assert by_family["tight"]["win_rate_lb"] > by_family["fat"]["win_rate_lb"]
    assert by_family["ungraded"]["win_rate_lb"] is None
    assert "n=30" in by_family["tight"]["win_rate_headline"]

    ranked = panel_module._rank_setup_types(enriched, min_closed=0)
    ordered = [row["setup_family"] for row in ranked]
    assert ordered == ["tight", "fat", "ungraded", "short_book"], ordered

    # And the column leads, ahead of the two rates that are not win rates.
    headers = [label for _field, label in panel_module.SETUP_TYPE_COLUMNS]
    assert headers.index("Win %") < headers.index("Target Hit")
    assert headers.index("Win %") < headers.index("Stop")
    assert headers.index("Win %") < headers.index("Closed R")


def test_the_setup_types_population_sentence_names_the_outcome_kind(panel_module):
    rows = [
        _type_row("fat", side="LONG", wins=54, losses=36),
        _type_row("tight", side="LONG", wins=24, losses=6),
    ]
    sentence = panel_module.setup_type_population_sentence(rows)

    assert OUTCOME_KIND in sentence, sentence
    assert "78 win(s)" in sentence, sentence
    assert "42 loss(es)" in sentence, sentence
    # Flat, unmeasured and pending are stated and are NOT losses.
    assert "2 flat" in sentence, sentence
    assert "4 closed-unmeasured" in sentence, sentence
    assert "6 still open" in sentence, sentence
    assert "Wilson lower bound inside each side" in sentence, sentence
    # M3's clause is kept verbatim - gate #72 clause 5 reads it.
    assert "8 expired unmeasured, excluded" in sentence, sentence


def test_a_setup_type_export_without_counts_says_so_rather_than_showing_a_zero(
    panel_module,
):
    old_rows = [
        {
            "side": "LONG",
            "setup_family": "written_before_the_counts",
            "closed_setups": "12",
            "target_hit_rate": "0.5",
            "stop_rate": "0.4",
        }
    ]
    enriched = panel_module._setup_type_headline_rows(old_rows)
    assert enriched[0]["win_rate_headline"] == "counts not exported yet"
    assert enriched[0]["win_rate_lb"] is None

    sentence = panel_module.setup_type_population_sentence(old_rows)
    assert "no win/loss counts in this export yet" in sentence, sentence
    assert OUTCOME_KIND not in sentence, sentence


# ---------------------------------------------------------------------------
# The two-session block is discovery by construction
# ---------------------------------------------------------------------------


def test_the_two_session_block_can_never_be_crowned_because_it_has_no_session():
    """`build_tracker_short_horizon_rows` exports no session column, so every
    adapted row is UNDATED and `select_leader` refuses to call one fresh. The
    number still reaches the screen - as labelled discovery."""
    from working_lately import select_leader, short_term_evidence_rows

    raw = [
        {"side": "LONG", "setup_family": "fast_follow", "samples_2d": "12",
         "win_rate_2d": "0.75", "avg_r_2d": "0.6"},
    ]
    adapted = short_term_evidence_rows(raw)
    # The rate here IS wins / n, so the pair is exact, not rounded from a
    # weighted mean.
    assert adapted[0]["n_wins"] == 9
    assert adapted[0]["n_losses"] == 3
    assert adapted[0]["latest_measured_session"] == ""

    verdict = select_leader(
        adapted,
        kind="swing_short_term",
        last_completed_session=_last_completed_session(),
        min_n=6,
    )
    assert verdict.state == "no_evidence"
    assert verdict.leader is None
    assert verdict.coverage["no_session_on_row"] == 1
    assert verdict.coverage["discovery_reason"] == "no_session"
    assert verdict.coverage["discovery_leader"]["setup_family"] == "fast_follow"


def _load_the_tracker(panel) -> None:
    """G7 trigger: the Setup Tracker's read is no longer a side effect of
    building the widget, so the test asks for it.

    `tests.conftest.refresh_setup_tracker` calls the panel's own `refresh()` -
    the slot the Refresh button calls - and, once G7.2 moves the twelve export
    reads onto a worker, waits for the render that lands on the Qt thread. It is
    a trigger and nothing else: no assertion moved with it.
    """
    from tests.conftest import refresh_setup_tracker

    refresh_setup_tracker(panel)
