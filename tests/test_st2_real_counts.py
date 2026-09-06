"""Packet ST2 - real integer counts, and ONE honest leader.

Written by the tester BEFORE the fix, on `claude/st2-real-counts` off `main`
at ``84ee24d6`` (2026-09-06). Every test here is red on that commit except
where the handoff says otherwise. The builder makes them pass and may only ADD.

Why these tests exist
---------------------
The Setup Tracker's recent-types table shows a win rate that nobody counted.
``build_recent_tracker_setup_family_rows`` computes ``win_rate_closed`` as a
RECENCY-WEIGHTED mean of win flags (``legacy.py`` ~7507-7528, weights at
~7402-7409), the panel hands that rate to ``swing_headline.headline_from_rate``
(``setup_tracker_panel.py:1392-1400``), and that function rebuilds an integer
pair with ``wins = round(rate * n)``. Reproduced on this branch's base:

    two 28-day-old wins (weight .25 each) + two same-day losses (weight 1.0)
    -> win_rate_closed 0.2, and the panel prints ``25% (>=5%, n=4)``
    where the truth is 2 wins and 2 losses, 50%.

The 25% is not a rounding artefact - it is a count that was never observed,
carrying a Wilson bound computed from it. And the "BEST PERFORMING RIGHT NOW"
banner (``_best_now_banner_html``, ``:1583-1637``) picks its swing winner by
``max(avg_closed_r)`` over ``closed_setups >= 3`` across BOTH namespaces, so a
three-example study with a big R outranks a ninety-episode live family and the
banner disagrees with the table right underneath it (the table already sorts by
the Wilson lower bound). Reproduced on the base: the table lists
``tight_and_hot`` first and the banner crowns ``fat_but_wide``.

Exactly what this file pins (so the builder is not guessing)
-----------------------------------------------------------
**ST2.1 - `master_avwap_lib.legacy.build_recent_tracker_setup_family_rows`**
gains these keys, ADDITIVE AND AT THE END of the row dict (the golden test
requires the existing column order to remain a PREFIX of the new one):

* ``n_wins`` / ``n_losses`` - integers, from each EPISODE's representative
  closed R (``representative_closed_r``, falling back to ``avg_closed_r``, the
  same field ``win_flags`` already reads). Unweighted. Never rounded from a rate.
* ``n_flats`` - representative closed R exactly ``0.0``.
* ``n_unmeasured`` - closed with no readable R.
* ``n_pending`` - episodes with no closed tradeable scenario (open).
* ``n_observations`` - rows BEFORE ``_dedupe_recent_tracker_family_rows``.
* ``n_episodes`` - rows AFTER dedupe; equals today's ``tracked_setups``.
* ``n_symbols`` - distinct symbols among the episodes.
* ``n_entry_sessions`` - distinct ``scan_date`` among the episodes.
* ``win_rate_closed_unweighted`` = ``n_wins / (n_wins + n_losses)``, ``None``
  when that denominator is 0.
* ``win_rate_closed_basis`` == ``"recency_weighted_half_life"``.
* ``outcome_kind`` == ``"trade_r_representative_exit"``.
* ``horizon_basis`` == ``f"{lookback_days}d lookback, representative exit"``.
* ``latest_measured_session`` (ST2.3's freshness input) - the max exit/scan
  session among the counted episodes, as an ISO date string.

``win_rate_closed`` KEEPS its name and its recency-weighted value.
``build_recent_setup_type_stat_rows`` passes every new key through.

**The panel** (`scripts/ui/panels/setup_tracker_panel.py`):

* ``_recent_type_headline_rows`` stops calling ``headline_from_rate`` and uses
  ``swing_headline.headline_from_counts(wins=n_wins, losses=n_losses,
  avg_r=avg_closed_r)``. It still writes ``win_rate_headline`` (the
  ``swing_headline.format_win_rate`` spelling) and ``win_rate_lb``.
* A row whose ``n_wins``/``n_losses`` are MISSING (an old CSV) or PRESENT AND
  EMPTY (a row written before the counts existed) gets
  ``win_rate_headline == "counts not exported yet"`` and ``win_rate_lb is
  None``. Never reconstructed.
* ``RECENT_TYPE_COLUMNS`` carries the header ``"Win % (unweighted)"`` before a
  separate ``("win_rate_closed", "Win % (recency-weighted)")`` column.
* ``SETUP_TYPE_COLUMNS`` carries a ``"Win %"`` header BEFORE ``"Target Hit"``
  and ``"Stop"``.

**ST2.2 - `build_tracker_setup_type_rows`** rows gain ``n_wins``, ``n_losses``,
``n_flats``, ``n_unmeasured``, ``n_pending``, ``win_rate`` and ``outcome_kind``
== ``"trade_r_representative_exit"``, all at ITS OWN grain
``(side, priority_bucket, setup_family, favorite_zone, retest_label,
compression_label)``. Every existing column keeps its value.

**ST2.3 - new pure module `scripts/working_lately.py`** (no Qt, no file I/O):

* ``LEADER_MARGIN_LB == 0.05`` and ``LEADER_FRESHNESS_SESSIONS == 2``.
* ``@dataclass(frozen=True) LeaderVerdict`` whose fields, IN ORDER, are
  ``("state", "leader", "runner_up", "reason", "as_of", "policy_line",
  "coverage")``.
* ``select_leader(rows, *, kind, last_completed_session, previous=None)
  -> LeaderVerdict``.
  - ``rows`` are evidence-table row MAPPINGS; ``verdict.leader`` is the winning
    ROW MAPPING itself (so ``verdict.leader["setup_family"]`` reads), or None.
  - ``state`` is one of ``"leader"``, ``"no_clear_leader"``,
    ``"last_reliable_reading"``, ``"no_evidence"``.
  - Eligible = ``namespace == "live"`` AND ``n_wins + n_losses >=
    evidence_stats.MIN_REPORTABLE_N`` AND ``latest_measured_session`` no more
    than ``LEADER_FRESHNESS_SESSIONS`` EXCHANGE SESSIONS before
    ``last_completed_session`` (``market_calendar``, never calendar days).
  - Order by the Wilson lower bound on the INTEGER counts
    (``swing_headline.wilson_lower_bound``), ties by n.
  - ``coverage["studies_excluded"]`` is the integer number of ``namespace ==
    "study"`` rows kept out.
  - ``reason`` is lower-case-searchable and names the cause: it contains
    ``"study"`` when studies were the only candidates, ``"floor"`` when every
    live row was under ``MIN_REPORTABLE_N``, and ``"not fresh"`` when the
    evidence is stale and there is no previous verdict.
  - ``policy_line`` is a non-empty sentence containing the leader's side and
    its ``outcome_kind``.
* ``_best_now_banner_html`` consumes ``select_leader(panel.recent_type_rows,
  kind="swing", ...)``; the banner NAMES the eligible leader, never names a
  study family, and carries the word ``"excluded"`` beside the count of study
  rows it kept out (the packet's "contains 'study' only in the exclusion
  count"). ``last_completed_session`` comes from ``market_calendar``, so these
  tests date their fixture rows with that same function.

**ST2.4** - ``ranking_score``, ``score_delta`` and every pre-existing column of
both builders stay byte-identical, pinned by
``tests/fixtures/st2_recent_rows_golden.csv`` and
``tests/fixtures/st2_setup_type_rows_golden.csv``, both generated FROM ``main``
at ``84ee24d6`` (see ``_write_goldens`` below - the builder must never run it).

Fixture realism notes
---------------------
* The synthetic setups here go through the REAL builders; nothing hand-writes a
  family row except where the row is the INPUT to ``select_leader`` or to the
  panel's CSV reader, and in those cases the CSV is read back through the
  panel's own ``_load_csv_rows_cached``, so every value arrives as a STRING
  exactly as it does on the desk.
* An "old" recent-types row is modelled BOTH ways: key absent (a CSV written
  before the column existed) and key PRESENT AND EMPTY (a row in the new CSV
  with nothing to report).
* ``build_recent_setup_type_stat_rows`` has no ``reference_date`` parameter and
  reads ``datetime.now()``, so the panel fixture's ``scan_date``s are computed
  relative to today; only the AGE (0 days and 28 days, half life 14.0) decides
  the weights, so the .25/.25/1/1 split is stable on any day the suite runs.
* Freshness is measured on the exchange calendar, so the session strings are
  computed with ``market_calendar``, never as "today minus N days".
"""

from __future__ import annotations

import csv
import dataclasses
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RECENT_GOLDEN = FIXTURES / "st2_recent_rows_golden.csv"
SETUP_TYPE_GOLDEN = FIXTURES / "st2_setup_type_rows_golden.csv"

#: The one outcome-kind string ST2 stamps on both builders' rows.
OUTCOME_KIND = "trade_r_representative_exit"

#: Reference day for every golden and every deterministic builder call.
GOLDEN_REFERENCE_DATE = date(2026, 4, 22)
GOLDEN_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Synthetic tracker setups, shaped like the real records the builders read
# ---------------------------------------------------------------------------


def _setup(
    symbol: str,
    scan_date: str,
    total_r,
    *,
    side: str = "LONG",
    family: str = "post_earnings_52w_break",
    bucket: str = "favorite_setup",
    scenario_status: str = "TARGET_HIT",
    favorite_zone: str = "",
    retest_label: str = "",
    compression_label: str = "N",
) -> dict:
    """One tracked setup with one tradeable scenario.

    ``stop_reference_label`` is the side's protective band so
    ``_representative_scenario`` finds a representative and the counts are read
    from ``representative_closed_r``, which is the production path, not from
    the cross-variant average fallback.

    ``total_r=None`` models a CLOSED scenario whose R cannot be read - the
    "unmeasured" case, which is not a loss.
    """
    setup_status = "CLOSED" if scenario_status in {"TARGET_HIT", "STOPPED"} else "OPEN"
    return {
        "setup_id": f"{scan_date}:{symbol}:{family}:{retest_label}:{compression_label}",
        "symbol": symbol,
        "side": side,
        "scan_date": scan_date,
        "priority_bucket": bucket,
        "setup_family": family,
        "favorite_zone": favorite_zone,
        "retest_label": retest_label,
        "compression_label": compression_label,
        "setup_tags": [],
        "favorite_signals": [],
        "setup_status": setup_status,
        "scenarios": {
            "baseline": {
                "experimental": False,
                "tradeable": True,
                "status": scenario_status,
                "total_r": total_r,
                "stop_reference_label": "LOWER_1" if side.upper() == "LONG" else "UPPER_1",
            }
        },
    }


def _legacy():
    from master_avwap_lib import legacy

    return legacy


def _family_row(rows: list[dict], family: str) -> dict:
    matches = [row for row in rows if row.get("setup_family") == family]
    assert matches, f"no row for family {family!r} in {[r.get('setup_family') for r in rows]}"
    assert len(matches) == 1, f"expected one row for {family!r}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------


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


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _last_completed_session() -> date:
    import market_calendar

    return market_calendar.last_completed_session(
        datetime.now(market_calendar.MARKET_TZ)
    )


def _sessions_back(day: date, count: int) -> date:
    import market_calendar

    walked = day
    for _ in range(count):
        walked = market_calendar.previous_session(walked)
    return walked


# ---------------------------------------------------------------------------
# Evidence-table rows, as the CSV hands them to the panel and to select_leader
# ---------------------------------------------------------------------------

RECENT_CSV_FIELDS = (
    "type_label",
    "namespace",
    "side",
    "priority_bucket",
    "setup_family",
    "lookback_days",
    "tracked_setups",
    "closed_setups",
    "avg_total_r",
    "avg_closed_r",
    "representative_total_r",
    "representative_closed_r",
    "win_rate_closed",
    "profit_factor",
    "target_hit_rate",
    "stop_rate",
    "sample_setups",
    "score_delta",
    "ranking_score",
    "status",
    "n_wins",
    "n_losses",
    "n_flats",
    "n_unmeasured",
    "n_pending",
    "win_rate_closed_unweighted",
    "win_rate_closed_basis",
    "outcome_kind",
    "horizon_basis",
    "latest_measured_session",
)


def _evidence_row(
    family: str,
    *,
    namespace: str,
    wins: int,
    losses: int,
    avg_closed_r: float,
    session: str,
    side: str = "LONG",
    status: str = "",
) -> dict:
    n = wins + losses
    return {
        "type_label": f"{side} | favorite_setup | family={family}",
        "namespace": namespace,
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "lookback_days": "30",
        "tracked_setups": str(n),
        "closed_setups": str(n),
        "avg_total_r": str(avg_closed_r),
        "avg_closed_r": str(avg_closed_r),
        "representative_total_r": str(avg_closed_r),
        "representative_closed_r": str(avg_closed_r),
        "win_rate_closed": str(wins / n) if n else "",
        "profit_factor": "1.5",
        "target_hit_rate": "0.5",
        "stop_rate": "0.4",
        "sample_setups": "",
        "score_delta": "0",
        "ranking_score": "0.0",
        "status": status,
        "n_wins": str(wins),
        "n_losses": str(losses),
        "n_flats": "0",
        "n_unmeasured": "0",
        "n_pending": "0",
        "win_rate_closed_unweighted": str(wins / n) if n else "",
        "win_rate_closed_basis": "recency_weighted_half_life",
        "outcome_kind": OUTCOME_KIND,
        "horizon_basis": "30d lookback, representative exit",
        "latest_measured_session": session,
    }


# ===========================================================================
# 1 - the weighted rate is labelled, and the counted one is the truth
# ===========================================================================


def test_a_recency_weighted_family_reports_its_true_two_of_four_win_count(
    panel_module, tmp_path
):
    """Two old wins at weight .25 and two fresh losses at weight 1.0.

    The recency-weighted rate is 0.2 and stays 0.2 under its own name. The
    COUNTS are 2 and 2, the unweighted rate is 0.5, and the panel's headline
    cell says 50% on n=4 - not the 25% it invents today from ``round(0.2 * 4)``.
    """
    import pandas as pd

    legacy = _legacy()

    # `build_recent_setup_type_stat_rows` reads `datetime.now()`, so the ages
    # (not the dates) are what this fixture fixes: 28 days is two half lives.
    today = date.today()
    old = (today - timedelta(days=28)).isoformat()
    fresh = today.isoformat()
    payload = {
        "setups": {
            "w1": _setup("OLDWA", old, 1.5),
            "w2": _setup("OLDWB", old, 1.2),
            "l1": _setup("NEWLA", fresh, -1.0, scenario_status="STOPPED"),
            "l2": _setup("NEWLB", fresh, -0.8, scenario_status="STOPPED"),
        }
    }

    rows = legacy.build_recent_setup_type_stat_rows(payload)
    row = _family_row(rows, "post_earnings_52w_break")

    # The existing weighted rate is untouched and now says what it is.
    assert row["win_rate_closed"] == pytest.approx(0.2, abs=1e-9)
    assert row["win_rate_closed_basis"] == "recency_weighted_half_life"
    # The counts nobody could see before.
    assert row["n_wins"] == 2
    assert row["n_losses"] == 2
    assert row["win_rate_closed_unweighted"] == pytest.approx(0.5, abs=1e-9)
    assert row["outcome_kind"] == OUTCOME_KIND
    assert row["horizon_basis"] == "30d lookback, representative exit"

    # ...and they survive the real export to CSV and the panel's own reader.
    csv_path = tmp_path / "master_avwap_setup_type_recent_stats.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    loaded = panel_module._load_csv_rows_cached(csv_path)
    enriched = panel_module._recent_type_headline_rows(loaded)
    headline = _family_row(enriched, "post_earnings_52w_break")["win_rate_headline"]
    assert "50%" in headline, headline
    assert "n=4" in headline, headline
    assert "25%" not in headline, headline

    # The recent table shows both rates and says which is which.
    headers = [label for _field, label in panel_module.RECENT_TYPE_COLUMNS]
    fields = [field for field, _label in panel_module.RECENT_TYPE_COLUMNS]
    assert "Win % (unweighted)" in headers, headers
    assert ("win_rate_closed", "Win % (recency-weighted)") in list(
        panel_module.RECENT_TYPE_COLUMNS
    ), panel_module.RECENT_TYPE_COLUMNS
    assert headers.index("Win % (unweighted)") < headers.index(
        "Win % (recency-weighted)"
    )
    assert "win_rate_closed" in fields


def test_a_row_without_exported_counts_says_so_instead_of_reconstructing_one(
    panel_module, tmp_path
):
    """Two SEPARATE files, because the two shapes cannot coexist in one CSV.

    * ``yesterdays_export.csv`` was written before the columns existed, so
      ``csv.DictReader`` never produces the key at all.
    * ``todays_export.csv`` has the columns and a family with nothing graded,
      so the key is PRESENT AND EMPTY - the shape a test that models an old row
      as "key missing" would sail straight past.

    Neither may be turned back into a count. Today both print
    ``25% (>=5%, n=4)`` from ``round(0.2 * 4)``.
    """
    old_export = {
        "setup_family": "written_before_the_counts",
        "namespace": "live",
        "win_rate_closed": "0.2",
        "closed_setups": "4",
        "avg_closed_r": "-0.45",
    }
    old_path = tmp_path / "yesterdays_export.csv"
    _write_csv(old_path, sorted(old_export), [old_export])

    new_export = dict(old_export)
    new_export["setup_family"] = "new_export_nothing_graded"
    new_export["n_wins"] = ""
    new_export["n_losses"] = ""
    new_path = tmp_path / "todays_export.csv"
    _write_csv(new_path, sorted(new_export), [new_export])

    absent = panel_module._load_csv_rows_cached(old_path)
    assert "n_wins" not in absent[0], absent[0]
    present_empty = panel_module._load_csv_rows_cached(new_path)
    assert present_empty[0]["n_wins"] == "", present_empty[0]

    for row in panel_module._recent_type_headline_rows(absent + present_empty):
        assert row["win_rate_headline"] == "counts not exported yet", row
        assert row["win_rate_lb"] is None, row


# ===========================================================================
# 2 - a study never leads, however big its R
# ===========================================================================


def test_a_high_r_study_never_becomes_the_leader_over_a_supported_live_family(
    panel_module, tmp_path, monkeypatch
):
    """The study here is the BEST row on every number a reader can see - 40-2,
    +3.10R, a higher Wilson bound than the live family. It is still not the
    leader, and the banner does not name it."""
    from evidence_stats import MIN_REPORTABLE_N
    from working_lately import select_leader

    session = _last_completed_session().isoformat()
    study = _evidence_row(
        "second_dev_breakout_study",
        namespace="study",
        wins=40,
        losses=2,
        avg_closed_r=3.10,
        session=session,
    )
    live = _evidence_row(
        "post_earnings_52w_break",
        namespace="live",
        wins=24,
        losses=16,
        avg_closed_r=0.50,
        session=session,
    )
    assert 24 + 16 >= MIN_REPORTABLE_N, "the live family must clear the floor"

    verdict = select_leader(
        [study, live],
        kind="swing",
        last_completed_session=_last_completed_session(),
    )
    assert verdict.state == "leader"
    assert verdict.leader["setup_family"] == "post_earnings_52w_break"
    assert verdict.coverage["studies_excluded"] == 1
    assert "LONG" in verdict.policy_line
    assert OUTCOME_KIND in verdict.policy_line

    csv_path = tmp_path / "recent.csv"
    _write_csv(csv_path, RECENT_CSV_FIELDS, [study, live])
    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", csv_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        html = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()

    assert "post_earnings_52w_break" in html, html
    assert "second_dev_breakout_study" not in html, html
    assert "excluded" in html.lower(), html


# ===========================================================================
# 3 - the banner and the table cannot disagree
# ===========================================================================


def test_the_banner_names_the_same_family_the_table_ranks_first_by_the_bound(
    panel_module, tmp_path, monkeypatch
):
    """``fat_but_wide`` has the biggest mean R (+2.50R on 90) and
    ``tight_and_hot`` the biggest Wilson bound (0.627 vs 0.497 on 30). The
    table already ranks by the bound; the banner picks max R, so today they
    name different families on the same screen."""
    session = _last_completed_session().isoformat()
    fat = _evidence_row(
        "fat_but_wide", namespace="live", wins=54, losses=36, avg_closed_r=2.50, session=session
    )
    tight = _evidence_row(
        "tight_and_hot", namespace="live", wins=24, losses=6, avg_closed_r=0.40, session=session
    )

    csv_path = tmp_path / "recent.csv"
    _write_csv(csv_path, RECENT_CSV_FIELDS, [fat, tight])
    monkeypatch.setattr(panel_module, "RECENT_SETUP_TYPE_STATS_FILE", csv_path)
    panel = panel_module.SetupTrackerPanel()
    try:
        ordered = [row["setup_family"] for row in panel.recent_type_rows]
        html = panel_module._best_now_banner_html(panel)
    finally:
        panel.deleteLater()

    # No NEW/RISING pin is set on either row, so this IS the performance order.
    assert ordered[0] == "tight_and_hot", ordered
    assert "tight_and_hot" in html, html
    assert "fat_but_wide" not in html, html


# ===========================================================================
# 4 - flat, unmeasured, pending: three facts that are not losses
# ===========================================================================


def test_a_flat_close_and_an_unreadable_close_are_counted_apart_from_wins_and_losses():
    """Six episodes from seven observations (one symbol re-scanned the next
    day): 2 wins, 1 loss, 1 flat, 1 closed-but-unreadable, 1 open."""
    legacy = _legacy()
    setups = {
        "a": _setup("WINA", "2026-04-20", 1.0),
        # the SAME episode re-scanned the next day - collapsed by
        # `_dedupe_recent_tracker_family_rows`, counted once
        "a_rescan": _setup("WINA", "2026-04-21", 1.0),
        "b": _setup("WINB", "2026-04-20", 1.0),
        "l": _setup("LOSS1", "2026-04-19", -1.0, scenario_status="STOPPED"),
        "f": _setup("FLAT1", "2026-04-19", 0.0),
        "u": _setup("UNMEA", "2026-04-18", None, scenario_status="STOPPED"),
        "o": _setup("OPEN1", "2026-04-18", 0.5, scenario_status="OPEN"),
    }
    rows = legacy.build_recent_tracker_setup_family_rows(
        setups,
        reference_date=GOLDEN_REFERENCE_DATE,
        lookback_days=GOLDEN_LOOKBACK_DAYS,
    )
    row = _family_row(rows, "post_earnings_52w_break")

    assert row["n_wins"] == 2
    assert row["n_losses"] == 1
    assert row["n_flats"] == 1
    assert row["n_unmeasured"] == 1
    assert row["n_pending"] == 1
    # A flat is not a loss and an unreadable close is not a loss.
    assert row["n_wins"] + row["n_losses"] == 3
    assert row["win_rate_closed_unweighted"] == pytest.approx(2 / 3, abs=1e-9)

    # Observations vs episodes, under their true names.
    assert row["n_observations"] == 7
    assert row["n_episodes"] == 6
    assert row["n_episodes"] == row["tracked_setups"]
    assert row["n_symbols"] == 6
    # 2026-04-20, 2026-04-19, 2026-04-18 - the rescanned episode keeps its first
    assert row["n_entry_sessions"] == 3
    # Every episode is accounted for exactly once.
    assert (
        row["n_wins"] + row["n_losses"] + row["n_flats"] + row["n_unmeasured"] + row["n_pending"]
        == row["n_episodes"]
    )
    assert row["latest_measured_session"] == "2026-04-20"


# ===========================================================================
# 5 - nothing eligible says why
# ===========================================================================


def test_nothing_eligible_reports_no_evidence_and_says_why():
    from evidence_stats import MIN_REPORTABLE_N
    from working_lately import select_leader

    last_session = _last_completed_session()
    session = last_session.isoformat()

    studies_only = [
        _evidence_row("study_a", namespace="study", wins=40, losses=2, avg_closed_r=3.1, session=session),
        _evidence_row("study_b", namespace="study", wins=35, losses=10, avg_closed_r=1.4, session=session),
    ]
    verdict = select_leader(studies_only, kind="swing", last_completed_session=last_session)
    assert verdict.state == "no_evidence"
    assert verdict.leader is None
    assert "study" in verdict.reason.lower(), verdict.reason
    assert verdict.coverage["studies_excluded"] == 2

    thin = MIN_REPORTABLE_N - 1
    under_floor = [
        _evidence_row("live_a", namespace="live", wins=thin, losses=0, avg_closed_r=1.1, session=session),
        _evidence_row("live_b", namespace="live", wins=2, losses=1, avg_closed_r=0.9, session=session),
    ]
    verdict = select_leader(under_floor, kind="swing", last_completed_session=last_session)
    assert verdict.state == "no_evidence"
    assert verdict.leader is None
    assert "floor" in verdict.reason.lower(), verdict.reason


# ===========================================================================
# 6 - stale evidence keeps the last reliable reading, and says it is old
# ===========================================================================


def test_stale_evidence_falls_back_to_the_last_reliable_reading():
    from working_lately import (
        LEADER_FRESHNESS_SESSIONS,
        LEADER_MARGIN_LB,
        LeaderVerdict,
        select_leader,
    )

    assert LEADER_MARGIN_LB == 0.05
    assert LEADER_FRESHNESS_SESSIONS == 2
    assert dataclasses.is_dataclass(LeaderVerdict)
    assert LeaderVerdict.__dataclass_params__.frozen is True
    assert [field.name for field in dataclasses.fields(LeaderVerdict)] == [
        "state",
        "leader",
        "runner_up",
        "reason",
        "as_of",
        "policy_line",
        "coverage",
    ]

    last_session = _last_completed_session()
    fresh_session = last_session.isoformat()
    # FIVE EXCHANGE SESSIONS old, walked on the calendar - never "minus 5 days".
    stale_session = _sessions_back(last_session, 5).isoformat()

    fresh_rows = [
        _evidence_row(
            "post_earnings_52w_break",
            namespace="live",
            wins=24,
            losses=16,
            avg_closed_r=0.5,
            session=fresh_session,
        )
    ]
    previous = select_leader(fresh_rows, kind="swing", last_completed_session=last_session)
    assert previous.state == "leader"

    stale_rows = [
        _evidence_row(
            "post_earnings_52w_break",
            namespace="live",
            wins=24,
            losses=16,
            avg_closed_r=0.5,
            session=stale_session,
        )
    ]

    carried = select_leader(
        stale_rows,
        kind="swing",
        last_completed_session=last_session,
        previous=previous,
    )
    assert carried.state == "last_reliable_reading"
    assert carried.leader is not None
    assert carried.leader["setup_family"] == previous.leader["setup_family"]
    assert carried.as_of == previous.as_of

    alone = select_leader(stale_rows, kind="swing", last_completed_session=last_session)
    assert alone.state == "no_evidence"
    assert alone.leader is None
    assert "not fresh" in alone.reason.lower(), alone.reason


# ===========================================================================
# 7 - Setup Types counts belong to the row, not to the group
# ===========================================================================


def _six_row_group() -> dict:
    """Eighteen setups in ONE (side, bucket, family, zone) group, split six ways
    by (retest_label, compression_label) - the grain the export actually uses.

    Each of the six carries a DIFFERENT outcome mix, so a coarse rate joined
    across them would be visibly wrong on five of the six rows.
    """
    plans = {
        ("AVWAPE", "N"): [(1.0, "TARGET_HIT"), (1.2, "TARGET_HIT"), (-1.0, "STOPPED")],
        ("AVWAPE", "Y"): [(1.0, "TARGET_HIT"), (0.0, "TARGET_HIT"), (None, "STOPPED")],
        ("EMA21", "N"): [(1.0, "TARGET_HIT"), (0.5, "OPEN"), (0.7, "OPEN")],
        ("EMA21", "Y"): [(-1.0, "STOPPED"), (-0.9, "STOPPED"), (-1.1, "STOPPED")],
        ("None", "N"): [(-1.0, "STOPPED"), (0.0, "TARGET_HIT"), (0.3, "OPEN")],
        ("None", "Y"): [(None, "STOPPED"), (None, "STOPPED"), (2.0, "TARGET_HIT")],
    }
    setups: dict[str, dict] = {}
    index = 0
    for (retest, comp), outcomes in plans.items():
        for total_r, status in outcomes:
            index += 1
            setups[f"s{index}"] = _setup(
                f"SYM{index:02d}",
                "2026-04-20",
                total_r,
                scenario_status=status,
                favorite_zone="Z1",
                retest_label=retest,
                compression_label=comp,
            )
    return setups


def test_setup_types_counts_belong_to_the_row_not_to_the_group(panel_module):
    legacy = _legacy()
    rows = legacy.build_tracker_setup_type_rows(_six_row_group())
    assert len(rows) == 6, [row["setup_type_id"] for row in rows]

    by_key = {
        (row["retest_label"], row["compression_label"]): row for row in rows
    }
    expected = {
        ("AVWAPE", "N"): dict(n_wins=2, n_losses=1, n_flats=0, n_unmeasured=0, n_pending=0),
        ("AVWAPE", "Y"): dict(n_wins=1, n_losses=0, n_flats=1, n_unmeasured=1, n_pending=0),
        ("EMA21", "N"): dict(n_wins=1, n_losses=0, n_flats=0, n_unmeasured=0, n_pending=2),
        ("EMA21", "Y"): dict(n_wins=0, n_losses=3, n_flats=0, n_unmeasured=0, n_pending=0),
        ("None", "N"): dict(n_wins=0, n_losses=1, n_flats=1, n_unmeasured=0, n_pending=1),
        ("None", "Y"): dict(n_wins=1, n_losses=0, n_flats=0, n_unmeasured=2, n_pending=0),
    }

    total = 0
    for key, counts in expected.items():
        row = by_key[key]
        for name, value in counts.items():
            assert row[name] == value, f"{key} {name}: {row[name]} != {value}"
        row_total = sum(row[name] for name in counts)
        # Every count is this ROW's own population - never the group's rate
        # copied down six rows.
        assert row_total == int(row["tradeable_setups"]), (key, row_total, row["tradeable_setups"])
        assert row["outcome_kind"] == OUTCOME_KIND
        total += row_total

    assert total == 18
    assert sum(int(row["tracked_setups"]) for row in rows) == 18

    # AVWAPE|N is 2-1 at its own grain; the group as a whole is 5-5.
    assert by_key[("AVWAPE", "N")]["win_rate"] == pytest.approx(2 / 3, abs=1e-9)
    assert by_key[("EMA21", "Y")]["win_rate"] == pytest.approx(0.0, abs=1e-9)
    # Nothing graded at all -> no rate, never a zero.
    assert by_key[("None", "Y")]["win_rate"] == pytest.approx(1.0, abs=1e-9)

    # The tab leads with the win rate, with target/stop beside it.
    headers = [label for _field, label in panel_module.SETUP_TYPE_COLUMNS]
    assert "Win %" in headers, headers
    assert headers.index("Win %") < headers.index("Target Hit"), headers
    assert headers.index("Win %") < headers.index("Stop"), headers


# ===========================================================================
# 8 - the champion's columns do not move
# ===========================================================================


def _golden_setups() -> dict:
    """A deterministic population covering both sides, two families, three
    zones, closed / open / flat / unreadable outcomes and one re-scan."""
    return {
        "g01": _setup("AAA", "2026-04-21", 1.8, favorite_zone="Z1", retest_label="AVWAPE"),
        "g02": _setup("AAA", "2026-04-22", 1.8, favorite_zone="Z1", retest_label="AVWAPE"),
        "g03": _setup("BBB", "2026-04-20", -1.0, scenario_status="STOPPED", favorite_zone="Z1", retest_label="AVWAPE"),
        "g04": _setup("CCC", "2026-04-18", 0.0, favorite_zone="Z2", retest_label="None"),
        "g05": _setup("DDD", "2026-04-15", None, scenario_status="STOPPED", favorite_zone="Z2", retest_label="None"),
        "g06": _setup("EEE", "2026-04-10", 0.6, scenario_status="OPEN", favorite_zone="Z2", retest_label="None", compression_label="Y"),
        "g07": _setup("FFF", "2026-04-08", 2.4, family="post_earnings_candle_break", favorite_zone="Z3"),
        "g08": _setup("GGG", "2026-04-05", -0.7, family="post_earnings_candle_break", scenario_status="STOPPED", favorite_zone="Z3"),
        "g09": _setup("HHH", "2026-04-02", 1.1, family="post_earnings_candle_break", favorite_zone="Z3", compression_label="Y"),
        "g10": _setup("III", "2026-04-21", -1.4, side="SHORT", scenario_status="STOPPED", favorite_zone="Z1"),
        "g11": _setup("JJJ", "2026-04-19", 2.2, side="SHORT", favorite_zone="Z1"),
        "g12": _setup("KKK", "2026-04-17", 0.4, side="SHORT", scenario_status="OPEN", favorite_zone="Z1", compression_label="Y"),
    }


def _golden_recent_rows():
    return _legacy().build_recent_tracker_setup_family_rows(
        _golden_setups(),
        reference_date=GOLDEN_REFERENCE_DATE,
        lookback_days=GOLDEN_LOOKBACK_DAYS,
    )


def _golden_setup_type_rows():
    return _legacy().build_tracker_setup_type_rows(_golden_setups())


def _render(rows: list[dict], columns) -> str:
    import pandas as pd

    return pd.DataFrame(rows)[list(columns)].to_csv(index=False)


def _number(value) -> str:
    """A canonical spelling for a numeric cell, so None and "" compare equal
    and a float from the CSV compares against the float that produced it."""
    if value is None or value == "":
        return ""
    return f"{float(value):.10g}"


def _golden_columns(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _golden_text(path: Path) -> str:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return handle.read()


def _write_goldens() -> None:  # pragma: no cover - run by hand, on `main` only
    """Pin both goldens FROM THE OLD CODE.

    **The builder must never run this.** A golden regenerated by the code it is
    meant to pin is a self-portrait. These two files were written on `main` at
    ``84ee24d6``, before any ST2 change existed, and are committed with the red
    tests. Re-running it requires ``ST2_REGENERATE_GOLDENS_FROM_MAIN=1`` on a
    checkout of that commit.
    """
    recent = _golden_recent_rows()
    types = _golden_setup_type_rows()
    for path, rows in ((RECENT_GOLDEN, recent), (SETUP_TYPE_GOLDEN, types)):
        text = _render(rows, list(rows[0].keys()))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write(text)


def test_regenerate_the_goldens_by_hand_on_main_only():
    """Not a test of the product. Skipped unless the env var is set."""
    if os.environ.get("ST2_REGENERATE_GOLDENS_FROM_MAIN") != "1":
        pytest.skip("goldens are pinned from main at 84ee24d6; set the env var to rewrite")
    _write_goldens()  # pragma: no cover


def test_the_original_export_columns_stay_byte_identical_while_the_counts_are_added():
    """Golden characterization first (this half is green on `main` by
    construction - it is the regression guard), then the additions."""
    recent_rows = _golden_recent_rows()
    type_rows = _golden_setup_type_rows()

    recent_columns = _golden_columns(RECENT_GOLDEN)
    type_columns = _golden_columns(SETUP_TYPE_GOLDEN)

    assert _render(recent_rows, recent_columns) == _golden_text(RECENT_GOLDEN)
    assert _render(type_rows, type_columns) == _golden_text(SETUP_TYPE_GOLDEN)

    # ranking_score and score_delta named explicitly: they are scoring inputs,
    # and plan.md sec 5 forbids moving one without a golden.
    for produced, columns, golden_path in (
        (recent_rows, recent_columns, RECENT_GOLDEN),
        (type_rows, type_columns, SETUP_TYPE_GOLDEN),
    ):
        with golden_path.open("r", newline="", encoding="utf-8") as handle:
            golden_rows = list(csv.DictReader(handle))
        assert len(golden_rows) == len(produced)
        for old, new in zip(golden_rows, produced):
            for column in ("ranking_score", "score_delta"):
                if column in columns:
                    assert _number(new[column]) == _number(old[column]), (
                        column,
                        new.get("type_label"),
                    )

    # ADDITIVE AT THE END: the old header is a PREFIX of the new recent header.
    produced_recent_columns = list(recent_rows[0].keys())
    assert produced_recent_columns[: len(recent_columns)] == recent_columns, (
        produced_recent_columns
    )
    for column in (
        "n_wins",
        "n_losses",
        "n_flats",
        "n_unmeasured",
        "n_observations",
        "n_episodes",
        "n_symbols",
        "n_entry_sessions",
        "n_pending",
        "win_rate_closed_unweighted",
        "win_rate_closed_basis",
        "outcome_kind",
        "horizon_basis",
        "latest_measured_session",
    ):
        assert column in produced_recent_columns, column

    produced_type_columns = list(type_rows[0].keys())
    assert set(type_columns).issubset(set(produced_type_columns))
    for column in (
        "n_wins",
        "n_losses",
        "n_flats",
        "n_unmeasured",
        "n_pending",
        "win_rate",
        "outcome_kind",
    ):
        assert column in produced_type_columns, column


# ===========================================================================
# 9 - the reconstruction seam is gone from the panel
# ===========================================================================


def test_the_recent_rows_no_longer_reconstruct_a_count_from_a_stored_rate():
    """A source scan, and it is only defensible BESIDE test 1, which drives the
    real path and asserts the number. ``headline_from_rate`` rebuilds
    ``round(rate * n)``; the recent rows must not reach it at all, because a
    later edit could re-introduce the call without changing today's numbers.

    ``headline_from_rate`` itself STAYS - the veto and like cohort CSVs write
    ``wins / n`` and are legitimate callers. Its docstring must name them and
    must forbid a weighted rate.
    """
    import swing_headline

    panel_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ui"
        / "panels"
        / "setup_tracker_panel.py"
    ).read_text(encoding="utf-8")
    assert "headline_from_rate" not in panel_source
    assert "headline_from_counts" in panel_source

    # The function survives for its legitimate callers, and its docstring has
    # to say what it must never be handed.
    doc = swing_headline.headline_from_rate.__doc__ or ""
    assert "weighted" in doc.lower(), doc
