"""Packet ST4 - the selected opportunity is fixed before its outcome is seen.

Written RED, before the fix, against the real seams in
``scripts/master_avwap_lib/legacy.py``. When it was written the default policy
had to stay byte-identical and the challenger was opt-in evidence only.

**Packet ST7 (2026-09-06, decision 0019) made ``first_actionable_v2`` the
DEFAULT**, on the trader's *"Yes a trade not yet completed should say pending. A
second entry after a first close is its own trade yes."* Not one assertion below
was removed for it: every leg that characterized v1 through the bare signature
now NAMES ``SELECTION_CLOSED_FIRST_V1`` and asserts the same numbers, so v1 stays
reproducible forever, and every "and the default agrees" leg now asserts the v2
answer. Test 8 pins BOTH whole outputs - ``st4_family_rows_golden.csv`` for v1 by
name, ``st7_family_rows_v2_default_golden.csv`` for the default.

Everything this file pins, so the builder cannot satisfy it by inventing a
different name (packet ST4, items ST4.1-ST4.5):

New module ``scripts/master_avwap_lib/selection_policy.py``
  * ``SELECTION_CLOSED_FIRST_V1`` - value ``"closed_first_v1"``, today's policy.
  * ``SELECTION_FIRST_ACTIONABLE_V2`` - value ``"first_actionable_v2"``.
  * ``DEFAULT_SELECTION_POLICY`` - was ``SELECTION_CLOSED_FIRST_V1``; since
    packet ST7 it IS ``SELECTION_FIRST_ACTIONABLE_V2``.
  * ``REENTRY_RULE_V2`` - a non-empty documentation string.
  * ``assign_attempts(rows, *, policy) -> list[dict]`` - each returned row
    carries ``attempt_index`` (1-based int) and ``role``, one of the pinned
    strings ``ATTEMPT_ROLE_FIRST_ACTIONABLE`` (``"first_actionable"``) /
    ``ATTEMPT_ROLE_RESCAN`` (``"rescan"``).
  * ``select_episode_rows(rows, *, policy) -> list[dict]``.

Row keys the attempt rule reads
  * ``representative_exit_date`` - the ISO date the row's representative
    scenario CLOSED, empty/absent while it is open. There is no scalar exit
    field on a scenario today (see PREMISES in the handoff): the recorded exit
    date is the ``trade_date`` of the scenario's last ``events`` entry, and the
    row-level key above is what the family builder must stamp from it.

Keyword arguments (additive, defaulted to today's behaviour)
  * ``_dedupe_recent_tracker_family_rows(rows, *, policy=...)``
  * ``_representative_scenario(tradeable, primary_stop_label, *, policy=...)``
  * ``_summarize_tracker_setup_outcome(setup, *, policy=...)``
  * ``build_recent_tracker_setup_family_rows(..., selection_policy=..., as_of_session=...)``

Summary / family-row keys added (additive; the golden proves no original
column moved)
  * ``representative_exit_template_id`` - already emitted today; under v2 it is
    ``REPRESENTATIVE_EXIT_TEMPLATE_ID_V2 == "full_band2"`` regardless of the
    ``scenarios`` dict order.
  * ``representative_status`` - the literal ``"pending"`` when the v2
    representative has not closed.
  * ``selection_policy``, ``as_of_session``, ``n_excluded``,
    ``excluded_reasons`` (a ``;``-joined ``reason=count`` string using the
    pinned tokens ``untradeable``, ``after_as_of`` and, for decision (b),
    ``expired_unmeasured_in_population``).

``scripts/tracker_selection_compare.py``
  * ``PROTECTED_DATA_ROOT`` - the live home folder the CLI refuses.
  * ``main(argv) -> int`` - non-zero for a ``--tracker`` or ``--out`` under
    that root, and it never overwrites a stamped output.

Deliberately NOT referenced here: any column packet ST2 introduces
(``n_observations``, ``n_episodes``, ``n_wins``, ...). ST2 is being built on a
parallel branch; ST4's own grain names on today's code are ``tracked_setups``
(deduped episodes) and ``closed_setups``.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_CSV = FIXTURES_DIR / "st4_family_rows_golden.csv"

GOLDEN_REFERENCE_DATE = date(2026, 2, 2)
GOLDEN_LOOKBACK_DAYS = 45


def _sp():
    """The new selection-policy module, imported at call time.

    Imported inside each test rather than at module scope so a test that has
    its own reachable seam (a missing keyword argument, a missing column) fails
    on THAT, not on a collection-time ImportError shared by the whole file.
    """
    from master_avwap_lib import selection_policy

    return selection_policy


# ---------------------------------------------------------------------------
# Synthetic tracker records. Real shape: a scenario's exit date lives on the
# LAST entry of its `events` list, and an old/open scenario has the key PRESENT
# and EMPTY (`"events": []`), never absent.
# ---------------------------------------------------------------------------
def _scenario(
    *,
    scenario_id,
    stop_label,
    exit_template_id,
    status,
    total_r,
    exit_date=None,
    tradeable=True,
    days_held=4,
):
    return {
        "scenario_id": scenario_id,
        "stop_reference_label": stop_label,
        "stop_reference_level": 95.0,
        "stop_source_type": "band",
        "exit_template_id": exit_template_id,
        "exit_template_label": exit_template_id,
        "framework_family": "baseline",
        "framework_version": "baseline",
        "experimental": False,
        "tradeable": bool(tradeable),
        "status": status,
        "total_r": total_r,
        "days_held": days_held,
        "entry_price": 100.0,
        "initial_risk_per_share": 5.0,
        "initial_risk_usd": 500.0,
        "direction": 1.0,
        "events": (
            [
                {
                    "trade_date": exit_date,
                    "reason": "FINAL_TARGET" if (total_r or 0) > 0 else "STOP",
                    "price": 110.0,
                    "shares": 100,
                }
            ]
            if exit_date
            else []
        ),
    }


def _setup(
    *,
    symbol,
    side,
    anchor_date,
    scan_date,
    setup_family,
    priority_bucket="tracked",
    setup_status="CLOSED",
    scenarios,
    regime_label="",
):
    record = {
        "symbol": symbol,
        "side": side,
        "anchor_date": anchor_date,
        "scan_date": scan_date,
        "priority_bucket": priority_bucket,
        "setup_family": setup_family,
        "setup_status": setup_status,
        "favorite_signals": [],
        "scenarios": {s["scenario_id"]: s for s in scenarios},
    }
    if regime_label:
        record["market_regime_label"] = regime_label
    return record


def _golden_setups():
    """A deterministic synthetic tracker population.

    Two sides, two families, a closed episode, an open episode, a rescan pair
    (same symbol/side/anchor/family on two scan dates) and a mixed-outcome
    setup, so the golden exercises the dedupe, the representative and the
    weighted aggregates rather than a single trivial row.
    """
    long_stop = "LOWER_1"
    short_stop = "UPPER_1"
    setups = {}

    setups["s1"] = _setup(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
        setup_family="avwape_bounce", setup_status="OPEN",
        scenarios=[
            _scenario(scenario_id="s1a", stop_label=long_stop, exit_template_id="full_band2",
                      status="OPEN", total_r=0.4),
            _scenario(scenario_id="s1b", stop_label=long_stop, exit_template_id="full_band3",
                      status="TARGET_HIT", total_r=2.5, exit_date="2026-01-12"),
        ],
    )
    setups["s2"] = _setup(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-14",
        setup_family="avwape_bounce", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="s2a", stop_label=long_stop, exit_template_id="full_band2",
                      status="TARGET_HIT", total_r=1.8, exit_date="2026-01-21"),
            _scenario(scenario_id="s2b", stop_label=long_stop, exit_template_id="full_band3",
                      status="STOPPED", total_r=-1.0, exit_date="2026-01-19"),
        ],
    )
    setups["s3"] = _setup(
        symbol="AMD", side="LONG", anchor_date="2026-01-06", scan_date="2026-01-08",
        setup_family="avwape_bounce", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="s3a", stop_label=long_stop, exit_template_id="full_band2",
                      status="STOPPED", total_r=-1.0, exit_date="2026-01-15", days_held=5),
        ],
    )
    setups["s4"] = _setup(
        symbol="TSLA", side="SHORT", anchor_date="2026-01-09", scan_date="2026-01-12",
        setup_family="post_earnings_52w_break", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="s4a", stop_label=short_stop, exit_template_id="full_band2",
                      status="TARGET_HIT", total_r=3.2, exit_date="2026-01-22", days_held=8),
            _scenario(scenario_id="s4b", stop_label=short_stop, exit_template_id="full_band3",
                      status="TIME_STOP", total_r=0.3, exit_date="2026-01-26", days_held=12),
        ],
    )
    setups["s5"] = _setup(
        symbol="MSFT", side="SHORT", anchor_date="2026-01-13", scan_date="2026-01-16",
        setup_family="post_earnings_52w_break", setup_status="OPEN",
        scenarios=[
            _scenario(scenario_id="s5a", stop_label=short_stop, exit_template_id="full_band2",
                      status="OPEN", total_r=-0.2, days_held=3),
        ],
        regime_label="bull_trend",
    )
    setups["s6"] = _setup(
        symbol="AAPL", side="LONG", anchor_date="2026-01-20", scan_date="2026-01-23",
        setup_family="post_earnings_52w_break", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="s6a", stop_label=long_stop, exit_template_id="full_band2",
                      status="TARGET_HIT", total_r=1.1, exit_date="2026-01-29", days_held=4),
        ],
    )
    return setups


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _rows_to_csv_text(rows, columns):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row[column]) for column in columns])
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 1 (ST4.1) - the old policy, named, still does exactly what it did.
# ---------------------------------------------------------------------------
def test_closed_first_v1_named_explicitly_reproduces_todays_dedupe():
    """`tests/test_tracker_methodology.py` is untouched; the same two fixtures
    are replayed here with the policy NAMED, and must answer identically."""
    policy = _sp()
    # ST7 moved the default; v1 keeps its name and its answers.
    assert policy.DEFAULT_SELECTION_POLICY == policy.SELECTION_FIRST_ACTIONABLE_V2

    prefers_closed = [
        {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
         "scan_date": "2026-01-05", "closed_setups": 0, "representative_exit_date": ""},
        {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
         "scan_date": "2026-01-07", "closed_setups": 1, "representative_exit_date": "2026-01-11"},
        {"symbol": "AMD", "side": "LONG", "anchor_date": "2026-01-03", "setup_family": "f",
         "scan_date": "2026-01-06", "closed_setups": 0, "representative_exit_date": ""},
    ]
    out = m._dedupe_recent_tracker_family_rows(
        prefers_closed, policy=policy.SELECTION_CLOSED_FIRST_V1
    )
    assert len(out) == 2
    nvda = next(row for row in out if row["symbol"] == "NVDA")
    assert nvda["closed_setups"] == 1
    assert nvda["scan_date"] == "2026-01-07"
    assert nvda["selection_policy"] == policy.SELECTION_CLOSED_FIRST_V1

    earliest_when_none_closed = [
        {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
         "scan_date": "2026-01-09", "closed_setups": 0, "representative_exit_date": ""},
        {"symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
         "scan_date": "2026-01-05", "closed_setups": 0, "representative_exit_date": ""},
    ]
    out = m._dedupe_recent_tracker_family_rows(
        earliest_when_none_closed, policy=policy.SELECTION_CLOSED_FIRST_V1
    )
    assert len(out) == 1
    assert out[0]["scan_date"] == "2026-01-05"
    assert out[0]["selection_policy"] == policy.SELECTION_CLOSED_FIRST_V1

    # ST7: the DEFAULT dedupe is v2 and stamps its own name. Neither of these
    # two fixtures holds a re-entry (the second row rescans a LIVE attempt in
    # the first, and nothing has closed in the second), so v2 selects one row
    # per thesis here too - and picks the EARLIEST scan, never the closed one.
    default_prefers_closed = m._dedupe_recent_tracker_family_rows(
        [dict(row) for row in prefers_closed]
    )
    assert len(default_prefers_closed) == 2
    default_nvda = next(row for row in default_prefers_closed if row["symbol"] == "NVDA")
    assert default_nvda["scan_date"] == "2026-01-05"
    assert default_nvda["selection_policy"] == policy.SELECTION_FIRST_ACTIONABLE_V2
    default_earliest = m._dedupe_recent_tracker_family_rows(
        [dict(row) for row in earliest_when_none_closed]
    )
    assert len(default_earliest) == 1
    assert default_earliest[0]["scan_date"] == "2026-01-05"
    assert default_earliest[0]["selection_policy"] == policy.SELECTION_FIRST_ACTIONABLE_V2


# ---------------------------------------------------------------------------
# 2 (ST4.1) - the review's 08-10 open vs 08-15 closed case.
# ---------------------------------------------------------------------------
def test_first_actionable_keeps_the_earlier_open_entry_over_the_later_closed_rescan():
    policy = _sp()
    rows = [
        {"symbol": "PLTR", "side": "LONG", "anchor_date": "2026-08-04", "setup_family": "f",
         "scan_date": "2026-08-10", "closed_setups": 0, "representative_exit_date": "",
         "representative_closed_r": None, "any_target_hit": False, "any_stopped": False},
        {"symbol": "PLTR", "side": "LONG", "anchor_date": "2026-08-04", "setup_family": "f",
         "scan_date": "2026-08-15", "closed_setups": 1, "representative_exit_date": "2026-08-19",
         "representative_closed_r": 2.0, "any_target_hit": True, "any_stopped": False},
    ]

    v2 = m._dedupe_recent_tracker_family_rows(
        [dict(row) for row in rows], policy=policy.SELECTION_FIRST_ACTIONABLE_V2
    )
    assert len(v2) == 1, "the rescan of an OPEN attempt is the same episode"
    assert v2[0]["scan_date"] == "2026-08-10"
    # The trade you could actually have taken is still open: it is pending, and
    # it contributes neither a win nor a loss.
    assert int(v2[0]["closed_setups"] or 0) == 0
    assert v2[0]["representative_closed_r"] is None

    # Characterized: today's policy takes the later, closed rescan.
    v1 = m._dedupe_recent_tracker_family_rows(
        [dict(row) for row in rows], policy=policy.SELECTION_CLOSED_FIRST_V1
    )
    assert len(v1) == 1
    assert v1[0]["scan_date"] == "2026-08-15"
    assert int(v1[0]["closed_setups"] or 0) == 1


# ---------------------------------------------------------------------------
# 3 (ST4.2) - an open representative is pending, never the mean of the others.
# ---------------------------------------------------------------------------
def test_open_representative_stays_pending_instead_of_borrowing_the_closed_mean():
    setup = _setup(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
        setup_family="avwape_bounce", setup_status="OPEN",
        scenarios=[
            # The representative (primary stop LOWER_1, first baseline template)
            # is still OPEN at +0.40R...
            _scenario(scenario_id="a", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="OPEN", total_r=0.4),
            # ...while an alternate exit plan on the same stop has closed +3.00R.
            _scenario(scenario_id="b", stop_label="LOWER_1", exit_template_id="full_band3",
                      status="TARGET_HIT", total_r=3.0, exit_date="2026-01-13"),
        ],
    )

    # Both policies stamp the representative's own state (ST4.2), so this key
    # is present on the DEFAULT call too.
    assert "representative_status" in m._summarize_tracker_setup_outcome(setup)

    policy = _sp()
    # ST7: the DEFAULT is the repaired read - pending stays pending.
    default = m._summarize_tracker_setup_outcome(setup)
    assert default["representative_closed_r"] is None
    assert default["representative_status"] == "pending"
    assert default["selection_policy"] == policy.SELECTION_FIRST_ACTIONABLE_V2
    v2 = m._summarize_tracker_setup_outcome(setup, policy=policy.SELECTION_FIRST_ACTIONABLE_V2)
    assert v2["representative_closed_r"] is None
    assert v2["representative_status"] == "pending"
    assert v2["representative_exit_template_id"] == "full_band2"

    # Characterized defect: today the open representative is reported as the
    # mean of the OTHER closed scenarios - +3.00R for a trade still running.
    v1 = m._summarize_tracker_setup_outcome(setup, policy=policy.SELECTION_CLOSED_FIRST_V1)
    assert v1["representative_closed_r"] == pytest.approx(3.0)
    assert v1["selection_policy"] == policy.SELECTION_CLOSED_FIRST_V1


# ---------------------------------------------------------------------------
# 4 (ST4.2) - the exit template is declared, not decided by dict order.
# ---------------------------------------------------------------------------
def test_representative_exit_template_survives_a_scenario_dict_reorder():
    assert m.REPRESENTATIVE_EXIT_TEMPLATE_ID_V2 == "full_band2"
    policy = _sp()
    band2 = _scenario(scenario_id="a", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="TARGET_HIT", total_r=2.0, exit_date="2026-01-13")
    band3 = _scenario(scenario_id="b", stop_label="LOWER_1", exit_template_id="full_band3",
                      status="STOPPED", total_r=-1.0, exit_date="2026-01-11")

    def _record(scenarios):
        return _setup(
            symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
            setup_family="avwape_bounce", setup_status="CLOSED", scenarios=scenarios,
        )

    forward = _record([band2, band3])
    reordered = _record([band3, band2])

    v2_forward = m._summarize_tracker_setup_outcome(
        forward, policy=policy.SELECTION_FIRST_ACTIONABLE_V2
    )
    v2_reordered = m._summarize_tracker_setup_outcome(
        reordered, policy=policy.SELECTION_FIRST_ACTIONABLE_V2
    )
    assert v2_forward["representative_exit_template_id"] == "full_band2"
    assert v2_reordered["representative_exit_template_id"] == "full_band2"
    assert v2_reordered["representative_total_r"] == pytest.approx(2.0)
    assert v2_forward["representative_total_r"] == pytest.approx(2.0)

    # ST7: the DEFAULT is v2, so the declared template survives the reorder.
    assert m._summarize_tracker_setup_outcome(reordered)[
        "representative_exit_template_id"
    ] == "full_band2"
    assert m._summarize_tracker_setup_outcome(reordered)[
        "representative_total_r"
    ] == pytest.approx(2.0)

    # Characterized under v1 BY NAME: the answer moves with the dict order, and
    # so does the headline R (+2.00R becomes -1.00R for the same setup). That is
    # the defect this fixture documents and ST7 retired from the default.
    v1_forward = m._summarize_tracker_setup_outcome(
        forward, policy=policy.SELECTION_CLOSED_FIRST_V1
    )
    v1_reordered = m._summarize_tracker_setup_outcome(
        reordered, policy=policy.SELECTION_CLOSED_FIRST_V1
    )
    assert v1_forward["representative_exit_template_id"] == "full_band2"
    assert v1_reordered["representative_exit_template_id"] == "full_band3"
    assert v1_forward["representative_total_r"] == pytest.approx(2.0)
    assert v1_reordered["representative_total_r"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 5 (ST4.1) - a legitimate re-entry vs a rescan of a live attempt.
# ---------------------------------------------------------------------------
def test_a_second_attempt_needs_the_first_to_have_closed_first():
    policy = _sp()
    first_entry = {
        "symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
        "scan_date": "2026-01-05", "closed_setups": 1,
        "representative_exit_date": "2026-01-09",
    }
    rescan_while_open = {
        "symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
        "scan_date": "2026-01-07", "closed_setups": 1,
        "representative_exit_date": "2026-01-09",
    }
    re_entry_after_the_exit = {
        "symbol": "NVDA", "side": "LONG", "anchor_date": "2026-01-02", "setup_family": "f",
        "scan_date": "2026-01-13", "closed_setups": 1,
        "representative_exit_date": "2026-01-16",
    }
    rows = [first_entry, rescan_while_open, re_entry_after_the_exit]

    assigned = policy.assign_attempts(
        [dict(row) for row in rows], policy=policy.SELECTION_FIRST_ACTIONABLE_V2
    )
    # Three observations of one thesis; two of them are one attempt.
    assert len(assigned) == 3
    by_scan = {row["scan_date"]: row for row in assigned}
    assert by_scan["2026-01-05"]["attempt_index"] == 1
    assert by_scan["2026-01-05"]["role"] == policy.ATTEMPT_ROLE_FIRST_ACTIONABLE
    assert by_scan["2026-01-07"]["attempt_index"] == 1
    assert by_scan["2026-01-07"]["role"] == policy.ATTEMPT_ROLE_RESCAN
    assert by_scan["2026-01-13"]["attempt_index"] == 2
    assert by_scan["2026-01-13"]["role"] == policy.ATTEMPT_ROLE_FIRST_ACTIONABLE
    assert sum(1 for row in assigned if row["attempt_index"] == 1) == 3 - 1

    selected = policy.select_episode_rows(
        [dict(row) for row in rows], policy=policy.SELECTION_FIRST_ACTIONABLE_V2
    )
    assert len(selected) == 2
    assert sorted(row["scan_date"] for row in selected) == ["2026-01-05", "2026-01-13"]
    assert sorted(row["attempt_index"] for row in selected) == [1, 2]

    # The same three rows through the real dedupe seam.
    assert len(m._dedupe_recent_tracker_family_rows(
        [dict(row) for row in rows], policy=policy.SELECTION_FIRST_ACTIONABLE_V2
    )) == 2

    # Characterized: today every rescan of one thesis collapses to ONE row, so
    # a real second entry is invisible.
    assert len(policy.select_episode_rows(
        [dict(row) for row in rows], policy=policy.SELECTION_CLOSED_FIRST_V1
    )) == 1
    assert len(m._dedupe_recent_tracker_family_rows(
        [dict(row) for row in rows], policy=policy.SELECTION_CLOSED_FIRST_V1
    )) == 1

    # ST7: the DEFAULT sees the second entry.
    assert len(m._dedupe_recent_tracker_family_rows([dict(row) for row in rows])) == 2


# ---------------------------------------------------------------------------
# 6 (ST4.3) - replay as-of.
# ---------------------------------------------------------------------------
def test_as_of_session_makes_a_later_exit_pending_and_excludes_later_scans():
    early = _setup(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
        setup_family="avwape_bounce", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="a", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="TARGET_HIT", total_r=2.0, exit_date="2026-01-09"),
        ],
    )
    scanned_after_the_cutoff = _setup(
        symbol="AMD", side="LONG", anchor_date="2026-01-08", scan_date="2026-01-12",
        setup_family="avwape_bounce", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="b", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="STOPPED", total_r=-1.0, exit_date="2026-01-15"),
        ],
    )

    rows = m.build_recent_tracker_setup_family_rows(
        {"early": early, "later": scanned_after_the_cutoff},
        reference_date=date(2026, 1, 20),
        lookback_days=45,
        as_of_session="2026-01-07",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["as_of_session"] == "2026-01-07"
    assert "after_as_of=1" in str(row["excluded_reasons"])
    assert int(row["n_excluded"] or 0) == 1
    # NVDA's only scenario exits on 01-09, after the cutoff: at 01-07 the trade
    # was still running, so it is pending and grades nothing.
    assert int(row["tracked_setups"] or 0) == 1
    assert int(row["closed_setups"] or 0) == 0

    # Without the cutoff both setups are present and NVDA has closed.
    unbounded = m.build_recent_tracker_setup_family_rows(
        {"early": early, "later": scanned_after_the_cutoff},
        reference_date=date(2026, 1, 20),
        lookback_days=45,
    )
    assert int(unbounded[0]["tracked_setups"] or 0) == 2
    assert int(unbounded[0]["closed_setups"] or 0) == 2


# ---------------------------------------------------------------------------
# 7 (ST4.4) - the decided cases are NAMED, not silently re-decided.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy_attribute", ["SELECTION_CLOSED_FIRST_V1", "SELECTION_FIRST_ACTIONABLE_V2"])
def test_population_accounting_names_untradeable_and_expired_unmeasured(policy_attribute):
    normal = _setup(
        symbol="NVDA", side="LONG", anchor_date="2026-01-02", scan_date="2026-01-05",
        setup_family="avwape_bounce", setup_status="CLOSED",
        scenarios=[
            _scenario(scenario_id="a", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="TARGET_HIT", total_r=2.0, exit_date="2026-01-09"),
        ],
    )
    # Decision (c), 2026-09-06: neither open nor closed - already outside every
    # count through the `tradeable` filter. This packet only makes it NAMED.
    untradeable = _setup(
        symbol="AMD", side="LONG", anchor_date="2026-01-03", scan_date="2026-01-06",
        setup_family="avwape_bounce", setup_status="UNTRADEABLE",
        scenarios=[
            _scenario(scenario_id="b", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="UNTRADEABLE", total_r=0.0, tradeable=False),
        ],
    )
    # Decision (b), 2026-09-06: EXPIRED_UNMEASURED stays IN the champion's
    # scoring population and is excluded from EXPORTS only. This builder feeds
    # the scoring, so the record is COUNTED and merely named.
    expired = _setup(
        symbol="TSLA", side="LONG", anchor_date="2026-01-04", scan_date="2026-01-07",
        setup_family="avwape_bounce", setup_status=m.SETUP_STATUS_EXPIRED_UNMEASURED,
        scenarios=[
            _scenario(scenario_id="c", stop_label="LOWER_1", exit_template_id="full_band2",
                      status="OPEN", total_r=0.1),
        ],
    )

    population = {"normal": normal, "untradeable": untradeable, "expired": expired}

    # ST4.4: the accounting columns are on EVERY family row, including the
    # default call that ships today.
    default_row = m.build_recent_tracker_setup_family_rows(
        population, reference_date=date(2026, 1, 20), lookback_days=45
    )[0]
    assert "n_excluded" in default_row
    assert "excluded_reasons" in default_row

    policy = _sp()
    selected_policy = getattr(policy, policy_attribute)
    rows = m.build_recent_tracker_setup_family_rows(
        population,
        reference_date=date(2026, 1, 20),
        lookback_days=45,
        selection_policy=selected_policy,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["selection_policy"] == selected_policy
    reasons = str(row["excluded_reasons"])
    assert "untradeable=1" in reasons
    assert "expired_unmeasured_in_population=1" in reasons
    assert int(row["n_excluded"] or 0) == 1, "only the untradeable record is excluded"
    # Decision (b) is not reopened: the expired record is still in the
    # population beside the normal one.
    assert int(row["tracked_setups"] or 0) == 2
    assert int(row["closed_setups"] or 0) == 1


# ---------------------------------------------------------------------------
# 8 (invariant) - two policies, two whole outputs, both pinned.
# ---------------------------------------------------------------------------
V2_GOLDEN_CSV = FIXTURES_DIR / "st7_family_rows_v2_default_golden.csv"


def test_v1_by_name_reproduces_the_golden_family_rows_and_the_default_is_the_v2_pin():
    """`tests/fixtures/st4_family_rows_golden.csv` was pinned from `main` at
    84ee24d6 by a scratch script, BEFORE any ST4 code existed, and is reproduced
    by NAMING `closed_first_v1`. `st7_family_rows_v2_default_golden.csv` was
    pinned on `main` at 68762909 through the explicit `first_actionable_v2`
    keyword, BEFORE the default flipped, and is what a DEFAULT build must now
    produce - so neither pin is a self-portrait of the code under test."""
    golden_text = GOLDEN_CSV.read_text(encoding="utf-8")
    golden_columns = next(csv.reader(io.StringIO(golden_text)))

    rows = m.build_recent_tracker_setup_family_rows(
        _golden_setups(),
        reference_date=GOLDEN_REFERENCE_DATE,
        lookback_days=GOLDEN_LOOKBACK_DAYS,
        selection_policy=_sp().SELECTION_CLOSED_FIRST_V1,
    )
    assert _rows_to_csv_text(rows, golden_columns) == golden_text

    v2_text = V2_GOLDEN_CSV.read_text(encoding="utf-8")
    v2_columns = next(csv.reader(io.StringIO(v2_text)))
    assert v2_text != golden_text, "the new default really moves the rows"

    default_rows = m.build_recent_tracker_setup_family_rows(
        _golden_setups(),
        reference_date=GOLDEN_REFERENCE_DATE,
        lookback_days=GOLDEN_LOOKBACK_DAYS,
    )
    assert _rows_to_csv_text(default_rows, v2_columns) == v2_text

    explicit_default = m.build_recent_tracker_setup_family_rows(
        _golden_setups(),
        reference_date=GOLDEN_REFERENCE_DATE,
        lookback_days=GOLDEN_LOOKBACK_DAYS,
        selection_policy=_sp().DEFAULT_SELECTION_POLICY,
    )
    assert _rows_to_csv_text(explicit_default, v2_columns) == v2_text


# ---------------------------------------------------------------------------
# 9 (ST4.5) - the comparison CLI cannot reach the live home folder.
# ---------------------------------------------------------------------------
def _write_scratch_tracker(path: Path) -> None:
    payload = m._default_setup_tracker_payload()
    payload["data_session"] = "2026-02-02"
    payload["setups"] = {
        "s6": _setup(
            symbol="AAPL", side="LONG", anchor_date="2026-01-20", scan_date="2026-01-23",
            setup_family="post_earnings_52w_break", setup_status="CLOSED",
            scenarios=[
                _scenario(scenario_id="s6a", stop_label="LOWER_1", exit_template_id="full_band2",
                          status="TARGET_HIT", total_r=1.1, exit_date="2026-01-29"),
            ],
        ),
        "s7": _setup(
            symbol="AMD", side="LONG", anchor_date="2026-01-21", scan_date="2026-01-26",
            setup_family="post_earnings_52w_break", setup_status="OPEN",
            scenarios=[
                _scenario(scenario_id="s7a", stop_label="LOWER_1", exit_template_id="full_band2",
                          status="OPEN", total_r=0.3),
            ],
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_cli_refuses_the_live_home_folder_and_never_overwrites(tmp_path):
    import tracker_selection_compare as compare

    protected = Path(str(compare.PROTECTED_DATA_ROOT))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # A --tracker under the protected root is refused, and nothing is written.
    rc = compare.main([
        "--tracker", str(protected / "data" / "runtime" / "st4_never_read.json"),
        "--out", str(out_dir),
    ])
    assert rc != 0
    assert list(out_dir.iterdir()) == []

    # So is an --out under it.
    tracker_copy = tmp_path / "tracker_copy.json"
    _write_scratch_tracker(tracker_copy)
    rc = compare.main([
        "--tracker", str(tracker_copy),
        "--out", str(protected / "st4_never_written"),
    ])
    assert rc != 0
    assert not (protected / "st4_never_written").exists()

    # A legitimate run writes a stamped json + csv...
    assert compare.main(["--tracker", str(tracker_copy), "--out", str(out_dir)]) == 0
    after_first = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
    assert any(name.startswith("selection_comparison_") and name.endswith(".json")
               for name in after_first)
    assert any(name.startswith("selection_comparison_") and name.endswith(".csv")
               for name in after_first)
    first_json = next(name for name in after_first
                      if name.startswith("selection_comparison_") and name.endswith(".json"))
    report = json.loads(after_first[first_json].decode("utf-8"))
    assert "README" in report

    # ...and a second run never rewrites the first run's bytes.
    compare.main(["--tracker", str(tracker_copy), "--out", str(out_dir)])
    for name, original_bytes in after_first.items():
        assert (out_dir / name).read_bytes() == original_bytes, f"{name} was overwritten"
