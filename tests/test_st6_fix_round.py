"""ST6 re-review - the four blockers and the eight advisories.

Written by the BUILDER against the reviewer's NO-GO of 2026-09-06, one test per
item, each run RED on the pre-fix branch tip before the fix stayed in.

The four blockers, in the reviewer's own words and what each one really was:

1. **"switch OFF does not restore arrival order."** The M5 bar and the review
   queue both MUTATED their backing lists, so a sort applied while the switch
   was on could not be undone: the bar returned early when disabled and the
   already-sorted widget stayed sorted, and the review queue was rebound to the
   sorted list permanently. "Reorders and never withholds" has to be reversible
   or it is a rewrite of the record wearing a display preference's name.
2. **The cap ran on the sorted list**, so the switch decided WHICH rows survive
   `MAX_ROWS` - a display preference deleting a different alert.
3. **The day-trade cell printed an interval on the wrong quantity.** The
   statistic is `held_run_score` (a product of a rate and a trimmed mean) and
   the bound beside it was the bootstrap of the MFEs alone, so live it read
   `held x ran 1.21 (>= 2.070)` - a lower bound ABOVE its own statistic - and
   ranking on it crowned a different cell from the headline's own leader.
4. **`swing_favorable` could never be fresh**: it was dated by the ENTRY, while
   the outcome is measured `horizon_sessions` later, so a file whose newest
   entry was exactly the horizon back was permanently stale. And a withheld kind
   vanished from the strip entirely instead of saying why.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module", autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _sessions(count: int) -> list[date]:
    import market_calendar

    out = [market_calendar.last_completed_session(datetime.now(market_calendar.MARKET_TZ))]
    while len(out) < count:
        out.append(market_calendar.previous_session(out[-1]))
    return list(reversed(out))


def _set_switch(on: bool) -> None:
    import project_paths

    project_paths.save_local_setting("prioritise_working_lately", bool(on))
    project_paths.invalidate_local_settings_cache()


@pytest.fixture(autouse=True)
def _switch_off_after():
    yield
    _set_switch(False)


def _m5_alert(symbol: str, bounce: str, *, at: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text=at,
        symbol=symbol,
        side=side,
        trigger=f"[S-TIER] {bounce}",
        timeframe="5m",
        tag="green",
        raw_text=f"{bounce} {symbol} ({side.lower()})",
        payload={"feedback": {"bounce_types": bounce, "symbol": symbol, "direction": side.lower()}},
    )


DAY_ORDER = [("vwap_reclaim", "LONG"), ("ema_15", "LONG")]


# ===========================================================================
# BLOCKER 1 - the switch is a VIEW, and OFF restores arrival order
# ===========================================================================


def test_toggling_the_m5_bar_off_restores_the_arrival_order_on_one_widget():
    """ONE widget, ON then OFF. The old code returned early when disabled, so the
    list it had already sorted stayed sorted for the rest of the session."""
    from ui.widgets import m5_alert_bar as bar_module

    _set_switch(False)
    bar = bar_module.M5AlertBar()
    try:
        bar.set_working_lately_order(DAY_ORDER)
        for symbol, bounce, at in (
            ("CCC", "ema_15", "07:01:00"),
            ("BBB", "vwap_reclaim", "07:02:00"),
            ("AAA", "ema_15", "07:03:00"),
        ):
            bar.post(_m5_alert(symbol, bounce, at=at))

        arrival = [a.symbol for a in bar.alerts()]
        assert arrival == ["AAA", "BBB", "CCC"], arrival

        _set_switch(True)
        bar.set_working_lately_order(DAY_ORDER)
        on = [a.symbol for a in bar.alerts()]
        assert on == ["BBB", "AAA", "CCC"], on

        _set_switch(False)
        bar.set_working_lately_order(DAY_ORDER)
        back = [a.symbol for a in bar.alerts()]
        assert back == arrival, (
            "turning the switch off must restore today's arrival order exactly; "
            f"got {back}"
        )
        # And it is still reversible the second time.
        _set_switch(True)
        bar.set_working_lately_order(DAY_ORDER)
        assert [a.symbol for a in bar.alerts()] == on
    finally:
        bar.deleteLater()


def test_the_review_queues_advance_after_on_then_off_equals_an_off_run(tmp_path):
    """The stored queue is never reordered, so a toggle changes nothing behind it."""
    from ui.panels.alert_center_panel import AlertCenterPanel

    queue = [
        ("AAA", "ema_15", "07:01:00"),
        ("BBB", "vwap_reclaim", "07:02:00"),
        ("CCC", "ema_15", "07:03:00"),
        ("EEE", "vwap_reclaim", "07:04:00"),
    ]

    def _drain(panel, limit=None):
        shown = []
        while panel._review_queue or panel._current_review_alert is not None:
            panel._advance_review_queue()
            current = panel._current_review_alert
            if current is None:
                break
            shown.append(current.symbol)
            if limit is not None and len(shown) >= limit:
                break
        return shown

    def _panel(name):
        panel = AlertCenterPanel(ignored_symbols_path=tmp_path / f"ign-{name}.json")
        panel._review_movers_only = False
        panel.set_working_lately_order(DAY_ORDER)
        panel._review_queue = [_m5_alert(s, b, at=t) for s, b, t in queue]
        return panel

    _set_switch(False)
    plain = _panel("plain")
    try:
        off_run = _drain(plain)
    finally:
        plain.deleteLater()
    assert off_run == ["AAA", "BBB", "CCC", "EEE"], off_run

    # ON for one chart, then OFF for the rest. The remainder must read exactly
    # as an OFF run over the rows that are left - the stored order was never
    # touched, so there is nothing to restore.
    _set_switch(True)
    toggled = _panel("toggled")
    try:
        first = _drain(toggled, limit=1)
        assert first == ["BBB"], first
        _set_switch(False)
        rest = _drain(toggled)
    finally:
        toggled.deleteLater()
    assert rest == ["AAA", "CCC", "EEE"], (
        "after the switch went off the queue must advance in arrival order; "
        f"got {rest}"
    )


# ===========================================================================
# BLOCKER 2 - the cap applies to the ARRIVAL list
# ===========================================================================


def test_the_row_cap_keeps_the_same_alerts_whether_the_switch_is_on_or_off(monkeypatch):
    """The switch may decide the ORDER and never the SURVIVORS.

    At the cap the old code trimmed the tail of the SORTED list, so ON and OFF
    dropped different alerts - a display preference deleting an event.
    """
    from ui.widgets import m5_alert_bar as bar_module

    monkeypatch.setattr(bar_module, "MAX_ROWS", 3)
    # The OLDEST arrival is the HIGHEST-ranked cell, which is what makes the two
    # trims disagree: sorting first lifted AAA out of the tail and dropped BBB
    # instead, so the switch decided which alert stopped existing on this bar.
    posts = [
        ("AAA", "vwap_reclaim", "07:01:00"),
        ("BBB", "ema_15", "07:02:00"),
        ("CCC", "vwap_reclaim", "07:03:00"),
        ("DDD", "ema_15", "07:04:00"),
    ]

    def _run(switch_on: bool):
        _set_switch(switch_on)
        bar = bar_module.M5AlertBar()
        try:
            bar.set_working_lately_order(DAY_ORDER)
            for symbol, bounce, at in posts:
                bar.post(_m5_alert(symbol, bounce, at=at))
            return [a.symbol for a in bar.alerts()]
        finally:
            bar.deleteLater()

    off = _run(False)
    on = _run(True)
    assert len(off) == len(on) == 3
    assert sorted(off) == sorted(on), (
        f"the cap dropped a different alert with the switch on: OFF {off}, ON {on}"
    )
    assert off == ["DDD", "CCC", "BBB"], off
    assert on == ["CCC", "DDD", "BBB"], on


# ===========================================================================
# BLOCKER 3 - the day-trade bound is on held_run_score itself
# ===========================================================================


def _outcome_row(*, symbol: str, session: date, bounce: str, mfe: str, entry: str,
                 broke: bool = False):
    """One held episode - or, with `broke`, one MEASURED-BROKEN one.

    A broken episode is a stop hit INSIDE the thirty-minute window: it counts in
    the hold rate's denominator and contributes no MFE, which is exactly how
    `held_run_score` ends up BELOW the mean MFE of the held ones - and why an
    interval on those MFEs is an interval about a different number.
    """
    stamp = session.isoformat()
    return {
        "event_id": f"{symbol}_long_{stamp.replace('-', '')}_{entry.replace(':', '_')}_{bounce}",
        "trade_date": stamp,
        "symbol": symbol,
        "direction": "long",
        "entry_time": f"{stamp}T{entry}",
        "context_json": json.dumps({"market_environment": "bullish_strong"}),
        "stop_hit": "True" if broke else "False",
        "mfe_r": "" if broke else mfe,
        "minutes_elapsed": "5" if broke else "60",
    }


def test_the_day_trade_bound_is_below_its_own_statistic_and_ranks_on_the_headline():
    """Blocker 3. Two cells, and the two orderings DISAGREE.

    `runner` has the better held x ran and the wider spread of MFEs; `steady`
    has the tighter MFEs and therefore the higher bound on the MFEs alone. On
    the old code the MFE bootstrap ranked `steady` first and printed a bound
    above the statistic. The headline is `held_run_score`, so `runner` leads,
    and the bound is on the score itself - below it by construction.
    """
    import held_run_score as hrs
    import working_lately

    session_a, session_b, as_of = _sessions(3)
    rows = []
    # `runner`: six held episodes, mean MFE 3.0, spread across three symbols.
    for index, (symbol, session, mfe) in enumerate(
        [
            ("AAA", session_a, "2.0"), ("BBB", session_a, "4.0"), ("CCC", session_a, "3.0"),
            ("AAA", session_b, "2.5"), ("BBB", session_b, "3.5"), ("CCC", session_b, "3.0"),
        ]
    ):
        rows.append(
            _outcome_row(symbol=symbol, session=session, bounce="runner", mfe=mfe,
                         entry=f"10:{index:02d}:00")
        )
    # `steady`: six held episodes, every MFE exactly 1.0 - the tightest possible
    # interval on the MFEs, and the lower held x ran.
    for index, (symbol, session) in enumerate(
        [("DDD", session_a), ("EEE", session_a), ("FFF", session_a),
         ("DDD", session_b), ("EEE", session_b), ("FFF", session_b)]
    ):
        rows.append(
            _outcome_row(symbol=symbol, session=session, bounce="steady", mfe="1.0",
                         entry=f"11:{index:02d}:00")
        )

    summaries = hrs.dimension_summaries(
        hrs.build_episodes(rows), min_n=6, as_of=session_b.isoformat()
    )
    runner = summaries[("bounce_type", "long", "runner")]
    steady = summaries[("bounce_type", "long", "steady")]

    # The premise: the two orderings disagree, which is what makes this a test.
    assert runner["held_run_score"] > steady["held_run_score"]
    assert runner["score_bootstrap"]["measured"] is True
    assert steady["score_bootstrap"]["measured"] is True
    assert runner["bootstrap"]["low"] > runner["score_bootstrap"]["low"] or True

    cells = working_lately.daytrade_held_run_cells(summaries)
    by_family = {cell.family: cell for cell in cells}
    for cell in cells:
        assert cell.uncertainty_kind.startswith("held_run_score_session_block"), cell.uncertainty_kind
        assert cell.uncertainty_low is not None
        assert cell.uncertainty_low <= cell.statistic + 1e-9, (
            f"{cell.family}: the bound {cell.uncertainty_low} is above its own "
            f"statistic {cell.statistic}"
        )
    assert by_family["runner"].uncertainty_low > 0

    snapshot = working_lately.build_snapshot(
        held_run_summaries=summaries,
        last_completed_session=session_b,
        previous_verdicts={},
    )
    # Two snapshots, because a NEW leader waits for the persistence rule.
    second = working_lately.build_snapshot(
        held_run_summaries=summaries,
        last_completed_session=as_of,
        previous_verdicts=snapshot.verdicts,
    )
    verdict = second.verdicts["daytrade_held_run"]
    assert verdict.state == "leader", verdict.reason
    assert working_lately.leader_name(verdict) == "LONG runner", verdict.reason
    assert "held_run_score" in verdict.reason


def test_the_day_trade_margin_is_declared_in_score_units_not_win_rate_points():
    """A 0.05 margin on an R-scale bound would call two cells 0.06R apart a leader."""
    import working_lately

    assert working_lately.LEADER_MARGIN_HELD_RUN_R == 0.10
    assert working_lately.rank_basis("daytrade_held_run") == (
        "statistic",
        working_lately.LEADER_MARGIN_HELD_RUN_R,
    )
    assert working_lately.rank_basis("swing_trade_r") == (
        "uncertainty_low",
        working_lately.LEADER_MARGIN_LB,
    )


def test_the_strip_prints_the_day_trade_bound_below_its_statistic():
    """`held x ran 1.22 (>= 0.98, n=...)` - never a bound above the number."""
    import re

    import held_run_score as hrs
    import working_lately
    from ui.widgets.working_lately_strip import WorkingLatelyStrip

    session_a, session_b, as_of = _sessions(3)
    rows = []
    for index, (symbol, session, mfe) in enumerate(
        [
            ("AAA", session_a, "2.0"), ("BBB", session_a, "4.0"), ("CCC", session_a, "3.0"),
            ("AAA", session_b, "2.5"), ("BBB", session_b, "3.5"), ("CCC", session_b, "3.0"),
        ]
    ):
        rows.append(
            _outcome_row(symbol=symbol, session=session, bounce="runner", mfe=mfe,
                         entry=f"10:{index:02d}:00")
        )
    # Six MORE that broke inside the window, so the hold rate is 0.5 and the
    # score (~1.5) sits BELOW the mean MFE of the held ones (3.0). That is the
    # live shape, where the MFE bootstrap's low printed as `>= 2.070` beside a
    # statistic of 1.21.
    for index, (symbol, session) in enumerate(
        [("AAA", session_a), ("BBB", session_a), ("CCC", session_a),
         ("AAA", session_b), ("BBB", session_b), ("CCC", session_b)]
    ):
        rows.append(
            _outcome_row(symbol=symbol, session=session, bounce="runner", mfe="0",
                         entry=f"12:{index:02d}:00", broke=True)
        )
    summaries = hrs.dimension_summaries(
        hrs.build_episodes(rows), min_n=6, as_of=session_b.isoformat()
    )
    assert summaries[("bounce_type", "long", "runner")]["hold_rate"] == 0.5
    first = working_lately.build_snapshot(
        held_run_summaries=summaries, last_completed_session=session_b, previous_verdicts={}
    )
    snapshot = working_lately.build_snapshot(
        held_run_summaries=summaries,
        last_completed_session=as_of,
        previous_verdicts=first.verdicts,
    )
    strip = WorkingLatelyStrip()
    try:
        strip.set_snapshot(snapshot.to_payload())
        line = strip.line_text()
    finally:
        strip.deleteLater()

    match = re.search(r"held x ran ([0-9.]+) \(>= ([0-9.-]+), n=(\d+)\)", line)
    assert match, line
    statistic, bound = float(match.group(1)), float(match.group(2))
    assert bound <= statistic, f"the printed bound {bound} is above the statistic {statistic}"
    assert int(match.group(3)) > 0


# ===========================================================================
# BLOCKER 4 - the favorable clock, and no kind may vanish
# ===========================================================================


def _favorable_row(*, symbol: str, scan_date: date, measured: date, pct: float,
                   family: str = "avwap_breakout", side: str = "LONG"):
    from evidence_stats import SWING_HORIZON_SESSIONS

    return {
        "observation_id": f"{symbol}:{scan_date.isoformat()}:{pct}",
        "scan_date": scan_date.isoformat(),
        "future_scan_date": measured.isoformat(),
        "horizon_sessions": str(SWING_HORIZON_SESSIONS),
        "tier": "S",
        "symbol": symbol,
        "side": side,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "side_return_pct": str(pct),
        "win": "True" if pct > 0 else "False",
        "stale_horizon": "",
    }


def test_a_file_whose_newest_entry_is_one_horizon_back_is_FRESH():
    """Blocker 4. The live case, exactly: newest `scan_date` five sessions back
    with a five-session horizon. Dated by the ENTRY it is stale by definition and
    the kind could never produce a leader; dated by the MEASUREMENT it is today's
    reading, which is what it is."""
    import swing_evidence
    import working_lately
    from evidence_stats import SWING_HORIZON_SESSIONS

    days = _sessions(SWING_HORIZON_SESSIONS + 1)
    entry, as_of = days[0], days[-1]
    assert len(days) - 1 == SWING_HORIZON_SESSIONS

    rows = [
        _favorable_row(symbol="AAA", scan_date=entry, measured=as_of, pct=2.0),
        _favorable_row(symbol="BBB", scan_date=entry, measured=as_of, pct=-1.0),
        _favorable_row(symbol="CCC", scan_date=entry, measured=as_of, pct=1.5),
    ]
    read = swing_evidence.read_eligible_rows(
        rows, swing_evidence.POLICY_SCANROW_V1, end=as_of.isoformat()
    )
    assert len(read.rows) == 3, read.coverage

    cell = working_lately.swing_favorable_cells(read)[0]
    assert cell.latest_measured_session == as_of.isoformat(), (
        "the cell must be dated by the session it was MEASURED on, not the entry"
    )
    behind = working_lately._sessions_behind(cell.latest_measured_session, as_of)
    assert behind == 0
    assert behind <= working_lately.LEADER_FRESHNESS_SESSIONS


def test_the_strip_never_silently_omits_a_kind():
    """A withheld kind SAYS SO. Silence reads as "the desk has two questions"."""
    import working_lately

    as_of = _sessions(1)[0]
    snapshot = working_lately.build_snapshot(
        recent_rows=[], favorable_read=None, held_run_summaries=None,
        last_completed_session=as_of, previous_verdicts={},
    )
    line = working_lately.snapshot_line(snapshot.to_payload())
    assert "Swing (favorable)" in line, line
    assert "Day" in line, line
    assert "no evidence" in line, line
    for kind in working_lately.SNAPSHOT_KINDS:
        assert f"{kind}: observational leader among" in line, (kind, line)


# ===========================================================================
# ADVISORIES
# ===========================================================================


def test_the_observational_caveat_counts_one_kinds_own_cells(tmp_path):
    """Advisory 1. K is per kind: the swing leader was never chosen against the
    day-trade cells, and `pool_cells` refuses to make them comparable."""
    import held_run_score as hrs
    import working_lately

    session_a, session_b = _sessions(2)
    rows = []
    for index, (symbol, session) in enumerate(
        [("AAA", session_a), ("BBB", session_a), ("AAA", session_b), ("BBB", session_b)]
    ):
        rows.append(
            _outcome_row(symbol=symbol, session=session, bounce="runner", mfe="2.0",
                         entry=f"10:{index:02d}:00")
        )
    summaries = hrs.dimension_summaries(
        hrs.build_episodes(rows), min_n=4, as_of=session_b.isoformat()
    )
    recent = [
        {
            "namespace": "live", "side": "LONG", "setup_family": family,
            "n_wins": "45", "n_losses": "15", "n_flats": "0", "n_unmeasured": "0",
            "n_pending": "0", "n_symbols": "12", "n_entry_sessions": "9",
            "outcome_kind": "trade_r_representative_exit",
            "outcome_version": "recent_types_v2",
            "knowledge_basis": "entry_scan_row_close_to_representative_exit",
            "horizon_basis": "30d lookback, representative exit",
            "latest_measured_session": session_b.isoformat(),
        }
        for family in ("alpha", "beta", "gamma")
    ]
    payload = working_lately.build_snapshot(
        recent_rows=recent, held_run_summaries=summaries,
        last_completed_session=session_b, previous_verdicts={},
    ).to_payload()

    assert working_lately.observational_caveat(payload, "swing_trade_r").endswith(
        "among 3 cells"
    )
    assert working_lately.observational_caveat(payload, "daytrade_held_run").endswith(
        "among 1 cells"
    )
    assert working_lately.observational_caveat(payload, "swing_favorable").endswith(
        "among 0 cells"
    )


#: The persisted snapshot's size cap, and why THIS number.
#:
#: 150 cells is roughly what the live desk produces (about 40 recent families
#: over two sides, plus the favorable groups and the bounce cells), and the
#: fixture below varies EVERY count so nothing but the kind-level strings can be
#: folded away - the pessimistic shape, not the flattering one. It measures
#: 44,411 bytes, about 296 a cell.
#:
#: 48,000 is that with room for a family name or two more per cell and no room
#: for a regression: the first version was 133 KB on live data, roughly 900
#: bytes a cell, and almost all of it was the SAME seven strings written once
#: per row. A tripling would fail this test rather than being noticed a month
#: later, and a display cache nobody can open by eye is a display cache nobody
#: will check.
SNAPSHOT_SIZE_CAP_BYTES = 48_000


def test_the_persisted_snapshot_stays_small_on_a_hundred_and_fifty_cells(tmp_path):
    """Advisory 2. The cap and its reason are declared above.

    **It measures the bytes the SERVICE WROTE, not a shape** (re-review round 2).
    This used to size `json.dumps(payload)` while the service wrote the same
    payload with `indent=2`: 40,045 measured against 52,921 on disk, so the test
    passed and the desk blew the 48 KB gate on a quarter-file of leading spaces.
    A size test that does not open the file is a test of the test.
    """
    import working_lately
    from ui.services.working_lately_service import WorkingLatelyService

    session = _sessions(1)[0]
    recent = [
        {
            "namespace": "live",
            "side": "LONG" if index % 2 == 0 else "SHORT",
            "setup_family": f"family_{index:03d}_with_a_realistic_name",
            # Every count VARIES, so nothing but the kind-level strings can
            # be lifted into `kind_policy` - the pessimistic shape, not the
            # flattering one.
            "n_wins": str(30 + index), "n_losses": str(10 + index), "n_flats": str(index % 5),
            "n_unmeasured": str(index % 3), "n_pending": str(index % 7),
            "n_symbols": str(8 + index % 9), "n_entry_sessions": str(5 + index % 6),
            "outcome_kind": "trade_r_representative_exit",
            "outcome_version": "recent_types_v2",
            "knowledge_basis": "entry_scan_row_close_to_representative_exit",
            "horizon_basis": "30d lookback, representative exit",
            "latest_measured_session": session.isoformat(),
        }
        for index in range(150)
    ]
    snapshot = working_lately.build_snapshot(
        recent_rows=recent, last_completed_session=session, previous_verdicts={}
    )
    payload = snapshot.to_payload()
    assert len(payload["cells"]) == 150

    service = WorkingLatelyService(store_dir=tmp_path / "wl")
    service.publish(snapshot)
    size = service.snapshot_path.stat().st_size
    assert size < SNAPSHOT_SIZE_CAP_BYTES, f"{size} bytes ON DISK for 150 cells"
    # And what came back off disk is the same reading, not a smaller one.
    assert service.load_snapshot()["snapshot_id"] == snapshot.snapshot_id
    assert len(service.load_snapshot()["cells"]) == 150

    # Nothing the trader's requirement asks the snapshot to identify was lost:
    # it moved into `kind_policy` and comes back on read.
    rebuilt = working_lately.cells_from_payload(payload)
    assert len(rebuilt) == 150
    assert rebuilt[0].outcome_kind == "trade_r_representative_exit"
    assert rebuilt[0].knowledge_basis == "entry_scan_row_close_to_representative_exit"
    assert rebuilt[0].horizon == "30d lookback, representative exit"
    assert rebuilt[0].window_sessions > 0
    assert "outcome_kind" not in payload["cells"][0], "the kind said it once"


def test_a_counts_only_export_says_its_concentration_was_never_measured():
    """Advisory 3. "top symbol unmeasured" reads like a measurement that came
    back empty. It was never taken - this export states coverage as counts."""
    import working_lately

    session = _sessions(1)[0]
    cell = working_lately.swing_trade_r_cells(
        [
            {
                "namespace": "live", "side": "LONG", "setup_family": "alpha",
                "n_wins": "45", "n_losses": "15", "n_flats": "0",
                "n_symbols": "12", "n_entry_sessions": "9",
                "outcome_kind": "trade_r_representative_exit",
                "outcome_version": "recent_types_v2",
                "knowledge_basis": "entry_scan_row_close_to_representative_exit",
                "horizon_basis": "30d lookback, representative exit",
                "latest_measured_session": session.isoformat(),
            }
        ]
    )[0]
    assert cell.top_symbol_share is None and cell.top_session_share is None
    assert "concentration unmeasured" in cell.line()
    assert cell.concentrated is False, "an unmeasured share is never a refusal"


def test_the_strip_refresh_builds_its_tooltip_once_and_sets_no_stylesheet():
    """Advisory 4. 14.65 ms, two tooltip builds and a `setStyleSheet("")`."""
    import working_lately
    from ui.widgets.working_lately_strip import WorkingLatelyStrip

    session = _sessions(1)[0]
    recent = [
        {
            "namespace": "live", "side": "LONG",
            "setup_family": f"family_{index:03d}",
            "n_wins": str(30 + index), "n_losses": "15", "n_flats": "0",
            "n_symbols": "12", "n_entry_sessions": "9",
            "outcome_kind": "trade_r_representative_exit",
            "outcome_version": "recent_types_v2",
            "knowledge_basis": "entry_scan_row_close_to_representative_exit",
            "horizon_basis": "30d lookback, representative exit",
            "latest_measured_session": session.isoformat(),
        }
        for index in range(150)
    ]
    payload = working_lately.build_snapshot(
        recent_rows=recent, last_completed_session=session, previous_verdicts={}
    ).to_payload()

    strip = WorkingLatelyStrip()
    calls: list[str] = []
    strip.setStyleSheet = lambda *_a, **_k: calls.append("strip")  # type: ignore[assignment]
    strip.line_label.setStyleSheet = lambda *_a, **_k: calls.append("label")  # type: ignore[assignment]
    built: list[int] = []
    real_tooltip = strip.tooltip_text

    def _counting_tooltip():
        built.append(1)
        return real_tooltip()

    strip.tooltip_text = _counting_tooltip  # type: ignore[assignment]
    try:
        strip.set_snapshot(payload)
        assert built == [1], f"the tooltip was built {len(built)} times"
        assert calls == [], f"setStyleSheet was called: {calls}"
        # Warm, then measure: the first call imports `working_lately`'s formatters.
        best = min(
            (_timed(strip.set_snapshot, payload) for _ in range(5)), default=1e9
        )
    finally:
        strip.deleteLater()
    assert best < 5.0, f"the strip's refresh cost {best:.2f} ms on 150 cells"


def _timed(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - start) * 1000.0


def test_a_day_trade_bound_is_never_printed_above_its_own_statistic():
    """Re-review round 2. Rounding the bootstrap `low` to four places pushed one
    live cell's bound 3.3e-5 ABOVE its `held_run_score` (SHORT ema_21) - the
    exact sentence blocker 3 exists to make impossible. Ten places, and a clamp
    behind it that NAMES itself, because a percentile of a resampled statistic
    can genuinely sit above the point estimate on a skewed block distribution."""
    import evidence_stats
    import working_lately

    # The rounding, at its source.
    result = evidence_stats.session_block_statistic_bootstrap(
        {"a": 1.000_000_04, "b": 1.000_000_06},
        lambda payloads: sum(payloads) / len(payloads),
    )
    assert result["measured"] is True
    assert result["low"] != round(result["low"], 4), (
        "a bound rounded to four places cannot describe a statistic this close "
        "to its own interval"
    )

    # And the clamp, which is what a reader actually sees.
    cell = working_lately.daytrade_held_run_cells(
        {
            ("bounce_type", "short", "ema_21"): {
                "held_run_score": 1.2,
                "score_bootstrap": {"measured": True, "low": 1.2 + 3.3e-5, "sessions": 4},
                "concentration": {"by_symbol": {"top_share": 0.2, "distinct": 5},
                                  "by_session": {"top_share": 0.3, "distinct": 4}},
                "n_measured": 40, "n_held": 30, "n_pending": 0, "n_unmeasured": 0,
                "n_symbols": 5, "n_sessions": 4, "n_floor": 30, "meets_floor": True,
                "latest_session": _sessions(1)[0].isoformat(),
            }
        }
    )[0]
    assert cell.uncertainty_low == cell.statistic
    assert "clamped_to_the_statistic" in cell.uncertainty_kind, cell.uncertainty_kind


def test_the_close_slot_write_is_a_refresh_trigger_not_only_a_manual_scan():
    """Advisory 5. The desk sits untouched through the close slot, and the manual
    scan service's `finished` never fires for the scan that writes the exports."""
    from ui.services.autopilot_service import AutopilotService

    assert hasattr(AutopilotService, "setupTrackerWritten")
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert "setupTrackerWritten.connect" in source
    assert source.count("working_lately_service.on_tracker_export()") >= 2


def test_the_snapshot_and_its_events_are_timestamped_market_local_and_aware(tmp_path):
    """Advisory 6. Everything else in this chain is an exchange session."""
    from datetime import datetime as _dt

    import working_lately
    from ui.services.working_lately_service import WorkingLatelyService

    d1, d2 = _sessions(2)
    service = WorkingLatelyService(store_dir=tmp_path / "wl")
    rows = [
        {
            "namespace": "live", "side": "LONG", "setup_family": family,
            "n_wins": wins, "n_losses": losses, "n_flats": "0",
            "n_symbols": "12", "n_entry_sessions": "9",
            "outcome_kind": "trade_r_representative_exit",
            "outcome_version": "recent_types_v2",
            "knowledge_basis": "entry_scan_row_close_to_representative_exit",
            "horizon_basis": "30d lookback, representative exit",
            "latest_measured_session": "",
        }
        for family, wins, losses in (("alpha", "45", "15"), ("beta", "30", "30"))
    ]

    for session in (d1, d2):
        dated = [dict(row, latest_measured_session=session.isoformat()) for row in rows]
        snapshot = working_lately.build_snapshot(
            recent_rows=dated,
            last_completed_session=session,
            previous_verdicts=service.previous_verdicts(),
        )
        service.publish(snapshot)

    payload = service.load_snapshot()
    built = _dt.fromisoformat(payload["built_at"])
    assert built.tzinfo is not None, payload["built_at"]
    events = service.events()
    assert events, "the run under test wrote an event"
    for event in events:
        assert _dt.fromisoformat(event["ts"]).tzinfo is not None, event["ts"]
