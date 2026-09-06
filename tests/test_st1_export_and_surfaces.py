"""Packet ST1, builder's half: the export seam and the three describe() lines.

The tester's `tests/test_st1_outcome_clock.py` pins the BUILDERS and the reader.
These pin what the packet asks for around them:

* the v2 file is written in the SAME export pass that writes the tier outcomes,
  from the caller's own completed bars, and a failure there never costs a v1
  export (ST1 item 2);
* every surface that shows the family rate also says WHAT it is - the Master
  AVWAP setups panel, the setup docs and the AWAY digest (ST1 item 3).

Nothing here opens a live store, fetches, or writes outside `tmp_path`.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from master_avwap_lib import legacy  # noqa: E402

JUNE_SESSIONS = (
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-05",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
)


def _scan_row(symbol: str, scan_date: str, close: float) -> dict:
    return {
        "run_id": f"run-{scan_date}",
        "run_timestamp": f"{scan_date}T13:00:00",
        "run_date": scan_date,
        "watchlist_label": "swing_longs",
        "symbol": symbol,
        "side": "LONG",
        "last_trade_date": scan_date,
        "last_close": float(close),
        "priority_bucket": "favorite_setup",
        "priority_score": 120.0,
        "setup_family": "avwap_breakout",
        "favorite_zone": "AVWAPE to UPPER_1",
        "current_band_zone": "AVWAPE to UPPER_1",
        "trend_20d": "UP",
        "assigned_tier": "S",
    }


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _scan_row("AAA", day, 100.0 + index)
            for index, day in enumerate(JUNE_SESSIONS)
        ]
    )


def _export_paths(tmp_path: Path) -> dict:
    return {
        "tier_list_path": tmp_path / "tier_list.csv",
        "tier_outcomes_path": tmp_path / "tier_outcomes.csv",
        "tier_performance_path": tmp_path / "tier_performance.csv",
        "tier_catch_rate_path": tmp_path / "tier_catch.csv",
        "session_horizon_path": tmp_path / "session_horizon.csv",
    }


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_the_export_writes_the_v2_file_beside_the_tier_outcomes(tmp_path, monkeypatch):
    """One pass, five files, and the v2 rows carry their own clock.

    `sessions_spanned == horizon_sessions` on every measured row is the whole
    point of the file: v1's column is whatever the symbol's scan membership made
    it, and v2's is the declared horizon by construction.
    """
    import market_calendar

    paths = _export_paths(tmp_path)
    closes = {
        date.fromisoformat(day): 100.0 + index * 2.0
        for index, day in enumerate(JUNE_SESSIONS)
    }
    monkeypatch.setattr(market_calendar, "last_completed_session", lambda now: date(2026, 6, 30))

    result = legacy.export_bot_tier_tracker_views(
        history_df=_history(),
        closes_for=lambda symbol: closes,
        **paths,
    )

    assert paths["session_horizon_path"].exists()
    rows = _read(paths["session_horizon_path"])
    assert rows, "the v2 export wrote no rows"
    assert result["session_horizon_outcome_count"] == len(rows)
    assert result["session_horizon_measured_count"] >= 1
    # Both counts reach the caller under their own names, so the log line and
    # the run result can never report a re-scan as a duplicate.
    assert result["session_horizon_dropped_duplicates"] == 0
    assert result["session_horizon_collapsed_same_session"] == 0

    for row in rows:
        assert row["outcome_kind"] == "favorable_direction_session_v2"
        assert row["knowledge_basis"] == "entry_session_close_to_target_session_close"
        if str(row["measured"]).strip().lower() == "true":
            assert row["sessions_spanned"] == row["horizon_sessions"]
            assert row["unmeasured_reason"] == ""
            assert row["target_close"] not in ("", None)
        else:
            assert row["unmeasured_reason"], "an unmeasured row must say why"

    # The v1 file is untouched by any of this and still carries its own column.
    v1_rows = _read(paths["tier_outcomes_path"])
    assert v1_rows
    assert {row["outcome_kind"] for row in v1_rows} == {"favorable_direction_scanrow_v1"}

    # The two files join on `observation_id`.
    v1_ids = {row["observation_id"] for row in v1_rows}
    v2_ids = {row["observation_id"] for row in rows}
    assert v1_ids and v1_ids.issubset(v2_ids)


def test_the_export_never_fetches_and_survives_a_closes_lookup_that_raises(tmp_path, monkeypatch):
    """A broken `closes_for` costs the v2 rows their measurement, never the v1 files.

    The evidence-store rule, applied to an export: a shadow file may never cost
    the thing it describes.
    """
    import market_calendar

    paths = _export_paths(tmp_path)
    monkeypatch.setattr(market_calendar, "last_completed_session", lambda now: date(2026, 6, 30))

    def exploding_closes(symbol):
        raise RuntimeError("no bars, and certainly no network")

    result = legacy.export_bot_tier_tracker_views(
        history_df=_history(), closes_for=exploding_closes, **paths
    )

    assert result["tier_outcome_count"] > 0
    assert paths["tier_outcomes_path"].exists()
    rows = _read(paths["session_horizon_path"])
    assert rows
    assert all(str(row["measured"]).strip().lower() == "false" for row in rows)
    assert {row["unmeasured_reason"] for row in rows} == {"no_bar_for_target_session"}


def test_a_failed_v2_write_never_costs_the_v1_exports(tmp_path, monkeypatch):
    """The guard, proven by breaking the builder the export calls."""
    import market_calendar

    from master_avwap_lib import session_horizon_outcomes

    paths = _export_paths(tmp_path)
    monkeypatch.setattr(market_calendar, "last_completed_session", lambda now: date(2026, 6, 30))
    monkeypatch.setattr(
        session_horizon_outcomes,
        "build_session_horizon_observation_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = legacy.export_bot_tier_tracker_views(
        history_df=_history(), closes_for=lambda symbol: None, **paths
    )

    assert result["tier_outcome_count"] > 0
    assert result["session_horizon_outcome_count"] == 0
    assert paths["tier_outcomes_path"].exists()


def test_two_scans_of_one_symbol_on_one_day_are_COLLAPSED_never_duplicates():
    """The desk ran FIFTEEN scans on 2026-08-31. None of them is a duplicate.

    Row identity is `_scan_factor_row_id` - `symbol:scan_date:run_id` - so two
    scans of AAA on 2026-06-01 under different run ids are two SCAN ROWS, not
    one recorded twice. Keying de-duplication on `(symbol, scan_date)` reported
    475,492 duplicates against 109,584 rows on the live history, where the truly
    repeated `scan_row_id`s numbered 75.

    The trader's lead, 2026-09-06, on what to do with them: the MEASUREMENT is
    the same number for every scan that day - entry-session close to
    target-session close - so the file keeps ONE row per
    `(symbol, side, scan_date, horizon)`, the session's LAST scan row, and says
    how many looks stand behind it. **`collapsed_same_session`, never
    `dropped_duplicates`**: the second number stays the count of a real input
    defect, and 127.5 MB of re-scans per export was the cost of confusing them.
    """
    from master_avwap_lib.session_horizon_outcomes import (
        build_session_horizon_observation_rows,
    )

    morning = _scan_row("AAA", "2026-06-01", 100.0)
    afternoon = dict(morning)
    afternoon["run_id"] = "run-2026-06-01-afternoon"
    afternoon["run_timestamp"] = "2026-06-01T16:00:00"
    afternoon["last_close"] = 102.0
    history = pd.DataFrame([morning, afternoon])

    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: {date(2026, 6, 1): 100.0, date(2026, 6, 2): 110.0},
        horizons=(1,),
        last_completed_session=date(2026, 6, 30),
    )

    # NOT a duplicate. That number is reserved for a repeated `scan_row_id`.
    assert built.dropped_duplicates == 0
    assert built.collapsed_same_session == 1
    assert len(built.rows) == 1
    row = built.rows[0]
    assert row["collapsed_same_session"] == 1
    # The session's LAST scan row is the representative - v1's choice too.
    assert row["scan_row_id"].endswith("run-2026-06-01-afternoon")
    assert row["target_session"] == "2026-06-02"


def test_the_collapsed_row_is_the_one_v1_names_so_the_files_join_1_to_1():
    """v1's `observation_id` IS the v2 row's, on a day the desk scanned twice.

    Both keep the session's last scan row off the same sort, so the join is one
    to one rather than one to many - which is what makes `outcome_kind` a
    comparison between two measurements of the same decision.
    """
    from master_avwap_lib.session_horizon_outcomes import (
        build_session_horizon_observation_rows,
    )

    rows = []
    for index, day in enumerate(JUNE_SESSIONS[:6]):
        rows.append(_scan_row("JOIN", day, 100.0 + index))
        second = dict(rows[-1])
        second["run_id"] = f"run-{day}-second"
        second["run_timestamp"] = f"{day}T16:00:00"
        second["last_close"] = 100.0 + index
        rows.append(second)
    history = pd.DataFrame(rows)
    closes = {
        date.fromisoformat(day): 100.0 + index for index, day in enumerate(JUNE_SESSIONS)
    }

    v1 = legacy.build_scan_factor_observation_rows(history, horizons=(1,))
    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: closes,
        horizons=(1,),
        last_completed_session=date(2026, 6, 30),
    )

    v1_ids = [row["observation_id"] for row in v1]
    v2_ids = [row["observation_id"] for row in built.rows]
    assert len(v1_ids) == len(set(v1_ids))
    assert len(v2_ids) == len(set(v2_ids))
    # ONE TO ONE on the sessions both cover: v1 has no row for the last session
    # (no later scan row to compare against), v2 has no row for a target that
    # has not closed - neither is a join failure.
    assert set(v1_ids).issubset(set(v2_ids))
    assert built.collapsed_same_session == 6
    assert all(row["collapsed_same_session"] == 1 for row in built.rows)


def test_the_build_is_a_rolling_window_and_counts_what_it_left_out():
    """Scan dates older than the declared window are excluded, and COUNTED.

    Unbounded, this rewrote the whole feature history on every scan to change
    nothing outside the newest sessions, because a settled row's target close
    does not move. The window is declared, and what falls outside it is reported
    rather than silently missing.
    """
    from evidence_stats import LATELY_SESSIONS
    from master_avwap_lib.session_horizon_outcomes import (
        BUILD_WINDOW_SESSIONS,
        build_session_horizon_observation_rows,
    )

    # 1.5x the widest window any reader uses - the lead's number, 2026-09-06.
    assert BUILD_WINDOW_SESSIONS == 30
    assert BUILD_WINDOW_SESSIONS > LATELY_SESSIONS

    history = pd.DataFrame(
        [
            _scan_row("OLD", "2026-01-05", 100.0),
            _scan_row("NEW", "2026-06-01", 100.0),
        ]
    )
    closes = {date(2026, 1, 5): 100.0, date(2026, 6, 1): 100.0, date(2026, 6, 2): 110.0}
    built = build_session_horizon_observation_rows(
        history,
        lambda symbol: closes,
        horizons=(1,),
        last_completed_session=date(2026, 6, 30),
    )

    assert {row["symbol"] for row in built.rows} == {"NEW"}
    assert built.excluded["outside_build_window"] == 1

    # `None` builds everything, for a caller that wants the whole history.
    everything = build_session_horizon_observation_rows(
        history,
        lambda symbol: closes,
        horizons=(1,),
        last_completed_session=date(2026, 6, 30),
        window_sessions=None,
    )
    assert {row["symbol"] for row in everything.rows} == {"NEW", "OLD"}
    assert everything.excluded["outside_build_window"] == 0


def test_the_daily_frame_lookup_reads_closes_and_never_fetches():
    """`closes_from_daily_frames` is the desk's `closes_for`: frames in, dict out."""
    from master_avwap_lib.session_horizon_outcomes import closes_from_daily_frames

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "close": [100.0, 101.5],
        }
    )
    closes_for = closes_from_daily_frames({"AAA": frame})
    assert closes_for("aaa") == {date(2026, 6, 1): 100.0, date(2026, 6, 2): 101.5}
    # A symbol with no frame is "we did not look", not "there was no bar".
    assert closes_for("ZZZ") is None


# ---------------------------------------------------------------------------
# The describe() line, on all three surfaces
# ---------------------------------------------------------------------------
def _outcome_row(index: int, *, stale: str = "False") -> dict:
    """One tier-outcome row, shaped like the file's."""
    return {
        "observation_id": f"OBS{index}:5",
        "scan_row_id": f"OBS{index}",
        "scan_date": "2026-06-03",
        "future_scan_date": "2026-06-10",
        "horizon_sessions": 5,
        "tier": "S",
        "tier_source": "assigned",
        "symbol": f"SYM{index}",
        "side": "LONG",
        "setup_family": "avwap_breakout",
        "entry_close": 100.0,
        "future_close": 105.0,
        "raw_return_pct": 5.0,
        "side_return_pct": 5.0,
        "win": "True",
        "spy_forward_return_pct": "",
        "spy_relative_side_return_pct": "",
        "sessions_spanned": 18 if stale.lower() == "true" else 5,
        "stale_horizon": stale,
        "positive_scan_factor_match_count": 0,
        "positive_scan_factor_matches": "",
        "outcome_kind": "favorable_direction_scanrow_v1",
    }


def _write_v1_csv(path: Path, *, stale_rows: int = 1) -> None:
    columns = list(legacy.TIER_OUTCOME_COLUMNS)
    rows = [
        _outcome_row(index, stale="True" if index < stale_rows else "False")
        for index in range(6)
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def test_the_setup_docs_surface_says_what_the_rate_is(tmp_path, monkeypatch):
    import setup_docs

    path = tmp_path / "master_avwap_tier_outcomes.csv"
    _write_v1_csv(path)
    setup_docs.clear_family_outcome_cache()
    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: path)
    monkeypatch.setattr(setup_docs, "_family_outcomes_window", lambda: ("2026-06-01", "2026-06-30"))
    try:
        rows = setup_docs.family_headline_rows()
        line = setup_docs.family_record_coverage_line()
    finally:
        setup_docs.clear_family_outcome_cache()

    assert rows["avwap_breakout"]["n"] == 5
    assert "favorable_direction_scanrow_v1" in line
    # The horizon is stated IN ITS OWN UNIT - the packet's whole point.
    assert "5 scan rows" in line
    assert "5 eligible" in line
    assert "stale_horizon 1" in line


def test_the_away_digest_says_what_it_ranked_on():
    import autopilot_core

    payload = {
        "swing_picks": [{"symbol": "AAA", "side": "LONG", "family": "avwap_breakout"}],
        "swing_family_records": {"avwap_breakout": {"wins": 5, "losses": 1}},
        "swing_family_record_line": (
            "favorable_direction_scanrow_v1 over 5 scan rows, last 20 sessions "
            "(2026-06-02..2026-06-30): 5 eligible / 0 pending / 1 excluded (stale_horizon 1)"
        ),
    }
    report = autopilot_core.render_away_report(payload)
    assert "Ranked on: favorable_direction_scanrow_v1 over 5 scan rows" in report

    # No picks, no claim: the line describes a ranking that happened.
    empty = autopilot_core.render_away_report(
        {"swing_picks": [], "swing_family_record_line": payload["swing_family_record_line"]}
    )
    assert "Ranked on:" not in empty


def test_the_setups_table_header_takes_its_noun_from_swing_headline():
    """The header and the cells under it can never disagree about the noun.

    `headline_labels` is the one place the word lives; the table composes its
    own "Family" around it. A label typed into the model would be a second
    spelling of the same claim.
    """
    pytest.importorskip("PySide6", reason="the setups table is a Qt model")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import swing_headline
    from ui.models.setup_table_model import FAMILY_RATE_HEADER, SetupTableModel

    labels = swing_headline.headline_labels(swing_headline.OUTCOME_KIND_FAVORABLE_DIRECTION)
    assert labels[0].lower() in FAMILY_RATE_HEADER.lower()
    assert swing_headline.headline_labels()[0] == "Win %"  # trade_r keeps its word
    column = [key for key, _label in SetupTableModel.COLUMNS].index("family_win_rate")
    assert SetupTableModel.COLUMNS[column][1] == FAMILY_RATE_HEADER

    # And the cell under it uses the same noun, off the row's own outcome kind.
    cell = swing_headline.format_win_rate(
        swing_headline.headline_from_tracker_rows(
            "avwap_breakout", [{"win": "1", "side_return_pct": "2.0"}]
        ).as_row()
    )
    assert labels[0].split()[0].lower() in cell.lower()


def test_the_digest_section_note_names_the_policy_rather_than_a_win_rate():
    """A model reading the index must not call a percent move a win rate."""
    from ai_jobs.digest import _SECTION_NOTES

    note = _SECTION_NOTES["swing_win_rates"]
    assert "favorable_direction_scanrow_v1" in note
    assert "scan-row offset" in note.lower()
    assert "5 scan rows" in note
    assert "20 exchange sessions" in note
    assert "never a stop-rule win rate" in note.lower()


def test_the_setups_panel_shows_the_coverage_line():
    pytest.importorskip("PySide6", reason="the setups table is a Qt panel")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ui.panels.master_avwap_panel import MasterAvwapPanel

    app = QApplication.instance() or QApplication([])
    assert app is not None
    panel = MasterAvwapPanel()
    try:
        assert panel.family_record_label.text() == ""
        line = (
            "favorable_direction_scanrow_v1 over 5 scan rows, last 20 sessions "
            "(2026-06-02..2026-06-30): 5 eligible / 0 pending / 1 excluded (stale_horizon 1)"
        )
        panel._on_family_records_ready(({"avwap_breakout": {"win_rate": 0.6}}, line))
        assert panel.family_record_label.text() == line
        assert "5 scan rows" in panel.family_record_label.toolTip()
    finally:
        panel.deleteLater()


def test_both_trader_facing_readers_go_THROUGH_the_one_reader(tmp_path, monkeypatch):
    """Behavioural, not a source scan: replace the reader and watch both follow.

    R4 B4's lesson - a source-text test passes for a verb that never runs the
    code - so this proves the wiring by making `read_eligible_rows` answer
    differently and checking that BOTH surfaces answer differently with it.
    """
    import autopilot_core
    import setup_docs
    import swing_evidence

    path = tmp_path / "master_avwap_tier_outcomes.csv"
    _write_v1_csv(path, stale_rows=0)
    window = ("2026-06-01", "2026-06-30")
    calls: list[str] = []
    real = swing_evidence.read_eligible_rows

    def only_the_first(rows_or_path, policy, **kwargs):
        calls.append(str(policy.outcome_kind))
        read = real(rows_or_path, policy, **kwargs)
        return swing_evidence.EligibleRead(
            policy=read.policy,
            rows=read.rows[:1],
            pending=read.pending,
            excluded=read.excluded,
            source_rows=read.source_rows,
            window=read.window,
        )

    monkeypatch.setattr(swing_evidence, "read_eligible_rows", only_the_first)
    setup_docs.clear_family_outcome_cache()
    monkeypatch.setattr(setup_docs, "_family_outcomes_path", lambda: path)
    monkeypatch.setattr(setup_docs, "_family_outcomes_window", lambda: window)
    try:
        docs_rows = setup_docs.family_headline_rows()
        records = autopilot_core.swing_family_records(path, window=window)
    finally:
        setup_docs.clear_family_outcome_cache()

    assert len(calls) == 2, "each surface must call the shared reader exactly once"
    assert docs_rows["avwap_breakout"]["n"] == 1
    assert records["avwap_breakout"]["wins"] + records["avwap_breakout"]["losses"] == 1


def test_the_stale_rule_is_ONE_function_the_reader_and_the_report_both_call(tmp_path):
    """Change `is_stale_horizon` and both the reader and the export follow it.

    The tier performance export cannot take a whole policy - its cells span
    every horizon over a 365-day lookback - so what it shares is the MISSINGNESS
    PREDICATE. If either side spells the rule out for itself, one of them will
    answer this differently.
    """
    import swing_evidence
    from swing_evidence import POLICY_SCANROW_V1, read_eligible_rows

    rows = [
        _outcome_row(1, stale="False"),
        _outcome_row(2, stale="False"),
        _outcome_row(3, stale="quarantined"),
    ]

    # The rule as shipped: only an explicit True drops, so "quarantined" stays.
    assert len(read_eligible_rows(rows, POLICY_SCANROW_V1, window=("2026-06-01", "2026-06-30")).rows) == 3
    performance = legacy.build_bot_tier_performance_rows(
        rows, rows, lookback_days=365, reference_date="2026-06-30"
    )
    cell = next(
        row for row in performance
        if row["tier"] == "S" and row["side"] == "LONG" and row["horizon_sessions"] == 5
    )
    assert cell["observation_count"] == 3

    # One rule, one function: widen it and BOTH answers move together.
    original = swing_evidence.is_stale_horizon
    try:
        swing_evidence.is_stale_horizon = lambda row: str(
            row.get("stale_horizon") or ""
        ).strip().lower() in {"true", "quarantined"}
        legacy.is_stale_horizon = swing_evidence.is_stale_horizon
        read = read_eligible_rows(rows, POLICY_SCANROW_V1, window=("2026-06-01", "2026-06-30"))
        assert len(read.rows) == 2
        assert read.excluded["stale_horizon"] == 1
        widened = legacy.build_bot_tier_performance_rows(
            rows, rows, lookback_days=365, reference_date="2026-06-30"
        )
        widened_cell = next(
            row for row in widened
            if row["tier"] == "S" and row["side"] == "LONG" and row["horizon_sessions"] == 5
        )
        assert widened_cell["observation_count"] == 2
        # The BASELINE is filtered like for like - an edge against a baseline
        # built on other rules is a subtraction of two different populations.
        assert widened_cell["baseline_observation_count"] == 2
    finally:
        swing_evidence.is_stale_horizon = original
        legacy.is_stale_horizon = original
