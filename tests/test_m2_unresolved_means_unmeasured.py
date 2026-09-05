"""Packet M2 - `unresolved` means UNMEASURED.

A swept trade that measured its bars is not unresolved. The after-close sweep
finalizes with no bars in hand by construction, so every trade it settles was
written `status="unresolved"` - including the 3,607 of the last twenty sessions
whose `finalization.basis` is `last_measured_bar` or
`stop_hit_from_prior_measurement`, i.e. trades whose bars WERE measured and
whose R is already readable under `setup_scoreboard.exit_policy_r`.

Two halves:

* **Reader side** (`outcome_semantics.terminal_kind`) - one truth that reads
  HISTORY correctly without rewriting a single row, so the 4,251 `unresolved`
  rows already on disk are classified by what they measured rather than by
  their label.
* **Writer side** (`bounce_bot_lib.legacy`) - new rows say what the finalizer
  did: `swept_measured` when a measurement was used, `unresolved` ONLY when
  nothing was ever measured. Additive value in an existing column; the header
  is never widened and no historical row is rewritten.

Nothing here may change a detector, a tier, a mute or the PROVEN stamp. The
champion aggregator's `eod_complete` filter is asserted UNCHANGED below.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (str(SCRIPTS_DIR), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import outcome_semantics  # noqa: E402

# The sweep harness already exists; duplicating its fifty-line host builder
# would be a second machine that could drift from the one under test.
from test_outcome_sweep import (  # noqa: E402
    AFTER_CLOSE,
    _context,
    _host,
    _seed,
    _state,
)


@pytest.fixture(autouse=True)
def _isolate_stores():
    """`_host` repoints the module's store paths. Never leave them repointed."""
    import bounce_bot_lib.legacy as legacy

    checkpoint = legacy.INTRADAY_BOUNCE_OUTCOME_STATE_JSON
    csv_path = legacy.INTRADAY_BOUNCE_OUTCOMES_CSV
    try:
        yield
    finally:
        legacy.INTRADAY_BOUNCE_OUTCOME_STATE_JSON = checkpoint
        legacy.INTRADAY_BOUNCE_OUTCOMES_CSV = csv_path


def _bars(rows):
    """`[(open, high, low, close)]` -> the frame the writer expects."""
    return pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-08-21 07:05:00") + pd.Timedelta(minutes=5 * index),
                "open": item[0],
                "high": item[1],
                "low": item[2],
                "close": item[3],
            }
            for index, item in enumerate(rows)
        ]
    )


def _row(status, basis="", reason="", *, event_type="final", event_id="a", **extra):
    """A row shaped exactly like the live CSV's: context in a JSON STRING."""
    context = {"tier": "B"}
    if basis:
        context["finalization"] = {"basis": basis, "measured_bars": 3}
        if reason:
            context["finalization"]["reason"] = reason
    row = {
        "event_id": event_id,
        "event_type": event_type,
        "status": status,
        "context_json": json.dumps(context),
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# M2.1 - one reader-side truth
# ---------------------------------------------------------------------------
def test_a_row_with_bars_through_the_close_is_measured_eod():
    assert (
        outcome_semantics.terminal_kind(_row("eod_complete", "measured"))
        == outcome_semantics.TERMINAL_MEASURED_EOD
    )


@pytest.mark.parametrize(
    "basis", ["last_measured_bar", "stop_hit_from_prior_measurement"]
)
def test_a_swept_row_whose_bars_were_measured_is_measured_swept(basis):
    """The 3,607. Labelled `unresolved` on disk; measured in fact."""
    row = _row("unresolved", basis, "no_eod_close", bars_elapsed="6")
    assert outcome_semantics.terminal_kind(row) == outcome_semantics.TERMINAL_MEASURED_SWEPT


@pytest.mark.parametrize(
    "reason",
    ["no_bars_after_entry", "no_measurement_in_checkpoint", "expired_no_data"],
)
def test_a_row_that_measured_nothing_is_unmeasured(reason):
    row = _row("unresolved", "unresolved", reason, bars_elapsed="0")
    assert outcome_semantics.terminal_kind(row) == outcome_semantics.TERMINAL_UNMEASURED


def test_a_row_with_no_final_is_open():
    assert (
        outcome_semantics.terminal_kind(_row("open", event_type="12_bar"))
        == outcome_semantics.TERMINAL_OPEN
    )


def test_a_final_row_carrying_a_status_the_registry_has_never_seen_is_unmeasured():
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    assert (
        outcome_semantics.terminal_kind(_row("something_new", ""))
        == outcome_semantics.TERMINAL_UNMEASURED
    )


def test_the_new_writer_status_reads_measured_swept_without_the_basis():
    """The status alone answers it, so the reader never parses 308 MB of JSON."""
    row = {"event_id": "a", "event_type": "final", "status": "swept_measured"}
    assert outcome_semantics.terminal_kind(row) == outcome_semantics.TERMINAL_MEASURED_SWEPT


def test_terminal_coverage_folds_rows_per_event_and_says_one_sentence():
    rows = [
        _row("open", event_type="registered", event_id="a"),
        _row("unresolved", "stop_hit_from_prior_measurement", "no_eod_close", event_id="a"),
        _row("open", event_type="registered", event_id="b"),
        _row("eod_complete", "measured", event_id="b"),
        _row("unresolved", "unresolved", "no_bars_after_entry", event_id="c"),
        # No final row at all: still open.
        _row("open", event_type="6_bar", event_id="d"),
    ]
    counts = outcome_semantics.terminal_coverage(rows)
    assert counts[outcome_semantics.TERMINAL_MEASURED_EOD] == 1
    assert counts[outcome_semantics.TERMINAL_MEASURED_SWEPT] == 1
    assert counts[outcome_semantics.TERMINAL_UNMEASURED] == 1
    assert counts[outcome_semantics.TERMINAL_OPEN] == 1
    assert counts["measured"] == 2
    assert counts["events"] == 4

    sentence = outcome_semantics.format_terminal_coverage(counts)
    assert sentence == (
        "Outcomes: measured 2 (eod 1 / swept 1), unmeasured 1, open 1 over the window."
    )


# ---------------------------------------------------------------------------
# M2.1 - the scoreboard counts a swept stop-out under the policy that measured it
# ---------------------------------------------------------------------------
_SCOREBOARD_COLUMNS = (
    "schema_version,event_id,event_type,logged_at,trade_date,symbol,direction,"
    "entry_time,entry_price,stop_price,risk_per_share,bars_elapsed,minutes_elapsed,"
    "close_r,mfe_r,mae_r,best_price,worst_price,target_1r_hit,target_2r_hit,stop_hit,"
    "status,milestone_bar,context_json,outcome_mode,eod_close,eod_move_pct,mfe_pct,mae_pct"
)


def _scoreboard_csv(tmp_path, status):
    """One swept stop-out, written with the status under test."""
    import setup_scoreboard

    context = {
        "finalization": {
            "basis": "stop_hit_from_prior_measurement",
            "measured_bars": 3,
            "reason": "no_eod_close",
        },
        "exit": {"stop_hit": True, "stop_exit_r": -1.0, "last_measured_close": 98.8},
    }
    path = tmp_path / "outcomes.csv"
    columns = _SCOREBOARD_COLUMNS.split(",")
    # The finding this test rests on: the scoreboard is NOT a status-keyed
    # reader. It never loads the column, and has always worked off
    # `finalization.basis`, which is why nothing about it changes for M2.
    assert "status" not in setup_scoreboard.OUTCOME_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "schema_version": "4",
                "event_id": "AAPL_long_20260821_06_30_00_vwap",
                "event_type": "final",
                "logged_at": "2026-08-21T14:30:00",
                "trade_date": "2026-08-21",
                "symbol": "AAPL",
                "direction": "long",
                "entry_time": "2026-08-21T07:00:00",
                "entry_price": "100.0",
                "stop_price": "99.0",
                "risk_per_share": "1.0",
                "bars_elapsed": "3",
                "minutes_elapsed": "15",
                "close_r": "",
                "mfe_r": "0.4",
                "mae_r": "-1.5",
                "best_price": "100.4",
                "worst_price": "98.5",
                "target_1r_hit": "False",
                "target_2r_hit": "False",
                "stop_hit": "True",
                "status": status,
                "milestone_bar": "",
                "context_json": json.dumps(context),
                "outcome_mode": "eod_hold",
                "eod_close": "",
                "eod_move_pct": "",
                "mfe_pct": "",
                "mae_pct": "",
            }
        )
    return path


@pytest.mark.parametrize("status", ["unresolved", "swept_measured"])
def test_a_swept_stop_out_counts_under_stop_exit_and_never_as_an_eod_hold(tmp_path, status):
    """The R is right there; only the LABEL said "unresolved".

    Run under BOTH the historical status and the one the writer now emits, so
    the new value cannot quietly fall out of the scoreboard.
    """
    import setup_scoreboard

    frame, coverage = setup_scoreboard.load_intraday_finals(
        _scoreboard_csv(tmp_path, status),
        window_start="2026-08-01",
        window_end="2026-08-31",
    )
    assert len(frame) == 1
    assert frame["r_stop_exit"].iloc[0] == pytest.approx(-1.0)
    assert pd.isna(frame["r_eod_hold"].iloc[0]), "there is no eod close, so no eod_hold R"
    assert coverage.policy_measured["stop_exit"] == 1
    assert coverage.policy_measured["eod_hold"] == 0
    assert coverage.unresolved == 0, "a measured stop-out is not unresolved"
    assert bool(frame["usable"].iloc[0])
    # And the ONE reader-side truth agrees with the scoreboard about it.
    row = frame.to_dict("records")[0]
    assert outcome_semantics.terminal_kind(row) == outcome_semantics.TERMINAL_MEASURED_SWEPT


# ---------------------------------------------------------------------------
# M2.2 - the writer says what it did
# ---------------------------------------------------------------------------
def test_the_sweep_writes_swept_measured_for_a_trade_that_measured_its_bars(tmp_path):
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {
        "bars": 6, "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2,
        "best_price": 100.8, "worst_price": 99.8, "last_close": 100.5,
        "stop_hit": False, "target_1r_hit": False, "target_2r_hit": False,
        "minutes_elapsed": 30, "at": "2026-08-21T08:00:00",
    }
    _seed(host, {"a": state})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["status"] == "swept_measured"
    # Everything else about the row is unchanged: `close_r` is still the
    # eod_hold number and there is still no eod close.
    assert row["close_r"] == "" and row["eod_close"] == ""
    assert _context(row)["finalization"]["basis"] == "last_measured_bar"
    assert _context(row)["finalization"]["reason"] == "no_eod_close"
    assert outcome_semantics.terminal_kind(row) == outcome_semantics.TERMINAL_MEASURED_SWEPT


def test_a_swept_stop_out_is_swept_measured_too(tmp_path):
    host = _host(tmp_path)
    state = _state(event_id="a")
    state["last_measured"] = {
        "bars": 3, "stop_hit": True, "best_price": 100.4, "worst_price": 98.5,
        "mfe_r": 0.4, "mae_r": -1.5, "last_close": 98.8,
    }
    _seed(host, {"a": state})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["status"] == "swept_measured"
    assert _context(row)["finalization"]["basis"] == "stop_hit_from_prior_measurement"
    assert _context(row)["exit"]["stop_exit_r"] == -1.0


def test_a_trade_that_measured_nothing_is_still_unresolved(tmp_path):
    """`unresolved` now means exactly one thing: nothing was ever measured."""
    host = _host(tmp_path)
    _seed(host, {"a": _state(event_id="a")})
    host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    row = host.rows[-1]
    assert row["status"] == "unresolved"
    assert _context(row)["finalization"]["basis"] == "unresolved"
    assert outcome_semantics.terminal_kind(row) == outcome_semantics.TERMINAL_UNMEASURED


def test_a_final_with_bars_through_the_close_is_untouched(tmp_path):
    host = _host(tmp_path)
    state = _state(event_id="a")
    host._append_bounce_outcome_row(
        state, "final", bars_elapsed=1, milestone_bar=None,
        rows_after_entry=_bars([(100, 101.2, 99.7, 101.0)]), finalize_eod=True,
    )
    row = host.rows[-1]
    assert row["status"] == "eod_complete"
    assert _context(row)["finalization"]["basis"] == "measured"


def test_the_sweep_counts_the_four_way_split(tmp_path):
    host = _host(tmp_path)
    measured = _state(event_id="a")
    measured["last_measured"] = {
        "bars": 6, "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2,
        "best_price": 100.8, "worst_price": 99.8, "last_close": 100.5,
        "stop_hit": False, "target_1r_hit": False, "target_2r_hit": False,
        "minutes_elapsed": 30, "at": "2026-08-21T08:00:00",
    }
    _seed(host, {"a": measured, "b": _state(event_id="b")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)
    assert counts["finalized"] == 2
    split = counts["by_terminal_kind"]
    assert split[outcome_semantics.TERMINAL_MEASURED_SWEPT] == 1
    assert split[outcome_semantics.TERMINAL_UNMEASURED] == 1
    assert split.get(outcome_semantics.TERMINAL_MEASURED_EOD, 0) == 0


def test_the_sweep_log_line_names_the_split(tmp_path):
    """The line the live gate reads off `trading_bot.log`."""
    import bounce_bot_lib.legacy as legacy

    host = _host(tmp_path)
    measured = _state(event_id="a")
    measured["last_measured"] = {
        "bars": 6, "close_r": 0.5, "mfe_r": 0.8, "mae_r": -0.2,
        "best_price": 100.8, "worst_price": 99.8, "last_close": 100.5,
        "stop_hit": False, "target_1r_hit": False, "target_2r_hit": False,
        "minutes_elapsed": 30, "at": "2026-08-21T08:00:00",
    }
    _seed(host, {"a": measured, "b": _state(event_id="b")})
    counts = host.sweep_pending_bounce_outcomes(now=AFTER_CLOSE)

    text = legacy.outcome_sweep_log_line(counts)
    assert "finalized 2" in text
    assert "measured_eod 0" in text
    assert "swept_measured 1" in text
    assert "unmeasured 1" in text
    assert "expired 0" in text


# ---------------------------------------------------------------------------
# the champion aggregator is UNCHANGED (no tier / mute / PROVEN movement)
# ---------------------------------------------------------------------------
def test_the_champion_aggregator_still_takes_eod_complete_rows_only():
    """A guard, not a change: `swept_measured` is excluded exactly as
    `unresolved` was, so the learned tier, the mute and the PROVEN stamp see
    the same population they saw before this packet."""
    from bounce_bot_lib.legacy import _latest_bounce_outcome_rows

    frame = pd.DataFrame(
        [
            {"event_id": "a", "event_type": "final", "status": "eod_complete",
             "bars_elapsed": 12, "logged_at": "2026-08-21T14:30:00"},
            {"event_id": "b", "event_type": "final", "status": "swept_measured",
             "bars_elapsed": 6, "logged_at": "2026-08-21T14:30:00"},
            {"event_id": "c", "event_type": "final", "status": "unresolved",
             "bars_elapsed": 0, "logged_at": "2026-08-21T14:30:00"},
        ]
    )
    kept = _latest_bounce_outcome_rows(frame)
    assert list(kept["event_id"]) == ["a"]


# ---------------------------------------------------------------------------
# M2.3 - coverage is shown, never hidden
# ---------------------------------------------------------------------------
def test_the_episode_carries_its_terminal_kind_and_the_report_counts_them(tmp_path):
    import held_run_score

    rows = [
        _row("open", event_type="registered", event_id="a", trade_date="2026-08-21",
             symbol="AAPL", direction="long"),
        _row("unresolved", "last_measured_bar", "no_eod_close", event_id="a",
             trade_date="2026-08-21", symbol="AAPL", direction="long"),
        _row("open", event_type="registered", event_id="b", trade_date="2026-08-21",
             symbol="MSFT", direction="long"),
        _row("eod_complete", "measured", event_id="b", trade_date="2026-08-21",
             symbol="MSFT", direction="long"),
        _row("open", event_type="6_bar", event_id="c", trade_date="2026-08-21",
             symbol="NVDA", direction="long"),
    ]
    episodes = held_run_score.build_episodes(rows, as_of="2026-08-24")
    kinds = {episode.event_id: episode.terminal_kind for episode in episodes}
    assert kinds["a"] == outcome_semantics.TERMINAL_MEASURED_SWEPT
    assert kinds["b"] == outcome_semantics.TERMINAL_MEASURED_EOD
    assert kinds["c"] == outcome_semantics.TERMINAL_OPEN

    counts = held_run_score.terminal_coverage(episodes)
    assert counts[outcome_semantics.TERMINAL_MEASURED_SWEPT] == 1
    assert counts[outcome_semantics.TERMINAL_MEASURED_EOD] == 1
    assert counts[outcome_semantics.TERMINAL_OPEN] == 1
    assert counts["events"] == 3


def test_the_tracker_status_line_shows_the_coverage(monkeypatch):
    """Q1's window sentence, then M2's coverage sentence, from ONE read."""
    from ui.panels import daytrade_tracker_panel as panel_module

    counts = outcome_semantics.terminal_coverage(
        [
            _row("eod_complete", "measured", event_id="a"),
            _row("unresolved", "last_measured_bar", "no_eod_close", event_id="b"),
            _row("unresolved", "unresolved", "no_bars_after_entry", event_id="c"),
            _row("open", event_type="registered", event_id="d"),
        ]
    )
    text = panel_module.outcome_coverage_text(counts)
    assert text == (
        "Outcomes: measured 2 (eod 1 / swept 1), unmeasured 1, open 1 over the window."
    )
    assert panel_module.outcome_coverage_text({}) == ""


def test_the_held_run_report_carries_the_coverage_from_one_read(tmp_path, monkeypatch):
    """No second pass over 308 MB: the coverage rides the existing read."""
    import held_run_score
    from ui.panels import daytrade_tracker_panel as panel_module

    episodes = held_run_score.build_episodes(
        [
            _row("eod_complete", "measured", event_id="a", trade_date="2026-08-21",
                 symbol="AAPL", direction="long"),
        ],
        as_of="2026-08-24",
    )
    calls = {"n": 0}

    def _one_read(**_kwargs):
        calls["n"] += 1
        return episodes

    monkeypatch.setattr(held_run_score, "load_episodes", _one_read)
    report = panel_module.load_held_run_report()
    assert calls["n"] == 1, "one read, not one per section"
    assert report["outcome_coverage"][outcome_semantics.TERMINAL_MEASURED_EOD] == 1


def test_the_away_digest_reports_the_coverage_where_it_reports_the_outcome_count():
    import autopilot_core

    rows = [
        _row("eod_complete", "measured", event_id="a"),
        _row("unresolved", "stop_hit_from_prior_measurement", "no_eod_close", event_id="b"),
        _row("unresolved", "unresolved", "expired_no_data", event_id="c"),
    ]
    line = autopilot_core.outcome_coverage_line(rows, window_text="today")
    assert line == "Outcomes: measured 2 (eod 1 / swept 1), unmeasured 1, open 0 today."
    assert autopilot_core.outcome_coverage_line([]) == ""

    report = autopilot_core.render_away_report(
        {"generated_at": "2026-09-05T14:00:00", "outcome_coverage_line": line}
    )
    assert line in report
