"""R10.A / D2 - a finalization may not write a number it did not measure.

The defect, reproduced from the live store: **1,164 of 6,907 in-window finals
(16.9%) carry `close_r` exactly 0, and every single one has `eod_close` exactly
equal to its entry price** - while 0 of the 5,743 non-zero finals do. The writer
defaulted the close to the entry whenever it had no bars in hand. 251 of those
never advanced a bar at all; **563 are trades whose own earlier rows had already
recorded a stop hit** and which therefore score about -1R, not 0.

**`close_r` means one thing everywhere** (trader decision, 2026-08-23 review):
R at the EOD close under `eod_hold`. Without bars through the close there is no
such number, so it stays blank and the row is `unresolved` - it does not become
-1.0 because a stop was hit, and it does not become the last mid-session close
either. Both would make the same trade report a different number depending only
on what the finalizer happened to have in hand.

What WAS measured is still written: mfe/mae, best/worst, the stop and target
flags. The exit-policy question `close_r` cannot answer lives in `context.exit`:
`stop_exit_r` (-1.0 **under a named assumption** - only the touch is measured),
`gap_through_stop`, and `ambiguous_interval_bars` for a bar whose own range
contains both the stop and the 1R target, where R10.0's predeclared stop-first
rule applies and the count is reported.

A 0R is a number a mean will happily average in. `unresolved` is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _Writer:
    """A minimal host for the one function that writes an outcome row."""


def _host():
    from bounce_bot_lib.legacy import BounceBot

    host = _Writer.__new__(_Writer)
    host.rows = []
    host.pending_bounce_outcomes = {}
    host._append_learning_row = lambda path, fieldnames, row: host.rows.append(dict(row))
    host._mirror_outcome_row_to_ledger = lambda row, state: None
    host._parse_bar_time = BounceBot._parse_bar_time.__get__(host, _Writer)
    host._json_for_learning = BounceBot._json_for_learning.__get__(host, _Writer)
    host._context_with_finalization = BounceBot._context_with_finalization.__get__(host, _Writer)
    host._exit_facts = BounceBot._exit_facts.__get__(host, _Writer)
    host._naive_market_local = BounceBot._naive_market_local
    host.OUTCOME_BAR_MINUTES = BounceBot.OUTCOME_BAR_MINUTES
    host._completed_session_rows = BounceBot._completed_session_rows.__get__(host, _Writer)
    host._rows_after_bounce_entry_for_session = (
        BounceBot._rows_after_bounce_entry_for_session.__get__(host, _Writer)
    )
    host._append_bounce_outcome_row = BounceBot._append_bounce_outcome_row.__get__(host, _Writer)
    return host


def _state(**kwargs):
    state = {
        "event_id": "AAPL_long_20260821_06_30_00_h1_ema10_bounce",
        "symbol": "AAPL",
        "direction": "long",
        "trade_date": "2026-08-21",
        "entry_time": "2026-08-21T07:00:00",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "risk_per_share": 1.0,
        "target_1r": 101.0,
        "target_2r": 102.0,
        "milestones_logged": [],
        "outcome_mode": "eod_hold",
        "context": {"tier": "B"},
    }
    state.update(kwargs)
    return state


def _bars(rows):
    return pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(f"2026-08-21 07:{5 * (i + 1):02d}:00"),
                "open": row[0], "high": row[1], "low": row[2], "close": row[3],
                "volume": 1000.0,
            }
            for i, row in enumerate(rows)
        ]
    )


def _final(host, state, bars=None):
    host._append_bounce_outcome_row(
        state, "final", bars_elapsed=0 if bars is None else len(bars),
        milestone_bar=None,
        rows_after_entry=pd.DataFrame() if bars is None else bars,
        finalize_eod=True,
    )
    return host.rows[-1]


import json  # noqa: E402


def _context(row) -> dict:
    return json.loads(row["context_json"])


# ---------------------------------------------------------------------------
# nothing was ever seen
# ---------------------------------------------------------------------------
def test_a_session_with_no_bars_finalizes_unresolved_not_at_zero():
    host = _host()
    row = _final(host, _state())
    assert row["status"] == "unresolved"
    assert row["close_r"] == ""
    assert row["eod_close"] == ""
    assert row["mfe_r"] == "" and row["mae_r"] == ""


def test_the_unresolved_row_says_why():
    host = _host()
    row = _final(host, _state())
    finalization = _context(row)["finalization"]
    assert finalization["basis"] == "unresolved"
    assert finalization["reason"] == "no_bars_after_entry"
    assert finalization["measured_bars"] == 0


def test_the_row_never_claims_the_close_was_the_entry():
    """The exact signature of the 1,164: close_r 0 and eod_close == entry."""
    host = _host()
    row = _final(host, _state())
    assert not (row["close_r"] == 0 and row["eod_close"] == row["entry_price"])


def test_the_existing_context_survives():
    host = _host()
    row = _final(host, _state())
    assert _context(row)["tier"] == "B"


# ---------------------------------------------------------------------------
# bars were seen earlier, but not through the close
# ---------------------------------------------------------------------------
def test_a_stop_hit_without_eod_bars_is_unresolved_with_its_stop_exit_recorded():
    """MAJOR-3, trader decision 2026-08-23: `close_r` means ONE thing.

    It is R at the EOD close under `eod_hold`, everywhere. Writing -1.0 into it
    because a stop was hit made the same trade report a different number
    depending only on whether the finalizer had bars in hand - and -1.0 assumes
    a fill at the stop when the measured fact is only that the low reached it.
    The stop exit is now its own field, under a named assumption.
    """
    host = _host()
    state = _state()
    host._append_bounce_outcome_row(
        state, "3_bar", bars_elapsed=3, milestone_bar=3,
        rows_after_entry=_bars([(100, 100.4, 98.5, 98.8)]),
    )
    assert state["last_measured"]["stop_hit"] is True

    row = _final(host, state)
    assert row["status"] == "unresolved", "no bars through the close, so no eod_hold number"
    assert row["close_r"] == "" and row["eod_close"] == ""
    assert row["stop_hit"] is True, "what WAS measured is still recorded"
    assert row["mae_r"] != ""

    context = _context(row)
    assert context["finalization"]["basis"] == "stop_hit_from_prior_measurement"
    assert context["finalization"]["reason"] == "no_eod_close"
    assert context["exit"]["stop_exit_r"] == -1.0
    assert "only the touch is measured" in context["exit"]["stop_exit_assumption"]


def test_a_gap_through_the_stop_is_flagged_because_the_fill_assumption_is_optimistic():
    host = _host()
    state = _state()
    host._append_bounce_outcome_row(
        state, "3_bar", bars_elapsed=3, milestone_bar=3,
        rows_after_entry=_bars([(100, 100.2, 97.0, 97.4)]),   # mae_r = -3.0
    )
    row = _final(host, state)
    assert _context(row)["exit"]["gap_through_stop"] is True


def test_a_stop_touched_but_not_gapped_is_not_flagged():
    host = _host()
    state = _state()
    host._append_bounce_outcome_row(
        state, "3_bar", bars_elapsed=3, milestone_bar=3,
        rows_after_entry=_bars([(100, 100.2, 99.0, 99.4)]),   # mae_r = -1.0 exactly
    )
    row = _final(host, state)
    assert _context(row)["exit"]["gap_through_stop"] is False


def test_no_stop_and_no_eod_bars_is_also_unresolved():
    """The last mid-session close is not the EOD close, and must not pose as it."""
    host = _host()
    state = _state()
    host._append_bounce_outcome_row(
        state, "3_bar", bars_elapsed=3, milestone_bar=3,
        rows_after_entry=_bars([(100, 100.8, 99.6, 100.5)]),
    )
    row = _final(host, state)
    assert row["status"] == "unresolved"
    assert row["close_r"] == "" and row["eod_close"] == ""
    context = _context(row)
    assert context["finalization"]["basis"] == "last_measured_bar"
    assert context["finalization"]["measured_bars"] == 3
    assert context["exit"]["last_measured_close"] == 100.5, "kept, but not as the close"


def test_a_measurement_is_kept_across_calls_on_the_state():
    host = _host()
    state = _state()
    host._append_bounce_outcome_row(
        state, "1_bar", bars_elapsed=1, milestone_bar=1,
        rows_after_entry=_bars([(100, 100.8, 99.6, 100.5)]),
    )
    measured = state["last_measured"]
    assert measured["bars"] == 1 and measured["last_close"] == 100.5
    assert measured["at"], "a measurement carries when it was taken"


# ---------------------------------------------------------------------------
# bars are in hand right now
# ---------------------------------------------------------------------------
def test_a_final_with_live_bars_is_unchanged_and_reads_measured():
    host = _host()
    state = _state()
    row = _final(host, state, _bars([(100, 101.2, 99.7, 101.0)]))
    assert row["status"] == "eod_complete"
    assert row["eod_close"] == 101.0
    assert row["close_r"] == pytest.approx(1.0)
    assert _context(row)["finalization"]["basis"] == "measured"


def test_a_live_stop_out_keeps_eod_hold_semantics_in_close_r():
    """The same trade must not report a different number with bars in hand."""
    host = _host()
    state = _state()
    row = _final(host, state, _bars([(100, 100.4, 98.5, 98.8)]))
    assert row["status"] == "eod_complete"
    assert row["close_r"] == pytest.approx(-1.2), "eod_hold: R at the close, not at the stop"
    assert row["stop_hit"] is True
    assert _context(row)["exit"]["stop_exit_r"] == -1.0


def test_a_bar_containing_both_the_stop_and_the_target_is_counted_as_ambiguous():
    """Within one bar there is no way to know which came first."""
    host = _host()
    state = _state()
    row = _final(host, state, _bars([(100, 101.5, 98.5, 100.2)]))
    exit_facts = _context(row)["exit"]
    assert exit_facts["ambiguous_interval_bars"] == 1
    assert exit_facts["ambiguity_rule"] == "stop first (R10.0 predeclared)"


def test_an_unambiguous_bar_is_not_counted():
    host = _host()
    state = _state()
    row = _final(host, state, _bars([(100, 100.4, 98.5, 98.8)]))
    assert _context(row)["exit"]["ambiguous_interval_bars"] == 0


def test_a_non_final_row_carries_no_finalization_block():
    host = _host()
    state = _state()
    host._append_bounce_outcome_row(
        state, "1_bar", bars_elapsed=1, milestone_bar=1,
        rows_after_entry=_bars([(100, 100.8, 99.6, 100.5)]),
    )
    assert "finalization" not in _context(host.rows[-1])


# ---------------------------------------------------------------------------
# the legacy rows are unaffected
# ---------------------------------------------------------------------------
def test_the_legacy_detector_and_the_registry_rule_agree_on_the_old_signature():
    """`unsettled_close_mask` and `fabricated_zero_v1` describe the same rows."""
    import evidence_rules
    from setup_scoreboard import unsettled_close_mask

    frame = pd.DataFrame(
        [
            {"close_r": 0.0, "eod_close": 12.34, "entry_price": 12.34},   # fabricated
            {"close_r": 0.0, "eod_close": 12.35, "entry_price": 12.34},   # real scratch
            {"close_r": 1.5, "eod_close": 13.00, "entry_price": 12.34},   # ordinary
        ]
    )
    mask = list(unsettled_close_mask(frame))
    tags = [
        evidence_rules.fabricated_zero_v1(
            close_r=row["close_r"], eod_close=row["eod_close"], entry_price=row["entry_price"]
        ).tagged
        for row in frame.to_dict("records")
    ]
    assert mask == tags == [True, False, False]


def test_new_unresolved_rows_are_excluded_without_calling_them_fabricated_zeroes():
    """Blank is the new honest encoding; it is unresolved, not legacy fabrication."""
    import evidence_rules
    from setup_scoreboard import unsettled_close_mask

    host = _host()
    row = _final(host, _state())
    frame = pd.DataFrame([{k: row[k] for k in ("close_r", "eod_close", "entry_price")}])
    assert any(unsettled_close_mask(frame))
    assert not evidence_rules.fabricated_zero_v1(
        close_r=row["close_r"], eod_close=row["eod_close"], entry_price=row["entry_price"]
    ).tagged
