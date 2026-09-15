r"""PCT-3: the LIVE scan's own row builder publishes the measure.

Added by the BUILDER in the fix round. The first review found the packet's
central premise false:

> The desk's scan builds its priority rows and `master_avwap_ai_state.json` in
> `runner._run_master_impl`, a near-duplicate of
> `legacy._evaluate_priority_snapshot_for_date`, which only
> `build_completed_bar_priority_rows` (its ai_state discarded) and the tracker
> backfill call. Today's live ai_state: 1003 symbols, 35 flagged, 0 with
> `compression_score`.

So the first build applied the copy-through at a seam the desk never runs, and
the tester's file - which drives `legacy._evaluate_priority_snapshot_for_date` -
could not see it. Both seams now call the same two functions, and THIS file is
the one that would go red if `runner.py`'s copy were dropped again.

It drives `runner._run_master_impl` for real, with one synthetic symbol and
every outside door closed (no IB client, no Yahoo, no earnings provider, no
tracker write), and then reads `master_avwap_ai_state.json` off disk - the file
the desk's setups table actually merges from. `tests/conftest.py` points
`project_paths` at a test directory, so nothing here touches a live store.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SYMBOL = "PCTX"


def _sessions(count: int) -> list[pd.Timestamp]:
    """`count` business days ending on the last completed weekday before today.

    The scan reads "today" off the wall clock, so the frame has to end in the
    real past or the last bar is a forming one and the priority row never
    arrives.
    """
    end = date.today() - timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    return list(pd.bdate_range(end=pd.Timestamp(end), periods=count))


def _compressed_frame() -> pd.DataFrame:
    """A clean run-up into an earnings anchor, then a tight box.

    The same shape as the tester's `COMPRESSED_FRAME`: whatever the score comes
    out as, the anchored slice after the anchor is narrow enough that
    `summarize_anchor_compression` has three real ratios to publish - which is
    all this file asserts about.
    """
    stamps = _sessions(120)
    anchor_index = 90
    rows = []
    for index, stamp in enumerate(stamps):
        if index < anchor_index:
            base = 80.0 + index * 0.25
            rows.append((stamp, base, base + 1.2, base - 1.2, base + 0.3))
        else:
            base = 103.0
            wiggle = 0.15 * ((index % 3) - 1)
            rows.append((stamp, base + wiggle, base + 0.30, base - 0.30, base + wiggle))
    frame = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    frame["volume"] = 1_000_000
    return frame, stamps[anchor_index].date()


@pytest.fixture
def scanned(monkeypatch, tmp_path):
    """One symbol through `runner._run_master_impl`, every outside door closed.

    Returns `(run_result, ai_state_payload)`.
    """
    from master_avwap_lib import runner

    frame, anchor_day = _compressed_frame()
    anchor_iso = anchor_day.isoformat()

    monkeypatch.setattr(
        runner, "resolve_master_scan_watchlist_paths",
        lambda **_kwargs: ([tmp_path / "longs.txt"], [tmp_path / "shorts.txt"], "test watchlist"),
    )
    monkeypatch.setattr(runner, "load_tickers_from_paths", lambda paths, **_kw: [SYMBOL] if "longs" in str(list(paths)[0]) else [])
    monkeypatch.setattr(
        runner, "append_master_avwap_d1_watchlist_symbols", lambda longs, shorts: (longs, shorts, 0)
    )
    monkeypatch.setattr(runner, "load_theta_long_symbols", lambda *a, **k: [])
    monkeypatch.setattr(
        runner, "load_scan_earnings_context", lambda symbols: ({SYMBOL: [anchor_iso]}, {SYMBOL: {}})
    )
    monkeypatch.setattr(runner, "collect_upcoming_earnings_dates", lambda symbols: {})
    monkeypatch.setattr(runner, "_maybe_run_setup_tracker_catchup", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "connect_daily_data_client", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "build_market_regime_snapshot", lambda ib, day: {"label": "mixed"})
    monkeypatch.setattr(runner, "fetch_daily_bars", lambda ib, sym, days, **kw: frame.copy())
    monkeypatch.setattr(runner, "record_theta_picks", lambda *a, **k: None)

    result = runner._run_master_impl(update_setup_tracker=False)

    import project_paths

    payload = json.loads(Path(project_paths.MASTER_AVWAP_AI_STATE_FILE).read_text(encoding="utf-8"))
    return result, payload


COPY_THROUGH_FIELDS = (
    "compression_score",
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "compression_rule_version",
)


def test_the_scan_that_writes_ai_state_publishes_the_compression_measure(scanned):
    """The file the desk's setups table merges from carries the numbers.

    This is the assertion the live ai_state failed: 1003 symbols, 35 flagged,
    **0** with `compression_score`.
    """
    _result, payload = scanned
    entry = (payload.get("symbols") or {}).get(SYMBOL)
    assert entry, f"the scan wrote no ai_state entry for {SYMBOL}: {sorted(payload.get('symbols') or {})}"

    missing = [field for field in COPY_THROUGH_FIELDS if field not in entry]
    assert not missing, f"the ai_state symbol entry is missing {missing}"
    assert entry["compression_rule_version"] == "anchor_compression_v1"
    assert isinstance(entry["compression_score"], int)
    for field in COPY_THROUGH_FIELDS[1:4]:
        assert entry[field] is not None, f"{field} was published as a blank"


def test_the_scan_s_priority_row_carries_them_too(scanned):
    """The row the report, the tracker and every downstream reader are built from."""
    result, _payload = scanned
    rows = [row for row in (result.get("priority_rows") or []) if row.get("symbol") == SYMBOL]
    assert rows, "the scan produced no priority row for the synthetic symbol"
    row = rows[0]
    missing = [field for field in COPY_THROUGH_FIELDS if field not in row]
    assert not missing, f"the priority row is missing {missing}"


def test_the_scan_evaluates_the_v1_break_rule_on_every_row(scanned):
    """`compression_break_recent` and its rule version travel together.

    The fixture is compressed rather than breaking, so the VALUE here is False -
    what this pins is that the rule RAN, which is what makes the tag reachable
    at all from the live scan.
    """
    result, payload = scanned
    row = [row for row in (result.get("priority_rows") or []) if row.get("symbol") == SYMBOL][0]
    entry = payload["symbols"][SYMBOL]
    for carrier, where in ((row, "the priority row"), (entry, "the ai_state entry")):
        assert "compression_break_recent" in carrier, where
        assert carrier["compression_break_rule_version"] == "compression_break_v1", where


def test_the_feature_history_columns_carry_the_measure(scanned):
    """`df_features` is built with `columns=feature_columns`, so a field the row
    carries and that allowlist does not is silently dropped - which is exactly
    how `assigned_tier` became a decision nothing could read."""
    from master_avwap_lib import runner
    import inspect

    source = inspect.getsource(runner._run_master_impl)
    start = source.index("feature_columns = [")
    end = source.index("]", start)
    listed = source[start:end]
    for field in (*COPY_THROUGH_FIELDS, "compression_break_recent", "compression_break_rule_version"):
        assert f'"{field}"' in listed, f"{field} is not in the feature_columns allowlist"


def test_a_record_with_no_measure_is_not_stamped_with_a_rule_that_never_ran():
    """An old tracker row reads as NOT MEASURED, never as a measured zero.

    A `compression_score` of 0 under `anchor_compression_v1` says "the rule
    looked and found nothing tight" - and that zero is what a calibration report
    would average.
    """
    from master_avwap_lib import legacy

    stale = {"symbol": "OLD", "side": "LONG", "compression_flag": True, "compression_penalty": 10}
    copied = legacy.compression_copy_through_from_row(stale)
    assert copied["compression_score"] is None
    assert copied["compression_stdev_atr_ratio"] is None
    assert "compression_rule_version" not in copied

    measured = dict(stale, compression_score=3, compression_stdev_atr_ratio=0.52)
    copied = legacy.compression_copy_through_from_row(measured)
    assert copied["compression_score"] == 3
    assert copied["compression_rule_version"] == "anchor_compression_v1"


def test_a_record_v1_never_evaluated_carries_no_break_verdict_at_all():
    """The flag and its version are one reading: both, or neither."""
    from master_avwap_lib import legacy

    assert legacy._compression_break_copy_through({"symbol": "OLD"}, {}) == {}
    evaluated = legacy._compression_break_copy_through(
        {"compression_break_recent": False, "compression_break_v1_note": ""}, {}
    )
    assert evaluated["compression_break_recent"] is False
    assert evaluated["compression_break_rule_version"] == "compression_break_v1"


def test_the_v1_note_has_its_own_field_that_phase_six_cannot_overwrite():
    """`enrich_priority_rows_with_phase6_studies` does `row.update(context)` and
    owns `compression_break_note`. v1's answer lives beside it."""
    from master_avwap_lib import legacy

    row = {"compression_break_recent": True, "compression_break_v1_note": "v1 said so"}
    phase6_context = {"compression_break_note": "", "compression_break_today": False}
    row.update(phase6_context)
    assert row["compression_break_v1_note"] == "v1 said so"
    assert legacy._compression_break_copy_through(row, {})["compression_break_v1_note"] == "v1 said so"


def test_the_break_tag_is_a_confirmation_and_never_displaces_an_older_one():
    """Six tags is the cap. A label added in 2026 may not cost a row a tag it
    has carried for a year, so `COMPRESSION_BREAK` is added in confirmation
    order rather than ahead of the confirmations."""
    from master_avwap_lib import setup_tagging

    crowded = {
        "side": "LONG",
        "setup_family": "avwap_breakout",
        "favorite_signals": ["CROSS_UP_UPPER_1"],
        "top_pattern_entry": True,
        "previous_day_range_break": True,
        "breakout_5d": True,
        "trend_20d": "UP",
        "trend_ma_alignment": True,
    }
    without = setup_tagging.derive_setup_tag_payload(crowded)["setup_tags"]
    with_break = setup_tagging.derive_setup_tag_payload(
        dict(crowded, compression_break_recent=True)
    )["setup_tags"]
    assert without == with_break, "the new label displaced an older confirmation"

    quiet = {"side": "LONG", "setup_family": "general", "compression_break_recent": True}
    payload = setup_tagging.derive_setup_tag_payload(quiet)
    assert "COMPRESSION_BREAK" in payload["setup_tags"]
    assert payload["setup_tag_roles"]["COMPRESSION_BREAK"] == "confirmation"
