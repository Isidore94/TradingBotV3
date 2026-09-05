"""V1 item 2 - did the level hold, then how far did it run. SHADOW ONLY.

Decision 0016 answer 4: *"the intraday level holds, then the name runs. Rank by
maximum favourable excursion - the most the move offered - not by any exit;
exiting well is the trader's job."* And: *"an M5 alert on a name that also
carries a D1 setup outranks the same alert on a name that does not."*

A SECOND score. The champion tier still gates alerts, still mutes and still
stamps PROVEN; this is displayed beside it and never instead of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _row(
    event_id,
    *,
    date="2026-09-01",
    symbol="AAA",
    entry="10:00:00",
    stop_hit=False,
    mfe="2.0",
    minutes="60",
    environment="bullish_strong",
):
    return {
        "event_id": event_id,
        "trade_date": date,
        "symbol": symbol,
        "direction": "long",
        "entry_time": f"{date}T{entry}",
        "context_json": json.dumps({"market_environment": environment}),
        "stop_hit": "True" if stop_hit else "False",
        "mfe_r": mfe,
        "minutes_elapsed": minutes,
    }


def _event(symbol, *, entry="10_00_00", kind="ema_15", date="20260901"):
    return f"{symbol}_long_{date}_{entry}_{kind}"


# ---------------------------------------------------------------------------
# The two halves, and why they multiply
# ---------------------------------------------------------------------------


def test_the_score_is_the_hold_rate_times_what_the_held_ones_offered():
    import held_run_score as hrs

    rows = [
        _row(_event("A"), mfe="3.0"),
        _row(_event("B"), mfe="1.0"),
        _row(_event("C"), stop_hit=True, minutes="10", mfe="0.1"),
        _row(_event("D"), stop_hit=True, minutes="5", mfe="0.2"),
    ]

    cells = hrs.build_segments(hrs.build_episodes(rows), min_n=1)

    assert len(cells) == 1
    cell = cells[0]
    assert cell["n"] == 4 and cell["n_held"] == 2
    assert cell["hold_rate"] == 0.5
    assert cell["mean_mfe_r_of_held"] == 2.0, "the mean of the HELD ones only"
    assert cell["held_run_score"] == 1.0


def test_a_stop_after_the_window_is_an_exit_and_not_a_broken_level():
    """A level that gives way an hour later gave the trade its chance first."""
    import held_run_score as hrs

    # Packet Q1: `stop_hit` is a boolean over ALL bars and the log carries no
    # first-break time, so a late stop is an exit only when an EARLIER row
    # already saw the window pass with the stop intact. Alone it is unknown.
    late = hrs.build_episodes(
        [_row(_event("A"), minutes="35"), _row(_event("A"), stop_hit=True, minutes="45")]
    )
    alone = hrs.build_episodes([_row(_event("C"), stop_hit=True, minutes="45")])
    early = hrs.build_episodes([_row(_event("B"), stop_hit=True, minutes="20")])

    assert late[0].held is True
    assert alone[0].held is False and alone[0].measurement == hrs.UNMEASURED
    assert early[0].held is False


def test_a_stop_that_cannot_be_placed_counts_as_broken():
    """Calling it late would quietly improve every hold rate on the board."""
    import held_run_score as hrs

    row = _row(_event("A"), stop_hit=True, minutes="")
    row["logged_at"] = ""
    assert hrs.build_episodes([row])[0].held is False


def test_the_many_rows_of_one_alert_fold_into_one_episode():
    """The log carries a registered row, updates, milestones and a final."""
    import held_run_score as hrs

    event = _event("A")
    episodes = hrs.build_episodes(
        [
            _row(event, mfe=""),
            _row(event, mfe="1.5"),
            _row(event, mfe="2.9"),
            _row(event, mfe="2.1"),
        ]
    )

    assert len(episodes) == 1
    assert episodes[0].mfe_r == 2.9, "the BEST the move offered, not the last"


# ---------------------------------------------------------------------------
# The segment, including the D1 dimension the trader named
# ---------------------------------------------------------------------------


def test_a_name_that_also_carries_a_d1_setup_is_its_own_segment():
    import held_run_score as hrs

    rows = [
        _row(_event("WITH"), symbol="WITH"),
        _row(_event("WITHOUT"), symbol="WITHOUT"),
    ]
    episodes = hrs.build_episodes(
        rows, d1_setups_by_session={"2026-09-01": {"WITH": {"LONG"}}}
    )

    by_symbol = {episode.symbol: episode for episode in episodes}
    assert by_symbol["WITH"].d1_setup_present is True
    assert by_symbol["WITHOUT"].d1_setup_present is False
    assert by_symbol["WITH"].segment() != by_symbol["WITHOUT"].segment()


def test_the_time_buckets_split_the_open_from_the_middle_of_the_day():
    """R4 fix round 1: the buckets are the CHAMPION'S, and so is the arithmetic.

    V1 declared four of its own here (`open_30m` / `morning` / `midday` /
    `power_hour`) and read the hour straight off the clock against Eastern
    cutoffs - on a Pacific desk, with a desk-local `entry_time`, that is the bug
    `bounce_bot_lib.learning.time_bucket_for` says it exists to have fixed. It
    also gave the Daytrade Tracker's Time of Day tab a vocabulary the aggregator
    does not use, so 8 of 10 live rows went blank on a spelling.

    Asserted against `time_bucket_for` rather than against literals: the point is
    that there is ONE definition, and a literal here would be a second one.
    """
    from datetime import datetime

    import held_run_score as hrs
    from bounce_bot_lib.learning import time_bucket_for

    for text in (
        "2026-09-01T06:35:00",
        "2026-09-01T07:30:00",
        "2026-09-01T10:00:00",
        "2026-09-01T12:30:00",
    ):
        assert hrs.time_bucket(text) == time_bucket_for(datetime.fromisoformat(text))
    assert set(hrs.TIME_BUCKETS) >= {"opening_drive", "midday", "closing_window"}
    assert hrs.time_bucket("") == "unknown"
    assert hrs.time_bucket("not a time") == "unknown"


def test_the_d1_setups_come_from_the_scan_output_and_only_the_two_buckets():
    """Read from the scan output files, never fetched."""
    import held_run_score as hrs

    mapping = hrs.d1_setups_by_session(
        [
            {"symbol": "AAA", "bucket": "favorite_setup", "scan_date": "2026-09-01", "side": "LONG"},
            {"symbol": "BBB", "bucket": "near_favorite_zone", "scan_date": "2026-09-01", "side": "SHORT"},
            {"symbol": "CCC", "bucket": "watch", "scan_date": "2026-09-01", "side": "LONG"},
            {"symbol": "DDD", "bucket": "favorite_setup", "scan_date": "", "side": "LONG"},
        ]
    )

    assert mapping == {"2026-09-01": {"AAA": {"LONG"}, "BBB": {"SHORT"}}}


def test_nothing_in_this_module_fetches():
    source = (ROOT / "scripts" / "held_run_score.py").read_text(encoding="utf-8")
    for forbidden in ("yfinance", "requests", "ibapi", "download(", "urlopen"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# The floor, and the row suffix
# ---------------------------------------------------------------------------


def test_a_thin_segment_is_reported_and_never_shown_on_an_alert_row():
    """"held 100% / ran 3.2R" over two episodes reads as a strong segment."""
    import held_run_score as hrs

    cells = hrs.build_segments(
        hrs.build_episodes([_row(_event("A")), _row(_event("B"))])
    )

    assert len(cells) == 1, "it is REPORTED - a thin cell is not a missing one"
    assert cells[0]["n"] == 2
    assert cells[0]["meets_floor"] is False
    assert hrs.alert_suffix(cells[0]) == "", "blank below the floor, never a number"


def test_the_suffix_reads_the_way_the_packet_asked():
    import held_run_score as hrs

    cell = {"meets_floor": True, "hold_rate": 0.714, "mean_mfe_r_of_held": 1.93}
    assert hrs.alert_suffix(cell) == "held 71% / ran 1.9R"
    assert hrs.alert_suffix(None) == ""
    assert hrs.alert_suffix({"meets_floor": True, "hold_rate": None}) == ""


def test_the_window_is_the_shared_lately_window_and_carries_no_regime_label():
    """Decision 0016 answer 6: "lately" is a rolling window and nothing else.

    Packet Q1: the window is `evidence_stats.lately_window` (exchange sessions),
    the same one the swing path uses - never "the last N dates in the file".
    """
    import evidence_stats
    import held_run_score as hrs

    rows = []
    for day in range(1, 26):
        date = f"2026-08-{day:02d}"
        rows.append(_row(_event("A", date=date.replace("-", "")), date=date))
    episodes = hrs.build_episodes(rows, as_of="2026-08-25")

    kept = hrs.recent_sessions(episodes, as_of="2026-08-25")
    start, end = evidence_stats.lately_window(end="2026-08-25")
    assert kept == {day for day in (row["trade_date"] for row in rows) if start <= day <= end}
    assert "2026-08-25" in kept
    assert hrs.window_bounds(as_of="2026-08-25") == (start, end)


# ---------------------------------------------------------------------------
# The champion is untouched
# ---------------------------------------------------------------------------


def test_the_champion_never_imports_the_challenger():
    """The tier gate, the mutes and the PROVEN stamp keep their own inputs."""
    learning = (ROOT / "scripts" / "bounce_bot_lib" / "learning.py").read_text(
        encoding="utf-8"
    )
    assert "held_run_score" not in learning

    # And the challenger never CALLS the champion's machinery. The check is on
    # code, not on prose: the module's own docstring names `review_policy.json`
    # precisely to say it never reaches it, and a naive substring search would
    # fail on the sentence that promises the thing it is checking.
    source = (ROOT / "scripts" / "held_run_score.py").read_text(encoding="utf-8")
    code = chr(10).join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    body = code.split('"""', 2)[-1]
    for forbidden in ("MUTE_DIMENSIONS", "review_policy", "alert_passes_min_tier"):
        assert forbidden not in body, forbidden
