"""Packet WS-ENV - one D1 market environment label per session, joined by SCAN DATE.

Written by the TESTER on `claude/ws-env-d1-environment`, off
`origin/claude/wishlist-sweep-2026-09-12`, and proven RED there before any of it
existed. The builder makes these pass; it may ADD tests and may not weaken, skip
or delete one.

WISHLIST item 7, as the lead ruled it: one D1 environment label per session for
SPY (QQQ and IWM as extra benchmarks, each its own row), from COMPLETED daily
bars only, the rule in ONE pure module under `scripts/indicators/`, versioned;
an append-only daily store never backfilled under a different rule without a
version bump; the label joined to every swing outcome by SCAN DATE
(point-in-time: the label of the scan date, never the exit date); `unknown` its
own cell, never pooled. **Shadow evidence only** - zero detector, score, alert
or Focus influence, and the label only LABELS a readout.

===========================================================================
THE API THESE TESTS PIN
===========================================================================

`scripts/indicators/d1_environment.py` - PURE (bars in, a frozen tuple out; no
clock, no I/O, no import from `master_avwap_lib`)::

    RULE_VERSION = "d1_environment_v1"
    LOOKBACK_SESSIONS = 10
    ATR_SESSIONS = 14
    SMA_SESSIONS = 20
    COMPRESSION_RANGE_ATR = 3.0
    TREND_SLOPE_ATR = 0.5
    WARMUP_SESSIONS = 34

    D1Environment(label, rule_version, as_of_session, range_atr, slope_atr,
                  sma20, atr14, bars_used, reason)          # frozen
    classify_environment(bars) -> D1Environment

**The one ambiguity in the packet, resolved here and pinned by the golden.**
The packet says ATR14 may be taken "over the bars before the window ... - state
it". It is STATED: `atr14` is Wilder's ATR at the LAST supplied bar computed
over the WHOLE supplied series (the same recurrence
`scripts/indicators/atr.wilder_atr` already owns), which includes the 10-session
range window. Every number in the golden below was hand-computed under that
definition. An implementation that stops the ATR before the window produces
different numbers and this file fails.

The rule, in order - **the compression test comes FIRST**, so a session may be
`compressed` while its slope and its close both say `trending_up`::

    len(bars) < 34                      -> unknown, reason "warmup"
    atr14 unmeasurable                  -> unknown, reason "unmeasurable"
    range_atr <= 3.0                    -> compressed
    slope_atr > +0.5 and close > sma20  -> trending_up
    slope_atr < -0.5 and close < sma20  -> trending_down
    otherwise                           -> mixed

where `range_atr` = (max high - min low over the last 10 bars) / atr14 and
`slope_atr` = (SMA20 at the last bar - SMA20 ten sessions earlier) / atr14.

`scripts/d1_environment_store.py` - append-only JSONL under the
`project_paths.D1_ENVIRONMENT_FILE` named constant (shared home)::

    BENCHMARKS = ("SPY", "QQQ", "IWM")
    ROW_FIELDS  (session, benchmark, label, rule_version, range_atr, slope_atr,
                 sma20, atr14, bars_used, reason, bars_through, written_at,
                 source)
    append_environment(env, *, benchmark, bars_through, source="scan",
                       path=None) -> bool      # False when already written
    label_for_session(session, benchmark="SPY", rule_version=RULE_VERSION,
                      path=None) -> str        # "unknown" when absent
    labels_by_session(benchmark="SPY", rule_version=RULE_VERSION,
                      path=None) -> dict       # ONE read, cached by mtime
    read_rows(path=None) -> list[dict]
    main(argv) -> int                          # `backfill`, DRY RUN by default

`scripts/d1_environment_join.py`::

    attach_environment(rows, *, date_field="scan_date", benchmark="SPY",
                       rule_version=RULE_VERSION, labels=None, path=None)
        # adds `d1_environment` to each row IN PLACE and returns the SAME list

`scripts/master_avwap_lib/runner.py`::

    record_d1_environment(ib=None, *, now, path=None, benchmarks=None)
        -> dict[str, str]   # {benchmark: label}, for the log line only
        # called as a sibling of `bridge_earnings_anchor_caches_to_csv`,
        # failure LOGGED never raised

`scripts/research_results.py`::

    ENVIRONMENT_SECTION_KEY = "by_environment"
    environment_section(rows, *, benchmark="SPY", rule_version=...)
        -> ResultsSection
    build_results_view(..., environment_rows=None)
        # bot x swing gains the section; every other selection is untouched

`scripts/ui/panels/research_results_panel.py`::

    _read_environment_rows(as_of) -> list[dict]     # on the worker
    _read(...)                                      # calls it for bot/swing ONLY

===========================================================================
THE FIXTURES
===========================================================================

`tests/fixtures/ws_env_spy_daily.csv` - 184 REAL SPY daily bars
(2025-12-17 .. 2026-09-11), copied read-only from the desk's own
`%LOCALAPPDATA%\\TradingBotV3\\machine_cache\\daily_bars\\SPY.csv` on
2026-09-12 and reduced to datetime/open/high/low/close.

`tests/fixtures/ws_env_spy_d1_environment_golden.csv` - the expected label for
EVERY one of those 184 sessions, computed POINT-IN-TIME (session i sees bars
0..i and nothing after), by a hand-written reference implementation of the rule
above that imports nothing from `scripts/`. It is **not** generated by the code
under test, which does not exist yet. Its distribution is 33 `unknown`
(the warm-up), 69 `compressed`, 43 `mixed`, 27 `trending_up`, 12
`trending_down` - all five labels are exercised on real bars, and 2026-09-11
lands at `range_atr = 3.0030`, three thousandths the wrong side of the
compression threshold.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import logging
import os
import sys
import textwrap
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPY_BARS_FIXTURE = FIXTURES / "ws_env_spy_daily.csv"
SPY_GOLDEN_FIXTURE = FIXTURES / "ws_env_spy_d1_environment_golden.csv"

RULE = "d1_environment_v1"


# ---------------------------------------------------------------------------
# fixture loading
# ---------------------------------------------------------------------------


def _recorded_spy_bars() -> list[dict]:
    """The 184 recorded SPY daily bars, oldest first, as dict bars."""
    bars: list[dict] = []
    with open(SPY_BARS_FIXTURE, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            bars.append(
                {
                    "dt": row["datetime"][:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
    return bars


def _golden_labels() -> list[tuple[str, str]]:
    with open(SPY_GOLDEN_FIXTURE, newline="", encoding="utf-8-sig") as handle:
        return [(row["session"], row["label"]) for row in csv.DictReader(handle)]


def _synthetic_bars(closes, *, half_range=1.0, start_day=2):
    """Hand-built daily bars: high = close + h, low = close - h.

    With a constant close-to-close step of 1.0 and h = 1.0 every true range is
    exactly 2.0, so ATR14 is exactly 2.0 and every expectation below is exact
    arithmetic rather than a number read off a run.
    """
    bars = []
    for index, close in enumerate(closes):
        day = start_day + index
        bars.append(
            {
                "dt": f"2026-01-{day:02d}" if day <= 31 else f"2026-02-{day - 31:02d}",
                "open": float(close),
                "high": float(close) + half_range,
                "low": float(close) - half_range,
                "close": float(close),
            }
        )
    return bars


# ---------------------------------------------------------------------------
# item 1 - the pure rule and its golden
# ---------------------------------------------------------------------------


def test_the_named_thresholds_are_the_packets_and_the_rule_is_versioned():
    from indicators import d1_environment

    assert d1_environment.RULE_VERSION == RULE
    assert d1_environment.LOOKBACK_SESSIONS == 10
    assert d1_environment.ATR_SESSIONS == 14
    assert d1_environment.SMA_SESSIONS == 20
    assert d1_environment.COMPRESSION_RANGE_ATR == 3.0
    assert d1_environment.TREND_SLOPE_ATR == 0.5
    assert d1_environment.WARMUP_SESSIONS == 34


def test_the_pure_module_imports_no_engine_and_opens_no_file():
    """Indicators are pure: bars in, tuples out. No `legacy`, no provider, no I/O."""
    from indicators import d1_environment

    source = Path(inspect.getfile(d1_environment)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {
        "master_avwap_lib",
        "bounce_bot_lib",
        "legacy",
        "yfinance",
        "ibapi",
        "project_paths",
        "d1_environment_store",
    }
    assert not (imported & forbidden), f"pure module imports {sorted(imported & forbidden)}"
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in called


def test_the_golden_spy_session_labels_match_the_hand_computed_fixture():
    """Every one of 184 recorded SPY sessions, labelled POINT-IN-TIME.

    Session i is classified from bars 0..i only. The expectations are the
    hand-written reference's, never the module's.
    """
    from indicators.d1_environment import classify_environment

    bars = _recorded_spy_bars()
    golden = _golden_labels()
    assert len(bars) == 184 and len(golden) == 184

    produced = []
    for index in range(len(bars)):
        env = classify_environment(bars[: index + 1])
        produced.append((bars[index]["dt"], env.label))
    assert produced == golden


def test_the_golden_measured_fields_are_the_hand_computed_numbers():
    """Six picked sessions, to the last bit. A changed formula cannot survive this."""
    from indicators.d1_environment import classify_environment

    bars = _recorded_spy_bars()
    index_of = {bar["dt"]: position for position, bar in enumerate(bars)}

    # (session, label, range_atr, slope_atr, sma20, atr14, bars_used)
    expected = (
        ("2026-02-05", "mixed", 3.078974988029741, 0.20889553660283078, 690.318, 7.161474219740262, 34),
        ("2026-03-19", "trending_down", 3.0839399513749917, -0.9263306682606458, 677.716, 9.14090431217099, 63),
        ("2026-04-17", "trending_up", 6.613460234443287, 0.8209942849328791, 666.617, 9.27351156972107, 83),
        ("2026-09-09", "compressed", 2.6123168933287135, 0.5767894309980219, 768.2914947509765, 6.05592964797632, 182),
        ("2026-09-10", "mixed", 3.0920654028648573, 0.15394030542935183, 767.55849609375, 6.034792513522654, 183),
        ("2026-09-11", "mixed", 3.0030207497941053, -0.1965005472158221, 766.8789947509765, 6.2137343359384465, 184),
    )
    for session, label, range_atr, slope_atr, sma20, atr14, bars_used in expected:
        env = classify_environment(bars[: index_of[session] + 1])
        assert env.label == label, session
        assert env.as_of_session == session
        assert env.rule_version == RULE
        assert env.bars_used == bars_used, session
        assert env.range_atr == pytest.approx(range_atr, rel=0, abs=1e-10), session
        assert env.slope_atr == pytest.approx(slope_atr, rel=0, abs=1e-9), session
        assert env.sma20 == pytest.approx(sma20, rel=0, abs=1e-9), session
        assert env.atr14 == pytest.approx(atr14, rel=0, abs=1e-11), session


def test_compression_is_decided_before_the_trend_so_a_quiet_uptrend_is_compressed():
    """2026-04-29 on the real bars: slope_atr 3.84 and close above SMA20 - and
    `compressed`, because range_atr is 2.19. Order of the branches, not taste."""
    from indicators.d1_environment import classify_environment

    bars = _recorded_spy_bars()
    index_of = {bar["dt"]: position for position, bar in enumerate(bars)}
    env = classify_environment(bars[: index_of["2026-04-29"] + 1])

    assert env.label == "compressed"
    assert env.range_atr == pytest.approx(2.1904852812087787, abs=1e-10)
    assert env.slope_atr == pytest.approx(3.842764501504945, abs=1e-9)
    assert bars[index_of["2026-04-29"]]["close"] > env.sma20


def test_the_compression_threshold_is_inclusive_at_three_atr():
    """Hand arithmetic, no fixture: 33 flat bars of TR 2.0, then one wide bar.

    A last bar of low 100 / high 107 makes ATR14 = (2*13 + 7)/14 = 33/14 and a
    10-session range of 7.0, so range_atr = 98/33 = 2.9697 -> compressed. Widen
    that bar to high 108 and ATR14 = 34/14 with a range of 8.0, so range_atr =
    112/34 = 3.2941 -> NOT compressed. The pair straddles the threshold with
    nothing but integers behind it.
    """
    from indicators.d1_environment import classify_environment

    def series(last_high):
        bars = _synthetic_bars([101.0] * 39)
        bars[-1] = {"dt": bars[-1]["dt"], "open": 101.0, "high": last_high, "low": 100.0, "close": 101.0}
        return bars

    tight = classify_environment(series(107.0))
    assert tight.atr14 == pytest.approx(33.0 / 14.0, abs=1e-12)
    assert tight.range_atr == pytest.approx(98.0 / 33.0, abs=1e-12)
    assert tight.label == "compressed"

    wide = classify_environment(series(108.0))
    assert wide.atr14 == pytest.approx(34.0 / 14.0, abs=1e-12)
    assert wide.range_atr == pytest.approx(112.0 / 34.0, abs=1e-12)
    assert wide.slope_atr == pytest.approx(0.0, abs=1e-12)
    assert wide.label == "mixed"


def test_a_clean_uptrend_and_a_clean_downtrend_are_named_by_exact_arithmetic():
    """40 bars stepping 1.0 a session: ATR14 is exactly 2.0, slope_atr exactly
    +/-5.0, range_atr exactly 5.5. Nothing here was read off a run."""
    from indicators.d1_environment import classify_environment

    up = classify_environment(_synthetic_bars([100.0 + step for step in range(40)]))
    assert up.atr14 == pytest.approx(2.0, abs=1e-12)
    assert up.range_atr == pytest.approx(5.5, abs=1e-12)
    assert up.slope_atr == pytest.approx(5.0, abs=1e-12)
    assert up.sma20 == pytest.approx(129.5, abs=1e-12)
    assert up.label == "trending_up"

    down = classify_environment(_synthetic_bars([140.0 - step for step in range(40)]))
    assert down.atr14 == pytest.approx(2.0, abs=1e-12)
    assert down.range_atr == pytest.approx(5.5, abs=1e-12)
    assert down.slope_atr == pytest.approx(-5.0, abs=1e-12)
    assert down.sma20 == pytest.approx(110.5, abs=1e-12)
    assert down.label == "trending_down"


def test_fewer_than_thirty_four_completed_bars_is_unknown_with_reason_warmup():
    from indicators.d1_environment import classify_environment

    bars = _recorded_spy_bars()
    short = classify_environment(bars[:33])
    assert short.label == "unknown"
    assert short.reason == "warmup"
    assert short.range_atr is None and short.slope_atr is None
    assert short.sma20 is None and short.atr14 is None
    assert short.bars_used == 33
    assert short.as_of_session == bars[32]["dt"]
    assert short.rule_version == RULE

    # One more bar and it measures. 34 is the warm-up, not 35.
    assert classify_environment(bars[:34]).label == "mixed"
    assert classify_environment([]).label == "unknown"
    assert classify_environment([]).reason == "warmup"


def test_an_unreadable_price_series_is_unknown_and_never_a_guess():
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    from indicators.d1_environment import classify_environment

    bars = _synthetic_bars([101.0] * 40)
    for bar in bars:
        bar["high"] = float("nan")
        bar["low"] = float("nan")
        bar["close"] = float("nan")
    env = classify_environment(bars)

    assert env.label == "unknown"
    assert env.reason == "unmeasurable"
    assert env.atr14 is None
    assert env.range_atr is None
    assert env.slope_atr is None


def test_the_environment_result_is_immutable():
    from indicators.d1_environment import classify_environment

    env = classify_environment(_recorded_spy_bars())
    with pytest.raises(Exception):
        env.label = "trending_up"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# item 2 - the append-only store
# ---------------------------------------------------------------------------


def _spy_env(session: str | None = None):
    from indicators.d1_environment import classify_environment

    bars = _recorded_spy_bars()
    if session is not None:
        index_of = {bar["dt"]: position for position, bar in enumerate(bars)}
        bars = bars[: index_of[session] + 1]
    return classify_environment(bars)


def test_the_store_path_is_a_named_project_paths_constant_in_the_shared_home():
    import project_paths

    path = project_paths.D1_ENVIRONMENT_FILE
    assert path.suffix == ".jsonl"
    assert path.parent == project_paths.PERSISTENT_DATA_DIR
    # The warehouse lives elsewhere by contract; this is desk evidence.
    assert "research" not in str(path).lower()


def test_the_store_row_carries_the_measured_fields_and_what_it_read_through(tmp_path):
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    env = _spy_env("2026-09-09")
    assert store.append_environment(env, benchmark="SPY", bars_through="2026-09-09", path=path) is True

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "2026-09-09"
    assert row["benchmark"] == "SPY"
    assert row["label"] == "compressed"
    assert row["rule_version"] == RULE
    assert row["bars_through"] == "2026-09-09"
    assert row["source"] == "scan"
    assert row["bars_used"] == 182
    assert row["range_atr"] == pytest.approx(2.6123168933287135, abs=1e-10)
    assert row["slope_atr"] == pytest.approx(0.5767894309980219, abs=1e-9)
    assert row["sma20"] == pytest.approx(768.2914947509765, abs=1e-9)
    assert row["atr14"] == pytest.approx(6.05592964797632, abs=1e-11)
    # `written_at` is AWARE - a naive stamp is a second opinion about the clock.
    written = datetime.fromisoformat(str(row["written_at"]))
    assert written.tzinfo is not None and written.utcoffset() is not None


def test_the_store_never_rewrites_a_session_for_one_benchmark_and_version(tmp_path):
    """Append-only. The second write for the same key changes nothing at all."""
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    env = _spy_env("2026-09-09")
    assert store.append_environment(env, benchmark="SPY", bars_through="2026-09-09", path=path) is True
    first = path.read_bytes()

    # A LATER session is a new key and is written.
    assert (
        store.append_environment(
            _spy_env("2026-09-11"), benchmark="SPY", bars_through="2026-09-11", path=path
        )
        is True
    )
    # A DIFFERENT reading of a session ALREADY written - a re-run, a repaired
    # bar file - changes nothing at all.
    assert (
        store.append_environment(
            _spy_env("2026-09-11"), benchmark="SPY", bars_through="2026-09-11", path=path
        )
        is False
    )
    assert (
        store.append_environment(env, benchmark="SPY", bars_through="2026-09-10", path=path)
        is False
    )
    rows = store.read_rows(path=path)
    assert [row["session"] for row in rows if row["benchmark"] == "SPY"] == [
        "2026-09-09",
        "2026-09-11",
    ]
    # Append-only: the first bytes on disk are still the first bytes on disk.
    assert path.read_bytes()[: len(first)] == first
    assert store.label_for_session("2026-09-09", benchmark="SPY", path=path) == "compressed"


def test_each_benchmark_keeps_its_own_row_for_the_same_session(tmp_path):
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    env = _spy_env("2026-09-09")
    assert store.append_environment(env, benchmark="SPY", bars_through="2026-09-09", path=path) is True
    assert store.append_environment(env, benchmark="QQQ", bars_through="2026-09-09", path=path) is True
    assert store.append_environment(env, benchmark="IWM", bars_through="2026-09-09", path=path) is True
    assert len(store.read_rows(path=path)) == 3
    assert store.BENCHMARKS == ("SPY", "QQQ", "IWM")


def test_a_second_rule_version_writes_beside_the_first_and_never_over_it(tmp_path):
    """A store is never backfilled under a different rule without a version bump,
    and the bump is what lets both live in one file."""
    import dataclasses

    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    env = _spy_env("2026-09-09")
    assert store.append_environment(env, benchmark="SPY", bars_through="2026-09-09", path=path) is True

    v2 = dataclasses.replace(env, rule_version="d1_environment_v2", label="trending_up")
    assert store.append_environment(v2, benchmark="SPY", bars_through="2026-09-09", path=path) is True

    rows = store.read_rows(path=path)
    assert len(rows) == 2
    assert store.label_for_session("2026-09-09", benchmark="SPY", path=path) == "compressed"
    assert (
        store.label_for_session(
            "2026-09-09", benchmark="SPY", rule_version="d1_environment_v2", path=path
        )
        == "trending_up"
    )


def test_label_for_session_returns_unknown_for_an_absent_session(tmp_path):
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    assert store.label_for_session("2026-09-09", benchmark="SPY", path=path) == "unknown"
    store.append_environment(_spy_env("2026-09-09"), benchmark="SPY", bars_through="2026-09-09", path=path)
    # A session nobody labelled, a benchmark nobody wrote, a version nobody ran.
    assert store.label_for_session("2026-09-10", benchmark="SPY", path=path) == "unknown"
    assert store.label_for_session("2026-09-09", benchmark="QQQ", path=path) == "unknown"
    assert store.label_for_session("2026-09-09", benchmark="SPY", rule_version="nope", path=path) == "unknown"
    assert store.labels_by_session(benchmark="SPY", path=path) == {"2026-09-09": "compressed"}


def test_labels_by_session_is_one_read_cached_by_the_files_mtime(tmp_path):
    """Cheap enough to call from a worker: the same mtime is answered from the
    cache, a newer mtime is re-read."""
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    store.append_environment(_spy_env("2026-09-09"), benchmark="SPY", bars_through="2026-09-09", path=path)
    assert store.labels_by_session(benchmark="SPY", path=path) == {"2026-09-09": "compressed"}

    stat = os.stat(path)
    rewritten = path.read_text(encoding="utf-8").replace('"compressed"', '"trending_up"')
    path.write_text(rewritten, encoding="utf-8")
    os.utime(path, (stat.st_atime, stat.st_mtime))
    assert store.labels_by_session(benchmark="SPY", path=path) == {"2026-09-09": "compressed"}

    os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))
    assert store.labels_by_session(benchmark="SPY", path=path) == {"2026-09-09": "trending_up"}


# ---------------------------------------------------------------------------
# item 3 - the runner hook
# ---------------------------------------------------------------------------


def _bars_frame(bars):
    import pandas as pd

    return pd.DataFrame(
        {
            "datetime": pd.to_datetime([bar["dt"] for bar in bars]),
            "open": [bar["open"] for bar in bars],
            "high": [bar["high"] for bar in bars],
            "low": [bar["low"] for bar in bars],
            "close": [bar["close"] for bar in bars],
            "volume": [1_000_000] * len(bars),
        }
    )


@pytest.fixture()
def spy_only_fetch(monkeypatch):
    """A fake `fetch_daily_bars` that serves the recorded SPY bars to every name."""
    from master_avwap_lib import runner

    bars = _recorded_spy_bars()
    state = {"through": "2026-09-11", "calls": []}

    def _fetch(ib, symbol, days):
        state["calls"].append(str(symbol).upper())
        index_of = {bar["dt"]: position for position, bar in enumerate(bars)}
        return _bars_frame(bars[: index_of[state["through"]] + 1])

    monkeypatch.setattr(runner, "fetch_daily_bars", _fetch)
    return state


def test_the_hook_excludes_a_forming_last_bar_and_labels_through_yesterday(
    spy_only_fetch, tmp_path
):
    """Mid-session on 2026-09-10 the desk has a FORMING 09-10 daily bar.

    Through 09-09 the label is `compressed`; through 09-10 it is `mixed`. A hook
    that keeps the forming bar prints `mixed` on a session that is not over.
    """
    from master_avwap_lib import runner

    spy_only_fetch["through"] = "2026-09-10"
    path = tmp_path / "d1_environment.jsonl"
    labels = runner.record_d1_environment(
        None, now=datetime(2026, 9, 10, 13, 0), path=path, benchmarks=("SPY",)
    )

    assert labels == {"SPY": "compressed"}
    rows = [row for row in json.loads("[" + ",".join(path.read_text(encoding="utf-8").splitlines()) + "]")]
    assert len(rows) == 1
    assert rows[0]["session"] == "2026-09-09"
    assert rows[0]["bars_through"] == "2026-09-09"
    assert rows[0]["bars_used"] == 182
    assert rows[0]["source"] == "scan"


def test_the_hook_counts_a_finished_session_and_writes_one_row_per_benchmark(
    spy_only_fetch, tmp_path, caplog
):
    from master_avwap_lib import runner

    spy_only_fetch["through"] = "2026-09-10"
    path = tmp_path / "d1_environment.jsonl"
    with caplog.at_level(logging.INFO):
        labels = runner.record_d1_environment(
            None, now=datetime(2026, 9, 11, 9, 0), path=path, benchmarks=("SPY", "QQQ", "IWM")
        )

    assert labels == {"SPY": "mixed", "QQQ": "mixed", "IWM": "mixed"}
    assert spy_only_fetch["calls"] == ["SPY", "QQQ", "IWM"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sorted(row["benchmark"] for row in rows) == ["IWM", "QQQ", "SPY"]
    assert {row["session"] for row in rows} == {"2026-09-10"}

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "D1 environment:" in text
    assert "SPY=mixed" in text
    assert RULE in text
    assert "2026-09-10" in text


def test_the_hook_failure_never_fails_the_scan(monkeypatch, tmp_path):
    """Evidence stores are never allowed to cost the thing they record."""
    from master_avwap_lib import runner

    def _boom(ib, symbol, days):
        raise RuntimeError("IB said no")

    monkeypatch.setattr(runner, "fetch_daily_bars", _boom)
    path = tmp_path / "d1_environment.jsonl"
    labels = runner.record_d1_environment(
        None, now=datetime(2026, 9, 11, 9, 0), path=path, benchmarks=("SPY",)
    )
    assert labels == {}
    assert not path.exists() or path.read_text(encoding="utf-8").strip() == ""


def test_the_hook_is_called_once_beside_the_anchor_bridge(spy_only_fetch):
    """Wiring, read structurally: one call, after the caches are saved and the
    anchor bridge has run, before `save_history`."""
    from master_avwap_lib import runner

    tree = ast.parse(textwrap.dedent(inspect.getsource(runner._run_master_impl)))
    lines: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, []).append(node.lineno)

    assert len(lines.get("record_d1_environment", [])) == 1, "exactly one call site"
    bridge = min(lines["bridge_earnings_anchor_caches_to_csv"])
    hook = lines["record_d1_environment"][0]
    history = min(lines["save_history"])
    assert bridge < hook < history


# ---------------------------------------------------------------------------
# item 4 - the backfill CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def cached_spy_csv(monkeypatch, tmp_path):
    """The recorded SPY bars where the desk's own daily-bar cache keeps them."""
    import project_paths

    cache_dir = tmp_path / "daily_bars"
    cache_dir.mkdir()
    (cache_dir / "SPY.csv").write_text(
        SPY_BARS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(project_paths, "DAILY_BARS_CACHE_DIR", cache_dir)
    return cache_dir


def test_the_backfill_is_a_dry_run_by_default_and_writes_nothing(
    cached_spy_csv, tmp_path, capsys
):
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    code = store.main(
        ["backfill", "--benchmark", "SPY", "--since", "2026-09-01", "--path", str(path)]
    )
    assert code == 0
    assert not path.exists()
    printed = capsys.readouterr().out
    # It prints where it is pointed BEFORE it does anything (2026-09-05 rule).
    assert "DATA_DIR" in printed
    assert "compressed" in printed and "mixed" in printed


def test_the_backfill_apply_labels_each_past_session_point_in_time(
    cached_spy_csv, tmp_path
):
    """2026-09-01..2026-09-11 is eight sessions: six `compressed`, then `mixed`
    on 09-10 and 09-11. A backfill that labels every session from the WHOLE
    series would print one label eight times."""
    import d1_environment_store as store

    path = tmp_path / "d1_environment.jsonl"
    code = store.main(
        [
            "backfill",
            "--benchmark",
            "SPY",
            "--since",
            "2026-09-01",
            "--path",
            str(path),
            "--apply",
        ]
    )
    assert code == 0
    rows = store.read_rows(path=path)
    assert [(row["session"], row["label"]) for row in rows] == [
        ("2026-09-01", "compressed"),
        ("2026-09-02", "compressed"),
        ("2026-09-03", "compressed"),
        ("2026-09-04", "compressed"),
        ("2026-09-08", "compressed"),
        ("2026-09-09", "compressed"),
        ("2026-09-10", "mixed"),
        ("2026-09-11", "mixed"),
    ]
    assert {row["source"] for row in rows} == {"backfill"}
    assert {row["rule_version"] for row in rows} == {RULE}
    # Re-running it appends nothing: append-only means never a second row.
    assert (
        store.main(
            ["backfill", "--benchmark", "SPY", "--since", "2026-09-01", "--path", str(path), "--apply"]
        )
        == 0
    )
    assert len(store.read_rows(path=path)) == 8


# ---------------------------------------------------------------------------
# item 5a - the join, by SCAN DATE
# ---------------------------------------------------------------------------


def _store_with_september(path):
    import d1_environment_store as store

    for session in ("2026-09-09", "2026-09-10", "2026-09-11"):
        store.append_environment(
            _spy_env(session), benchmark="SPY", bars_through=session, path=path
        )
    return path


def test_attach_environment_joins_on_the_scan_date_and_never_the_exit_date(tmp_path):
    """Point-in-time: the label of the scan date, never the exit date.

    The row is scanned on 2026-09-09 (`compressed`) and exits 2026-09-11
    (`mixed`). Both sessions are in the store, so a join on the wrong column is
    not a blank - it is a confidently wrong label.
    """
    import d1_environment_join as join

    path = _store_with_september(tmp_path / "d1_environment.jsonl")
    rows = [
        {"observation_id": "a", "scan_date": "2026-09-09", "future_scan_date": "2026-09-11", "win": "True"},
        {"observation_id": "b", "scan_date": "2026-09-10", "future_scan_date": "2026-09-11", "win": "False"},
        {"observation_id": "c", "scan_date": "2026-08-03", "future_scan_date": "2026-09-09", "win": "True"},
    ]
    out = join.attach_environment(rows, date_field="scan_date", benchmark="SPY", path=path)

    assert [row["d1_environment"] for row in out] == ["compressed", "mixed", "unknown"]
    # No copy: the caller's own list and the caller's own dicts.
    assert out is rows
    assert out[0] is rows[0]


def test_attach_environment_reads_a_timestamped_scan_date_and_a_present_but_empty_one(
    tmp_path,
):
    """The live CSV carries `2026-09-09` but an older row has the column PRESENT
    AND EMPTY, and a rebuilt one carries a full timestamp."""
    import d1_environment_join as join

    path = _store_with_september(tmp_path / "d1_environment.jsonl")
    rows = [
        {"scan_date": "2026-09-09T13:02:12"},
        {"scan_date": ""},
        {"scan_date": None},
    ]
    join.attach_environment(rows, path=path)
    assert [row["d1_environment"] for row in rows] == ["compressed", "unknown", "unknown"]


# ---------------------------------------------------------------------------
# item 5b - the Results readout
# ---------------------------------------------------------------------------


def _outcome_rows():
    """Tier-outcome rows shaped like the live CSV, with a KNOWN true answer.

    trending_up  / LONG : 20-10  -> 66.7%, n=30,  Wilson lb 0.4878
    compressed   / LONG : 62-38  -> 62.0%, n=100, Wilson lb 0.5221
    unknown      / SHORT:  5-5   -> 50.0%, n=10,  below the n=30 floor
    mixed        / SHORT:  1-1   -> 50.0%, n=2,   below the floor

    So ranking by the RATE puts trending_up first and ranking by the BOUND puts
    compressed first. The page sorts by the bound.
    """
    rows = []
    plan = (
        ("trending_up", "LONG", "2026-09-01", 20, 10),
        ("compressed", "LONG", "2026-09-02", 62, 38),
        ("unknown", "SHORT", "2026-09-03", 5, 5),
        ("mixed", "SHORT", "2026-09-04", 1, 1),
    )
    counter = 0
    for environment, side, scan_date, wins, losses in plan:
        for index in range(wins + losses):
            counter += 1
            win = index < wins
            rows.append(
                {
                    "observation_id": f"obs-{counter}",
                    "scan_date": scan_date,
                    "horizon_sessions": "5",
                    "side": side,
                    "symbol": f"SYM{counter}",
                    "win": "True" if win else "False",
                    "side_return_pct": "1.5" if win else "-1.0",
                    "stale_horizon": "False",
                    "d1_environment": environment,
                }
            )
    # An old row with the verdict column PRESENT AND EMPTY. Unmeasured is not a
    # loss: it must not move compressed/LONG's n off 100.
    rows.append(
        {
            "observation_id": "obs-blank",
            "scan_date": "2026-09-02",
            "horizon_sessions": "5",
            "side": "LONG",
            "symbol": "BLANK",
            "win": "",
            "side_return_pct": "",
            "stale_horizon": "False",
            "d1_environment": "compressed",
        }
    )
    return rows


def test_the_environment_section_is_one_row_per_environment_and_side_sorted_by_the_bound():
    import research_results
    import swing_headline

    section = research_results.environment_section(
        _outcome_rows(), benchmark="SPY", rule_version=RULE
    )

    assert section.key == research_results.ENVIRONMENT_SECTION_KEY == "by_environment"
    assert "By environment" in section.title
    assert "SPY" in section.title and RULE in section.title

    keys = [(row.values["environment"], row.values["side"]) for row in section.rows]
    assert keys == [
        ("compressed", "LONG"),
        ("trending_up", "LONG"),
        ("unknown", "SHORT"),
        ("mixed", "SHORT"),
    ]

    by_key = {
        (row.values["environment"], row.values["side"]): row.values for row in section.rows
    }
    assert by_key[("compressed", "LONG")]["n"] == 100
    assert by_key[("compressed", "LONG")]["win_rate"] == pytest.approx(0.62, abs=1e-12)
    assert by_key[("compressed", "LONG")]["win_rate_lb"] == pytest.approx(
        swing_headline.wilson_lower_bound(62, 100), abs=1e-12
    )
    assert by_key[("trending_up", "LONG")]["n"] == 30
    assert by_key[("trending_up", "LONG")]["win_rate"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    # The trap: the higher RATE sorts SECOND because its bound is lower.
    assert by_key[("trending_up", "LONG")]["win_rate"] > by_key[("compressed", "LONG")]["win_rate"]
    assert by_key[("trending_up", "LONG")]["win_rate_lb"] < by_key[("compressed", "LONG")]["win_rate_lb"]


def test_the_environment_section_leads_with_the_rate_and_says_what_the_rate_is():
    """Win rate FIRST, with n and the ONE Wilson bound. And these rows are
    `favorable_direction` (ST1), so the column may not be headed 'Win %'."""
    import research_results
    import swing_headline

    section = research_results.environment_section(_outcome_rows(), benchmark="SPY", rule_version=RULE)

    assert tuple(section.stats["columns"]) == (
        ("environment", "side") + swing_headline.HEADLINE_COLUMNS
    )
    assert tuple(section.stats["labels"]) == (
        ("Environment", "Side")
        + swing_headline.headline_labels(swing_headline.OUTCOME_KIND_FAVORABLE_DIRECTION)
    )
    for row in section.rows:
        assert row.values["outcome_kind"] == swing_headline.OUTCOME_KIND_FAVORABLE_DIRECTION
        assert row.values["avg_unit"] == "%"
        assert set(swing_headline.HEADLINE_COLUMNS) <= set(row.values)


def test_the_environment_section_keeps_unknown_as_its_own_row_and_pools_nothing():
    import research_results

    rows = _outcome_rows()
    section = research_results.environment_section(rows, benchmark="SPY", rule_version=RULE)

    unknown = [row for row in section.rows if row.values["environment"] == "unknown"]
    assert len(unknown) == 1
    assert unknown[0].values["n"] == 10
    # Every graded row lands in exactly one cell, and `unknown` is a cell, not a
    # bucket folded into the others.
    assert sum(row.values["n"] for row in section.rows) == 30 + 100 + 10 + 2
    assert section.stats["n_unknown"] == 10
    assert section.stats["rule_version"] == RULE
    assert section.stats["benchmark"] == "SPY"


def test_the_environment_section_applies_the_reportable_floor_without_hiding_a_row():
    import evidence_stats
    import research_results

    section = research_results.environment_section(_outcome_rows(), benchmark="SPY", rule_version=RULE)
    by_key = {(row.values["environment"], row.values["side"]): row for row in section.rows}

    assert evidence_stats.MIN_REPORTABLE_N == 30
    assert by_key[("compressed", "LONG")].eligible is True
    assert by_key[("trending_up", "LONG")].eligible is True
    assert by_key[("unknown", "SHORT")].eligible is False
    assert by_key[("mixed", "SHORT")].eligible is False
    for key in (("unknown", "SHORT"), ("mixed", "SHORT")):
        assert by_key[key].reason
        assert by_key[key].values["meets_floor"] is False
    # Floored rows are LABELLED, never dropped, and they sort after the eligible ones.
    assert len(section.rows) == 4


def test_the_environment_section_sentence_names_the_rule_and_what_it_covers():
    import research_results

    section = research_results.environment_section(_outcome_rows(), benchmark="SPY", rule_version=RULE)

    assert section.stats["sessions_covered"] == 3
    assert section.stats["sessions_uncovered"] == 1
    assert RULE in section.sentence
    assert "SPY" in section.sentence
    assert "3" in section.sentence and "1" in section.sentence
    assert "session" in section.sentence.lower()


# ---------------------------------------------------------------------------
# item 5c - the section reaches the page, and the champion sections do not move
# ---------------------------------------------------------------------------


def _bot_snapshot():
    """A small real `EvidenceSnapshot.to_payload()` with both swing kinds."""
    from working_lately import (
        SNAPSHOT_KINDS,
        EvidenceCell,
        EvidenceSnapshot,
        select_cell_leader,
    )

    def cell(**kwargs):
        base = dict(
            outcome_kind="trade_r_representative_exit",
            outcome_version="recent_types_v2",
            knowledge_basis="entry_scan_row_close_to_representative_exit",
            horizon="30d lookback, representative exit",
            window_sessions=20,
            latest_measured_session="2026-09-11",
            n_pending=0,
            n_excluded=0,
            n_symbols=17,
            n_sessions=12,
            top_symbol_share=0.21,
            top_session_share=0.18,
            statistic_name="win rate (closed, unweighted)",
            uncertainty_kind="wilson lower bound",
            namespace="live",
            n_floor=30,
            meets_floor=True,
        )
        base.update(kwargs)
        return EvidenceCell(**base)

    cells = [
        cell(kind="swing_trade_r", side="LONG", family="alpha", n_eligible=613, n_graded=613, statistic=0.72, uncertainty_low=0.61),
        cell(kind="swing_trade_r", side="SHORT", family="beta", n_eligible=614, n_graded=614, statistic=0.64, uncertainty_low=0.53),
        cell(
            kind="swing_favorable",
            side="LONG",
            family="fav_a",
            n_eligible=615,
            n_graded=615,
            statistic=58.0,
            uncertainty_low=49.0,
            outcome_kind="favorable_direction",
            outcome_version="favorable_direction_session_v2",
            knowledge_basis="scan row close to target session close",
            horizon="5 sessions",
            statistic_name="favorable direction (percent)",
            uncertainty_kind="wilson lower bound (percent)",
        ),
    ]
    verdicts = {
        kind: select_cell_leader(
            cells,
            kind=kind,
            last_completed_session=date(2026, 9, 11),
            previous=None,
            source_rows=len([one for one in cells if one.kind == kind]),
        )
        for kind in SNAPSHOT_KINDS
    }
    return EvidenceSnapshot(
        snapshot_id="wsenvfixture000000000000000000000000000",
        as_of="2026-09-11",
        built_at="2026-09-11T17:31:00-04:00",
        cells=tuple(cells),
        verdicts=verdicts,
        sources={},
    ).to_payload()


def test_the_bot_swing_page_gains_the_environment_section_and_nothing_else_does():
    import research_results

    snapshot = _bot_snapshot()
    rows = _outcome_rows()

    swing = research_results.build_results_view(
        population="bot",
        horizon="swing",
        window="recent",
        snapshot=snapshot,
        as_of=date(2026, 9, 11),
        environment_rows=rows,
    )
    assert [section.key for section in swing.sections] == [
        "swing_trade_r",
        "swing_favorable",
        "by_environment",
    ]

    day = research_results.build_results_view(
        population="bot",
        horizon="day",
        window="recent",
        snapshot=snapshot,
        as_of=date(2026, 9, 11),
        environment_rows=rows,
    )
    assert "by_environment" not in [section.key for section in day.sections]

    mine = research_results.build_results_view(
        population="mine",
        horizon="swing",
        window="recent",
        snapshot=snapshot,
        journal_trades=[],
        as_of=date(2026, 9, 11),
        environment_rows=rows,
    )
    assert "by_environment" not in [section.key for section in mine.sections]


def test_the_existing_results_sections_are_byte_for_byte_what_they_were():
    """The champion aggregates are untouched: the same two sections, equal as
    objects, with or without the environment cut, and their lines are the ones
    the CURRENT code produced before any of this existed."""
    import research_results

    snapshot = _bot_snapshot()
    before = research_results.build_results_view(
        population="bot", horizon="swing", window="recent", snapshot=snapshot, as_of=date(2026, 9, 11)
    )
    after = research_results.build_results_view(
        population="bot",
        horizon="swing",
        window="recent",
        snapshot=snapshot,
        as_of=date(2026, 9, 11),
        environment_rows=_outcome_rows(),
    )

    assert before.sections[0] == after.sections[0]
    assert before.sections[1] == after.sections[1]
    assert before.freshness_line == after.freshness_line
    assert before.window_sentence == after.window_sentence
    assert before.window_applies is after.window_applies is False

    # Pinned by running the PRE-CHANGE code on this branch (2026-09-12), never
    # by running the code these tests are written for.
    assert [row.line for row in after.sections[0].rows] == [
        "LONG alpha [swing_trade_r]; win rate (closed, unweighted) 0.72; wilson lower bound >= 0.61;"
        " n=613 (613 graded, 0 pending, 0 excluded); 17 symbol(s) / 12 session(s); top symbol 0.21,"
        " top session 0.18; outcome trade_r_representative_exit (recent_types_v2); basis"
        " entry_scan_row_close_to_representative_exit; horizon 30d lookback, representative exit;"
        " window 20 sessions through 2026-09-11; namespace live",
        "SHORT beta [swing_trade_r]; win rate (closed, unweighted) 0.64; wilson lower bound >= 0.53;"
        " n=614 (614 graded, 0 pending, 0 excluded); 17 symbol(s) / 12 session(s); top symbol 0.21,"
        " top session 0.18; outcome trade_r_representative_exit (recent_types_v2); basis"
        " entry_scan_row_close_to_representative_exit; horizon 30d lookback, representative exit;"
        " window 20 sessions through 2026-09-11; namespace live",
    ]
    assert [row.line for row in after.sections[1].rows] == [
        "LONG fav_a [swing_favorable]; favorable direction (percent) 58; wilson lower bound (percent)"
        " >= 49; n=615 (615 graded, 0 pending, 0 excluded); 17 symbol(s) / 12 session(s); top symbol"
        " 0.21, top session 0.18; outcome favorable_direction (favorable_direction_session_v2); basis"
        " scan row close to target session close; horizon 5 sessions; window 20 sessions through"
        " 2026-09-11; namespace live",
    ]


def test_the_results_panel_reads_the_environment_rows_on_the_worker_for_bot_swing(
    monkeypatch,
):
    """The cut is computed on the Results worker, and only where it is shown -
    a My-trades page never pays for it."""
    from ui.panels import research_results_panel as panel

    calls: list[object] = []

    def _rows(as_of):
        calls.append(as_of)
        return _outcome_rows()

    monkeypatch.setattr(panel, "_read_environment_rows", _rows)
    monkeypatch.setattr(panel, "read_persisted_snapshot", lambda: _bot_snapshot())
    monkeypatch.setattr(panel, "load_trades", lambda: [])

    payload = panel._read(("bot", "swing", "recent"), "recent", None)
    assert len(calls) == 1
    assert "by_environment" in [section.key for section in payload["view"].sections]

    panel._read(("mine", "swing", "recent"), "recent", None)
    panel._read(("bot", "day", "recent"), "recent", None)
    assert len(calls) == 1


def test_the_panels_environment_read_joins_the_store_to_the_outcome_file(
    monkeypatch, tmp_path
):
    """The real read: the tier outcome CSV through `swing_evidence`, the store
    through the join, `unknown` for a session nobody labelled."""
    import project_paths
    from ui.panels import research_results_panel as panel

    store_path = _store_with_september(tmp_path / "d1_environment.jsonl")
    outcomes = tmp_path / "master_avwap_tier_outcomes.csv"
    fields = [
        "observation_id",
        "scan_date",
        "horizon_sessions",
        "side",
        "symbol",
        "win",
        "side_return_pct",
        "stale_horizon",
    ]
    with open(outcomes, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (scan_date, win) in enumerate(
            (("2026-09-09", "True"), ("2026-09-11", "False"), ("2026-08-03", "True"))
        ):
            writer.writerow(
                {
                    "observation_id": f"o{index}",
                    "scan_date": scan_date,
                    "horizon_sessions": "5",
                    "side": "LONG",
                    "symbol": f"S{index}",
                    "win": win,
                    "side_return_pct": "1.0",
                    "stale_horizon": "False",
                }
            )
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_TIER_OUTCOMES_FILE", outcomes)
    monkeypatch.setattr(project_paths, "D1_ENVIRONMENT_FILE", store_path)

    rows = panel._read_environment_rows(date(2026, 9, 11))
    assert [row["d1_environment"] for row in rows] == ["compressed", "mixed", "unknown"]


# ---------------------------------------------------------------------------
# the invariant that binds the whole packet
# ---------------------------------------------------------------------------


def test_nothing_in_this_chain_reaches_a_detector_a_score_or_an_alert():
    """Shadow evidence only. The store, the join and the rule import no engine,
    no alert module and no Focus store."""
    forbidden = {
        "bounce_bot_lib",
        "m5_signal_engines",
        "focus_pick_store",
        "regime_pause_focus",
        "review_learning",
        "alert_center",
        "setup_points",
    }
    for name in ("d1_environment_store", "d1_environment_join"):
        module = __import__(name)
        tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden), f"{name} imports {sorted(imported & forbidden)}"
