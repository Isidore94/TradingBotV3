r"""PCT-3: the LIVE scan's own row builder publishes the measure.

Added by the BUILDER in the first fix round. Review round 1 found the packet's
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

The scan runs IN A SUBPROCESS, and that is not incidental
---------------------------------------------------------

Review round 2 caught what the first version of this file cost: a real scan
writes about thirty-seven files, and `tests/conftest.py` gives the WHOLE SESSION
one shared `TRADINGBOTV3_DATA_DIR`. So running the scan in-process left an
`ai_state`, a priority report, a `d1_features_history.csv` and a
`master_avwap_tier_outcomes.csv` behind in a directory every other test shares -
and `tests/test_r4fix_digest_horizon.py` stops SKIPPING the moment that tier
outcomes file exists, then fails on the empty file this scan wrote. One test's
fixture turned another test red, three thousand tests later.

Monkeypatching the `project_paths` constants would work only for the ones I
remembered. A child process with `TRADINGBOTV3_DATA_DIR` pointed at `tmp_path`
before any `scripts/` import is the whole answer at once, and it is the same
idiom the tester's file uses for the CLI. `test_the_scan_leaves_the_shared_test_
directory_untouched` measures the directory before and after and fails if one
byte lands there.

`tests/conftest.py` points `project_paths` at a test directory; the child points
its own at `tmp_path`; nothing here touches a live store.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SYMBOL = "PCTX"

#: The child prints one line of JSON behind this marker, so the scan's own
#: logging on stdout cannot be mistaken for the payload.
RESULT_MARKER = "PCT3-RESULT::"

CHILD_SOURCE = r'''
"""Run ONE symbol through the real scan, in a home of this process's own.

Every outside door is closed: no IB client, no market-data provider, no
earnings feed, no tracker write, no theta report. What is left is the row
builder, which is the thing under test.
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

SCRATCH = Path(sys.argv[1])
SCRIPTS_DIR = sys.argv[2]
SYMBOL = "PCTX"
RESULT_MARKER = "PCT3-RESULT::"

os.environ["TRADINGBOTV3_DATA_DIR"] = str(SCRATCH / "home")
os.environ["LOCALAPPDATA"] = str(SCRATCH / "localappdata")
os.environ["TRADINGBOT_DIAGNOSTICS_DIR"] = str(SCRATCH / "diag")
os.environ["TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, SCRIPTS_DIR)

import pandas as pd

import project_paths

if "TradingBotData" in str(project_paths.DATA_DIR):
    raise SystemExit(f"ABORT: the child resolved under the live folder: {project_paths.DATA_DIR}")

from master_avwap_lib import runner


def _sessions(count):
    """`count` business days ending on the last completed weekday before today.

    The scan reads "today" off the wall clock, so the frame has to end in the
    real past or the last bar is a forming one and no priority row arrives.
    """
    end = date.today() - timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    return list(pd.bdate_range(end=pd.Timestamp(end), periods=count))


def _compressed_frame():
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


frame, anchor_day = _compressed_frame()
anchor_iso = anchor_day.isoformat()

runner.resolve_master_scan_watchlist_paths = lambda **kwargs: (
    [SCRATCH / "longs.txt"],
    [SCRATCH / "shorts.txt"],
    "test watchlist",
)
runner.load_tickers_from_paths = lambda paths, **kw: (
    [SYMBOL] if "longs" in str(list(paths)[0]) else []
)
runner.append_master_avwap_d1_watchlist_symbols = lambda longs, shorts: (longs, shorts, 0)
runner.load_theta_long_symbols = lambda *a, **k: []
runner.load_scan_earnings_context = lambda symbols: ({SYMBOL: [anchor_iso]}, {SYMBOL: {}})
runner.collect_upcoming_earnings_dates = lambda symbols: {}
runner._maybe_run_setup_tracker_catchup = lambda **kwargs: {}
runner.connect_daily_data_client = lambda **kwargs: None
runner.build_market_regime_snapshot = lambda ib, day: {"label": "mixed"}
runner.fetch_daily_bars = lambda ib, sym, days, **kw: frame.copy()
runner.record_theta_picks = lambda *a, **k: None

result = runner._run_master_impl(update_setup_tracker=False)

ai_state = json.loads(
    Path(project_paths.MASTER_AVWAP_AI_STATE_FILE).read_text(encoding="utf-8")
)
rows = [row for row in (result.get("priority_rows") or []) if row.get("symbol") == SYMBOL]

feature_header = []
features_path = Path(project_paths.D1_FEATURES_FILE)
if features_path.exists():
    with features_path.open("r", encoding="utf-8-sig", newline="") as handle:
        import csv

        feature_header = next(csv.reader(handle), [])

payload = {
    "priority_row": rows[0] if rows else None,
    "ai_state_entry": (ai_state.get("symbols") or {}).get(SYMBOL),
    "ai_state_symbols": sorted(ai_state.get("symbols") or {}),
    "feature_header": feature_header,
}
print(RESULT_MARKER + json.dumps(payload, default=str))
'''


COPY_THROUGH_FIELDS = (
    "compression_score",
    "compression_stdev_atr_ratio",
    "compression_range_atr_ratio",
    "compression_close_range_atr_ratio",
    "compression_rule_version",
)


def _shared_test_dir() -> Path:
    """The ONE directory `tests/conftest.py` hands the whole session."""
    return Path(os.environ["TRADINGBOTV3_DATA_DIR"])


def _listing(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def _run_scan(tmp_path: Path) -> tuple[dict, set[str], set[str]]:
    """The child, and the shared directory's listing either side of it."""
    scratch = tmp_path / "scan"
    for name in ("home", "localappdata", "diag"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    child = scratch / "child_scan.py"
    child.write_text(CHILD_SOURCE, encoding="utf-8")

    shared = _shared_test_dir()
    before = _listing(shared)

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(child), str(scratch), str(SCRIPTS_DIR)],
        capture_output=True,
        text=True,
        timeout=900,
        env=environment,
        cwd=str(SCRIPTS_DIR),
    )
    after = _listing(shared)
    assert completed.returncode == 0, (
        f"the scan child failed ({completed.returncode})\n"
        f"stdout tail:\n{completed.stdout[-4000:]}\nstderr tail:\n{completed.stderr[-4000:]}"
    )
    line = next(
        (
            text[len(RESULT_MARKER) :]
            for text in reversed(completed.stdout.splitlines())
            if text.startswith(RESULT_MARKER)
        ),
        "",
    )
    assert line, f"the child printed no result payload:\n{completed.stdout[-4000:]}"
    return json.loads(line), before, after


@pytest.fixture(scope="module")
def scanned(tmp_path_factory) -> tuple[dict, set[str], set[str]]:
    """One symbol through `runner._run_master_impl`, in a home of its own.

    Module-scoped: a real scan is the expensive part of this file, and every
    test below reads the same run.
    """
    return _run_scan(tmp_path_factory.mktemp("pct3-runner"))


# ---------------------------------------------------------------------------
# The blocker review round 2 found: a fixture that writes into the shared home
# ---------------------------------------------------------------------------


def test_the_scan_leaves_the_shared_test_directory_untouched(scanned):
    """A fixture may not write into the directory the whole suite shares.

    A real scan writes about thirty-seven files. In the shared
    `TRADINGBOTV3_DATA_DIR` one of them is `master_avwap_tier_outcomes.csv`, and
    `tests/test_r4fix_digest_horizon.py` stops SKIPPING the moment that file
    exists - so this fixture turned a passing test red three thousand tests
    later. The child owns its own home; this measures it.
    """
    _payload, before, after = scanned
    leaked = sorted(after - before)
    assert not leaked, (
        "the scan wrote into the session-shared test data directory: " + ", ".join(leaked[:20])
    )


# ---------------------------------------------------------------------------
# What the live row builder must publish
# ---------------------------------------------------------------------------


def test_the_scan_that_writes_ai_state_publishes_the_compression_measure(scanned):
    """The file the desk's setups table merges from carries the numbers.

    This is the assertion the live ai_state failed: 1003 symbols, 35 flagged,
    **0** with `compression_score`.
    """
    payload, _before, _after = scanned
    entry = payload["ai_state_entry"]
    assert entry, f"the scan wrote no ai_state entry for {SYMBOL}: {payload['ai_state_symbols']}"

    missing = [field for field in COPY_THROUGH_FIELDS if field not in entry]
    assert not missing, f"the ai_state symbol entry is missing {missing}"
    assert entry["compression_rule_version"] == "anchor_compression_v1"
    assert isinstance(entry["compression_score"], int)
    for field in COPY_THROUGH_FIELDS[1:4]:
        assert entry[field] is not None, f"{field} was published as a blank"


def test_the_scan_s_priority_row_carries_them_too(scanned):
    """The row the report, the tracker and every downstream reader are built from."""
    payload, _before, _after = scanned
    row = payload["priority_row"]
    assert row, "the scan produced no priority row for the synthetic symbol"
    missing = [field for field in COPY_THROUGH_FIELDS if field not in row]
    assert not missing, f"the priority row is missing {missing}"


def test_the_scan_evaluates_the_v1_break_rule_on_every_row(scanned):
    """`compression_break_recent` and its rule version travel together.

    The fixture is compressed rather than breaking, so the VALUE here is False -
    what this pins is that the rule RAN, which is what makes the tag reachable
    at all from the live scan.
    """
    payload, _before, _after = scanned
    for carrier, where in (
        (payload["priority_row"], "the priority row"),
        (payload["ai_state_entry"], "the ai_state entry"),
    ):
        assert "compression_break_recent" in carrier, where
        assert carrier["compression_break_rule_version"] == "compression_break_v1", where


def test_the_feature_csv_the_scan_wrote_carries_the_measure_columns(scanned):
    """Read the file, not the source.

    `df_features` is built with `columns=feature_columns`, so a field the row
    carries and that allowlist does not is silently dropped - which is how
    `assigned_tier` became a decision nothing could read. The proof is the
    header of the `d1_features.csv` this scan actually wrote; the earlier
    version of this test string-matched `inspect.getsource`, which would pass on
    a list that was built and never used.
    """
    payload, _before, _after = scanned
    header = payload["feature_header"]
    assert header, "the scan wrote no d1_features.csv"
    missing = [
        field
        for field in (*COPY_THROUGH_FIELDS, "compression_break_recent", "compression_break_rule_version")
        if field not in header
    ]
    assert not missing, f"the feature CSV header is missing {missing}"


# ---------------------------------------------------------------------------
# The two "or nothing at all" rules, at both helpers
# ---------------------------------------------------------------------------


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


def test_an_unmeasurable_slice_is_not_stamped_either_at_the_live_helper():
    """The SAME rule at `compression_copy_through`, which the live scan calls.

    `summarize_anchor_compression` returns its default dict - `compression_score`
    0 and three `None` ratios - whenever the slice is empty, the ATR-20 is
    missing or the anchor has no sigma. Publishing that as `0` under
    `anchor_compression_v1` is the measured zero again, on the path the desk
    actually runs. Both helpers answer the same way.
    """
    import pandas as pd

    from master_avwap_lib import legacy

    unmeasurable = legacy.summarize_anchor_compression(pd.DataFrame(), None, None)
    assert unmeasurable["compression_score"] == 0, "the summary's own default is a zero"
    assert unmeasurable["compression_stdev_atr_ratio"] is None

    copied = legacy.compression_copy_through(unmeasurable)
    assert copied["compression_score"] is None, "an unmeasured slice is not a score of zero"
    assert copied["compression_stdev_atr_ratio"] is None
    assert "compression_rule_version" not in copied, (
        "a rule version beside no measurement claims the rule ran"
    )

    # A real reading is published exactly as before.
    measured = {
        "compression_score": 3,
        "compression_stdev_atr_ratio": 0.52,
        "compression_range_atr_ratio": 2.1,
        "compression_close_range_atr_ratio": 1.4,
    }
    copied = legacy.compression_copy_through(measured)
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


# ---------------------------------------------------------------------------
# The six-tag cap
# ---------------------------------------------------------------------------


#: The reviewer's own fixture. Six tags before `COMPRESSION_BREAK` is offered:
#: FIRST_DEV_BREAKOUT (primary, from the family + signal), RANGE_BREAK_CONFIRMED,
#: FIVE_DAY_BREAKOUT, D1_TREND_ALIGNED, SMA_BREAKOUT_CONFIRMED and D1_RS - and
#: `D1_RS` is a CONTEXT tag, so a new confirmation added before the context
#: block still pushes it off the end.
_CROWDED_ROW = {
    "side": "LONG",
    "setup_family": "avwap_breakout",
    "favorite_signals": ["CROSS_UP_UPPER_1"],
    "previous_day_range_break": True,
    "breakout_5d": True,
    "trend_20d": "UP",
    "sma_breakout_confirmed": True,
    "daily_relative_strength_score": 1.5,
}


def test_the_break_tag_never_displaces_an_older_tag_under_the_cap():
    """Six tags is the cap, and a label added in 2026 may not cost a row a tag
    it has carried for a year - including a CONTEXT tag at the very end."""
    from master_avwap_lib import setup_tagging

    without = setup_tagging.derive_setup_tag_payload(_CROWDED_ROW)["setup_tags"]
    with_break = setup_tagging.derive_setup_tag_payload(
        dict(_CROWDED_ROW, compression_break_recent=True)
    )["setup_tags"]
    assert len(without) == setup_tagging.DEFAULT_MAX_SETUP_TAGS, (
        f"the fixture has to fill the cap to test it: {without}"
    )
    assert "D1_RS" in without, without
    assert without == with_break, (
        f"the new label displaced an older tag: {sorted(set(without) - set(with_break))}"
    )


def test_the_break_tag_is_still_offered_and_is_a_confirmation():
    from master_avwap_lib import setup_tagging

    quiet = {"side": "LONG", "setup_family": "general", "compression_break_recent": True}
    payload = setup_tagging.derive_setup_tag_payload(quiet)
    assert "COMPRESSION_BREAK" in payload["setup_tags"]
    assert payload["setup_tag_roles"]["COMPRESSION_BREAK"] == "confirmation"
