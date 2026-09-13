"""Packet WS-ENV - the builder's ADDED coverage, beside the tester's 35.

Nothing here weakens or restates a tester assertion; each test covers a seam
the tester's file leaves open and that the build has to get right anyway:

* the BACKFILL's completed-bar rule (the tester pins the live hook's, not the
  CLI's - and the desk has had 100 cached daily files ending in a forming
  candle at once);
* a short series still WRITES a row, because `unknown` is an answer;
* `attach_environment(labels=...)` opens no store, which is what lets a caller
  join two row sets against one read;
* the environment section fills the shortlist's OWN column names, so the cut
  renders in the existing Bot table instead of a blank row per environment.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPY_BARS_FIXTURE = FIXTURES / "ws_env_spy_daily.csv"


def _bars() -> list[dict]:
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


def test_the_backfill_drops_a_forming_last_bar_the_way_the_live_hook_does():
    """A cached file that ends in TODAY's half-finished candle labels through
    yesterday. The CLI reads the same `completed_bars` rule the hook does."""
    import d1_environment_store as store

    bars = _bars()
    mid_session = store.backfill_rows(
        "SPY", since="2026-09-01", bars=bars, now=datetime(2026, 9, 11, 12, 30)
    )
    assert [session for session, _env in mid_session][-1] == "2026-09-10"

    after_close = store.backfill_rows(
        "SPY", since="2026-09-01", bars=bars, now=datetime(2026, 9, 12, 6, 0)
    )
    assert [session for session, _env in after_close][-1] == "2026-09-11"


def test_a_warmup_session_is_still_written_because_unknown_is_an_answer(tmp_path):
    """A row that says `unknown` with its reason is evidence; a MISSING row is
    silence, and the two are not the same fact."""
    import d1_environment_store as store
    from indicators.d1_environment import classify_environment

    path = tmp_path / "d1_environment.jsonl"
    env = classify_environment(_bars()[:20])
    assert env.label == "unknown" and env.reason == "warmup"
    assert store.append_environment(
        env, benchmark="SPY", bars_through=env.as_of_session, path=path
    ) is True

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == "unknown"
    assert row["reason"] == "warmup"
    assert row["range_atr"] is None and row["atr14"] is None
    assert row["bars_used"] == 20
    assert store.label_for_session(row["session"], benchmark="SPY", path=path) == "unknown"


def test_attach_environment_with_a_handed_in_table_opens_no_store(tmp_path):
    """One read, two row sets. The path is deliberately a directory that holds
    no store: a join that reached for it would raise or return `unknown`."""
    import d1_environment_join as join

    labels = {"2026-09-09": "compressed", "2026-09-10": "mixed"}
    rows = [{"scan_date": "2026-09-09"}, {"scan_date": "2026-09-10"}, {"scan_date": "2026-01-02"}]
    out = join.attach_environment(rows, labels=labels, path=tmp_path / "nothing.jsonl")

    assert [row["d1_environment"] for row in out] == ["compressed", "mixed", "unknown"]
    assert join.environment_counts(out) == {"compressed": 1, "mixed": 1, "unknown": 1}


def test_the_environment_rows_fill_the_bot_shortlists_own_columns():
    """The cut renders in the ONE bot table. Without these keys every
    environment row would draw as a line of empty cells."""
    import research_results
    from ui.panels.research_results_panel import BOT_COLUMNS

    rows = [
        {
            "observation_id": f"o{index}",
            "scan_date": "2026-09-02",
            "horizon_sessions": "5",
            "side": "LONG",
            "symbol": f"S{index % 7}",
            "win": "True" if index % 3 else "False",
            "side_return_pct": "1.0",
            "d1_environment": "compressed",
        }
        for index in range(40)
    ]
    section = research_results.environment_section(rows)
    assert section.rows
    for row in section.rows:
        for key, _label in BOT_COLUMNS:
            assert str(row.display.get(key, "")).strip(), f"{key} is blank on an environment row"
        assert row.values["n_symbols"] == 7
        assert row.values["n_sessions"] == 1


def test_the_environment_section_is_absent_unless_the_caller_hands_rows_in():
    """`environment_rows=None` is "not read", not "empty": a page that could not
    open the store shows no section rather than an empty one claiming zero."""
    import research_results

    view = research_results.build_results_view(
        population="bot", horizon="swing", window="recent", snapshot={}, as_of=date(2026, 9, 11)
    )
    assert research_results.ENVIRONMENT_SECTION_KEY not in [
        section.key for section in view.sections
    ]
