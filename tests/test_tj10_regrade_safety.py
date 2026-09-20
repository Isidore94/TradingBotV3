"""TJ-10 fix round - the nightly re-grade may never lose an open read.

Reviewer, 2026-09-20 (blocker 1). `regrade_matured` revisited only `pending*`
rows and appended whenever the verdict CHANGED, and its daily loader called
`chart_snapshot.load_d1_bars` alone - which answers EMPTY for SPY, QQQ, IWM and
VXX on this desk. Reproduced: a correct ``pending 2026-09-25`` was superseded by
``unmeasured:no_anchor_close``; a later run WITH real closes wrote nothing,
because the row was no longer pending. The true answer was ``right`` (+5.0 ATR).

Two rules pinned here:

* **An `unmeasured` result NEVER supersedes a `pending` row** - missing data is
  uncertainty, never confirmation (plan.md sec 5). A verdict may only move UP
  (`verdict_rank`), and a row left unmeasured for a DATA reason is revisited on
  every later run. Only `unmeasured:not_a_call` is final.
* **ONE daily-bar loader, one tape loader and one ATR**, shared by the Day
  Review page and the night: `daily_bars_for_symbol` reads the durable store and
  falls back to the desk's own machine-local daily cache, and it is what
  `regrade_matured` uses when nothing is injected.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj10_support as fx  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

#: The reviewer's own numbers: a +9.00 move on a 0.75-ATR-per-band tape is
#: +5.0 ATR against `daily_atr`, three band-widths clear of 0.25.
FULL_CLOSES = {
    fx.SESSION: 100.0,
    fx.NEXT_SESSIONS[0]: 104.0,
    fx.NEXT_SESSIONS[1]: 105.0,
    fx.NEXT_SESSIONS[2]: 106.0,
    fx.NEXT_SESSIONS[3]: 107.0,
    fx.NEXT_SESSIONS[4]: 109.0,
}
AFTER_THE_HORIZON = datetime(2026, 9, 26, 2, 0, tzinfo=PACIFIC)
MID_HORIZON = datetime(2026, 9, 21, 17, 0, tzinfo=PACIFIC)


def _extracted_d1_read():
    """A read the ledger will STORE without a snapshot.

    An extracted row may carry the named absence; a CLICKED one may not (lead
    decision, 2026-09-20), and this test is about the re-grade, not the guard.
    """
    import market_read_grades as grader

    entry = fx.old_entry(
        "empty", text="D1 SPY is downtrending into next week", timeframe="D1"
    )
    return grader.read_rows([entry], session=fx.SESSION)[0]


def _pending_grade(root):
    import market_read_grades as grader

    row = dict(_extracted_d1_read())
    row["direction"] = "up"
    grade = grader.grade_read(
        row,
        daily_bars=fx.daily_bars({fx.SESSION: 100.0, fx.NEXT_SESSIONS[0]: 104.0}),
        atr=fx.atr_for(9.0, 3.0),
        now=MID_HORIZON,
    )
    assert grade["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"
    grader.append_grades(fx.SESSION, [grade], root=root)
    return grade


def test_an_absent_daily_store_never_retires_a_pending_read(monkeypatch, tmp_path):
    """The reviewer's three steps, in order, ending at the true answer."""
    import chart_snapshot
    import d1_environment_store
    import market_read_grades as grader

    pending = _pending_grade(tmp_path)

    # Step 2: the night runs on a desk where BOTH daily stores are empty.
    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])
    monkeypatch.setattr(d1_environment_store, "_cached_daily_bars", lambda _symbol: [])
    blind = grader.regrade_matured(AFTER_THE_HORIZON, root=tmp_path)

    assert blind == [], "an absent store wrote a verdict"
    stored = grader.read_grades(fx.SESSION, root=tmp_path)
    assert len(stored) == 1
    assert stored[0]["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"

    # Step 3: the closes arrive. The row is still open, so it is still looked at.
    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars",
        lambda _symbol: fx.daily_bars(FULL_CLOSES),
    )
    written = grader.regrade_matured(
        AFTER_THE_HORIZON, root=tmp_path,
        atr_for=lambda _symbol, _session: fx.atr_for(9.0, 3.0),
    )

    assert len(written) == 1
    assert written[0]["verdict"] == grader.VERDICT_RIGHT
    assert written[0]["supersedes"] == pending["grade_id"]
    final = grader.read_grades(fx.SESSION, root=tmp_path)
    assert len(final) == 2, "the file must only ever grow"
    assert final[0]["verdict"] == f"pending {fx.NEXT_SESSIONS[4]}"


def test_a_row_left_unmeasured_for_a_data_reason_is_looked_at_again(monkeypatch, tmp_path):
    """Last night there were no bars. Tonight there are."""
    import chart_snapshot
    import d1_environment_store
    import market_read_grades as grader

    row = dict(_extracted_d1_read())
    row["direction"] = "up"
    blind = grader.grade_read(row, daily_bars=(), atr=None, now=AFTER_THE_HORIZON)
    assert blind["verdict"] == "unmeasured:no_anchor_close"
    grader.append_grades(fx.SESSION, [blind], root=tmp_path)

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])
    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars",
        lambda _symbol: fx.daily_bars(FULL_CLOSES),
    )
    written = grader.regrade_matured(
        AFTER_THE_HORIZON, root=tmp_path,
        atr_for=lambda _symbol, _session: fx.atr_for(9.0, 3.0),
    )

    assert len(written) == 1
    assert written[0]["verdict"] == grader.VERDICT_RIGHT
    assert written[0]["supersedes"] == blind["grade_id"]


def test_a_no_view_answer_is_never_revisited(monkeypatch, tmp_path):
    """`unmeasured:not_a_call` is FINAL: no bar arriving later makes it a call."""
    import d1_environment_store
    import market_read_grades as grader

    entry = fx.mentor_entry(direction="no_view", horizon="rest_of_day",
                            timeframe="M5", confidence="")
    row = grader.read_rows([entry], session=fx.SESSION)[0]
    declined = grader.grade_read(row, m5_bars=(), atr=None, now=fx.AFTER_THE_CLOSE)
    assert declined["verdict"] == "unmeasured:not_a_call"
    grader.append_grades(fx.SESSION, [declined], root=tmp_path)

    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars",
        lambda _symbol: fx.daily_bars(FULL_CLOSES),
    )
    written = grader.regrade_matured(AFTER_THE_HORIZON, root=tmp_path)

    assert written == []
    assert len(grader.read_grades(fx.SESSION, root=tmp_path)) == 1


def test_a_closed_horizon_is_still_closed(monkeypatch, tmp_path):
    """Idempotence, restated over the new rule: a measured row is never revisited."""
    import d1_environment_store
    import market_read_grades as grader

    row = dict(_extracted_d1_read())
    row["direction"] = "up"
    closed = grader.grade_read(
        row, daily_bars=fx.daily_bars(FULL_CLOSES), atr=fx.atr_for(9.0, 3.0),
        now=AFTER_THE_HORIZON,
    )
    assert closed["verdict"] == grader.VERDICT_RIGHT
    grader.append_grades(fx.SESSION, [closed], root=tmp_path)

    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars",
        lambda _symbol: fx.daily_bars({**FULL_CLOSES, fx.NEXT_SESSIONS[4]: 50.0}),
    )
    written = grader.regrade_matured(
        AFTER_THE_HORIZON, root=tmp_path,
        atr_for=lambda _symbol, _session: fx.atr_for(9.0, 3.0),
    )

    assert written == []
    assert len(grader.read_grades(fx.SESSION, root=tmp_path)) == 1


def test_one_loader_serves_the_page_and_the_night(monkeypatch):
    """The durable store is EMPTY on this desk; the machine cache is the one
    that answers, and both callers go through this function."""
    import chart_snapshot
    import d1_environment_store
    import market_read_grades as grader

    monkeypatch.setattr(chart_snapshot, "load_d1_bars", lambda _symbol: [])
    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars",
        lambda symbol: fx.daily_bars(FULL_CLOSES) if symbol == "SPY" else [],
    )

    assert len(grader.daily_bars_for_symbol("SPY")) == len(FULL_CLOSES)
    assert grader.daily_bars_for_symbol("NOPE") == []
    # Six closes cannot make an ATR(14): unmeasurable is None, never 0.
    assert grader.atr_for_session("SPY", fx.SESSION) is None
    long_history = fx.daily_bars({
        f"2026-08-{day:02d}": 100.0 + day for day in range(3, 31)
    })
    monkeypatch.setattr(
        d1_environment_store, "_cached_daily_bars", lambda _symbol: long_history
    )
    assert grader.atr_for_session("SPY", "2026-08-31") is not None
    # The durable store WINS when it has the symbol - the cache is a fallback.
    monkeypatch.setattr(
        chart_snapshot, "load_d1_bars",
        lambda _symbol: fx.daily_bars({fx.SESSION: 1.0}),
    )
    assert len(grader.daily_bars_for_symbol("SPY")) == 1


def test_the_verdict_rank_only_ever_moves_up():
    import market_read_grades as grader

    assert grader.verdict_rank(grader.VERDICT_RIGHT) > grader.verdict_rank(
        f"{grader.PENDING_PREFIX} 2026-09-25"
    )
    assert grader.verdict_rank(f"{grader.PENDING_PREFIX} 2026-09-25") > grader.verdict_rank(
        "unmeasured:no_anchor_close"
    )
    assert grader.verdict_rank("unmeasured:no_anchor_close") > grader.verdict_rank(
        "unmeasured:not_a_call"
    )
