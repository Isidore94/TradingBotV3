r"""TJ-15 fix round - the two blockers, the three advisories, and the floors.

Reviewer, 2026-09-19, on the real 2026-09-18 window:

* **Blocker 1.** The pack NAMED a leader off four rows against one.
  `MIN_REPORTABLE_N` gates the RATE, whose denominator is every measured
  decision; a FEATURE is only measured on the decisions that also carried a
  point-in-time scan row, and that was a different and much smaller number.
  `veto / sma_incoming` had 30 measured decisions, cleared the rate floor, and
  then named `atr20` at ``n_a=4 n_b=1 auc=1.0``.
* **Blocker 2.** Half the population was M5 and was pooled under the D1 ruler:
  1,026 M5 against 1,013 D1 over the window, `like` printed as ONE rate over 320
  D1 and 358 M5 decisions, and the top-named leader `not_today` was 317 of 317
  M5 decisions judged on a five-session swing horizon with D1-scan features.

Seven assertions in the tester's two files became unsatisfiable under the lead's
new rules and are left RED and UNEDITED, each for the same reason - the fixture
is smaller than the feature floor - with the arithmetic named in the handoff.
Every contract they pinned is re-proven here at a population that clears the
floor, plus the new rules the fix round adds.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SESSION = "2026-09-18"
DECIDED = "2026-09-11"
#: The exchange session before `DECIDED`, and the one before that.
PRIOR = "2026-09-10"
TWO_BACK = "2026-09-09"
NOW = datetime(2026, 9, 19, 2, 0)

FEATURE_COLUMNS = ("run_id", "run_timestamp", "run_date", "symbol", "side",
                   "pct_from_current_vwap", "atr20")


@pytest.fixture(autouse=True)
def _pin_the_desk_zone(monkeypatch):
    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Los_Angeles")


def _rows(**columns) -> list[dict]:
    length = len(next(iter(columns.values())))
    return [
        {name: ("" if values[i] is None else str(values[i])) for name, values in columns.items()}
        for i in range(length)
    ]


def _sessions_ending(session: str, count: int) -> list[str]:
    from market_calendar import previous_session

    cursor = date.fromisoformat(session)
    out = [cursor.isoformat()]
    for _ in range(count - 1):
        cursor = previous_session(cursor)
        out.append(cursor.isoformat())
    return list(reversed(out))


def _sessions_after(session: str, count: int) -> list[str]:
    from market_calendar import is_session

    cursor = date.fromisoformat(session)
    out: list[str] = []
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_session(cursor):
            out.append(cursor.isoformat())
    return out


def _daily(*, run: bool, session: str = DECIDED, forward: int = 5) -> list[dict]:
    flat = [
        {"dt": f"{day}T00:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for day in _sessions_ending(session, 15)
    ]
    first = (100.0, 103.0, 99.9, 102.5) if run else (100.0, 100.5, 98.5, 98.6)
    rest = (102.5, 103.0, 102.0, 102.5) if run else (98.6, 99.0, 98.0, 98.5)
    after = ([first] + [rest] * 4)[:forward]
    days = _sessions_after(session, len(after))
    return flat + [
        {"dt": f"{day}T00:00:00", "open": o, "high": h, "low": low, "close": c}
        for day, (o, h, low, c) in zip(days, after, strict=False)
    ]


def _decision(
    symbol: str,
    *,
    stamp: str = f"{DECIDED}T10:05:00-04:00",
    reason: str = "extended",
    verdict: str = "veto",
    timeframe: str = "D1",
    session: str = DECIDED,
) -> dict:
    row = {
        "session_date": session, "symbol": symbol, "side": "LONG",
        "category": "chart_review", "verdict": verdict, "source": "annotations",
        "stamp": stamp, "capture_id": f"e-{symbol}", "reason": reason,
        "decision_session": session,
    }
    if timeframe is not None:
        row["timeframe"] = timeframe
    return row


def _history(tmp_path: Path, rows, *, name: str = "d1_features_history.csv") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _scan(symbol: str, *, run_date: str, at: str, vwap: float, atr: float = 2.0) -> dict:
    return {
        "run_id": f"{run_date}-{at.replace(':', '')}",
        "run_timestamp": f"{run_date}T{at}", "run_date": run_date,
        "symbol": symbol, "side": "LONG",
        "pct_from_current_vwap": vwap, "atr20": atr,
    }


def _group(pack, reason: str = "extended", verdict: str = "veto") -> dict:
    found = [
        g for g in pack["groups"]
        if g.get("reason_code") == reason and g.get("verdict") == verdict
    ]
    assert len(found) == 1, pack["groups"]
    return found[0]


def _feature(group, name: str = "pct_from_current_vwap") -> dict:
    rows = [row for row in group["features"] if row["feature"] == name]
    assert len(rows) == 1, group["features"]
    return rows[0]


def _build(tmp_path, decisions, features, bars, *, session=SESSION, now=NOW):
    from ai_jobs import miss_contrast

    out = miss_contrast.run_miss_contrast(
        session_date=session, now=now, root=tmp_path / "packs",
        decisions=decisions, features=features, daily_bars=bars,
    )
    assert out["status"] == "ok", out
    return json.loads(Path(out["outputs"][0]).read_text(encoding="utf-8"))


#: The feature floor needs this many rows a side. Sized from the module so a
#: change to the constant changes the fixtures rather than silently passing.
def _over_the_floor() -> tuple[int, int]:
    import evidence_contrast
    from evidence_stats import MIN_REPORTABLE_N

    side = evidence_contrast.MIN_CONTRAST_SIDE_N
    while side * 2 < MIN_REPORTABLE_N:
        side += 1
    return side, side


# ===========================================================================
# BLOCKER 1 - the feature floor
# ===========================================================================


def test_a_feature_measured_on_forty_rows_against_one_is_thin_and_never_first():
    """The lead's named case. Forty against one is AUC 1.0 by construction.

    `lopsided` separates perfectly and cannot be wrong; `honest` is measured on
    both sides in numbers a reader can argue with. Only the second may be
    ranked, and the first is still NAMED with both counts.
    """
    import evidence_contrast

    group_a = [
        {"lopsided": 9.0, "honest": 6.0 if index < 30 else 1.0}
        for index in range(40)
    ]
    group_b = [
        {"lopsided": 1.0 if index == 0 else None, "honest": 1.0}
        for index in range(41)
    ]

    out = evidence_contrast.contrast(group_a, group_b)

    names = [row["feature"] for row in out["features"]]
    assert names == ["honest"], names
    assert out["compared"] == 1
    assert out["thin"] == 1
    thin = out["thin_features"][0]
    assert thin["feature"] == "lopsided"
    assert (thin["n_a"], thin["n_b"]) == (40, 1)
    assert "auc" not in thin, "a thin feature carries no rank statistic"
    assert thin["note"] == "too few to call"
    assert "too thin to call" in out["statement"], out["statement"]


def test_the_floor_is_both_sides_and_the_total_and_the_boundary_is_inclusive():
    """`min(n_a, n_b) >= min_side` AND `n_a + n_b >= min_total`, exactly."""
    import evidence_contrast
    from evidence_stats import MIN_REPORTABLE_N

    side = evidence_contrast.MIN_CONTRAST_SIDE_N

    def _pair(count_a: int, count_b: int):
        a = [{"f": float(index)} for index in range(count_a)]
        b = [{"f": float(index) + 100.0} for index in range(count_b)]
        return evidence_contrast.contrast(a, b)

    # One row under on the thin side is thin, even with a huge other side.
    assert _pair(side - 1, 200)["compared"] == 0
    assert _pair(200, side - 1)["compared"] == 0
    # Both sides over the side floor but under the TOTAL floor is still thin.
    thin_total = MIN_REPORTABLE_N - 1
    assert side * 2 <= thin_total, "fixture needs a gap between the two floors"
    assert _pair(side, thin_total - side)["compared"] == 0
    # Exactly on both floors is compared.
    on_both = _pair(side, MIN_REPORTABLE_N - side)
    assert on_both["compared"] == 1
    assert on_both["thin"] == 0


def test_a_group_with_a_reportable_rate_but_no_ranked_feature_is_not_a_leader(tmp_path):
    """The live shape: 35 measured decisions, features on 4 against 1.

    The rate clears `MIN_REPORTABLE_N` and is printed. The feature does NOT
    clear the feature floor, so the group is not ranked, is not named, and the
    pack's headline never claims the floor for it.
    """
    from evidence_stats import MIN_REPORTABLE_N

    runs, duds = 15, 20
    assert runs + duds > MIN_REPORTABLE_N
    decisions, scans, bars = [], [], {}
    for index in range(runs + duds):
        ran = index < runs
        symbol = f"S{index:03d}"
        decisions.append(_decision(symbol))
        bars[symbol] = _daily(run=ran)
        # Only four of the runners and one of the duds were ever scanned.
        if (ran and index < 4) or (not ran and index == runs):
            scans.append(_scan(symbol, run_date=DECIDED, at="06:40:00",
                               vwap=9.0 if ran else 1.0))

    pack = _build(tmp_path, decisions, _history(tmp_path, scans), bars)
    group = _group(pack)

    assert group["measured"] == runs + duds
    assert group["reportable"] is True, "the RATE clears its own floor"
    assert (group["misses"], group["correct"]) == (runs, duds)
    assert (group["n_a"], group["n_b"]) == (4, 1)
    assert group["no_point_in_time_scan"] == runs + duds - 5
    assert group["compared"] == 0
    assert group["features"] == []
    assert group["feature_note"] == "no feature had enough rows on both sides"
    thin = {row["feature"]: row for row in group["thin_features"]}
    assert thin["atr20"]["n_a"] == 4 and thin["atr20"]["n_b"] == 1

    assert tuple(pack["leaders"]) == (), pack["leaders"]
    assert "extended" not in pack["statement"], pack["statement"]
    assert "no feature had enough rows on both sides" in pack["statement"]


# ===========================================================================
# BLOCKER 2 - D1 only
# ===========================================================================


def test_only_d1_decisions_are_judged_and_the_rest_are_counted_by_timeframe(tmp_path):
    """The lead's named case: 3 D1 likes and 5 M5 likes.

    An M5 click has no D1-scan features and is not a five-session swing call.
    It is EXCLUDED and COUNTED, never pooled into the same rate.
    """
    decisions, scans, bars = [], [], {}
    for index in range(3):
        symbol = f"D{index:03d}"
        decisions.append(_decision(symbol, verdict="like", reason="clean", timeframe="D1"))
        scans.append(_scan(symbol, run_date=DECIDED, at="06:40:00", vwap=4.0))
        bars[symbol] = _daily(run=index == 0)
    for index in range(5):
        symbol = f"M{index:03d}"
        decisions.append(_decision(symbol, verdict="like", reason="clean", timeframe="M5"))
        scans.append(_scan(symbol, run_date=DECIDED, at="06:40:00", vwap=4.0))
        bars[symbol] = _daily(run=True)

    pack = _build(tmp_path, decisions, _history(tmp_path, scans), bars)

    likes = [g for g in pack["groups"] if g["verdict"] == "like"]
    assert len(likes) == 1, pack["groups"]
    assert likes[0]["n"] == 3
    assert pack["excluded_by_timeframe"] == {"M5": 5}
    assert pack["timeframe"] == "D1"
    assert "5 M5" in pack["statement"], pack["statement"]
    assert "D1 swing ruler" in pack["statement"], pack["statement"]


def test_a_decision_with_no_timeframe_is_counted_and_never_read_as_d1(tmp_path):
    """Missing data is uncertainty. It is not the timeframe we happen to want."""
    decisions = [
        _decision("AAA", timeframe="D1"),
        _decision("BBB", timeframe=None),
        _decision("CCC", timeframe="h1"),
    ]
    scans = [_scan(s, run_date=DECIDED, at="06:40:00", vwap=4.0) for s in ("AAA", "BBB", "CCC")]
    bars = {s: _daily(run=True) for s in ("AAA", "BBB", "CCC")}

    pack = _build(tmp_path, decisions, _history(tmp_path, scans), bars)

    assert pack["decisions"] == 1
    assert pack["no_timeframe"] == 1
    # Case-insensitive: `h1` is H1 and is excluded, not silently judged.
    assert pack["excluded_by_timeframe"] == {"H1": 1}
    assert "no timeframe recorded" in pack["statement"], pack["statement"]
    assert _group(pack)["n"] == 1


# ===========================================================================
# ADVISORY A - the point-in-time window reaches back ONE session
# ===========================================================================


def _scan_cohort(tmp_path, scans_for, *, stamp, runs=None, duds=None):
    """`runs` + `duds` D1 vetoes, each with the scan rows `scans_for` gives."""
    side_a, side_b = _over_the_floor()
    runs = side_a if runs is None else runs
    duds = side_b if duds is None else duds
    decisions, scans, bars = [], [], {}
    for index in range(runs + duds):
        ran = index < runs
        symbol = f"S{index:03d}"
        decisions.append(_decision(symbol, stamp=stamp))
        bars[symbol] = _daily(run=ran)
        scans.extend(scans_for(symbol, ran))
    return decisions, _history(tmp_path, scans), bars, runs, duds


def test_an_evening_decision_takes_that_sessions_last_scan(tmp_path):
    """Scans ran 07:00 and 13:00; the click was at 18:30. The 13:00 wins."""
    stamp = f"{DECIDED}T18:30:00-07:00"

    def _scans(symbol, ran):
        return [
            _scan(symbol, run_date=PRIOR, at="13:00:00", vwap=0.1),
            _scan(symbol, run_date=DECIDED, at="07:00:00", vwap=0.5),
            _scan(symbol, run_date=DECIDED, at="13:00:00", vwap=8.0 if ran else 2.0),
        ]

    decisions, features, bars, runs, duds = _scan_cohort(tmp_path, _scans, stamp=stamp)
    pack = _build(tmp_path, decisions, features, bars)
    group = _group(pack)

    assert (group["scan_same_session"], group["scan_prior_session"]) == (runs + duds, 0)
    assert group["no_point_in_time_scan"] == 0
    row = _feature(group)
    assert row["median_a"] == pytest.approx(8.0)
    assert row["median_b"] == pytest.approx(2.0)


def test_a_pre_market_decision_takes_yesterdays_last_scan(tmp_path):
    """05:30 on the session: no scan has run yet, and one ran at 13:00 yesterday.

    That is what the trader was looking at. The 07:15 scan that day is LATER
    than the click and may never be used.
    """
    stamp = f"{DECIDED}T05:30:00-07:00"

    def _scans(symbol, ran):
        return [
            _scan(symbol, run_date=PRIOR, at="07:00:00", vwap=0.5),
            _scan(symbol, run_date=PRIOR, at="13:00:00", vwap=7.0 if ran else 3.0),
            _scan(symbol, run_date=DECIDED, at="07:15:00", vwap=9.9),
        ]

    decisions, features, bars, runs, duds = _scan_cohort(tmp_path, _scans, stamp=stamp)
    pack = _build(tmp_path, decisions, features, bars)
    group = _group(pack)

    assert (group["scan_same_session"], group["scan_prior_session"]) == (0, runs + duds)
    row = _feature(group)
    assert row["median_a"] == pytest.approx(7.0)
    assert row["median_b"] == pytest.approx(3.0)


def test_a_scan_two_sessions_old_is_never_used(tmp_path):
    """One session back is what they were looking at; two is another day."""
    stamp = f"{DECIDED}T05:30:00-07:00"

    def _scans(symbol, ran):
        return [_scan(symbol, run_date=TWO_BACK, at="13:00:00", vwap=7.0 if ran else 3.0)]

    decisions, features, bars, runs, duds = _scan_cohort(tmp_path, _scans, stamp=stamp)
    pack = _build(tmp_path, decisions, features, bars)
    group = _group(pack)

    assert group["no_point_in_time_scan"] == runs + duds
    assert (group["scan_same_session"], group["scan_prior_session"]) == (0, 0)
    assert group["compared"] == 0
    assert group["features"] == []


# ===========================================================================
# ADVISORY B - only a veto has a reason code
# ===========================================================================


def test_free_text_reasons_do_not_split_a_verdict_into_one_group_per_sentence(tmp_path):
    """Live: `dislike` split into two groups of one over two phrasings."""
    decisions, scans, bars = [], [], {}
    phrases = ["[other] rejecting 1stdev", "[other] rejecting the 1stdev"]
    for index in range(6):
        symbol = f"S{index:03d}"
        decisions.append(
            _decision(symbol, verdict="dislike", reason=phrases[index % 2])
        )
        scans.append(_scan(symbol, run_date=DECIDED, at="06:40:00", vwap=4.0))
        bars[symbol] = _daily(run=index % 2 == 0)

    pack = _build(tmp_path, decisions, _history(tmp_path, scans), bars)

    dislikes = [g for g in pack["groups"] if g["verdict"] == "dislike"]
    assert len(dislikes) == 1, pack["groups"]
    assert dislikes[0]["reason_code"] == ""
    assert dislikes[0]["n"] == 6
    assert "free text, not a code" in pack["pooling"], pack["pooling"]


def test_a_veto_still_groups_by_its_vocabulary_code(tmp_path):
    """The one verdict whose reason IS a code keeps its per-code groups."""
    decisions, scans, bars = [], [], {}
    for index in range(6):
        symbol = f"S{index:03d}"
        code = "extended" if index < 4 else "compressed"
        decisions.append(_decision(symbol, verdict="veto", reason=code))
        scans.append(_scan(symbol, run_date=DECIDED, at="06:40:00", vwap=4.0))
        bars[symbol] = _daily(run=index % 2 == 0)

    pack = _build(tmp_path, decisions, _history(tmp_path, scans), bars)

    codes = sorted(g["reason_code"] for g in pack["groups"] if g["verdict"] == "veto")
    assert codes == ["compressed", "extended"], pack["groups"]


# ===========================================================================
# ADVISORY C - a pack must be keyed to a readable date
# ===========================================================================


@pytest.mark.parametrize("bad", ["", "   ", "not-a-date", "2026-13-99"])
def test_an_unreadable_session_date_refuses_and_writes_nothing(tmp_path, bad):
    """It used to publish `miss_contrast-.json`: one file, no date, no reader."""
    from ai_jobs import miss_contrast

    root = tmp_path / "packs"
    out = miss_contrast.run_miss_contrast(
        session_date=bad, now=NOW, root=root,
        decisions=[_decision("AAA")], features=None, daily_bars={},
    )

    assert out["status"] == "failed", out
    assert "readable date" in out["reason"], out
    assert out["outputs"] == []
    assert not root.exists() or list(root.glob("*.json")) == []


# ===========================================================================
# the tester's seven, re-proven over a population that clears the floor
# ===========================================================================


def test_a_feature_that_separates_the_groups_reports_counts_medians_and_the_precedent_auc():
    """The tester's first case at 15 a side instead of 3 (floor: 10 and 30)."""
    import compression_calibration
    import evidence_contrast

    misses = _rows(
        pct_from_current_vwap=[3.0 + index * 0.1 for index in range(15)],
        spy_five_day_return_pct=[1.0] * 15,
    )
    correct = _rows(
        pct_from_current_vwap=[1.0 + index * 0.1 for index in range(15)],
        spy_five_day_return_pct=[1.0] * 15,
    )

    out = evidence_contrast.contrast(
        misses, correct, label_a="real_miss", label_b="correct_rejection"
    )

    assert (out["n_a"], out["n_b"]) == (15, 15)
    assert out["compared"] == 2
    by_name = {row["feature"]: row for row in out["features"]}
    separated = by_name["pct_from_current_vwap"]
    assert separated["median_a"] == pytest.approx(3.7)
    assert separated["median_b"] == pytest.approx(1.7)
    assert (separated["n_a"], separated["n_b"]) == (15, 15)
    assert separated["auc"] == pytest.approx(
        compression_calibration.auc(
            [3.0 + index * 0.1 for index in range(15)],
            [1.0 + index * 0.1 for index in range(15)],
        )
    )
    assert separated["auc"] == pytest.approx(1.0)
    assert by_name["spy_five_day_return_pct"]["auc"] == pytest.approx(0.5)


def test_only_the_top_three_are_shown_and_both_numbers_are_stated():
    """`observational, top K of N compared; M more too thin to call`."""
    import evidence_contrast

    high = [9.0] * 15
    low = [1.0] * 15
    same = [5.0] * 15
    misses = _rows(
        atr20=high, priority_score=low, compression_penalty=same,
        days_to_next_earnings=same, pct_from_current_vwap=same,
        recent_band_extension_days=same, spy_one_day_return_pct=[6.0] + [5.0] * 14,
    )
    correct = _rows(
        atr20=low, priority_score=high, compression_penalty=same,
        days_to_next_earnings=same, pct_from_current_vwap=same,
        recent_band_extension_days=same, spy_one_day_return_pct=same,
    )

    out = evidence_contrast.contrast(misses, correct, label_a="a", label_b="b")

    assert out["compared"] == 7
    assert out["thin"] == 0
    assert out["top"] == 3
    assert [row["feature"] for row in out["features"]] == [
        "atr20", "priority_score", "spy_one_day_return_pct",
    ]
    assert "observational" in out["statement"], out["statement"]
    assert "top 3 of 7" in out["statement"], out["statement"]
    assert "0 more too thin to call" in out["statement"], out["statement"]


def test_a_tie_breaks_by_feature_name_and_never_by_how_big_the_group_is():
    """A SIZE rule orders nothing here; the rank statistic and the name do."""
    import evidence_contrast

    high = [9.0] * 16
    strong_b = [0.0] * 16
    tied_b = [1.0] * 8 + [9.0] * 8
    misses = _rows(z_strongest=high, b_tied=high, a_tied=high, mid=[6.0] + [5.0] * 15)
    correct = _rows(
        z_strongest=strong_b, b_tied=tied_b, a_tied=tied_b, mid=[5.0] * 16
    )

    out = evidence_contrast.contrast(misses, correct, label_a="a", label_b="b", top=3)

    names = [row["feature"] for row in out["features"]]
    assert names == ["z_strongest", "a_tied", "b_tied"], names
    by_name = {row["feature"]: row for row in out["features"]}
    assert by_name["a_tied"]["auc"] == pytest.approx(by_name["b_tied"]["auc"])


def test_a_blank_cell_is_left_out_of_the_median_and_is_never_read_as_zero():
    """An old row has the key PRESENT and EMPTY. Missing data is not a value."""
    import evidence_contrast

    # `atr20` is measured on 16 of the 24 misses: median 3.5, which a zero-fill
    # would drag to 2.33. `mid_earnings_zone_streak_days` was never written on
    # the other side, so it is NAMED unmeasured and is not one of the K.
    atr_values = [3.0] * 8 + [4.0] * 8 + [None] * 4 + ["n/a"] * 4
    misses = _rows(
        atr20=atr_values,
        mid_earnings_zone_streak_days=[2.0] * 24,
    )
    correct = _rows(
        atr20=[1.0] * 24,
        mid_earnings_zone_streak_days=[None] * 12 + [""] * 12,
    )

    out = evidence_contrast.contrast(misses, correct, label_a="a", label_b="b")

    assert (out["n_a"], out["n_b"]) == (24, 24)
    assert out["compared"] == 1
    assert "mid_earnings_zone_streak_days" in tuple(out["unmeasured_features"])
    row = out["features"][0]
    assert row["feature"] == "atr20"
    assert row["median_a"] == pytest.approx(3.5)
    assert (row["n_a"], row["n_b"]) == (16, 24)


def test_the_features_come_from_the_last_scan_at_or_before_the_decision(tmp_path):
    """The tester's point-in-time case at 15 a side. 07:30 did not exist yet."""
    stamp = f"{DECIDED}T10:05:00-04:00"   # 07:05 desk-local

    def _scans(symbol, ran):
        return [
            _scan(symbol, run_date=DECIDED, at="06:40:00", vwap=3.0 if ran else 1.0),
            _scan(symbol, run_date=DECIDED, at="07:30:00", vwap=9.9),
        ]

    decisions, features, bars, runs, duds = _scan_cohort(tmp_path, _scans, stamp=stamp)
    pack = _build(tmp_path, decisions, features, bars)
    group = _group(pack)

    assert (group["misses"], group["correct"]) == (runs, duds)
    row = _feature(group)
    assert row["median_a"] == pytest.approx(3.0)
    assert row["median_b"] == pytest.approx(1.0)


def test_the_same_decision_written_in_another_zone_picks_the_same_scan_row(tmp_path):
    """07:05-07:00 and 14:05+00:00 are ONE instant and must read alike."""

    def _scans(symbol, ran):
        return [
            _scan(symbol, run_date=DECIDED, at="06:40:00", vwap=3.0 if ran else 1.0),
            _scan(symbol, run_date=DECIDED, at="07:30:00", vwap=9.9),
        ]

    local_decisions, features, bars, _r, _d = _scan_cohort(
        tmp_path, _scans, stamp=f"{DECIDED}T07:05:00-07:00"
    )
    utc_decisions = [
        {**row, "stamp": f"{DECIDED}T14:05:00+00:00"} for row in local_decisions
    ]

    local = _feature(_group(_build(tmp_path, local_decisions, features, bars)))
    utc = _feature(_group(_build(tmp_path, utc_decisions, features, bars)))

    assert local["median_a"] == pytest.approx(3.0)
    assert utc["median_a"] == pytest.approx(local["median_a"])
    assert utc["median_b"] == pytest.approx(local["median_b"])


def test_a_scan_from_a_later_session_never_reaches_an_earlier_decision(tmp_path):
    """Later scans are in the window too; not one of them may be read back."""
    later = _sessions_after(DECIDED, 1)[0]
    assert later <= SESSION

    def _scans(symbol, ran):
        return [
            _scan(symbol, run_date=DECIDED, at="06:40:00", vwap=3.0 if ran else 1.0),
            _scan(symbol, run_date=later, at="06:40:00", vwap=9.9),
        ]

    decisions, features, bars, _r, _d = _scan_cohort(
        tmp_path, _scans, stamp=f"{DECIDED}T10:05:00-04:00"
    )
    row = _feature(_group(_build(tmp_path, decisions, features, bars)))

    assert row["median_a"] == pytest.approx(3.0)
    assert row["median_b"] == pytest.approx(1.0)
