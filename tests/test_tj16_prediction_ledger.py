"""TJ-16 items 1 and 2 - the prediction ledger's reader, its naive baselines
and its calibration by `How sure`. RED before the build.

`plan.md` §12.4 TJ-16 item 2: *"Skill against naive baselines, never a bare hit
rate. Every accuracy cell is shown beside what `always Up`, `same as the last
hour` and `with the D1 environment` would have scored on the SAME stamps ...
**Calibration:** accuracy by `How sure` - High must beat Low or the page says it
does not."*

The contract these tests pin (the builder may add keys, never remove one):

    scripts/prediction_ledger.py

    read_ledger(sessions, *, root=None, source="") -> list[dict]
        The CURRENT grade row per read - a superseded row is hidden, never
        counted twice - oldest session first. `source` filters
        (`market_read_grades.SOURCE_CLICK` / `SOURCE_EXTRACTED`); `""` is
        everything, which is why `build_readout` has to refuse a mix.

    build_readout(rows) -> dict
        Raises `market_read_grades.PoolingError` when the rows do not share one
        `source` (decision 0021 answer 29).

        {"schema", "source", "empty", "statement", "sessions": [...],
         "horizons": {"rest_of_day": H, "next_5_sessions": H}}

        H = {"accuracy": cell,
             "baselines": {name: cell for name in market_read_grades.BASELINES},
             "calibration": {"cells": [...], "high_beats_low": bool | None,
                             "statement": str},
             "statement": str}

        cell = `market_read_grades.accuracy`'s keys - right / wrong / flat /
        pending / unmeasured / n / rate / rate_lb / meets_floor. `n` is the
        CLOSED horizons; a `flat` reading is IN it; `rate` is None when nothing
        was measured, never 0.0.

        a calibration cell adds `confidence` and keeps `n`.

    your_reads(session, *, root=None, source=SOURCE_CLICK) -> dict
        {"text", "n", "right", "empty", "baselines": {...}} - Day Review's
        **Your reads** line (TJ-12), a FILE READ meant for a worker: it takes a
        root, it never raises on a folder that does not exist, and today on the
        live desk that folder does not exist.

Nothing here calls a model, opens a network or touches a live store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj16_support as fx  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cells(payload: Any) -> list[Mapping[str, Any]]:
    """Every statistics cell anywhere in a readout.

    A cell is a mapping that carries a `rate`. Walking for them is how "every
    cell carries `n`" is asserted without listing the shape twice.
    """
    found: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        if "rate" in payload:
            found.append(payload)
        for value in payload.values():
            found.extend(_cells(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(_cells(item))
    return found


def _horizon(readout: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    horizons = readout["horizons"]
    assert name in horizons, sorted(horizons)
    return horizons[name]


def _confidence(block: Mapping[str, Any], level: str) -> Mapping[str, Any]:
    cells = [row for row in block["cells"] if str(row.get("confidence")) == level]
    assert len(cells) == 1, block["cells"]
    return cells[0]


def _ledger(tmp_path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    root = tmp_path / "day_review"
    fx.store_ledger(root, list(rows))
    return root


# ---------------------------------------------------------------------------
# item 2 - the baselines, on the trader's own stamps
# ---------------------------------------------------------------------------
def test_accuracy_is_printed_beside_three_baselines_on_the_same_forty_stamps(tmp_path):
    """Ten sessions of four clicks: 22 right of 40 closed rows = 0.55.

    The three naive rules answer the SAME forty stamps, so each of their cells
    has the same `n` - a baseline measured on a different population is not a
    comparison. Their scores are hand-counted in `tj16_support`:
    `always_up` 24/40, `same_as_the_last_hour` 32/40, `with_the_d1_environment`
    32/40. All three beat the trader, which is exactly the sentence item 2
    exists to make sayable: *"right 55% (n 40); always-Up 60%"*.
    """
    import market_read_grades as grades
    import prediction_ledger

    sessions, rows = fx.two_weeks_of_clicks()
    root = _ledger(tmp_path, rows)

    stored = prediction_ledger.read_ledger(sessions, root=root, source=grades.SOURCE_CLICK)
    assert len(stored) == 40, len(stored)
    readout = prediction_ledger.build_readout(stored)
    day = _horizon(readout, "rest_of_day")

    assert day["accuracy"]["right"] == 22
    assert day["accuracy"]["wrong"] == 18
    assert day["accuracy"]["n"] == 40
    assert day["accuracy"]["rate"] == pytest.approx(0.55)

    assert set(day["baselines"]) == set(grades.BASELINES), sorted(day["baselines"])
    assert day["baselines"]["always_up"]["n"] == 40
    assert day["baselines"]["always_up"]["right"] == 24
    assert day["baselines"]["always_up"]["rate"] == pytest.approx(0.60)
    assert day["baselines"]["same_as_the_last_hour"]["n"] == 40
    assert day["baselines"]["same_as_the_last_hour"]["right"] == 32
    assert day["baselines"]["with_the_d1_environment"]["n"] == 40
    assert day["baselines"]["with_the_d1_environment"]["right"] == 32

    cells = _cells(readout)
    assert cells, readout
    for cell in cells:
        assert "n" in cell, cell


def test_a_baseline_that_cannot_answer_is_unmeasured_and_never_counted_wrong(tmp_path):
    """`compressed` is not a direction, so the D1-environment rule has no answer.

    Missing data is uncertainty, never confirmation (plan.md sec 5): those four
    stamps leave that baseline's fraction entirely - they are not four losses
    for the naive rule, which would flatter the trader.
    """
    import prediction_ledger

    sessions = fx.sessions_ending(count=2)
    rows = fx.graded_session(
        sessions[0],
        [
            {"hour": hour, "direction": "up", "confidence": "medium",
             "d1_environment": "compressed"}
            for hour in fx.CLICK_HOURS
        ],
        rising=True,
    ) + fx.graded_session(
        sessions[1],
        [
            {"hour": hour, "direction": "up", "confidence": "medium",
             "d1_environment": "trending_up"}
            for hour in fx.CLICK_HOURS
        ],
        rising=True,
    )
    root = _ledger(tmp_path, rows)

    readout = prediction_ledger.build_readout(
        prediction_ledger.read_ledger(sessions, root=root)
    )
    day = _horizon(readout, "rest_of_day")

    assert day["accuracy"]["n"] == 8
    cell = day["baselines"]["with_the_d1_environment"]
    assert cell["n"] == 4, "the four compressed stamps are not in the fraction"
    assert cell["wrong"] == 0
    assert cell["unmeasured"] == 4
    assert cell["rate"] == pytest.approx(1.0)


def test_the_baseline_is_judged_by_the_graders_own_band_and_never_a_copy(
    tmp_path, monkeypatch
):
    """One band rule, CALLED - the precedent `real_miss.verdict` set in TJ-15.

    The trader's verdicts are STORED and cannot move; a baseline's has to be
    measured now, and it is measured with `market_read_grades`' own band. Swap
    that rule and every baseline follows it while the stored rows do not. A
    reader carrying its own `abs(move) <= 0.25` drifts the day the band moves.
    """
    import market_read_grades as grades
    import prediction_ledger

    sessions, rows = fx.two_weeks_of_clicks()
    root = _ledger(tmp_path, rows)
    stored = prediction_ledger.read_ledger(sessions, root=root)

    monkeypatch.setattr(grades, "_verdict_for", lambda *_a, **_k: grades.VERDICT_WRONG)
    readout = prediction_ledger.build_readout(stored)
    day = _horizon(readout, "rest_of_day")

    assert day["accuracy"]["right"] == 22, "a stored verdict is never re-measured"
    for name in grades.BASELINES:
        assert day["baselines"][name]["right"] == 0, name
        assert day["baselines"][name]["n"] == 40, name


# ---------------------------------------------------------------------------
# item 2 - calibration
# ---------------------------------------------------------------------------
def test_high_confidence_that_did_not_beat_low_is_said_so(tmp_path):
    """The honest sentence, on a fixture built to embarrass the flattering one.

    Twenty-four sessions, 96 clicks, one `How sure` per session and every
    bucket over `MIN_REPORTABLE_N` on its own: low 28/32, medium 16/32, high
    8/32. High is the WORST bucket here. plan.md TJ-16 item 2: *"High must beat
    Low or the page says it does not"* - so the page says it.
    """
    import evidence_stats
    import prediction_ledger

    sessions, rows = fx.calibration_ledger()
    root = _ledger(tmp_path, rows)

    readout = prediction_ledger.build_readout(
        prediction_ledger.read_ledger(sessions, root=root)
    )
    calibration = _horizon(readout, "rest_of_day")["calibration"]

    low = _confidence(calibration, "low")
    medium = _confidence(calibration, "medium")
    high = _confidence(calibration, "high")
    assert (low["n"], low["right"]) == (32, 28)
    assert (medium["n"], medium["right"]) == (32, 16)
    assert (high["n"], high["right"]) == (32, 8)
    assert low["rate"] == pytest.approx(0.875)
    assert high["rate"] == pytest.approx(0.25)
    for cell in (low, medium, high):
        assert cell["n"] >= evidence_stats.MIN_REPORTABLE_N
        assert cell["meets_floor"] is True

    assert calibration["high_beats_low"] is False
    said = str(calibration["statement"]).lower()
    assert "did not beat" in said, calibration["statement"]
    assert "high" in said and "low" in said, calibration["statement"]


# ---------------------------------------------------------------------------
# item 1 - what the ledger reader may never do
# ---------------------------------------------------------------------------
def test_the_two_horizons_are_never_pooled_into_one_number(tmp_path):
    """A rest-of-day call and a five-session call answer different questions.

    Forty of the first and five of the second. 45 is the number a reader that
    pooled them would print, so no cell anywhere in the readout may hold it.
    """
    import prediction_ledger

    sessions, day_rows = fx.two_weeks_of_clicks()
    rows = day_rows + fx.five_session_clicks(sessions)
    root = _ledger(tmp_path, rows)

    readout = prediction_ledger.build_readout(
        prediction_ledger.read_ledger(sessions, root=root)
    )

    assert _horizon(readout, "rest_of_day")["accuracy"]["n"] == 40
    week = _horizon(readout, "next_5_sessions")["accuracy"]
    assert week["n"] == 5
    assert week["right"] == 3
    for cell in _cells(readout):
        assert cell["n"] != 45, cell


def test_pooling_a_clicked_read_with_an_extracted_one_raises(tmp_path):
    """Decision 0021 answer 29, and `market_read_grades.pooled_accuracy`'s rule.

    An extracted stance is what the desk INFERRED from prose; a click is what
    the trader STATED. One rate over both describes neither, so the reader
    refuses rather than averaging them.
    """
    import market_journal
    import market_read_grades as grades
    import prediction_ledger

    sessions, rows = fx.two_weeks_of_clicks()
    session = sessions[-1]
    typed = market_journal.build_entry(
        text="SPY is leaking under the open and the sellers keep showing up.",
        session_date=session,
        timeframe="M5",
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        now=fx.stamp_at(session, 11),
    )
    extracted = grades.read_rows([typed], session=session)
    assert extracted and extracted[0]["source"] == grades.SOURCE_EXTRACTED, extracted
    rows = rows + [
        grades.grade_read(
            extracted[0],
            m5_bars=fx.session_tape(session, rising=False),
            atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
            now=fx.morning_after(session),
            context=fx.context(
                hour=11, direction=extracted[0]["direction"], confidence="",
                last_hour_spy="down", d1_environment="trending_down",
            ),
        )
    ]
    root = _ledger(tmp_path, rows)

    both = prediction_ledger.read_ledger(sessions, root=root)
    assert len({str(row.get("source")) for row in both}) == 2, both

    with pytest.raises(grades.PoolingError):
        prediction_ledger.build_readout(both)

    clicked = prediction_ledger.read_ledger(
        sessions, root=root, source=grades.SOURCE_CLICK
    )
    assert len(clicked) == 40
    assert _horizon(prediction_ledger.build_readout(clicked), "rest_of_day")[
        "accuracy"
    ]["n"] == 40


def test_a_regraded_read_is_counted_once_at_its_current_verdict(tmp_path):
    """Append-only: a matured grade is a NEW row naming the old one (TJ-10).

    The ledger therefore holds two rows for one read, and a reader that counted
    both would report two calls where the trader made one - and would count a
    `pending` that has since closed.
    """
    import market_read_grades as grades
    import prediction_ledger

    session = fx.sessions_ending(count=1)[0]
    entry = fx.click_entry(session=session, hour=9, direction="up", confidence="high")
    read = grades.read_rows([entry], session=session)[0]
    snapshot = fx.context(
        hour=9, direction="up", confidence="high",
        last_hour_spy="up", d1_environment="trending_up",
    )
    # First night: the bell has not rung, so the row is open.
    first = grades.grade_read(
        read, m5_bars=(), atr=None, now=fx.stamp_at(session, 11), context=snapshot
    )
    assert first["verdict"].startswith(grades.PENDING_PREFIX), first["verdict"]
    # Second night: the session has closed and the horizon is measured.
    second = grades.grade_read(
        read,
        m5_bars=fx.session_tape(session, rising=True),
        atr=fx.atr_for(fx.SESSION_MOVE, 3.0),
        now=fx.morning_after(session),
        supersedes=first["grade_id"],
        context=snapshot,
    )
    assert second["verdict"] == grades.VERDICT_RIGHT, second["verdict"]
    root = _ledger(tmp_path, [first, second])
    assert len(grades.read_grades(session, root=root)) == 2, "both rows stay on disk"

    current = prediction_ledger.read_ledger([session], root=root)
    assert len(current) == 1, current
    assert current[0]["verdict"] == grades.VERDICT_RIGHT

    cell = _horizon(prediction_ledger.build_readout(current), "rest_of_day")["accuracy"]
    assert (cell["n"], cell["right"], cell["pending"]) == (1, 1, 0)


def test_an_open_horizon_is_printed_and_left_out_of_both_halves(tmp_path):
    """TJ-11's blocker 2, inherited: `pending` is counted and shown, and it is
    in NEITHER half of the fraction. Four closed rows and two open ones is a
    rate over four, not over six.
    """
    import market_read_grades as grades
    import prediction_ledger

    session = fx.sessions_ending(count=1)[0]
    closed = fx.graded_session(
        session,
        [{"hour": hour, "direction": "up", "confidence": "medium"}
         for hour in fx.CLICK_HOURS],
        rising=True,
    )
    open_rows = []
    for hour in (11, 12):
        entry = fx.click_entry(
            session=session, hour=hour, direction="up", confidence="medium"
        )
        read = grades.read_rows([entry], session=session)[0]
        open_rows.append(
            grades.grade_read(
                read, m5_bars=(), atr=None, now=fx.stamp_at(session, 12, 30),
                context=fx.context(
                    hour=hour, direction="up", confidence="medium",
                    last_hour_spy="up", d1_environment="trending_up",
                ),
            )
        )
    root = _ledger(tmp_path, closed + open_rows)

    cell = _horizon(
        prediction_ledger.build_readout(prediction_ledger.read_ledger([session], root=root)),
        "rest_of_day",
    )["accuracy"]

    assert cell["n"] == 4
    assert cell["right"] == 4
    assert cell["pending"] == 2
    assert cell["rate"] == pytest.approx(1.0), "an open horizon never dilutes a rate"


# ---------------------------------------------------------------------------
# the empty state - which is the state the desk is in TODAY
# ---------------------------------------------------------------------------
def test_the_ledger_is_empty_until_the_first_click_and_says_so(tmp_path):
    """Measured 2026-09-20: `DAY_REVIEW_READS_DIR` does not exist on the desk.

    The click card went live today, so every surface TJ-16 builds opens on an
    empty ledger first. Nothing raises, no rate is 0.0 (that would read as "you
    are never right"), and the page says there are no clicked calls yet.
    """
    import prediction_ledger

    missing = tmp_path / "never_written"
    assert not missing.exists()

    rows = prediction_ledger.read_ledger(fx.sessions_ending(count=3), root=missing)
    assert rows == []

    readout = prediction_ledger.build_readout(rows)
    assert readout["empty"] is True
    assert "no clicked calls yet" in str(readout["statement"]).lower(), readout["statement"]
    for name in ("rest_of_day", "next_5_sessions"):
        cell = _horizon(readout, name)["accuracy"]
        assert cell["n"] == 0
        assert cell["rate"] is None, "nothing measured is not a rate of zero"

    line = prediction_ledger.your_reads(fx.LAST_SESSION, root=missing)
    assert line["empty"] is True
    assert line["n"] == 0
    assert "no clicked calls yet" in str(line["text"]).lower(), line["text"]


def test_your_reads_line_shows_one_sessions_tally_beside_its_baseline(tmp_path):
    """TJ-12 / item 5: Day Review's **Your reads** line, a file read for a worker.

    The oldest fixture session is a rising day the trader called up four times:
    four right of four, and `always Up` scored the same four. The line carries
    the counts as numbers so the page never re-derives them on the Qt thread.
    """
    import prediction_ledger

    sessions, rows = fx.two_weeks_of_clicks()
    root = _ledger(tmp_path, rows)
    session = sessions[0]

    line = prediction_ledger.your_reads(session, root=root)

    assert line["empty"] is False
    assert line["n"] == 4
    assert line["right"] == 4
    assert line["baselines"]["always_up"]["n"] == 4
    assert line["baselines"]["always_up"]["right"] == 4
    assert "4" in str(line["text"]), line["text"]
