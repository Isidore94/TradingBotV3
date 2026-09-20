r"""TJ-4 review round 2 - the three blockers in round 1's fix.

1. **Three queued sessions with NO PACK starved the sweep for ever.** The
   budget was taken off the LIST, and `queued_sessions` sorts oldest first, so
   three markers whose sessions had no pack sat at the head of the queue every
   night and no real request behind them was ever attempted. It was live: the
   home folder held three session folders and ZERO `pack.json`, so every Redo
   the trader pressed queued a packless session. Three fixes: the budget counts
   sessions NARRATED, the Redo click BUILDS the pack before it queues anything,
   and `request_redo` validates what it is handed.
2. **Both reply caps were below one real full day.** A regular session with
   every Mentor card answered carries 8 read rows and 25 citable ids (two
   scheduled cards store two entries each) before a forecast, a trade, a
   walk-away row, a congruence line or an internals mark. The caps now come
   FROM THE PACK, with absolute ceilings only to bound the render.
3. **A worst-case night declared 45 minutes against a 10-minute reserve.** The
   slot now asks the launch window itself before every call after its first.

**NO MODEL IS EVER CALLED.** Every path hands the module a fake `request`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj4_support as fx  # noqa: E402

ET = ZoneInfo("America/New_York")
SESSION = fx.SESSION

#: 02:00 ET inside the shipped 18:30-08:00 window: 360 minutes left.
DEEP_NIGHT = datetime(2026, 9, 21, 2, 0, tzinfo=ET)
#: 07:55 ET: FIVE minutes left, less than one call's worth.
WINDOW_CLOSING = datetime(2026, 9, 21, 7, 55, tzinfo=ET)

#: Real, closed exchange sessions with no pack ever built for them.
PACKLESS = ("2026-08-03", "2026-08-04", "2026-08-05")
#: Real, closed exchange sessions the trader queued for real.
QUEUED = ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11")


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path, monkeypatch):
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    base = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", base, raising=False)
    return base


def _pack_for(day: str, *, root: Path):
    import day_review_pack

    entry = fx.observation_only_entry(text=f"A quiet session ({day}).")
    pack = day_review_pack.build_pack(day, entries=[entry], now=fx.AFTER_THE_CLOSE)
    day_review_pack.write_pack(pack, root=root)
    return pack


def _already_narrated(day: str, *, root: Path, pack, headline="the OLD story"):
    from ai_jobs import day_review_narration as nar

    path = nar.narration_path(day, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema": nar.SCHEMA, "session_date": day,
            "inputs_hash": pack["inputs_hash"], "prompt_version": nar.PROMPT_VERSION,
            "model": "local-test-medium", "narration": {"headline": headline},
        }),
        encoding="utf-8",
    )
    return path


def _reply_for(pack, *, headline="a NEW story"):
    import day_review_pack

    allowed = list(day_review_pack.allowed_source_ids(pack))
    return {
        "model": "local-test-medium",
        "summary": {
            "headline": headline,
            "what_happened": "SPY drifted.",
            "what_you_thought": "You said little.",
            "were_you_right": [],
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "One note.",
            "sources": [allowed[0]],
        },
    }


def _spy(root: Path):
    import day_review_pack
    from ai_jobs import day_review_narration as nar

    seen: list[str] = []

    def request(**kwargs):
        evidence = kwargs.get("evidence") or {}
        day = str(evidence.get("session_date") or "")
        if kwargs.get("prompt_version") == nar.D1_VIEW_PROMPT_VERSION:
            seen.append(f"d1:{day}")
            ids = list(evidence.get("allowed_source_ids") or ())
            return {
                "model": "local-test-medium",
                "summary": {"belief_now": "Long.", "open_theses": [],
                            "sources": [ids[0]] if ids else []},
            }
        seen.append(day)
        return _reply_for(day_review_pack.read_pack(day, root=root),
                          headline=f"a NEW story for {day}")

    return request, seen


def _run(root, request, *, session=SESSION, now=DEEP_NIGHT, **kwargs):
    from ai_jobs.day_review_narration import run_day_review_narration

    return run_day_review_narration(
        session_date=session, now=now, root=root, request=request, **kwargs
    )


def _headline(day, *, root):
    from ai_jobs.day_review_narration import read_narration

    return str(((read_narration(day, root=root) or {}).get("narration") or {}).get("headline") or "")


# ---------------------------------------------------------------------------
# BLOCKER 1 - the packless queue must not starve the real one
# ---------------------------------------------------------------------------


def test_three_packless_markers_never_hold_up_a_real_redo(root):
    """The reviewer's reproduction: markers for three August days with no pack
    ever built, and one real queued redo behind them. Before the fix the real
    one was never attempted - on night 1, night 2 or any night after."""
    import day_review_pack

    _pack_for(SESSION, root=root)
    real = "2026-09-16"
    older = _pack_for(real, root=root)
    _already_narrated(real, root=root, pack=older)
    for day in PACKLESS:
        day_review_pack.request_redo(day, root=root)
    day_review_pack.request_redo(real, root=root)

    request, seen = _spy(root)
    outcome = _run(root, request)

    assert real in seen, seen
    assert _headline(real, root=root) == f"a NEW story for {real}"
    assert day_review_pack.redo_requested(real, root=root) is False
    for day in PACKLESS:
        assert day_review_pack.redo_requested(day, root=root) is True, day
    assert "no pack yet" in outcome["reason"], outcome["reason"]


def test_the_budget_counts_sessions_narrated_not_names_in_a_list(root):
    """Three packless markers cost NOTHING, so three real ones still run."""
    import day_review_pack
    from ai_jobs.day_review_narration import REDO_SWEEP_LIMIT

    _pack_for(SESSION, root=root)
    for day in PACKLESS:
        day_review_pack.request_redo(day, root=root)
    for day in QUEUED:
        pack = _pack_for(day, root=root)
        _already_narrated(day, root=root, pack=pack)
        day_review_pack.request_redo(day, root=root)

    request, seen = _spy(root)
    outcome = _run(root, request)

    narrated = [day for day in seen if day in QUEUED]
    assert narrated == list(QUEUED[:REDO_SWEEP_LIMIT]), narrated
    assert day_review_pack.redo_requested(QUEUED[REDO_SWEEP_LIMIT], root=root) is True
    assert "1 more queued session" in outcome["reason"], outcome["reason"]


def test_a_long_list_of_packless_sessions_is_named_up_to_five_then_counted(root):
    import day_review_pack
    from ai_jobs.day_review_narration import MAX_NAMED_UNBUILT

    _pack_for(SESSION, root=root)
    days = ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
            "2026-08-10", "2026-08-11")
    for day in days:
        day_review_pack.request_redo(day, root=root)

    request, _seen = _spy(root)
    outcome = _run(root, request)

    named = [day for day in days if day in outcome["reason"]]
    assert len(named) == MAX_NAMED_UNBUILT, named
    assert f"+{len(days) - MAX_NAMED_UNBUILT} more" in outcome["reason"], outcome["reason"]
    for day in days:
        assert day_review_pack.redo_requested(day, root=root) is True, day


# ---------------------------------------------------------------------------
# BLOCKER 1(c) - a marker goes where the trader meant it, or nowhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "asked",
    ["..", "2026-09-18-extra", "not-a-date", "", "2026-13-40",
     "2026-09-19",   # a Saturday
     "2099-01-05"],  # a real session, in the future
)
def test_request_redo_refuses_anything_but_a_closed_session_and_writes_nothing(
    root, asked
):
    """`session_dir` slices to ten characters, so `"2026-09-18-extra"` used to
    write into a REAL day's folder and `".."` wrote `redo_requested.json` at the
    `day_review` ROOT, outside `sessions/` (reviewer round 2). A write the
    trader asked for fails CLOSED: it lands where they meant it or not at all.
    """
    import day_review_pack

    with pytest.raises(ValueError):
        day_review_pack.request_redo(asked, root=root)

    assert list(root.rglob("redo_requested.json")) == []


def test_a_real_closed_session_is_still_accepted(root):
    import day_review_pack

    marker = day_review_pack.request_redo(SESSION, root=root)

    assert marker == root / "sessions" / SESSION / "redo_requested.json"
    assert marker.exists()


def test_a_folder_that_is_not_a_session_date_is_not_a_queue_entry(root):
    """Whatever else a future packet leaves beside the packs is not a request."""
    from ai_jobs.day_review_narration import queued_sessions

    stray = root / "sessions" / "notes"
    stray.mkdir(parents=True)
    (stray / "redo_requested.json").write_text("{}", encoding="utf-8")

    assert queued_sessions(root) == []


# ---------------------------------------------------------------------------
# BLOCKER 2 - a real full day fits
# ---------------------------------------------------------------------------


def _full_day_pack(root):
    """A regular session with every Mentor card answered: 8 reads, 25+ ids.

    The reviewer's measured shape (2026-09-20): six scheduled cards, two of
    which (`m5_d1`) store TWO entries each, so eight entries and eight read
    rows through the REAL grader.
    """
    import day_review_pack

    entries = []
    for hour, timeframes in (
        (7, ("M5",)), (8, ("M5", "D1")), (9, ("M5",)),
        (10, ("M5",)), (11, ("M5",)), (12, ("M5", "D1")),
    ):
        for timeframe in timeframes:
            entries.append(fx.observing_and_predicting_entry(
                timeframe=timeframe,
                horizon="rest_of_day" if timeframe == "M5" else "next_5_sessions",
                observation=f"What I see at {hour:02d}:00 on the {timeframe}.",
                stamp=fx.session_moment(SESSION, hour),
            ))
    reads, _grades = fx.graded_reads(entries)
    assert len(reads) == 8, f"fixture drift: {len(reads)} read rows"
    pack = day_review_pack.build_pack(
        SESSION,
        entries=entries,
        story=fx.daily_story(entries),
        reads=reads,
        congruence=fx.congruence(reads),
        trades=fx.trades(),
        now=fx.AFTER_THE_CLOSE,
    )
    assert len(day_review_pack.allowed_source_ids(pack)) >= 25, "fixture drift"
    day_review_pack.write_pack(pack, root=root)
    return pack


def _grade_everything(pack, *, headline="You called the day."):
    import day_review_pack

    claims = [
        {
            "claim": f"read {index + 1}",
            "source_id": next(
                item["source_id"] for item in pack["trader_said"]
                if item["entry_id"] == row["entry_id"] and item["kind"] == "prediction"
            ),
            "verdict": row["verdict"],
            "evidence_id": row["source_id"],
        }
        for index, row in enumerate(pack["reads"])
    ]
    return {
        "model": "local-test-medium",
        "summary": {
            "headline": headline,
            "what_happened": "SPY closed +2.00%.",
            "what_you_thought": "You called it up, six times.",
            "were_you_right": claims,
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "Every card answered.",
            "sources": list(day_review_pack.allowed_source_ids(pack)),
        },
    }


def test_a_real_full_day_grading_every_read_is_accepted(root):
    """8 reads, 25+ citable ids, every read graded. Under the round-1 caps
    (6 claims, 24 sources) this exact day was rejected WHOLE and the trader got
    no story at all on the days they answered every prompt."""
    from ai_jobs.day_review_narration import narration_path, read_narration

    from ai_jobs import day_review_narration as nar

    pack = _full_day_pack(root)
    reply = _grade_everything(pack)
    assert len(reply["summary"]["were_you_right"]) == 8
    assert len(reply["summary"]["sources"]) >= 25

    def request(**kwargs):
        # A full day carries D1 cards too, so the rolling view is asked as well.
        if kwargs.get("prompt_version") == nar.D1_VIEW_PROMPT_VERSION:
            ids = list((kwargs.get("evidence") or {}).get("allowed_source_ids") or ())
            return {
                "model": "local-test-medium",
                "summary": {"belief_now": "Long.", "open_theses": [], "sources": ids[:1]},
            }
        return reply

    outcome = _run(root, request)

    assert outcome["status"] == "ok", outcome
    assert narration_path(SESSION, root=root).exists()
    stored = read_narration(SESSION, root=root)
    assert stored["graded"] == {"reads_graded": 8, "reads_in_pack": 8}


def test_the_schema_the_model_is_given_carries_this_packs_own_numbers(root):
    """The model is TOLD the real bound, so it never has to choose two reads to
    drop in silence."""
    import day_review_pack

    pack = _full_day_pack(root)
    seen: list[dict] = []

    def request(**kwargs):
        seen.append(kwargs)
        return _grade_everything(pack)

    _run(root, request)

    schema = seen[0]["schema"]
    assert schema["properties"]["were_you_right"]["maxItems"] == len(pack["reads"])
    assert schema["properties"]["sources"]["maxItems"] == len(
        day_review_pack.allowed_source_ids(pack)
    )
    assert schema["additionalProperties"] is False


def test_grading_one_read_twice_is_rejected_whole(root):
    """One read, one verdict. Two claims on one row would print as two
    measurements of a single measured thing.

    TWO claims on an eight-read pack, so neither the per-pack cap nor any
    older fixed cap can be what rejects it: only the uniqueness rule can.
    """
    from ai_jobs.day_review_narration import narration_path

    pack = _full_day_pack(root)
    path = narration_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last night"}\n', encoding="utf-8")
    before = path.read_bytes()

    reply = _grade_everything(pack)
    first = reply["summary"]["were_you_right"][0]
    reply["summary"]["were_you_right"] = [first, dict(first)]
    assert len(reply["summary"]["were_you_right"]) == 2 < len(pack["reads"])

    outcome = _run(root, lambda **_k: reply)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_more_claims_than_the_pack_has_reads_is_rejected_whole(root):
    """The per-pack cap is a real gate, not a hint in the prompt.

    ONE read in the pack and TWO claims in the reply - both citing that read
    with its measured verdict, so every other rule is satisfied and the number
    is the only thing wrong with it.
    """
    import day_review_pack
    from ai_jobs.day_review_narration import narration_path

    entry = fx.observing_and_predicting_entry()
    reads, _grades = fx.graded_reads([entry])
    assert len(reads) == 1
    pack = day_review_pack.build_pack(
        SESSION, entries=[entry], reads=reads, now=fx.AFTER_THE_CLOSE
    )
    day_review_pack.write_pack(pack, root=root)
    path = narration_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last night"}\n', encoding="utf-8")
    before = path.read_bytes()

    read = pack["reads"][0]
    said = next(item for item in pack["trader_said"] if item["kind"] == "prediction")
    claim = {
        "claim": "Rest of day: up", "source_id": said["source_id"],
        "verdict": read["verdict"], "evidence_id": read["source_id"],
    }
    reply = _reply_for(pack, headline="one claim too many")
    reply["summary"]["were_you_right"] = [claim, dict(claim)]

    outcome = _run(root, lambda **_k: reply)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_the_ceilings_are_named_and_generous(root):
    """They bound the RENDER, not the day: a real day must never meet one."""
    from ai_jobs.day_review_narration import (
        MAX_GRADED_CLAIMS,
        MAX_OPEN_THESES,
        MAX_SOURCES,
    )

    pack = _full_day_pack(root)
    import day_review_pack

    assert MAX_GRADED_CLAIMS >= 8 * 4, MAX_GRADED_CLAIMS
    assert MAX_SOURCES >= len(day_review_pack.allowed_source_ids(pack)) * 4
    assert MAX_OPEN_THESES >= 8


# ---------------------------------------------------------------------------
# BLOCKER 3 - the window is asked again before every call after the first
# ---------------------------------------------------------------------------


def _queue_three(root):
    import day_review_pack

    _pack_for(SESSION, root=root)
    for day in QUEUED[:3]:
        pack = _pack_for(day, root=root)
        _already_narrated(day, root=root, pack=pack)
        day_review_pack.request_redo(day, root=root)


def test_a_night_with_the_whole_window_ahead_of_it_sweeps_three(root):
    import day_review_pack

    _queue_three(root)
    request, seen = _spy(root)

    outcome = _run(root, request, now=DEEP_NIGHT)

    assert [day for day in seen if day in QUEUED] == list(QUEUED[:3]), seen
    for day in QUEUED[:3]:
        assert day_review_pack.redo_requested(day, root=root) is False, day
    assert outcome["status"] == "ok", outcome


def test_a_sweep_that_would_run_past_the_window_stops_and_keeps_its_markers(root):
    """07:55 ET is five minutes from the window's close and one call is bounded
    at nine. The night's OWN story still runs - the slot's reserve bought that
    one - and nothing else starts, with every marker kept and SAID."""
    import day_review_pack

    _queue_three(root)
    request, seen = _spy(root)

    outcome = _run(root, request, now=WINDOW_CLOSING)

    assert seen == [SESSION], seen
    for day in QUEUED[:3]:
        assert day_review_pack.redo_requested(day, root=root) is True, day
    assert "window closed" in outcome["reason"], outcome["reason"]
    assert "3 more queued session" in outcome["reason"], outcome["reason"]
    assert outcome["status"] == "ok", outcome


def test_the_rolling_view_waits_for_the_next_night_when_the_window_is_closing(root):
    import day_review_pack
    from ai_jobs.day_review_narration import d1_view_path

    _pack_for(SESSION, root=root)
    d1 = fx.d1_note_entry(SESSION, text="The index grinds higher.")
    pack = day_review_pack.build_pack(SESSION, entries=[d1], now=fx.AFTER_THE_CLOSE)
    day_review_pack.write_pack(pack, root=root)
    request, seen = _spy(root)

    outcome = _run(root, request, now=WINDOW_CLOSING)

    assert not [name for name in seen if name.startswith("d1:")], seen
    assert not d1_view_path(root=root).exists()
    assert "rolling D1 view waits" in outcome["reason"], outcome["reason"]


def test_the_window_is_asked_for_one_calls_worth_of_room(root, monkeypatch):
    from ai_jobs import day_review_narration as nar
    from ai_jobs import window

    asked: list[tuple] = []

    def _launch_allowed(moment=None, *, reserve_minutes=0.0):
        asked.append((moment, reserve_minutes))
        return True, "window open"

    monkeypatch.setattr(window, "launch_allowed", _launch_allowed)
    _queue_three(root)
    request, _seen = _spy(root)

    _run(root, request, now=DEEP_NIGHT)

    assert asked, "the window was never asked"
    assert {reserve for _moment, reserve in asked} == {nar.SWEEP_CALL_MINUTES}
    assert nar.SWEEP_CALL_MINUTES == nar.TIMEOUT_SECONDS / 60.0
    for moment, _reserve in asked:
        assert moment is not None, "the window must be asked about a MOMENT"


def test_the_clock_is_injectable_and_the_run_never_sleeps(root, monkeypatch):
    """A window re-read needs a moving clock, and it may never move by
    sleeping: the seam takes `now` plus the time the run has actually spent."""
    import day_review_pack

    def _no_sleep(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("a narration slot may never sleep")

    monkeypatch.setattr(time, "sleep", _no_sleep)
    _queue_three(root)
    request, seen = _spy(root)
    # This root carries no D1 row, so the rolling view asks nothing: these are
    # the SWEEP's own asks, one before each queued session. The third crosses
    # the window's close and the sweep stops there.
    moments = [DEEP_NIGHT, DEEP_NIGHT, WINDOW_CLOSING]

    def clock():
        return moments.pop(0) if moments else WINDOW_CLOSING

    outcome = _run(root, request, now=DEEP_NIGHT, clock=clock)

    swept = [day for day in seen if day in QUEUED]
    assert swept == list(QUEUED[:2]), seen
    for day in swept:
        assert day_review_pack.redo_requested(day, root=root) is False, day
    assert day_review_pack.redo_requested(QUEUED[2], root=root) is True
    assert "window closed" in outcome["reason"], outcome["reason"]


# ---------------------------------------------------------------------------
# BLOCKER 1(b) - the page builds the pack before it queues anything
# ---------------------------------------------------------------------------
pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _PackService:
    """A reader WITH the pack seam, recording the thread it was built on."""

    def __init__(self, *, built=True) -> None:
        self.calls: list[tuple[str, int]] = []
        self._built = built

    def read_day(self, session_date, **_kwargs):
        return {"session_date": session_date}

    def build_pack_for(self, session_date, **_kwargs):
        self.calls.append((str(session_date), threading.get_ident()))
        return {"schema": "day_review_pack_v1"} if self._built else None


def _panel(qapp, monkeypatch, service, *, launched, window_open):
    from ui.panels import day_review_panel as module
    from ui.panels.day_review_panel import DayReviewPanel

    monkeypatch.setattr(
        module.window, "launch_allowed",
        lambda *_a, **_k: (window_open, "window open" if window_open else "outside"),
        raising=False,
    )
    panel = DayReviewPanel(
        service=service,
        clock=lambda: datetime(2026, 9, 19, 8, 0),
        redo_launcher=lambda session: launched.append(session),
    )
    monkeypatch.setattr(panel, "reload", lambda: None)
    return panel


def _drain(qapp, panel, *, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and panel._redo_workers:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_a_daytime_redo_builds_the_pack_off_the_qt_thread_before_queuing(
    qapp, tmp_path, monkeypatch
):
    """The night narrates a session it has a PACK for. A click that only wrote
    a marker queued a request nobody could ever answer - which, with the live
    home folder's three packless session folders, was every click."""
    import day_review_pack
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    service = _PackService(built=True)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=False)
    qt_thread = threading.get_ident()
    try:
        panel.redo_story()
        _drain(qapp, panel)

        assert [name for name, _thread in service.calls] == [panel.session_date()]
        assert service.calls[0][1] != qt_thread, "the pack build ran on the Qt thread"
        assert day_review_pack.redo_requested(panel.session_date(), root=root) is True
        assert launched == []
        assert "tonight" in panel.status.text().lower()
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()


def test_a_redo_whose_pack_cannot_be_built_queues_nothing_and_says_so(
    qapp, tmp_path, monkeypatch
):
    import day_review_pack
    import project_paths
    from ui.panels.day_review_panel import REDO_NO_PACK_NOTE

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    service = _PackService(built=False)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=False)
    try:
        panel.redo_story()
        _drain(qapp, panel)

        session = panel.session_date()
        assert day_review_pack.redo_requested(session, root=root) is False
        assert launched == []
        assert panel.status.text() == REDO_NO_PACK_NOTE.format(session=session)
        assert list(root.rglob("redo_requested.json")) == []
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()


def test_a_redo_at_night_builds_the_pack_before_it_starts_the_process(
    qapp, tmp_path, monkeypatch
):
    import day_review_pack
    import project_paths

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    service = _PackService(built=True)
    launched: list[str] = []
    panel = _panel(qapp, monkeypatch, service, launched=launched, window_open=True)
    try:
        panel.redo_story()
        _drain(qapp, panel)

        assert [name for name, _thread in service.calls] == [panel.session_date()]
        assert launched == [panel.session_date()]
        assert day_review_pack.redo_requested(panel.session_date(), root=root) is False
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()


def test_the_page_says_how_many_reads_the_story_graded(qapp, monkeypatch):
    """A SIZE statement, and only when it is fewer than the session held."""
    from ui.panels.day_review_panel import DayReviewPanel
    from ui.services.day_review_service import empty_payload

    story = {
        "schema": "day_review_narration_v1",
        "session_date": SESSION,
        "graded": {"reads_graded": 1, "reads_in_pack": 8},
        "narration": {
            "headline": "You called the open.",
            "what_happened": "SPY closed +2.00%.",
            "what_you_thought": "You called it up.",
            "were_you_right": [{
                "claim": "Rest of day: up", "source_id": "said:1:prediction",
                "verdict": "right", "evidence_id": "read:1",
            }],
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "One call.",
            "sources": ["said:1:prediction"],
        },
    }
    panel = DayReviewPanel(service=None, clock=lambda: datetime(2026, 9, 19, 8, 0)) \
        if False else DayReviewPanel(
            service=type("_S", (), {"read_day": lambda self, s, **k: {}})(),
            clock=lambda: datetime(2026, 9, 19, 8, 0),
        )
    monkeypatch.setattr(panel, "reload", lambda: None)
    try:
        payload = empty_payload(SESSION)
        payload["day_story"] = story
        panel.render(payload)
        assert "graded 1 of 8 reads" in panel.story_body.text()

        whole = dict(story)
        whole["graded"] = {"reads_graded": 8, "reads_in_pack": 8}
        payload["day_story"] = whole
        panel.render(payload)
        assert "graded" not in panel.story_body.text()
    finally:
        try:
            panel.shutdown()
        except Exception:  # noqa: BLE001
            pass
        panel.deleteLater()
        qapp.processEvents()
