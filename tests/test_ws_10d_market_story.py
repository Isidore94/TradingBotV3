"""Packet WS-10D - the Market Journal tells the market story and challenges my thesis.

WISHLIST 10D steps 1-3 plus 10K's forecast import. These tests were written BEFORE the
feature and are RED on ``claude/ws-10d-market-story``'s base commit
(``origin/claude/wishlist-sweep-2026-09-12``). The builder makes them pass and may only
ADD; nothing here may be renamed, relaxed or skipped to make a red go green.

===========================================================================
THE CONTRACT THESE TESTS DEFINE - the names the builder inherits
===========================================================================

``scripts/market_story.py`` (PURE: no Qt, no clock, no I/O, no model)
    ``BENCHMARKS == ("SPY", "QQQ", "IWM", "VXX", "TLT", "USO")`` - the trader's primary
    scope, in that order.
    ``KIND_TRADER = "trader"`` / ``KIND_MEASURED = "measured"`` / ``KIND_AI = "ai"`` -
    the three visibly distinct kinds.
    ``DailyStory`` with ``session_date``, ``trader_said``, ``external_forecasts``,
    ``measured``, ``ai_said``, ``context``, ``sources``, ``notes``. Every sequence is a
    tuple; ``ai_said`` is ALWAYS empty in this packet (code computes; the model explains
    later, in the narration stage).
    ``build_daily_story(session_date, *, entries=(), captures=None, context_row=None,
    index_bars=None, benchmarks=None) -> DailyStory``.

    It uses the entries it is HANDED, in ``created_at`` order, and does not re-filter
    them by the row's ``session_date`` - see "the ledger stamp trap" below.

``scripts/market_thesis.py`` (PURE extraction + an append-only store)
    ``EXTRACTOR_VERSION`` (a non-empty string; **never asserted literally** -
    CLAUDE.md's rule about versioned vocabularies), ``UNSTATED == "unstated"``,
    ``UNKNOWN == "unknown"``, the five stances
    ``STANCE_BULLISH/BEARISH/NEUTRAL/CAUTIOUS/UNSTATED``, the statuses
    ``STATUS_OPEN/STATUS_CLOSED/STATUS_INVALIDATED``, the link kinds
    ``LINK_SUPPORTS/LINK_CONTRADICTS/LINK_MENTIONS``, and
    ``KIND_THESIS = "thesis"`` / ``KIND_FORECAST = "forecast"``.
    ``ThesisDraft`` with ``entry_id``, ``session_date``, ``created_at``,
    ``extractor_version``, ``kind``, ``claim``, ``horizon``, ``horizon_sessions``,
    ``stance``, ``condition``, ``invalidation``, ``benchmarks``, ``spans``,
    ``is_prediction``.
    ``extract_thesis(entry) -> ThesisDraft``,
    ``link_entries(draft, later_entries) -> list[dict]``,
    ``resolve_status(draft, *, as_of, links=()) -> tuple[str, str]``,
    ``questions_for(draft) -> tuple[str, ...]`` (one or two),
    ``draft_row(draft, *, now=None) -> dict`` (the JSONL row, carrying ``thesis_id``),
    ``record_draft(draft, *, path=None, now=None) -> dict``,
    ``record_interpretation(*, entry_id, supersedes, text, path=None, now=None) -> dict``,
    ``read_rows(path=None) -> list[dict]``, ``current_theses(rows) -> list[dict]``,
    ``active_theses(rows, *, as_of) -> list[dict]``.
    Store: ``project_paths.MARKET_THESES_FILE`` (``market_theses.jsonl``), append-only,
    keyed on ``entry_id`` + ``extractor_version``. A row carries ``thesis_id``,
    ``entry_id``, ``extractor_version``, ``kind``, ``supersedes`` and ``recorded_at``.

``scripts/market_story_rollups.py`` (PURE builders + the nightly entry point)
    ``KIND_WEEKLY = "weekly"`` / ``KIND_MONTHLY = "monthly"`` /
    ``KIND_QUARTERLY = "quarterly"``,
    ``build_rollups(daily_stories, *, open_theses=()) -> dict[str, list[dict]]``,
    ``run_market_story_rollups(*, session_date="", now=None, stories=None,
    out_dir=None, **_ignored) -> dict``.
    A pack carries ``kind``, ``period_id`` (``2026-W37`` / ``2026-09`` / ``2026-Q3``),
    ``sessions_covered``, ``sessions_expected``, ``sessions_missing``, ``complete``,
    ``coverage_note``, ``open_theses``, ``inputs_hash``, and the period ids it was built
    from (``weeks`` on a month, ``months`` on a quarter).
    Written to ``out_dir / <kind> / <period_id>.json``; the default is
    ``project_paths.MARKET_STORY_ROLLUPS_DIR``.

``ui/services/market_journal_service.py``
    ``daily_story(session_date, *, index_bars=None) -> DailyStory`` (worker-thread call),
    ``theses_for(session_date) -> list[dict]``,
    ``save_interpretation(*, entry_id, supersedes, text) -> dict``,
    ``import_weekly_forecast(*, text, source_model="", created_at_claimed="",
    target_week="", scenarios=(), links=(), session_date="", now=None,
    theses_path=None) -> dict``.

``ui/panels/market_journal_panel.py``
    ``STORY_TRADER_HEADING == "You said"``, ``STORY_MEASURED_HEADING == "The market
    did"``, ``STORY_SOURCES_HEADING == "Sources"``,
    ``STORY_FORECAST_HEADING == "External forecast"``; the widgets ``story_view``
    (a ``QTextBrowser``), ``theses_list``, ``thesis_questions``,
    ``interpretation_box``, ``save_interpretation_button``. The existing worker payload
    gains ``"story"`` and ``"theses"`` and ``_render`` consumes them (G7: no store read
    on the Qt thread).

``scripts/market_journal.py``
    ``ORIGIN_EXTERNAL_FORECAST == "external_forecast"``.

``ai_jobs/runner.default_slots()``
    gains ``market_story_rollups`` as the LAST name of the deterministic stage, directly
    ahead of ``ai_summary``. ``EXPECTED_SLOT_ORDER`` in ``tests/test_ai_jobs_runner.py``
    gains the same name in the same place (edited in this commit, red until the slot
    exists).

===========================================================================
WHY EACH ASSERTION IS A NUMBER AND NOT A SHAPE
===========================================================================

* **The measured part is arithmetic on hand-built bars.** Twenty flat bars
  (O=100 H=101 L=99 C=100) then one session bar (O=100 H=101.5 L=99.5 C=101). Every
  true range is 2.0, so Wilder ATR(14) is EXACTLY 2.0 whether or not the session bar is
  included - which removes the one ambiguity in "range in ATR". The session range is
  2.0, so ``range_atr`` is 1.0; ``change_pct`` is 1.0 (percent); SMA20 over the last
  twenty closes is 100.05 and the close sits 0.475 ATR above it. A formula that divides
  by price, or averages the wrong window, produces a different number and fails.
* **The hindsight rule is proven by two notes with IDENTICAL text.** The same sentence
  typed at 11:00 Pacific and at 21:00 Pacific on 2026-09-11 gets
  ``written_after_the_session`` False and True from the REAL ``build_entry``, and
  therefore ``predicts_this_session`` True and False. A text-only classifier passes the
  first and fails the second.
* **The horizon is counted in SESSIONS, never calendar days.** A "this week" thesis
  opened on Tue 2026-09-08 is still OPEN on Mon 2026-09-14 (four sessions elapsed) and
  CLOSED on Tue 2026-09-15 (five). Five calendar days lands on Sun 2026-09-13, so a
  ``timedelta(days=5)`` implementation closes it on the Monday and fails.
* **The month is not the sum of its weeks.** September 2026 has 21 sessions. ISO week 36
  starts Mon 2026-08-31 and week 40 ends Fri 2026-10-02, so the five weeks that touch
  September hold 24 sessions between them. The monthly pack must name 21, each exactly
  once - a "sum the weeks" or "concatenate" implementation gives 24.
* **A short week is COMPLETE.** Week 37 of 2026 has four sessions (Labor Day is
  2026-09-07). A pack that assumes five sessions calls a complete week incomplete.
* **Idempotence is counted, not inspected.** Two runs over unchanged inputs: the second
  reports ``rebuilt == 0`` AND every file is byte-identical. Change ONE session's story
  and exactly three packs rebuild (its week, its month, its quarter) while the other
  week stays cached.

===========================================================================
THE SUBJECT-SESSION CONTRACT (repaired in Phase 0.31)
===========================================================================

``EvidenceLedger.append`` now preserves an explicit subject in ``session_date`` and
stores the market-local write day in ``written_session_date``. Older rows remain
readable through the created-at fallback. ``daily_story`` is driven here through a
REAL ``MarketJournalService`` over a REAL ``EvidenceLedger`` in ``tmp_path`` so a
selection that loses an evening note cannot pass.

Nothing here touches a live store: every path is ``tmp_path`` and every clock is
injected. No test sleeps.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PACIFIC = ZoneInfo("America/Los_Angeles")

#: A plain Friday. Regular close 16:00 New York = 13:00 Pacific.
SESSION = "2026-09-11"
#: The Tuesday a "this week" thesis is opened on.
THESIS_SESSION = date(2026, 9, 8)

#: The same sentence, typed twice on the same day either side of the close.
PREDICTION_TEXT = "I expect SPY to hold 5,400 into the close."


# ---------------------------------------------------------------------------
# helpers - hand-built inputs, nothing generated by the code under test
# ---------------------------------------------------------------------------
def _sessions_ending(last: date, count: int) -> list[date]:
    """The `count` exchange sessions ending at `last`, oldest first."""
    import market_calendar

    out: list[date] = []
    day = last
    while len(out) < count:
        if market_calendar.is_session(day):
            out.append(day)
        day = date.fromordinal(day.toordinal() - 1)
    return list(reversed(out))


def _flat_bars(last: date = date(2026, 9, 11)) -> list[dict]:
    """Twenty flat bars then one session bar. Every number below is exact.

    TR is 2.0 on every bar including the last, so Wilder ATR(14) == 2.0 and the
    "range in ATR" answer is the same whether or not today is in the average.
    """
    days = _sessions_ending(last, 21)
    bars = [
        {
            "dt": day.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        }
        for day in days[:-1]
    ]
    bars.append(
        {
            "dt": days[-1].isoformat(),
            "open": 100.0,
            "high": 101.5,
            "low": 99.5,
            "close": 101.0,
            "volume": 1_000_000.0,
        }
    )
    return bars


def _entry(text: str, *, hour: int = 11, session: str = SESSION, symbols=("SPY",)):
    """One entry through the REAL builder, with an injected Pacific clock."""
    import market_journal

    return market_journal.build_entry(
        text=text,
        session_date=session,
        timeframe="D1",
        symbols=symbols,
        origin=market_journal.ORIGIN_JOURNAL_PAGE,
        now=datetime(
            int(session[:4]), int(session[5:7]), int(session[8:10]), hour, 0, tzinfo=PACIFIC
        ),
    )


def _service(tmp_path):
    """A real service over a real ledger in tmp_path (the pattern in test_market_journal)."""
    import market_journal
    from evidence_ledger import EvidenceLedger
    from ui.services.market_journal_service import MarketJournalService

    instance = MarketJournalService()
    instance._ledger = EvidenceLedger(
        stream=market_journal.STREAM,
        schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        directory=Path(tmp_path) / "ledger",
    )
    return instance


def _ledger_bytes(tmp_path) -> dict[str, str]:
    """Every ledger segment's sha256, so "the original is untouched" is provable."""
    out: dict[str, str] = {}
    for path in sorted((Path(tmp_path) / "ledger").glob("*.jsonl")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _cell(story, symbol: str) -> dict:
    for row in story.measured:
        if str(row.get("symbol") or "") == symbol:
            return dict(row)
    raise AssertionError(f"no measured cell for {symbol}: {[r.get('symbol') for r in story.measured]}")


# ===========================================================================
# Item 1 - the daily story: the trader's words, the measured facts, the sources
# ===========================================================================
def test_a_session_with_no_note_has_an_empty_trader_said_and_says_so():
    """No note = no invented thesis. The facts still stand; the words do not.

    The story is still built - the market did what it did whether or not the trader
    wrote about it - but `trader_said` is EMPTY and one of the story's own sentences
    says the session carries no note. An implementation that summarises the measured
    part into a sentence attributed to the trader fails here.
    """
    import market_story

    story = market_story.build_daily_story(
        SESSION, entries=(), index_bars={"SPY": _flat_bars()}
    )

    assert story.trader_said == ()
    assert story.ai_said == (), "this packet writes no model text into the story"
    said = " ".join(story.notes).lower()
    assert "no note" in said, story.notes
    assert SESSION in " ".join(story.notes)
    # The facts are unaffected by the silence.
    assert _cell(story, "SPY")["status"] == "measured"


def test_two_opposing_notes_on_one_day_are_both_kept_in_created_order():
    """The trader is allowed to change their mind inside a session.

    Nothing here resolves, reconciles, averages or picks between two notes. Both are
    in `trader_said`, verbatim, oldest first.
    """
    import market_story

    bullish = _entry("SPY is strong here; I like the reclaim.", hour=7)
    bearish = _entry("Taking that back - SPY is weak and I am wrong.", hour=11)

    story = market_story.build_daily_story(
        SESSION, entries=(bullish, bearish), index_bars={"SPY": _flat_bars()}
    )

    assert len(story.trader_said) == 2
    assert [row["text"] for row in story.trader_said] == [bullish["text"], bearish["text"]]
    assert [row["entry_id"] for row in story.trader_said] == [
        bullish["entry_id"],
        bearish["entry_id"],
    ]
    assert all(row["kind"] == market_story.KIND_TRADER for row in story.trader_said)


def test_a_benchmark_with_no_bars_is_unmeasured_and_never_invented():
    """An unsupported claim gets no measured part.

    Six benchmarks are in scope and only SPY has bars, so five cells must say
    UNMEASURED with every number None and a reason naming the symbol. "Missing data is
    uncertainty, never confirmation" (plan.md sec 5): a zero, a carried-forward close
    or a silently dropped cell would all read as a measurement.
    """
    import market_story

    assert market_story.BENCHMARKS == ("SPY", "QQQ", "IWM", "VXX", "TLT", "USO")

    story = market_story.build_daily_story(
        SESSION,
        entries=(_entry("USO looks ready to run."),),
        index_bars={"SPY": _flat_bars()},
    )

    assert len(story.measured) == 6
    assert [row["symbol"] for row in story.measured] == list(market_story.BENCHMARKS)

    uso = _cell(story, "USO")
    assert uso["status"] == "unmeasured"
    assert uso["close"] is None
    assert uso["change_pct"] is None
    assert uso["range_atr"] is None
    assert uso["position_vs_sma20"]["sma20"] is None
    assert uso["position_vs_sma20"]["distance_atr"] is None
    assert uso["position_vs_sma20"]["side"] == "unknown"
    assert uso["bars_through"] == ""
    assert "USO" in str(uso["reason"]), uso["reason"]
    assert all(row["kind"] == market_story.KIND_MEASURED for row in story.measured)


def test_the_measured_part_is_the_arithmetic_and_carries_its_rule_versions():
    """Twenty flat bars and one session bar - every number below is exact.

    ATR(14) is 2.0 because every true range is 2.0. The session range is 2.0, so
    `range_atr` is 1.0. The close moved 100.0 -> 101.0, so `change_pct` is 1.0 PERCENT
    (a fraction of 0.01 fails). SMA20 over the last twenty closes is (19*100 + 101)/20
    = 100.05, and 101.0 sits (101.0 - 100.05) / 2.0 = 0.475 ATR above it.
    """
    import market_story

    story = market_story.build_daily_story(SESSION, index_bars={"SPY": _flat_bars()})
    spy = _cell(story, "SPY")

    assert spy["status"] == "measured"
    assert spy["close"] == pytest.approx(101.0)
    assert spy["change_pct"] == pytest.approx(1.0), "percent, not a fraction"
    assert spy["range_atr"] == pytest.approx(1.0)
    assert spy["position_vs_sma20"]["sma20"] == pytest.approx(100.05)
    assert spy["position_vs_sma20"]["side"] == "above"
    assert spy["position_vs_sma20"]["distance_atr"] == pytest.approx(0.475)
    assert spy["bars_through"] == SESSION
    assert spy["bars_used"] == 21
    # Named rule versions, so a later change to the arithmetic is visible in the row.
    versions = spy["rule_versions"]
    assert set(versions) == {"change_pct", "range_atr", "position_vs_sma20"}
    assert all(isinstance(name, str) and name.strip() for name in versions.values())


def test_a_bar_dated_after_the_session_is_not_measured():
    """Completed bars only (plan.md sec 5). Tomorrow's forming bar is not today's fact.

    The same series plus one wild bar dated the NEXT session must produce a byte-equal
    cell. A reader that takes "the last bar" reports a 19% day and fails.
    """
    import market_story

    clean = _flat_bars()
    contaminated = clean + [
        {
            "dt": "2026-09-14",
            "open": 101.0,
            "high": 121.0,
            "low": 100.0,
            "close": 120.0,
            "volume": 5.0,
        }
    ]

    before = _cell(market_story.build_daily_story(SESSION, index_bars={"SPY": clean}), "SPY")
    after = _cell(
        market_story.build_daily_story(SESSION, index_bars={"SPY": contaminated}), "SPY"
    )

    assert after == before
    assert after["close"] == pytest.approx(101.0)
    assert after["bars_through"] == SESSION


def test_corrected_price_inputs_change_only_the_measured_part():
    """A bad print fixed overnight rewrites the facts and never the words.

    Same entries, same captures, same context row, different bars: `trader_said` and
    `sources` come back identical and only `measured` moves.
    """
    import market_story

    note = _entry("Held the level all day.")
    captures = {note["entry_id"]: {"entry_id": note["entry_id"], "digest": "SPY D1 ..."}}
    context = {"event_at": "2026-09-11T20:35:00+00:00", "session_date": SESSION}

    corrected = [dict(bar) for bar in _flat_bars()]
    corrected[-1]["close"] = 99.0
    corrected[-1]["low"] = 98.5
    corrected[-1]["high"] = 100.5

    first = market_story.build_daily_story(
        SESSION,
        entries=(note,),
        captures=captures,
        context_row=context,
        index_bars={"SPY": _flat_bars()},
    )
    second = market_story.build_daily_story(
        SESSION,
        entries=(note,),
        captures=captures,
        context_row=context,
        index_bars={"SPY": corrected},
    )

    assert second.trader_said == first.trader_said
    assert second.sources == first.sources
    assert second.measured != first.measured
    assert _cell(second, "SPY")["close"] == pytest.approx(99.0)
    assert _cell(second, "SPY")["change_pct"] == pytest.approx(-1.0)
    # The sources name every source the story was built from.
    assert first.sources["entry_ids"] == (note["entry_id"],)
    assert first.sources["capture_entry_ids"] == (note["entry_id"],)
    assert first.sources["context_row_id"] == "2026-09-11T20:35:00+00:00"


def test_the_same_sentence_predicts_the_session_only_when_it_was_typed_during_it(tmp_path):
    """A late-written note is never shown as a prediction about that session.

    ONE sentence, typed twice on 2026-09-11: 11:00 Pacific (the tape is moving) and
    21:00 Pacific (eight hours after the close). `build_entry` COMPUTES
    `written_after_the_session` from the real exchange close, so the two rows differ
    only in when they were typed - and the story must read the second as hindsight.

    Driven through a REAL service and ledger because the 21:00 Pacific write is the
    next New-York date. The subject must stay 2026-09-11 while the write day records
    2026-09-12.
    """
    service = _service(tmp_path)

    midday = service.write_entry(
        text=PREDICTION_TEXT,
        session_date=SESSION,
        timeframe="D1",
        origin="journal_page",
        now=datetime(2026, 9, 11, 11, 0, tzinfo=PACIFIC),
    )
    evening = service.write_entry(
        text=PREDICTION_TEXT,
        session_date=SESSION,
        timeframe="D1",
        origin="journal_page",
        now=datetime(2026, 9, 11, 21, 0, tzinfo=PACIFIC),
    )
    assert midday["ok"] is True and evening["ok"] is True
    assert midday["entry"]["written_after_the_session"] is False
    assert evening["entry"]["written_after_the_session"] is True
    assert evening["entry"]["session_date"] == SESSION
    assert evening["entry"]["written_session_date"] == "2026-09-12"

    story = service.daily_story(SESSION, index_bars={"SPY": _flat_bars()})

    assert [row["entry_id"] for row in story.trader_said] == [
        midday["entry"]["entry_id"],
        evening["entry"]["entry_id"],
    ], "the evening note belongs to the session it was written about"
    early, late = story.trader_said
    assert early["written_after_the_session"] is False
    assert late["written_after_the_session"] is True
    assert early["predicts_this_session"] is True
    assert late["predicts_this_session"] is False, (
        "the same words typed after the close describe the session, they do not predict it"
    )


# ===========================================================================
# Item 2 - deterministic thesis extraction
# ===========================================================================
CONDITIONED = (
    "I expect SPY to hold above 5,400 this week, as long as VIX stays under 20. "
    "If SPY loses 5,400 on a closing basis I am wrong."
)
AMBIGUOUS = "I am not sure whether SPY is strong or weak here; watching."
BARE = "SPY was heavy into the close."


def test_an_ambiguous_note_has_an_unstated_stance_and_no_stance_span():
    """Both a bullish and a bearish word and no view: `unstated`, not a coin flip.

    A keyword counter that takes the first match reads "strong" and calls it bullish.
    An unstated field carries NO span, because a span is a quotation and there is
    nothing to quote.
    """
    import market_thesis

    draft = market_thesis.extract_thesis(_entry(AMBIGUOUS))

    assert draft.stance == market_thesis.STANCE_UNSTATED == "unstated"
    assert "stance" not in draft.spans
    assert draft.benchmarks == ("SPY",)
    assert isinstance(draft.extractor_version, str) and draft.extractor_version.strip()
    assert draft.kind == market_thesis.KIND_THESIS


def test_a_stated_invalidation_is_quoted_by_its_span_and_an_absent_one_stays_unstated():
    """Every field carries the source span, and the span must reproduce the field.

    The invariant is not "there is a span" - it is that slicing the entry's own text
    with the span gives back exactly what the field claims. A paraphrase cannot pass.
    """
    import market_thesis

    entry = _entry(CONDITIONED)
    draft = market_thesis.extract_thesis(entry)

    assert draft.stance == market_thesis.STANCE_BULLISH
    assert draft.benchmarks == ("SPY",)
    assert draft.horizon_sessions == 5, "'this week' is five exchange sessions"

    assert draft.invalidation != market_thesis.UNSTATED
    assert "5,400" in draft.invalidation
    assert draft.invalidation in entry["text"], "verbatim, never paraphrased"
    start, end = draft.spans["invalidation"]
    assert entry["text"][start:end] == draft.invalidation

    assert draft.condition != market_thesis.UNSTATED
    assert "VIX" in draft.condition
    c_start, c_end = draft.spans["condition"]
    assert entry["text"][c_start:c_end] == draft.condition

    bare = market_thesis.extract_thesis(_entry(BARE))
    assert bare.invalidation == market_thesis.UNSTATED == "unstated"
    assert bare.condition == market_thesis.UNSTATED
    assert "invalidation" not in bare.spans
    assert "condition" not in bare.spans


def test_a_thesis_stays_open_across_a_weekend_and_closes_on_its_fifth_session():
    """The horizon is counted in SESSIONS, not calendar days.

    Opened Tue 2026-09-08 with a five-session horizon. Four sessions have elapsed by
    Mon 2026-09-14 (09-09, 09-10, 09-11, 09-14), so it is still OPEN; the fifth is Tue
    2026-09-15, where it CLOSES. Five calendar days from 09-08 is Sun 09-13, so a
    `timedelta(days=...)` window closes it a session early and fails on the Monday.
    """
    import market_thesis

    entry = _entry(CONDITIONED, session=THESIS_SESSION.isoformat())
    draft = market_thesis.extract_thesis(entry)
    assert draft.horizon_sessions == 5

    status, reason = market_thesis.resolve_status(draft, as_of=date(2026, 9, 14))
    assert status == market_thesis.STATUS_OPEN == "open", reason

    status, reason = market_thesis.resolve_status(draft, as_of=date(2026, 9, 15))
    assert status == market_thesis.STATUS_CLOSED, reason
    assert "5" in reason or "session" in reason.lower(), reason


def test_a_later_note_contradicts_only_on_a_stance_reversal_on_the_same_benchmark():
    """A contradiction is a stance reversal on the same benchmark. Nothing subtler.

    Four later notes and four different answers: agreeing on SPY SUPPORTS, reversing on
    SPY CONTRADICTS, naming SPY without a stance only MENTIONS, and a note about QQQ is
    not linked at all. A note after the horizon is not linked either - a thesis about
    this week is not evidence about the week after.
    """
    import market_thesis

    draft = market_thesis.extract_thesis(
        _entry(CONDITIONED, session=THESIS_SESSION.isoformat())
    )

    supporting = _entry("SPY held 5,400 again into the close; still strong.", session="2026-09-09")
    contradicting = _entry("SPY lost 5,400 and closed weak. I was wrong.", session="2026-09-10")
    mentioning = _entry("SPY printed 5,398 at one point today.", session="2026-09-10")
    other = _entry("QQQ is strong and reclaimed its anchor.", session="2026-09-10", symbols=("QQQ",))
    too_late = _entry("SPY is weak again.", session="2026-09-21")

    links = market_thesis.link_entries(
        draft, (supporting, contradicting, mentioning, other, too_late)
    )
    by_entry = {row["entry_id"]: row["link"] for row in links}

    assert by_entry.get(supporting["entry_id"]) == market_thesis.LINK_SUPPORTS
    assert by_entry.get(contradicting["entry_id"]) == market_thesis.LINK_CONTRADICTS
    assert by_entry.get(mentioning["entry_id"]) == market_thesis.LINK_MENTIONS
    assert other["entry_id"] not in by_entry, "a different benchmark is not this thesis"
    assert too_late["entry_id"] not in by_entry, "outside the horizon is not this thesis"
    assert all(row["benchmark"] == "SPY" for row in links)


def test_a_grounded_question_quotes_a_stated_condition_and_invents_none():
    """"Your caution was conditioned on X" is asked only when X was stated.

    One or two questions, never more. For the conditioned note one of them carries the
    condition VERBATIM; for the bare note nothing is asked about a condition and the
    word `unstated` never reaches the trader as a question.
    """
    import market_thesis

    draft = market_thesis.extract_thesis(_entry(CONDITIONED))
    questions = market_thesis.questions_for(draft)
    assert 1 <= len(questions) <= 2, questions
    assert any(draft.condition in question for question in questions), questions

    bare = market_thesis.questions_for(market_thesis.extract_thesis(_entry(BARE)))
    assert 1 <= len(bare) <= 2, bare
    assert not any(market_thesis.UNSTATED in question for question in bare), bare
    assert not any("conditioned on" in question.lower() for question in bare), bare


def test_an_edited_interpretation_supersedes_the_draft_and_leaves_the_entry_untouched(
    tmp_path,
):
    """A trader edit is a NEW row. The draft stays; the journal entry is never rewritten.

    Proven against disk twice over: both thesis rows are still readable with their own
    text, and the journal ledger's bytes are identical before and after.
    """
    import market_thesis
    import project_paths

    assert Path(project_paths.MARKET_THESES_FILE).name == "market_theses.jsonl"

    service = _service(tmp_path)
    written = service.write_entry(
        text=CONDITIONED,
        session_date=SESSION,
        timeframe="D1",
        origin="journal_page",
        now=datetime(2026, 9, 11, 11, 0, tzinfo=PACIFIC),
    )
    assert written["ok"] is True
    entry = written["entry"]
    before = _ledger_bytes(tmp_path)
    assert before, "the entry reached disk"

    theses = Path(tmp_path) / "market_theses.jsonl"
    draft = market_thesis.extract_thesis(entry)
    draft_row = market_thesis.record_draft(
        draft, path=theses, now=datetime(2026, 9, 11, 19, 0, tzinfo=timezone.utc)
    )
    assert draft_row["entry_id"] == entry["entry_id"]
    assert draft_row["extractor_version"] == draft.extractor_version
    assert draft_row["supersedes"] == ""

    edited = market_thesis.record_interpretation(
        entry_id=entry["entry_id"],
        supersedes=draft_row["thesis_id"],
        text="What I actually meant: I want 5,400 to hold on a CLOSING basis only.",
        path=theses,
        now=datetime(2026, 9, 12, 19, 0, tzinfo=timezone.utc),
    )

    rows = market_thesis.read_rows(theses)
    assert len(rows) == 2, "append-only: the draft is still on disk"
    stored_draft = next(row for row in rows if row["thesis_id"] == draft_row["thesis_id"])
    assert stored_draft["claim"] == draft_row["claim"], "the draft was not rewritten"

    current = market_thesis.current_theses(rows)
    assert len(current) == 1
    assert current[0]["thesis_id"] == edited["thesis_id"]
    assert current[0]["supersedes"] == draft_row["thesis_id"]
    assert current[0]["text"] == (
        "What I actually meant: I want 5,400 to hold on a CLOSING basis only."
    )

    assert _ledger_bytes(tmp_path) == before, "the journal entry itself is never touched"


# ===========================================================================
# Item 4 - the weekly forecast import (WISHLIST 10K)
# ===========================================================================
FORECAST_TEXT = (
    "Week of Sept 14: base case SPY grinds to 5,500 on soft CPI.\n"
    "Bear case: a hot print takes it back to 5,350."
)


def test_a_pasted_forecast_keeps_an_unknown_creation_time_unknown_and_is_not_a_thesis(
    tmp_path,
):
    """Outside commentary, imported. Never the trader's adopted view.

    Three refusals in one: the creation time nobody supplied stays the literal string
    `unknown` and is NOT quietly filled from the import moment; the sidecar is
    `kind=forecast`, so `active_theses` never returns it; and the story keeps it out of
    `trader_said` and under its own heading. A later import is not information known
    earlier - that is what `created_at_claimed` vs `imported_at` records.
    """
    import market_journal
    import market_story
    import market_thesis

    assert market_journal.ORIGIN_EXTERNAL_FORECAST == "external_forecast"

    service = _service(tmp_path)
    theses = Path(tmp_path) / "market_theses.jsonl"
    imported_at = datetime(2026, 9, 12, 17, 30, tzinfo=timezone.utc)

    result = service.import_weekly_forecast(
        text=FORECAST_TEXT,
        source_model="",
        created_at_claimed="",
        target_week="2026-W38",
        scenarios=("base: 5,500", "bear: 5,350"),
        session_date=SESSION,
        now=imported_at,
        theses_path=theses,
    )

    assert result["ok"] is True
    entry = result["entry"]
    assert entry["origin"] == market_journal.ORIGIN_EXTERNAL_FORECAST
    assert entry["text"] == FORECAST_TEXT, "verbatim, newlines and all"

    sidecar = result["forecast"]
    assert sidecar["kind"] == market_thesis.KIND_FORECAST == "forecast"
    assert sidecar["created_at_claimed"] == market_thesis.UNKNOWN == "unknown"
    assert sidecar["source_model"] == market_thesis.UNKNOWN
    assert sidecar["imported_at"].startswith("2026-09-12T17:30")
    assert sidecar["created_at_claimed"] != sidecar["imported_at"]
    assert sidecar["target_week"] == "2026-W38"
    assert list(sidecar["scenarios"]) == ["base: 5,500", "bear: 5,350"]
    assert sidecar["entry_id"] == entry["entry_id"]

    rows = market_thesis.read_rows(theses)
    assert len(rows) == 1
    assert market_thesis.active_theses(rows, as_of=date(2026, 9, 14)) == [], (
        "a forecast is outside commentary, never an adopted thesis"
    )

    story = market_story.build_daily_story(
        SESSION, entries=(entry,), index_bars={"SPY": _flat_bars()}
    )
    assert story.trader_said == (), "the trader wrote none of these words"
    assert len(story.external_forecasts) == 1
    assert story.external_forecasts[0]["entry_id"] == entry["entry_id"]
    assert story.external_forecasts[0]["text"] == FORECAST_TEXT


def test_a_supplied_creation_time_is_kept_exactly_as_supplied(tmp_path):
    """Known stays known, and the import moment never overwrites it."""
    import market_thesis

    service = _service(tmp_path)
    theses = Path(tmp_path) / "market_theses.jsonl"

    result = service.import_weekly_forecast(
        text=FORECAST_TEXT,
        source_model="gpt-5-thinking",
        created_at_claimed="2026-09-12T08:00:00-07:00",
        target_week="2026-W38",
        scenarios=(),
        session_date=SESSION,
        now=datetime(2026, 9, 12, 17, 30, tzinfo=timezone.utc),
        theses_path=theses,
    )

    sidecar = result["forecast"]
    assert sidecar["created_at_claimed"] == "2026-09-12T08:00:00-07:00"
    assert sidecar["source_model"] == "gpt-5-thinking"
    assert sidecar["created_at_claimed"] != market_thesis.UNKNOWN


# ===========================================================================
# Item 5 - the rollups and their nightly slot
# ===========================================================================
def _story_dicts(sessions, *, marker: str = "") -> list[dict]:
    """The minimum a rollup needs: a dated story per session."""
    return [
        {
            "session_date": day,
            "trader_said": ({"entry_id": f"mj-{day}", "text": f"{day} {marker}".strip()},),
            "measured": (),
            "sources": {"entry_ids": (f"mj-{day}",)},
        }
        for day in sessions
    ]


SEPTEMBER_SESSIONS = [day.isoformat() for day in _sessions_ending(date(2026, 9, 30), 21)]
AUGUST_TAIL = ["2026-08-31"]
OCTOBER_HEAD = ["2026-10-01", "2026-10-02"]
WEEK_37 = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]
WEEK_38 = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]


def test_a_session_inside_two_overlapping_weeks_is_counted_once_in_the_month():
    """September 2026 has 21 sessions; the five weeks touching it hold 24.

    ISO week 36 starts Mon 2026-08-31 and week 40 ends Fri 2026-10-02, so a monthly
    pack built by concatenating its weeks names 24 sessions and three that are not in
    September. The month must name 21, each exactly once, and none from August or
    October. Week 37 has FOUR sessions (Labor Day is 2026-09-07) and is COMPLETE - an
    implementation that assumes five calls a full week short.
    """
    import market_story_rollups as rollups

    assert len(SEPTEMBER_SESSIONS) == 21
    stories = _story_dicts(AUGUST_TAIL + SEPTEMBER_SESSIONS + OCTOBER_HEAD)
    open_thesis = {"entry_id": "mj-open", "claim": "SPY holds 5,400", "status": "open"}

    packs = rollups.build_rollups(stories, open_theses=(open_thesis,))

    weekly = {pack["period_id"]: pack for pack in packs[rollups.KIND_WEEKLY]}
    monthly = {pack["period_id"]: pack for pack in packs[rollups.KIND_MONTHLY]}

    week37 = weekly["2026-W37"]
    assert list(week37["sessions_covered"]) == WEEK_37
    assert list(week37["sessions_expected"]) == WEEK_37
    assert week37["complete"] is True
    assert list(week37["sessions_missing"]) == []

    september = monthly["2026-09"]
    covered = list(september["sessions_covered"])
    assert len(covered) == 21, covered
    assert len(set(covered)) == 21, "a session is named once, never once per week"
    assert covered == sorted(covered)
    assert "2026-08-31" not in covered
    assert "2026-10-01" not in covered and "2026-10-02" not in covered
    assert september["complete"] is True

    # It really was built from the overlapping weeks: they hold 24 sessions between them.
    assert list(september["weeks"]) == [
        "2026-W36",
        "2026-W37",
        "2026-W38",
        "2026-W39",
        "2026-W40",
    ]
    assert sum(len(weekly[week]["sessions_covered"]) for week in september["weeks"]) == 24

    # Open theses carry forward into every period.
    assert open_thesis in list(september["open_theses"])
    assert open_thesis in list(week37["open_theses"])


def test_an_incomplete_period_names_the_sessions_it_is_missing():
    """Explicit session coverage at the boundaries, in words as well as counts."""
    import market_story_rollups as rollups

    kept = [day for day in SEPTEMBER_SESSIONS if day not in {"2026-09-10", "2026-09-11"}]
    assert len(kept) == 19

    packs = rollups.build_rollups(_story_dicts(AUGUST_TAIL + kept + OCTOBER_HEAD))
    weekly = {pack["period_id"]: pack for pack in packs[rollups.KIND_WEEKLY]}
    monthly = {pack["period_id"]: pack for pack in packs[rollups.KIND_MONTHLY]}
    quarterly = {pack["period_id"]: pack for pack in packs[rollups.KIND_QUARTERLY]}

    week37 = weekly["2026-W37"]
    assert week37["complete"] is False
    assert list(week37["sessions_missing"]) == ["2026-09-10", "2026-09-11"]
    assert "2026-09-10" in week37["coverage_note"]
    assert "2026-09-11" in week37["coverage_note"]

    september = monthly["2026-09"]
    assert len(september["sessions_covered"]) == 19
    assert september["complete"] is False
    assert list(september["sessions_missing"]) == ["2026-09-10", "2026-09-11"]
    assert september["coverage_note"].strip()

    q3 = quarterly["2026-Q3"]
    assert list(q3["months"]) == ["2026-09"], "July and August were never written"
    assert q3["complete"] is False
    assert len(q3["sessions_covered"]) == 19


def test_a_second_run_over_unchanged_inputs_rebuilds_nothing_and_a_changed_day_rebuilds_three(
    tmp_path,
):
    """Rebuilt only when an input's hash changed - counted, not inspected.

    Two weeks in, so four packs exist: W37, W38, 2026-09 and 2026-Q3. Run twice over
    identical inputs and nothing is rebuilt and no file moves a byte. Change ONE session
    in week 37 and exactly three packs rebuild - its week, its month, its quarter -
    while week 38 stays cached.
    """
    import market_story_rollups as rollups

    out = Path(tmp_path) / "story_rollups"
    stories = _story_dicts(WEEK_37 + WEEK_38)
    clock = datetime(2026, 9, 19, 6, 0, tzinfo=timezone.utc)

    first = rollups.run_market_story_rollups(
        session_date="2026-09-18", now=clock, stories=stories, out_dir=out
    )
    assert first["rebuilt"] == 4, first
    assert first["cached"] == 0, first

    snapshot = {
        path.relative_to(out).as_posix(): path.read_bytes() for path in sorted(out.rglob("*.json"))
    }
    assert len(snapshot) == 4, sorted(snapshot)
    assert "weekly/2026-W37.json" in snapshot
    assert "monthly/2026-09.json" in snapshot
    assert "quarterly/2026-Q3.json" in snapshot

    second = rollups.run_market_story_rollups(
        session_date="2026-09-18", now=clock, stories=stories, out_dir=out
    )
    assert second["rebuilt"] == 0, second
    assert second["cached"] == 4, second
    assert {
        path.relative_to(out).as_posix(): path.read_bytes() for path in sorted(out.rglob("*.json"))
    } == snapshot

    changed = _story_dicts(WEEK_37, marker="revised") + _story_dicts(WEEK_38)
    third = rollups.run_market_story_rollups(
        session_date="2026-09-18", now=clock, stories=changed, out_dir=out
    )
    assert third["rebuilt"] == 3, third
    assert third["cached"] == 1, third
    after = {
        path.relative_to(out).as_posix(): path.read_bytes() for path in sorted(out.rglob("*.json"))
    }
    assert after["weekly/2026-W38.json"] == snapshot["weekly/2026-W38.json"]
    assert after["weekly/2026-W37.json"] != snapshot["weekly/2026-W37.json"]

    pack = json.loads((out / "weekly" / "2026-W37.json").read_text(encoding="utf-8"))
    assert pack["period_id"] == "2026-W37"
    assert list(pack["sessions_covered"]) == WEEK_37
    assert pack["inputs_hash"], "a cache key that is empty caches nothing"


def test_the_measured_report_precedes_the_day_facts_stage_tail():
    """Decision 0018: a later phase APPENDS inside its stage and never reorders.

    `market_story_rollups` remains after `daily_digest` and `theta_pick_grading`; WS-RP
    appends `measured_report` immediately after it; Day Review facts then close
    the deterministic stage before `ai_summary`.
    Both are deterministic and retain `journal_import`'s attempt budget rather than the
    briefs'.
    """
    from ai_jobs import runner

    names = tuple(slot.name for slot in runner.default_slots())
    assert "market_story_rollups" in names, names
    # S12 (2026-09-26): the SP4 family evidence closes stage 1 after the facts.
    assert names[names.index("ai_summary") - 1] == "family_side_evidence", names
    assert names[names.index("family_side_evidence") - 1] == "day_review_facts", names
    assert names[names.index("day_review_facts") - 1] == "measured_report", names
    assert names[names.index("measured_report") - 1] == "market_story_rollups", names
    assert names.index("market_story_rollups") > names.index("daily_digest")
    assert names.index("market_story_rollups") > names.index("theta_pick_grading")

    slot = {item.name: item for item in runner.default_slots()}["market_story_rollups"]
    assert slot.enabled
    assert slot.max_attempts == 3
    assert slot.reserve_minutes == 5.0


# ===========================================================================
# Item 3 - the Story pane and the Active theses list, rendered offscreen
# ===========================================================================
PANEL_NOTE = "SPY reclaimed the anchor and I finally sized up."


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6", reason="the Market Journal page is Qt")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _panel_story():
    """A story payload with one trader note, one measured cell and one unmeasured."""
    import market_story

    return market_story.build_daily_story(
        SESSION,
        entries=(_entry(PANEL_NOTE),),
        index_bars={"SPY": _flat_bars()},
    )


def _panel_theses():
    """Two drafts: one with a stated invalidation, one without."""
    import market_thesis

    stated = market_thesis.extract_thesis(_entry(CONDITIONED))
    bare = market_thesis.extract_thesis(_entry(BARE))
    return [market_thesis.draft_row(stated), market_thesis.draft_row(bare)]


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    import project_paths

    monkeypatch.setattr(project_paths, "RUNTIME_DATA_DIR", tmp_path, raising=False)

    from PySide6.QtCore import QObject, Signal

    from ui.panels.market_journal_panel import MarketJournalPanel

    story = _panel_story()
    theses = _panel_theses()

    class _Stub(QObject):
        statusChanged = Signal(str)
        entryWritten = Signal(dict)
        chartCaptured = Signal(dict)

        def __init__(self):
            super().__init__()
            self.saved: list[dict] = []

        def entries_for(self, session_date: str = ""):
            return [dict(row) for row in story.trader_said]

        def sessions_with_entries(self):
            return [SESSION]

        def regime_timeline(self, **_kwargs):
            return {"shifts": [], "agreement": {"rate": None, "note": "none"}}

        def day_context(self, _session):
            return {"measured": False, "reason": "not measured"}

        def chart_digests(self):
            return {}

        def chart_capture(self, entry_id: str):
            return None

        def daily_story(self, session_date: str, **_kwargs):
            return story

        def theses_for(self, session_date: str = ""):
            return [dict(row) for row in theses]

        def save_interpretation(self, *, entry_id, supersedes, text):
            self.saved.append(
                {"entry_id": entry_id, "supersedes": supersedes, "text": text}
            )
            return {"ok": True}

    widget = MarketJournalPanel(service=_Stub())
    widget._render(
        {
            "session_date": SESSION,
            "entries": [dict(row) for row in story.trader_said],
            "sessions": [SESSION],
            "timeline": {"shifts": [], "agreement": {"rate": None, "note": "none"}},
            "context": {"measured": False, "reason": "not measured"},
            "digests": {},
            "story": story,
            "theses": [dict(row) for row in theses],
        }
    )
    yield widget
    widget.shutdown()
    widget.deleteLater()


@pytest.mark.qt
def test_the_story_pane_shows_you_said_then_the_market_did_then_the_sources(panel):
    """The three kinds, visibly distinct, in reading order.

    The trader's words sit under "You said" and BEFORE "The market did"; the measured
    part sits between the second and third headings; the sources name the entry the
    story was built from. An unmeasured benchmark is named as unmeasured rather than
    omitted, because a missing row reads as "nothing happened".
    """
    from ui.panels import market_journal_panel as page

    assert page.STORY_TRADER_HEADING == "You said"
    assert page.STORY_MEASURED_HEADING == "The market did"
    assert page.STORY_SOURCES_HEADING == "Sources"

    body = panel.story_view.toPlainText()
    trader = body.index(page.STORY_TRADER_HEADING)
    measured = body.index(page.STORY_MEASURED_HEADING)
    sources = body.index(page.STORY_SOURCES_HEADING)
    assert trader < measured < sources, body[:400]

    assert trader < body.index(PANEL_NOTE) < measured, "the words go under 'You said'"
    assert measured < body.index("SPY", measured)  # lead fix 2026-09-13: the note itself names SPY above the measured heading < sources
    assert measured < body.index("USO") < sources, "an unmeasured benchmark is still named"

    entry_id = panel.service.entries_for()[0]["entry_id"]
    assert entry_id in body[sources:], body[sources:]


@pytest.mark.qt
def test_the_active_theses_list_shows_an_unstated_invalidation_as_unstated(panel):
    """Claim, horizon, stance, condition, invalidation - and `unstated` when it was not said."""
    import market_thesis

    assert panel.theses_list.count() == 2
    labels = [panel.theses_list.item(i).text() for i in range(panel.theses_list.count())]
    stated = next(label for label in labels if "5,400" in label)
    bare = next(label for label in labels if "5,400" not in label)

    assert "5,400" in stated
    assert market_thesis.UNSTATED in bare, bare


@pytest.mark.qt
def test_a_grounded_question_appears_only_for_a_stated_condition(panel):
    """Selecting the conditioned thesis asks about the condition, in the trader's words."""
    labels = [panel.theses_list.item(i).text() for i in range(panel.theses_list.count())]
    row = next(i for i, label in enumerate(labels) if "5,400" in label)
    panel.theses_list.setCurrentRow(row)

    asked = panel.thesis_questions.text()
    assert "VIX" in asked, asked
    assert asked.count("?") in (1, 2), asked


@pytest.mark.qt
def test_saving_an_interpretation_writes_a_superseding_row_and_never_the_original(panel):
    """The editable box writes a NEW row naming the draft. The draft is not mutated."""
    labels = [panel.theses_list.item(i).text() for i in range(panel.theses_list.count())]
    row = next(i for i, label in enumerate(labels) if "5,400" in label)
    panel.theses_list.setCurrentRow(row)

    before = panel.service.theses_for(SESSION)
    draft = next(item for item in before if "5,400" in json.dumps(item, default=str))

    panel.interpretation_box.setPlainText("Closing basis only, and only while VIX is calm.")
    panel.save_interpretation_button.click()

    assert len(panel.service.saved) == 1, panel.service.saved
    saved = panel.service.saved[0]
    assert saved["entry_id"] == draft["entry_id"]
    assert saved["supersedes"] == draft["thesis_id"]
    assert saved["text"] == "Closing basis only, and only while VIX is calm."
    assert panel.service.theses_for(SESSION) == before, "the draft row is untouched"
