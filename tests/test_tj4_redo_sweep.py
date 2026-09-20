r"""TJ-4 review round 1 - the night sweeps the markers, and bounds the reply.

Blocker 1 (reviewer, 2026-09-20): a daytime **Redo story** for any session but
the one the night narrates was dropped SILENTLY while the page said "Queued for
tonight". The only reader of `redo_requested.json` was the slot, for the session
IT was called with, and the page's default pick during a session day is the
PREVIOUS session - so the DEFAULT click was the broken one and the marker was
never cleared.

Lead decision, built here: **the night sweeps.** After narrating its own
session, an unattended run scans `sessions/*/redo_requested.json`, OLDEST first,
re-narrates at most `REDO_SWEEP_LIMIT` of them (the marker overrides the
unchanged-hash skip), clears each marker only after a GOOD run, and SAYS what it
narrated, what is still queued and how many wait for tomorrow.

Also here, from the same review: the reply's own bounds (`maxItems` and array
item lengths, which the shared validator does not enforce), an empty headline,
and duplicate `source_id`s.

**NO MODEL IS EVER CALLED.** Every path hands the module a fake `request`, and
the paths that must call nothing are handed one that raises.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj4_support as fx  # noqa: E402

SESSION = fx.SESSION            # 2026-09-18, the night's own session
YESTERDAY = "2026-09-17"
TWO_BACK = "2026-09-16"
THREE_BACK = "2026-09-15"
FOUR_BACK = "2026-09-14"


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------
def _pack_for(day: str, *, root: Path, text: str = "A quiet session."):
    """One small REAL pack for `day`, written under `root`."""
    import day_review_pack

    entry = fx.observation_only_entry(text=f"{text} ({day})")
    entry["session_date"] = day
    pack = day_review_pack.build_pack(
        day, entries=[entry], now=fx.AFTER_THE_CLOSE
    )
    day_review_pack.write_pack(pack, root=root)
    return pack


def _reply_for(pack, *, headline: str):
    """A minimal well-formed narration of `pack`. It grades nothing."""
    import day_review_pack

    allowed = list(day_review_pack.allowed_source_ids(pack))
    assert allowed, "fixture drift: a pack with no citable id"
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


def _already_narrated(day: str, *, root: Path, pack, headline: str = "the OLD story"):
    """A verified story for `day` whose hash MATCHES its pack.

    This is the state the reviewer reproduced: the pack has not moved, so
    without the marker the night would skip it.
    """
    from ai_jobs import day_review_narration as nar

    path = nar.narration_path(day, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema": nar.SCHEMA,
            "session_date": day,
            "inputs_hash": pack["inputs_hash"],
            "prompt_version": nar.PROMPT_VERSION,
            "model": "local-test-medium",
            "narration": {"headline": headline},
        }),
        encoding="utf-8",
    )
    return path


def _spy(root: Path, *, headline: str = "a NEW story"):
    """A fake `request` that answers whatever pack it is shown, and records."""
    import day_review_pack
    from ai_jobs import day_review_narration as nar

    seen: list[str] = []

    def request(**kwargs):
        evidence = kwargs.get("evidence") or {}
        day = str(evidence.get("session_date") or "")
        seen.append(f"{kwargs.get('prompt_version')}:{day}")
        if kwargs.get("prompt_version") == nar.D1_VIEW_PROMPT_VERSION:
            ids = list(evidence.get("allowed_source_ids") or ())
            return {
                "model": "local-test-medium",
                "summary": {
                    "belief_now": "You lean long.",
                    "open_theses": [],
                    "sources": [ids[0]] if ids else [],
                },
            }
        pack = day_review_pack.read_pack(day, root=root)
        return _reply_for(pack, headline=f"{headline} for {day}")

    return request, seen


def _never_called(**_kwargs):  # pragma: no cover - must never run
    raise AssertionError("no model may be called on this path")


@pytest.fixture
def root(tmp_path, monkeypatch):
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    base = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", base, raising=False)
    return base


def _run(root, request, *, session: str = SESSION, **kwargs):
    from ai_jobs.day_review_narration import run_day_review_narration

    return run_day_review_narration(
        session_date=session, now=fx.OVERNIGHT, root=root, request=request, **kwargs
    )


def _headline(day: str, *, root: Path) -> str:
    from ai_jobs.day_review_narration import read_narration

    stored = read_narration(day, root=root) or {}
    return str((stored.get("narration") or {}).get("headline") or "")


# ---------------------------------------------------------------------------
# BLOCKER 1 - the reviewer's own reproduction, first
# ---------------------------------------------------------------------------


def test_a_redo_queued_for_another_day_is_narrated_by_tonights_run(root):
    """The reviewer's reproduction, verbatim: the marker is for 2026-09-16 and
    the night narrates 2026-09-18.

    Before the fix the night made ZERO calls, left the old story in place and
    left the marker on disk for ever - while the page had told the trader it was
    queued for tonight.
    """
    import day_review_pack

    tonight = _pack_for(SESSION, root=root)
    older = _pack_for(TWO_BACK, root=root)
    _already_narrated(SESSION, root=root, pack=tonight)
    _already_narrated(TWO_BACK, root=root, pack=older)
    day_review_pack.request_redo(TWO_BACK, root=root)     # the page's daytime click

    request, seen = _spy(root)
    outcome = _run(root, request)

    assert outcome["status"] == "ok", outcome
    assert _headline(TWO_BACK, root=root) == "a NEW story for 2026-09-16"
    assert day_review_pack.redo_requested(TWO_BACK, root=root) is False
    assert TWO_BACK in outcome["reason"], outcome["reason"]
    # The night's own session had not moved, so it was skipped as unchanged -
    # the sweep is the only thing that called a model.
    assert seen == [f"day_review_narration_v1:{TWO_BACK}"], seen


def test_the_sweep_takes_the_oldest_first_and_says_how_many_are_left(root):
    """`REDO_SWEEP_LIMIT` is a SIZE rule on a ten-minute reserve, and what it
    leaves behind stays queued and is SAID rather than dropped."""
    import day_review_pack
    from ai_jobs.day_review_narration import REDO_SWEEP_LIMIT

    queued = [FOUR_BACK, THREE_BACK, TWO_BACK, YESTERDAY]
    assert len(queued) > REDO_SWEEP_LIMIT, "fixture drift: nothing would be left over"
    _pack_for(SESSION, root=root)
    for day in queued:
        pack = _pack_for(day, root=root)
        _already_narrated(day, root=root, pack=pack)
        day_review_pack.request_redo(day, root=root)

    request, seen = _spy(root)
    outcome = _run(root, request)

    narrated = [
        name.split(":", 1)[1]
        for name in seen
        if name.startswith("day_review") and not name.endswith(SESSION)
    ]
    assert narrated == queued[:REDO_SWEEP_LIMIT], narrated
    for day in queued[:REDO_SWEEP_LIMIT]:
        assert day_review_pack.redo_requested(day, root=root) is False, day
    for day in queued[REDO_SWEEP_LIMIT:]:
        assert day_review_pack.redo_requested(day, root=root) is True, day
    assert "1 more queued session" in outcome["reason"], outcome["reason"]


def test_every_session_the_night_narrated_is_named_in_its_reason(root):
    import day_review_pack

    _pack_for(SESSION, root=root)
    for day in (TWO_BACK, YESTERDAY):
        pack = _pack_for(day, root=root)
        _already_narrated(day, root=root, pack=pack)
        day_review_pack.request_redo(day, root=root)

    request, _seen = _spy(root)
    outcome = _run(root, request)

    for day in (TWO_BACK, YESTERDAY):
        assert day in outcome["reason"], (day, outcome["reason"])


def test_a_queued_session_with_no_pack_keeps_its_marker_and_is_named(root):
    """The post-close tick may simply not have reached it yet. That is not a
    failed night and it is never a lost request."""
    import day_review_pack

    _pack_for(SESSION, root=root)
    # A marker with no pack beside it: the folder exists, `pack.json` does not.
    day_review_pack.request_redo(TWO_BACK, root=root)
    assert day_review_pack.read_pack(TWO_BACK, root=root) is None

    request, seen = _spy(root)
    outcome = _run(root, request)

    assert outcome["status"] != "failed"
    assert day_review_pack.redo_requested(TWO_BACK, root=root) is True
    assert "no pack yet" in outcome["reason"], outcome["reason"]
    assert not [name for name in seen if name.endswith(TWO_BACK)], seen


def test_a_rejected_redo_keeps_the_marker_and_the_prior_story(root):
    """A rejected redo is the same rejection as any other: the prior file is
    byte-identical. The marker STAYS, so the next night tries again."""
    import day_review_pack

    tonight = _pack_for(SESSION, root=root)
    older = _pack_for(TWO_BACK, root=root)
    _already_narrated(SESSION, root=root, pack=tonight)
    prior = _already_narrated(TWO_BACK, root=root, pack=older)
    before = prior.read_bytes()
    day_review_pack.request_redo(TWO_BACK, root=root)

    def request(**kwargs):
        reply = _reply_for(
            day_review_pack.read_pack(TWO_BACK, root=root), headline="nope"
        )
        # An id no pack carries: the grounding rejection.
        reply["summary"]["sources"] = ["journal:invented"]
        return reply

    outcome = _run(root, request)

    assert prior.read_bytes() == before, "the prior story was overwritten"
    assert day_review_pack.redo_requested(TWO_BACK, root=root) is True
    assert "still queued" in outcome["reason"], outcome["reason"]


def test_one_bad_queued_session_never_costs_the_nights_own_story(root):
    """The trader opens THIS morning's story. A queued day from last week that
    the model fumbled may not take it away."""
    import day_review_pack

    _pack_for(SESSION, root=root)                     # tonight: not yet narrated
    older = _pack_for(TWO_BACK, root=root)
    _already_narrated(TWO_BACK, root=root, pack=older)
    day_review_pack.request_redo(TWO_BACK, root=root)

    def request(**kwargs):
        evidence = kwargs.get("evidence") or {}
        day = str(evidence.get("session_date") or "")
        pack = day_review_pack.read_pack(day, root=root)
        reply = _reply_for(pack, headline=f"story for {day}")
        if day == TWO_BACK:
            reply["summary"]["sources"] = ["journal:invented"]
        return reply

    outcome = _run(root, request)

    assert _headline(SESSION, root=root) == f"story for {SESSION}"
    assert outcome["status"] == "ok", outcome
    assert day_review_pack.redo_requested(TWO_BACK, root=root) is True


def test_the_rolling_view_is_built_once_however_many_days_were_swept(root):
    """The D1 view is a view of the WINDOW, not of a session. One per run."""
    import day_review_pack
    from ai_jobs.day_review_narration import D1_VIEW_PROMPT_VERSION

    _pack_for(SESSION, root=root)
    # A D1 row so the rolling view has something to be built from at all.
    d1 = fx.d1_note_entry(SESSION, text="I still think the index grinds higher.")
    pack = day_review_pack.build_pack(SESSION, entries=[d1], now=fx.AFTER_THE_CLOSE)
    day_review_pack.write_pack(pack, root=root)
    for day in (TWO_BACK, YESTERDAY):
        older = _pack_for(day, root=root)
        _already_narrated(day, root=root, pack=older)
        day_review_pack.request_redo(day, root=root)

    request, seen = _spy(root)
    _run(root, request)

    views = [name for name in seen if name.startswith(D1_VIEW_PROMPT_VERSION)]
    assert len(views) == 1, seen


def test_a_night_with_no_pack_of_its_own_still_drains_the_queue(root):
    """A missing pack tonight is a skipped row for TONIGHT, never a stranded
    queue: the trader's request is not the night's to lose."""
    import day_review_pack

    older = _pack_for(TWO_BACK, root=root)
    _already_narrated(TWO_BACK, root=root, pack=older)
    day_review_pack.request_redo(TWO_BACK, root=root)
    assert day_review_pack.read_pack(SESSION, root=root) is None

    request, _seen = _spy(root)
    outcome = _run(root, request)

    assert outcome["status"] == "skipped", outcome
    assert _headline(TWO_BACK, root=root) == f"a NEW story for {TWO_BACK}"
    assert day_review_pack.redo_requested(TWO_BACK, root=root) is False


def test_an_operator_who_named_one_day_gets_that_day_and_no_sweep(root):
    """`--session` is a targeted redo. It does what it was asked for; the
    unattended night is what sweeps."""
    import day_review_pack

    asked = _pack_for(YESTERDAY, root=root)
    _already_narrated(YESTERDAY, root=root, pack=asked)
    other = _pack_for(TWO_BACK, root=root)
    _already_narrated(TWO_BACK, root=root, pack=other)
    day_review_pack.request_redo(YESTERDAY, root=root)
    day_review_pack.request_redo(TWO_BACK, root=root)

    request, seen = _spy(root)
    _run(root, request, session=YESTERDAY, only_this_session=True)

    assert seen == [f"day_review_narration_v1:{YESTERDAY}"], seen
    assert day_review_pack.redo_requested(YESTERDAY, root=root) is False
    assert day_review_pack.redo_requested(TWO_BACK, root=root) is True


def test_a_night_with_nothing_queued_calls_nothing_extra(root):
    """The sweep is free when the queue is empty: no scan turns into a call."""
    tonight = _pack_for(SESSION, root=root)
    _already_narrated(SESSION, root=root, pack=tonight)

    outcome = _run(root, _never_called)

    assert outcome["status"] == "ok", outcome
    assert "unchanged" in outcome["reason"]


# ---------------------------------------------------------------------------
# A1 - the reply's own bounds
# ---------------------------------------------------------------------------


def _prior(root):
    tonight = _pack_for(SESSION, root=root)
    from ai_jobs.day_review_narration import narration_path

    path = narration_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last night"}\n', encoding="utf-8")
    return tonight, path, path.read_bytes()


def test_a_reply_with_five_thousand_sources_is_rejected_whole(root):
    """Measured by the reviewer: `validate_structured_output` enforces no
    `maxItems`, so a 5,000-source reply was written (372 KB) and then rendered
    line by line on the Qt thread."""
    import day_review_pack
    from ai_jobs.day_review_narration import MAX_SOURCES

    pack, path, before = _prior(root)
    one = day_review_pack.allowed_source_ids(pack)[0]
    reply = _reply_for(pack, headline="too many")
    reply["summary"]["sources"] = [one] * 5000

    outcome = _run(root, lambda **_k: reply)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before
    assert MAX_SOURCES < 5000


def test_a_reply_with_five_hundred_graded_claims_is_rejected_whole(root):
    import day_review_pack
    from ai_jobs.day_review_narration import MAX_GRADED_CLAIMS

    pack, path, before = _prior(root)
    one = day_review_pack.allowed_source_ids(pack)[0]
    reply = _reply_for(pack, headline="too many claims")
    reply["summary"]["were_you_right"] = [
        {"claim": "x", "source_id": one, "verdict": "right", "evidence_id": one}
    ] * 500

    outcome = _run(root, lambda **_k: reply)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before
    assert MAX_GRADED_CLAIMS < 500


def test_a_source_id_longer_than_the_schema_allows_is_rejected_whole(root):
    """The shared validator checks a STRING's length, never an array item's.

    The over-long id is one the pack ACTUALLY CARRIES (reviewer round 2: with
    an invented id this test passed on the old code for the wrong reason - the
    grounding rule rejected it before any length rule could). Here the id is
    allowed, so only its LENGTH can reject the reply.
    """
    import day_review_pack
    from ai_jobs.day_review_narration import narration_path

    long_id = "said:" + "x" * 300
    entry = fx.observation_only_entry(text="One long-winded note.")
    pack = day_review_pack.build_pack(
        SESSION, entries=[entry], now=fx.AFTER_THE_CLOSE
    )
    pack["trader_said"][0]["source_id"] = long_id
    day_review_pack.write_pack(pack, root=root)
    assert long_id in day_review_pack.allowed_source_ids(pack), "the id must be ALLOWED"

    path = narration_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last night"}\n', encoding="utf-8")
    before = path.read_bytes()

    reply = _reply_for(pack, headline="a very long id")
    reply["summary"]["sources"] = [long_id]

    outcome = _run(root, lambda **_k: reply)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


def test_the_rolling_view_bounds_its_lists_too(root):
    """One rule, both artifacts."""
    import day_review_pack
    from ai_jobs.day_review_narration import D1_VIEW_PROMPT_VERSION, d1_view_path

    _pack_for(SESSION, root=root)
    d1 = fx.d1_note_entry(SESSION, text="The index grinds higher.")
    pack = day_review_pack.build_pack(SESSION, entries=[d1], now=fx.AFTER_THE_CLOSE)
    day_review_pack.write_pack(pack, root=root)
    path = d1_view_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last week"}\n', encoding="utf-8")
    before = path.read_bytes()

    def request(**kwargs):
        if kwargs.get("prompt_version") != D1_VIEW_PROMPT_VERSION:
            return _reply_for(
                day_review_pack.read_pack(SESSION, root=root), headline="today"
            )
        ids = list((kwargs.get("evidence") or {}).get("allowed_source_ids") or ())
        return {
            "model": "local-test-medium",
            "summary": {
                "belief_now": "Long.",
                "open_theses": [{
                    "claim": "up", "since": SESSION,
                    "still_true": "unknown", "evidence_id": ids[0],
                }] * 400,
                "sources": [ids[0]],
            },
        }

    _run(root, request)

    assert path.read_bytes() == before, "an unbounded rolling view was written"


# ---------------------------------------------------------------------------
# A2 - a story with no headline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("headline", ["", "   ", "\n\t "])
def test_a_story_with_no_headline_is_rejected_whole(root, headline):
    """The page keeps its own "No story yet" line when there is nothing to put
    in its place, so a headline-less story is read as no story AT ALL over a
    body of text (reviewer, 2026-09-20). An answer that says nothing is not an
    answer.
    """
    pack, path, before = _prior(root)

    outcome = _run(root, lambda **_k: _reply_for(pack, headline=headline))

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# A3 - one id, one row
# ---------------------------------------------------------------------------


def test_the_pack_never_mints_one_id_for_two_rows(root):
    """Two identical read rows, two congruence lines of one kind, the same
    entry twice: a derived id can collide, and the pack's minter is what stops
    two rows sharing one."""
    import day_review_pack

    entry = fx.observation_only_entry(text="Only one sentence today.")
    reads, _grades = fx.graded_reads([entry])
    assert reads, "fixture drift: no read row"
    line = {"kind": "desk_d1_label", "text": "a line", "verdict": "unmeasured"}

    pack = day_review_pack.build_pack(
        SESSION,
        entries=[entry, dict(entry)],          # the same row, appended twice
        reads=list(reads) + [dict(reads[0])],  # the same read, twice
        congruence=[line, dict(line)],         # two lines of one kind
        trades=[{"trade_id": "t-1", "symbol": "A", "realized_pnl": 1.0},
                {"trade_id": "t-1", "symbol": "A", "realized_pnl": 2.0}],
        now=fx.AFTER_THE_CLOSE,
    )

    ids = [
        str(item.get("source_id") or "")
        for name in day_review_pack.LIST_SECTIONS
        for item in pack.get(name) or ()
    ] + [str(row.get("source_id") or "") for row in pack["trades"]["rows"]]
    assert len(ids) == len(set(ids)), sorted(ids)
    assert all(value.strip() for value in ids)
    allowed = day_review_pack.allowed_source_ids(pack)
    assert len(allowed) == len(set(allowed))
    assert set(ids) <= set(allowed)


def test_a_pack_whose_id_names_two_rows_is_refused_by_the_reader(root):
    """The reader's half of the same rule, for a pack an older build wrote: a
    dict keyed on `source_id` silently keeps the LAST row, so a narration could
    quote the second row's verdict while naming the first."""
    import day_review_pack
    from ai_jobs.day_review_narration import narration_path

    entry = fx.observing_and_predicting_entry()
    reads, _grades = fx.graded_reads([entry])
    pack = day_review_pack.build_pack(
        SESSION, entries=[entry], reads=reads, now=fx.AFTER_THE_CLOSE
    )
    # Hand-duplicate an id, the way an older build could have.
    doubled = dict(pack["reads"][0])
    doubled["verdict"] = "wrong"
    pack["reads"] = list(pack["reads"]) + [doubled]
    day_review_pack.write_pack(pack, root=root)

    path = narration_path(SESSION, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"verified":"last night"}\n', encoding="utf-8")
    before = path.read_bytes()

    read = pack["reads"][0]
    said = next(
        item for item in pack["trader_said"] if item["kind"] == "prediction"
    )
    reply = {
        "model": "local-test-medium",
        "summary": {
            "headline": "You were right.",
            "what_happened": "SPY rose.",
            "what_you_thought": "You called it up.",
            "were_you_right": [{
                "claim": "Rest of day: up",
                "source_id": said["source_id"],
                "verdict": "wrong",              # the SECOND row's verdict
                "evidence_id": read["source_id"],
            }],
            "chased_against_news": {"verdict": "unknown", "evidence_id": ""},
            "process": "One call.",
            "sources": [said["source_id"], read["source_id"]],
        },
    }

    outcome = _run(root, lambda **_k: reply)

    assert outcome["status"] == "degraded_no_narrative", outcome
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# A6 / A7 - one rule for the redo's ledger row
# ---------------------------------------------------------------------------


def test_the_window_refusal_row_is_keyed_to_the_session_that_was_asked_for(
    tmp_path, monkeypatch
):
    """A6. The forced DAYTIME redo of an old session is refused by the clock -
    night-only holds - and the row it leaves must name the day the trader asked
    about, not tonight's session (reviewer, 2026-09-20: `runner.py:406`)."""
    import local_writer_lock as lock_mod
    from contextlib import contextmanager

    from ai_jobs import runner, window

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(tmp_path / "ai_store"))
    (tmp_path / "ai_store").mkdir()
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(
        window, "launch_allowed", lambda *_a, **_k: (False, "outside the night window")
    )

    called: list[str] = []
    slot = runner.JobSlot(
        name="day_review_narration",
        run=lambda **kwargs: called.append(kwargs["session_date"]) or {"status": "ok"},
        reserve_minutes=1.0,
        max_attempts=3,
        uses_model=True,
    )
    led = tmp_path / "ledger.jsonl"
    report = runner.run_slots(
        [slot],
        now=fx.OVERNIGHT,
        force=True,
        only="day_review_narration",
        ledger_path=led,
        session_override=TWO_BACK,
    )

    assert called == [], "night-only must still hold"
    row = report.results[0]
    assert row["status"] == "skipped", row
    assert row["session_date"] == TWO_BACK, row


def test_a_redo_of_an_old_day_is_not_masked_by_tonights_own_completion(
    tmp_path, monkeypatch
):
    """A7. The already-done check for this slot under `--session` looks at the
    OVERRIDDEN session's rows. Tonight being covered says nothing about whether
    last Wednesday was ever narrated."""
    from contextlib import contextmanager

    import local_writer_lock as lock_mod
    from ai_jobs import runner, window

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(tmp_path / "ai_store"))
    (tmp_path / "ai_store").mkdir()
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (True, "open"))

    seen: list[str] = []
    slot = runner.JobSlot(
        name="day_review_narration",
        run=lambda **kwargs: seen.append(kwargs["session_date"]) or {"status": "ok"},
        reserve_minutes=1.0,
        max_attempts=3,
        uses_model=True,
    )
    led = tmp_path / "ledger.jsonl"
    # Tonight's normal run completes first.
    runner.run_slots([slot], now=fx.OVERNIGHT, ledger_path=led)
    assert seen == [runner.session_date_for(fx.OVERNIGHT)]

    # Then a redo of an older day, WITHOUT --force.
    runner.run_slots(
        [slot],
        now=fx.OVERNIGHT + timedelta(minutes=30),
        only="day_review_narration",
        ledger_path=led,
        session_override=TWO_BACK,
    )

    assert seen[-1] == TWO_BACK, seen


def test_an_operator_run_of_one_day_tells_the_slot_it_is_only_that_day(
    tmp_path, monkeypatch
):
    """The runner's half of the no-sweep rule, and no other slot ever sees the
    keyword: it is added only for the slot the override names."""
    from contextlib import contextmanager

    import local_writer_lock as lock_mod
    from ai_jobs import runner, window

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(tmp_path / "ai_store"))
    (tmp_path / "ai_store").mkdir()
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (True, "open"))

    seen: list[dict] = []

    def _run_slot(**kwargs):
        seen.append(dict(kwargs))
        return {"status": "ok"}

    slot = runner.JobSlot(
        name="day_review_narration", run=_run_slot, reserve_minutes=1.0, max_attempts=3
    )
    led = tmp_path / "ledger.jsonl"
    runner.run_slots(
        [slot],
        now=fx.OVERNIGHT,
        only="day_review_narration",
        ledger_path=led,
        session_override=TWO_BACK,
    )
    assert seen[0].get("only_this_session") is True

    seen.clear()
    runner.run_slots([slot], now=fx.OVERNIGHT, ledger_path=led, force=True)
    assert "only_this_session" not in seen[0], seen[0]
