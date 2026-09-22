r"""TJ-16 item 4 - the model labels the WORDS, grounded, and never sees a verdict.

`plan.md` §12.4 TJ-16 item 4: *"each `observation` and `because` gets codes from
a closed, versioned vocabulary (`ui/annotations/vocabularies/
observation_tags_v1.json` ...), each code with the exact source span that must
reproduce it (`market_thesis`'s rule) or it is rejected. The codes become
context fields in item 3 on the NEXT run. **The model never sees a verdict while
tagging**, so a tag cannot be derived from the outcome."*

Decision 0021 answer 29: tendencies are found by MATH; the model only tags words
with a closed span-grounded vocabulary.

**No test here ever reaches a model.** Every call goes through an injected fake
`post`, the endpoint setting is a patched string, and the one runner test
injects the fake into the registered slot rather than letting the real one dial
out.

The contract these tests pin (the builder may add keys, never remove one):

    ai_jobs.observation_tags.load_vocabulary(*, directory=None)
        -> {"vocabulary_id", "vocab_version": int, "codes": (str, ...),
            "entries": ({"code", "label", "hint"}, ...)}

    ai_jobs.observation_tags.notes_for(entries)
        -> [{"note_id", "entry_id", "field": "observation" | "because", "text"}]

    ai_jobs.observation_tags.build_evidence(notes, *, vocabulary=None) -> dict
        the payload handed to the model: the NOTES and the VOCABULARY, and no
        verdict, no grade, no bar and no price anywhere in it.

    ai_jobs.observation_tags.run_observation_tags(
        *, session_date="", now=None, root=None, entries=None, post=None,
    ) -> {"status", "model", "reason", "outputs"}

    ai_jobs.observation_tags.read_latest(session_date, *, root=None)

    THE REPLY the model must return, and the only one accepted::

        {"tags": [{"note_id": ..., "code": ..., "span": [start, end],
                   "quote": ...}, ...]}

    `quote` must be `text[start:end]` EXACTLY, `code` must be in the loaded
    vocabulary and `note_id` must be one the payload offered. One bad row
    rejects the WHOLE reply and the last verified file stays byte-identical.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from unittest import mock
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj16_support as fx  # noqa: E402

ET = ZoneInfo("America/New_York")
OVERNIGHT = datetime(2026, 9, 19, 2, 0, tzinfo=ET)

SLOT = "observation_tags"
VOCABULARY_DIR = SCRIPTS_DIR / "ui" / "annotations" / "vocabularies"
VOCABULARY_FAMILY = "observation_tags"
#: The house's permanent-identifier rule (`ui.annotations.vocabulary._CODE_RE`).
CODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

ENDPOINT = "http://127.0.0.1:11434/v1"
MEDIUM_TAG = "gemma3:12b-tbv3ctx-64k"

#: Two notes whose words are the trader's and whose spans are checkable. Neither
#: contains a grade word, so a payload that leaked one is obvious.
OBSERVATION = "SPY is holding the 50 day and breadth is leading on the sectors."
BECAUSE = "The 50 day held on the retest and volatility is coming in."

#: Keys that may never appear anywhere in what the model is handed: the outcome
#: of this call, the outcome of the last one, and any bar or price at all.
FORBIDDEN_KEYS = frozenset(
    {
        "verdict", "verdicts", "grade", "grades", "grade_id", "graded_at",
        "previous_call_verdict", "move", "move_atr", "anchor_price",
        "anchor_at", "final_price", "final_at", "checkpoints", "rate",
        "rate_lb", "right", "wrong", "flat", "pending",
        "open", "high", "low", "close", "volume", "bars", "m5_bars",
        "daily_bars", "tape", "atr", "flat_band_atr",
    }
)


# ---------------------------------------------------------------------------
# the fake endpoint - this file's only "model"
# ---------------------------------------------------------------------------
def _settings(**values):
    import ai_summary
    import project_paths

    def _get(key, default=None):
        return values.get(key, default)

    class _Both:
        def __enter__(self):
            self._a = mock.patch.object(ai_summary, "get_local_setting", _get)
            self._b = mock.patch.object(project_paths, "get_local_setting", _get)
            self._a.start()
            self._b.start()
            return self

        def __exit__(self, *exc):
            self._b.stop()
            self._a.stop()
            return False

    return _Both()


def _live_settings(**extra):
    values = {
        "ai_local_endpoint_url": ENDPOINT,
        "ai_local_model_medium": MEDIUM_TAG,
        "ai_local_model_large": MEDIUM_TAG,
        "ai_local_context_tokens": 65536,
        "ai_offhours_start": "01:00",
        "ai_offhours_end": "09:00",
    }
    values.update(extra)
    return _settings(**values)


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _fake_model(reply, calls):
    """A fake local endpoint that answers `reply` and records what it was sent."""

    def _post(url, **kwargs):
        calls.append(dict(kwargs))
        return _Response(
            {
                "id": "chatcmpl-tags",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(reply)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 900, "completion_tokens": 120,
                          "total_tokens": 1_020},
            }
        )

    return _post


# ---------------------------------------------------------------------------
# fixture entries, notes and replies
# ---------------------------------------------------------------------------
def _entries(session: str = fx.LAST_SESSION, *, with_context: bool = True):
    """Two Mentor rows carrying What I see and Because, as TJ-14A stores them.

    The first row's stored context carries the PREVIOUS call's verdict, which
    is exactly the field a payload built by dumping the entry would leak.
    """
    entry = fx.click_entry(
        session=session, hour=9, direction="up", confidence="high",
        observation=OBSERVATION, because=BECAUSE,
    )
    if with_context:
        entry["mentor"]["context"] = fx.context(
            hour=9, direction="up", confidence="high",
            last_hour_spy="up", d1_environment="trending_up",
            previous_call_verdict="wrong",
        )
    second = fx.click_entry(
        session=session, hour=10, direction="down", confidence="low",
        observation="The sellers keep showing up into the highs.",
        because="No follow through on the last push.",
    )
    return [entry, second]


def _vocabulary():
    from ai_jobs import observation_tags

    return observation_tags.load_vocabulary()


def _note(notes, text: str) -> Mapping[str, Any]:
    found = [note for note in notes if str(note.get("text")) == text]
    assert len(found) == 1, [note.get("text") for note in notes]
    return found[0]


def _good_reply(notes, code: str, *, phrase: str = "breadth"):
    """One tag whose span really does reproduce its quote."""
    note = _note(notes, OBSERVATION)
    start = OBSERVATION.index(phrase)
    return {
        "tags": [
            {
                "note_id": note["note_id"],
                "code": code,
                "span": [start, start + len(phrase)],
                "quote": phrase,
            }
        ]
    }


def _forbidden_keys(payload: Any) -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.append(str(key))
            found.extend(_forbidden_keys(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(_forbidden_keys(item))
    return found


# ---------------------------------------------------------------------------
# the vocabulary is a closed, versioned asset
# ---------------------------------------------------------------------------
def test_the_tag_vocabulary_is_a_closed_versioned_file_with_permanent_codes():
    """`ui/annotations/vocabularies/` is where a versioned picklist lives, and a
    code is a permanent identifier: never renamed, never reused, readable years
    after the rows that carry it were written.

    The VERSION is never a literal in a test (CLAUDE.md): what is pinned is
    that the file's declared version matches its own filename, so a later v2
    ships beside it and rows stamped with this one stay interpretable.
    """
    files = sorted(VOCABULARY_DIR.glob(f"{VOCABULARY_FAMILY}_v*.json"))
    assert files, f"no {VOCABULARY_FAMILY}_v*.json under {VOCABULARY_DIR}"

    loaded = _vocabulary()
    assert loaded["vocabulary_id"] == VOCABULARY_FAMILY
    assert isinstance(loaded["vocab_version"], int)
    named = sorted(
        int(re.fullmatch(rf"{VOCABULARY_FAMILY}_v(\d+)", path.stem).group(1))
        for path in files
    )
    assert loaded["vocab_version"] == named[-1], (loaded["vocab_version"], named)

    codes = list(loaded["codes"])
    assert len(codes) >= 8, codes
    assert len(set(codes)) == len(codes), codes
    for code in codes:
        assert CODE_RE.fullmatch(code), code
    for entry in loaded["entries"]:
        assert str(entry.get("label") or "").strip(), entry


# ---------------------------------------------------------------------------
# what the model is handed
# ---------------------------------------------------------------------------
def test_the_payload_handed_to_the_model_carries_no_verdict_and_no_bar(tmp_path):
    """The assertion is on the EXACT request body the fake endpoint received.

    A tag derived from the outcome is not a label of the words, it is a
    rationalisation of the result - so the grade ledger of this very session
    sits on disk beside the tagger and none of it reaches the prompt. Neither
    does a bar: the model is given the trader's sentences and the vocabulary,
    and nothing that happened afterwards.
    """
    import market_read_grades as grades
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = _entries(session)
    stored = fx.graded_session(
        session,
        [{"hour": hour, "direction": "up", "confidence": "high"}
         for hour in fx.CLICK_HOURS],
        rising=True,
    )
    ledger_root = tmp_path / "day_review"
    fx.store_ledger(ledger_root, stored)
    assert grades.read_grades(session, root=ledger_root), "the grades are on disk"

    notes = observation_tags.notes_for(entries)
    code = _vocabulary()["codes"][0]
    calls: list[dict] = []
    with _live_settings():
        out = observation_tags.run_observation_tags(
            session_date=session,
            now=fx.morning_after(session),
            root=tmp_path / "packs",
            entries=entries,
            post=_fake_model(_good_reply(notes, code), calls),
        )
    assert out["status"] == "ok", out
    assert len(calls) == 1, calls

    body = calls[0]["json"]
    assert body["model"] == MEDIUM_TAG, "item 4: the MEDIUM local model tags the words"
    sent = json.dumps(body, default=str)

    assert OBSERVATION in sent, "the model must see the trader's own sentence"
    assert "previous_call_verdict" not in sent, sent[:2000]
    assert "move_atr" not in sent
    assert "final_price" not in sent
    assert "graded_at" not in sent
    # The grade ids are this session's own verdict rows. None of them, and no
    # price off the tape they were measured on, may be in the prompt. (The WORD
    # "wrong" is deliberately not asserted on: an honest instruction may well
    # tell the model not to judge whether the trader was right or wrong.)
    for row in stored:
        assert str(row["grade_id"]) not in sent, row["grade_id"]
    for level in (fx.CLOSE_RISING, fx.BEFORE_RISING):
        assert str(level) not in sent, level


def test_the_evidence_offers_the_closed_vocabulary_and_nothing_about_the_outcome():
    """The same rule stated on the structure rather than the wire.

    Every code the model may use is IN the payload - a closed vocabulary the
    model picks from, never invents - and no key anywhere in it names a
    verdict, a grade, a price or a bar.
    """
    from ai_jobs import observation_tags

    vocabulary = _vocabulary()
    notes = observation_tags.notes_for(_entries())
    evidence = observation_tags.build_evidence(notes, vocabulary=vocabulary)

    offered = json.dumps(evidence, default=str)
    for code in vocabulary["codes"]:
        assert code in offered, code
    assert _forbidden_keys(evidence) == [], _forbidden_keys(evidence)

    texts = {str(note["text"]) for note in notes}
    assert OBSERVATION in texts and BECAUSE in texts, texts


def test_both_of_the_two_texts_are_offered_and_nothing_else_is(tmp_path):
    """TJ-14A: `mentor.observation` and `mentor.prediction.because` are the ONLY
    texts. The tagger labels those two fields, each note naming which it is.
    """
    from ai_jobs import observation_tags

    entries = _entries()
    notes = observation_tags.notes_for(entries)

    assert {str(note["field"]) for note in notes} == {"observation", "because"}
    assert len(notes) == 4, notes
    assert {str(note["entry_id"]) for note in notes} == {
        str(entry["entry_id"]) for entry in entries
    }
    assert len({str(note["note_id"]) for note in notes}) == 4, notes


# ---------------------------------------------------------------------------
# the grounding rule - a span that does not reproduce its quote
# ---------------------------------------------------------------------------
def test_a_span_that_does_not_reproduce_its_quote_rejects_the_whole_reply(tmp_path):
    """`market_thesis`' rule: a span is a QUOTATION, and one that does not
    reproduce the text is not evidence of anything.

    The reply here has one good tag and one whose span is off by two
    characters. The WHOLE reply is rejected - not the bad row - and the last
    verified file stays byte-identical, because a half-accepted answer is a
    file nobody can trust and nobody can tell apart from a whole one.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = _entries(session)
    notes = observation_tags.notes_for(entries)
    code = _vocabulary()["codes"][0]
    root = tmp_path / "packs"

    calls: list[dict] = []
    with _live_settings():
        first = observation_tags.run_observation_tags(
            session_date=session, now=fx.morning_after(session), root=root,
            entries=entries, post=_fake_model(_good_reply(notes, code), calls),
        )
    assert first["status"] == "ok", first
    written = sorted(root.glob("*.json"))
    assert written, "the first verified answer is on disk"
    before = {path: path.read_bytes() for path in written}

    note = _note(notes, OBSERVATION)
    start = OBSERVATION.index("breadth")
    bad = {
        "tags": [
            dict(_good_reply(notes, code)["tags"][0]),
            {
                "note_id": note["note_id"],
                "code": code,
                # Two characters to the left: the slice reads "eadth", not what
                # the model said it quoted.
                "span": [start + 2, start + 2 + len("breadth")],
                "quote": "breadth",
            },
        ]
    }
    assert OBSERVATION[start + 2 : start + 2 + len("breadth")] != "breadth"

    with _live_settings():
        second = observation_tags.run_observation_tags(
            session_date=session, now=fx.morning_after(session), root=root,
            entries=entries, post=_fake_model(bad, []),
        )

    assert second["status"] != "ok", second
    assert str(second.get("reason") or "").strip(), second
    assert sorted(root.glob("*.json")) == written, "a rejected reply wrote a file"
    for path, payload in before.items():
        assert path.read_bytes() == payload, path


def test_a_code_outside_the_vocabulary_rejects_the_whole_reply(tmp_path):
    """Closed vocabulary: the model picks from the list or it is not believed.

    An unknown code is the same failure as an unknown `source_id` in every
    other grounded slot - the reply is rejected whole and the prior file stays.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = _entries(session)
    notes = observation_tags.notes_for(entries)
    vocabulary = _vocabulary()
    root = tmp_path / "packs"

    invented = "cites_the_lunar_cycle"
    assert invented not in vocabulary["codes"]
    reply = _good_reply(notes, vocabulary["codes"][0])
    reply["tags"].append(
        {**dict(reply["tags"][0]), "code": invented}
    )

    with _live_settings():
        out = observation_tags.run_observation_tags(
            session_date=session, now=fx.morning_after(session), root=root,
            entries=entries, post=_fake_model(reply, []),
        )

    assert out["status"] != "ok", out
    assert list(root.glob("*.json")) == [], "nothing was published"
    assert observation_tags.read_latest(session, root=root) is None


def test_a_verified_reply_stores_every_tag_with_its_exact_span_and_its_version(tmp_path):
    """What a believed answer looks like on disk.

    Each tag keeps the span AND the quote it was accepted for, so a reader
    years later can re-check the grounding against the note itself, and the
    file is stamped with the vocabulary version that produced it.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = _entries(session)
    notes = observation_tags.notes_for(entries)
    vocabulary = _vocabulary()
    root = tmp_path / "packs"

    with _live_settings():
        out = observation_tags.run_observation_tags(
            session_date=session, now=fx.morning_after(session), root=root,
            entries=entries,
            post=_fake_model(_good_reply(notes, vocabulary["codes"][0]), []),
        )
    assert out["status"] == "ok", out

    stored = observation_tags.read_latest(session, root=root)
    assert stored is not None
    assert stored["session_date"] == session
    assert stored["vocab_version"] == vocabulary["vocab_version"]
    assert str(stored.get("model") or "") == MEDIUM_TAG

    tags = stored["tags"]
    assert len(tags) == 1, tags
    tag = tags[0]
    assert tag["code"] == vocabulary["codes"][0]
    start, end = tag["span"]
    assert OBSERVATION[start:end] == tag["quote"] == "breadth"
    assert tag["entry_id"] == entries[0]["entry_id"]


def test_the_tagger_is_never_asked_anything_when_no_local_model_is_configured(tmp_path):
    """An unconfigured desk dials nothing and keeps the last verified file."""
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    root = tmp_path / "packs"
    calls: list[dict] = []

    with _settings():  # no endpoint, no model
        out = observation_tags.run_observation_tags(
            session_date=session, now=fx.morning_after(session), root=root,
            entries=_entries(session), post=_fake_model({"tags": []}, calls),
        )

    assert out["status"] != "ok", out
    assert calls == [], "an unconfigured desk asked a model anyway"
    assert list(root.glob("*.json")) == []


# ---------------------------------------------------------------------------
# the slot, its stage, and the run it may not join
# ---------------------------------------------------------------------------
def test_the_tagger_is_a_stage_two_model_slot_that_runs_before_the_briefs():
    """plan.md TJ-13 item 9: Stage 2 holds `observation_tags`, and item 4 puts
    it before `ticker_briefs` - the two-hour slot it must not queue behind.

    `uses_model` is declared honestly (it loads a local model), `max_attempts`
    is set and never 0, and the pair WS-10D and WS-RP pin stays untouched.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    slot = {item.name: item for item in runner.default_slots()}[SLOT]

    assert slot.enabled is True
    assert slot.uses_model is True, "it loads a local model; --force may not buy the day"
    assert isinstance(slot.max_attempts, int) and slot.max_attempts > 0
    assert slot.reserve_minutes > 0
    assert slot.description.strip()

    here = names.index(SLOT)
    assert here > names.index("measured_report"), "stage 1 finishes first"
    assert here < names.index("ticker_briefs"), names
    assert names[names.index("ai_summary") - 1] == "day_review_facts", names
    assert names[names.index("day_review_facts") - 1] == "measured_report", names


def test_the_tagger_runs_on_a_weeknight_and_not_on_sundays_deterministic_slate(tmp_path):
    """Item 4: *"local medium, weeknights, seconds per note"*. Sunday's slate is
    the deterministic stage plus the weekend's backlog, so a stage 2 slot that
    nobody attempted is not offered there.
    """
    from ai_jobs import runner

    led = tmp_path / "never_written.jsonl"
    assert SLOT in [slot.name for slot in runner.slots_for("weeknight")]
    sunday = [
        slot.name
        for slot in runner.slots_for(
            "sunday", session_date=fx.LAST_SESSION, ledger_path=led
        )
    ]
    assert SLOT not in sunday, sunday


def test_a_forced_daytime_run_of_the_tagger_is_skipped_and_never_ok(tmp_path, monkeypatch):
    """TJ-13A item 1: `--force` may not buy the clock for a slot that starts
    local inference. *"I always want the bot to run overnight never during the
    day"* - a 14 GB model load in front of the trader's market prep is the thing
    that rule is about.
    """
    import local_writer_lock as lock_mod
    from ai_jobs import runner, window

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(tmp_path / "ai_store"))
    (tmp_path / "ai_store").mkdir()
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (False, "window closed"))

    slot = {item.name: item for item in runner.default_slots()}[SLOT]
    report = runner.run_slots(
        [slot], now=OVERNIGHT, force=True, ledger_path=tmp_path / "ledger.jsonl"
    )

    row = report.results[0]
    assert row["status"] == "skipped", row
    assert "window" in str(row["reason"]).lower(), row


def test_tonights_tags_reach_the_contrast_on_the_NEXT_run_and_never_the_same_one(
    tmp_path, monkeypatch
):
    """Item 4: *"the codes become context fields in item 3 on the NEXT run"*.

    Both slots in ONE runner pass, with the fake model injected into the
    registered slot. The contrast is stage 1 and the tagger is stage 2, so
    tonight's pack cannot hold tonight's codes - and the pack proves it by
    holding no `tag:` feature at all. Run the contrast again with the tags file
    on disk and the codes are there.
    """
    import dataclasses

    import project_paths
    import local_writer_lock as lock_mod
    from ai_jobs import observation_tags, prediction_contrast, runner, window

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)
    monkeypatch.setattr(window, "market_session_block", lambda *_a, **_k: "")
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (True, "open"))
    store_root = tmp_path / "ai_store"
    store_root.mkdir()
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(store_root))

    session = fx.LAST_SESSION
    entries = _entries(session)
    rows = fx.graded_session(
        session,
        [{"hour": hour, "direction": "up", "confidence": "high"}
         for hour in fx.CLICK_HOURS],
        rising=True,
    )
    ledger_root = tmp_path / "day_review"
    fx.store_ledger(ledger_root, rows)
    monkeypatch.setattr(project_paths, "DAY_REVIEW_READS_DIR", ledger_root / "reads")

    notes = observation_tags.notes_for(entries)
    code = _vocabulary()["codes"][0]
    registered = {slot.name: slot for slot in runner.default_slots()}
    tagger = dataclasses.replace(
        registered[SLOT],
        run=lambda **kwargs: observation_tags.run_observation_tags(
            entries=entries,
            post=_fake_model(_good_reply(notes, code), []),
            **kwargs,
        ),
    )

    with _live_settings():
        report = runner.run_slots(
            [registered["prediction_contrast"], tagger],
            now=OVERNIGHT,
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert [row["status"] for row in report.results] == ["ok", "ok"], report.results

    tonight = prediction_contrast.read_latest(session)
    assert tonight is not None and tonight["reads"] == 4, tonight
    assert "tag:" not in json.dumps(tonight, default=str), "tonight's tags are tonight's"
    assert observation_tags.read_latest(session) is not None

    tomorrow = prediction_contrast.run_prediction_contrast(
        session_date=session, now=fx.morning_after(session)
    )
    assert tomorrow["status"] == "ok", tomorrow
    again = json.loads(Path(tomorrow["outputs"][0]).read_text(encoding="utf-8"))
    assert f"tag:{code}" in json.dumps(again, default=str), again


def test_a_tagged_note_that_did_not_get_a_code_is_a_zero_and_an_untagged_one_is_absent(
    tmp_path,
):
    """The honest encoding, the same one every other context field gets.

    A note the tagger READ and did not give code X is a measured 0.0 for
    `tag:X`. A note nobody has tagged yet contributes NOTHING - it is not a
    note without the code, it is a note nobody has looked at.
    """
    from ai_jobs import prediction_contrast

    sessions, rows = fx.two_weeks_of_clicks()
    code = _vocabulary()["codes"][0]
    tagged = {}
    for row in rows:
        session = str(row["session"])
        if session in (sessions[0], sessions[7]):
            continue  # nobody has tagged these eight notes
        tagged[str(row["entry_id"])] = [code] if row["verdict"] == "right" else []

    out = prediction_contrast.run_prediction_contrast(
        session_date=fx.LAST_SESSION,
        now=fx.morning_after(fx.LAST_SESSION),
        root=tmp_path / "packs",
        rows=rows,
        tags=tagged,
    )
    assert out["status"] == "ok", out
    pack = json.loads(Path(out["outputs"][0]).read_text(encoding="utf-8"))

    everything = pack["horizons"]["rest_of_day"]["all"]
    feature = [
        item for item in everything["features"] if item["feature"] == f"tag:{code}"
    ]
    assert len(feature) == 1, [item["feature"] for item in everything["features"]]
    assert (feature[0]["n_a"], feature[0]["n_b"]) == (18, 14), feature
