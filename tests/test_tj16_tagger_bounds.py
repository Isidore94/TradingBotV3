r"""TJ-16 review round 1 - the tagger's own bounds, its windowed read, and the
hindsight label. RED before the fix.

Three findings from the reviewer's reproduction (2026-09-20), all in
`scripts/ai_jobs/observation_tags.py`:

**Blocker 1 - a JSON schema is a GRAMMAR HINT, never a guard.** `MAX_TAGS` and
`additionalProperties: false` live in `TAGS_JSON_SCHEMA`, which is sent to the
provider in `response_format`. A local backend that does not compile the grammar
- the exact fallback `ai_summary._request_local_summary` already ships for -
answers whatever it likes, and `validate_structured_output` checks only the TOP
level and only declared types. So a 10,000-row reply was published `ok`, a tag
row carrying an extra key was believed, and two byte-identical rows were both
stored. **A bound the verifier does not re-check is not a bound**, and this file
is what says so.

The lead's rule (2026-09-20): more than `MAX_TAGS` rows, an unknown key at
either level, or a byte-identical duplicate row REJECTS THE WHOLE REPLY. A model
that repeats itself is not verified - simpler and stricter than a silent
de-dupe. Two tags on one note with OVERLAPPING spans and DIFFERENT codes stay
ACCEPTED: one sentence can cite a level and be hedged, and that is the finding,
not an error.

**Blocker 2 - the read was unbounded.** `EvidenceLedger.read` filters by
`session_date` WHILE STREAMING (`start=` / `end=`); the slot read every row ever
written and then filtered in Python. The packet's lead decision 2 says "never
stream the whole journal unbounded". Measured by the reviewer on a copy of the
live stream: 84 rows against 7 for one session, identical answer - a correction
carries the ORIGINAL `session_date`, so both halves of a supersede pair stay
inside a one-session window.

**Advisory 1 - a note written AFTER the session is tagged, unmarked.** The
machine adds no outcome and the trader's own words are the artifact under study,
so such a note is NOT dropped and NOTHING it produces is re-ranked. It is
LABELLED, so a later reader can partition hindsight words from live ones - and
the label never reaches the model, because a payload that said "this was written
afterwards" would be telling the tagger something about the outcome.

No test here reaches a model: every call goes through an injected fake `post`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

import tj16_support as fx  # noqa: E402
from test_tj16_observation_tags import (  # noqa: E402
    BECAUSE,
    OBSERVATION,
    _entries,
    _fake_model,
    _good_reply,
    _live_settings,
    _note,
    _vocabulary,
)


def _fingerprint(root: Path) -> dict[str, str]:
    """Every published file and its sha256 - what "byte-identical" is measured on."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.json"))
    }


def _first_good_run(tmp_path, session=fx.LAST_SESSION):
    """One verified answer on disk, and everything needed to send a second."""
    from ai_jobs import observation_tags

    entries = _entries(session)
    notes = observation_tags.notes_for(entries)
    code = _vocabulary()["codes"][0]
    root = tmp_path / "packs"
    with _live_settings():
        out = observation_tags.run_observation_tags(
            session_date=session,
            now=fx.morning_after(session),
            root=root,
            entries=entries,
            post=_fake_model(_good_reply(notes, code), []),
        )
    assert out["status"] == "ok", out
    before = _fingerprint(root)
    assert before, "the first verified answer is on disk"
    return entries, notes, code, root, before


def _send(entries, reply, root, session=fx.LAST_SESSION):
    from ai_jobs import observation_tags

    with _live_settings():
        return observation_tags.run_observation_tags(
            session_date=session,
            now=fx.morning_after(session),
            root=root,
            entries=entries,
            post=_fake_model(reply, []),
        )


def _assert_rejected(out, root, before):
    assert out["status"] != "ok", out
    assert str(out.get("reason") or "").strip(), out
    assert _fingerprint(root) == before, "a rejected reply changed the published files"


# ---------------------------------------------------------------------------
# blocker 1 - the reply's own bounds, re-checked where they can be enforced
# ---------------------------------------------------------------------------
def test_a_reply_over_the_tag_cap_is_rejected_whole(tmp_path):
    """`MAX_TAGS` is a bound or it is a comment.

    Every row here is individually VALID - a real note, a real code, a span that
    reproduces its quote - so the only thing wrong with the answer is that there
    are more of them than the contract allows. The reviewer reproduced this with
    ten thousand rows published `ok`; the cap is the boundary, so the boundary is
    what is pinned.
    """
    from ai_jobs import observation_tags

    entries, notes, code, root, before = _first_good_run(tmp_path)
    note = _note(notes, OBSERVATION)
    over = observation_tags.MAX_TAGS + 1
    assert over + 1 < len(OBSERVATION), "the fixture note is long enough for distinct spans"
    reply = {
        "tags": [
            {
                "note_id": note["note_id"],
                "code": code,
                "span": [index, index + 2],
                "quote": OBSERVATION[index : index + 2],
            }
            for index in range(over)
        ]
    }
    assert len({tuple(row["span"]) for row in reply["tags"]}) == over, "no duplicates"

    _assert_rejected(_send(entries, reply, root), root, before)


def test_a_tag_row_carrying_a_key_the_contract_forbids_is_rejected_whole(tmp_path):
    """`additionalProperties: false` is enforced by the PROVIDER or by nobody.

    `validate_structured_output` walks the top level only; an array of objects is
    passed through untouched. So an extra key on a tag row reaches the verifier,
    and the verifier is the last place it can be refused.
    """
    entries, notes, code, root, before = _first_good_run(tmp_path)
    reply = _good_reply(notes, code)
    reply["tags"][0]["why"] = "smuggled"

    _assert_rejected(_send(entries, reply, root), root, before)


def test_an_unknown_key_at_the_top_of_the_reply_is_rejected_by_the_verifier(tmp_path):
    """The verifier stands on its own, not on the provider path around it.

    `verify_reply` is a public function and the one place the answer is believed.
    Asked directly - the way a later caller might - it refuses a reply carrying a
    key the contract does not declare.
    """
    import pytest

    from ai_jobs import observation_tags

    entries = _entries()
    notes = observation_tags.notes_for(entries)
    vocabulary = _vocabulary()
    reply = _good_reply(notes, vocabulary["codes"][0])
    reply["notes"] = [{"note_id": "nt-smuggled"}]

    with pytest.raises(observation_tags.ReplyRejected):
        observation_tags.verify_reply(reply, notes, vocabulary)


def test_two_byte_identical_tag_rows_reject_the_whole_reply(tmp_path):
    """A model that repeats itself has not been verified.

    De-duping silently would store a file that does not say what the model
    actually returned, and `codes_by_entry` already collapses codes per entry so
    no number would move - which is precisely why the duplicate would never be
    noticed. Refusing is simpler and stricter (lead decision, 2026-09-20).
    """
    entries, notes, code, root, before = _first_good_run(tmp_path)
    reply = _good_reply(notes, code)
    reply["tags"].append(dict(reply["tags"][0]))
    assert reply["tags"][0] == reply["tags"][1]

    _assert_rejected(_send(entries, reply, root), root, before)


def test_two_codes_overlapping_on_one_sentence_are_still_believed(tmp_path):
    """The guard against over-tightening blocker 1's fix.

    One sentence can cite a level AND be hedged, and the two spans can overlap
    or be identical while the CODES differ. That is a finding about the words,
    not a repetition, and it is accepted and stored.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = _entries(session)
    notes = observation_tags.notes_for(entries)
    vocabulary = _vocabulary()
    first, second = vocabulary["codes"][0], vocabulary["codes"][1]
    note = _note(notes, OBSERVATION)
    start = OBSERVATION.index("holding the 50 day")
    reply = {
        "tags": [
            {
                "note_id": note["note_id"],
                "code": first,
                "span": [start, start + len("holding the 50 day")],
                "quote": "holding the 50 day",
            },
            {
                "note_id": note["note_id"],
                "code": second,
                "span": [start + 8, start + len("holding the 50 day")],
                "quote": OBSERVATION[start + 8 : start + len("holding the 50 day")],
            },
        ]
    }
    root = tmp_path / "packs"

    out = _send(entries, reply, root, session=session)

    assert out["status"] == "ok", out
    stored = observation_tags.read_latest(session, root=root)
    assert stored is not None
    assert {tag["code"] for tag in stored["tags"]} == {first, second}


# ---------------------------------------------------------------------------
# blocker 2 - the nightly read asks for ONE session
# ---------------------------------------------------------------------------
def _spy_ledger(monkeypatch, rows):
    """Replace `EvidenceLedger` with something that records how it was read."""
    import evidence_ledger

    calls: list[dict[str, Any]] = []

    class _Spy:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def read(self, **kwargs):
            calls.append(dict(kwargs))
            return SimpleNamespace(rows=tuple(rows), unreadable=0, files=())

    monkeypatch.setattr(evidence_ledger, "EvidenceLedger", _Spy)
    return calls


def test_the_nightly_read_asks_the_ledger_for_the_one_session(tmp_path, monkeypatch):
    """`EvidenceLedger.read` filters by `session_date` WHILE STREAMING.

    The slot used to read every row ever written and then filter in Python.
    Measured by the reviewer on a copy of the live stream: 84 rows against 7 for
    one session, the same answer either way. The packet's rule is "never stream
    the whole journal unbounded", and a filter after the fact is not that.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = _entries(session)
    calls = _spy_ledger(monkeypatch, entries)

    rows, note = observation_tags._read_entries(session)

    assert not note, note
    assert len(calls) == 1, calls
    window = calls[0]
    assert str(window.get("start") or "")[:10] == session, window
    assert str(window.get("end") or "")[:10] == session, window
    assert {str(row["entry_id"]) for row in rows} == {
        str(entry["entry_id"]) for entry in entries
    }


def test_a_corrected_note_still_resolves_to_its_latest_text_in_that_window(
    tmp_path, monkeypatch
):
    """A correction carries the ORIGINAL `session_date` (`market_journal.
    build_entry`: *"session_date is what the entry is ABOUT"*), so both halves of
    a supersede pair are inside a one-session window and `resolve_entries` can
    still hide the old one. Narrowing the read must not narrow the answer.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    original = fx.click_entry(
        session=session, hour=9, direction="up", confidence="high",
        observation="SPY is holding the 50 day.", because="The retest held.",
    )
    corrected = fx.click_entry(
        session=session, hour=9, direction="up", confidence="high",
        observation="SPY is holding the 20 day, not the 50.",
        because="The retest held.",
    )
    corrected["supersedes"] = original["entry_id"]
    calls = _spy_ledger(monkeypatch, [original, corrected])

    rows, _note = observation_tags._read_entries(session)

    assert str(calls[0].get("start") or "")[:10] == session
    assert [str(row["entry_id"]) for row in rows] == [str(corrected["entry_id"])]
    texts = {str(item["text"]) for item in observation_tags.notes_for(rows)}
    assert "SPY is holding the 20 day, not the 50." in texts
    assert "SPY is holding the 50 day." not in texts


# ---------------------------------------------------------------------------
# advisory 1 - hindsight is LABELLED, never dropped and never re-ranked
# ---------------------------------------------------------------------------
def _hindsight_entry(session: str = fx.LAST_SESSION):
    """One Mentor row written in the evening, after the bell. Real writer."""
    import market_journal

    day = fx.LAST_SESSION if session is None else session
    moment = fx.stamp_at(day, 20)  # 20:00 Pacific - hours after the close
    mentor = {
        "slot_id": "m5-2000",
        "prompt_kind": "m5",
        "scheduled_at": (moment - timedelta(minutes=2)).isoformat(),
        "responded_at": moment.isoformat(),
        "observation": OBSERVATION,
        "prediction": market_journal.build_prediction(
            direction="up", horizon="rest_of_day", confidence="high", because=BECAUSE
        ),
    }
    entry = market_journal.build_entry(
        text=OBSERVATION,
        session_date=day,
        timeframe=market_journal.TIMEFRAME_M5,
        origin=market_journal.ORIGIN_TRADE_MENTOR,
        mentor=mentor,
        now=moment,
    )
    assert entry["written_after_the_session"] is True, entry["written_after_the_session"]
    return entry


def test_a_note_written_after_the_bell_is_labelled_and_never_reaches_the_model(tmp_path):
    """Advisory 1. `prediction.because` CAN carry hindsight - the reviewer wrote
    "In hindsight the 50 day failed and I lost on this call" and watched it reach
    the payload verbatim. The machine adds no outcome and the trader's words are
    the artifact under study, so the note is tagged like any other and NOTHING it
    produces is re-ranked: it is LABELLED, and the label stays out of the prompt.

    A payload that said "this was written after the close" would be telling the
    tagger something about the outcome, which is the one thing this slot may
    never do.
    """
    from ai_jobs import observation_tags

    session = fx.LAST_SESSION
    entries = [_hindsight_entry(session)]
    notes = observation_tags.notes_for(entries)
    assert notes, notes
    for note in notes:
        assert note["written_after_the_session"] is True, note

    evidence = observation_tags.build_evidence(notes, vocabulary=_vocabulary())
    offered = json.dumps(evidence, default=str)
    assert "written_after_the_session" not in offered, offered[:2000]

    code = _vocabulary()["codes"][0]
    root = tmp_path / "packs"
    calls: list[dict] = []
    out = None
    with _live_settings():
        out = observation_tags.run_observation_tags(
            session_date=session,
            now=fx.morning_after(session),
            root=root,
            entries=entries,
            post=_fake_model(_good_reply(notes, code), calls),
        )
    assert out["status"] == "ok", out
    assert "written_after_the_session" not in json.dumps(calls[0]["json"], default=str)

    stored = observation_tags.read_latest(session, root=root)
    assert stored is not None
    assert stored["entries_written_after"] == 1, stored
    assert stored["tags"], stored
    for tag in stored["tags"]:
        assert tag["written_after_the_session"] is True, tag


def test_the_contrast_pack_counts_the_tagged_notes_written_after_the_session(tmp_path):
    """The label has to survive the join, or nobody can partition on it.

    It is a COUNT beside the tag block, never a filter: no feature is dropped, no
    row is re-weighted, and `tags.codes` is unchanged. A reader who wants to know
    how much of the tagged evidence is hindsight can now ask.
    """
    from ai_jobs import prediction_contrast

    _sessions, rows = fx.two_weeks_of_clicks()
    code = _vocabulary()["codes"][0]
    tagged = {str(row["entry_id"]): [code] for row in rows}
    hindsight = sorted(tagged)[:3]

    out = prediction_contrast.run_prediction_contrast(
        session_date=fx.LAST_SESSION,
        now=fx.morning_after(fx.LAST_SESSION),
        root=tmp_path / "packs",
        rows=rows,
        tags=tagged,
        written_after=hindsight,
    )
    assert out["status"] == "ok", out
    pack = json.loads(Path(out["outputs"][0]).read_text(encoding="utf-8"))

    assert pack["tags"]["entries_tagged"] == 40
    assert pack["tags"]["reads_matched"] == 40
    assert pack["tags"]["entries_written_after"] == 3, pack["tags"]

    everything = pack["horizons"]["rest_of_day"]["all"]
    feature = [
        item for item in everything["features"] if item["feature"] == f"tag:{code}"
    ]
    assert len(feature) == 1, [item["feature"] for item in everything["features"]]
    assert (feature[0]["n_a"], feature[0]["n_b"]) == (22, 18), (
        "a hindsight note is LABELLED, never dropped and never re-weighted"
    )


def test_a_pack_with_no_hindsight_says_zero_rather_than_leaving_the_field_out(tmp_path):
    """Present and zero, never absent: a reader that has to tell "none" from
    "this build did not measure it" is reading two different absences as one.
    """
    from ai_jobs import prediction_contrast

    _sessions, rows = fx.two_weeks_of_clicks()
    out = prediction_contrast.run_prediction_contrast(
        session_date=fx.LAST_SESSION,
        now=fx.morning_after(fx.LAST_SESSION),
        root=tmp_path / "packs",
        rows=rows,
    )
    assert out["status"] == "ok", out
    pack: Mapping[str, Any] = json.loads(
        Path(out["outputs"][0]).read_text(encoding="utf-8")
    )
    assert pack["tags"]["entries_written_after"] == 0, pack["tags"]
