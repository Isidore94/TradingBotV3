r"""TJ-9E - the NIGHT reads the words and drafts three fields, blind to the result.

Lead decisions 4, 5 and 6. The pattern copied is `scripts/ai_jobs/observation_tags.py`
(TJ-16): a closed versioned vocabulary with its own loader, an exact span + quote
per value, `verify_reply` re-checking EVERY bound itself because a JSON schema is
a grammar hint and never a guard, and a rejection that leaves the prior file
byte-identical. The pass-counting half copies `tests/test_tj6_one_ask_a_night.py`:
the REAL `runner.run_slots`, the real ledger, a PER-TEST runner lock key, and no
model anywhere.

**NO MODEL IS EVER CALLED HERE.** Every run is handed this file's own `request=`
or `post=`. **The real `ai_jobs_runner` lock is never touched** - each pass-
counting test points `runner.RUNNER_LOCK_KEY` at a key of its own, so the nightly
task holding the real one can neither block these tests nor be blocked by them.

RED FOR: `scripts/ai_jobs/exit_note_fields.py` and `scripts/exit_reasons.py` do
not exist on this branch (verified 2026-09-21 at `05988440`), so every test here
fails at `ModuleNotFoundError` / `AttributeError` until the builder writes them.
`test_the_slot_order_pins_are_the_ones_this_packet_moves` is the exception and is
a STATED GUARD over the pins as they stand today.

SLOT POSITION - WHAT IS **NOT** PINNED HERE. The packet asks for
`exit_note_fields` "directly after `observation_tags`, before
`week_review_narration`". That position breaks an IMMEDIATE-ADJACENCY pin -
`tests/test_tj5_week_slot_and_slate.py:137` asserts
``names[names.index("observation_tags") + 1] == "week_review_narration"`` - so
the tester refused to choose and asked the lead (handoff QUESTION 1). What is
pinned below is only the part no resolution changes: stage 2, `uses_model`,
after `observation_tags`, before `ticker_briefs`.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as tj5  # noqa: E402
import tj9e_support as fx  # noqa: E402

SLOT = "exit_note_fields"

#: 02:00 ET on Monday 2026-09-14 - inside the live 01:00-09:00 window, on a
#: session day, so the night's session is Friday 2026-09-11.
NIGHT = datetime(2026, 9, 14, 2, 0, tzinfo=tj5.EASTERN)
#: A Monday AFTERNOON. The market is open and the trader is at the desk.
DAYTIME = datetime(2026, 9, 14, 14, 0, tzinfo=tj5.EASTERN)

LIVE_START = "01:00"
LIVE_END = "09:00"

#: Keys that may never appear anywhere in what the model is handed, at any
#: depth. The trade's money, the trade's prices, its fills and anything measured
#: off the tape afterwards. `TJ-16`'s list, plus the exit's own money words.
FORBIDDEN_KEYS = frozenset(
    {
        "net_pnl", "gross_pnl", "pnl", "pnl_usd", "net_pnl_usd", "net_pnl_cad",
        "net_amount", "gross_amount", "commission", "fees", "r_multiple", "r",
        "expected_r", "mfe", "mfe_r", "mae", "average_entry_price",
        "average_exit_price", "entry_price", "exit_price", "price",
        "quantity", "quantity_opened", "quantity_closed", "position_qty",
        "legs", "trade_legs", "raw_executions", "executions", "fills",
        "verdict", "verdicts", "grade", "grades", "graded_at", "outcome",
        "win", "won", "loss", "walkaway", "open", "high", "low", "close",
        "volume", "bars", "m5_bars", "daily_bars", "tape", "atr",
    }
)

#: The EXACT top-level keys the evidence package may carry: the note's words,
#: the symbol, the side and the two closed code lists, plus the housekeeping a
#: grounded slot needs. Asserted as an EQUALITY, so a key nobody thought about
#: fails here rather than travelling to a model.
EXPECTED_EVIDENCE_KEYS = frozenset(
    {"package_id", "evidence_hash", "instructions", "note", "vocabularies"}
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _settings(**values):
    import ai_summary
    import project_paths
    from ai_jobs import store

    def _get(key, default=None):
        return values.get(key, default)

    class _All:
        def __enter__(self):
            self._patches = [
                mock.patch.object(ai_summary, "get_local_setting", _get),
                mock.patch.object(project_paths, "get_local_setting", _get),
                mock.patch.object(store._paths(), "get_local_setting", _get),
            ]
            for patch in self._patches:
                patch.start()
            return self

        def __exit__(self, *exc):
            for patch in reversed(self._patches):
                patch.stop()
            return False

    return _All()


def _live_settings(**extra):
    values = {
        "ai_local_endpoint_url": "http://127.0.0.1:1/v1/chat/completions",
        "ai_local_model_medium": "tj9e-medium",
        "ai_local_model_large": "tj9e-medium",
        "ai_local_context_tokens": 65536,
        "ai_offhours_start": LIVE_START,
        "ai_offhours_end": LIVE_END,
    }
    values.update(extra)
    return _settings(**values)


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


def _why_code() -> str:
    import exit_reasons

    return exit_reasons.codes()[0]


def _felt_codes(count: int = 1) -> list[str]:
    import trader_state_tags

    return list(trader_state_tags.codes()[:count])


def _ledger_rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if row.get("job") == SLOT]


@pytest.fixture
def night(tmp_path, monkeypatch):
    """A scratch night: a pack root, the window, the store check, and NO real lock."""
    import project_paths
    from ai_jobs import runner, window

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "packs"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(window, "market_session_block", lambda now=None: "")
    # NEVER the real `ai_jobs_runner` lock: the nightly task holds that one.
    monkeypatch.setattr(runner, "RUNNER_LOCK_KEY", f"tj9e-test-{uuid.uuid4().hex}")
    return {"root": root, "ledger": tmp_path / "ledger.jsonl"}


def _passes(night, store, answer, *, count: int = 3, request_calls=None) -> list[dict]:
    """`count` consecutive task firings at the SAME moment, through `run_slots`.

    The real runner, the real slot function, the real ledger - only the model is
    this file's own. Three passes, because the scheduled task fires every 30
    minutes and the SECOND one is where a re-ask shows up.
    """
    from ai_jobs import exit_note_fields, runner
    from ai_jobs import store as job_store

    calls: list[dict] = [] if request_calls is None else request_calls
    request = fx.fake_request(answer, calls=calls)
    slate = [
        replace(
            slot,
            run=partial(
                exit_note_fields.run_exit_note_fields,
                request=request,
                store=store,
                root=night["root"],
            ),
        )
        for slot in runner.default_slots()
        if slot.name == SLOT
    ]
    assert len(slate) == 1, f"{SLOT} is not a registered slot"
    with mock.patch.object(job_store, "store_available", return_value=(True, "ready")):
        with _live_settings():
            for _firing in range(count):
                runner.run_slots(slate, now=NIGHT, only=SLOT, ledger_path=night["ledger"])
    return calls


# ---------------------------------------------------------------------------
# the two vocabularies
# ---------------------------------------------------------------------------
def test_the_exit_reason_vocabulary_is_a_closed_versioned_file_with_permanent_codes():
    """A NEW family `exit_reasons_v*.json` beside the others, with its OWN loader.

    The VERSION is never a literal in a test (CLAUDE.md): what is pinned is that
    the file's declared version matches its own FILENAME, so a v2 ships beside
    it and rows stamped with this one stay interpretable. Hand-counted: the
    packet names TWELVE starting codes, so the list is at least that long.
    """
    import exit_reasons
    from ui.annotations.vocabulary import VOCABULARY_DIR

    family = exit_reasons.VOCABULARY_FAMILY
    files = sorted(Path(VOCABULARY_DIR).glob(f"{family}_v*.json"))
    assert files, f"no {family}_v*.json under {VOCABULARY_DIR}"

    book = exit_reasons.load_vocabulary()
    named = sorted(
        int(re.fullmatch(rf"{re.escape(family)}_v(\d+)", path.stem).group(1))
        for path in files
    )
    assert isinstance(book["vocab_version"], int)
    assert book["vocab_version"] == named[-1], (book["vocab_version"], named)

    codes = list(exit_reasons.codes())
    assert len(codes) >= 12, codes
    assert len(set(codes)) == len(codes), codes
    assert "target_hit" in codes and "stop_hit" in codes and "other" in codes
    for entry in book["entries"]:
        assert str(entry.get("label") or "").strip(), entry


def test_the_felt_codes_come_from_tj7s_own_loader_and_follow_it(monkeypatch, tmp_path):
    """ONE vocabulary of feelings on the desk, not two.

    Proved by SWAPPING the owner: `trader_state_tags` is pointed at a directory
    holding a different picklist, and the codes the exit slot offers follow it.
    A slot that had copied the list would keep offering the shipped codes and
    fail here. The cap is read from `trader_state_tags.MAX_STATE_TAGS`, never
    typed, and no version is asserted.
    """
    import trader_state_tags
    from ai_jobs import exit_note_fields

    folder = tmp_path / "vocabularies"
    folder.mkdir()
    (folder / f"{trader_state_tags.VOCABULARY_FAMILY}_v1.json").write_text(
        json.dumps(
            {
                "vocabulary_id": trader_state_tags.VOCABULARY_FAMILY,
                "vocab_version": 1,
                "tags": [
                    {"code": "swapped_alpha", "label": "Swapped alpha"},
                    {"code": "swapped_beta", "label": "Swapped beta"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trader_state_tags, "_vocabulary_dir", lambda directory=None: folder)

    offered = tuple(exit_note_fields.felt_codes())
    assert offered == ("swapped_alpha", "swapped_beta"), offered
    assert exit_note_fields.MAX_FELT == trader_state_tags.MAX_STATE_TAGS


# ---------------------------------------------------------------------------
# what the model is handed - and what it is NOT
# ---------------------------------------------------------------------------
def test_the_request_body_holds_the_words_the_symbol_the_side_and_two_code_lists(tmp_path, night):
    """THE OUTCOME FENCE, asserted on the EXACT body the fake endpoint received.

    The trade really did make money and the store the slot reads from holds all
    of it: entry 137.41, exit 191.83, 73 shares, net 3972.66. A draft derived
    from the result is not a reading of the words, it is a rationalisation - and
    the P&L is one join away from the note.

    Hand-counted: 1 model call, 0 forbidden keys at any depth, 0 of the four
    money strings anywhere in the body.
    """
    from ai_jobs import exit_note_fields

    store, _trade_id = fx.swing_with_money(tmp_path)
    calls: list[dict] = []
    reply = fx.good_reply(_why_code(), _felt_codes(1))

    with _live_settings():
        out = exit_note_fields.run_exit_note_fields(
            session_date=fx.REVIEWED,
            now=NIGHT,
            root=night["root"],
            store=store,
            post=fx.fake_post(reply, calls),
        )
    assert out["status"] == "ok", out
    assert len(calls) == 1, calls

    body = calls[0]["json"]
    # LEAD AMENDMENT 2026-09-21: `json.dumps` escapes non-ASCII by default and
    # the fixture note carries an en dash ON PURPOSE (the multi-byte case), so
    # `EXIT_NOTE in sent` could not be true for any implementation. Serialise
    # the way the words really travel. The shared provider path also embeds
    # the evidence as a JSON STRING inside a chat message, so the note sits one
    # layer down, escaped again: unwrap every message's content before looking.
    layers = [json.dumps(body, default=str, ensure_ascii=False)]
    for message in body.get("messages") or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        layers.append(content)
        try:
            layers.append(json.dumps(json.loads(content), default=str, ensure_ascii=False))
        except ValueError:
            pass
    sent = " ".join(layers)

    # The evidence is embedded in the prompt as JSON text, so the en dash may
    # arrive escaped (\u2013): the words are there either way.
    escaped = json.dumps(fx.EXIT_NOTE)[1:-1]
    assert fx.EXIT_NOTE in sent or escaped in sent, "the model must see the trader's own words"
    assert fx.SWING in sent, "the symbol travels"
    assert "LONG" in sent, "the side travels"
    assert _why_code() in sent and _felt_codes(1)[0] in sent, "both code lists travel"

    assert _forbidden_keys(body) == [], _forbidden_keys(body)
    # LEAD-GRANTED AMENDMENT 2026-09-21 (review 1 blocker 2): this searched the
    # whole serialised body for `str(number)`, and the body carries
    # `evidence_hash` - a sha256 over a RANDOM note id. `MONEY_QUANTITY` was
    # `73`, which a fresh digest contains about half the time, so the packet's
    # headline fence failed 4 runs in 8 with nothing leaking. The money values
    # are now hex-proof and the search is `fx.money_that_leaked`, which strips
    # the long hex ids and matches the quantity on word boundaries. It is
    # STRICTER, not looser: it reports every value it finds instead of stopping
    # at the first.
    assert fx.money_that_leaked(body) == [], fx.money_that_leaked(body)


def test_the_evidence_package_carries_these_keys_and_no_others(tmp_path):
    """The same rule stated on the STRUCTURE, as an equality.

    `build_evidence` is BUILT from the note and the two picklists - not filtered
    down from a trade row - so there is no key to forget to delete. The equality
    is what makes that structural: a key nobody thought about fails here.
    """
    from ai_jobs import exit_note_fields

    store, _trade_id = fx.swing_with_money(tmp_path)
    notes = exit_note_fields.notes_waiting(store, fx.REVIEWED)
    assert len(notes) == 1, notes

    evidence = exit_note_fields.build_evidence(notes[0])
    assert set(evidence) == EXPECTED_EVIDENCE_KEYS, sorted(set(evidence))
    assert _forbidden_keys(evidence) == [], _forbidden_keys(evidence)
    assert set(evidence["note"]) == {"note_id", "text", "symbol", "side", "exit_session"}


# ---------------------------------------------------------------------------
# the grounding check - all of it, or none of it
# ---------------------------------------------------------------------------
def _run_once(store, root, reply, *, calls=None):
    from ai_jobs import exit_note_fields

    seen: list[dict] = [] if calls is None else calls
    with _live_settings():
        return exit_note_fields.run_exit_note_fields(
            session_date=fx.REVIEWED,
            now=NIGHT,
            root=root,
            store=store,
            post=fx.fake_post(reply, seen),
        )


def _stored_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(Path(root).glob("*.json"))}


@pytest.mark.parametrize(
    "spoil, label",
    [
        (lambda r, w, f: r["fields"]["why"].update({"code": "a_code_nobody_shipped"}),
         "an unknown why code"),
        (lambda r, w, f: r["fields"]["felt"][0].update({"code": "a_feeling_nobody_shipped"}),
         "an unknown felt code"),
        (lambda r, w, f: r["fields"]["why"].update({"span": [0, 4]}),
         "a span that does not reproduce its quote"),
        (lambda r, w, f: r["fields"]["why"].update({"confidence": 0.9}),
         "an extra key on a value"),
        (lambda r, w, f: r.update({"notes": "extra"}),
         "an extra key at the top level"),
        (lambda r, w, f: r["fields"].update({"pnl": 3972.66}),
         "an extra key inside fields"),
        (lambda r, w, f: r["fields"].__setitem__("felt", list(r["fields"]["felt"]) * 2),
         "a duplicate felt row"),
        (lambda r, w, f: r["fields"].__setitem__(
            "watching",
            [fx.value(fx.QUOTE_WATCHING), fx.value(fx.QUOTE_WHY), fx.value(fx.QUOTE_FELT),
             fx.value("the retest")],
        ), "four watching quotes, over the bound of three"),
        (lambda r, w, f: r["fields"].__setitem__(
            "felt", [fx.value(fx.QUOTE_FELT, code) for code in f]
        ), "more felt codes than MAX_STATE_TAGS"),
    ],
)
def test_a_broken_reply_is_rejected_whole_and_publishes_nothing(tmp_path, night, spoil, label):
    """One bad value rejects the WHOLE reply, not the row.

    A half-accepted answer is a file nobody can trust and nobody can tell apart
    from a whole one. Hand-counted per case: 1 model call, 0 files published,
    and the previously verified file byte-identical afterwards.

    The `felt` over-count case offers `MAX_STATE_TAGS + 1` codes, read from
    `trader_state_tags` and never typed.
    """
    import trader_state_tags
    from ai_jobs import ledger

    store, _trade_id = fx.swing_with_money(tmp_path)
    why = _why_code()
    too_many = _felt_codes(trader_state_tags.MAX_STATE_TAGS + 1)
    assert len(too_many) > trader_state_tags.MAX_STATE_TAGS, too_many

    # A good night first, so there IS a prior verified file to keep.
    good = _run_once(store, night["root"], fx.good_reply(why, _felt_codes(1)))
    assert good["status"] == ledger.STATUS_OK, good
    before = _stored_bytes(night["root"])
    assert before, "the good night published nothing"

    # A second note on the same trade puts the slot back to work.
    import trade_mentor_trade_check as check

    check.save_exit_note(
        store, _trade_id, fx.EXIT_NOTE, exit_session=fx.REVIEWED,
        now=datetime.fromisoformat("2026-09-14T09:40:00-04:00"),
    )

    reply = fx.good_reply(why, _felt_codes(1))
    spoil(reply, why, too_many)
    calls: list[dict] = []
    out = _run_once(store, night["root"], reply, calls=calls)

    assert len(calls) == 1, calls
    assert out["status"] != ledger.STATUS_OK, (label, out)
    assert _stored_bytes(night["root"]) == before, f"{label}: the prior file moved"
    leftovers = [path.name for path in Path(night["root"]).iterdir() if path.suffix != ".json"]
    assert leftovers == [], f"{label}: a temp file was left behind: {leftovers}"


def test_a_field_the_note_does_not_speak_to_is_absent_and_never_guessed(tmp_path, night):
    """A note that says WHY and nothing else drafts `why` alone.

    Hand-counted: 1 draft, 1 key under `fields`. An absent field is the honest
    answer; a guessed `felt` would be the machine putting a feeling in the
    trader's mouth, and TJ-7's whole rule is that only the trader names one.
    """
    from ai_jobs import exit_note_fields, ledger

    store, _trade_id = fx.swing_with_money(tmp_path)
    reply = {"fields": {"why": fx.value(fx.QUOTE_WHY, _why_code())}}

    out = _run_once(store, night["root"], reply)
    assert out["status"] == ledger.STATUS_OK, out

    stored = exit_note_fields.read_latest(fx.REVIEWED, root=night["root"])
    drafts = stored["drafts"]
    assert len(drafts) == 1, drafts
    assert set(drafts[0]["fields"]) == {"why"}, drafts[0]["fields"]


def test_every_stored_value_carries_a_span_that_reproduces_its_quote(tmp_path, night):
    """Hand-counted: 1 why + 1 felt + 1 watching = 3 grounded values.

    Every span is re-read out of the note the trader actually wrote, and the
    note holds a MULTI-BYTE character (an en dash), so a builder who measured
    the note in bytes lands on the wrong characters and fails here.
    """
    from ai_jobs import exit_note_fields, ledger

    store, _trade_id = fx.swing_with_money(tmp_path)
    out = _run_once(store, night["root"], fx.good_reply(_why_code(), _felt_codes(1)))
    assert out["status"] == ledger.STATUS_OK, out

    stored = exit_note_fields.read_latest(fx.REVIEWED, root=night["root"])
    draft = stored["drafts"][0]
    values = [draft["fields"]["why"], *draft["fields"]["felt"], *draft["fields"]["watching"]]
    assert len(values) == 3, values
    for value in values:
        start, end = value["span"]
        assert fx.EXIT_NOTE[start:end] == value["quote"], value
    assert len(fx.EXIT_NOTE.encode("utf-8")) != len(fx.EXIT_NOTE), (
        "the fixture note lost its multi-byte character"
    )


# ---------------------------------------------------------------------------
# the night - through the REAL runner
# ---------------------------------------------------------------------------
def test_a_night_with_no_note_waiting_never_loads_the_model(night, tmp_path):
    """Hand-counted: 3 firings -> 0 model calls, 3 `skipped` rows, 0 files.

    `RECALLED_RAW` holds ZERO live rows today and `EXIT_NOTE_RAW` does not exist
    yet, so "nothing waiting" is the FIRST state this slot will ever be in, on
    the first night after the merge and on every night the trader writes nothing.
    """
    from ai_jobs import ledger

    store = fx.new_store(tmp_path)
    fx.mark_covered(store, fx.REVIEWED)

    calls = _passes(night, store, fx.good_reply(_why_code(), _felt_codes(1)))
    assert calls == [], f"{len(calls)} model call(s) for a night with nothing to read"

    rows = _ledger_rows(night["ledger"])
    assert len(rows) == 3, [row.get("status") for row in rows]
    assert {row["status"] for row in rows} == {ledger.STATUS_SKIPPED}
    assert list(Path(night["root"]).glob("*.json")) == []


def test_three_passes_ask_the_model_exactly_once_per_note(night, tmp_path):
    """Hand-counted: 2 notes, 3 firings -> exactly 2 model calls, ONE `ok` row.

    The scheduled task fires every 30 minutes for eight hours. A slot whose
    already-done check never bites asks sixteen times, which is the blocker TJ-6
    was written for.
    """
    from ai_jobs import ledger

    store, ordered = fx.notes_waiting_store(tmp_path, 2)
    assert len(ordered) == 2

    calls = _passes(night, store, fx.good_reply(_why_code(), _felt_codes(1)))
    assert len(calls) == 2, f"the night asked the model {len(calls)} times"

    rows = _ledger_rows(night["ledger"])
    assert len(rows) == 1, [(row["status"], row["reason"]) for row in rows]
    assert rows[0]["status"] == ledger.STATUS_OK


def test_at_most_a_nights_worth_is_drafted_oldest_first_and_the_rest_are_said(night, tmp_path):
    """Hand-counted: `EXIT_NOTES_PER_NIGHT + 1` notes waiting -> exactly
    `EXIT_NOTES_PER_NIGHT` model calls, the OLDEST first, and 1 kept for tomorrow.

    The cap is read from the module, never typed. "The rest are SAID" is the
    half that stops a cap being a silent loss: the ledger row names how many are
    still waiting.
    """
    from ai_jobs import exit_note_fields, ledger

    cap = int(exit_note_fields.EXIT_NOTES_PER_NIGHT)
    store, ordered = fx.notes_waiting_store(tmp_path, cap + 1)

    calls = _passes(night, store, fx.good_reply(_why_code(), _felt_codes(1)), count=1)
    assert len(calls) == cap, f"{len(calls)} calls for a cap of {cap}"

    stored = exit_note_fields.read_latest(fx.REVIEWED, root=night["root"])
    drafted = [draft["trade_id"] for draft in stored["drafts"]]
    assert drafted == ordered[:cap], "the night did not take the oldest notes first"

    rows = _ledger_rows(night["ledger"])
    assert rows[0]["status"] == ledger.STATUS_OK
    assert "1" in str(rows[0]["reason"]), rows[0]["reason"]
    assert "wait" in str(rows[0]["reason"]).lower() or "kept" in str(rows[0]["reason"]).lower()


def test_the_window_is_asked_again_before_every_call_after_the_first(night, tmp_path, monkeypatch):
    """A run near the window close stops CLEANLY, mid-queue.

    Hand-counted: 3 notes waiting, the window open for the first launch and
    closed for every later check -> exactly 1 model call and 1 draft stored. The
    two it did not reach are kept, not dropped.
    """
    from ai_jobs import exit_note_fields, ledger, window

    store, ordered = fx.notes_waiting_store(tmp_path, 3)
    answers = iter([(True, "open")])

    def _launch_allowed(*_args, **_kwargs):
        try:
            return next(answers)
        except StopIteration:
            return (False, "the window closed")

    monkeypatch.setattr(window, "launch_allowed", _launch_allowed)

    calls: list[dict] = []
    with _live_settings():
        out = exit_note_fields.run_exit_note_fields(
            session_date=fx.REVIEWED,
            now=NIGHT,
            root=night["root"],
            store=store,
            post=fx.fake_post(fx.good_reply(_why_code(), _felt_codes(1)), calls),
        )

    assert len(calls) == 1, f"{len(calls)} calls after the window closed"
    assert out["status"] == ledger.STATUS_OK, out
    assert "window" in str(out["reason"]).lower(), out["reason"]

    stored = exit_note_fields.read_latest(fx.REVIEWED, root=night["root"])
    assert [draft["trade_id"] for draft in stored["drafts"]] == ordered[:1]


def test_a_forced_daytime_run_is_skipped_and_calls_nothing(night, tmp_path, monkeypatch):
    """TJ-13A item 1: `--force` may not buy the clock for a slot that starts
    local inference. *"I always want the bot to run overnight never during the
    day"*. Hand-counted: 1 firing at 14:00 -> 0 model calls, 1 `skipped` row."""
    import local_writer_lock as lock_mod
    from ai_jobs import exit_note_fields, ledger, runner, window

    @contextmanager
    def _open(_key, **_kwargs):
        yield None

    monkeypatch.setattr(lock_mod, "local_writer_lock", _open)
    monkeypatch.setenv("TRADINGBOTV3_AI_STORE_DIR", str(tmp_path / "ai_store"))
    (tmp_path / "ai_store").mkdir()
    monkeypatch.setattr(window, "launch_allowed", lambda *_a, **_k: (False, "window closed"))

    store, _ordered = fx.notes_waiting_store(tmp_path, 1)
    calls: list[dict] = []
    slot = {item.name: item for item in runner.default_slots()}[SLOT]
    slot = replace(
        slot,
        run=partial(
            exit_note_fields.run_exit_note_fields,
            request=fx.fake_request(fx.good_reply(_why_code(), _felt_codes(1)), calls=calls),
            store=store,
            root=night["root"],
        ),
    )
    report = runner.run_slots(
        [slot], now=DAYTIME, force=True, ledger_path=night["ledger"]
    )

    row = report.results[0]
    assert row["status"] == ledger.STATUS_SKIPPED, row
    assert calls == [], "a forced daytime run asked a model anyway"
    assert list(Path(night["root"]).glob("*.json")) == []


def test_an_unconfigured_desk_dials_nothing(night, tmp_path):
    """No endpoint, no model: the slot asks nothing and keeps the last file."""
    from ai_jobs import exit_note_fields, ledger

    store, _ordered = fx.notes_waiting_store(tmp_path, 1)
    calls: list[dict] = []
    with _settings():  # nothing configured at all
        out = exit_note_fields.run_exit_note_fields(
            session_date=fx.REVIEWED,
            now=NIGHT,
            root=night["root"],
            store=store,
            post=fx.fake_post(fx.good_reply(_why_code(), _felt_codes(1)), calls),
        )
    assert out["status"] != ledger.STATUS_OK, out
    assert calls == []
    assert list(Path(night["root"]).glob("*.json")) == []


# ---------------------------------------------------------------------------
# the slot's own declaration
# ---------------------------------------------------------------------------
def test_the_slot_is_a_stage_two_model_slot_between_the_tags_and_the_briefs():
    """What no resolution of the adjacency question changes.

    Stage 1 finishes first, the slot loads a local model so `--force` may not
    buy it the day, it has a real attempt cap, and it sits after
    `observation_tags` and before `ticker_briefs` - the two-hour slot it must
    not queue behind.
    """
    from ai_jobs import runner

    names = [slot.name for slot in runner.default_slots()]
    slot = {item.name: item for item in runner.default_slots()}[SLOT]

    assert slot.enabled is True
    assert slot.uses_model is True
    assert slot.model_free_kwargs is None, "there is no half of this that runs without a model"
    assert isinstance(slot.max_attempts, int) and slot.max_attempts > 0
    assert slot.reserve_minutes > 0
    assert slot.description.strip()

    here = names.index(SLOT)
    assert here > names.index(runner._STAGE_ONE_LAST_SLOT), "stage 1 finishes first"
    assert here > names.index("observation_tags"), names
    assert here < names.index("ticker_briefs"), names
    assert SLOT in [item.name for item in runner.slots_for("weeknight")]


def test_the_slot_reports_statuses_the_ledger_owns():
    """The slot's return values are `ai_jobs.ledger`'s constants, IMPORTED.

    An unrecognised status is filed as `failed` by the runner
    (`scripts/ai_jobs/runner.py:456-470`), so a slot that spells its own is a
    night recorded as a failure. The module is read as SOURCE: a status string
    typed as a literal beside an imported constant is the drift this refuses.
    """
    import ast

    from ai_jobs import ledger

    path = ROOT_DIR / "scripts" / "ai_jobs" / "exit_note_fields.py"
    assert path.is_file(), path
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imports_ledger = any(
        (isinstance(node, ast.ImportFrom) and node.module and "ledger" in node.module)
        or (isinstance(node, ast.ImportFrom) and node.module == "ai_jobs"
            and any(alias.name == "ledger" for alias in node.names))
        or (isinstance(node, ast.Import) and any("ledger" in alias.name for alias in node.names))
        for node in ast.walk(tree)
    )
    assert imports_ledger, "the slot does not import the ledger's status vocabulary"

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    spelled = literals & {
        ledger.STATUS_OK, ledger.STATUS_FAILED, ledger.STATUS_SKIPPED, ledger.STATUS_DEGRADED,
    }
    assert not spelled, f"status string(s) typed as literals: {sorted(spelled)}"


def test_the_slot_order_pins_are_the_ones_this_packet_moves():
    """STATED GUARD over the pins AS THEY STAND TODAY, so the builder can see
    exactly which files a new slot moves and the handoff's list is checkable.

    The adjacency pin quoted here is the one the packet's requested position
    breaks; the tester did not choose a resolution (handoff QUESTION 1).
    """
    week = (ROOT_DIR / "tests" / "test_tj5_week_slot_and_slate.py").read_text(encoding="utf-8")
    assert 'names[names.index("observation_tags") + 1] == SLOT' in week, (
        "the TJ-5 adjacency pin moved; re-read it before editing"
    )
    scopes = (ROOT_DIR / "tests" / "test_opt_in_evidence_scopes.py").read_text(encoding="utf-8")
    # R1 put day_review_show after day_review_narration.
    assert '"day_review_narration", "day_review_show", "observation_tags"' in scopes, (
        "the opt-in `allowed` list moved; re-read it before editing"
    )
