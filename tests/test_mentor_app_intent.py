"""P18 review round 5: one table-driven intent resolver. Every sentence from the four review rounds, with the book."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach, intent  # noqa: E402

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
#: The default book: AMD and NVDA held long, TSLA held short. ALL and MSFT are watched, not held.
BOOK = {"AMD": "LONG", "NVDA": "LONG", "TSLA": "SHORT"}
KNOWN = {**BOOK, "ALL": "SHORT", "MSFT": ""}
NOT_HELD: dict[str, str] = {}

EXIT_L = ("exit", "LONG")
EXIT_S = ("exit", "SHORT")
ADD_L = ("add", "LONG")
ADD_S = ("add", "SHORT")
NEW_L = ("new", "LONG")
NEW_S = ("new", "SHORT")
NONE = ("none_to_exit", "")

#: (sentence, book, {ticker: (kind, side)}). Tickers not listed get no gate.
TABLE = [
    # round 2-3: selling or covering a held position is an exit
    ("should I sell my position in AMD", BOOK, {"AMD": EXIT_L}),
    ("AMD looks weak here, should I sell?", BOOK, {"AMD": EXIT_L}),
    ("AMD is extended, thinking about selling", BOOK, {"AMD": EXIT_L}),
    ("should i sell all of my AMD", BOOK, {"AMD": EXIT_L}),
    ("AMD close is weak, sell?", BOOK, {"AMD": EXIT_L}),
    ("TSLA ripping, should I cover?", BOOK, {"TSLA": EXIT_S}),
    ("should I sell AMD here?", BOOK, {"AMD": EXIT_L}),
    ("selling AMD, taking profit", BOOK, {"AMD": EXIT_L}),
    ("about to sell half my NVDA", BOOK, {"NVDA": EXIT_L}),
    ("sell AMD here", BOOK, {"AMD": EXIT_L}),
    ("trim NVDA", BOOK, {"NVDA": EXIT_L}),
    ("take profit on AMD", BOOK, {"AMD": EXIT_L}),
    ("cover TSLA", BOOK, {"TSLA": EXIT_S}),
    ("sell half my NVDA", BOOK, {"NVDA": EXIT_L}),
    ("get out of AMD", BOOK, {"AMD": EXIT_L}),
    ("dump TSLA", BOOK, {"TSLA": EXIT_S}),  # rule: "dump" closes the position held, whichever side
    # "close" is an action only with an object; as a price word it is never an exit
    ("should I close AMD", BOOK, {"AMD": EXIT_L}),
    ("thinking about closing NVDA", BOOK, {"NVDA": EXIT_L}),
    ("close TSLA", BOOK, {"TSLA": EXIT_S}),
    ("close the TSLA short", BOOK, {"TSLA": EXIT_S}),
    ("close out AMD", BOOK, {"AMD": EXIT_L}),
    ("close my NVDA", BOOK, {"NVDA": EXIT_L}),
    ("did AMD close above vwap", BOOK, {}),
    ("what is the AMD close today", BOOK, {}),
    ("where did AMD close yesterday?", BOOK, {}),
    ("AMD closed red", BOOK, {}),
    ("AMD closing strong, add more?", BOOK, {"AMD": ADD_L}),
    # each verb binds to its own ticker
    ("sell AMD and buy NVDA", BOOK, {"AMD": EXIT_L, "NVDA": ADD_L}),
    ("buy NVDA and sell AMD", BOOK, {"NVDA": ADD_L, "AMD": EXIT_L}),
    ("NVDA is weak, sell AMD?", BOOK, {"AMD": EXIT_L}),
    # not held: the verb's own side is a new trade
    ("about to sell AMD", NOT_HELD, {"AMD": NEW_S}),
    ("about to buy NVDA", NOT_HELD, {"NVDA": NEW_L}),
    ("I'm about to buy NVDA", NOT_HELD, {"NVDA": NEW_L}),
    ("I'm thinking of going long AMD here, size 200 stop 151", NOT_HELD, {"AMD": NEW_L}),
    ("I want to short TSLA here", NOT_HELD, {"TSLA": NEW_S}),
    ("going long MSFT here", NOT_HELD, {"MSFT": NEW_L}),
    ("going short MSFT here", NOT_HELD, {"MSFT": NEW_S}),
    ("close TSLA", NOT_HELD, {"TSLA": NONE}),
    ("trim NVDA", NOT_HELD, {"NVDA": NONE}),
    # adds: the held side; an add on a name not held is nothing
    ("adding to AMD", {"AMD": "SHORT"}, {"AMD": ADD_S}),
    ("adding to NVDA", BOOK, {"NVDA": ADD_L}),
    ("add NVDA", NOT_HELD, {}),
    ("NVDA sell-off today", BOOK, {}),
    ("NVDA sell-off today", NOT_HELD, {}),
    # no trade verb: no gate for a held name; the intent words still gate a name not held
    ("entering TSLA", NOT_HELD, {"TSLA": NEW_S}),
    ("thinking of taking a short like ALL, thoughts?", BOOK, {"ALL": NEW_S}),
    # pure questions, no verb
    ("how is AMD trading right now", BOOK, {}),
    ("what's the news on NVDA", BOOK, {}),
    ("is TSLA reporting this week", BOOK, {}),
    ("where is NVDA currently", BOOK, {}),
    ("how much am I up on AMD", BOOK, {}),
    ("what did the night say about TSLA", BOOK, {}),
    ("show me the AMD chart levels", BOOK, {}),
    ("is NVDA above its AVWAP", BOOK, {}),
]


def _gates(text, book):
    requests = attach.plan_attachments(text, {**KNOWN, **book}, NOW, book=list(book))
    out = {}
    for r in requests:
        if r.name == "gate_pack":
            kind = "exit" if r.args.get("exit") else "add" if r.args.get("add") else "new"
            out[r.args["symbol"]] = (kind, r.args["side"])
        elif r.name == "pick_pack" and str(r.args.get("note") or "").startswith("No position in"):
            out[r.args["symbol"]] = NONE
    return out


def test_the_table_covers_at_least_forty_phrasings():
    assert len(TABLE) >= 40


@pytest.mark.parametrize("text,book,want", TABLE, ids=[f"{n}:{row[0][:40]}" for n, row in enumerate(TABLE)])
def test_every_phrasing_resolves_against_the_book(text, book, want):
    assert _gates(text, book) == want


def test_one_gate_per_ticker_and_intent_with_its_flags():
    requests = attach.plan_attachments("sell AMD and buy NVDA", KNOWN, NOW, book=list(BOOK))
    gates = [dict(r.args) for r in requests if r.name == "gate_pack"]
    assert gates == [{"side": "LONG", "symbol": "AMD", "exit": True}, {"side": "LONG", "symbol": "NVDA", "add": True}]


def test_close_needs_an_object_and_sell_off_is_one_token():
    assert intent.verbs("AMD close is weak", ["AMD"]) == []
    assert intent.verbs("the close", []) == []
    assert intent.verbs("close my NVDA", ["NVDA"]) == [(0, "close")]
    assert intent.verbs("NVDA sell-off", ["NVDA"]) == []
    assert intent.verbs("sell short AMD", ["AMD"]) == [(0, "short")]
    assert intent.verbs("buy it back", []) == [(0, "cover")]


def test_the_gate_names_the_intent_and_a_name_not_held_gets_a_no_position_note(tmp_path):
    from mentor_packs import gate_pack, pick_pack

    src = gate_pack.fixture_sources(tmp_path)
    req = {f: gate_pack.build("LONG", "ALL", sources=src, **{f: True}).rows[0]["text"] for f in ("add", "exit")}
    assert req["add"].startswith("Request: ADD to a held LONG ALL")
    assert req["exit"].startswith("Request: EXIT of a held LONG ALL")
    assert gate_pack.build("LONG", "ALL", sources=src).rows[0]["text"].startswith("Request: new LONG ALL")
    assert gate_pack.build("LONG", "ALL", sources=src, add="false").rows[0]["add"] is False
    paths = pick_pack.write_fixture_world(tmp_path / "w")
    pack = pick_pack.build("ALL", paths=paths, note="No position in ALL to exit")
    assert pack.rows[0] == {"id": "pick:ALL:note", "kind": "note", "symbol": "ALL",
                            "text": "No position in ALL to exit"}
