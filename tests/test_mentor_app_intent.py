"""P18 intent resolver (review rounds 5-10): the TABLE of phrasings with their book, and the GUARD table.

Every reviewer sentence from rounds 1-10 is a TABLE row; ``test_the_table_size_is_pinned`` pins the count."""

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
KNOWN = {**BOOK, "ALL": "SHORT", "MSFT": "", "QCOM": ""}
NOT_HELD: dict[str, str] = {}

EXIT_L = ("exit", "LONG")
EXIT_S = ("exit", "SHORT")
ADD_L = ("add", "LONG")
ADD_S = ("add", "SHORT")
NEW_L = ("new", "LONG")
NEW_S = ("new", "SHORT")
HISTORY = ("history", "")
STATUS = ("status", "")
WHICH = ("ask_side", "")
NONE = ("none_to_exit", "")
FLIP_TO_LONG = [("exit", "SHORT"), ("new", "LONG")]
FLIP_TO_SHORT = [("exit", "LONG"), ("new", "SHORT")]

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
    # round 6: bare short/long are verbs only in a verb frame
    ("what's the short interest on QCOM", BOOK, {}),
    ("is the TSLA short squeeze over", BOOK, {}),
    ("how long has NVDA been basing", BOOK, {}),
    ("how long should I hold AMD", BOOK, {}),
    ("does NVDA trade long", BOOK, {}),
    ("any long ideas besides QCOM", BOOK, {}),
    ("I'm short TSLA, cover?", BOOK, {"TSLA": EXIT_S}),
    ("go long QCOM", BOOK, {"QCOM": NEW_L}),
    ("short QCOM here", BOOK, {"QCOM": NEW_S}),
    ("the short side looks better", BOOK, {}),
    ("how long can TSLA keep running", BOOK, {}),
    ("buy TSLA", BOOK, {"TSLA": EXIT_S}),
    ("go long TSLA", BOOK, {"TSLA": FLIP_TO_LONG}),
    ("go short AMD", BOOK, {"AMD": FLIP_TO_SHORT}),
    # round 6: past tense is history, never a live intent
    ("sold AMD at 150", BOOK, {"AMD": HISTORY}),
    ("I closed AMD at 150", BOOK, {"AMD": HISTORY}),
    ("I sold AMD, should I buy QCOM?", BOOK, {"AMD": HISTORY, "QCOM": NEW_L}),
    ("covered TSLA this morning", BOOK, {"TSLA": HISTORY}),
    ("bought NVDA at the open", BOOK, {"NVDA": HISTORY}),
    ("trimmed NVDA into the pop", BOOK, {"NVDA": HISTORY}),
    ("exited AMD early", BOOK, {"AMD": HISTORY}),
    ("I'm out of AMD now", BOOK, {"AMD": HISTORY}),
    ("got out of NVDA", BOOK, {"NVDA": HISTORY}),
    ("stop out of AMD", BOOK, {"AMD": HISTORY}),
    # round 6: quiet exits
    ("take some off AMD", BOOK, {"AMD": EXIT_L}),
    ("take NVDA off", BOOK, {"NVDA": EXIT_L}),
    ("cut AMD", BOOK, {"AMD": EXIT_L}),
    ("flat TSLA", BOOK, {"TSLA": EXIT_S}),
    ("go flat AMD", BOOK, {"AMD": EXIT_L}),
    ("NVDA is flat today", BOOK, {}),
    # round 6: a buyback is a company's, not a cover
    ("AMD buy back program announced", BOOK, {}),
    ("does TSLA have a buyback", BOOK, {}),
    ("should I buy back TSLA", BOOK, {"TSLA": EXIT_S}),
    # round 7: no resolver verb never opens a gate (QCOM not held)
    ("should I worry about the short interest on QCOM", BOOK, {}),
    ("how long should I hold QCOM", BOOK, {}),
    ("QCOM buy back program", BOOK, {}),
    ("is QCOM a long term hold", BOOK, {}),
    ("QCOM short squeeze coming?", BOOK, {}),
    ("any long setups like QCOM", BOOK, {}),
    ("what's the QCOM buyback size", BOOK, {}),
    # round 7: the legacy phrasings are resolver rows now
    ("thinking of taking ALL", BOOK, {"ALL": NEW_S}),
    ("thinking about getting into ALL on the open", BOOK, {"ALL": NEW_S}),
    ("should I take ALL here", BOOK, {"ALL": NEW_S}),
    ("thinking of taking AMD", BOOK, {"AMD": ADD_L}),
    ("size up ALL short", BOOK, {"ALL": NEW_S}),
    ("what should my stop be on a QCOM long at 230", BOOK, {"QCOM": NEW_L}),
    ("what should my stop be on a AMD long at 150", BOOK, {"AMD": ADD_L}),
    ("walk me through a pre-trade checklist for a long on QCOM", BOOK, {"QCOM": NEW_L}),
    # round 7: a first-person status about a held name is not an add
    ("I'm long AMD, how does it look?", BOOK, {"AMD": STATUS}),
    ("I'm short TSLA here, thoughts?", BOOK, {"TSLA": STATUS}),
    ("I'm short TSLA, cover?", BOOK, {"TSLA": EXIT_S}),
    ("I'm long QCOM here", BOOK, {"QCOM": NEW_L}),
    ("I'm short ALL, thoughts?", BOOK, {"ALL": NEW_S}),
    ("I'm long TSLA now", BOOK, {"TSLA": FLIP_TO_LONG}),
    # round 7: cut / size down / an add after history
    ("cut my losses on AMD?", BOOK, {"AMD": EXIT_L}),
    ("cut my losses on QCOM?", BOOK, {"QCOM": NONE}),
    ("size down AMD", BOOK, {"AMD": EXIT_L}),
    ("size down QCOM", BOOK, {"QCOM": NONE}),
    ("size down TSLA", BOOK, {"TSLA": EXIT_S}),
    ("I bought NVDA yesterday, add?", BOOK, {"NVDA": ADD_L}),
    ("I bought QCOM yesterday, add?", BOOK, {"QCOM": HISTORY}),
    ("I bought NVDA yesterday", BOOK, {"NVDA": HISTORY}),
    # round 8: a status frame carries across coordinated side words
    ("I'm short TSLA and long AMD, thoughts?", BOOK, {"TSLA": STATUS, "AMD": STATUS}),
    ("I'm long QCOM and short TSLA", BOOK, {"QCOM": NEW_L, "TSLA": STATUS}),
    ("I'm long NVDA and AMD", BOOK, {"NVDA": STATUS, "AMD": STATUS}),
    ("I'm long on AMD, thoughts?", BOOK, {"AMD": STATUS}),
    ("I'm long on QCOM, thoughts?", BOOK, {"QCOM": NEW_L}),
    # round 8: got into / got in are history; a past marker makes any verb history
    ("I got into AMD this morning", BOOK, {"AMD": HISTORY}),
    ("got in ALL at the open, how's it look?", BOOK, {"ALL": HISTORY}),
    ("I got into QCOM yesterday", BOOK, {"QCOM": HISTORY}),
    ("I cut half my AMD this morning", BOOK, {"AMD": HISTORY}),
    ("cut out the AMD noise", BOOK, {}),
    ("cut out the QCOM noise", BOOK, {}),
    ("cut my losses on AMD?", BOOK, {"AMD": EXIT_L}),
    ("get into QCOM here", NOT_HELD, {}),  # QCOM has no known side: no gate
    ("get into ALL here", BOOK, {"ALL": NEW_S}),
    # round 8 advisories
    ("get me out of NVDA", BOOK, {"NVDA": EXIT_L}),
    ("get me out of QCOM", BOOK, {"QCOM": NONE}),
    ("I'm long AMD and it's breaking down, out?", BOOK, {"AMD": EXIT_L}),
    ("TSLA shorts covering, should I?", BOOK, {}),
    ("QCOM shorts covering, should I?", BOOK, {}),
    ("I'm taking AMD off my watchlist", BOOK, {}),
    ("I'm taking QCOM off my watchlist", BOOK, {}),
    # round 9: "flat" after a status frame is a status, never an exit
    ("I'm flat AMD", BOOK, {"AMD": STATUS}),
    ("I'm flat NVDA now", BOOK, {"NVDA": STATUS}),
    ("I'm flat on AMD", BOOK, {"AMD": STATUS}),
    ("I'm already flat AMD", BOOK, {"AMD": STATUS}),
    ("I am flat AMD", BOOK, {"AMD": STATUS}),
    ("went flat AMD", BOOK, {"AMD": HISTORY}),
    ("I'm flat QCOM", BOOK, {"QCOM": STATUS}),
    ("I'm flat AMD, buy back in?", BOOK, {"AMD": NEW_L}),  # rule: re-entry on a name he says he is flat is a new LONG
    ("I'm long AMD, short TSLA, flat NVDA", BOOK, {"AMD": STATUS, "TSLA": STATUS, "NVDA": STATUS}),
    # round 9: tense - narration with no ask cue is history
    ("I cut AMD at 155", BOOK, {"AMD": HISTORY}),
    ("I cut AMD for a loss", BOOK, {"AMD": HISTORY}),
    ("I cut half my NVDA", BOOK, {"NVDA": HISTORY}),
    ("I cut TSLA at 340", BOOK, {"TSLA": HISTORY}),
    ("I cut QCOM at 220", BOOK, {"QCOM": HISTORY}),
    ("I was buying AMD all morning", BOOK, {"AMD": HISTORY}),
    ("I was going to buy QCOM", BOOK, {"QCOM": HISTORY}),
    ("I wanted to sell AMD", BOOK, {"AMD": HISTORY}),
    ("going to cut AMD", BOOK, {"AMD": EXIT_L}),
    ("going to cut QCOM", BOOK, {"QCOM": NONE}),
    # round 9: verb words used as nouns
    ("is NVDA a buy or sell", BOOK, {}),
    ("is QCOM a buy or sell", BOOK, {}),
    ("what's my AMD exit", BOOK, {}),
    ("is AMD a sell here", BOOK, {}),
    # round 9: "at the open" is past only with no future word
    ("add AMD at the open tomorrow", BOOK, {"AMD": ADD_L}),
    ("should I buy QCOM at the open?", BOOK, {"QCOM": NEW_L}),
    ("I bought QCOM at the open", BOOK, {"QCOM": HISTORY}),
    # the reviewers' probe lists (scratchpad rv_p18*, rv_p18r6/q.txt, rv8/b.txt)
    ("the sell-off in AMD, should i add?", BOOK, {"AMD": ADD_L}),
    ("AMD sell-off today", BOOK, {}),
    ("add to NVDA", BOOK, {"NVDA": ADD_L}),
    ("add NVDA", BOOK, {"NVDA": ADD_L}),
    ("closing AMD here", BOOK, {"AMD": EXIT_L}),
    ("exit NVDA?", BOOK, {"NVDA": EXIT_L}),
    ("should i sell NVDA", BOOK, {"NVDA": EXIT_L}),
    ("i want to take profits on NVDA", BOOK, {"NVDA": EXIT_L}),
    ("buy back TSLA", BOOK, {"TSLA": EXIT_S}),
    ("sell half of AMD", BOOK, {"AMD": EXIT_L}),
    ("should I sell some AMD", BOOK, {"AMD": EXIT_L}),
    ("selling my AMD shares", BOOK, {"AMD": EXIT_L}),
    ("should I sell AMD and add to NVDA", BOOK, {"AMD": EXIT_L, "NVDA": ADD_L}),
    ("trim AMD, NVDA looks fine", BOOK, {"AMD": EXIT_L}),
    ("take a look at the QCOM short squeeze", BOOK, {}),
    ("how long should I wait on QCOM", BOOK, {}),
    ("does QCOM have a stock buy back program", BOOK, {}),
    ("is the QCOM short interest worth a trade", BOOK, {}),
    ("should I care how long QCOM has based", BOOK, {}),
    ("I sold QCOM yesterday", BOOK, {"QCOM": HISTORY}),
    ("trim NVDA into strength", BOOK, {"NVDA": EXIT_L}),
    ("cover TSLA at 350", BOOK, {"TSLA": EXIT_S}),
    # round 10 (f): someone else's verb is never his ask
    ("should he sell TSLA?", BOOK, {}),
    ("did he sell TSLA?", BOOK, {}),
    ("he's selling AMD?", BOOK, {}),
    ("who is buying NVDA?", BOOK, {}),
    ("is Cathie buying NVDA?", BOOK, {}),
    ("is anyone shorting QCOM?", BOOK, {}),
    ("the guy on twitter is shorting TSLA?", BOOK, {}),
    ("my wife says sell AMD?", BOOK, {}),
    ("analysts say buy QCOM?", BOOK, {}),
    ('"sell AMD" he said', BOOK, {}),
    ("my brother wants to buy AMD", BOOK, {}),
    ("I say sell AMD", BOOK, {"AMD": EXIT_L}),
    ("I say buy QCOM", BOOK, {"QCOM": NEW_L}),
    # round 10: first-person present progressive is a present intent; with a duration it is narration
    ("I'm buying AMD", BOOK, {"AMD": ADD_L}),
    ("I'm selling AMD", BOOK, {"AMD": EXIT_L}),
    ("I'm covering TSLA", BOOK, {"TSLA": EXIT_S}),
    ("I'm buying QCOM", BOOK, {"QCOM": NEW_L}),
    ("I'm buying AMD all morning", BOOK, {"AMD": HISTORY}),
    ("I'm selling NVDA since the open", BOOK, {"NVDA": HISTORY}),
    # round 10: coordinated objects, an ask after a noun, "it" bound to the clause's ticker
    ("should I buy QCOM or AMD", NOT_HELD, {"QCOM": NEW_L, "AMD": NEW_L}),
    ("should I buy QCOM or AMD", BOOK, {"QCOM": NEW_L, "AMD": ADD_L}),
    ("QCOM buyback \u2014 buy?", BOOK, {"QCOM": NEW_L}),
    ("NVDA: short it", BOOK, {"NVDA": FLIP_TO_SHORT}),
    ("QCOM: short it", BOOK, {"QCOM": NEW_S}),
    ("ALL short \u2014 take it?", BOOK, {"ALL": NEW_S}),
    # round 11: a ":" hands its left side to the right clause as the speaker
    ("Tom: sell AMD", BOOK, {}),
    ("my wife: sell AMD", BOOK, {}),
    ("Cramer: short TSLA", BOOK, {}),
    ("Jensen: buy NVDA", BOOK, {}),
    ("Analysts: buy QCOM", BOOK, {}),
    ("Goldman: buy QCOM", BOOK, {}),
    ("me: sell AMD?", BOOK, {"AMD": EXIT_L}),
    # round 11: a coordinated object never crosses into its own predicate
    ("should I buy QCOM and AMD looks weak", BOOK, {"QCOM": NEW_L}),
    ("buy QCOM and AMD is extended", BOOK, {"QCOM": NEW_L}),
    ("should I sell NVDA or AMD is better", BOOK, {"NVDA": EXIT_L}),
    # round 11: "it" binds its own, the next or the previous clause's ticker, with that clause's side tag
    ("should I take it? AMD short", BOOK, {"AMD": FLIP_TO_SHORT}),
    ("should I take it? QCOM short", BOOK, {"QCOM": NEW_S}),
    ("QCOM long. take it?", BOOK, {"QCOM": NEW_L}),
    # round 11: habits are narration; Capitalized weekdays are not names
    ("I'm buying AMD every dip", BOOK, {"AMD": HISTORY}),
    ("I always trim NVDA into strength", BOOK, {"NVDA": HISTORY}),
    ("On Monday should I sell AMD?", BOOK, {"AMD": EXIT_L}),
    # round 12: a habit or someone else's clause never turns his own ask in the next clause into history
    ("I always sell too early, sell AMD now?", BOOK, {"AMD": EXIT_L}),
    ("I usually sell AMD here, sell?", BOOK, {"AMD": EXIT_L}),
    ("Buffett is buying, buy QCOM?", BOOK, {"QCOM": NEW_L}),
    ("AMD CEO is selling, should I?", BOOK, {"AMD": EXIT_L}),
    ("I usually sell AMD here", BOOK, {"AMD": HISTORY}),
    # round 12: comma lists, plural pronouns, a lowercase ticker right after a trade verb
    ("sell AMD, NVDA", BOOK, {"AMD": EXIT_L, "NVDA": EXIT_L}),
    ("sell AMD, NVDA and buy QCOM", BOOK, {"AMD": EXIT_L, "NVDA": EXIT_L, "QCOM": NEW_L}),
    ("sell them both? AMD NVDA", BOOK, {"AMD": EXIT_L, "NVDA": EXIT_L}),
    ("AMD and NVDA both look done, sell?", BOOK, {"AMD": EXIT_L, "NVDA": EXIT_L}),
    ("sell amd", BOOK, {"AMD": EXIT_L}),
    ("should I sell all of my AMD", BOOK, {"AMD": EXIT_L}),
    # round 12: a bare "sell" on a held short asks which side; "sell more" is an add
    ("sell TSLA", BOOK, {"TSLA": WHICH}),
    ("sell more TSLA", BOOK, {"TSLA": ADD_S}),
    ("sell AMD and TSLA", BOOK, {"AMD": EXIT_L, "TSLA": WHICH}),
    ("sell more QCOM", BOOK, {"QCOM": NEW_S}),
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
    out: dict = {}
    for r in requests:
        if r.name == "gate_pack":
            kind = "exit" if r.args.get("exit") else "add" if r.args.get("add") else "new"
            got = (kind, r.args["side"])
            if r.args.get("flip"):
                out.setdefault(r.args["symbol"], []).append(got)
            else:
                out[r.args["symbol"]] = got
        elif r.name == "pick_pack" and str(r.args.get("note") or "").startswith("No position in"):
            out[r.args["symbol"]] = NONE
        elif r.name == "pick_pack" and r.reason.startswith("history on "):
            out[r.args["symbol"]] = HISTORY
        elif r.name == "pick_pack" and r.reason.startswith("status of a held "):
            out[r.args["symbol"]] = STATUS
        elif r.name == "pick_pack" and r.reason.startswith("which side on a held short "):
            out[r.args["symbol"]] = WHICH
    return out


def test_the_table_size_is_pinned():
    assert len(TABLE) == 241 and len(GUARD) == 20


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
    assert intent.verbs("sell short AMD", ["AMD"]) == [(0, "go_short")]
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


def test_a_flip_names_both_halves_in_its_request_rows(tmp_path):
    from mentor_packs import gate_pack

    src = gate_pack.fixture_sources(tmp_path)
    out = gate_pack.build("SHORT", "ALL", sources=src, exit=True, flip=True).rows[0]["text"]
    new = gate_pack.build("LONG", "ALL", sources=src, flip=True).rows[0]["text"]
    assert out.startswith("Request: EXIT of a held SHORT ALL") and out.endswith("; a FLIP: then a new LONG ALL")
    assert new.startswith("Request: new LONG ALL") and new.endswith("; a FLIP from a held SHORT ALL")
    assert gate_pack.build("LONG", "ALL", sources=src, flip="0").rows[0]["flip"] is False


def test_history_attaches_the_journal_and_the_pick_never_a_gate():
    names = [r.name for r in attach.plan_attachments("sold AMD at 150", KNOWN, NOW, book=list(BOOK))]
    assert "journal_pack" in names and "pick_pack" in names and "gate_pack" not in names


def test_a_status_attaches_the_pick_and_the_book_and_no_gate():
    names = [r.name for r in attach.plan_attachments("I'm long AMD, how does it look?", KNOWN, NOW, book=list(BOOK))]
    assert "pick_pack" in names and "book_pack" in names and "gate_pack" not in names


def test_the_legacy_side_words_are_gone_from_attach():
    assert not hasattr(attach, "_side_words") and not hasattr(attach, "_SHORT_WORD")


#: The final guard, as its own table: each row breaks one condition, so no gate opens.
GUARD = [
    # (a) no present-tense trade verb
    ("AMD looks heavy into the close", "a"),
    ("QCOM sold off hard", "a"),
    ("bought NVDA at 140", "a"),
    # (b) the ticker is not the direct object (an adjective, or the verb has another object)
    ("buy the AMD dip?", "b"),
    ("cut out the NVDA noise", "b"),
    ("sell the TSLA news?", "b"),
    ("add the QCOM idea to my list", "b"),
    # (c) a past-time marker in the clause
    ("sell AMD yesterday was right", "c"),
    ("I trimmed NVDA earlier", "c"),
    ("cover TSLA last week worked", "c"),
    ("add QCOM at the open was a mistake", "c"),
    # (d) a status frame on a held name
    ("I'm long AMD", "d"),
    ("I am short TSLA here", "d"),
    # (e) no ask cue and the verb is not first in its clause: narration
    ("I cut AMD at 155", "e"),
    ("I sell NVDA when it breaks the low", "e"),
    ("I usually trim NVDA into strength", "e"),
    ("I cover TSLA on red days", "e"),
    ("we add AMD on pullbacks", "e"),
    ("my plan was I buy QCOM above 230", "e"),
    ("I was buying AMD all morning", "e"),
]


@pytest.mark.parametrize("text,why", GUARD, ids=[f"{why}:{text[:40]}" for text, why in GUARD])
def test_the_final_guard_opens_no_gate(text, why):
    requests = attach.plan_attachments(text, KNOWN, NOW, book=list(BOOK))
    assert not [r for r in requests if r.name == "gate_pack"], why


def test_a_bare_sell_on_a_held_short_attaches_the_book_and_says_which():
    requests = attach.plan_attachments("sell TSLA", KNOWN, NOW, book=list(BOOK))
    notes = [r.args.get("note") for r in requests if r.name == "pick_pack"]
    assert "book_pack" in [r.name for r in requests] and not [r for r in requests if r.name == "gate_pack"]
    assert notes == ["You are short TSLA: 'sell more' means add, 'cover' means exit - say which"]


def test_a_lowercase_word_is_a_ticker_only_right_after_a_trade_verb():
    assert attach.find_symbols("sell amd", KNOWN) == ["AMD"]
    assert attach.find_symbols("amd looks weak", KNOWN) == []
    assert attach.find_symbols("sell all of my AMD", KNOWN) == ["AMD"]
