"""TJ-12 item 2 - the card computes NO new statistic; it calls the owners.

*"It computes NO new statistic: every number is read from a TJ-9/10/11
function."* (packet TJ-12 item 2.)

The proof shape is TJ-16's: swap the OWNER and watch the card follow it. A test
that only checked the card's arithmetic against a hand-typed expectation would
pass just as happily for a second, drifting copy of the rule.

This file also pins what the card must NOT be able to do: no model endpoint, no
Qt import, no store read. `build` is pure.

Two DORMANT Mentor kinds name this module as their reader
(`scripts/mentor_questions.py:626` `report_card.process_line` /
`:671` `report_card.long_hold_lines`). TJ-12 does not WAKE them - that is a
one-field edit plus a lane in `MainWindow._mentor_question_state`, and this
packet does not grant it - but the reader has to exist and has to read the key
the answer is filed under, which is what the registry's own probe checks.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tj12_support as fx  # noqa: E402

CARD_MODULE = SCRIPTS_DIR / "day_report_card.py"

#: Anything that can reach a model, a socket or a screen.
FORBIDDEN_IMPORTS = (
    "openai", "requests", "urllib", "httpx", "socket",
    "ai_summary", "ai_jobs.provider", "ai_jobs.runner",
    "PySide6", "PyQt5", "qtawesome", "pyqtgraph",
)


def _line(card, key):
    for line in card.lines:
        if line["key"] == key:
            return line
    raise AssertionError(f"no {key!r} line: {[item['key'] for item in card.lines]}")


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# ---------------------------------------------------------------------------
# the owners
# ---------------------------------------------------------------------------
def test_the_your_reads_line_prints_the_owners_integers_and_re_tallies_nothing(tmp_path):
    """A sentinel tally no arithmetic could have produced from the ledger."""
    import day_report_card

    inputs = fx.day_inputs(tmp_path, with_reads=False)
    inputs["your_reads"] = {
        "text": "Your reads: 41 right of 57 - always up scored 13 of 19 on the same stamps",
        "session": fx.SESSION,
        "n": 57, "right": 41, "wrong": 16, "flat": 0, "pending": 0,
        "empty": False,
        "baselines": {
            "always_up": {"right": 13, "n": 19, "rate": 13 / 19, "meets_floor": False},
            "same_as_the_last_hour": {"right": 0, "n": 0, "rate": None, "meets_floor": False},
            "with_the_d1_environment": {"right": 0, "n": 0, "rate": None, "meets_floor": False},
        },
    }
    line = _line(day_report_card.build(inputs), "your_reads")
    assert line["n"] == 57
    assert "41" in line["text"] and "57" in line["text"], line["text"]


def test_the_did_well_line_quotes_the_skill_sentence_verbatim(tmp_path):
    """Swap TJ-11's sentence and the card's quote follows it."""
    import day_report_card
    import walkaway_day

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    sentinel = "Of this scan: SENTINEL BASE RATES (n 140, measured 140, pending 0)."
    window = dict(fx.skill_window())
    window["sentence"] = sentinel
    day = inputs["walkaway"]
    inputs["walkaway"] = walkaway_day.WalkawayDay(
        liked_not_traded=day.liked_not_traded,
        rejected=day.rejected,
        skill={"session": window, "lately": window},
        sentences=day.sentences,
        money=day.money,
    )
    built = day_report_card.build(inputs)
    assert sentinel in _line(built, "did_well")["text"]
    assert sentinel in _line(built, "missed")["text"]


def test_the_process_line_follows_trade_origin_and_never_its_own_rule(tmp_path, monkeypatch):
    """`trade_origin.planned_state` is the ONE rule (TJ-9, decision 0021 a25)."""
    import day_report_card
    import trade_origin

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    assert _line(day_report_card.build(inputs), "process")["planned"] == 2

    monkeypatch.setattr(trade_origin, "planned_state", lambda *_a, **_k: trade_origin.UNPLANNED)
    line = _line(day_report_card.build(inputs), "process")
    assert line["planned"] == 0
    assert line["unplanned"] == 4, "every trade follows the swapped rule"


def test_the_missed_line_follows_walkaways_own_reason_clause(tmp_path):
    """The shared reason is TJ-11's sentence, not a second count here."""
    import day_report_card
    import walkaway_day

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)
    day = inputs["walkaway"]
    sentences = dict(day.sentences)
    sentences["rejected"] = "You vetoed 5. 3 were real misses; 3 share the reason SENTINEL_CODE."
    inputs["walkaway"] = walkaway_day.WalkawayDay(
        liked_not_traded=day.liked_not_traded,
        rejected=day.rejected,
        skill=day.skill,
        sentences=sentences,
        money=day.money,
    )
    assert "SENTINEL_CODE" in _line(day_report_card.build(inputs), "missed")["text"]


def test_the_day_card_computes_no_wilson_of_its_own(tmp_path, monkeypatch):
    """`walkaway_day` already put the ONE Wilson on every cell as `low`.

    A card that recomputed it would be a second interval on the same numbers -
    and the first thing to drift the day `swing_headline`'s z moves.
    """
    import day_report_card
    import swing_headline

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)

    def _never(*_a, **_k):
        raise AssertionError("the day card must read the cell's own `low`")

    monkeypatch.setattr(swing_headline, "wilson_lower_bound", _never)
    built = day_report_card.build(inputs)
    best = fx.family_cell(fx.skill_window(), fx.BEST_BOUND_FAMILY)
    assert _line(built, "did_well")["best_family"]["low"] == best["low"]


# ---------------------------------------------------------------------------
# what the card may not reach
# ---------------------------------------------------------------------------
def test_the_card_imports_no_model_no_socket_and_no_qt():
    assert CARD_MODULE.exists(), f"{CARD_MODULE} does not exist"
    imported = _imported_names(CARD_MODULE)
    for name in FORBIDDEN_IMPORTS:
        assert not any(
            found == name or found.startswith(name + ".") for found in imported
        ), f"{name} is imported by day_report_card"


def test_build_opens_no_file(tmp_path, monkeypatch):
    """`build` is PURE: the worker did the reading before it was called."""
    import builtins

    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    inputs = fx.day_inputs(tmp_path)

    def _no_open(*_a, **_k):
        raise AssertionError("`build` opened a file")

    monkeypatch.setattr(builtins, "open", _no_open)
    monkeypatch.setattr(Path, "open", _no_open)
    built = day_report_card.build(inputs)
    assert len(built.lines) == 6


# ---------------------------------------------------------------------------
# the two dormant Mentor kinds whose answers this module is the reader for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind_name", ("trade_origin", "open_position_check"))
def test_the_card_holds_the_reader_each_dormant_kind_names(kind_name):
    """`consumer_report`'s own probe, pointed at the function the registry names.

    The registry says the answer to `trade_origin` is read by
    `<module>.process_line` under the key `trade_origin`, and the answer to
    `open_position_check` by `<module>.long_hold_lines` under
    `open_position_state`. The MODULE name in the registry is
    `report_card` while the packet names the file `scripts/day_report_card.py` -
    so this asks for the function the registry names on the module the packet
    names, and the lead owns which of the two names moves.

    Waking the kinds is NOT tested: that is `dormant_until=""` plus a lane in
    `MainWindow._mentor_question_state`, and this packet does not grant it.
    """
    import day_report_card
    import mentor_questions

    kind = mentor_questions.kind_named(kind_name)
    attribute = kind.consumer.rsplit(".", 1)[-1]
    reader = getattr(day_report_card, attribute, None)
    assert reader is not None, f"day_report_card has no {attribute!r}"
    assert mentor_questions._reads_key(reader, kind.answer_key), (
        f"{attribute} never reads {kind.answer_key!r} as a string constant"
    )
