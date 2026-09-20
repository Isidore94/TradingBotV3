r"""TJ-4 item 1 - the hash-stable day pack. RED before the build.

Packet `.claude/packets/TJ-4.md` item 1; `plan.md` §12.4 "TJ-4" change 1 and its
**AMENDED 2026-09-19** block; decision 0021 answers 8 and 14.

The contract these tests pin (the builder may ADD keys, never remove one)
------------------------------------------------------------------------

``scripts/day_review_pack.py`` - PURE. It opens no store, starts no thread, has
no clock of its own, never reads the 1.1 GB setup tracker and never imports an
ask-first module. Everything it needs is handed in::

    SCHEMA = "day_review_pack_v1"
    SECTIONS = ("trader_said", "forecast", "environment", "measured",
                "internals", "walkaway", "skill", "reads", "congruence",
                "trades", "report_card", "mood")

    build_pack(session_date, *, entries=(), forecast=None, story=None,
               environment=(), d1_label="", internals=(), walkaway=None,
               reads=(), congruence=(), trades=(), now=None) -> dict

    allowed_source_ids(pack) -> tuple[str, ...]

    default_root() -> Path               # project_paths.DAY_REVIEW_DIR, at CALL time
    pack_path(session, *, root=None)     # <root>/sessions/<date>/pack.json
    write_pack(pack, *, root=None) -> Path
    read_pack(session, *, root=None) -> dict | None

The pack carries ``schema``, ``session_date``, ``inputs_hash``, ``built_at`` and
the twelve sections. Six of them are LISTS whose every item names its source;
``trades`` and ``walkaway`` are mappings whose rows do the same; ``forecast``
carries the verbatim text plus one citable cell per `forecast_brief` field.
``inputs_hash`` is over the INPUTS and never over the clock, so the same session
built twice hashes the same.

Two hooks, empty until their packets land: ``report_card`` (TJ-12) and ``mood``
(TJ-7). They are PRESENT and empty, never absent - a reader with one shape has
no special cases.

NOTE FOR THE BUILDER: `day_review_index._prune` unlinks every child of a
`sessions/<date>/` folder older than `KEEP_SESSIONS` (40) and removes the
folder, so a `pack.json` written there is deleted with the index it sits beside.
That is survivable only because the pack is rebuildable from durable inputs -
which is what `test_two_builds_of_the_same_session_hash_equal_...` proves.
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

SESSION = fx.SESSION

#: The twelve sections, in the order `plan.md` TJ-4 change 1 names them.
PLAN_SECTIONS = (
    "trader_said", "forecast", "environment", "measured", "internals",
    "walkaway", "skill", "reads", "congruence", "trades", "report_card", "mood",
)

#: The sections that are a LIST of items, each of which must name its source.
ITEM_SECTIONS = (
    "trader_said", "environment", "measured", "internals", "reads", "congruence",
)


def _list_items(pack) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for name in ITEM_SECTIONS:
        for item in pack.get(name) or ():
            out.append((name, dict(item)))
    return out


# ---------------------------------------------------------------------------
# the shape
# ---------------------------------------------------------------------------


def test_the_pack_carries_every_section_the_plan_names_and_two_empty_hooks():
    """`plan.md` TJ-4 change 1 as amended. `report_card` and `mood` are hooks.

    A hook is PRESENT and empty. Leaving the key out would make TJ-12 and TJ-7
    editors of this module rather than fillers of a slot, and would give the
    narration prompt two shapes for the same session.
    """
    import day_review_pack

    pack = fx.build()

    assert pack["schema"] == day_review_pack.SCHEMA
    assert pack["session_date"] == SESSION
    assert tuple(day_review_pack.SECTIONS) == PLAN_SECTIONS
    for name in PLAN_SECTIONS:
        assert name in pack, f"the pack has no {name} section"
    assert not pack["report_card"], "TJ-12's hook ships empty"
    assert not pack["mood"], "TJ-7's hook ships empty"
    assert str(pack["inputs_hash"]).strip()


def test_an_observation_and_a_prediction_from_one_card_are_two_items():
    """TJ-14A's rule, carried into the pack (packet: "SEPARATE items").

    One Mentor card holds what the trader SAW and what they CALLED. Folded into
    one item, the story could quote a description as a call - which is the one
    thing TJ-4's amendment forbids. Only a `prediction` may be called a call.
    """
    pack = fx.build()

    clicked = fx.observing_and_predicting_entry()
    mine = [
        item for item in pack["trader_said"]
        if str(item.get("entry_id") or "") == clicked["entry_id"]
    ]
    kinds = sorted(str(item.get("kind") or "") for item in mine)
    assert kinds == ["observation", "prediction"], mine
    assert len({str(item["source_id"]) for item in mine}) == 2, (
        "the two halves of one card must be citable apart"
    )
    call = next(item for item in mine if item["kind"] == "prediction")
    assert call["direction"] == "up"
    assert call["horizon"] == "rest_of_day"
    seen = next(item for item in mine if item["kind"] == "observation")
    assert "Breadth is better" in str(seen.get("text") or "")
    assert not seen.get("direction"), "an observation carries no direction"


def test_a_note_with_no_click_is_an_observation_and_never_a_prediction():
    """The 13 live rows whose `mentor` key is PRESENT and EMPTY.

    `market_journal.prediction_of` answers `None` for every one of them, so the
    pack may not manufacture a call out of the words.
    """
    pack = fx.build()

    noted = fx.observation_only_entry()
    mine = [
        item for item in pack["trader_said"]
        if str(item.get("entry_id") or "") == noted["entry_id"]
    ]
    assert [str(item.get("kind") or "") for item in mine] == ["observation"], mine


def test_a_wordless_click_puts_no_empty_observation_in_the_pack():
    """TJ-14A: "a card answered with clicks and no words has an empty `text`".

    Nobody wrote a sentence and the desk does not write one for them (decision
    0021 answer 29), so an empty `observation` is no item at all - an empty
    quote in `trader_said` is a source id the model can cite and say nothing
    about.
    """
    wordless = fx.observing_and_predicting_entry(observation="")
    reads, _grades = fx.graded_reads([wordless])
    pack = fx.build(
        with_forecast=False,
        with_machine_row=False,
        entries=[wordless],
        story=fx.daily_story([wordless]),
        reads=reads,
        congruence=fx.congruence(reads),
    )

    mine = [
        item for item in pack["trader_said"]
        if str(item.get("entry_id") or "") == wordless["entry_id"]
    ]
    assert [str(item.get("kind") or "") for item in mine] == ["prediction"], mine


def test_the_desks_own_row_never_enters_the_pack():
    """TJ-1 item 2: `is_machine_entry` is the ONE filter, and it applies here.

    The machine row is a real `auto_mode_flip` entry, not a dict with a `[desk]`
    prefix in its text. It may appear in no section and may not be citable.
    """
    import day_review_pack

    pack = fx.build(with_machine_row=True)

    machine = fx.machine_entry()
    assert machine["entry_id"], "fixture drift: the machine row has no id"

    for name, item in _list_items(pack):
        assert str(item.get("entry_id") or "") != machine["entry_id"], (name, item)
    cited = " ".join(str(item) for item in day_review_pack.allowed_source_ids(pack))
    assert machine["entry_id"] not in cited, cited


def test_the_pasted_forecast_is_never_quoted_as_the_traders_own_words():
    """It is somebody else's brief. `trader_said` is the trader's.

    `market_read_grades.read_rows` already refuses it as a read for exactly this
    reason; the pack's own section has to refuse it too.
    """
    pack = fx.build(with_forecast=True)

    pasted = fx.forecast_entry()
    assert all(
        str(item.get("entry_id") or "") != pasted["entry_id"]
        for item in pack["trader_said"]
    ), pack["trader_said"]
    assert str(pack["forecast"].get("entry_id") or "") == pasted["entry_id"]


# ---------------------------------------------------------------------------
# source ids
# ---------------------------------------------------------------------------


def test_every_item_names_its_source_and_every_id_is_allowed_and_unique():
    """The grounding contract: a model may cite only what the pack carries.

    `market_story_narration` is the pattern - an output citing anything outside
    `allowed_source_ids` is rejected whole. So every item has to name a source,
    no two items may share one, and the allowed list has to hold them all.
    """
    import day_review_pack

    pack = fx.build()
    items = _list_items(pack)
    assert items, "fixture drift: the pack has no items at all"

    seen: list[str] = []
    for name, item in items:
        source = str(item.get("source_id") or "").strip()
        assert source, f"{name} item carries no source_id: {item}"
        seen.append(source)
    assert len(seen) == len(set(seen)), "two items share one source_id"

    allowed = tuple(str(value) for value in day_review_pack.allowed_source_ids(pack))
    assert len(allowed) == len(set(allowed)), "the allowed list repeats an id"
    assert set(seen) <= set(allowed), sorted(set(seen) - set(allowed))
    assert all(value.strip() for value in allowed)


def test_the_forecasts_own_fields_are_cited_one_by_one():
    """plan.md TJ-4 change 2: the brief enters "as the `forecast_brief` fields
    (playbook, bottom line, turbulence), each with its own `source_id`, plus the
    verbatim text".

    `chased_against_news` is judged against the brief's stated bearish-reversal
    conditions, so that field has to be citable on its own.
    """
    import day_review_pack

    pack = fx.build(with_forecast=True)
    forecast = pack["forecast"]

    assert str(forecast.get("text") or "").lstrip().startswith("# Market Morning Brief")
    fields = forecast.get("fields") or {}
    for name in ("playbook_bullish", "playbook_bearish", "bottom_line", "turbulence"):
        assert name in fields, (name, sorted(fields))
        assert str(fields[name].get("source_id") or "").strip(), name
    assert "Brent snaps back above" in str(fields["playbook_bearish"].get("value") or "")

    allowed = set(str(value) for value in day_review_pack.allowed_source_ids(pack))
    for name, cell in fields.items():
        assert str(cell["source_id"]) in allowed, name


def test_the_walkaway_rows_and_the_skill_block_are_citable_too():
    """A story that says "you vetoed ABCL and it ran 4%" has to cite that row."""
    import day_review_pack

    pack = fx.build()
    allowed = set(str(value) for value in day_review_pack.allowed_source_ids(pack))

    top = pack["walkaway"]["top"]
    assert top, pack["walkaway"]
    for row in top:
        assert str(row.get("source_id") or "") in allowed, row
    assert str(pack["skill"].get("source_id") or "") in allowed, pack["skill"]
    for row in pack["trades"]["rows"]:
        assert str(row.get("source_id") or "") in allowed, row


# ---------------------------------------------------------------------------
# the sections that come from other packets
# ---------------------------------------------------------------------------


def test_the_reads_section_is_tj10s_measured_verdict_and_not_a_second_opinion():
    """Decision 0021 answer 14. The pack carries the row; it never re-grades.

    The fixture tape rises 100.00 -> 102.00 after the 07:02 stamp and the ATR is
    chosen so that move is exactly three flat-band widths, so the `up` call is
    `right` under any band the builder declared.
    """
    pack = fx.build()

    reads = {str(row["entry_id"]): row for row in pack["reads"]}
    clicked = fx.observing_and_predicting_entry()
    row = reads[clicked["entry_id"]]
    assert row["verdict"] == "right", row
    assert row["source"] == "click"
    assert str(row.get("read_id") or "").strip(), "a verdict must name the read it graded"


def test_the_congruence_lines_travel_whole_with_their_kinds_in_order():
    """TJ-10 item 6's spine, unchanged. The pack reformats nothing."""
    import market_read_grades as grades

    pack = fx.build()
    kinds = tuple(str(line.get("kind") or "") for line in pack["congruence"])
    for kind in grades.CONGRUENCE_KINDS:
        assert kind in kinds, (kind, kinds)
    assert [k for k in kinds if k in grades.CONGRUENCE_KINDS] == list(
        grades.CONGRUENCE_KINDS
    ), kinds


def test_the_skill_line_is_tj11s_and_the_pack_computes_no_rate_of_its_own():
    """TJ-11 item 6. A rate counts CLOSED horizons only, and TJ-11 owns it.

    The fixture's skill cell says 1 of 2 closed horizons ran. A pack that
    recomputed anything would have to reach a population it was never handed.
    """
    pack = fx.build()

    cells = pack["skill"]["session"]["cells"]
    assert len(cells) == 1, cells
    assert cells[0]["measured"] == 2
    assert cells[0]["pending"] == 0
    assert cells[0]["rate"] == pytest.approx(0.5)
    assert pack["skill"]["session"]["sentence"].startswith("This session:")


def test_the_walkaway_section_counts_every_population_and_tops_by_ran_after():
    """plan TJ-4 change 1: "TJ-2 counts per population + top three by Ran after"."""
    pack = fx.build()

    walkaway = pack["walkaway"]
    assert walkaway["counts"]["rejected"] == 1
    assert walkaway["counts"]["liked_not_traded"] == 0
    assert walkaway["counts"]["traded_left_early"] == 0
    assert walkaway["counts"]["claimed_d1"] == 0
    assert len(walkaway["top"]) <= 3
    assert [row["symbol"] for row in walkaway["top"]] == ["ABCL"]
    assert walkaway["top"][0]["real_miss"] == "real_miss"
    assert walkaway["top"][0]["ran_after_pct"] == pytest.approx(4.0)


def test_the_measured_section_is_build_daily_storys_own_cells():
    """`market_story.build_daily_story` measured SPY at +2.00% on this tape."""
    pack = fx.build()

    cells = {str(cell["symbol"]): cell for cell in pack["measured"]}
    assert cells["SPY"]["status"] == "measured"
    assert cells["SPY"]["change_pct"] == pytest.approx(2.0)
    assert cells["SPY"]["close"] == pytest.approx(102.0)


def test_the_environment_section_keeps_both_regime_shifts_and_the_d1_label():
    """plan TJ-4 change 1: "the day's regime shifts + the D1 label"."""
    pack = fx.build()

    shifts = [row for row in pack["environment"] if row.get("kind") == "regime_shift"]
    labels = [row for row in pack["environment"] if row.get("kind") == "d1_label"]
    assert [row["to_regime"] for row in shifts] == ["risk_on", "neutral"]
    assert [row["source"] for row in shifts] == ["auto", "user"]
    assert len(labels) == 1, pack["environment"]
    assert labels[0]["label"] == "trending_up"


def test_the_trades_section_counts_the_day_and_names_every_line():
    """plan TJ-4 change 1: "count, wins/losses, net R or P&L, one line each"."""
    pack = fx.build()

    trades = pack["trades"]
    assert trades["n"] == 2
    assert trades["wins"] == 1
    assert trades["losses"] == 1
    assert trades["net_pnl"] == pytest.approx(120.0)
    assert [row["symbol"] for row in trades["rows"]] == ["ABCL", "ERAS"]


# ---------------------------------------------------------------------------
# internals (TJ-14A item 6)
# ---------------------------------------------------------------------------


def test_the_internals_are_the_open_each_mentor_hour_and_the_close_in_order():
    pack = fx.build()

    marks = pack["internals"]
    assert [str(mark["kind"]) for mark in marks] == ["open", "mentor", "close"]
    assert [str(mark["at"]) for mark in marks] == sorted(str(m["at"]) for m in marks)


def test_an_internals_mark_travels_compacted_with_the_scalar_the_ai_can_read():
    """CLAUDE.md: `compact_for_ai` with the SCALAR `common.internals`.

    Measured 2026-09-19: `ai_summary._bounded` stops six levels down, exactly
    where a derived line's `inputs` sit, so the model saw "[nested content
    omitted]" where the sector read should be. The scalar is what survives, so
    the pack carries the compacted projection and not the raw v2 block.
    """
    import trade_mentor_context

    pack = fx.build()
    mark = pack["internals"][1]
    context = mark["context"]

    assert "common" in context and "rows" in context, "the raw v2 block was stored"
    assert "readings" not in context
    assert context["common"]["schema"] == trade_mentor_context.SCHEMA
    assert isinstance(context["common"]["internals"], str)
    assert context["common"]["internals"].strip()


# ---------------------------------------------------------------------------
# hash stability, and what is missing
# ---------------------------------------------------------------------------


def test_two_builds_of_the_same_session_hash_equal_however_late_the_second_is():
    """"one function, hash-stable" (plan TJ-4 change 1).

    The post-close tick and the nightly slot build the SAME pack, hours apart.
    A clock inside the hash would make change 2's `inputs_hash` skip unreachable
    and the model would be paid every night for a session that had not moved.
    """
    first = fx.build(now=fx.AFTER_THE_CLOSE)
    second = fx.build(now=fx.AFTER_THE_CLOSE + timedelta(hours=9))

    assert first["inputs_hash"] == second["inputs_hash"]
    assert {k: v for k, v in first.items() if k != "built_at"} == {
        k: v for k, v in second.items() if k != "built_at"
    }


def test_a_changed_note_changes_the_hash():
    """The other half: a stable hash that never moves is not a hash."""
    before = fx.build(with_forecast=False, with_machine_row=False)

    entries = [
        fx.observing_and_predicting_entry(observation="I changed my mind at 07:02."),
        fx.observation_only_entry(),
    ]
    reads, _grades = fx.graded_reads(entries)
    after = fx.build(
        with_forecast=False,
        with_machine_row=False,
        entries=entries,
        story=fx.daily_story(entries),
        reads=reads,
        congruence=fx.congruence(reads),
    )

    assert before["inputs_hash"] != after["inputs_hash"]


def test_a_session_with_no_pasted_forecast_says_so_rather_than_inventing_one():
    """plan TJ-4 change 2: "A forecast absent -> `chased_against_news` unknown".

    The pack's job is to make that reachable: the section is present and empty,
    and no forecast field is citable.
    """
    import day_review_pack

    pack = fx.build(with_forecast=False)

    assert pack["forecast"] in (None, {}, ()), pack["forecast"]
    allowed = " ".join(str(value) for value in day_review_pack.allowed_source_ids(pack))
    assert "forecast" not in allowed, allowed


# ---------------------------------------------------------------------------
# where it is stored
# ---------------------------------------------------------------------------


def test_the_pack_is_written_under_day_review_dir_and_read_back(tmp_path):
    """`DAY_REVIEW_DIR / "sessions" / <date> / "pack.json"`, root read at CALL
    time so a test can redirect it - the idiom `day_review_index` already uses.
    """
    import day_review_pack
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    assert day_review_pack.default_root() == Path(project_paths.DAY_REVIEW_DIR)

    root = tmp_path / "day_review"
    pack = fx.build()
    path = day_review_pack.write_pack(pack, root=root)

    assert path == day_review_pack.pack_path(SESSION, root=root)
    assert path == root / "sessions" / SESSION / "pack.json"
    assert path.exists()
    again = day_review_pack.read_pack(SESSION, root=root)
    assert again is not None
    assert again["inputs_hash"] == pack["inputs_hash"]
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == day_review_pack.SCHEMA


def test_an_absent_pack_is_none_and_never_an_exception(tmp_path):
    """A page and a night both ask for a pack that may not exist yet."""
    import day_review_pack

    assert day_review_pack.read_pack("2026-09-17", root=tmp_path / "nothing") is None


def test_the_pack_builder_takes_its_moment_and_reaches_no_ask_first_module():
    """PURE, like `grade_read` and `walkaway_day.build` beside it.

    The 1.1 GB tracker and the ask-first detectors are all reachable by import
    from `scripts/`; a pure builder reaches none of them, and a test that only
    read the output would never notice.
    """
    import day_review_pack

    pack = fx.build(now=fx.AFTER_THE_CLOSE)
    assert str(pack["built_at"]).startswith("2026-09-19"), pack["built_at"]

    source = Path(day_review_pack.__file__).read_text(encoding="utf-8")
    for forbidden in ("legacy", "master_avwap", "bounce_bot", "setup_tracker"):
        assert forbidden not in source, forbidden
