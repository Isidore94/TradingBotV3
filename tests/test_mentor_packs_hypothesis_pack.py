"""Mentor app P11: hypothesis_pack - a night hypothesis is a query into the shadow permutation grid,
looked up (never run, never applied), validated against the report's own vocabulary, and graded
against the NEXT report (hit = still a key, miss = listed but failed, gone = not there)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_search as sps  # noqa: E402
from mentor_app import challenge  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402
from mentor_packs import hypothesis_pack as hp  # noqa: E402

Q = hp.FIXTURE_QUERY


@pytest.fixture
def report():
    return hp.Report(path="2026-09-26.json", asof="2026-09-26", body=hp.fixture_report())


def _write(folder: Path, day: str, body: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------- lookup
def test_a_key_cell_returns_the_reports_numbers(report):
    cell, why = hp.find(Q, report)
    assert why == "" and cell is not None and hp.lookup(Q, report) == cell
    key = hp.fixture_report()["populations"]["swing"]["horizons"]["5"]["families"]["avwap_breakout LONG"]["keys"][0]
    assert (cell.n, cell.sessions, cell.wins) == (64, 22, 41)
    assert cell.win_rate == key["selection"]["win_rate"] and cell.wilson_lb == key["selection"]["wilson_lb"]
    assert (cell.holdout_n, cell.holdout_wins, cell.holdout_passed, cell.status) == (18, 13, True, "key")
    assert cell.holdout_lb99 == round(sps.wilson_lower_bound(13, 18, sps.Z99), 4), "the report's 99% bound"
    assert cell.report_asof == "2026-09-26" and cell.horizon_name == "5d"
    text = cell.text()
    assert "n=64" in text and "LB95 0.52" in text and "hold-out n=18" in text and "passed" in text


def test_the_horizon_name_and_facet_order_do_not_matter(report):
    other = {**Q, "horizon": "5d", "facets": ["spy_trend=up", "sma100_support=held"]}
    assert hp.lookup(other, report) == hp.lookup(Q, report)
    assert hp.lookup({**Q, "facets": {"spy_trend": "up", "sma100_support": "held"}}, report) is not None


def test_a_listed_rejected_cell_is_found_with_its_reason(report):
    cell, _ = hp.find({**Q, "facets": ["rs_vs_spy=weak"]}, report)
    assert cell.status == "rejected" and not cell.holdout_passed
    assert "lower bound does not beat the baseline" in cell.holdout_reason


@pytest.mark.parametrize("query, reason", [
    ({**Q, "facets": ["made_up=yes"]}, "not in vocabulary: facet made_up"),
    ({**Q, "facets": ["rs_vs_spy=strong"]}, "not in vocabulary: rs_vs_spy=strong"),
    ({**Q, "family": "gap_fill"}, "not in vocabulary: family gap_fill LONG"),
    ({**Q, "horizon": "20"}, "not in vocabulary: horizon 20"),
    ({**Q, "facets": ["spy_trend=up"]}, hp.MISS_NOT_IN_GRID),
    ({**Q, "facets": ["a=1", "b=2", "c=3", "d=4"]}, "grid stops at depth 3"),
    ({**Q, "population": "crypto"}, "population must be one of swing, m5"),
    ({**Q, "side": "UP"}, "side (LONG or SHORT)"),
    ({**Q, "population": "m5", "horizon": "held30"}, "the report refused this horizon"),
])
def test_a_miss_says_why_and_never_guesses(report, query, reason):
    cell, why = hp.find(query, report)
    assert cell is None and reason in why
    assert hp.MISS_NOT_IN_GRID.startswith("not in the grid (depth > 3 or under the floor")


def test_no_report_is_a_miss_not_a_guess():
    assert hp.find(Q, None) == (None, "no permutation report to look in")


def test_the_vocabulary_comes_from_the_reports_published_cells(report):
    vocab = hp.vocabulary(report)
    assert vocab["swing"]["facets"] == {"rs_vs_spy": ["weak"], "sma100_support": ["held"], "spy_trend": ["up"]}
    assert vocab["swing"]["families"] == ["avwap_breakout LONG"] and vocab["swing"]["horizons"] == ["5d"]
    assert vocab["m5"] == {"horizons": [], "families": [], "facets": {}}, "a refused horizon lends no vocabulary"


def test_latest_report_reads_the_newest_history_copy_else_the_live_file(tmp_path):
    live = tmp_path / "permutation_report.json"
    live.write_text(json.dumps(hp.fixture_report("2026-09-12")), encoding="utf-8")
    history = tmp_path / "history"
    assert hp.latest_report(history, live).asof == "2026-09-12"
    _write(history, "2026-09-19", hp.fixture_report("2026-09-19"))
    _write(history, "2026-09-26", hp.fixture_report("2026-09-26"))
    assert hp.latest_report(history, live).asof == "2026-09-26"
    assert hp.latest_report(tmp_path / "none", tmp_path / "none.json") is None


# ---------------------------------------------------------------- the pack
def test_the_pack_rows_have_unique_tz_aware_ids(tmp_path):
    world = hp.write_fixture_world(tmp_path)
    pack = hp.build(now=hp.FIXTURE_NOW, chat_db=world["chat"], history_dir=world["history"],
                    report_file=world["report_file"])
    assert pack.ids == ("hyp:report:asof", "hyp:2026-09-28:1", "hyp:2026-09-28:1:cell", "hyp:2026-09-21:1",
                        "hyp:2026-09-21:1:cell")
    assert len(set(pack.ids)) == len(pack.ids)
    asof = pack.rows[0]
    assert datetime.fromisoformat(asof["built_utc"]).tzinfo is not None
    assert "data date 2026-09-26" in asof["text"] and "1 open, 1 graded" in asof["text"]
    assert "n=64" in pack.rows[2]["text"] and "GONE" in pack.rows[4]["text"]
    card = hp.card_markdown(pack)
    assert card.rstrip().endswith(hp.FOOTER) and "fixtures and the ladder" in hp.FOOTER
    assert "[hyp:2026-09-28:1:cell]" in card


def test_no_hypothesis_yet_is_one_row(tmp_path):
    pack = hp.build(chat_db=tmp_path / "none.sqlite3", history_dir=tmp_path / "h", report_file=tmp_path / "r.json")
    assert pack.ids == ("hyp:report:asof", "hyp:none") and "No permutation report found" in pack.rows[0]["text"]
    assert not (tmp_path / "none.sqlite3").exists(), "the pack never creates the chat store"


def test_the_pack_reads_the_chat_db_read_only(tmp_path):
    world = hp.write_fixture_world(tmp_path)
    before = world["chat"].read_bytes()
    hp.build(chat_db=world["chat"], history_dir=world["history"], report_file=world["report_file"])
    assert world["chat"].read_bytes() == before
    source = (SCRIPTS_DIR / "mentor_packs" / "hypothesis_pack.py").read_text(encoding="utf-8")
    assert "mode=ro" in source and "PySide6" not in source


# ---------------------------------------------------------------- grading
def _open(store: MentorChatStore, hyp_id: str, query: dict, report) -> None:
    store.add_challenge(hyp_id, kind="hypothesis", symbol="", claim="why", evidence_ids=["fact:turns"],
                        issued_utc="2026-09-27T05:00:00+00:00",
                        outcome={"status": "open", "query": query, "lookup": hp.lookup_record(query, report)})


def test_grading_waits_for_a_newer_report_then_hit_miss_or_gone(tmp_path):
    history = tmp_path / "history"
    _write(history, "2026-09-26", hp.fixture_report("2026-09-26"))
    first = hp.latest_report(history, tmp_path / "none.json")
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    _open(store, "hyp:2026-09-26:1", Q, first)
    _open(store, "hyp:2026-09-26:2", {**Q, "facets": ["rs_vs_spy=weak"]}, first)
    _open(store, "hyp:2026-09-26:3", {**Q, "facets": ["spy_trend=up"]}, first)
    now = datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc)
    assert hp.grade_open(store, now, history_dir=history, report_file=tmp_path / "none.json") == 0, \
        "the same report never grades its own hypotheses"
    _write(history, "2026-10-03", hp.fixture_report("2026-10-03"))
    assert hp.grade_open(store, now, history_dir=history, report_file=tmp_path / "none.json") == 3
    got = {row["id"]: json.loads(row["outcome_json"]) for row in store.challenges(kind="hypothesis")}
    assert got["hyp:2026-09-26:1"]["result"] == "hit" and got["hyp:2026-09-26:1"]["hit"] is True
    assert got["hyp:2026-09-26:2"]["result"] == "miss" and got["hyp:2026-09-26:2"]["hit"] is False
    assert got["hyp:2026-09-26:3"]["result"] == "gone" and "hit" not in got["hyp:2026-09-26:3"]
    assert got["hyp:2026-09-26:1"]["graded_lookup"]["report_asof"] == "2026-10-03"
    assert all(row["graded_utc"] for row in store.challenges(kind="hypothesis"))
    assert hp.grade_open(store, now, history_dir=history, report_file=tmp_path / "none.json") == 0


def test_a_key_that_fails_the_next_reports_bar_is_a_miss(tmp_path):
    history = tmp_path / "history"
    _write(history, "2026-09-26", hp.fixture_report("2026-09-26"))
    store = MentorChatStore(tmp_path / "chat.sqlite3")
    _open(store, "hyp:x:1", Q, hp.latest_report(history, tmp_path / "none.json"))
    _write(history, "2026-10-03", hp.fixture_report("2026-10-03", key_passes=False))
    challenge.grade_open(store, datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc), veto_outcomes=tmp_path / "v.csv",
                         journal=tmp_path / "j.sqlite3", permutation_history=history,
                         permutation_report=tmp_path / "none.json")
    outcome = json.loads(store.challenges(kind="hypothesis")[0]["outcome_json"])
    assert outcome["result"] == "miss" and "grid median" in outcome["graded_lookup"]["cell"]["holdout_reason"]


def test_the_scorecard_lists_hypotheses_with_n():
    class Store:
        def challenges(self, **_):
            return [{"kind": "hypothesis", "graded_utc": "x", "outcome_json": json.dumps({"hit": True})},
                    {"kind": "hypothesis", "graded_utc": "x", "outcome_json": json.dumps({"result": "gone"})}]

    text = challenge.scorecard(Store(), floor=30)
    assert "**Scorecard: hypothesis challenges**" in text and "issued 2, fully graded 2" in text
    assert "still a key in the next weekly permutation report" in text and "too few (n=1, floor 30)" in text
