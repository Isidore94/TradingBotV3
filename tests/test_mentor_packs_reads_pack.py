"""P18 C: reads_pack - the trader's own Market Journal reads, calls and grades; the gate carries today's read."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mentor_app import attach  # noqa: E402
from mentor_packs import gate_pack, reads_pack, registry  # noqa: E402

NOW = reads_pack.FIXTURE_NOW


def _ids(pack):
    return [row["id"] for row in pack.rows]


def test_registered_and_the_fixture_reads_no_live_source(monkeypatch):
    assert "reads_pack" in registry.names()

    def boom(*_a, **_k):
        raise AssertionError("the fixture read a live source")

    monkeypatch.setattr(reads_pack, "live_sources", boom)
    pack = reads_pack.fixture()
    assert len(set(_ids(pack))) == len(pack.rows) and all(row.get("text") for row in pack.rows)


def test_the_traders_own_reads_newest_first_never_a_machine_row():
    pack = reads_pack.fixture()
    reads = [row for row in pack.rows if row["kind"] in ("current", "read")]
    assert [row["id"] for row in reads] == ["read:mj-d", "read:mj-b", "read:mj-c", "read:mj-a"]
    assert reads[0]["kind"] == "current" and "Current read Wed 2026-09-30 11:00 ET M5" in reads[0]["text"]
    assert "call: bearish rest of day (medium)" in reads[0]["text"] and "not graded yet" in reads[0]["text"]
    assert "rest of day up -> wrong, moved -0.40 ATR" in reads[1]["text"]
    assert not any("mj-m" in row["id"] or "Auto mode flipped" in row["text"] for row in pack.rows)


def test_n_limits_the_reads_and_yesterdays_newest_is_not_current():
    one = reads_pack.build(n=1, now=NOW, sources=reads_pack.fixture_sources())
    assert [row["id"] for row in one.rows if row["kind"] in ("current", "read")] == ["read:mj-d"]
    tomorrow = datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)
    later = reads_pack.build(n=1, now=tomorrow, sources=reads_pack.fixture_sources())
    assert [row["kind"] for row in later.rows if row["id"] == "read:mj-d"] == ["read"]


def test_accuracy_by_horizon_and_regime_with_n_and_the_floor():
    rows = {row["id"]: row for row in reads_pack.fixture().rows}
    rod = rows["read:acc:rest_of_day"]
    assert (rod["right"], rod["wrong"], rod["n"], rod["meets_floor"]) == (1, 1, 2, False)
    assert "n=2, 50% right" in rod["text"] and "too few to call (n=2, floor 30)" in rod["text"]
    assert rows["read:acc:next_5_sessions"]["pending"] == 1 and rows["read:acc:next_5_sessions"]["n"] == 0
    # The trader's typed regime on the read's day: chop from 09-29; nothing typed before -> unknown.
    assert rows["read:acc:regime:chop"]["wrong"] == 1 and rows["read:acc:regime:chop"]["pending"] == 1
    assert rows["read:acc:regime:unknown"]["right"] == 1


def test_a_clicked_call_is_never_pooled_with_an_extracted_stance():
    src = reads_pack.fixture_sources()
    extra = {"grade_id": "gr-x", "entry_id": "mj-a", "session": "2026-09-28", "horizon": "rest_of_day",
             "direction": "down", "source": "extracted", "verdict": "right", "supersedes": ""}
    grades = [*src.grades(), extra]
    pack = reads_pack.build(now=NOW, sources=reads_pack.Sources(entries=src.entries, grades=lambda: grades,
                                                                regime_rows=src.regime_rows))
    rows = {row["id"]: row for row in pack.rows}
    assert rows["read:acc:rest_of_day"]["n"] == 2
    assert rows["read:acc:rest_of_day:extracted"]["n"] == 1 and "never pooled" in rows[
        "read:acc:rest_of_day:extracted"]["text"]


def test_open_predictions_await_a_grade_oldest_first():
    rows = [row for row in reads_pack.fixture().rows if row["kind"] == "open"]
    assert [row["id"] for row in rows] == ["read:open:1", "read:open:2"]
    assert "range next 5 sessions (low), awaiting a grade (pending 2026-10-06)" in rows[0]["text"]
    assert rows[1]["entry_id"] == "mj-d" and "no grade row yet" in rows[1]["text"]


def test_an_unreadable_ledger_is_unknown_and_unreadable_grades_say_so():
    def boom():
        raise OSError("gone")

    pack = reads_pack.build(now=NOW, sources=reads_pack.Sources(entries=boom, grades=lambda: []))
    assert not pack.rows and "unknown" in pack.empty_text
    src = reads_pack.fixture_sources()
    pack = reads_pack.build(now=NOW, sources=reads_pack.Sources(entries=src.entries, grades=boom))
    assert pack.rows[-1]["id"] == "read:grades:unknown"
    empty = reads_pack.build(now=NOW, sources=reads_pack.Sources(entries=lambda: [], grades=lambda: []))
    assert not empty.rows and "no reads of yours" in empty.empty_text


def test_grade_files_are_read_with_superseded_rows_hidden(tmp_path):
    folder = tmp_path / "reads"
    folder.mkdir()
    rows = [{"grade_id": "gr-1", "entry_id": "mj-c", "verdict": "pending 2026-10-06", "supersedes": ""},
            {"grade_id": "gr-2", "entry_id": "mj-c", "verdict": "right", "supersedes": "gr-1"}]
    (folder / "2026-09-29.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (folder / "2026-01-02.jsonl").write_text(json.dumps({"grade_id": "old"}) + "\n", encoding="utf-8")
    got = reads_pack.read_grade_files(folder, since="2026-06-01")
    assert [row["grade_id"] for row in got] == ["gr-2"]
    assert reads_pack.read_grade_files(tmp_path / "missing") == []


def test_the_gate_carries_todays_read_and_says_when_it_disagrees(tmp_path):
    src = gate_pack.fixture_sources(tmp_path)
    src = gate_pack.Sources(**{**src.__dict__, "reads_sources": reads_pack.fixture_sources()})
    long = gate_pack.build("LONG", "NVDA", now=NOW, sources=src)
    rows = {row["id"]: row for row in long.rows}
    assert "gate:NVDA:read:mj-d" in rows
    assert rows["gate:NVDA:read:conflict"]["text"] == "Your 11:00 read said bearish rest of day; this is a long."
    short = gate_pack.build("SHORT", "NVDA", now=NOW, sources=src)
    assert "gate:NVDA:read:mj-d" in _ids(short) and "gate:NVDA:read:conflict" not in _ids(short)
    tomorrow = datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)
    later = gate_pack.build("LONG", "NVDA", now=tomorrow, sources=src)
    assert "gate:NVDA:read:none" in _ids(later) and "gate:NVDA:read:conflict" not in _ids(later)


def test_read_words_attach_the_reads_pack():
    for question in ("what was my read this morning", "what did I say the market would do",
                     "was I right about SPY yesterday", "my call on SPY", "how are my predictions doing"):
        names = [request.name for request in attach.plan_attachments(question, set(), NOW)]
        assert "reads_pack" in names, question
    assert "reads_pack" not in [r.name for r in attach.plan_attachments("give me a read on NVDA", set(), NOW)]
