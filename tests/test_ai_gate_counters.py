"""P2 item 4: the five AI phase gates, on the page named after them.

Every counter already existed and none had a surface. `digest_gate_state` and
`enrichment.gate_state` are functions nothing rendered; `synthesis.gate_state`
needs a count only its caller has; the policy draft states its window inside a
prose `notes` sentence; the evidence report prints its window into a Markdown
file in the report store. So "why is the weekly synthesis only scaffolding?"
had no answer on the A.I. Summary page.

The rule these tests exist to hold: every number is READ from the source that
owns it, never recomputed and never hardcoded, and a source that cannot be read
says so rather than showing a zero - a blank cell reads as zero, and zero is a
claim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ai_jobs import gate_counters  # noqa: E402


# ==========================================================================
# the counters
# ==========================================================================
def test_every_gate_is_reported_in_reading_order():
    """Fail-before-fix: the module does not exist."""
    keys = [counter.key for counter in gate_counters.gate_counters()]
    assert keys == ["digest", "enrichment", "synthesis", "policy_draft", "evidence"]


def test_the_digest_counter_reads_the_function_that_owns_it(monkeypatch, tmp_path):
    from ai_jobs import digest

    monkeypatch.setattr(
        digest,
        "digest_gate_state",
        lambda root: {
            "sessions_collected": 6,
            "sessions_required": 10,
            "window_met": False,
            "statement": "DIGEST GATE NOT MET: 6 of 10 session fact packs exist.",
        },
    )
    counter = gate_counters._digest_counter(tmp_path)

    assert (counter.have, counter.need, counter.met) == (6, 10, False)
    assert counter.text() == "Digest 6/10"
    assert "NOT MET" in counter.detail


def test_a_met_gate_says_met_rather_than_only_a_ratio(monkeypatch, tmp_path):
    from ai_jobs import digest

    monkeypatch.setattr(
        digest,
        "digest_gate_state",
        lambda root: {
            "sessions_collected": 12,
            "sessions_required": 10,
            "window_met": True,
            "statement": "Digest collection window met by count.",
        },
    )
    assert gate_counters._digest_counter(tmp_path).text() == "Digest met (12/10)"


def test_an_unreadable_source_says_unavailable_and_never_zero(monkeypatch, tmp_path):
    """A blank cell reads as zero, and zero is a claim about the trader's
    collection window that nobody measured."""
    from ai_jobs import digest

    def boom(root):
        raise OSError("the digest store is gone")

    monkeypatch.setattr(digest, "digest_gate_state", boom)
    counter = gate_counters._digest_counter(tmp_path)

    assert counter.have is None and counter.need is None
    assert counter.readable is False
    assert counter.met is False
    assert counter.text() == "Digest unavailable"
    assert "the digest store is gone" in counter.detail


def test_the_synthesis_counter_counts_the_way_the_job_counts(monkeypatch):
    """Through `_read_cohort` + `graded_sessions`, the two functions
    `run_weekly_synthesis` uses. A second counting rule here could disagree
    with the document it is reporting on."""
    from ai_jobs import synthesis

    seen: list[str] = []

    def fake_read(name, unavailable):
        seen.append(name)
        return []

    monkeypatch.setattr(synthesis, "_read_cohort", fake_read)
    monkeypatch.setattr(synthesis, "graded_sessions", lambda v, l: 2)

    counter = gate_counters._synthesis_counter()
    assert seen == ["veto", "like"]
    assert counter.have == 2
    assert counter.need == synthesis.REQUIRED_GRADED_SESSIONS
    assert counter.met is False


def test_an_unavailable_cohort_is_named_in_the_detail(monkeypatch):
    from ai_jobs import synthesis

    def fake_read(name, unavailable):
        unavailable[f"{name} cohort"] = "file missing"
        return []

    monkeypatch.setattr(synthesis, "_read_cohort", fake_read)
    counter = gate_counters._synthesis_counter()
    assert "veto cohort (file missing)" in counter.detail


def test_the_policy_draft_counter_reads_the_published_notes(tmp_path):
    """Read from the file the desk actually wrote, not recomputed: a
    recomputed number could be right while the file says something else, and
    the file is what the model was handed."""
    draft = tmp_path / "review_policy_draft.json"
    draft.write_text(
        json.dumps(
            {
                "notes": (
                    "POLICY DRAFT WINDOW NOT MET. 5 of 10 drafted session(s). "
                    "This file is a DRAFT and is not authoritative."
                )
            }
        ),
        encoding="utf-8",
    )
    counter = gate_counters._draft_counter(draft)

    assert (counter.have, counter.need, counter.met) == (5, 10, False)
    assert counter.text() == "Policy draft 5/10"


def test_a_missing_policy_draft_is_stated_not_counted(tmp_path):
    counter = gate_counters._draft_counter(tmp_path / "nothing.json")
    assert counter.readable is False
    assert "no draft on disk" in counter.detail


def test_a_notes_sentence_with_no_counts_is_absent_rather_than_guessed(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"notes": "Something with no ratio in it."}), encoding="utf-8")
    counter = gate_counters._draft_counter(draft)
    assert counter.have is None and counter.need is None


def test_the_evidence_counter_reads_the_published_window_block(tmp_path):
    report = tmp_path / "evidence_report.json"
    report.write_text(
        json.dumps(
            {
                "window": {
                    "sessions_collected": 6,
                    "sessions_required": 10,
                    "window_met": False,
                    "statement": "COLLECTION WINDOW NOT MET.",
                }
            }
        ),
        encoding="utf-8",
    )
    counter = gate_counters._evidence_counter(report)

    assert (counter.have, counter.need, counter.met) == (6, 10, False)
    assert counter.text() == "Evidence window 6/10"


def test_a_missing_evidence_report_is_stated_not_counted(tmp_path):
    counter = gate_counters._evidence_counter(tmp_path / "nothing.json")
    assert counter.readable is False
    assert "no report on disk" in counter.detail


def test_the_strip_and_its_tooltip_carry_every_gate():
    counters = [
        gate_counters.GateCounter("a", "Digest", 6, 10, False, "digest statement"),
        gate_counters.GateCounter("b", "Enrichment", 6, 10, False, ""),
    ]
    assert gate_counters.strip_text(counters) == "Digest 6/10 · Enrichment 6/10"
    tooltip = gate_counters.strip_tooltip(counters)
    assert "Digest: digest statement" in tooltip
    # A gate with nothing published says so rather than showing a blank line.
    assert "Enrichment: No statement published." in tooltip


def test_gate_counters_never_raises(monkeypatch):
    """A gate is a report on a run, never part of one."""
    from ai_jobs import digest, enrichment, synthesis

    def boom(*args, **kwargs):
        raise RuntimeError("everything is broken")

    monkeypatch.setattr(digest, "digest_gate_state", boom)
    monkeypatch.setattr(enrichment, "gate_state", boom)
    monkeypatch.setattr(synthesis, "_read_cohort", boom)

    counters = gate_counters.gate_counters()
    assert len(counters) == 5
    assert all(not counter.readable for counter in counters[:3])
    assert gate_counters.strip_text(counters)


# ==========================================================================
# the panel strip
# ==========================================================================
@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _settle(predicate, timeout=10.0):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def test_the_panel_reads_the_gates_off_the_qt_thread(qapp, monkeypatch):
    """Five file reads - a directory listing, two CSVs and two JSON documents -
    none of which belongs on the Qt thread."""
    import threading

    from ui.panels.ai_summary_panel import AiSummaryPanel

    seen: list[int] = []

    def recording():
        seen.append(threading.get_ident())
        return {"counters": [], "text": "Digest 6/10", "tooltip": "because"}

    monkeypatch.setattr(gate_counters, "counters_payload", recording)

    panel = AiSummaryPanel()
    try:
        assert _settle(lambda: panel.gate_strip.text() == "Digest 6/10")
        assert seen and seen[0] != threading.main_thread().ident
        assert panel.gate_strip.toolTip() == "because"
    finally:
        panel.shutdown()
        panel.deleteLater()


def test_a_failed_gate_read_states_it_on_the_strip(qapp, monkeypatch):
    from ui.panels.ai_summary_panel import AiSummaryPanel

    def boom():
        raise RuntimeError("gates unreadable")

    monkeypatch.setattr(gate_counters, "counters_payload", boom)

    panel = AiSummaryPanel()
    try:
        assert _settle(lambda: "unavailable" in panel.gate_strip.text())
        assert "gates unreadable" in panel.gate_strip.text()
    finally:
        panel.shutdown()
        panel.deleteLater()
