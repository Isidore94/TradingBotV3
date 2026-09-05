"""The AI phase gates, in one read (P2 item 4).

Five counters already exist and every one of them is somewhere the trader does
not look: `digest.digest_gate_state` and `enrichment.gate_state` are functions
nothing renders, `synthesis.gate_state` needs a count only its caller has,
`review_policy_draft.json` states its own window inside a `notes` sentence, and
the evidence report prints its window line into a Markdown file in the report
store. So the A.I. Summary page - the one surface named after all of this -
showed none of them, and "why is the weekly synthesis only scaffolding?" had no
answer on screen.

This module answers it. Every number is READ from the source that owns it; not
one is recomputed here, and none is hardcoded. A source that cannot be read
yields a counter whose ``detail`` says why, because a blank cell reads as zero
and zero is a claim.

Pure and Qt-free on purpose: the panel calls :func:`gate_counters` once, on a
worker thread, and renders the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateCounter:
    """One gate: where it stands, and whether that is a reading or a failure."""

    key: str
    label: str
    have: int | None = None
    need: int | None = None
    met: bool = False
    detail: str = ""

    @property
    def readable(self) -> bool:
        return self.have is not None and self.need is not None

    def text(self) -> str:
        """The compact form for the strip. Never a fabricated zero."""
        if not self.readable:
            return f"{self.label} unavailable"
        if self.met:
            return f"{self.label} met ({self.have}/{self.need})"
        return f"{self.label} {self.have}/{self.need}"


def _digest_counter(root: Path | None = None) -> GateCounter:
    """Phase 1's counter. `root` is injectable so a desk with no AI store
    configured - which raises before the gate function is even reached - can be
    told apart in a test from a store that exists and is short of sessions."""
    try:
        from ai_jobs import digest, store

        target = Path(root) if root is not None else store.digests_dir(create=False)
        state = digest.digest_gate_state(target)
    except Exception as exc:  # noqa: BLE001
        return GateCounter("digest", "Digest", detail=f"digest store unreadable: {exc}")
    return GateCounter(
        "digest",
        "Digest",
        have=int(state.get("sessions_collected") or 0),
        need=int(state.get("sessions_required") or 0),
        met=bool(state.get("window_met")),
        detail=str(state.get("statement") or ""),
    )


def _enrichment_counter() -> GateCounter:
    """Phase 2's gate, which IS the digest gate - `enrichment.gate_state`
    delegates to it. Reported separately anyway, because "enrichment is gated"
    is the fact the trader is looking for and inferring it from the digest
    line is exactly the inference a counter exists to remove.

    **`met` reads `gate_met`, not `window_met`** (Q4, 2026-09-04). Since the
    audit half became real, the window being met is no longer what decides
    whether this slot runs - so a counter reporting `window_met` would say
    "Enrichment met" on the exact nights the slot refuses. `window_met` is the
    fallback for a caller that predates the key.
    """
    try:
        from ai_jobs import enrichment

        state = enrichment.gate_state()
    except Exception as exc:  # noqa: BLE001
        return GateCounter("enrichment", "Enrichment", detail=f"unreadable: {exc}")
    met = state.get("gate_met")
    if met is None:
        met = state.get("window_met")
    return GateCounter(
        "enrichment",
        "Enrichment",
        have=int(state.get("sessions_collected") or 0),
        need=int(state.get("sessions_required") or 0),
        met=bool(met),
        detail=str(state.get("statement") or ""),
    )


def _synthesis_counter() -> GateCounter:
    """Counted the way `run_weekly_synthesis` counts it, through the same two
    functions - a second counting rule here could disagree with the job."""
    try:
        from ai_jobs import synthesis

        unavailable: dict[str, str] = {}
        veto_rows = synthesis._read_cohort("veto", unavailable)
        like_rows = synthesis._read_cohort("like", unavailable)
        state = synthesis.gate_state(synthesis.graded_sessions(veto_rows, like_rows))
    except Exception as exc:  # noqa: BLE001
        return GateCounter("synthesis", "Weekly synthesis", detail=f"unreadable: {exc}")
    detail = str(state.get("statement") or "")
    if unavailable:
        detail += " Cohorts unavailable: " + "; ".join(
            f"{name} ({why})" for name, why in sorted(unavailable.items())
        )
    return GateCounter(
        "synthesis",
        "Weekly synthesis",
        have=int(state.get("sessions_graded") or 0),
        need=int(state.get("sessions_required") or 0),
        met=bool(state.get("window_met")),
        detail=detail,
    )


def _draft_counter(path: Path | None = None) -> GateCounter:
    """READ from the published draft's own `notes`, not recomputed.

    The point of this line is what the trader's desk actually wrote last night.
    A recomputed number could be right while the file on disk says something
    else, and the file is what the AI was handed.
    """
    try:
        import project_paths

        target = Path(path) if path is not None else Path(project_paths.REVIEW_POLICY_DRAFT_FILE)
        payload = json.loads(target.read_text(encoding="utf-8"))
        notes = str(payload.get("notes") or "")
    except Exception as exc:  # noqa: BLE001
        return GateCounter("policy_draft", "Policy draft", detail=f"no draft on disk: {exc}")
    have, need = _counts_in(notes)
    return GateCounter(
        "policy_draft",
        "Policy draft",
        have=have,
        need=need,
        met=("NOT MET" not in notes.upper()) if notes else False,
        detail=notes,
    )


def _evidence_counter(path: Path | None = None) -> GateCounter:
    """READ from the published evidence report's own `window` block."""
    try:
        if path is not None:
            target = Path(path)
        else:
            from ai_jobs import evidence_report

            target = Path(evidence_report._default_report_dir()) / "evidence_report.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        window = payload.get("window") or {}
    except Exception as exc:  # noqa: BLE001
        return GateCounter("evidence", "Evidence window", detail=f"no report on disk: {exc}")
    return GateCounter(
        "evidence",
        "Evidence window",
        have=int(window.get("sessions_collected") or 0),
        need=int(window.get("sessions_required") or 0),
        met=bool(window.get("window_met")),
        detail=str(window.get("statement") or ""),
    )


def _counts_in(text: str) -> tuple[int | None, int | None]:
    """"5 of 10 drafted session(s)" -> (5, 10). Absent rather than guessed.

    The draft states its window in prose because that sentence is also what the
    model is handed. Parsing it is the honest way to report the SAME number;
    inventing a parallel counter here is how two surfaces start disagreeing.
    """
    import re

    match = re.search(r"(\d+)\s+of\s+(\d+)", str(text or ""))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def gate_counters() -> list[GateCounter]:
    """Every gate, in reading order. Never raises: a gate is a report, not a run."""
    return [
        _digest_counter(),
        _enrichment_counter(),
        _synthesis_counter(),
        _draft_counter(),
        _evidence_counter(),
    ]


def strip_text(counters: list[GateCounter]) -> str:
    """The one-line strip. Middle dots, in the order given."""
    return " · ".join(counter.text() for counter in counters)


def strip_tooltip(counters: list[GateCounter]) -> str:
    """Each gate's own statement, so the strip is a summary of something the
    reader can open rather than five numbers with no provenance."""
    lines: list[str] = []
    for counter in counters:
        detail = counter.detail.strip() or "No statement published."
        lines.append(f"{counter.label}: {detail}")
    return "\n\n".join(lines)


def counters_payload() -> dict[str, Any]:
    """What the panel's worker hands back to the Qt thread."""
    counters = gate_counters()
    return {
        "counters": counters,
        "text": strip_text(counters),
        "tooltip": strip_tooltip(counters),
    }
