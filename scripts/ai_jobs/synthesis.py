"""The weekly trader-judgement synthesis — machinery BUILT, runs GATED (W5).

`docs/LOCAL_AI_AUTOMATION_PLAN.md` §7.3 listed this under "What is NOT built",
with two things already decided and one thing missing. Decided: the **cadence**
(weekly, on the weekend surface — recorded against R8 in `plan.md`) and the
**gate** (two weeks of graded cohort rows). Missing: authorization. The trader
gave it on 2026-08-24, for the MACHINERY only, on the R10.I scaffolding pattern:
build it now, run it gated, and until the gate passes let it produce
deterministic scaffolding that says so on its own first line rather than a
finding.

**What the gate actually counts.** Two weeks means two weeks of forward
evidence, so it counts SESSIONS in which at least one graded row has a matured
horizon — pooled across the veto and LIKE cohorts, because they are the two
halves of one judgement and reading either alone gives the flattering half.
Counting rows instead would let one busy afternoon of vetoes clear a gate whose
whole purpose is waiting for the market to answer them. With the first cohort
day at 2026-08-20, ten graded sessions land in early September.

**Below the gate, no model is called at all.** §7.2's reason for keeping
`trader_judgement` out of the nightly slate applies with more force here: an
unattended read over a stream still filling narrates "too early" every time
until a reader stops looking. So an unmet gate produces the deterministic pack
and the NOT-MET statement, and asks nobody anything.

**Not nightly.** Absent from `default_slots()`; it lives in `optional_slots()`
and is reached by `run_ai_jobs.py --weekly-synthesis`, constructed per call the
way `--scopes` is, so it cannot become nightly by being set once.

**Not frontier.** Medium tier or nothing (§6.4a D7). Phase 5's frontier
synthesis pass is a separate thing and is **not authorized**.

**Not a control signal.** Nothing here may reach a detector, a score, an alert,
a watchlist, Focus, the review queue or `review_policy.json`. Tests walk this
module's AST rather than trusting this paragraph.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_log = logging.getLogger(__name__)

#: Schema NAME (R10 ground rule 5). A changed meaning is a new name.
SYNTHESIS_SCHEMA = "weekly_synthesis_facts_v1"
NARRATION_SCHEMA = "weekly_synthesis_narration_v1"

#: Two weeks of TRADING, measured in sessions that produced a matured graded row.
REQUIRED_GRADED_SESSIONS = 10

#: Horizons the cohort files carry, weakest-evidence last.
HORIZONS = ("h1", "h3", "h5", "h10")

#: Cells are (cohort x side x horizon) and grow with the vocabulary. Capped, and
#: what the cap drops is counted and printed - a silent top-N reads as "that was
#: all of it".
MAX_CELLS = 40

FACT_PACK_SOURCE_ID = "synthesis.facts"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded_no_narrative"

#: Printed verbatim, first, until the gate passes. The wording matters: a reader
#: who skims must not be able to mistake it for a hedge on a real finding.
GATE_NOT_MET_PREFIX = "SYNTHESIS GATE NOT MET."

GATE_MET_STATEMENT = (
    "Gate met: two weeks of graded cohort sessions exist. Every figure is "
    "still labelled `discovery` - the window was not declared in advance, and a "
    "large post-hoc sample is a large discovery, never a confirmation."
)


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _matured(row: Mapping[str, Any]) -> bool:
    """Did any horizon on this row actually grade?

    `matured_horizons` is a LIST, not a count: `human_focus_tracking` writes
    `",".join(matured)`, so the values on disk are `"1"`, `"1,3"`, `"1,3,5"`.

    This used to be `int(float(value)) > 0`, which RAISES on every one of those
    except a bare `"1"` - and the raise was swallowed as "not matured". So the
    counter saw only rows with exactly ONE matured horizon, and it went DOWN as
    evidence accrued: a row whose value grew from `"1"` to `"1,3"` stopped
    parsing and left the count. That is exactly what the desk showed - 2 of 10
    on 2026-09-01, 1 of 10 on 2026-09-02 - while the files held 176 matured veto
    rows and 53 matured like rows across FOUR distinct sessions.

    A predicate that can flip to False as more evidence arrives is wrong by
    construction, whatever number it happens to produce.
    """
    raw = row.get("matured_horizons")
    if raw is None:
        return False
    return any(
        part.strip() and part.strip() != "0"
        for part in str(raw).split(",")
    )


def graded_sessions(veto_rows: Sequence[Mapping[str, Any]],
                    like_rows: Sequence[Mapping[str, Any]]) -> int:
    """Sessions in which at least one cohort row actually GRADED.

    Not rows, and not calendar days. A pick registered is not a pick graded, and
    a day on which nothing matured taught nothing.
    """
    sessions: set[str] = set()
    for row in list(veto_rows or []) + list(like_rows or []):
        if not _matured(row):
            continue
        day = str(row.get("trade_date") or "")[:10]
        if day:
            sessions.add(day)
    return len(sessions)


def gate_state(sessions: int) -> dict[str, Any]:
    graded = max(0, int(sessions or 0))
    met = graded >= REQUIRED_GRADED_SESSIONS
    return {
        "sessions_graded": graded,
        "sessions_required": REQUIRED_GRADED_SESSIONS,
        "window_met": met,
        "statement": GATE_MET_STATEMENT if met else (
            f"{GATE_NOT_MET_PREFIX} {graded} of {REQUIRED_GRADED_SESSIONS} graded "
            "cohort session(s) exist. This document is deterministic scaffolding, "
            "not a finding: every figure is labelled `discovery`, carries its n, "
            "and must not be used to promote, demote, or change anything. No "
            "model was asked to narrate it."
        ),
    }


# ---------------------------------------------------------------------------
# the fact pack
# ---------------------------------------------------------------------------


def _cohort_of(row: Mapping[str, Any]) -> str:
    return str(row.get("source") or "").strip() or "unstated"


def _side_of(row: Mapping[str, Any]) -> str:
    side = str(row.get("side") or "").strip().upper()
    return side if side in {"LONG", "SHORT"} else "UNKNOWN"


def _cells(rows: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    """One `evidence_stats` summary per (cohort, side, horizon).

    Compacted to the fields a reader uses, for the same reason the digest
    compacts its slices: the full summary repeats conventions that do not change
    between cells, and a document nobody can hold is not more honest for it. The
    contract itself is stated once, at the top of the pack.
    """
    import evidence_stats

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows or ():
        for horizon in HORIZONS:
            raw = row.get(f"{horizon}_return")
            if raw is None or str(raw).strip() == "":
                continue
            grouped.setdefault((_cohort_of(row), _side_of(row), horizon), []).append(row)

    built: list[dict[str, Any]] = []
    for (cohort, side, horizon), members in grouped.items():
        values = []
        symbols = []
        sessions = []
        for row in members:
            try:
                values.append(float(row.get(f"{horizon}_return")))
            except (TypeError, ValueError):
                continue
            symbols.append(str(row.get("symbol") or "").upper())
            sessions.append(str(row.get("trade_date") or "")[:10])
        if not values:
            continue
        summary = evidence_stats.summarize(values, symbols=symbols, sessions=sessions)
        raw = summary.get("raw") or {}
        boot = summary.get("bootstrap") or {}
        concentration = (summary.get("concentration") or {}).get("by_symbol") or {}
        built.append(
            {
                "family": family,
                "cohort": cohort,
                "side": side,
                "horizon": horizon,
                "schema": summary.get("schema"),
                "n": summary.get("n"),
                "symbols": (summary.get("counts") or {}).get("symbols"),
                "sessions": (summary.get("counts") or {}).get("sessions"),
                "mean": raw.get("mean"),
                "median": raw.get("median"),
                "trimmed_mean": raw.get("trimmed_mean"),
                "p10": raw.get("p10"),
                "p90": raw.get("p90"),
                "profit_factor": (summary.get("profit_factor") or {}).get("value"),
                "top_symbol_share": concentration.get("top_share"),
                "ci_low": boot.get("low") if boot.get("measured") else None,
                "ci_high": boot.get("high") if boot.get("measured") else None,
                "ci_basis": (
                    str(boot.get("interval")) if boot.get("measured")
                    else f"unmeasured: {boot.get('reason', '')}"
                ),
                "meets_n_floor": summary.get("meets_n_floor"),
                "n_floor": summary.get("n_floor"),
                "evidence_label": summary.get("evidence_label"),
            }
        )
    return built


def build_fact_pack(
    *,
    veto_rows: Sequence[Mapping[str, Any]] = (),
    like_rows: Sequence[Mapping[str, Any]] = (),
    digest_root: Path | None = None,
    since: str = "",
    until: str = "",
    unavailable: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One deterministic rollup over both graded cohorts. No model is called."""
    import evidence_stats

    moment = _now(now)
    state = gate_state(graded_sessions(veto_rows, like_rows))
    cells = _cells(veto_rows, "veto") + _cells(like_rows, "like")
    cells.sort(key=lambda cell: (-int(cell.get("n") or 0), cell["family"], cell["cohort"],
                                 cell["side"], cell["horizon"]))
    kept, cut = cells[:MAX_CELLS], cells[MAX_CELLS:]

    return {
        "schema": SYNTHESIS_SCHEMA,
        "generated_at": moment.isoformat(timespec="seconds"),
        # First key, first line rendered, and the one a skimming reader sees.
        "gate": state,
        "evidence_label": evidence_stats.LABEL_DISCOVERY,
        "statistics_contract": {
            "module": "evidence_stats",
            "schema": evidence_stats.SUMMARY_SCHEMA,
            "n_floor": evidence_stats.MIN_REPORTABLE_N,
            "n_floor_note": "necessary, never sufficient",
            "profit_factor_convention": evidence_stats.PROFIT_FACTOR_CONVENTION,
        },
        "window": {"since": since, "until": until},
        "cohorts": {
            "veto_rows": len(veto_rows or ()),
            "like_rows": len(like_rows or ()),
            "cells": kept,
            "cells_dropped": {
                "cells": len(cut),
                "events": sum(int(cell.get("n") or 0) for cell in cut),
                "basis": (
                    f"kept the {MAX_CELLS} cells with the largest n; what is listed "
                    "here is what that cap dropped, so a reader never mistakes the "
                    "table for the whole record"
                ),
            },
            "reading_note": (
                "Veto returns are side-adjusted, so POSITIVE means the pick you "
                "REJECTED would have worked. LIKE returns read the opposite way: "
                "positive means the pick you endorsed did. The two are the halves "
                "of one judgement and reading either alone gives the flattering "
                "half. A blank horizon has not matured and is absent, never zero."
            ),
        },
        "digest": _digest_block(digest_root, since=since, until=until),
        "unavailable": {str(k): str(v) for k, v in (unavailable or {}).items()},
        "not_a_control_signal": (
            "Advisory only. Nothing derived from this document may reach a "
            "detector, a score, an alert, a watchlist, Focus, the review queue "
            "or review_policy.json."
        ),
    }


def _digest_block(root: Path | None, *, since: str, until: str) -> dict[str, Any]:
    """The Phase 2 fact packs, once at least one exists.

    An absent digest is an ABSENT MEASUREMENT and says so in words. Rendering a
    zero here would claim the sessions were flat rather than unrecorded.
    """
    if root is None:
        return {"sessions": 0, "note": "no digest root supplied; nothing was read"}
    try:
        from ai_jobs import digest

        rolled = digest.rollup(Path(root), since=since, until=until)
    except Exception as exc:  # noqa: BLE001
        return {"sessions": 0, "note": f"digest packs unreadable: {exc}"}
    if not rolled.get("sessions"):
        return {
            "sessions": 0,
            "note": (
                "no fact pack exists for this window yet - an absent measurement, "
                "not a flat set of sessions"
            ),
        }
    rolled["note"] = (
        "Computed on demand from the daily fact packs (D8); nothing is stored. "
        + str(rolled.get("note") or "")
    )
    return rolled


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_markdown(pack: Mapping[str, Any]) -> str:
    """The human half. The gate statement is the first thing on it."""
    gate = pack.get("gate") or {}
    cohorts = pack.get("cohorts") or {}
    lines = [
        f"# Weekly trader-judgement synthesis - {str(pack.get('generated_at'))[:10]}\n\n",
        f"**{gate.get('statement', '')}**\n\n",
        f"{cohorts.get('reading_note', '')}\n\n",
        f"Every figure is labelled `{pack.get('evidence_label')}`. Statistics come "
        f"from `{(pack.get('statistics_contract') or {}).get('module')}`; "
        f"n >= {(pack.get('statistics_contract') or {}).get('n_floor')} is "
        "necessary, never sufficient.\n\n",
        f"## Cohort cells (n={len(cohorts.get('cells') or [])})\n\n",
    ]
    cells = cohorts.get("cells") or []
    if not cells:
        lines.append("Nothing has graded yet. That is an absent measurement.\n\n")
    else:
        lines.append("| family | cohort | side | horizon | n | mean | median | PF | CI |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for cell in cells:
            interval = (
                f"[{cell.get('ci_low')}, {cell.get('ci_high')}]"
                if cell.get("ci_low") is not None else "unmeasured"
            )
            lines.append(
                f"| {cell.get('family')} | {cell.get('cohort')} | {cell.get('side')} | "
                f"{cell.get('horizon')} | {cell.get('n')} | {cell.get('mean')} | "
                f"{cell.get('median')} | {cell.get('profit_factor')} | {interval} |\n"
            )
        lines.append("\n")
    dropped = cohorts.get("cells_dropped") or {}
    if dropped.get("cells"):
        lines.append(
            f"{dropped['cells']} further cell(s) holding {dropped['events']} "
            f"measurement(s) are not shown: {dropped.get('basis')}.\n\n"
        )
    digest_block = pack.get("digest") or {}
    lines.append(
        f"## Daily digest over the window\n\n{digest_block.get('note', '')}\n\n"
    )
    missing = pack.get("unavailable") or {}
    if missing:
        named = ", ".join(f"{name} ({reason})" for name, reason in sorted(missing.items()))
        lines.append(
            f"**INCOMPLETE**: {len(missing)} source(s) could not be read: {named}.\n\n"
        )
    lines.append(f"{pack.get('not_a_control_signal')}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# narration (medium tier or nothing)
# ---------------------------------------------------------------------------


def narration_evidence_package(pack: Mapping[str, Any]) -> dict[str, Any]:
    """A package holding the fact pack and NOTHING else."""
    import hashlib

    encoded = json.dumps(pack, sort_keys=True, default=str).encode("utf-8")
    source = {
        "source_id": FACT_PACK_SOURCE_ID,
        "label": "Deterministic weekly synthesis fact pack",
        "status": "available",
        "observed_at": pack.get("generated_at"),
        "content_through": str(pack.get("generated_at"))[:10],
        "content_through_basis": "the moment the rollup was computed",
        "session_date": str(pack.get("generated_at"))[:10],
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "truncated": False,
        "content": dict(pack),
    }
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": pack.get("generated_at"),
        "session_date": str(pack.get("generated_at"))[:10],
        "selected_scopes": ["weekly_synthesis"],
        "scope_labels": ["Deterministic weekly trader-judgement rollup"],
        "source_count": 1,
        "sources": [source],
        "coverage": {
            "counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0},
            "note": (
                "Every number here was computed by code. Do not compute new ones, "
                "and do not cite any source that is not listed."
            ),
        },
        "safety_contract": {
            "purpose": "advisory narration of an already-complete rollup",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Veto returns are side-adjusted: positive means the REJECTED pick "
            "would have worked. LIKE returns read the opposite way.",
            "Every figure is DISCOVERY. The window was not declared in advance, "
            "so nothing here confirms anything.",
            "A blank horizon has not matured. It is absent, never zero.",
        ],
    }
    import hashlib as _hashlib

    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = _hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def _narrate(*, pack: Mapping[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Medium tier, over the fact pack only. Raises on any failure.

    The frontier synthesis pass is Phase 5 and is **not authorized**; nothing
    here selects a provider other than the local endpoint.
    """
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError(
            "local AI provider is not configured (ai_local_endpoint_url unset); "
            "the deterministic rollup stands on its own"
        )
    package = narration_evidence_package(pack)
    result = ai_summary.request_ai_summary(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=package,
        timeout_seconds=900,
    )
    return {
        "schema": NARRATION_SCHEMA,
        "generated_at": _now(now).isoformat(timespec="seconds"),
        "facts_sha256": package["sources"][0]["sha256"],
        "facts_package_id": package["package_id"],
        "model": result.get("model", ""),
        "narration": result.get("summary") or {},
        "note": (
            "Narration only, over a rollup whose every number was computed by "
            "code. Advisory: it changes nothing the desk decides."
        ),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _publish(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _superseding(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}.{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def run_weekly_synthesis(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    veto_rows: Sequence[Mapping[str, Any]] | None = None,
    like_rows: Sequence[Mapping[str, Any]] | None = None,
    digest_root: Path | None = None,
    narrate: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """Roll both graded cohorts up, then narrate ONLY if the gate has passed."""
    moment = _now(now)
    unavailable: dict[str, str] = {}

    if veto_rows is None:
        veto_rows = _read_cohort("veto", unavailable)
    if like_rows is None:
        like_rows = _read_cohort("like", unavailable)

    try:
        target = Path(root) if root is not None else _default_root()
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_FAILED, "model": "",
                "reason": f"AI store unavailable: {exc}", "outputs": []}

    pack = build_fact_pack(
        veto_rows=veto_rows,
        like_rows=like_rows,
        digest_root=digest_root if digest_root is not None else _digest_root_or_none(),
        since=_window_start(veto_rows, like_rows),
        until=session_date or moment.date().isoformat(),
        unavailable=unavailable,
        now=moment,
    )

    stamp = moment.date().isoformat()
    try:
        json_path = _superseding(Path(target) / str(moment.year) / f"{stamp}.json")
        outputs = [str(_publish(json_path, json.dumps(pack, indent=1, sort_keys=True,
                                                      default=str) + "\n"))]
        outputs.append(str(_publish(json_path.with_suffix(".md"), render_markdown(pack))))
    except OSError as exc:
        return {"status": STATUS_FAILED, "model": "",
                "reason": f"synthesis could not be published: {exc}", "outputs": []}

    gate = pack["gate"]
    base = (
        f"{gate['sessions_graded']} of {gate['sessions_required']} graded cohort "
        f"session(s); {len(pack['cohorts']['cells'])} cell(s)"
        + (f"; {len(unavailable)} source(s) unreadable" if unavailable else "")
    )
    if not gate["window_met"]:
        # Nothing is asked of a model below the gate. There is nothing to
        # narrate, and a narration over a stream still filling says "too early"
        # until a reader stops looking.
        return {
            "status": STATUS_OK,
            "model": "",
            "reason": f"synthesis gate not met - scaffolding only, no model called; {base}",
            "outputs": outputs,
        }
    if not narrate:
        return {"status": STATUS_OK, "model": "", "reason": base, "outputs": outputs}

    try:
        narration = _narrate(pack=pack, now=moment)
    except Exception as exc:  # noqa: BLE001
        _log.info("Weekly synthesis: narration unavailable (%s); the rollup stands.", exc)
        return {"status": STATUS_DEGRADED, "model": "",
                "reason": f"{base}; narration absent: {exc}", "outputs": outputs}
    try:
        outputs.append(str(_publish(
            _superseding(json_path.with_name(f"{stamp}.narration.json")),
            json.dumps(narration, indent=1, sort_keys=True, default=str) + "\n",
        )))
    except OSError as exc:
        return {"status": STATUS_DEGRADED, "model": str(narration.get("model") or ""),
                "reason": f"{base}; narration could not be published: {exc}",
                "outputs": outputs}
    return {"status": STATUS_OK, "model": str(narration.get("model") or ""),
            "reason": f"{base}; narrated", "outputs": outputs}


def _default_root() -> Path:
    from ai_jobs import store

    return store.retros_dir()


def _digest_root_or_none() -> Path | None:
    try:
        from ai_jobs import store

        return store.digests_dir(create=False)
    except Exception:  # noqa: BLE001 - no store configured is a normal state
        return None


def _read_cohort(family: str, unavailable: dict[str, str]) -> list[dict[str, Any]]:
    import csv

    try:
        import project_paths

        path = Path(
            project_paths.VETO_COHORT_OUTCOMES_FILE if family == "veto"
            else project_paths.LIKE_COHORT_OUTCOMES_FILE
        )
        if not path.is_file():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception as exc:  # noqa: BLE001
        unavailable[f"{family} cohort outcomes"] = str(exc)
        return []


def _window_start(veto_rows, like_rows) -> str:
    days = sorted(
        str(row.get("trade_date") or "")[:10]
        for row in list(veto_rows or []) + list(like_rows or [])
        if row.get("trade_date")
    )
    return days[0] if days else ""
