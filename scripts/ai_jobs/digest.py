"""The Daily Digest Ledger — LOCAL-AI Phase 2, built 2026-08-24 (packet W4).

`docs/LOCAL_AI_AUTOMATION_PLAN.md` §3.2 and §6.4a. The 2026-08-08 trader
decision forbade building or freezing any digest schema until six open questions
were answered; they were answered on 2026-08-24 and are recorded in
`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md` §1. This is §6.4a's
design with those answers frozen into it, and the answers travel INSIDE every
pack so a reader six months from now knows the rules it was built under.

**Two artifacts, not one (D1), and that split is the whole design.**

| Artifact | Written by | When the model is down |
|---|---|---|
| `facts/<YYYY>/<YYYY-MM-DD>.json` | code only, zero LLM | **written normally** |
| `narration/<YYYY>/<YYYY-MM-DD>.json` | medium tier, reads the fact pack ONLY | absent |

A missing narration file is a normal state, not a degraded record. The frontier
reducer reads facts; narration is a convenience for the human, disposable and
regenerable (answer 4). And because the narrator is handed the fact pack and
nothing else, its prompt is bounded by the 16 KB cap plus a fixed scaffold — so
the 2026-08-10 truncation failure, where a model handed a sheared prompt produced
confident schema-valid output about evidence it never saw, cannot recur here by
CONSTRUCTION rather than by vigilance.

**Numbers are computed by code, never by the model** (§3.2). Every measured value
carries `{value, n, source_id, selector, as_of}` and `n` is mandatory: the
−0.18R vs +1.01R finding that reordered the Away report was only actionable
because both sample sizes were known.

The six answers, as built:

1. **Winning is BOTH** — R at scenario close AND MFE/MAE, side by side, never
   blended. Close-R is result, MFE/MAE is opportunity (R10 ground rule 12), and
   no field here combines them.
2. **Slices are `env_key` (environment × day-part) × side.** No setup-family
   slice in v1; adding one is a v2 decision.
3. **Shadow-engine output is EXCLUDED.** Champion facts only — a reducer reading
   a challenger beside a champion will treat the challenger as live (plan.md
   sec 7). A test walks this module's AST to keep that true.
4. **Narration is disposable; fact packs are the permanent record.**
5. **16 KB hard cap.** Over-cap FAILS the job and writes nothing, rather than
   truncating — a truncated fact pack is exactly the sheared prompt above.
6. **A non-session writes an EMPTY fact pack**, so the gap is visible. A missing
   file and a quiet day must not look the same.

**Rollups are a read, not a second store** (D8): weekly and monthly views are
computed on demand from the packs, because a derived aggregate store is a second
thing to keep in sync and a second thing to be wrong.

Nothing in this chain may reach a detector, a score, an alert, a watchlist,
Focus, the review queue or `review_policy.json`. It reads and it writes its own
two files.

**The live gate is owed and building it never marks it met**: ten consecutive
session days of digests, with the trader spot-auditing at least three against
raw evidence and finding no fabricated fact. `clean_digest_sessions` counts;
counting is not passing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_log = logging.getLogger(__name__)

#: Schema NAMES (R10 ground rule 5). A changed meaning is a new name.
#: v2 (2026-08-28) hoisted the per-cell/per-row provenance pointer to its block
#: (`_hoist_block_pointer`). Same facts, same provenance, ~30% fewer bytes. It is
#: a NEW NAME because the SHAPE changed: a v1 reader looking for `source_id` on a
#: cell must not silently find nothing. Every v1 pack on disk stays a v1 pack and
#: stays readable, and `clean_digest_sessions` counts by session, not by schema,
#: so the Phase 2 collection window is unaffected by the bump.
FACTS_SCHEMA = "daily_digest_facts_v2"
NARRATION_SCHEMA = "daily_digest_narration_v1"

#: D5. Target and hard cap. 90 packs at the target is well under 1.5 MB, which
#: is the entire point: a trivial context load for a frontier reducer.
FACT_PACK_TARGET_BYTES = 8_192
FACT_PACK_HARD_CAP_BYTES = 16_384

#: Slices are what can grow without bound, so they are capped - and what the cap
#: drops is COUNTED and PRINTED, because a silent top-N reads as "that was all".
#:
#: Sixteen is chosen so the pack fits its cap BY CONSTRUCTION (D5) rather than
#: by truncation, and well inside the 16 KB cap on the busiest measured day. A
#: larger cap here would mean a busy session failing the job and losing its
#: facts entirely, which is the opposite of what over-cap-fails is protecting.
#:
#: MEASURED, not estimated (2026-08-27, 14 slices): this line used to claim the
#: pack lands "near the 8 KB target" and it did not - it rendered at 14,070
#: bytes, 72% of it the outcomes block. The v2 pointer hoist took that down
#: without dropping a single figure. The target is still worth aiming at, but
#: the number that actually matters is the one the target exists to protect:
#: ninety packs as a trivial context load for a frontier reducer, which holds
#: comfortably at the post-hoist size. Cutting real slices to reach 8,192
#: exactly would trade evidence for a round number.
MAX_SLICES = 16

#: Phase 2's exit gate: ten consecutive session days of digests plus a trader
#: spot-audit of at least three against raw evidence.
REQUIRED_CLEAN_SESSIONS = 10

#: The one source id the narrator may cite. It reads the fact pack and nothing
#: else, so there is exactly one.
FACT_PACK_SOURCE_ID = "digest.facts"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
#: Facts written, narration absent. Not "ok": the ledger retries, and the night
#: is honestly recorded as half-done rather than as a healthy one.
STATUS_DEGRADED = "degraded_no_narrative"

#: Frozen into every pack, so the rules travel with the record.
ANSWERS = {
    "winning": (
        "BOTH, side by side: R at scenario close AND MFE/MAE, never blended. "
        "Close-R is the result; MFE/MAE is the opportunity."
    ),
    "slices": "env_key (market environment x day-part) x side. No setup-family slice in v1.",
    "shadow_engines": (
        "Shadow-engine output is excluded; champion facts only, so a reducer "
        "cannot mistake a challenger for a live engine."
    ),
    "retention": "Narration is disposable and regenerable; fact packs are the permanent record.",
    "cap": f"{FACT_PACK_HARD_CAP_BYTES} bytes hard cap. Over-cap fails the job and writes nothing.",
    "non_sessions": "A weekend or holiday writes an EMPTY fact pack so the gap is visible.",
}

#: One session cannot have a session-block interval - there is one block. Said
#: ONCE here rather than repeated as "unmeasured" on every slice row.
ONE_SESSION_NOTE = (
    "This pack covers ONE session, so a session-block interval is unmeasurable "
    "by construction: every figure below is a single day's discovery, and n is "
    "the only thing separating a reading from an anecdote."
)


# ---------------------------------------------------------------------------
# measured values (D2)
# ---------------------------------------------------------------------------


def measured(value: Any, *, n: Any, source_id: str, selector: str, as_of: str) -> dict[str, Any]:
    """One measured value with its provenance. ``n`` is mandatory.

    Raising on a missing ``n`` rather than defaulting it is deliberate: a
    default would make the omission invisible, and an average whose sample size
    nobody knows is not evidence.
    """
    if n is None:
        raise ValueError(
            "a measured value must carry its n; an average without a sample "
            "size cannot be read (LOCAL_AI_AUTOMATION_PLAN sec 6.4a D2)"
        )
    try:
        count = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n must be an integer, got {n!r}") from exc
    number: float | None
    if value is None:
        number = None
    else:
        try:
            number = round(float(value), 4)
        except (TypeError, ValueError):
            number = None
    return {
        "value": number,
        "n": max(0, count),
        "source_id": str(source_id),
        "selector": str(selector),
        "as_of": str(as_of),
    }


# ---------------------------------------------------------------------------
# slice identity (answer 2)
# ---------------------------------------------------------------------------


def env_key_of(row: Mapping[str, Any]) -> str:
    """This row's ``<environment>|<day_part>`` key, READ rather than derived.

    Answer 2 names "the `env_key` R10.A already stamps", and that is exactly
    what this reads. The alert path computes it at registration - the
    environment from live state, the day-part from
    `bounce_bot_lib.learning.time_bucket_for`, the same function the learning
    state keys its own segments by - and carries it in `context_json`.

    **The digest does not re-derive it, for two reasons.** A second copy of the
    day-part cutoffs would let the digest and the learning state disagree about
    what "midday" means, so they would be describing different days. And
    `ai_jobs` is deliberately kept out of live decision modules - a test asserts
    no module in this package imports `bounce_bot*`, `autopilot_core`,
    `master_avwap`, `technical_integrity`, `price_alert` or `d1_level_feed` -
    so importing the learning module to borrow one function would cross a
    boundary that exists to stop the advisory layer reaching into the alert
    path at all.

    A row written before R10.A began stamping carries no key. Its environment
    is still known, so the day-part alone is `unknown`: uncertainty, never a
    guess and never a quiet default to some bucket (plan.md sec 5).
    """
    stamped = str(row.get("env_key") or "").strip()
    if stamped:
        return stamped
    env = str(row.get("market_environment") or "").strip() or "unknown"
    return f"{env}|unknown"


def day_part_of(env_key: Any) -> str:
    """The day-part half of an env_key. Split, never recomputed."""
    text = str(env_key or "").strip()
    if "|" not in text:
        return "unknown"
    return text.split("|", 1)[1].strip() or "unknown"


def _side_of(direction: Any) -> str:
    text = str(direction or "").strip().upper()
    if text.startswith("SHORT"):
        return "SHORT"
    if text.startswith("LONG"):
        return "LONG"
    return "UNKNOWN"


def _numbers(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field)
        if raw is None or raw == "":
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
            continue
        values.append(number)
    return values


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


# ---------------------------------------------------------------------------
# the fact pack
# ---------------------------------------------------------------------------


def build_fact_pack(
    *,
    session_date: str,
    is_session: bool = True,
    finals: Sequence[Mapping[str, Any]] | None = None,
    coverage: Mapping[str, Any] | None = None,
    review_rows: Sequence[Mapping[str, Any]] = (),
    job_rows: Sequence[Mapping[str, Any]] = (),
    unavailable: Mapping[str, str] | None = None,
    supersedes: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """One session's deterministic fact pack. No model is called from here.

    ``finals`` are CHAMPION intraday outcome rows that already cleared the
    outcome store's own exclusions - settled, above the risk floor, and
    entry-claiming (R10.B). Rows that do not claim an entry are not trades and
    are never averaged as one; the caller's coverage block says how many there
    were.
    """
    moment = _now(now)
    rows = [dict(row) for row in (finals or [])] if is_session else []
    missing = {str(name): str(reason) for name, reason in (unavailable or {}).items()}
    as_of = moment.isoformat(timespec="seconds")

    overall, overall_pointer = _hoist_block_pointer(_overall_block(rows, session_date, as_of))
    slices, dropped = _slice_blocks(rows, session_date, as_of)
    slices, slice_pointer = _hoist_slice_pointer(slices, session_date)
    behaviour, behaviour_pointer = _hoist_block_pointer(
        _behaviour_block(review_rows, session_date, as_of)
    )
    operations, operations_pointer = _hoist_block_pointer(
        _operations_block(job_rows, session_date, as_of)
    )

    pack: dict[str, Any] = {
        "schema": FACTS_SCHEMA,
        "session_date": str(session_date or ""),
        "generated_at": as_of,
        "is_session": bool(is_session),
        "empty_reason": (
            ""
            if is_session
            else (
                "not a trading session; this pack is deliberately empty so the "
                "gap in the ledger is visible rather than looking like a "
                "missing file"
            )
        ),
        # The rules this pack was built under, carried with the record.
        "answers": dict(ANSWERS),
        "evidence_label": _discovery_label(),
        "sampling_note": ONE_SESSION_NOTE,
        "outcomes": {
            "pointer": overall_pointer,
            "overall": overall,
            "slice_pointer": slice_pointer,
            "slices": slices,
            "slices_dropped": dropped,
            "slice_pointer_note": (
                "Provenance is carried ONCE per block, not per cell and not per "
                "row: 'pointer' covers every cell under 'overall', and "
                "'slice_pointer' covers every row under 'slices' - its "
                "selector_template rebuilds any row's exact selector from that "
                "row's own env_key and side. Every metric cell still carries its "
                "own value and n, and the metric name is the key. Restating a "
                "query that does not change would spend a fifth of the size cap "
                "on two constants."
            ),
        },
        "behaviour": {"pointer": behaviour_pointer, **behaviour},
        "operations": {"pointer": operations_pointer, **operations},
        "coverage": dict(coverage or {}),
        "unavailable": missing,
        "supersedes": str(supersedes or ""),
    }
    pack["summary"] = _summary(pack)
    return pack


#: Lifted out of every measured cell in a block when they are constant across
#: it (D5 sizing). This is the same argument `slice_pointer_note` already makes
#: one level down -- a query that does not change should not be restated - and
#: the 2026-08-27 pack showed it applies across ROWS too: one `source_id` and
#: one `as_of` were printed 21 times for 21 cells that all shared them, which is
#: a fifth of the pack spent on two constants. Nothing is lost: the pointer is
#: still attached to every number, one level up, and D2's rule that a measured
#: value never travels without its provenance and its n is unchanged.
POINTER_KEYS = ("source_id", "as_of")


def _is_measured_cell(value: Any) -> bool:
    return isinstance(value, Mapping) and all(key in value for key in POINTER_KEYS)


def _hoist_block_pointer(cells: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Lift the constant pointer fields out of a block's measured cells.

    A field is only lifted when EVERY measured cell in the block agrees on it.
    A block that ever mixes two stores or two clocks keeps the field per cell,
    because then it is not a constant and hoisting it would state something
    false.
    """
    measured_cells = [value for value in cells.values() if _is_measured_cell(value)]
    if not measured_cells:
        return dict(cells), {}
    pointer = {}
    for key in POINTER_KEYS:
        values = {str(cell.get(key)) for cell in measured_cells}
        if len(values) == 1:
            pointer[key] = values.pop()
    if not pointer:
        return dict(cells), {}
    trimmed = {
        name: (
            {key: value for key, value in cell.items() if key not in pointer}
            if _is_measured_cell(cell)
            else cell
        )
        for name, cell in cells.items()
    }
    return trimmed, pointer


def _hoist_slice_pointer(
    slices: Sequence[Mapping[str, Any]], session_date: Any
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Lift source_id, as_of and the selector SHAPE off the slice rows.

    A row's selector differs only in the two fields the row already prints, so
    the template plus the row rebuilds it exactly. Fourteen rows on 2026-08-27
    spent roughly 1.2 KB restating that shape.
    """
    rows = [dict(row) for row in slices]
    if not rows:
        return rows, {}
    pointer: dict[str, str] = {}
    for key in POINTER_KEYS:
        values = {str(row.get(key)) for row in rows if key in row}
        if len(values) == 1 and len([row for row in rows if key in row]) == len(rows):
            pointer[key] = values.pop()
    template = (
        f"trade_date={session_date}&env_key={{env_key}}&side={{side}}&usable=true"
    )
    rebuilt_everywhere = all(
        str(row.get("selector") or "")
        == template.format(env_key=row.get("env_key"), side=row.get("side"))
        for row in rows
    )
    if rebuilt_everywhere:
        pointer["selector_template"] = template
    for row in rows:
        for key in pointer:
            if key == "selector_template":
                row.pop("selector", None)
            else:
                row.pop(key, None)
    return rows, pointer


def _discovery_label() -> str:
    try:
        import evidence_stats

        return evidence_stats.LABEL_DISCOVERY
    except Exception:  # pragma: no cover - the module ships beside this one
        return "discovery"


def _overall_block(rows, session_date, as_of) -> dict[str, Any]:
    """The session's two win metrics side by side, plus its full ground-rule-10
    summary on close-R.

    The complete `evidence_stats` summary is carried ONCE, here, rather than on
    every slice: a per-slice copy would multiply the pack past its cap while
    saying the same thing about the same single session.
    """
    close = _numbers(rows, "close_r")
    mfe = _numbers(rows, "mfe_r")
    mae = _numbers(rows, "mae_r")
    stop_exit = _numbers(rows, "r_stop_exit")
    last_measured = _numbers(rows, "r_last_measured")
    selector = f"trade_date={session_date}&event_type=final&usable=true"
    block = {
        # Result.
        "close_r": measured(_mean(close), n=len(close), source_id="outcomes.intraday_finals",
                            selector=selector + "&metric=close_r", as_of=as_of),
        "win_rate_close_r": measured(
            (len([value for value in close if value > 0]) / len(close)) if close else None,
            n=len(close), source_id="outcomes.intraday_finals",
            selector=selector + "&metric=win_rate(close_r>0)", as_of=as_of,
        ),
        # Opportunity. Reported BESIDE the result, never folded into it.
        "mfe_r": measured(_mean(mfe), n=len(mfe), source_id="outcomes.intraday_finals",
                          selector=selector + "&metric=mfe_r", as_of=as_of),
        "mae_r": measured(_mean(mae), n=len(mae), source_id="outcomes.intraday_finals",
                          selector=selector + "&metric=mae_r", as_of=as_of),
        "symbols": measured(
            len({str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")}),
            n=len(rows), source_id="outcomes.intraday_finals",
            selector=selector + "&metric=distinct_symbols", as_of=as_of,
        ),
        # Decision A: the two policies the after-close sweep can measure. Their
        # own n, beside close_r, never blended with it or with each other.
        "stop_exit_r": measured(
            _mean(stop_exit), n=len(stop_exit), source_id="outcomes.intraday_finals",
            selector=selector + "&metric=stop_exit_r", as_of=as_of,
        ),
        "last_measured_r": measured(
            _mean(last_measured), n=len(last_measured), source_id="outcomes.intraday_finals",
            selector=selector + "&metric=last_measured_r", as_of=as_of,
        ),
        "metric_note": (
            "close_r is the RESULT at scenario close; mfe_r/mae_r are the "
            "OPPORTUNITY the path offered. They are reported side by side and "
            "are never blended into one number. stop_exit_r and "
            "last_measured_r are the two frozen exit policies an after-close "
            "sweep CAN measure (Decision A, 2026-08-25); a session finalized by "
            "the sweep has no eod-hold close_r at all, so an n of 0 there beside "
            "a real n here is the honest reading, not a missing number."
        ),
    }
    block["statistics"] = _statistics_summary(close, rows)
    return block


def _statistics_summary(values: Sequence[float], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Ground rule 10's summary on the session's close-R, through the one module."""
    try:
        import evidence_stats

        summary = evidence_stats.summarize(
            values,
            symbols=[str(row.get("symbol") or "") for row in rows],
            sessions=[str(row.get("trade_date") or "") for row in rows],
        )
    except Exception as exc:  # pragma: no cover - the module ships beside this one
        return {"unavailable": str(exc)}
    # Only the parts a reducer reads. The whole summary would spend a quarter of
    # the cap restating conventions that do not change between sessions.
    raw = summary.get("raw") or {}
    concentration = (summary.get("concentration") or {}).get("by_symbol") or {}
    return {
        "schema": summary.get("schema"),
        "n": summary.get("n"),
        "mean": raw.get("mean"),
        "median": raw.get("median"),
        "trimmed_mean": raw.get("trimmed_mean"),
        "p10": raw.get("p10"),
        "p90": raw.get("p90"),
        "profit_factor": (summary.get("profit_factor") or {}).get("value"),
        "top_symbol_share": concentration.get("top_share"),
        "meets_n_floor": summary.get("meets_n_floor"),
        "n_floor": summary.get("n_floor"),
        "n_floor_note": "necessary, never sufficient",
        "evidence_label": summary.get("evidence_label"),
        "interval": "unmeasurable: one session is one block (see sampling_note)",
    }


def _slice_blocks(rows, session_date, as_of) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (env_key_of(row), _side_of(row.get("direction")))
        grouped.setdefault(key, []).append(row)

    built: list[dict[str, Any]] = []
    for (env_key, side), members in grouped.items():
        close = _numbers(members, "close_r")
        mfe = _numbers(members, "mfe_r")
        mae = _numbers(members, "mae_r")
        # ONE pointer per ROW rather than one per metric, and the metric name is
        # the key. D2 says a measured value may never be written without its
        # provenance and its n - both are here - but repeating the same
        # source_id, selector and as_of four times per row would spend a third
        # of the cap restating a query that does not change within the row, and
        # a pack that cannot fit its own session is not more honest for it.
        built.append(
            {
                "env_key": env_key,
                "side": side,
                "events": len(members),
                "symbols": len({str(row.get("symbol") or "").upper() for row in members}),
                "source_id": "outcomes.intraday_finals",
                "selector": f"trade_date={session_date}&env_key={env_key}&side={side}&usable=true",
                "as_of": as_of,
                "close_r": _cell(_mean(close), close),
                "win_rate_close_r": _cell(
                    (len([value for value in close if value > 0]) / len(close)) if close else None,
                    close,
                ),
                "mfe_r": _cell(_mean(mfe), mfe),
                "mae_r": _cell(_mean(mae), mae),
            }
        )
    # Largest n first, then a stable name order, so two runs over identical
    # inputs produce identical bytes.
    built.sort(key=lambda row: (-row["events"], row["env_key"], row["side"]))
    kept = built[:MAX_SLICES]
    cut = built[MAX_SLICES:]
    dropped = {
        "slices": len(cut),
        "events": sum(row["events"] for row in cut),
        "basis": (
            f"kept the {MAX_SLICES} slices with the largest n; what is listed "
            "here is what that cap dropped, so a reader never mistakes the "
            "table for the whole session"
        ),
    }
    return kept, dropped


def _cell(value: Any, sample: Sequence[float]) -> dict[str, Any]:
    """One slice metric: the value and its n, under the row's own pointer.

    Never a bare number. `n` travels with every cell for the same reason it is
    mandatory everywhere else - an average whose sample size nobody knows is
    not evidence - and an unmeasurable cell is `None` with n, never a zero.
    """
    number: float | None
    try:
        number = None if value is None else round(float(value), 4)
    except (TypeError, ValueError):
        number = None
    return {"value": number, "n": len(sample)}


def _behaviour_block(review_rows, session_date, as_of) -> dict[str, Any]:
    """The habits half of the mission. None of it is a market fact."""
    rows = [
        row for row in (review_rows or [])
        if str(row.get("trade_date") or "")[:10] == str(session_date)
    ]
    actions: dict[str, int] = {}
    for row in rows:
        action = str(row.get("action") or "unstated").strip().lower() or "unstated"
        actions[action] = actions.get(action, 0) + 1
    dwells = _numbers(rows, "dwell_ms")
    return {
        "reviewed": measured(len(rows), n=len(rows), source_id="review.alert_review_events",
                             selector=f"trade_date={session_date}", as_of=as_of),
        "by_action": actions,
        "median_dwell_ms": measured(
            sorted(dwells)[len(dwells) // 2] if dwells else None,
            n=len(dwells), source_id="review.alert_review_events",
            selector=f"trade_date={session_date}&metric=median(dwell_ms)", as_of=as_of,
        ),
        "note": (
            "Decisions the trader made about what fired. A day with no reviews "
            "is a day nobody reviewed, not a day nothing fired."
        ),
    }


def _operations_block(job_rows, session_date, as_of) -> dict[str, Any]:
    """Without this an infrastructure week reads as a bad trading week."""
    rows = [
        row for row in (job_rows or [])
        if str(row.get("session_date") or "") == str(session_date)
    ]
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unstated").strip().lower() or "unstated"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "job_rows": measured(len(rows), n=len(rows), source_id="ops.ai_job_ledger",
                             selector=f"session_date={session_date}", as_of=as_of),
        "by_status": statuses,
        "note": (
            "Overnight job outcomes for this session. 2026-08-10 would have read "
            "as 'no setups worth pushing' rather than 'the writer was "
            "unconfigured' without this block."
        ),
    }


def _summary(pack: Mapping[str, Any]) -> str:
    overall = (pack.get("outcomes") or {}).get("overall") or {}
    close = overall.get("close_r") or {}
    mfe = overall.get("mfe_r") or {}
    missing = pack.get("unavailable") or {}
    if not pack.get("is_session"):
        return (
            f"{pack.get('session_date')}: {pack.get('empty_reason')}. "
            "Nothing was measured because nothing traded."
        )
    parts = [
        f"{pack.get('session_date')}: n={close.get('n', 0)} settled entry-claim "
        f"outcome(s); mean close_r {close.get('value')}, mean mfe_r "
        f"{mfe.get('value')} (result and opportunity, side by side, never blended); "
        f"{len((pack.get('outcomes') or {}).get('slices') or [])} slice(s) kept."
    ]
    if missing:
        named = ", ".join(f"{name} ({reason})" for name, reason in sorted(missing.items()))
        parts.append(
            f"{len(missing)} source(s) could not be read, so this pack is "
            f"INCOMPLETE rather than empty: {named}."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# sizing, paths, and the append-only rule
# ---------------------------------------------------------------------------


def render_fact_pack(pack: Mapping[str, Any]) -> str:
    """The canonical bytes. Sorted and compact enough to fit the budget."""
    return json.dumps(pack, indent=1, sort_keys=True, default=str) + "\n"


def fact_pack_bytes(pack: Mapping[str, Any]) -> int:
    return len(render_fact_pack(pack).encode("utf-8"))


def facts_path(root: Path, session_date: str) -> Path:
    return Path(root) / "facts" / str(session_date)[:4] / f"{session_date}.json"


def narration_path(root: Path, session_date: str) -> Path:
    return Path(root) / "narration" / str(session_date)[:4] / f"{session_date}.json"


def superseding_path(path: Path) -> Path:
    """The next free sibling. A pack is never edited (D6).

    A correction is a new file naming what it supersedes, so the history of what
    was believed on the day survives the correction.
    """
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    index = 1
    while True:
        candidate = path.with_name(f"{stem}.{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _publish(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# narration (D1, D5, D7)
# ---------------------------------------------------------------------------


def provenance_ids(pack: Mapping[str, Any]) -> list[str]:
    """Every ``source_id`` the pack prints anywhere inside itself, sorted.

    Walks the built pack rather than listing the ids by hand, so a block added
    later cannot introduce a provenance id the narrator is shown but forbidden
    to name -- which is the exact shape of the 2026-08-25..27 digest failure.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "source_id" and isinstance(value, str) and value.strip():
                    found.add(value.strip())
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(pack)
    return sorted(found)


def narration_evidence_package(pack: Mapping[str, Any]) -> dict[str, Any]:
    """An evidence package holding the fact pack and NOTHING else.

    Built here rather than through `ai_summary.build_evidence_package`, whose
    job is to assemble many raw sources: the entire point of D5 is that this
    narrator sees one bounded document. Reusing the package SHAPE keeps the
    existing validation - a summary may only cite source ids that are present -
    so the narrator cannot cite a store it never saw.
    """
    import hashlib

    aliases = provenance_ids(pack)
    encoded = json.dumps(pack, sort_keys=True, default=str).encode("utf-8")
    source = {
        "source_id": FACT_PACK_SOURCE_ID,
        "label": f"Deterministic fact pack for {pack.get('session_date')}",
        "status": "available",
        "observed_at": pack.get("generated_at"),
        "content_through": pack.get("session_date"),
        "content_through_basis": "the session the pack describes",
        "session_date": pack.get("session_date"),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "truncated": False,
        "content": dict(pack),
    }
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": pack.get("generated_at"),
        "session_date": pack.get("session_date"),
        "selected_scopes": ["daily_digest"],
        "scope_labels": ["Deterministic daily digest fact pack"],
        "source_count": 1,
        "sources": [source],
        # The provenance ids the pack PRINTS on its own cells. The narrator is
        # told to cite exact source_id values and is handed a document full of
        # them, so it cites them; before 2026-08-28 that was rejected as
        # unusable evidence and threw the whole narration away three nights
        # running. They name real stores, they are visible in the one document
        # the narrator was given, and citing one is more informative than
        # citing the pack as a whole -- so they are citable, and nothing that
        # is not in the pack is.
        "citable_aliases": aliases,
        "coverage": {
            "counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0},
            "note": (
                "The narrator reads this fact pack and nothing else. Every "
                "number in it was computed by code; do not compute new ones, "
                "and do not cite any source that is not listed here. You may "
                f"cite '{FACT_PACK_SOURCE_ID}' for anything in the pack, or the "
                "exact source_id printed on the cell you are describing "
                + (f"({', '.join(aliases)})" if aliases else "(none present)")
                + ". Cite nothing else."
            ),
        },
        "safety_contract": {
            "purpose": "advisory narration of an already-complete fact pack",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders"],
        },
        "scope_caveats": [
            "Every figure is one session's DISCOVERY. Do not describe it as a "
            "trend, a confirmation, or evidence about a setup.",
            "close_r and mfe_r/mae_r are result and opportunity. Never combine them.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def _narrate(*, pack: Mapping[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Ask the medium tier to narrate the pack. Raises on any failure.

    Medium tier or nothing (D7): no 27B-class local model loads beside the
    running desk. The caller turns a raise into an absent narration file, which
    is a normal state.
    """
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError(
            "local AI provider is not configured (ai_local_endpoint_url unset); "
            "the fact pack stands on its own"
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
        "session_date": pack.get("session_date"),
        "generated_at": _now(now).isoformat(timespec="seconds"),
        "facts_sha256": package["sources"][0]["sha256"],
        "facts_package_id": package["package_id"],
        "model": result.get("model", ""),
        "narration": result.get("summary") or {},
        "note": (
            "Narration only. Every number it refers to was computed by code in "
            "the fact pack this file names; regenerating it changes nothing on "
            "the record."
        ),
    }


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------


def run_daily_digest(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    is_session: bool | None = None,
    finals: Sequence[Mapping[str, Any]] | None = None,
    narrate: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """One session's digest: facts always, narration when the model answers."""
    moment = _now(now)
    day = str(session_date or date.today().isoformat())
    unavailable: dict[str, str] = {}

    if is_session is None:
        is_session = _is_session(day, unavailable)

    coverage: dict[str, Any] = {}
    if finals is None and is_session:
        finals, coverage = _read_champion_finals(day, unavailable)
    rows = list(finals or [])

    review_rows = _read_rows("alert review events", _read_review_events, unavailable)
    job_rows = _read_rows("ai job ledger", _read_job_rows, unavailable)

    try:
        target_root = Path(root) if root is not None else _default_root()
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_FAILED, "model": "",
                "reason": f"AI store unavailable: {exc}", "outputs": []}

    facts_target = superseding_path(facts_path(target_root, day))
    pack = build_fact_pack(
        session_date=day,
        is_session=is_session,
        finals=rows,
        coverage=coverage,
        review_rows=review_rows,
        job_rows=job_rows,
        unavailable=unavailable,
        supersedes=facts_path(target_root, day).name if facts_target.name != facts_path(target_root, day).name else "",
        now=moment,
    )

    size = fact_pack_bytes(pack)
    if size > FACT_PACK_HARD_CAP_BYTES:
        # D5: over-cap FAILS rather than truncating. A truncated fact pack is
        # the sheared prompt that produced confident output about evidence it
        # never saw.
        return {
            "status": STATUS_FAILED,
            "model": "",
            "reason": (
                f"fact pack is {size} bytes, over the {FACT_PACK_HARD_CAP_BYTES}-byte "
                "hard cap; nothing was written rather than truncating the record"
            ),
            "outputs": [],
        }

    try:
        written = _publish(facts_target, render_fact_pack(pack))
    except OSError as exc:
        return {"status": STATUS_FAILED, "model": "",
                "reason": f"fact pack could not be published: {exc}", "outputs": []}
    outputs = [str(written)]

    over_target = f" (over the {FACT_PACK_TARGET_BYTES}-byte target)" if size > FACT_PACK_TARGET_BYTES else ""
    facts_reason = (
        f"facts for {day}: {size} bytes{over_target}, "
        f"n={((pack['outcomes']['overall'].get('close_r') or {}).get('n', 0))} outcome(s)"
        + (f", {len(unavailable)} source(s) unreadable" if unavailable else "")
    )

    if not narrate:
        return {"status": STATUS_OK, "model": "", "reason": facts_reason, "outputs": outputs}

    try:
        narration = _narrate(pack=pack, now=moment)
    except Exception as exc:  # noqa: BLE001 - a dead model is a normal state here
        _log.info("Daily digest: narration unavailable (%s); the fact pack stands.", exc)
        return {
            "status": STATUS_DEGRADED,
            "model": "",
            "reason": f"{facts_reason}; narration absent: {exc}",
            "outputs": outputs,
        }

    try:
        outputs.append(str(_publish(
            superseding_path(narration_path(target_root, day)),
            json.dumps(narration, indent=1, sort_keys=True, default=str) + "\n",
        )))
    except OSError as exc:
        return {
            "status": STATUS_DEGRADED,
            "model": str(narration.get("model") or ""),
            "reason": f"{facts_reason}; narration could not be published: {exc}",
            "outputs": outputs,
        }
    return {
        "status": STATUS_OK,
        "model": str(narration.get("model") or ""),
        "reason": facts_reason + "; narrated",
        "outputs": outputs,
    }


def _default_root() -> Path:
    from ai_jobs import store

    return store.digests_dir()


def _is_session(day: str, unavailable: dict[str, str]) -> bool:
    """Is this a trading session? An unanswerable calendar is recorded, not guessed."""
    try:
        from market_calendar import is_session

        return bool(is_session(date.fromisoformat(day)))
    except Exception as exc:  # noqa: BLE001
        unavailable["session calendar"] = str(exc)
        return True


def _read_rows(name: str, loader, unavailable: dict[str, str]) -> list[dict[str, Any]]:
    try:
        return list(loader())
    except Exception as exc:  # noqa: BLE001
        unavailable[name] = str(exc)
        return []


def _read_review_events() -> list[dict[str, Any]]:
    from review_events import load_review_events

    return load_review_events()


def _read_job_rows() -> list[dict[str, Any]]:
    from ai_jobs import ledger

    return ledger._read_rows(ledger.ledger_path(create=False))


def _read_champion_finals(day: str, unavailable: dict[str, str]):
    """This session's settled, entry-claiming CHAMPION outcomes.

    Read through `setup_scoreboard.load_intraday_finals`, which already applies
    the outcome store's exclusions - unsettled closes, sub-risk-floor rows, and
    families that do not CLAIM an entry (R10.B). Reusing it means the digest and
    the scoreboard cannot drift into two definitions of "usable".

    **Only the champion store is read.** No shadow engine's output reaches this
    pack (answer 3).
    """
    try:
        from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE
        from setup_scoreboard import load_intraday_finals

        path = Path(INTRADAY_BOUNCE_OUTCOMES_FILE)
        if not path.is_file():
            return [], {"outcome_store": "absent"}
        frame, coverage = load_intraday_finals(path, window_start=day, window_end=day)
    except Exception as exc:  # noqa: BLE001
        unavailable["intraday outcome store"] = str(exc)
        return [], {}
    if not len(frame):
        return [], _coverage_dict(coverage)
    usable = frame[frame["usable"]] if "usable" in frame else frame
    rows = [
        {
            "symbol": row.get("symbol"),
            "direction": row.get("direction"),
            "trade_date": row.get("trade_date"),
            "entry_time": row.get("entry_time"),
            "market_environment": row.get("market_environment"),
            # Lifted here rather than by widening `setup_scoreboard.CONTEXT_FIELDS`:
            # that tuple decides what the SCOREBOARD reads, and this is the
            # digest's need, not the scoreboard's.
            "env_key": _stamped_env_key(row.get("context_json")),
            "close_r": row.get("close_r"),
            "mfe_r": row.get("mfe_r"),
            "mae_r": row.get("mae_r"),
            # Decision A (2026-08-25). A sweep-finalized trade has no eod-hold
            # `close_r` at all; the R it DID reach is under one of these two
            # policies. Carried beside `close_r`, never folded into it - a
            # number that is a stop exit for some rows and an eod close for
            # others is a different statistic wearing one name.
            "r_stop_exit": row.get("r_stop_exit"),
            "r_last_measured": row.get("r_last_measured"),
        }
        for row in usable.to_dict("records")
    ]
    return rows, _coverage_dict(coverage)


def _stamped_env_key(context_json: Any) -> str:
    """The `env_key` the alert path stamped, or blank. Never derived here."""
    if not isinstance(context_json, str) or not context_json.strip():
        return ""
    try:
        payload = json.loads(context_json)
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    return str(payload.get("env_key") or "").strip()


def _coverage_dict(coverage: Any) -> dict[str, Any]:
    if coverage is None:
        return {}
    return {
        "rows_scanned": getattr(coverage, "rows_scanned", 0),
        "finals": getattr(coverage, "finals", 0),
        "in_window": getattr(coverage, "in_window", 0),
        "unsettled": getattr(coverage, "unsettled", 0),
        "never_measured": getattr(coverage, "never_measured", 0),
        "below_risk_floor": getattr(coverage, "below_risk_floor", 0),
        "not_entry_claim": getattr(coverage, "not_entry_claim", 0),
        "by_claim_kind": dict(getattr(coverage, "by_claim_kind", {}) or {}),
        "usable": getattr(coverage, "usable", 0),
        "usable_eod_hold_only": getattr(coverage, "usable_eod_hold_only", 0),
        "policy_measured": dict(getattr(coverage, "policy_measured", {}) or {}),
        "unresolved": getattr(coverage, "unresolved", 0),
        "unresolved_by_reason": dict(getattr(coverage, "unresolved_by_reason", {}) or {}),
        "note": (
            "Excluded rows are counted by reason, never silently dropped. A "
            "family that does not CLAIM an entry is not a trade and is never "
            "averaged as one."
        ),
    }


# ---------------------------------------------------------------------------
# rollups (D8) and the gate
# ---------------------------------------------------------------------------


def read_fact_packs(root: Path, *, since: str = "", until: str = "") -> list[dict[str, Any]]:
    """Every pack in the window, newest last. Reads only; writes nothing."""
    base = Path(root) / "facts"
    if not base.is_dir():
        return []
    packs: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        day = str(payload.get("session_date") or "")
        if since and day < since:
            continue
        if until and day > until:
            continue
        packs.append(dict(payload))
    packs.sort(key=lambda pack: (str(pack.get("session_date")), str(pack.get("generated_at"))))
    return packs


def rollup(root: Path, *, since: str = "", until: str = "") -> dict[str, Any]:
    """A weekly/monthly view, COMPUTED from the packs on demand.

    D8: a derived aggregate store would be a second thing to keep in sync and a
    second thing to be wrong. This writes nothing.

    Sessions are weighted by their n, not averaged as day-means: a 40-outcome
    session and a 2-outcome session are not two equal observations.
    """
    packs = [pack for pack in read_fact_packs(root, since=since, until=until) if pack.get("is_session")]
    latest: dict[str, dict[str, Any]] = {}
    for pack in packs:
        latest[str(pack.get("session_date"))] = pack  # a superseding sibling wins
    totals: dict[str, dict[str, float]] = {}
    for metric in ("close_r", "mfe_r", "mae_r"):
        weighted = 0.0
        count = 0
        for pack in latest.values():
            block = ((pack.get("outcomes") or {}).get("overall") or {}).get(metric) or {}
            value, n = block.get("value"), int(block.get("n") or 0)
            if value is None or not n:
                continue
            weighted += float(value) * n
            count += n
        totals[metric] = {"value": round(weighted / count, 4) if count else None, "n": count}
    return {
        "since": since,
        "until": until,
        "sessions": len(latest),
        "evidence_label": _discovery_label(),
        "note": (
            "Computed from the fact packs on demand (D8); nothing is stored. "
            "Session means are weighted by n - a 40-outcome session and a "
            "2-outcome session are not two equal observations."
        ),
        **totals,
    }


def clean_digest_sessions(root: Path) -> int:
    """How many SESSION fact packs exist. Counting is not passing.

    Phase 2's exit gate is ten consecutive session days of digests plus the
    trader spot-auditing at least three against raw evidence. This answers only
    the first half, and only the counting part of it: a number here never marks
    a live gate met.

    An empty non-session pack does not count. It exists to make a gap visible,
    which is the opposite of evidence that the digest ran over a real day.
    """
    return len({
        str(pack.get("session_date"))
        for pack in read_fact_packs(root)
        if pack.get("is_session") and pack.get("session_date")
    })


def digest_gate_state(root: Path) -> dict[str, Any]:
    """The R10.I-shaped statement for any surface that reports on this phase."""
    collected = clean_digest_sessions(root)
    met = collected >= REQUIRED_CLEAN_SESSIONS
    return {
        "sessions_collected": collected,
        "sessions_required": REQUIRED_CLEAN_SESSIONS,
        "window_met": met,
        "statement": (
            "Digest collection window met by count. The trader spot-audit of at "
            "least three packs against raw evidence is a separate half of the "
            "gate and is not answered here."
            if met else
            f"DIGEST GATE NOT MET: {collected} of {REQUIRED_CLEAN_SESSIONS} session "
            "fact packs exist. Phase 3 and anything downstream of it must not run "
            "on this evidence."
        ),
    }
