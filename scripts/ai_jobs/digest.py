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

**Both halves of that gate are now MEASURED** (packet Q4, 2026-09-04).

* The window half counts a RUN of consecutive exchange sessions ending at the
  newest pack, walked through `market_calendar` and never by weekday
  arithmetic, where every session in the run has a pack whose own failure
  record (`unavailable`) is empty. It counted DISTINCT session packs until
  then, so ten packs scattered across a month read as a met window and a pack
  that named a source it could not read counted as clean. A non-session pack
  neither counts nor breaks a run, and `first_gap_session` says where the run
  stopped.
* The audit half is a FILE - `digest_audit_approval.json` beside the packs -
  written only by `python -m ai_jobs.digest approve-audit`, which the trader
  runs. Nothing automatic writes it; the runner cannot approve its own
  evidence. `journal_enrichment` refuses until both halves are true.

**`entry_index.json` is the compact handoff** (Q4.4): one deterministic,
superseding-written index of what the packs in the `LATELY_SESSIONS` window
hold - their versions, their failures, four never-merged evidence sections,
what crossed the evidence FLOOR since the prior equal-length window, and the
registered experiments that are still collecting. No model, no ranking of
immature cells, and a failure to write it never fails the digest.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
#: v3 (2026-09-23) added the capped per-name `names` block (top D1 setups,
#: settled swing outcomes, M5 alerts, journal trades). v2 packs on disk stay
#: v2 and every reader here treats a missing `names` block as absent.
FACTS_SCHEMA = "daily_digest_facts_v3"
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
    "names": (
        "v3: per-name rows are capped, result-selected examples that print their "
        "full n; they are never slices or rates. null means unknown."
    ),
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
    top_setups: Mapping[str, Any] | None = None,
    swing_outcomes: Mapping[str, Any] | None = None,
    journal_trades: Mapping[str, Any] | None = None,
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
        "names": build_names_block(
            is_session=bool(is_session),
            finals=rows,
            top_setups=top_setups,
            swing_outcomes=swing_outcomes,
            journal_trades=journal_trades,
        ),
    }
    pack["summary"] = _summary(pack)
    _fit_names(pack)
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


#: Fact-file key for the night's telemetry lines. Added to the published file
#: only, after the narration package is built, so the narrator never sees it.
NIGHT_TELEMETRY_KEY = "night_telemetry"
#: The status words a goal line counts, in print order.
_GOAL_STATUS_WORDS = (
    ("ok", ("ok",)),
    ("degraded", ("degraded_no_narrative",)),
    ("failed", ("failed",)),
    ("skipped", ("skipped",)),
)


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _row_started(row: Mapping[str, Any]) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(row.get("started_at") or ""))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else None


def _stage_three_slots() -> frozenset[str]:
    """Stage-3 slot names: the runner's slate from `journal_enrichment` on."""
    try:
        from ai_jobs import runner

        names = [slot.name for slot in runner.default_slots() + runner.optional_slots()]
        first = names.index(STAGE_THREE_FIRST_SLOT)
        return frozenset(names[first:])
    except Exception:  # noqa: BLE001 - without it only the clock rule decides
        return frozenset()


#: The first slot of stage 3 (decision 0018); a night that reached it is complete.
STAGE_THREE_FIRST_SLOT = "journal_enrichment"


def last_complete_night(
    job_rows, session_date: str, *, now: datetime | None = None
) -> tuple[str, list[Mapping[str, Any]]]:
    """The newest earlier session whose night is complete, and its ledger rows.

    Complete: it has a stage-3 row, or every one of its rows started before the
    current night began (the first row for `session_date`, else `now`).
    Returns ("", []) when no earlier night qualifies.
    """
    rows = [row for row in (job_rows or []) if isinstance(row, Mapping)]
    tonight = [row for row in rows if str(row.get("session_date") or "") == str(session_date)]
    starts = [stamp for stamp in (_row_started(row) for row in tonight) if stamp is not None]
    night_start = min(starts) if starts else _now(now)
    by_session: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        day = str(row.get("session_date") or "")
        if day and day < str(session_date):
            by_session.setdefault(day, []).append(row)
    stage_three = _stage_three_slots()
    for day in sorted(by_session, reverse=True):
        night = by_session[day]
        if any(str(row.get("job") or "") in stage_three for row in night):
            return day, night
        stamps = [_row_started(row) for row in night]
        if all(stamp is None or stamp < night_start for stamp in stamps):
            return day, night
    return "", []


def night_telemetry_lines(
    job_rows, session_date: str, *, now: datetime | None = None
) -> list[str]:
    """'slots per goal' and 'tokens' lines for the last COMPLETE night, named by date.

    The digest runs early in the night, so tonight is half-run; the lines
    describe the newest earlier night that finished (`last_complete_night`).
    A slot counts once, by its newest row for that session (the last row wins);
    attempt-cap markers and corrections are not a slot's outcome. Tokens sum
    every row, because every attempt spent them.
    """
    from ai_jobs.runner import SLOT_GOALS

    night, rows = last_complete_night(job_rows, session_date, now=now)
    if not night:
        return [
            "slots per goal: unknown (no complete night in the ledger yet)",
            "tokens: unknown (no complete night in the ledger yet)",
        ]
    last: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        job = str(row.get("job") or "")
        if not job or str(row.get("status") or "") == "correction" or row.get("terminal"):
            continue
        last[job] = row
    counts = {goal: {word: 0 for word, _ in _GOAL_STATUS_WORDS} for goal in SLOT_GOALS}
    # Rows written before slots declared a goal are unknown, never counted as 0.
    no_goal = sum(1 for row in last.values() if not str(row.get("goal") or ""))
    for row in last.values():
        goal = str(row.get("goal") or "")
        if goal not in counts:
            continue
        status = str(row.get("status") or "")
        for word, statuses in _GOAL_STATUS_WORDS:
            if status in statuses:
                counts[goal][word] += 1
    if last and no_goal == len(last):
        goal_line = f"slots per goal (night of {night}): unknown (rows carry no goal)"
    else:
        goal_line = f"slots per goal (night of {night}): " + "; ".join(
            f"{goal}: " + " / ".join(f"{word} {n}" for word, n in counts[goal].items())
            for goal in SLOT_GOALS
        )
        if no_goal:
            goal_line += f"; unknown goal: {no_goal} slot" + ("s" if no_goal != 1 else "")

    # Rows written before slots summed their tokens carry `{}`: unknown, never 0.
    if not any(
        isinstance(row.get("tokens"), Mapping)
        and any(key in row["tokens"] for key in ("prompt_tokens", "completion_tokens", "calls"))
        for row in rows
    ):
        return [goal_line, f"tokens (night of {night}): unknown (rows carry no tokens)"]
    prompt = completion = calls = 0
    by_slot: dict[str, int] = {}
    for row in rows:
        tokens = row.get("tokens")
        if not isinstance(tokens, Mapping):
            continue
        slot_prompt = _int_or_zero(tokens.get("prompt_tokens"))
        prompt += slot_prompt
        completion += _int_or_zero(tokens.get("completion_tokens"))
        calls += _int_or_zero(tokens.get("calls"))
        job = str(row.get("job") or "")
        if job and slot_prompt:
            by_slot[job] = by_slot.get(job, 0) + slot_prompt
    top = sorted(by_slot.items(), key=lambda item: (-item[1], item[0]))[:3]
    token_line = (
        f"tokens (night of {night}): {prompt}/{completion} over {calls} calls; top 3 slots by "
        "prompt tokens: " + (", ".join(f"{job} {n}" for job, n in top) if top else "none")
    )
    return [goal_line, token_line]


# ---------------------------------------------------------------------------
# per-name rows (v3, trader 2026-09-23)
# ---------------------------------------------------------------------------

#: Row caps per list. Result-selected examples, each printed with its full n.
TOP_SETUPS_CAP = 10
SWING_OUTCOMES_CAP = 5
M5_ALERTS_CAP = 3
JOURNAL_TRADES_CAP = 10

#: Bytes kept free under the hard cap when named rows are trimmed to fit.
NAMES_FIT_MARGIN_BYTES = 256

#: M5 exit policies a ranking may use: (row key, finals field).
_M5_RANK_POLICIES = (
    ("close_r", "close_r"),
    ("stop_exit_r", "r_stop_exit"),
    ("last_measured_r", "r_last_measured"),
)

NAME_STATUS_OK = "ok"
NAME_STATUS_ABSENT = "absent"
NAME_STATUS_UNKNOWN = "unknown"
NAME_STATUS_NOT_A_SESSION = "not_a_session"

_TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}

#: The lists inside `names`. When a pack is over budget the longest list loses
#: its last row first; ties go to the earlier entry here.
_NAME_LISTS = (
    ("journal_trades", "rows"),
    ("d1_top_setups", "rows"),
    ("swing_settled", "worst"),
    ("swing_settled", "best"),
    ("m5_alerts", "worst"),
    ("m5_alerts", "best"),
)


def names_source(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    status: str = NAME_STATUS_OK,
    reason: str = "",
) -> dict[str, Any]:
    """One reader's answer: its rows, or why there are none."""
    return {
        "status": str(status),
        "reason": str(reason or ""),
        "rows": [dict(row) for row in (rows or [])] if status == NAME_STATUS_OK else [],
    }


def _number_or_none(value: Any, digits: int = 4) -> float | None:
    values = _numbers([{"v": value}], "v")
    return round(values[0], digits) if values else None


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _section_head(source: Mapping[str, Any] | None, source_id: str, is_session: bool) -> dict[str, Any]:
    if not is_session:
        return {"status": NAME_STATUS_NOT_A_SESSION, "reason": "not a trading session",
                "source_id": source_id, "n": 0}
    if source is None:
        return {"status": NAME_STATUS_UNKNOWN, "reason": "not read for this pack",
                "source_id": source_id, "n": 0}
    status = str(source.get("status") or NAME_STATUS_UNKNOWN)
    head: dict[str, Any] = {"status": status, "source_id": source_id, "n": 0}
    if status != NAME_STATUS_OK:
        head["reason"] = str(source.get("reason") or "no reason recorded")
    return head


def _source_rows(source: Mapping[str, Any] | None, section: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if section["status"] != NAME_STATUS_OK:
        return []
    return list((source or {}).get("rows") or [])


def _best_worst(rows: list[dict[str, Any]], key: str, cap: int) -> tuple[list, list]:
    """Best and worst by `key`, one row per name and side, never listed twice.

    With fewer than 2 x cap names the best list takes the upper half, so a
    short day still shows its worst.
    """
    ranked = sorted(
        (row for row in rows if row.get(key) is not None),
        key=lambda row: (-row[key], row["symbol"], row.get("side") or ""),
    )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in ranked:
        ident = (row["symbol"], row.get("side") or "")
        if ident not in seen:
            seen.add(ident)
            unique.append(row)
    best_n = min(cap, (len(unique) + 1) // 2)
    best = unique[:best_n]
    worst = list(reversed(unique[best_n:]))[:cap]
    return best, worst


def _top_setups_section(source, is_session: bool) -> dict[str, Any]:
    section = _section_head(source, "scan.tier_list", is_session)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _source_rows(source, section):
        symbol = _text(row.get("symbol")).upper()
        side = _text(row.get("side")).upper() or None
        if not symbol or (symbol, side or "") in seen:
            continue
        seen.add((symbol, side or ""))
        rows.append({
            "symbol": symbol,
            "side": side,
            "tier": _text(row.get("tier")).upper() or None,
            "score": _number_or_none(row.get("priority_score"), 1),
            "setup": _text(row.get("setup_family")) or None,
            "zone": _text(row.get("favorite_zone")) or _text(row.get("current_band_zone")) or None,
        })
    rows.sort(key=lambda row: (
        _TIER_ORDER.get(row["tier"] or "", 9),
        -(row["score"] if row["score"] is not None else float("-inf")),
        row["symbol"],
    ))
    section["n"] = len(rows)
    section["capped"] = max(0, len(rows) - TOP_SETUPS_CAP)
    section["rows"] = rows[:TOP_SETUPS_CAP]
    section["basis"] = "this session's D1 scan, by tier (S>A>B>C) then priority score"
    return section


def _swing_section(source, is_session: bool) -> dict[str, Any]:
    section = _section_head(source, "scan.session_horizon_outcomes", is_session)
    rows: list[dict[str, Any]] = []
    for row in _source_rows(source, section):
        symbol = _text(row.get("symbol")).upper()
        value = _number_or_none(row.get("side_return_pct"), 2)
        if not symbol or value is None:
            continue
        horizon = _number_or_none(row.get("horizon_sessions"), 0)
        rows.append({
            "symbol": symbol,
            "side": _text(row.get("side")).upper() or None,
            "setup": _text(row.get("setup_family")) or None,
            "tier": _text(row.get("tier")).upper() or None,
            "scan_date": _text(row.get("scan_date"))[:10] or None,
            "h": int(horizon) if horizon is not None else None,
            "ret_pct": value,
        })
    section["n"] = len(rows)
    section["best"], section["worst"] = _best_worst(rows, "ret_pct", SWING_OUTCOMES_CAP)
    section["basis"] = (
        "scan rows whose h-session horizon settled this session; ret_pct is the "
        "side return from scan-day close to this close. This source measures no R."
    )
    return section


def _m5_section(finals: Sequence[Mapping[str, Any]], is_session: bool) -> dict[str, Any]:
    section = _section_head(names_source(finals), "outcomes.intraday_finals", is_session)
    finals = list(finals) if is_session else []
    # Rank by ONE exit policy, the one measured on the most rows today; the
    # policies are never mixed in one ranking.
    rank_by = max(
        _M5_RANK_POLICIES,
        key=lambda item: (len(_numbers(finals, item[1])), -_M5_RANK_POLICIES.index(item)),
    )[0]
    source_field = dict(_M5_RANK_POLICIES)[rank_by]
    rows: list[dict[str, Any]] = []
    for row in finals:
        symbol = _text(row.get("symbol")).upper()
        if not symbol:
            continue
        entry = _text(row.get("entry_time"))
        rows.append({
            "symbol": symbol,
            "side": _side_of(row.get("direction")),
            "trigger": _text(row.get("bounce_type")) or None,
            "time": entry[11:16] if len(entry) >= 16 else None,
            rank_by: _number_or_none(row.get(source_field), 2),
            "mfe_r": _number_or_none(row.get("mfe_r"), 2),
        })
    section["n"] = len(rows)
    section["rank_by"] = rank_by
    section["unranked"] = len([row for row in rows if row[rank_by] is None])
    section["best"], section["worst"] = _best_worst(rows, rank_by, M5_ALERTS_CAP)
    section["basis"] = (
        "champion M5 alerts that settled with an entry claim, ranked by rank_by "
        "(the exit policy measured on the most rows this session)"
    )
    return section


def _journal_section(source, is_session: bool) -> dict[str, Any]:
    section = _section_head(source, "journal.trades", is_session)
    rows: list[dict[str, Any]] = []
    for row in _source_rows(source, section):
        symbol = _text(row.get("symbol")).upper()
        if not symbol:
            continue
        status = _text(row.get("status")).upper() or None
        rows.append({
            "symbol": symbol,
            "side": _text(row.get("direction")).upper() or None,
            "status": status,
            # An open trade's P&L is not settled: unknown, never zero.
            "net_pnl": _number_or_none(row.get("net_pnl"), 2) if status == "CLOSED" else None,
            "ccy": _text(row.get("currency")).upper() or None,
            "r": _number_or_none(row.get("r"), 2),
        })
    section["n"] = len(rows)
    section["capped"] = max(0, len(rows) - JOURNAL_TRADES_CAP)
    section["rows"] = rows[:JOURNAL_TRADES_CAP]
    section["basis"] = (
        "the trader's journal trades opened or closed this session; r is null "
        "because the journal stores no stop"
    )
    return section


def build_names_block(
    *,
    is_session: bool,
    finals: Sequence[Mapping[str, Any]],
    top_setups: Mapping[str, Any] | None,
    swing_outcomes: Mapping[str, Any] | None,
    journal_trades: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The per-name rows the narrator may name. Deterministic; no model."""
    return {
        "note": (
            "Per-name examples, result-selected and capped; each list prints its "
            "full n. null means unknown, never zero. status: ok (read), absent "
            "(no file), unknown (unreadable or not this session's data)."
        ),
        "d1_top_setups": _top_setups_section(top_setups, is_session),
        "swing_settled": _swing_section(swing_outcomes, is_session),
        "m5_alerts": _m5_section(list(finals or []), is_session),
        "journal_trades": _journal_section(journal_trades, is_session),
    }


def _fit_names(pack: dict[str, Any]) -> None:
    """Trim named rows (never other facts) until the pack fits under its cap."""
    names = pack.get("names") or {}
    budget = FACT_PACK_HARD_CAP_BYTES - NAMES_FIT_MARGIN_BYTES
    while fact_pack_bytes(pack) > budget:
        size, _order, section, key = max(
            (len((names.get(section) or {}).get(key) or []), -index, section, key)
            for index, (section, key) in enumerate(_NAME_LISTS)
        )
        if size <= 0:
            return
        names[section][key].pop()
        names[section]["trimmed"] = int(names[section].get("trimmed", 0)) + 1


def _named_rows(pack: Mapping[str, Any], sections: Sequence[str] | None = None):
    for name, section in (pack.get("names") or {}).items():
        if not isinstance(section, Mapping) or (sections is not None and name not in sections):
            continue
        for key in ("rows", "best", "worst"):
            for row in section.get(key) or []:
                if isinstance(row, Mapping) and row.get("symbol"):
                    yield row


def named_symbols(pack: Mapping[str, Any], sections: Sequence[str] | None = None) -> set[str]:
    """Every ticker printed as a row `symbol` in the pack's names block."""
    return {str(row["symbol"]).upper() for row in _named_rows(pack, sections)}


# ---------------------------------------------------------------------------
# per-name readers
# ---------------------------------------------------------------------------


def _csv_rows(path: Path):
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_top_setups(day: str, *, path: Path | None = None) -> dict[str, Any]:
    """This session's D1 scan tier list. Another day's scan is unknown, not today's."""
    if path is None:
        from project_paths import MASTER_AVWAP_TIER_LIST_FILE

        path = Path(MASTER_AVWAP_TIER_LIST_FILE)
    path = Path(path)
    if not path.is_file():
        return names_source(status=NAME_STATUS_ABSENT, reason=f"{path.name} not found")
    rows: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for row in _csv_rows(path):
        scan_date = _text(row.get("scan_date"))[:10]
        seen_dates.add(scan_date)
        if scan_date == day:
            rows.append(row)
    if not rows:
        latest = max(seen_dates - {""}, default="nothing")
        return names_source(
            status=NAME_STATUS_UNKNOWN,
            reason=f"the scan tier list holds {latest}, not {day}",
        )
    return names_source(rows)


def read_swing_outcomes(day: str, *, path: Path | None = None) -> dict[str, Any]:
    """Measured scan-row horizons that settled on this session."""
    if path is None:
        from project_paths import MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE

        path = Path(MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE)
    path = Path(path)
    if not path.is_file():
        return names_source(status=NAME_STATUS_ABSENT, reason=f"{path.name} not found")
    rows = [
        row for row in _csv_rows(path)
        if _text(row.get("target_session"))[:10] == day
        and _text(row.get("measured")).lower() in {"true", "1"}
    ]
    return names_source(rows)


def read_journal_trades(day: str, *, path: Path | None = None) -> dict[str, Any]:
    """The journal's trades for this session, opened read-only (never created)."""
    import sqlite3
    from urllib.parse import quote

    if path is None:
        from project_paths import JOURNAL_DB_FILE

        path = Path(JOURNAL_DB_FILE)
    path = Path(path)
    if not path.is_file():
        return names_source(status=NAME_STATUS_ABSENT, reason=f"{path.name} not found")
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, direction, status, net_pnl, currency, opened_at, closed_at "
            "FROM trades WHERE substr(opened_at, 1, 10) = ? OR substr(closed_at, 1, 10) = ? "
            "OR trade_date = ? ORDER BY opened_at, trade_id",
            (day, day, day),
        ).fetchall()
    finally:
        conn.close()
    return names_source([dict(row) for row in rows])


def _read_names_sources(day: str, unavailable: dict[str, str]) -> dict[str, Any]:
    """Every per-name reader. A read that raises is recorded and reads unknown."""
    sources: dict[str, Any] = {}
    for key, name, reader in (
        ("top_setups", "scan tier list", read_top_setups),
        ("swing_outcomes", "scan horizon outcomes", read_swing_outcomes),
        ("journal_trades", "trade journal", read_journal_trades),
    ):
        try:
            sources[key] = reader(day)
        except Exception as exc:  # noqa: BLE001 - one unreadable source costs its rows only
            unavailable[name] = str(exc)
            sources[key] = names_source(status=NAME_STATUS_UNKNOWN, reason=str(exc))
    return sources


# ---------------------------------------------------------------------------
# narration check: only tickers the pack holds, with their numbers
# ---------------------------------------------------------------------------



#: A ticker-shaped token: 2-5 capitals (optional class suffix), not part of a word.
_TICKER_RE = re.compile(r"(?<![A-Za-z0-9_])\$?([A-Z]{2,5}(?:\.[A-Z]{1,2})?)(?![A-Za-z0-9_])")
_DECIMAL_RE = re.compile(r"(?<![\d.])[-+]?(\d+\.\d+)(?![\d.])")

#: Capitalised words a narration may use that are not tickers.
NON_TICKER_WORDS = frozenset({
    "LONG", "SHORT", "EOD", "MFE", "MAE", "AVWAP", "AVWAPE", "VWAP", "ATR", "RVOL",
    "SMA", "EMA", "PNL", "USD", "CAD", "ET", "PT", "AM", "PM", "OK", "NA", "AI",
    "RTH", "ETF", "IPO", "EPS", "YTD", "QTD", "TBD", "UPPER", "LOWER", "NOT", "ONLY",
    "AND", "OR", "THE", "NO", "ALL", "NONE", "BUT", "NEW", "TOP", "BEST", "WORST",
    "HIGH", "LOW", "CLOSED", "OPEN", "UNKNOWN", "NULL", "DAY", "SWING", "TIER",
})

_BEST_CANDIDATE_SECTIONS = ("d1_top_setups", "swing_settled", "m5_alerts")


def _narration_texts(summary: Mapping[str, Any]) -> list[str]:
    texts = [str(summary.get("executive_summary") or "")]
    for value in summary.values():
        if isinstance(value, list):
            texts.extend(
                str(item.get("statement") or "") for item in value if isinstance(item, Mapping)
            )
    return texts


def check_narration_names(summary: Mapping[str, Any], pack: Mapping[str, Any]) -> list[str]:
    """Problems with a narration's names; empty when it may be published.

    Every ticker-shaped word must be a pack symbol (or a word the pack itself
    prints). When the pack names rows, best_candidates must name them, and a
    decimal in a best candidate must be one of that ticker's own numbers.
    """
    symbols = named_symbols(pack)
    printed = set(_TICKER_RE.findall(render_fact_pack(pack)))
    allowed = symbols | printed | NON_TICKER_WORDS
    problems: list[str] = []

    unknown = sorted({
        token for text in _narration_texts(summary) for token in _TICKER_RE.findall(text)
        if token not in allowed
    })
    if unknown:
        problems.append(
            "named tickers that are not in the fact pack: " + ", ".join(unknown)
            + ". Name only a `symbol` printed in names.*."
        )

    candidates = [
        str(item.get("statement") or "")
        for item in summary.get("best_candidates") or [] if isinstance(item, Mapping)
    ]
    if named_symbols(pack, _BEST_CANDIDATE_SECTIONS):
        if not candidates:
            problems.append(
                "best_candidates is empty although names.d1_top_setups, "
                "names.swing_settled or names.m5_alerts list tickers; name them "
                "with their numbers"
            )
        for text in candidates:
            named = [token for token in _TICKER_RE.findall(text) if token in symbols]
            if not named:
                problems.append(f"best_candidates statement names no pack ticker: {text!r}")
                continue
            values = [
                value for symbol in named for row in _named_rows(pack)
                if str(row["symbol"]).upper() == symbol
                for value in row.values() if isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
            for written in _DECIMAL_RE.findall(text):
                places = len(written.split(".", 1)[1])
                wanted = float(written)
                if not any(abs(round(abs(value), places) - wanted) < 1e-9 for value in values):
                    problems.append(
                        f"best_candidates number {written} is not a number the pack "
                        f"prints for {', '.join(named)}"
                    )
    return problems


#: Told to the narrator inside its package; `check_narration_names` enforces them.
NARRATION_RULES = (
    "best_candidates: name the tickers listed in names.d1_top_setups, "
    "names.swing_settled and names.m5_alerts, each with its own numbers copied "
    "from its row (tier and score, ret_pct and h, the m5 rank_by R). If those lists are "
    "all empty, best_candidates may be empty.",
    "Name no ticker that is not a `symbol` in names.*. A narration that names "
    "any other ticker, or a number that is not that ticker's own, is rejected.",
    "Cite 'digest.facts' in evidence_refs. A statement with a percent or a "
    "decimal R needs metric_ref {source_id: 'digest.facts', key: the row field "
    "(ret_pct, close_r, net_pnl), horizon: e.g. 'session' or 'h sessions', "
    "denominator: e.g. 'names.swing_settled.n'}.",
    "names.journal_trades are trades the trader made this session; do not call "
    "them held positions. null means unknown; never write it as zero.",
)


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
    names = pack.get("names") or {}
    if names:
        parts.append(
            f"Named rows: {len(named_symbols(pack))} ticker(s) across top setups, "
            "settled swings, M5 alerts and journal trades (see names)."
        )
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
    """Temp-and-rename, and **no litter when the rename fails**.

    The share can drop out between the write and the replace. Without the
    cleanup that leaves a half-published `<name>.tmp` beside the packs, which
    the next reader has to be told to ignore - and a store that needs a
    told-to-ignore file is a store nobody trusts. The last good file is
    untouched either way, which is what the rename buys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:  # pragma: no cover - nothing more can be done here
            _log.warning("Daily digest: could not remove the temp file %s.", tmp)
        raise
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
        "narration_rules": list(NARRATION_RULES),
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
    # One retry, told exactly what was wrong; a second miss publishes nothing.
    previous_error = ""
    for _attempt in range(2):
        result = ai_summary.request_ai_summary(
            provider="local",
            model=ai_summary.local_model("medium"),
            api_key="",
            evidence=package,
            timeout_seconds=900,
            previous_error=previous_error,
        )
        problems = check_narration_names(result.get("summary") or {}, pack)
        if not problems:
            break
        previous_error = "; ".join(problems)
    else:
        raise RuntimeError(f"narration rejected: {previous_error}")
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
    name_sources = _read_names_sources(day, unavailable) if is_session else {}

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
        top_setups=name_sources.get("top_setups"),
        swing_outcomes=name_sources.get("swing_outcomes"),
        journal_trades=name_sources.get("journal_trades"),
    )

    # The telemetry lines go in the published file only; `pack` (what the
    # narrator is handed) stays without them.
    published = dict(pack)
    try:
        import slot_output_reads

        lines = night_telemetry_lines(job_rows, day, now=moment) + [
            slot_output_reads.unread_line()
        ]
        published[NIGHT_TELEMETRY_KEY] = {"lines": lines}
    except Exception as exc:  # noqa: BLE001 - telemetry never costs the digest
        _log.info("Daily digest: night telemetry not built (%s).", exc)

    size = fact_pack_bytes(published)
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
        written = _publish(facts_target, render_fact_pack(published))
    except OSError as exc:
        return {"status": STATUS_FAILED, "model": "",
                "reason": f"fact pack could not be published: {exc}", "outputs": []}
    outputs = [str(written)]

    # Q4.4. The pack is on disk; the index is a convenience over it. A failure
    # here is logged and never fails the digest, and the last good index
    # survives because the write is temp-and-rename.
    try:
        outputs.append(str(write_entry_index(target_root, as_of=day)))
    except Exception as exc:  # noqa: BLE001 - never costs the record it describes
        _log.info("Daily digest: entry index not refreshed (%s); the packs stand.", exc)

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
            # The alert's trigger, for the names block only (never a slice).
            "bounce_type": row.get("bounce_type"),
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


def read_fact_pack_files(
    root: Path, *, since: str = "", until: str = ""
) -> list[tuple[Path, dict[str, Any]]]:
    """`(path, pack)` for every pack in the window, newest last.

    The path is carried because a session can have SUPERSEDING siblings
    (`2026-08-20.1.json` corrects `2026-08-20.json`, D6) and any reader that
    cites a pack must cite the file its numbers came from. `read_fact_packs`
    discarded the path, so the entry index cited `facts_path()` - always
    version 1 - beside values read from the newest sibling, which pointed a
    reader at the pack that had been corrected.
    """
    base = Path(root) / "facts"
    if not base.is_dir():
        return []
    entries: list[tuple[Path, dict[str, Any]]] = []
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
        entries.append((path, dict(payload)))
    entries.sort(key=lambda item: (
        str(item[1].get("session_date")),
        str(item[1].get("generated_at")),
        _supersession_index(item[0]),
    ))
    return entries


def _supersession_index(path: Path) -> int:
    """`2026-08-25.json` -> 0, `2026-08-25.1.json` -> 1, and so on.

    The tiebreak when two siblings carry the same `generated_at`, which a
    same-second re-run produces. Sorting by NAME would put `.1` before the
    base file - '1' < 'j' - and hand the correction's place to the pack it
    corrected.
    """
    tail = path.stem.rsplit(".", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def read_fact_packs(root: Path, *, since: str = "", until: str = "") -> list[dict[str, Any]]:
    """Every pack in the window, newest last. Reads only; writes nothing."""
    return [pack for _path, pack in read_fact_pack_files(root, since=since, until=until)]


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


def pack_failures(pack: Mapping[str, Any]) -> dict[str, str]:
    """The pack's OWN failure record: the sources it could not read.

    `unavailable` is the field the pack carries and the field its own summary
    calls INCOMPLETE. There is no separate `failures`/`errors` key, and none is
    invented here - a gate reads what the record says, not what a reader wishes
    it said.
    """
    return {str(name): str(reason) for name, reason in (pack.get("unavailable") or {}).items()}


def latest_pack_files_by_session(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Newest `(path, pack)` per session date. A superseding sibling wins (D6).

    `read_fact_pack_files` already sorts by `(session_date, generated_at,
    name)`, so the last entry seen for a day is the one that corrects the ones
    before it - and the path travels with it, because a citation to a
    superseded file is a citation to a pack that was corrected.
    """
    latest: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, pack in read_fact_pack_files(root):
        day = str(pack.get("session_date") or "")
        if day:
            latest[day] = (path, pack)
    return latest


def latest_packs_by_session(root: Path) -> dict[str, dict[str, Any]]:
    """Newest pack per session date. A superseding sibling wins (D6)."""
    return {day: pack for day, (_path, pack) in latest_pack_files_by_session(root).items()}


def collected_digest_sessions(root: Path) -> int:
    """How many DISTINCT session fact packs exist. The pre-Q4 count.

    Kept because every surface that reports "N of 10 collected" is reporting
    this, and because a run that broke last night did not un-collect the packs
    it already had. It is no longer what the gate turns on.
    """
    return len({
        day for day, pack in latest_packs_by_session(root).items()
        if pack.get("is_session")
    })


#: The walk back through the calendar is bounded so a calendar defect can never
#: become an unbounded loop. Four times the required run plus a margin is far
#: more history than the gate can ever need.
_MAX_GATE_WALK_SESSIONS = 400


def consecutive_clean_state(root: Path, *, as_of: str | date | None = None) -> dict[str, Any]:
    """The RUN of consecutive clean exchange sessions ending at the newest pack.

    Q4.1. Ten packs are not ten consecutive sessions: the pre-Q4 count was
    `len(distinct session dates)`, so a month with a hole in the middle read as
    a met window, and a pack that recorded an unreadable source counted as
    clean. Both are now false.

    * the run walks `market_calendar.previous_session`, **never weekday
      arithmetic** - a weekend or a holiday is not a gap because it is not a
      session;
    * a session in the run is CLEAN when its newest pack is `is_session` and
      `pack_failures` is empty;
    * a pack the digest wrote as a NON-session on a day the calendar calls one
      (an unscheduled closure the calendar cannot know about) neither counts
      nor breaks - it is skipped;
    * `first_gap_session` is the newest session that stopped the run, so the
      statement can say WHERE rather than only how many.
    """
    packs = latest_packs_by_session(root)
    limit = ""
    if as_of is not None:
        limit = as_of.isoformat() if isinstance(as_of, date) else str(as_of)[:10]
    sessions = {
        day: pack for day, pack in packs.items()
        if pack.get("is_session") and (not limit or day <= limit)
    }
    state: dict[str, Any] = {
        "sessions_consecutive_clean": 0,
        "first_gap_session": None,
        "newest_session": None,
        "gap_reason": "",
    }
    if not sessions:
        state["gap_reason"] = "no session fact pack exists"
        return state

    newest = max(sessions)
    state["newest_session"] = newest
    try:
        from market_calendar import previous_session

        cursor = date.fromisoformat(newest)
    except Exception as exc:  # noqa: BLE001 - an unreadable calendar is recorded
        state["gap_reason"] = f"exchange calendar unavailable: {exc}"
        return state

    count = 0
    for _ in range(_MAX_GATE_WALK_SESSIONS):
        day = cursor.isoformat()
        pack = packs.get(day)
        if pack is None:
            state["first_gap_session"] = day
            state["gap_reason"] = f"no fact pack for the session {day}"
            break
        if not pack.get("is_session"):
            # Recorded as not a trading session. Neither counts nor breaks.
            pass
        else:
            failures = pack_failures(pack)
            if failures:
                state["first_gap_session"] = day
                state["gap_reason"] = (
                    f"the pack for {day} is INCOMPLETE: "
                    + ", ".join(f"{name} ({why})" for name, why in sorted(failures.items()))
                )
                break
            count += 1
        try:
            cursor = previous_session(cursor)
        except Exception as exc:  # noqa: BLE001
            state["gap_reason"] = f"exchange calendar refused a date: {exc}"
            break
    state["sessions_consecutive_clean"] = count
    return state


def clean_digest_sessions(root: Path, *, as_of: str | date | None = None) -> int:
    """The length of the run of consecutive CLEAN sessions. Counting is not passing.

    Phase 2's exit gate is ten consecutive session days of digests plus the
    trader spot-auditing at least three against raw evidence. This answers the
    first half only; :func:`audit_approval_recorded` answers the second, and a
    number here never marks a live gate met.
    """
    return int(consecutive_clean_state(root, as_of=as_of)["sessions_consecutive_clean"])


# ---------------------------------------------------------------------------
# the audit half of the gate (Q4.2)
# ---------------------------------------------------------------------------

#: Beside the packs, one level above `facts/`, so it travels with the store it
#: describes rather than with a year directory.
AUDIT_APPROVAL_FILENAME = "digest_audit_approval.json"
AUDIT_APPROVAL_SCHEMA = "digest_audit_approval_v1"

#: The plan's own number: "the trader spot-auditing at least three".
REQUIRED_AUDITED_PACKS = 3


def audit_approval_path(root: Path) -> Path:
    return Path(root) / AUDIT_APPROVAL_FILENAME


def read_audit_approval(root: Path) -> dict[str, Any]:
    """The recorded spot-audit, or `{}`. Unreadable reads as absent.

    Conservative in the only direction that matters: an approval nobody can
    parse must not let Phase 3 through.
    """
    try:
        payload = json.loads(audit_approval_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def record_audit_approval(
    root: Path,
    *,
    packs: Sequence[str],
    note: str = "",
    approved_by: str = "trader",
    now: datetime | None = None,
) -> Path:
    """Record that a human read these packs against raw evidence.

    **Only the CLI calls this.** No nightly slot may: a runner that can approve
    its own evidence has not been audited, it has been asserted. Refuses fewer
    than :data:`REQUIRED_AUDITED_PACKS` packs, and refuses any date that has no
    pack on disk - an approval naming a session that was never written is the
    exact false record the gate exists to prevent.
    """
    days = []
    for entry in packs or ():
        day = str(entry)[:10].strip()
        if day and day not in days:
            days.append(day)
    if len(days) < REQUIRED_AUDITED_PACKS:
        raise ValueError(
            f"the digest audit needs at least {REQUIRED_AUDITED_PACKS} packs; "
            f"{len(days)} named. Phase 2's gate is ten consecutive clean sessions "
            "PLUS a spot-audit of at least three of them."
        )
    on_disk = latest_packs_by_session(root)
    missing = [day for day in days if day not in on_disk]
    if missing:
        raise ValueError(
            "no fact pack exists for " + ", ".join(missing) +
            "; an approval must name packs that were actually read"
        )
    payload = {
        "schema": AUDIT_APPROVAL_SCHEMA,
        "approved_at": _now(now).isoformat(timespec="seconds"),
        "approved_by": str(approved_by or "trader"),
        "packs": sorted(days),
        "note": str(note or ""),
        "how": (
            "Written by `python -m ai_jobs.digest approve-audit`, which a human "
            "runs. No nightly job writes this file."
        ),
    }
    return _publish(
        audit_approval_path(root),
        json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n",
    )


def audit_approval_recorded(root: Path) -> tuple[bool, list[str]]:
    """`(recorded, packs)`. An approval under the floor does not count."""
    payload = read_audit_approval(root)
    packs = [str(day)[:10] for day in (payload.get("packs") or []) if str(day).strip()]
    return (len(packs) >= REQUIRED_AUDITED_PACKS and bool(payload.get("approved_at")), packs)


def digest_gate_state(root: Path, *, as_of: str | date | None = None) -> dict[str, Any]:
    """The R10.I-shaped statement for any surface that reports on this phase.

    Two halves, both measured (Q4). `sessions_collected` keeps its pre-Q4
    meaning - the distinct count - so every existing reader still reads what it
    always read; `window_met` now turns on the CONSECUTIVE run, and `gate_met`
    additionally requires the recorded spot-audit.
    """
    run = consecutive_clean_state(root, as_of=as_of)
    consecutive = int(run["sessions_consecutive_clean"])
    collected = collected_digest_sessions(root)
    met = consecutive >= REQUIRED_CLEAN_SESSIONS
    recorded, audit_packs = audit_approval_recorded(root)
    if met:
        window_words = (
            f"Digest collection window met: {consecutive} consecutive clean "
            "session(s)."
        )
    else:
        where = run["first_gap_session"]
        window_words = (
            f"DIGEST GATE NOT MET: {consecutive} of {REQUIRED_CLEAN_SESSIONS} "
            "consecutive clean session fact pack(s)"
            + (f"; the run stops at {where}" if where else "")
            + (f" ({run['gap_reason']})" if run["gap_reason"] else "")
            + ". Phase 3 and anything downstream of it must not run on this evidence."
        )
    if recorded:
        audit_words = (
            f"Trader spot-audit recorded for {len(audit_packs)} pack(s): "
            + ", ".join(audit_packs) + "."
        )
    else:
        audit_words = (
            f"Trader spot-audit NOT recorded: {AUDIT_APPROVAL_FILENAME} is absent "
            f"or names fewer than {REQUIRED_AUDITED_PACKS} packs. Run "
            "`python -m ai_jobs.digest approve-audit --pack <date> ...` after "
            "reading the packs against raw evidence."
        )
    return {
        "sessions_collected": collected,
        "sessions_consecutive_clean": consecutive,
        "first_gap_session": run["first_gap_session"],
        "sessions_required": REQUIRED_CLEAN_SESSIONS,
        "window_met": met,
        "audit_recorded": recorded,
        "audit_packs": audit_packs,
        "audit_packs_required": REQUIRED_AUDITED_PACKS,
        "gate_met": bool(met and recorded),
        "statement": f"{window_words} {audit_words}",
    }


# ---------------------------------------------------------------------------
# entry_index.json - the compact, auditable handoff (Q4.4)
# ---------------------------------------------------------------------------

ENTRY_INDEX_FILENAME = "entry_index.json"
ENTRY_INDEX_SCHEMA = "digest_entry_index_v1"

#: FOUR sections, never merged. Each answers a different question on a
#: different population, and a reader who sums them has invented a number
#: nobody measured. The order is the order they are written in.
ENTRY_INDEX_SECTIONS = (
    "intraday_held_run",
    "swing_win_rates",
    "preference_observations",
    "journal_execution",
)

#: The one statistics contract (ground rule 10). A cell under it is UNMEASURED,
#: never a weak edge - which is why `changes_vs_prior_window` reports FLOOR
#: STATUS and never a ranking of immature cells. "Lately" is counted in trading
#: SESSIONS.
#:
#: RE-EXPORTS, imported hard and with no literal fallback: a fallback is a
#: second copy of the contract that silently wins whenever the import is the
#: thing that broke, and this repo has a rule against a list written in two
#: places for exactly that reason.
from evidence_stats import LATELY_SESSIONS as ENTRY_INDEX_WINDOW_SESSIONS  # noqa: E402
from evidence_stats import MIN_REPORTABLE_N as ENTRY_INDEX_FLOOR  # noqa: E402


def entry_index_path(root: Path) -> Path:
    return Path(root) / ENTRY_INDEX_FILENAME


def repo_commit(repo_root: Path | None = None) -> str:
    """HEAD, read from `.git` without spawning a process. `""` if unreadable.

    `research_warehouse.manifest.definitions_git_commit` is the precedent and
    reads `.git/HEAD` directly, which is right in a normal checkout and empty
    in a git WORKTREE - where `.git` is a FILE holding `gitdir: <path>` and the
    refs live in the COMMON dir the worktree points back at. Every agent builds
    in a worktree, so an index built there carried no commit at all.

    Provenance is evidence, not a gate: an unreadable repo yields `""` rather
    than failing the index.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        dot_git = root / ".git"
        if dot_git.is_file():
            pointer = dot_git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return ""
            git_dir = Path(pointer.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
        else:
            git_dir = dot_git
        common = git_dir
        common_file = git_dir / "commondir"
        if common_file.exists():
            target = Path(common_file.read_text(encoding="utf-8").strip())
            common = target if target.is_absolute() else (git_dir / target).resolve()

        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        for base in (git_dir, common):
            candidate = base / ref
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip()
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {ref}"):
                    return line.split(" ", 1)[0].strip()
    except OSError:
        return ""
    return ""


def read_entry_index(root: Path) -> dict[str, Any]:
    """The last good index, or `{}`. For the System Health / Research readers.

    Nothing in this packet consumes it; it exists so the next reader does not
    have to walk ninety packs to answer "what is here, and what changed?".
    """
    try:
        payload = json.loads(entry_index_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _cell_n(cell: Any) -> int:
    if isinstance(cell, Mapping):
        try:
            return int(cell.get("n") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _cell_value(cell: Any) -> Any:
    return cell.get("value") if isinstance(cell, Mapping) else None


def _held_run_cells(pack: Mapping[str, Any]) -> list[tuple[str, Any, int]]:
    """MFE/MAE cells only - the OPPORTUNITY half.

    `close_r` is the RESULT and is deliberately absent: the day-trade headline
    is "did the level hold, and how far did it run", and ground rule 12 forbids
    blending the two. This section indexes what that headline is computed from.
    """
    out: list[tuple[str, Any, int]] = []
    overall = ((pack.get("outcomes") or {}).get("overall")) or {}
    for metric in ("mfe_r", "mae_r"):
        cell = overall.get(metric)
        if isinstance(cell, Mapping):
            out.append((f"outcomes.overall.{metric}", _cell_value(cell), _cell_n(cell)))
    for row in ((pack.get("outcomes") or {}).get("slices")) or []:
        if not isinstance(row, Mapping):
            continue
        env = str(row.get("env_key") or "")
        side = str(row.get("side") or "")
        for metric in ("mfe_r", "mae_r"):
            cell = row.get(metric)
            if isinstance(cell, Mapping):
                out.append((
                    f"outcomes.slices[{env}|{side}].{metric}",
                    _cell_value(cell), _cell_n(cell),
                ))
    return out


def _preference_cells(pack: Mapping[str, Any]) -> list[tuple[str, Any, int, str]]:
    """The trader's own verdicts, classified by the ONE classifier.

    `review_learning.TAKE_ACTIONS` / `REJECT_ACTIONS` decide which action is a
    take and which is a reject; restating that here would be a second list to
    drift. Machine events, `*_fired`, `*_expired` and every `disarm_*` are in
    neither set and so are not indexed.
    """
    try:
        from review_learning import REJECT_ACTIONS, TAKE_ACTIONS
    except Exception:  # noqa: BLE001
        return []
    counts = ((pack.get("behaviour") or {}).get("by_action")) or {}
    out: list[tuple[str, Any, int, str]] = []
    for action, count in sorted(counts.items()):
        if action in TAKE_ACTIONS:
            lane = "take"
        elif action in REJECT_ACTIONS:
            lane = "reject"
        else:
            continue
        try:
            value = int(count)
        except (TypeError, ValueError):
            continue
        out.append((f"behaviour.by_action.{action}", value, value, lane))
    return out


def _section_cells(pack: Mapping[str, Any], pack_path: str) -> dict[str, list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = {name: [] for name in ENTRY_INDEX_SECTIONS}
    for key, value, n in _held_run_cells(pack):
        sections["intraday_held_run"].append({
            "pack_path": pack_path, "cell_key": key, "value": value, "n": n,
            "meets_floor": n >= ENTRY_INDEX_FLOOR,
        })
    for key, value, n, lane in _preference_cells(pack):
        sections["preference_observations"].append({
            "pack_path": pack_path, "cell_key": key, "value": value, "n": n,
            "lane": lane, "meets_floor": n >= ENTRY_INDEX_FLOOR,
        })
    return sections


#: Why two of the four sections are empty. A blank is right where the question
#: cannot be asked of this record, and an honest empty beats a number lifted
#: off a different grain.
_SECTION_NOTES = {
    "intraday_held_run": (
        "MFE/MAE cells - the OPPORTUNITY half of the day-trade headline. "
        "close_r is the RESULT and is never blended with these, so it is not "
        "indexed here."
    ),
    "swing_win_rates": (
        "EMPTY BY CONSTRUCTION: the daily fact pack carries CHAMPION INTRADAY "
        "outcomes only (answer 3), so no swing rate is derivable from it. "
        "The swing record lives in master_avwap_tier_outcomes.csv and is read "
        "through swing_headline; it is not restated here from a different grain. "
        "AND IT IS NOT A WIN RATE (ST1, 2026-09-06): that file's declared policy "
        "is swing_evidence.POLICY_SCANROW_V1 - outcome kind "
        "favorable_direction_scanrow_v1, the sign of a close-to-close percent "
        "move at a SCAN-ROW offset, one declared horizon of 5 scan rows, the "
        "lately window of 20 exchange sessions, explicit stale_horizon rows "
        "dropped. A model reading this index must call it a favorable-direction "
        "rate and never a stop-rule win rate or R."
    ),
    "preference_observations": (
        "The trader's own verdicts, classified by review_learning's TAKE/REJECT "
        "sets. Cohort GRADES are not here - they are the cohort slots' own "
        "files; this is what was decided on the day."
    ),
    "journal_execution": (
        "EMPTY BY CONSTRUCTION: the fact pack carries no journal or execution "
        "block. journal_import, journal_auto_tag and preference_trade_outcomes "
        "own that evidence and this index does not restate it from another store."
    ),
}


def _pending_experiments() -> dict[str, Any]:
    """Registered trials with their FROZEN windows, listed and UNRANKED.

    A ranked list of open experiments is a recommendation, and a trial that has
    not concluded has nothing to recommend. Unavailable is a NOTE, never zero.
    """
    try:
        from research_warehouse.config import get_research_store_dir
        from research_warehouse.trial_ledger import (
            STATUS_ABANDONED,
            STATUS_CONCLUDED,
            load,
        )

        store_root = get_research_store_dir()
        if store_root is None:
            return {
                "entries": [],
                "note": "research_store_dir is unset; the trial ledger was not read.",
            }
        rows = load(store_root)
    except Exception as exc:  # noqa: BLE001 - the ledger is evidence, not a gate
        return {"entries": [], "note": f"the trial ledger could not be read: {exc}"}
    entries = [
        {
            "trial_id": str(row.get("trial_id") or ""),
            "family": str(row.get("family") or ""),
            "question": str(row.get("question") or ""),
            "status": str(row.get("status") or ""),
            "registered_at": str(row.get("registered_at") or ""),
            "declared_window": dict(row.get("declared_window") or {}),
            "declared_floors": dict(row.get("declared_floors") or {}),
            "declared_cell_count": row.get("declared_cell_count"),
        }
        for row in rows
        if str(row.get("status") or "") not in (STATUS_ABANDONED, STATUS_CONCLUDED)
    ]
    entries.sort(key=lambda row: (row["registered_at"], row["trial_id"]))
    return {
        "entries": entries,
        "note": (
            "Registered before any outcome was inspected, listed in registration "
            "order and NEVER ranked. A frozen window is what makes a trial "
            "readable later; it is printed, not re-cut."
        ),
    }


def _window_cell_totals(
    packs: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, str],
    days: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Total n per (section, cell_key) across a window; the newest path is kept."""
    totals: dict[str, dict[str, Any]] = {}
    for day in sorted(days):
        pack = packs.get(day)
        if not pack:
            continue
        for section, rows in _section_cells(pack, paths.get(day, "")).items():
            for row in rows:
                key = f"{section}|{row['cell_key']}"
                bucket = totals.setdefault(key, {
                    "section": section, "cell_key": row["cell_key"],
                    "n": 0, "pack_path": "",
                })
                bucket["n"] += int(row["n"])
                if row["pack_path"]:
                    bucket["pack_path"] = row["pack_path"]  # newest wins
    return totals


def _window_days(end: str, sessions: int) -> list[str]:
    """The `sessions`-session window ending at `end`, inclusive, by the calendar."""
    try:
        from market_calendar import is_session, previous_session

        cursor = date.fromisoformat(end)
        if not is_session(cursor):
            cursor = previous_session(cursor)
        days = [cursor.isoformat()]
        for _ in range(max(0, int(sessions) - 1)):
            cursor = previous_session(cursor)
            days.append(cursor.isoformat())
        return sorted(days)
    except Exception:  # noqa: BLE001 - a window is never worth a blank index
        return []


def _latest_measured_report(root: Path) -> dict[str, Any]:
    """The newest measured report under `root`, named not summarised (WS-RP).

    The index points at the file the numbers were READ from - the newest
    version, because a `_v2` supersedes the `_v1` beside it - and carries its
    `report_id` so a reader can tell whether the page they are looking at and
    the export they were handed are the same document. It computes nothing: a
    failure to read one is an absent key, never a wrong number.
    """
    try:
        from ai_jobs import measured_report_publish

        payload = measured_report_publish.latest_published(Path(root))
    except Exception as exc:  # noqa: BLE001 - the index never costs the packs
        _log.info("Entry index: no measured report could be read (%s).", exc)
        return {}
    if not payload:
        return {}
    path = Path(str(payload.get("_path") or ""))
    cells = payload.get("cells") or []
    return {
        "path": str(path),
        "markdown_path": str(path.with_suffix(".md")),
        "report_id": str(payload.get("report_id") or ""),
        "as_of": str(payload.get("as_of") or ""),
        "session_date": str(payload.get("session_date") or ""),
        "cells": len(cells),
        "cells_measured": sum(
            1 for cell in cells
            if isinstance(cell, Mapping) and str(cell.get("state") or "") == "measured"
        ),
        "note": (
            "One measured report per session, published by the `measured_report` "
            "slot. The `.md` sibling is the readable brief; the JSON carries "
            "every cell with its population, window, clock and sources."
        ),
    }


def build_entry_index(root: Path, *, as_of: str | date | None = None) -> dict[str, Any]:
    """The whole index, computed. Deterministic; no model is called from here."""
    root = Path(root)
    newest = latest_pack_files_by_session(root)
    versions: dict[str, int] = {}
    for pack in read_fact_packs(root):
        day = str(pack.get("session_date") or "")
        if day:
            versions[day] = versions.get(day, 0) + 1
    sessions = {day: pack for day, (_path, pack) in newest.items() if pack.get("is_session")}

    limit = ""
    if as_of is not None:
        limit = as_of.isoformat() if isinstance(as_of, date) else str(as_of)[:10]
    in_scope = [day for day in sorted(sessions) if not limit or day <= limit]
    latest_session = in_scope[-1] if in_scope else ""

    # The file each session's numbers were READ from - the newest sibling, not
    # `facts_path`'s version 1. A citation to a superseded pack points a reader
    # at the record that was corrected.
    paths = {day: str(newest[day][0]) for day in sessions}

    window = _window_days(latest_session, ENTRY_INDEX_WINDOW_SESSIONS) if latest_session else []
    prior_end = ""
    if window:
        try:
            from market_calendar import previous_session

            prior_end = previous_session(date.fromisoformat(window[0])).isoformat()
        except Exception:  # noqa: BLE001
            prior_end = ""
    prior = _window_days(prior_end, ENTRY_INDEX_WINDOW_SESSIONS) if prior_end else []
    in_window = [day for day in (window or in_scope) if day in sessions]

    rows = []
    for day in in_window:
        pack = sessions[day]
        failures = pack_failures(pack)
        rows.append({
            "session_date": day,
            "pack_path": paths.get(day, ""),
            "versions": int(versions.get(day, 1)),
            "superseded": int(versions.get(day, 1)) > 1,
            "clean": not failures,
            "failures": failures,
            "coverage": dict(pack.get("coverage") or {}),
        })

    aggregated: dict[str, list[dict[str, Any]]] = {name: [] for name in ENTRY_INDEX_SECTIONS}
    for day in in_window:
        for name, entries in _section_cells(sessions[day], paths.get(day, "")).items():
            aggregated[name].extend(entries)
    section_payload = {
        name: {"entries": aggregated[name], "note": _SECTION_NOTES[name]}
        for name in ENTRY_INDEX_SECTIONS
    }

    this_totals = _window_cell_totals(sessions, paths, window or in_scope)
    prior_totals = _window_cell_totals(sessions, paths, prior)
    cleared: list[dict[str, Any]] = []
    fell: list[dict[str, Any]] = []
    for key in sorted(set(this_totals) | set(prior_totals)):
        section, _, cell_key = key.partition("|")
        now_row = this_totals.get(key) or {"n": 0, "pack_path": ""}
        was_row = prior_totals.get(key) or {"n": 0, "pack_path": ""}
        now_met = int(now_row["n"]) >= ENTRY_INDEX_FLOOR
        was_met = int(was_row["n"]) >= ENTRY_INDEX_FLOOR
        if now_met == was_met:
            continue
        change = {
            "section": section,
            "cell_key": cell_key,
            "pack_path": now_row.get("pack_path") or was_row.get("pack_path", ""),
            "this_window": {"n": int(now_row["n"]), "meets_floor": now_met},
            "prior_window": {"n": int(was_row["n"]), "meets_floor": was_met},
        }
        (cleared if now_met else fell).append(change)

    newest_pack = sessions.get(latest_session) or {}
    statistics = ((newest_pack.get("outcomes") or {}).get("overall") or {}).get("statistics") or {}

    return {
        "schema_version": ENTRY_INDEX_SCHEMA,
        "generated_at": _now().isoformat(timespec="seconds"),
        "git_commit": repo_commit(),
        "latest_complete_session": latest_session,
        "versions": {
            # Read off a pack. Nothing invented: the pack carries no recipe id
            # and no vocabulary version, so neither is listed.
            "facts_schema": FACTS_SCHEMA,
            "narration_schema": NARRATION_SCHEMA,
            "statistics_schema": str(statistics.get("schema") or ""),
            "evidence_label": str(newest_pack.get("evidence_label") or ""),
            "n_floor": int(statistics.get("n_floor") or ENTRY_INDEX_FLOOR),
            "note": (
                "The identifiers the packs themselves carry. A fact pack records "
                "no recipe id and no vocabulary version, so none is printed here "
                "rather than one being invented."
            ),
        },
        "window": {
            "sessions": ENTRY_INDEX_WINDOW_SESSIONS,
            "since": window[0] if window else "",
            "until": window[-1] if window else "",
            "prior_since": prior[0] if prior else "",
            "prior_until": prior[-1] if prior else "",
            "note": "Counted in TRADING SESSIONS through the exchange calendar.",
        },
        "sessions": rows,
        "changes_vs_prior_window": {
            "cleared_the_floor": cleared,
            "fell_below_the_floor": fell,
            "floor": ENTRY_INDEX_FLOOR,
            # Both counts, because "46 cleared, 0 fell" is a finding when the
            # prior window had packs and an artefact of a young store when it
            # had none - and the two look identical without this line.
            "this_window_packs": len([day for day in (window or in_scope) if day in sessions]),
            "prior_window_packs": len([day for day in prior if day in sessions]),
            "note": (
                "By FLOOR STATUS only. A cell that crossed the evidence floor "
                "since the prior equal-length window is worth a look; an "
                "immature cell is never ranked and no value is compared. "
                "`prior_window_packs` is 0 when there was no pack in the prior "
                "window at all, in which case every 'cleared' row is a first "
                "sighting rather than a change."
            ),
        },
        **section_payload,
        # WS-RP (2026-09-13): a TOP-LEVEL key, deliberately NOT a fifth
        # section. `ENTRY_INDEX_SECTIONS` is a published four-tuple whose
        # members are cell families from the fact packs; the measured report is
        # a different document with its own id, so it is named here and read
        # from its own file rather than folded into a contract it does not
        # belong to.
        "measured_report": _latest_measured_report(root),
        "pending_experiments": _pending_experiments(),
        # A frontier model opens a ticker brief only for a STATED question.
        # This list is deliberately empty: nothing here may invent one.
        "open_questions_for_a_ticker_brief": [],
        "note": (
            "Deterministic. Every number was computed by code from the fact "
            "packs; no model was called, and nothing here ranks, scores, gates "
            "or alerts."
        ),
    }


def write_entry_index(root: Path, *, as_of: str | date | None = None) -> Path:
    """Publish the index with a temp-and-rename write. Raises on a write failure.

    The caller (`run_daily_digest`) logs and swallows: the index is a
    convenience over a record that is already on disk, and it must never cost
    the pack that was just published.
    """
    index = build_entry_index(root, as_of=as_of)
    return _publish(
        entry_index_path(root),
        json.dumps(index, indent=1, sort_keys=True, default=str) + "\n",
    )


# ---------------------------------------------------------------------------
# the CLI (Q4.2) - the only writer of the audit approval
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m ai_jobs.digest ...`, run from `scripts/`.

    Four commands. `approve-audit` writes the file the trader - and nothing
    automatic - writes; `rebuild` writes a SUPERSEDING sibling pack for a day
    whose pack read the sources too early (the 2026-09-06 audit found two
    packs at `in_window` 0 over 78 and 447 finals, both generated before the
    after-close sweep of a frozen desk wrote the day) - facts only, no model,
    and like `approve-audit` a human runs it; the other two only read.
    """
    import argparse

    # `--root` on BOTH the top parser and every subparser, so it may be typed
    # on either side of the command name. A flag that only works in one
    # position is a flag the trader gets wrong once and then stops using.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="", help="digest store root (defaults to the AI store)")

    parser = argparse.ArgumentParser(
        prog="ai_jobs.digest",
        parents=[common],
        description="Daily digest gate and entry index (no model is called).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    approve = sub.add_parser(
        "approve-audit",
        parents=[common],
        help="record that the trader read these fact packs against raw evidence",
    )
    approve.add_argument("--pack", action="append", default=[], metavar="YYYY-MM-DD")
    approve.add_argument("--note", default="")

    rebuild = sub.add_parser(
        "rebuild",
        parents=[common],
        help="rebuild a day's fact pack from the live sources as a superseding "
             "sibling (D6: the early pack is never edited); no model is called",
    )
    rebuild.add_argument("--pack", action="append", default=[], metavar="YYYY-MM-DD")

    sub.add_parser("gate", parents=[common], help="print the two halves of the Phase 2 gate")
    sub.add_parser(
        "entry-index", parents=[common],
        help="rebuild entry_index.json from the packs on disk",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        root = Path(args.root) if args.root else _default_root()
    except Exception as exc:  # noqa: BLE001
        print(f"AI store unavailable: {exc}")
        return 2

    if args.command == "approve-audit":
        try:
            written = record_audit_approval(root, packs=args.pack, note=args.note)
        except ValueError as exc:
            print(f"refused: {exc}")
            return 2
        except OSError as exc:
            print(f"could not write the approval: {exc}")
            return 2
        print(f"recorded {written}")
        return 0

    if args.command == "rebuild":
        days = []
        for entry in args.pack:
            day = str(entry)[:10].strip()
            if day and day not in days:
                days.append(day)
        if not days:
            print("refused: name at least one --pack YYYY-MM-DD to rebuild")
            return 2
        failed = 0
        for day in days:
            result = run_daily_digest(session_date=day, root=root, narrate=False)
            status = str(result.get("status") or "")
            if status == STATUS_OK:
                print(f"rebuilt {day}: {result.get('reason')}")
                for output in result.get("outputs") or ():
                    print(f"  {output}")
            else:
                failed += 1
                print(f"failed {day}: {result.get('reason')}")
        return 2 if failed else 0

    if args.command == "entry-index":
        try:
            print(f"wrote {write_entry_index(root)}")
        except OSError as exc:
            print(f"could not write the index: {exc}")
            return 2
        return 0

    state = digest_gate_state(root)
    print(json.dumps(state, indent=1, sort_keys=True, default=str))
    return 0 if state["gate_met"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
