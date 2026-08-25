"""Benchmark trends over the diagnostics already on disk — P1.4 (packet W7).

Phase 1's exit gate asks for "representative benchmark/golden fixtures and
trends for timings, provider calls, failures, coverage, and scan-stage latency".
Every one of those is already measured: `run_manifest_v1` records per-phase
seconds and the whole `provider.<family>.<event>[.<source>]` counter tree, and
`ai_job_ledger.jsonl` records every overnight job's outcome. What was missing was
a reader that turns them into a TREND, so "the scan feels slower" becomes a
number with an n beside it.

**Zero new measurement.** Nothing here instruments a hot path, times anything,
or runs during a scan. It opens files that were written hours ago, and a test
walks its AST to keep that true.

Three honesty rules, the same three every evidence surface in this repo keeps:

* **n on every figure.** A median over two runs is not a trend, and the only
  thing that says so is the count printed beside it.
* **A comparison needs both halves.** With no baseline behind the recent window
  the recent number is reported and the absence is NAMED - never a change
  computed against nothing.
* **Absent is not zero.** A manifest that never recorded a phase is missing that
  phase, not a phase that took no time; a family with no attempts has no failure
  rate, rather than a 0% one.

Everything is labelled `discovery`. These windows are chosen after the fact.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Schema NAME (R10 ground rule 5).
TRENDS_SCHEMA = "observability_trends_v1"

#: The golden fixture that freezes this report's arithmetic.
FIXTURE_NAME = "observability_trends_v1"

#: Runs in the recent window, and in the baseline behind it.
DEFAULT_WINDOW = 10

#: Errors quoted verbatim in the run block. A failure with no message is worse
#: to debug than one with a bad message, so the ones there are travel.
MAX_QUOTED_ERRORS = 5


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _numbers(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values or ():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:  # NaN
            continue
        out.append(number)
    return out


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _p90(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = 0.9 * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


# ---------------------------------------------------------------------------
# stage latency
# ---------------------------------------------------------------------------


def _phase_seconds(manifest: Mapping[str, Any]) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for phase in manifest.get("phases") or ():
        if not isinstance(phase, Mapping):
            continue
        label = str(phase.get("label") or "").strip()
        if not label or label.upper().startswith("TOTAL"):
            # The total is `total_seconds`, and counting it as a stage would
            # make every report claim one phase that is the sum of the others.
            continue
        value = _numbers([phase.get("seconds")])
        if value:
            seconds[label] = value[0]
    return seconds


def _stage_latency(recent, baseline) -> list[dict[str, Any]]:
    recent_phases = [_phase_seconds(manifest) for manifest in recent]
    baseline_phases = [_phase_seconds(manifest) for manifest in baseline]
    labels = sorted({label for mapping in recent_phases + baseline_phases for label in mapping})

    rows: list[dict[str, Any]] = []
    for label in labels:
        values = [mapping[label] for mapping in recent_phases if label in mapping]
        prior = [mapping[label] for mapping in baseline_phases if label in mapping]
        median = _median(values)
        baseline_median = _median(prior)
        change = None
        basis = ""
        if median is None:
            basis = "not recorded in the recent window"
        elif baseline_median is None:
            basis = (
                "no baseline: this phase was not recorded in the earlier window, "
                "so there is nothing to compare against"
            )
        elif baseline_median == 0:
            basis = "baseline median is zero; a percentage change would be undefined"
        else:
            change = round((median - baseline_median) / baseline_median * 100.0, 2)
            basis = "median of the recent window against the median of the one before it"
        rows.append(
            {
                "label": label,
                "n": len(values),
                "median": median,
                "p90": _p90(values),
                "baseline_n": len(prior),
                "baseline_median": baseline_median,
                "change_pct": change,
                "change_basis": basis,
                "direction": (
                    None if change is None
                    else "slower" if change > 0 else "faster" if change < 0 else "unchanged"
                ),
                # Named rather than inferred from a smaller n: a phase that some
                # runs skip and a phase that got faster look identical in a
                # median alone.
                "runs_missing_phase": len(recent_phases) - len(values),
            }
        )
    rows.sort(key=lambda row: (-(row["median"] or 0.0), row["label"]))
    return rows


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


def _provider_rows(manifests) -> list[dict[str, Any]]:
    """Fold `provider.<family>.<event>[.<source>]` across the window.

    The counter tree is the scan's own; nothing is re-counted here. Only the
    fold and the two rates are this module's, and both refuse to divide by a
    denominator of zero.
    """
    families: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        counters = manifest.get("counters") or {}
        flat = _flatten(counters)
        for key, value in flat.items():
            if not key.startswith("provider."):
                continue
            parts = key.split(".")
            if len(parts) < 3:
                continue
            family, event = parts[1], parts[2]
            source = parts[3] if len(parts) > 3 else ""
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            entry = families.setdefault(
                family,
                {"family": family, "lookups": 0, "cache_hits": 0, "attempts": 0,
                 "failures": 0, "by_source": {}, "runs": 0},
            )
            if event == "lookup":
                entry["lookups"] += count
            elif event == "cache_hit":
                entry["cache_hits"] += count
            elif event == "attempt":
                entry["attempts"] += count
            elif event in ("failure", "success"):
                if event == "failure":
                    entry["failures"] += count
            if source:
                bucket = entry["by_source"].setdefault(
                    source, {"attempts": 0, "failures": 0, "successes": 0}
                )
                if event == "attempt":
                    bucket["attempts"] += count
                elif event == "failure":
                    bucket["failures"] += count
                elif event == "success":
                    bucket["successes"] += count
    rows = []
    for family, entry in families.items():
        entry["runs"] = len(list(manifests))
        entry["failure_rate"] = (
            round(entry["failures"] / entry["attempts"], 4) if entry["attempts"] else None
        )
        entry["cache_hit_rate"] = (
            round(entry["cache_hits"] / entry["lookups"], 4) if entry["lookups"] else None
        )
        entry["rate_basis"] = (
            "failures / attempts"
            if entry["attempts"]
            else "no attempt was made in this window, so there is no failure rate - "
                 "zero failures out of zero attempts is not a 0% rate"
        )
        rows.append(entry)
    rows.sort(key=lambda row: (-row["lookups"], row["family"]))
    return rows


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Counters arrive either dotted or nested; both spellings are read."""
    flat: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


# ---------------------------------------------------------------------------
# runs, jobs, coverage
# ---------------------------------------------------------------------------


def _run_block(manifests) -> dict[str, Any]:
    totals = _numbers(manifest.get("total_seconds") for manifest in manifests)
    failed = [
        manifest for manifest in manifests
        if str(manifest.get("status") or "").lower() not in ("ok", "")
    ]
    errors = [
        str(manifest.get("error") or "").strip()
        for manifest in failed
        if str(manifest.get("error") or "").strip()
    ]
    by_type: dict[str, int] = {}
    for manifest in manifests:
        job_type = str(manifest.get("job_type") or "unstated")
        by_type[job_type] = by_type.get(job_type, 0) + 1
    return {
        "n": len(list(manifests)),
        "failed": len(failed),
        "failure_rate": round(len(failed) / len(manifests), 4) if manifests else None,
        "by_job_type": by_type,
        "total_seconds_median": _median(totals),
        "total_seconds_p90": _p90(totals),
        "recent_errors": errors[-MAX_QUOTED_ERRORS:],
        "errors_not_quoted": max(0, len(errors) - MAX_QUOTED_ERRORS),
    }


def _job_rows(rows) -> list[dict[str, Any]]:
    folded: dict[str, dict[str, int]] = {}
    for row in rows or ():
        job = str(row.get("job") or "").strip()
        if not job:
            continue
        status = str(row.get("status") or "").strip().lower() or "unstated"
        entry = folded.setdefault(job, {})
        entry[status] = entry.get(status, 0) + 1
    built = []
    for job, statuses in folded.items():
        total = sum(statuses.values())
        failed = statuses.get("failed", 0)
        built.append({
            "job": job,
            "n": total,
            "ok": statuses.get("ok", 0),
            "failed": failed,
            "degraded": statuses.get("degraded_no_narrative", 0),
            "skipped": statuses.get("skipped", 0),
            "by_status": dict(sorted(statuses.items())),
            "failure_rate": round(failed / total, 4) if total else None,
        })
    built.sort(key=lambda row: (-row["failed"], row["job"]))
    return built


def _coverage_block(manifests) -> dict[str, Any]:
    counters = [_flatten(manifest.get("counters") or {}) for manifest in manifests]
    block: dict[str, Any] = {}
    for name in ("symbols_processed", "tracked_rows"):
        values = _numbers(counter.get(name) for counter in counters)
        block[name] = {
            "n": len(values),
            "median": _median(values),
            "p90": _p90(values),
            # Runs that never recorded the counter are NAMED: a smaller n and a
            # smaller number mean different things.
            "runs_missing": len(counters) - len(values),
        }
    return block


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def build_trends(
    *,
    manifests: Sequence[Mapping[str, Any]] = (),
    job_rows: Sequence[Mapping[str, Any]] = (),
    window: int = DEFAULT_WINDOW,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One trend report over already-written diagnostics. Reads only.

    ``manifests`` arrive oldest first, the way `load_recent_manifests` returns
    them. The last ``window`` are the recent window and the ``window`` before
    those are the baseline.
    """
    ordered = list(manifests or ())
    size = max(1, int(window or DEFAULT_WINDOW))
    recent = ordered[-size:]
    baseline = ordered[-2 * size: -size] if len(ordered) > size else []

    report = {
        "schema": TRENDS_SCHEMA,
        "generated_at": _now(now).isoformat(timespec="seconds"),
        "evidence_label": "discovery",
        "window": {
            "runs_per_window": size,
            "recent_n": len(recent),
            "baseline_n": len(baseline),
            "basis": (
                "the last N runs against the N before them. Both windows were "
                "chosen after the fact, so every figure is discovery: a change "
                "here is a thing to look at, never a thing to conclude."
            ),
        },
        "runs": _run_block(recent),
        "stage_latency": _stage_latency(recent, baseline),
        "providers": _provider_rows(recent),
        "jobs": _job_rows(job_rows),
        "coverage": _coverage_block(recent),
    }
    report["summary"] = _summary(report)
    return report


def _summary(report: Mapping[str, Any]) -> str:
    runs = report.get("runs") or {}
    if not runs.get("n"):
        return (
            "No run manifests in the window: nothing was measured, which is not "
            "the same as nothing going wrong."
        )
    slowest = (report.get("stage_latency") or [{}])[0]
    parts = [
        f"{runs['n']} run(s), {runs['failed']} failed; median total "
        f"{runs.get('total_seconds_median')}s, p90 {runs.get('total_seconds_p90')}s."
    ]
    if slowest.get("label"):
        change = slowest.get("change_pct")
        parts.append(
            f"Slowest stage: {slowest['label']} at {slowest.get('median')}s "
            f"(n={slowest.get('n')})"
            + (f", {change:+.1f}% against the previous window." if change is not None
               else f" - {slowest.get('change_basis')}.")
        )
    failing = [row for row in (report.get("jobs") or []) if row.get("failed")]
    if failing:
        named = ", ".join(f"{row['job']} {row['failed']}/{row['n']}" for row in failing[:3])
        parts.append(f"Failing overnight job(s): {named}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# the live read and the CLI
# ---------------------------------------------------------------------------


def read_live(*, window: int = DEFAULT_WINDOW, now: datetime | None = None) -> dict[str, Any]:
    """Build the report from this machine's own diagnostics. Opens files only."""
    from diagnostics.run_manifest import load_recent_manifests

    manifests = load_recent_manifests(limit=max(2 * window, window))
    jobs: list[Mapping[str, Any]] = []
    try:
        from ai_jobs import ledger

        jobs = ledger._read_rows(ledger.ledger_path(create=False))
    except Exception:  # noqa: BLE001 - an absent AI store is a normal state
        jobs = []
    return build_trends(manifests=manifests, job_rows=jobs, window=window, now=now)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="runs in the recent window (and in the baseline behind it)")
    parser.add_argument("--json", action="store_true", help="print the whole report")
    args = parser.parse_args(argv)
    report = read_live(window=args.window)
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True, default=str))
    else:
        print(report["summary"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
